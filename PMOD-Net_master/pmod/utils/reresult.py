import os
import h5py
import torch
import argparse
from rich.progress import track
from typing import List, Dict , Tuple
import torch.nn as nn
from torch import Tensor
from typing import Any, NamedTuple
import torch.nn.functional as F
from openpyxl.worksheet.worksheet import Worksheet
import openpyxl as xl
from typing import Union
from datetime import datetime
import csv

METRIC_MAE = "MAE"
METRIC_RMSE = "RMSE"
METRIC_MAPE = "MAPE"
METRIC_IOU = "IoU"

DIR_RESULTS = 'results'

class METRIC_SUM_OUTPUT(NamedTuple):
    sum: Tensor
    count: Tensor



class METRIC_RESULT:
    def __init__(self, name: str = '', use_class: bool = False, use_thresholds: bool = False, all_tag: str = 'All') -> None:
        self.name: str = name
        self.use_class: bool = use_class
        self.use_thresholds: bool = use_thresholds
        self.all_tag: str = all_tag

        self.all_metric: Tensor = None
        self.class_metric: Tensor = None
        self.threshold_metric: Tensor = None

        self.counts_metric: Tensor = None
        self.sum_class_metric: Tensor = None
        self.sum_threshold_metric: Tensor = None

    def add(self, count_metric: Tensor = None, class_metric: Tensor = None, threshold_metric: Tensor = None):
        if isinstance(count_metric, Tensor):
            if self.counts_metric is None:
                self.counts_metric = torch.zeros_like(count_metric)
            self.counts_metric = self.counts_metric + count_metric

        if isinstance(class_metric, Tensor):
            tmp_class_metric = class_metric
        elif isinstance(self.class_metric, Tensor):
            tmp_class_metric = self.class_metric
        else:
            tmp_class_metric = None
        if tmp_class_metric is not None:
            if self.sum_class_metric is None:
                self.sum_class_metric = torch.zeros_like(tmp_class_metric)
            self.sum_class_metric = self.sum_class_metric + tmp_class_metric

        if isinstance(threshold_metric, Tensor):
            tmp_threshold_metric = threshold_metric
        elif isinstance(self.threshold_metric, Tensor):
            tmp_threshold_metric = self.threshold_metric
        else:
            tmp_threshold_metric = None
        if tmp_threshold_metric is not None:
            if self.sum_threshold_metric is None:
                self.sum_threshold_metric = torch.zeros_like(
                    tmp_threshold_metric)
            self.sum_threshold_metric = self.sum_threshold_metric + tmp_threshold_metric

#predとgtが一致しているピクセル数をクラスごとにカウント
class MetricIntersection(nn.Module):
    def __init__(self):
        super(MetricIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1) #一番確率が高いクラスを取得
        pred = F.one_hot(pred, num_classes=num_classes) #one-hotエンコーディング(1,256,512)-.> (1,256,512,5)
        gt = gt.to(torch.int64)
        gt = F.one_hot(gt, num_classes=num_classes) #one-hotエンコーディング(1,256,512)-.> (1,256,512,5)

        intersection = torch.logical_and(pred, gt)

        return torch.sum(intersection, dim=(1, 2)) #H*Wで一致しているピクセル数をカウント (1,5) ex[[75378, 50570,     0,  4024,     0]]

class MetricSumIntersection(MetricIntersection):
    def __init__(self):
        super(MetricSumIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt), dim=0) #MetricIntersectionで計算された(B,C)をバッチ方向に結合してクラスごとのtotal intersection

class MetricUnion(nn.Module):
    def __init__(self, smooth: float = 1e-10):
        super(MetricUnion, self).__init__()
        self.smooth = torch.tensor(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = gt.to(torch.int64)

        gt = F.one_hot(gt, num_classes=num_classes)

        union = torch.logical_or(pred, gt) #分母の和集合の計算(1,256,512,5)

        return torch.sum(union, (1, 2)) + self.smooth # predかGTのどちらかが1のピクセル数をカウント (1,5) ex[[75378, 50570,     0,  4024,     0]]

class MetricSumUnion(MetricUnion):
    def __init__(self, smooth: float = 1e-10):
        super().__init__(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt) - self.smooth, dim=0) + self.smooth #MetricIntersectionで計算された(B,C)をバッチ方向に結合してクラスごとのtotal intersection 最初にself.smoothしてるから引いてからまた加算する


# 単一画像またはバッチに対して深度（gt）が指定された範囲に入っているかを判断し,その範囲内で各クラスごとのマスクを作成してどのクラスがどれくらい「その範囲内にあるか」をカウント
class MetricInRange(nn.Module):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricInRange, self).__init__()

        thresholds.sort()
        self.thresholds: Tensor
        self.register_buffer('thresholds', torch.tensor(thresholds))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        _range_list: List[Tensor] = [] #_range_list: 範囲ごとのマスクを格納するリスト
        _prev_th: float = 0.0 #前のしきい値（初期値は0.0）
        seg_gt = seg_gt.long() #seg_gt を整数化（one-hot化する準備）
        if self.num_classes > 0:
            _one_hot: Tensor = F.one_hot(seg_gt, num_classes=self.num_classes).bool() #[B, H, W] → [B, H, W, C] に変換
        for th_idx in range(self.thresholds.shape[0]):
            _th: Tensor = self.thresholds[th_idx].to(gt.device)#各しきい値に対してループ
            if _th <= _prev_th:
                continue
            _in_range: Tensor = (_prev_th < gt) & (gt <= _th) & (gt <= 1.0)#(_prev_th, _th]」の範囲に入っていて、1.0以下であるピクセルをマスクにする
            if self.num_classes > 0:
                _in_range: Tensor = torch.squeeze(_in_range, dim=1)#[B, 1, H, W] → [B, H, W] に変換（次の処理のため）
                _in_range_oh: Tensor = _one_hot.clone()
                _in_range_oh[~_in_range] = False#範囲外のピクセルは全部 False にして、範囲内だけ保持
                _range_list.append(_in_range_oh.bool().permute(
                    0, 3, 1, 2).unsqueeze(dim=1)) #[B, H, W, C] → [B, 1, C, H, W] に変換してリストへ
            else: #クラスなし
                _range_list.append(_in_range.unsqueeze(dim=1))#[B, 1, H, W] としてそのまま追加
            _prev_th = _th.clone()#次のしきい値の準備

        range_cat: Tensor = torch.cat(_range_list, dim=1)#[B, T, C, H, W] or [B, T, 1, H, W] のテンソルに変換
        range_cnt: Tensor = range_cat.sum(
            dim=(3, 4)) if self.num_classes > 0 else range_cat.sum(dim=(2, 3, 4)) #各クラス or 範囲ごとのピクセル数カウント セグメントあり：[B, T, C] or セグメントなし：[B, T] のテンソルに変換
        return range_cat, range_cnt #range_cat: 範囲ごとのマスクテンソル（True/Falseでどこが該当か）, range_cnt: その範囲ごとのピクセル数カウントテンソル

class MetricBatchSumAPE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ape = torch.abs(pred - gt) / gt
        ape = ape.unsqueeze(dim=1).expand_as(in_range)
        zeros = torch.zeros_like(ape)
        ape = torch.where(in_range, ape, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ape, cnts), METRIC_SUM_OUTPUT(ape.sum(dim=(3, 4)), cnts)
        else:
            return METRIC_SUM_OUTPUT(ape.sum(dim=(2, 3, 4)), cnts)

class MetricSumAPE(MetricBatchSumAPE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ape = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ape.sum, dim=0), torch.sum(ape.count, dim=0))

class MetricBatchSumAE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ae = torch.abs(pred - gt)

        ae = ae.unsqueeze(dim=1).expand_as(in_range)#(B, H, W) → (B, T, C, H, W)
        zeros = torch.zeros_like(ae)
        ae = torch.where(in_range, ae, zeros)# マスク対象外は0にする

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ae, cnts), METRIC_SUM_OUTPUT(ae.sum(dim=(3, 4)), cnts)#sum: AEの合計 cnts: AEのピクセル数 (B, T, C)
        else:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(2, 3, 4)), cnts)
        
class MetricSumAE(MetricBatchSumAE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ae = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ae.sum, dim=0), torch.sum(ae.count, dim=0))

# クラスの分布を出力
# class MetricBatchSumVehicle_Pos(MetricInRange):
#     def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
#         super(MetricBatchSumVehicle_Pos, self).__init__(thresholds, num_classes)

#     def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
#         in_range, cnts = self.in_range(gt, seg_gt)
#         vehicle_pos = gt

#         vehicle_pos = vehicle_pos.unsqueeze(dim=1).expand_as(in_range)
#         zeros = torch.zeros_like(vehicle_pos)
#         vehicle_pos = torch.where(in_range, vehicle_pos, zeros)

#         if self.num_classes > 0:
#             return METRIC_SUM_OUTPUT(vehicle_pos.sum(dim=(3, 4)), cnts)
#         else:
#             return METRIC_SUM_OUTPUT(vehicle_pos.sum(dim=(2, 3, 4)), cnts)

# class MetricSumVehicle_Pos(MetricBatchSumVehicle_Pos):
#     def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
#         super(MetricSumVehicle_Pos, self).__init__(thresholds, num_classes)

#     def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
#         vehicle_pos = super().forward(pred, gt, seg_gt)
#         return METRIC_SUM_OUTPUT(torch.sum(vehicle_pos.sum, dim=0), torch.sum(vehicle_pos.count, dim=0))

class MetricBatchSumSE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        se = torch.pow(pred - gt, 2)

        se = se.unsqueeze(dim=1).expand_as(in_range)
        zeros = torch.zeros_like(se)
        se = torch.where(in_range, se, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(se.sum(dim=(3, 4)), cnts)
        else:
            return METRIC_SUM_OUTPUT(se.sum(dim=(2, 3, 4)), cnts)

class MetricSumSE(MetricBatchSumSE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        se = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(se.sum, dim=0), torch.sum(se.count, dim=0))
    
def _excel_overview_output(metric_results: Dict[str, METRIC_RESULT], excel_sheet: Worksheet, range_norm: float, result_dict: Dict[str, float] = None, dynamic_class: List[str] = None, static_class: List[str] = None) -> None:
    for row, (metric_key, metric_result) in enumerate(metric_results.items(), 2):
        excel_sheet.cell(row=row, column=1, value=metric_result.name)

        if metric_key == METRIC_IOU or metric_key.startswith("METRIC_RANGE_IOU_"):
            mean_all: torch.Tensor = (
                metric_result.sum_class_metric / metric_result.counts_metric).mean()
            mean_all_value: float = mean_all.item() * 100.0
            # if metric_key == METRIC_RANGE_IOU:
                # print(f"Range IoU: {metric_result.sum_class_metric}")
                # print(f"Counts: {metric_result.counts_metric}")
                # print()
        else:
            mean_all: torch.Tensor = metric_result.sum_class_metric.sum() / \
                metric_result.counts_metric.sum()

            if metric_key == METRIC_RMSE:
                mean_all_value: float = torch.sqrt(
                    mean_all).item() * range_norm
            else:
                mean_all_value: float = mean_all.item()

        excel_sheet.cell(row=row, column=2, value=mean_all_value)
        if isinstance(result_dict, dict):
            result_dict[metric_key] = mean_all_value

        if metric_result.use_class is True:
            col: int = 3

            if metric_result.use_class is True:
                if metric_result.counts_metric.ndim == 2:
                    mean_class: torch.Tensor = metric_result.sum_class_metric / \
                        metric_result.counts_metric.sum(dim=0)
                else:
                    mean_class: torch.Tensor = metric_result.sum_class_metric / \
                        metric_result.counts_metric

                dynamic_indices = [int(cls) for cls in dynamic_class]
                dynamic_sum = sum(metric_result.sum_class_metric[idx] for idx in dynamic_indices)
                if metric_key == METRIC_IOU or metric_key.startswith("METRIC_RANGE_IOU_"):
                    dynamic_count = sum(metric_result.counts_metric[idx] for idx in dynamic_indices)
                    if dynamic_count > 0:
                        dynamic_class_metric = dynamic_sum / dynamic_count
                    else:
                        dynamic_class_metric = torch.tensor(0.0, device=dynamic_sum.device)
                else:
                    dynamic_count = sum(metric_result.counts_metric[:, idx].sum() for idx in dynamic_indices)
                    if metric_key == METRIC_RMSE:
                        dynamic_class_metric = torch.sqrt(dynamic_sum / dynamic_count) * range_norm
                    else:
                        dynamic_class_metric = dynamic_sum / dynamic_count

                static_indices = [int(cls) for cls in static_class]
                static_sum = sum(metric_result.sum_class_metric[idx] for idx in static_indices)
                if metric_key == METRIC_IOU or metric_key.startswith("METRIC_RANGE_IOU_"):
                    static_count = sum(metric_result.counts_metric[idx] for idx in static_indices)
                    if static_count > 0:
                        static_class_metric = static_sum / static_count
                    else:
                        static_class_metric = torch.tensor(0.0, device=static_sum.device)
                else:
                    static_count = sum(metric_result.counts_metric[:, idx].sum() for idx in static_indices)
                    if metric_key == METRIC_RMSE:
                        static_class_metric = torch.sqrt(static_sum / static_count) * range_norm
                    else:
                        static_class_metric = static_sum / static_count
                excel_sheet.cell(row=row, column=col, value=dynamic_class_metric.item())
                col += 1
                excel_sheet.cell(row=row, column=col, value=static_class_metric.item())
                col += 1

                for i in range(mean_class.shape[0]):
                    if metric_key == METRIC_RMSE:
                        excel_sheet.cell(row=row, column=col, value=torch.sqrt(
                            mean_class[i]).item() * range_norm)
                    elif metric_key == METRIC_IOU or metric_key.startswith("METRIC_RANGE_IOU_"):
                        excel_sheet.cell(row=row, column=col,
                                         value=mean_class[i].item() * 100.0)
                    else:
                        excel_sheet.cell(row=row, column=col,
                                         value=mean_class[i].item())
                    col += 1

            if metric_result.use_thresholds is True:
                if metric_result.counts_metric.ndim == 2:
                    mean_threshold: torch.Tensor = metric_result.sum_threshold_metric / \
                        metric_result.counts_metric.sum(dim=1)
                else:
                    mean_threshold: torch.Tensor = metric_result.sum_threshold_metric / \
                        metric_result.counts_metric

                for i in range(mean_threshold.shape[0]):
                    if metric_key == METRIC_RMSE:
                        excel_sheet.cell(row=row, column=col, value=torch.sqrt(
                            mean_threshold[i]).item() * range_norm)
                    else:
                        excel_sheet.cell(row=row, column=col,
                                         value=mean_threshold[i].item())
                    col += 1

def process_class_data(data_t, range_norm, csv_file, class_indices, total_value, total_count):
    """
    クラスごとのデータを処理し、結果をCSVに保存する関数 t検定用
    """
    all_non_zero_values = []
    for class_idx in class_indices:
        pixel_flattened = (data_t.sum.sum(dim=1)[:, class_idx, :, :] * range_norm).cpu().numpy().flatten()
        non_zero_values = pixel_flattened[pixel_flattened != 0]

        total_value += non_zero_values.sum()
        total_count += len(non_zero_values)

        all_non_zero_values.extend(non_zero_values)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in all_non_zero_values])

    return total_value, total_count

def tensor_to_str(tensor: torch.Tensor) -> str:
    return f"{tensor.item()}"

def main(args: dict):
    # Device Setting
    if torch.cuda.is_available():
        device: torch.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device: torch.device = torch.device('cpu')
        print(f'{"Device":11s}: CPU')
    
    args = parse_args()

    if args['version'] in ['kitti', 'real']:
        class_dict = {
            '0': 'Void',
            '1': 'Ground',
            '2': 'Obstacle',
            '3': 'Vehicle',
            '4': 'Person'
        }
        dynamic_class = ['3', '4']
        static_class = [key for key in class_dict.keys() if key not in dynamic_class]
    else:
        class_dict = {
            '0': 'Unlabeled',
            '1': 'Building',
            '2': 'Fence',
            '3': 'Other',
            '4': 'Pedestrian',
            '5': 'Pole',
            '6': 'RoadLine',
            '7': 'Road',
            '8': 'SideWalk',
            '9': 'Vegetation',
            '10': 'Vehicles',
            '11': 'Wall',
            '12': 'TrafficSign',
            '13': 'Sky',
            '14': 'TrafficLight'
        }
        dynamic_class = ['4', '10']
        static_class = [key for key in class_dict.keys() if key not in dynamic_class]

    # Metric Setting
    if args['version'] in ['kitti', 'real']:
        num_classes = 5
    else:
        num_classes = 15
    range_norm: float = args['range_norm']
    thresholds: List[float] = args['thresholds']
    normalized_thresholds: List[float] = [th / range_norm for th in thresholds]

    metricIntersection: nn.Module = MetricIntersection().to(device)
    metricUnion: nn.Module = MetricUnion().to(device)
    metricBatchSumAE: nn.Module = MetricBatchSumAE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    metricBatchSumSE: nn.Module = MetricBatchSumSE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    metricBatchSumAPE: nn.Module = MetricBatchSumAPE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    

    metric_results: Dict[str, METRIC_RESULT] = {
        # METRIC_IOU: METRIC_RESULT(name='IoU [%]', use_class=True, all_tag='Mean'),
        METRIC_MAE: METRIC_RESULT(name='MAE [m]', use_class=True, use_thresholds=True),
        METRIC_RMSE: METRIC_RESULT(name='RMSE [m]', use_class=True, use_thresholds=True),
        METRIC_MAPE: METRIC_RESULT(name='MAPE', use_class=True, use_thresholds=True)
    }
    
    # for threshold in thresholds:
    #     metric_key = f"METRIC_RANGE_IOU_{int(threshold)}"
    #     metric_results[metric_key] = METRIC_RESULT(
    #         name=f"Range IoU {int(threshold)} [%]", use_class=True, all_tag='Mean'
    #     )

    # for key, value in metric_results.items():
    #     print(f"{key}: {value.name}")

    result_dir: str = args['output']
    print(f'{"Result Directory":11s}: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)

    # Excel Book
    eval_book = xl.Workbook()
    eval_sheet_overview = eval_book.worksheets[0]

    eval_sheet_overview.title = 'Overview'
    eval_sheet_overview.cell(row=1, column=1, value='Metric')
    eval_sheet_overview.cell(row=1, column=2, value='All')
    eval_sheet_overview.cell(row=1, column=3, value='Dynamic')
    eval_sheet_overview.cell(row=1, column=4, value='Static')
    for col, (label, tag) in enumerate(class_dict.items(), 5):
        eval_sheet_overview.cell(row=1, column=col, value=f'{label}:{tag}')
    tmp_thresholds = [0.0] + args['thresholds']
    for col, (th_min, th_max) in enumerate(zip(tmp_thresholds[:-1], tmp_thresholds[1:]), col+1):
        eval_sheet_overview.cell(
            row=1, column=col, value=f'{th_min}~{th_max}[m]')

    eval_sheet_detail = eval_book.create_sheet('Detail')
    eval_sheet_detail.cell(row=1, column=1, value='Step')
    eval_sheet_detail.merge_cells(
        start_row=1, end_row=2, start_column=1, end_column=1)

    metric_col = 2
    for metric_result in metric_results.values():
        eval_sheet_detail.cell(row=1, column=metric_col,
                               value=metric_result.name)
        if metric_result.use_class is True:
            cols: int = len(class_dict) + 1
            eval_sheet_detail.merge_cells(
                start_row=1, end_row=1, start_column=metric_col, end_column=metric_col+cols-1)
            eval_sheet_detail.cell(
                row=2, column=metric_col, value=metric_result.all_tag)
            for col, (label, tag) in enumerate(class_dict.items(), metric_col + 1):
                eval_sheet_detail.cell(
                    row=2, column=col, value=f'{label}:{tag}')
        else:
            cols: int = 1
            eval_sheet_detail.merge_cells(
                start_row=1, end_row=2, start_column=metric_col, end_column=metric_col)
        metric_col += cols

    itr: int = 0
    result_dict: Dict[str, Union[float, str]] = {DIR_RESULTS: result_dir}
    # for eval_data_path in args['eval_data']:
    #     with h5py.File(eval_data_path, 'r') as f:
    # for eval_data_dir in args['eval_data']:

#####################################################################################################################
# t検定用
    # total_vehicle_ae = 0
    # total_vehicle_ae_cnts = 0.0
    # total_pedestrian_ae = 0.0
    # total_pedestrian_ae_cnts = 0.0
    # total_static_ae = 0.0
    # total_static_ae_cnts = 0.0
    # total_vehicle_ape = 0.0
    # total_vehicle_ape_cnts = 0.0
    # total_pedestrian_ape = 0.0
    # total_pedestrian_ape_cnts = 0.0
    # total_static_ape = 0.0
    # total_static_ape_cnts = 0.0

    # vehicle_class_idx = [3]
    # pedestrian_class_idx = [4]
    # static_class_idx = [0, 1, 2]
 
    # csv_dir = os.path.join(result_dir, "csv_outputs")
    # os.makedirs(csv_dir, exist_ok=True)
 
    # csv_files = {
    #     "vehicle_ae": os.path.join(csv_dir, "vehicle_ae.csv"),
    #     "vehicle_ape": os.path.join(csv_dir, "vehicle_ape.csv"),
    #     "pedestrian_ae": os.path.join(csv_dir, "pedestrian_ae.csv"),
    #     "pedestrian_ape": os.path.join(csv_dir, "pedestrian_ape.csv"),
    #     "static_ae": os.path.join(csv_dir, "static_ae.csv"),
    #     "static_ape": os.path.join(csv_dir, "static_ape.csv")
    # }
 
    # for key, file_name in csv_files.items():
    #     with open(file_name, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    # FP用
    # total_false_positive_numerator = 0
    # total_false_positive_denominator = 0
#####################################################################################################################

    hdf5_files = [os.path.join(args['eval_data'], file) for file in os.listdir(args['eval_data']) if file.endswith('.hdf5')]
    for eval_data_path in hdf5_files:
        with h5py.File(eval_data_path, 'r') as f:
            print(f"Processing file: {eval_data_path}")
            num = len(list(f['data'].keys()))

            for i in track(range(num)):
                # Eval Data Setting
                
                if args['type'] == 'pred':
                    depth = torch.from_numpy(f['data'][str(i)]['Pred-Depth'][()]/range_norm).to(device).unsqueeze(0)    
                else:
                    depth = torch.from_numpy(f['data'][str(i)]['Input-FBX-Map'][()]/range_norm).to(device).unsqueeze(0)
                gt_depth = torch.from_numpy(f['data'][str(i)]['GT-Depth'][()]/range_norm).to(device).unsqueeze(0)
                gt_label = torch.from_numpy(f['data'][str(i)]['GT-Label'][()]).to(device).unsqueeze(0)
                # label = torch.from_numpy(f['data'][str(i)]['Pred-Label'][()]).to(device).unsqueeze(0)
                # label = F.one_hot(label.squeeze(0).to(torch.int64), num_classes=num_classes)
                # label = label.permute(2, 0, 1).unsqueeze(0)

                # False Positiveの計算　
                # # (gt_labelでclass7(Road)である部分に対して、labelがclass4(Pedestrian)orclass10(Vehicles)であるpx数の総和) / (gt_labelでclass7(Road)であるpxの総和)
                # if args['version'] in ['kitti', 'real']:
                #     road_mask = (gt_label == 1)
                #     # print(f"Road Mask: {road_mask}")
                #     vehicle_person_mask = (_label == 3) | (_label == 4)
                #     # print(f"Vehicle/Person Mask: {vehicle_person_mask}")
                # else:
                #     road_mask = (gt_label == 7)
                #     # print(f"Road Mask: {road_mask}")
                #     vehicle_person_mask = (_label == 4) | (_label == 10)
                #     # print(f"Vehicle/Person Mask: {vehicle_person_mask}")
                # false_positive_numerator = (road_mask & vehicle_person_mask).sum().item()
                # # print(f"False Positive Numerator: {false_positive_numerator}")
                # false_positive_denominator = road_mask.sum().item()
                # # print(f"False Positive Denominator: {false_positive_denominator}")
                # total_false_positive_numerator += false_positive_numerator
                # total_false_positive_denominator += false_positive_denominator

                # AE/MAE
                
                ae_t, ae = metricBatchSumAE(
                    depth, gt_depth, gt_label)
                metric_results[METRIC_MAE].class_metric = ae.sum.sum(
                    dim=1) / ae.count.sum(dim=1) * range_norm
                metric_results[METRIC_MAE].threshold_metric = ae.sum.sum(
                    dim=2) / ae.count.sum(dim=2) * range_norm
                metric_results[METRIC_MAE].all_metric = ae.sum.sum(
                    dim=(1, 2)) / ae.count.sum(dim=(1, 2)) * range_norm
                metric_results[METRIC_MAE].add(ae.count.sum(dim=0), ae.sum.sum(
                    dim=(0, 1)) * range_norm, ae.sum.sum(dim=(0, 2)) * range_norm)
                
                # SE/RMSE
                se: METRIC_SUM_OUTPUT = metricBatchSumSE(
                    depth, gt_depth, gt_label)
                metric_results[METRIC_RMSE].class_metric = torch.sqrt(
                    se.sum.sum(dim=1) / se.count.sum(dim=1)) * range_norm
                metric_results[METRIC_RMSE].threshold_metric = torch.sqrt(
                    se.sum.sum(dim=2) / se.count.sum(dim=2)) * range_norm
                metric_results[METRIC_RMSE].all_metric = torch.sqrt(
                    se.sum.sum(dim=(1, 2)) / se.count.sum(dim=(1, 2))) * range_norm
                metric_results[METRIC_RMSE].add(se.count.sum(
                    dim=0), se.sum.sum(dim=(0, 1)), se.sum.sum(dim=(0, 2)))

                # APE/MAPE
                ape_t, ape = metricBatchSumAPE(
                    depth, gt_depth, gt_label)
                metric_results[METRIC_MAPE].class_metric = ape.sum.sum(
                    dim=1) / ape.count.sum(dim=1)
                metric_results[METRIC_MAPE].threshold_metric = ape.sum.sum(
                    dim=2) / ape.count.sum(dim=2)
                metric_results[METRIC_MAPE].all_metric = ape.sum.sum(
                    dim=(1, 2)) / ape.count.sum(dim=(1, 2))
                metric_results[METRIC_MAPE].add(ape.count.sum(
                    dim=0), ape.sum.sum(dim=(0, 1)), ape.sum.sum(dim=(0, 2)))

                # IoU
                # print(label.shape, gt_label.shape, type(label), type(gt_label))
                # intersection: torch.Tensor = metricIntersection(
                #     label, gt_label)
                # union: torch.Tensor = metricUnion(label, gt_label)
                # metric_results[METRIC_IOU].class_metric = intersection / union
                # metric_results[METRIC_IOU].all_metric = metric_results[METRIC_IOU].class_metric.mean(
                #     dim=1)
                # metric_results[METRIC_IOU].add(
                #     union.sum(dim=0), intersection.sum(dim=0))

#####################################################################################################################
                # IoU (Range)
                # 閾値作成用に拡張されたしきい値リスト
                # extended_thresholds = [0.0] + thresholds

                # 各しきい値ごとに処理
                # for t_idx, threshold in enumerate(extended_thresholds[:-1]):
                    # next_threshold = extended_thresholds[t_idx + 1]
                    # metric_key = f"METRIC_RANGE_IOU_{int(next_threshold)}"  # 動的にキーを作成

                    # 距離範囲に対応するマスクを作成　depthは正規化されているのでrange_normを掛ける
                    # range_mask = (gt_depth * range_norm >= threshold) & (gt_depth * range_norm < next_threshold)

                    # マスクされたピクセル数を計算 ここまで合ってる
                    # num_masked_pixels = range_mask.sum().item()
                    # print(f"  Number of masked pixels: {num_masked_pixels}")

                    # gt_filtered = gt_label * range_mask
                    # label_filtered = label * range_mask

                    # 確認用：元の gt_label の総ピクセル数と gt_filtered の有効ピクセル数を比較
                    # total_pixels = gt_label.numel()
                    # filtered_pixels = (gt_filtered > 0).sum().item()
                    # print(f"  Total pixels in gt_label: {total_pixels}")
                    # print(f"  Non-zero pixels in gt_filtered: {filtered_pixels}")
                    # print(f"  Percentage of pixels retained: {filtered_pixels / total_pixels * 100:.2f}%")

                    # intersection_range: torch.Tensor = metricIntersection(label_filtered, gt_filtered)
                    # union_range: torch.Tensor = metricUnion(label_filtered, gt_filtered)
                    # iou_range = intersection_range / (union_range + 1e-10)

                    # metric_results[metric_key].class_metric = iou_range
                    # metric_results[metric_key].all_metric = iou_range.mean(dim=1)
                    # metric_results[metric_key].add(
                        # union_range.sum(dim=0), intersection_range.sum(dim=0)
                    # )
                
                #一旦t検定用の脳筋csv保存コード
                #vehicle ae
                #flattenしてcsvに書き込み(データの線形化)t検定用に1行にしておく
                #行数＝px数 -> total_~_cntsは確認用
                # # Vehicle AE
                # total_vehicle_ae, total_vehicle_ae_cnts = process_class_data(
                #     ae_t, range_norm, csv_files["vehicle_ae"], vehicle_class_idx, total_vehicle_ae, total_vehicle_ae_cnts
                # )

                # # Pedestrian AE
                # total_pedestrian_ae, total_pedestrian_ae_cnts = process_class_data(
                #     ae_t, range_norm, csv_files["pedestrian_ae"], pedestrian_class_idx, total_pedestrian_ae, total_pedestrian_ae_cnts
                # )

                # # Static AE
                # total_static_ae, total_static_ae_cnts = process_class_data(
                #     ae_t, range_norm, csv_files["static_ae"], static_class_idx, total_static_ae, total_static_ae_cnts
                # )

                # # Vehicle APE
                # total_vehicle_ape, total_vehicle_ape_cnts = process_class_data(
                #     ape_t, 1.0, csv_files["vehicle_ape"], vehicle_class_idx, total_vehicle_ape, total_vehicle_ape_cnts
                # )

                # # Pedestrian APE
                # total_pedestrian_ape, total_pedestrian_ape_cnts = process_class_data(
                #     ape_t, 1.0, csv_files["pedestrian_ape"], pedestrian_class_idx, total_pedestrian_ape, total_pedestrian_ape_cnts
                # )

                # # Static APE
                # total_static_ape, total_static_ape_cnts = process_class_data(
                #     ape_t, 1.0, csv_files["static_ape"], static_class_idx, total_static_ape, total_static_ape_cnts
                # )
                
    # FP結果の出力確認
    # if total_false_positive_denominator > 0:
    #     final_false_positive_ratio = total_false_positive_numerator / total_false_positive_denominator
    #     print(total_false_positive_numerator, total_false_positive_denominator)
    # else:
    #     final_false_positive_ratio = 0.0
    # print(f"Final False Positive Ratio: {final_false_positive_ratio:.10f}")
    
    #　確認用　MAE問題なしOK
    # final_vehicle_ae = total_vehicle_ae / total_vehicle_ae_cnts
    # print(f"Final Vehicle AE: {final_vehicle_ae:.4f}")
    # final_pedestrian_ae = total_pedestrian_ae / total_pedestrian_ae_cnts
    # print(f"Final Pedestrian AE: {final_pedestrian_ae:.4f}")
    # final_static_ae = total_static_ae / total_static_ae_cnts
    # print(f"Final Static AE: {final_static_ae:.4f}")
    # #確認用　APE問題なしOK
    # final_vehicle_ape = total_vehicle_ape / total_vehicle_ape_cnts
    # print(f"Final Vehicle APE: {final_vehicle_ape:.4f}")
    # final_pedestrian_ape = total_pedestrian_ape / total_pedestrian_ape_cnts
    # print(f"Final Pedestrian APE: {final_pedestrian_ape:.4f}")
    # final_static_ape = total_static_ape / total_static_ape_cnts
    # print(f"Final Static APE: {final_static_ape:.4f}")
#####################################################################################################################

    # -> Excel (Overview)
    _excel_overview_output(
        metric_results, eval_sheet_overview, range_norm, result_dict, dynamic_class, static_class)
    # print(
    #     f'{"":11s}: {"Metric":14s}: {"mIoU":14s}:{result_dict[METRIC_IOU]:9.2f} [ % ]')
    print(
        f'{"":11s}: {"":14s}: {"MAE":14s}:{result_dict[METRIC_MAE]:9.4f} [ m ]')
    print(
        f'{"":11s}: {"":14s}: {"RMSE":14s}:{result_dict[METRIC_RMSE]:9.4f} [ m ]')
    print(
        f'{"":11s}: {"":14s}: {"MAPE":14s}:{result_dict[METRIC_MAPE]:9.4f}')

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_name = str(args['eval_data']).split('/')[0]
    filename = f'result_{current_time}_{data_name}.xlsx'
    eval_book.save(os.path.join(result_dir, filename))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--eval_data', type=str, metavar='DIR', required=True, help='Path to a directory containing HDF5 files for evaluation.')
    parser.add_argument('-o', '--output', type=str, metavar='PATH', default='result_dir',required=False, help='result Directory.')
    parser.add_argument('-v', '--version', type=str, default='kitti',required=False, help='data version.')
    parser.add_argument('-t', '--type', type=str, default='pred',required=False, help='data type.')
    parser.add_argument('-r', '--range_norm',type=float, default=80.0, help='norm of range.')
    parser.add_argument('-th', '--thresholds',type=float, nargs='+', default=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0], help='Thresholds of depth.')
    args: dict = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    main()


