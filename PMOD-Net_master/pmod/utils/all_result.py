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

METRIC_MAE = "MAE"
METRIC_RMSE = "RMSE"
METRIC_MAPE = "MAPE"

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

class MetricInRange(nn.Module):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricInRange, self).__init__()

        thresholds.sort()
        self.thresholds: Tensor
        self.register_buffer('thresholds', torch.tensor(thresholds))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        _range_list: List[Tensor] = []
        _prev_th: float = 0.0
        seg_gt = seg_gt.long()
        if self.num_classes > 0:
            _one_hot: Tensor = F.one_hot(seg_gt, num_classes=self.num_classes).bool()
        for th_idx in range(self.thresholds.shape[0]):
            _th: Tensor = self.thresholds[th_idx].to(gt.device)
            if _th <= _prev_th:
                continue
            _in_range: Tensor = (_prev_th < gt) & (gt <= _th) & (gt <= 1.0)
            if self.num_classes > 0:
                _in_range: Tensor = torch.squeeze(_in_range, dim=1)
                _in_range_oh: Tensor = _one_hot.clone()
                _in_range_oh[~_in_range] = False
                _range_list.append(_in_range_oh.bool().permute(
                    0, 3, 1, 2).unsqueeze(dim=1))
            else:
                _range_list.append(_in_range.unsqueeze(dim=1))
            _prev_th = _th.clone()

        range_cat: Tensor = torch.cat(_range_list, dim=1)
        range_cnt: Tensor = range_cat.sum(
            dim=(3, 4)) if self.num_classes > 0 else range_cat.sum(dim=(2, 3, 4))
        return range_cat, range_cnt

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
            return METRIC_SUM_OUTPUT(ape.sum(dim=(3, 4)), cnts)
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

        ae = ae.unsqueeze(dim=1).expand_as(in_range)
        zeros = torch.zeros_like(ae)
        ae = torch.where(in_range, ae, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(3, 4)), cnts)
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
    
def _excel_detail_output(idx: int, batch_size: int, metric_results: Dict[str, METRIC_RESULT], excel_sheet: Worksheet) -> int:
    itr: int = idx
    for i in range(batch_size):
        row: int = itr + 3
        col: int = 1
        excel_sheet.cell(row=row, column=col, value=itr)
        col += 1
        for metric_key, metric_result in metric_results.items():
            value: float = metric_result.all_metric[i].item()
            excel_sheet.cell(row=row, column=col, value=value)
            col += 1
            if metric_result.use_class is True:
                for j in range(metric_result.class_metric.shape[1]):
                    value: float = metric_result.class_metric[i, j].item()
                    excel_sheet.cell(row=row, column=col, value=value)
                    col += 1
        itr += 1
    return itr

def _excel_overview_output(metric_results: Dict[str, METRIC_RESULT], excel_sheet: Worksheet, range_norm: float, result_dict: Dict[str, float] = None, dynamic_class: List[str] = None, static_class: List[str] = None) -> None:
    for row, (metric_key, metric_result) in enumerate(metric_results.items(), 2):
        excel_sheet.cell(row=row, column=1, value=metric_result.name)
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
                dynamic_count = sum(metric_result.counts_metric[:, idx].sum() for idx in dynamic_indices)
                if metric_key == METRIC_RMSE:
                    dynamic_class_metric = torch.sqrt(dynamic_sum / dynamic_count) * range_norm
                else:    
                    dynamic_class_metric = dynamic_sum / dynamic_count

                static_indices = [int(cls) for cls in static_class]
                static_sum = sum(metric_result.sum_class_metric[idx] for idx in static_indices)
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

def main(args: dict):
    # Device Setting
    if torch.cuda.is_available():
        device: torch.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device: torch.device = torch.device('cpu')
        print(f'{"Device":11s}: CPU')
    
    args = parse_args()

    if args['version'] == 'kitti':
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
    if args['version'] == 'kitti':
        num_classes = 5
    else:
        num_classes = 15
    range_norm: float = args['range_norm']
    thresholds: List[float] = args['thresholds']
    normalized_thresholds: List[float] = [th / range_norm for th in thresholds]

    metricBatchSumAE: nn.Module = MetricBatchSumAE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    metricBatchSumSE: nn.Module = MetricBatchSumSE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    metricBatchSumAPE: nn.Module = MetricBatchSumAPE(
        thresholds=normalized_thresholds, num_classes=num_classes).to(device)
    
    # #class
    # metricBatchSumVehicle_Pos: nn.Module = MetricBatchSumVehicle_Pos(
    #     thresholds=normalized_thresholds, num_classes=num_classes).to(device)

    metric_results: Dict[str, METRIC_RESULT] = {
        METRIC_MAE: METRIC_RESULT(name='MAE [m]', use_class=True, use_thresholds=True),
        METRIC_RMSE: METRIC_RESULT(name='RMSE [m]', use_class=True, use_thresholds=True),
        METRIC_MAPE: METRIC_RESULT(name='MAPE', use_class=True, use_thresholds=True),
        # METRIC_VEHICLE_POSE: METRIC_RESULT(name='Vehicle_Pos', use_class=True, use_thresholds=True),
    }

    result_dir: str = args['output']
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

    with h5py.File(args['eval_data'], 'r') as f:
        num = len(list(f['data'].keys()))
        itr: int = 0

        result_dict: Dict[str, Union[float, str]] = {DIR_RESULTS: result_dir}
        for i in track(range(num)):
            # Eval Data Setting
            
            if args['type'] == 'pred':
                depth = torch.from_numpy(f['data'][str(i)]['Pred-Depth'][()]/80).to(device).unsqueeze(0)    
            else:
                depth = torch.from_numpy(f['data'][str(i)]['Input-Mono_Depth'][()]/80).to(device).unsqueeze(0)
            
            gt_depth = torch.from_numpy(f['data'][str(i)]['GT-Depth'][()]/80).to(device).unsqueeze(0)
            gt_label = torch.from_numpy(f['data'][str(i)]['GT-Label'][()]).to(device).unsqueeze(0)

            # AE/MAE
            ae: METRIC_SUM_OUTPUT = metricBatchSumAE(
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
            ape: METRIC_SUM_OUTPUT = metricBatchSumAPE(
                depth, gt_depth, gt_label)
            metric_results[METRIC_MAPE].class_metric = ape.sum.sum(
                dim=1) / ape.count.sum(dim=1)
            metric_results[METRIC_MAPE].threshold_metric = ape.sum.sum(
                dim=2) / ape.count.sum(dim=2)
            metric_results[METRIC_MAPE].all_metric = ape.sum.sum(
                dim=(1, 2)) / ape.count.sum(dim=(1, 2))
            metric_results[METRIC_MAPE].add(ape.count.sum(
                dim=0), ape.sum.sum(dim=(0, 1)), ape.sum.sum(dim=(0, 2)))
            
            # vehicle_pos: METRIC_SUM_OUTPUT = metricBatchSumVehicle_Pos(
            #     depth, gt_depth, gt_label)
            # metric_results[METRIC_VEHICLE_POSE].class_metric = vehicle_pos.sum.sum(
            #     dim=1) / vehicle_pos.count.sum(dim=1) * range_norm
            # metric_results[METRIC_VEHICLE_POSE].threshold_metric = vehicle_pos.sum.sum(
            #     dim=2) / vehicle_pos.count.sum(dim=2) * range_norm
            # metric_results[METRIC_VEHICLE_POSE].all_metric = vehicle_pos.sum.sum(
            #     dim=(1, 2)) / vehicle_pos.count.sum(dim=(1, 2)) * range_norm
            # metric_results[METRIC_VEHICLE_POSE].add(vehicle_pos.count.sum(dim=0), vehicle_pos.sum.sum(
                # dim=(0, 1)) * range_norm, vehicle_pos.sum.sum(dim=(0, 2)) * range_norm)
            
            itr = _excel_detail_output(
                idx=itr, batch_size=1, metric_results=metric_results, excel_sheet=eval_sheet_detail)

        # -> Excel (Overview)
        _excel_overview_output(
            metric_results, eval_sheet_overview, range_norm, result_dict, dynamic_class, static_class)
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
    parser.add_argument('-d', '--eval_data', type=str, metavar='PATH', required=True, help='Root directory of eval data path.')
    parser.add_argument('-o', '--output', type=str, metavar='PATH', default='result_dir',required=False, help='result Directory.')
    parser.add_argument('-v', '--version', type=str, default='kitti',required=False, help='data version.')
    parser.add_argument('-t', '--type', type=str, default='pred',required=False, help='data type.')
    parser.add_argument('-r', '--range_norm',type=float, default=80.0, help='norm of range.')
    parser.add_argument('-th', '--thresholds',type=float, nargs='+', default=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,30.0, 40.0, 50.0, 60.0, 70.0, 80.0], help='Thresholds of depth.')
    args: dict = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    main()


