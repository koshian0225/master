# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
from tqdm import tqdm
import hydra
from PIL import Image
import os
from omegaconf import OmegaConf
from utils import *
import warnings
from models.self_metric import *
from openpyxl.worksheet.worksheet import Worksheet
import openpyxl as xl
import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

METRIC_MAE = "MAE"
METRIC_RMSE = "RMSE"
METRIC_MAPE = "MAPE"
METRIC_IOU = "IoU"
DIR_RESULTS = 'results'

def _excel_overview_output(metric_results: Dict[str, METRIC_RESULT], excel_sheet: Worksheet, range_norm: float, result_dict: Dict[str, float] = None, dynamic_class: List[str] = None, static_class: List[str] = None) -> None:
    for row, (metric_key, metric_result) in enumerate(metric_results.items(), 2):
        excel_sheet.cell(row=row, column=1, value=metric_result.name)

        if metric_key == METRIC_IOU or metric_key.startswith("METRIC_RANGE_IOU_"):
            mean_all: torch.Tensor = (
                metric_result.sum_class_metric / metric_result.counts_metric).mean()
            mean_all_value: float = mean_all.item() * 100.0
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

# def _hdf5_output(h5_saver: H5Dataset, rgb, in_map, gt_depth, gt_label, pred_depth, pred_segmentation, version):
#     if version == 'carla':
#         label_tag: str = "carla-0.9.14"
#         frame_id_camera = "front"
#         frame_id_map = "front"
#         frame_id_label = "front"
#         frame_id_depth = "front"
#         frame_id_segmentation = "front"
#     elif version == 'kitti360':
#         label_tag: str = "5class"
#         frame_id_camera = "cam0"
#         frame_id_map = "cam0"
#         frame_id_label = "cam0"
#         frame_id_depth = "cam0"
#         frame_id_segmentation = "cam0" 
#     depth_norm_range_max: float = 80.0
#     data_group = h5_saver.get_next_data_group()
#     converter = LabelConverter(version)
#     pred_seg = converter.convert(pred_segmentation)
#     set_bgr8(data_group, 'Input-Camera',
#                 rgb,
#                 frame_id_camera
#                 )
#     set_depth(data_group, 'Input-Map',
#                 in_map * depth_norm_range_max,
#                 frame_id_map
#                 )
#     set_semantic2d(data_group, 'GT-Label',
#                     gt_label,
#                     frame_id_label,
#                     label_tag
#                     )
#     set_depth(data_group, 'GT-Depth',
#                 gt_depth * depth_norm_range_max,
#                 frame_id_depth
#                 )
#     set_depth(data_group, 'Pred-Depth',
#                 pred_depth * depth_norm_range_max,
#                 frame_id_depth
#                 )
#     set_semantic2d(data_group, 'Pred-Label',
#             pred_seg.argmax(axis=0).astype(np.uint8),
#             frame_id_segmentation,
#             label_tag
#             )

def test(run, mode='selval', save=False):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()  
    now = datetime.datetime.now().strftime("%m-%d-%H%M%S")
    dir_path = f"results/{run.cfg.name}-{now}/"
    os.makedirs(dir_path, exist_ok=True)    
    if run.cfg.version == 'carla':
        label_tag = "carla-0.9.14"
        # ラベルの変更に伴い、以下のコードは不要
        # class_dict = {
        #     '0': 'Unlabeled', '1': 'Building', '2': 'Fence', '3': 'Other', '4': 'Pedestrian',
        #     '5': 'Pole', '6': 'RoadLine', '7': 'Road', '8': 'SideWalk', '9': 'Vegetation',
        #     '10': 'Vehicles', '11': 'Wall', '12': 'TrafficSign', '13': 'Sky', '14': 'TrafficLight'
        # }
        # label_color_configs = [
        #     {'tag': tag, 'label': int(label), 'color': np.array(color, dtype=np.uint8)}
        #     for label, (tag, color) in zip(class_dict.keys(), [
        #         ('Unlabeled', [0, 0, 0]), ('Building', [70, 70, 70]), ('Fence', [40, 40, 100]),
        #         ('Other', [80, 90, 55]), ('Pedestrian', [60, 20, 220]), ('Pole', [153, 153, 153]),
        #         ('RoadLine', [50, 234, 157]), ('Road', [128, 64, 128]), ('SideWalk', [232, 35, 244]),
        #         ('Vegetation', [35, 142, 107]), ('Vehicles', [142, 0, 0]), ('Wall', [156, 102, 102]),
        #         ('TrafficSign', [0, 220, 220]), ('Sky', [180, 130, 70]), ('TrafficLight', [30, 170, 250])
        #     ])
        # ]
        # dynamic_class = ['4', '10']
        # static_class = [key for key in class_dict if key not in dynamic_class]
        # num_classes = 15
        # KITTI-360の5クラスに合わせる
        class_dict = {
            '0': 'Void', '1': 'Ground', '2': 'Obstacle', '3': 'Vehicle', '4': 'Person'
        }
        label_color_configs = [
            {'tag': tag, 'label': int(label), 'color': np.array(color, dtype=np.uint8)}
            for label, (tag, color) in zip(class_dict.keys(), [
                ('Void', [0, 0, 0]), ('Ground', [180, 130, 70]), ('Obstacle', [70, 70, 70]),
                ('Vehicle', [142, 0, 0]), ('Person', [60, 20, 220])
            ])
        ]
        dynamic_class = ['3', '4']
        static_class = [key for key in class_dict if key not in dynamic_class]
        num_classes = 5
    elif run.cfg.version == 'kitti360':
        label_tag = "5class"
        class_dict = {
            '0': 'Void', '1': 'Ground', '2': 'Obstacle', '3': 'Vehicle', '4': 'Person'
        }
        label_color_configs = [
            {'tag': tag, 'label': int(label), 'color': np.array(color, dtype=np.uint8)}
            for label, (tag, color) in zip(class_dict.keys(), [
                ('Void', [0, 0, 0]), ('Ground', [180, 130, 70]), ('Obstacle', [70, 70, 70]),
                ('Vehicle', [142, 0, 0]), ('Person', [60, 20, 220])
            ])
        ]
        dynamic_class = ['3', '4']
        static_class = [key for key in class_dict if key not in dynamic_class]
        num_classes = 5
    if save:
        filename = f"{run.cfg.name}.hdf5"
        h5_saver = H5Dataset(path=os.path.join(dir_path, filename))
        label_group = h5_saver.get_label_group(label_tag)
        for label_config in label_color_configs:
            bgr: np.ndarray = label_config['color']
            set_label_config(
                label_group,
                label_config['label'],
                label_config['tag'],
                bgr[2], bgr[1], bgr[0]
            )
    
    range_norm: float = 80.0
    thresholds: List[float] = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    normalized_thresholds: List[float] = [th / range_norm for th in thresholds]

    metricIntersection: nn.Module = MetricIntersection()
    metricUnion: nn.Module = MetricUnion()
    metricBatchSumAE: nn.Module = MetricBatchSumAE(
        thresholds=normalized_thresholds, num_classes=num_classes)
    metricBatchSumSE: nn.Module = MetricBatchSumSE(
        thresholds=normalized_thresholds, num_classes=num_classes)
    metricBatchSumAPE: nn.Module = MetricBatchSumAPE(
        thresholds=normalized_thresholds, num_classes=num_classes)

    metric_results: Dict[str, METRIC_RESULT] = {
        METRIC_IOU: METRIC_RESULT(name='IoU [%]', use_class=True, all_tag='Mean'),
        METRIC_MAE: METRIC_RESULT(name='MAE [m]', use_class=True, use_thresholds=True),
        METRIC_RMSE: METRIC_RESULT(name='RMSE [m]', use_class=True, use_thresholds=True),
        METRIC_MAPE: METRIC_RESULT(name='MAPE', use_class=True, use_thresholds=True)
    }

    eval_book = xl.Workbook()
    eval_sheet_overview = eval_book.worksheets[0]

    eval_sheet_overview.title = 'Overview'
    eval_sheet_overview.cell(row=1, column=1, value='Metric')
    eval_sheet_overview.cell(row=1, column=2, value='All')
    eval_sheet_overview.cell(row=1, column=3, value='Dynamic')
    eval_sheet_overview.cell(row=1, column=4, value='Static')
    for col, (label, tag) in enumerate(class_dict.items(), 5):
        eval_sheet_overview.cell(row=1, column=col, value=f'{label}:{tag}')
    tmp_thresholds = [0.0] + thresholds
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

    result_dict: Dict[str, Union[float, str]] = {DIR_RESULTS: dir_path}

    with torch.no_grad():
        for idx, datas in enumerate(
                tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank)):
            datas = run.init_cuda(*datas)
            output, pred_segmentation= net(datas[0],datas[1],datas[2], datas[5])  # 0:rgb, 1:in_map, 2:K_cam, 3:depth, 4:gt_label, 5:rgb_nd, 6:rgb_hdf5
            if isinstance(output, (list, tuple)):
                output = output[-1] /255.0
            if isinstance(pred_segmentation, (list, tuple)):
                pred_segmentation = pred_segmentation[0]

            gt_depth = datas[3] /255
            gt_label = datas[4].squeeze(1).to(torch.int64)  # 4:gt_label torch.Size([1, 256, 512]) torch.int64

            # output =nd

            # AE/MAE
            ae = metricBatchSumAE(
                output, gt_depth, gt_label)
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
                output, gt_depth, gt_label)
            metric_results[METRIC_RMSE].class_metric = torch.sqrt(
                se.sum.sum(dim=1) / se.count.sum(dim=1)) * range_norm
            metric_results[METRIC_RMSE].threshold_metric = torch.sqrt(
                se.sum.sum(dim=2) / se.count.sum(dim=2)) * range_norm
            metric_results[METRIC_RMSE].all_metric = torch.sqrt(
                se.sum.sum(dim=(1, 2)) / se.count.sum(dim=(1, 2))) * range_norm
            metric_results[METRIC_RMSE].add(se.count.sum(
                dim=0), se.sum.sum(dim=(0, 1)), se.sum.sum(dim=(0, 2)))

            # APE/MAPE
            ape = metricBatchSumAPE(
                output, gt_depth, gt_label)
            metric_results[METRIC_MAPE].class_metric = ape.sum.sum(
                dim=1) / ape.count.sum(dim=1)
            metric_results[METRIC_MAPE].threshold_metric = ape.sum.sum(
                dim=2) / ape.count.sum(dim=2)
            metric_results[METRIC_MAPE].all_metric = ape.sum.sum(
                dim=(1, 2)) / ape.count.sum(dim=(1, 2))
            metric_results[METRIC_MAPE].add(ape.count.sum(
                dim=0), ape.sum.sum(dim=(0, 1)), ape.sum.sum(dim=(0, 2)))
            
            # IoU
            intersection: torch.Tensor = metricIntersection(
                pred_segmentation, gt_label)
            union: torch.Tensor = metricUnion(pred_segmentation, gt_label)
            metric_results[METRIC_IOU].class_metric = intersection / union
            metric_results[METRIC_IOU].all_metric = metric_results[METRIC_IOU].class_metric.mean(
                dim=1)
            metric_results[METRIC_IOU].add(
                union.sum(dim=0), intersection.sum(dim=0))
            # if save:
            #     for i in range(output.shape[0]):
            #         # datas = ["0:rgb", "1:in_map", "2:K_cam", "3:depth", "4:gt_label", "5:rgb_hdf5"             
            #         rgb_hdf5 = datas[6][i].detach().cpu().numpy()
            #         in_map = np.squeeze(datas[1][i].detach().cpu().numpy()) / 255.0
            #         gt_depth = np.squeeze(datas[3][i].detach().cpu().numpy()) / 255.0
            #         gt_label = np.squeeze(datas[4][i].detach().cpu().numpy().astype(np.uint8))
            #         depth_output = np.squeeze(output[i].detach().cpu().numpy(), axis=0)
            #         pred_segmentation_i = pred_segmentation[i].detach().cpu().numpy()
            #         _hdf5_output(h5_saver, rgb_hdf5, in_map, gt_depth, gt_label, depth_output, pred_segmentation_i, run.cfg.version)
            
        # -> Excel (Overview)
        _excel_overview_output(
            metric_results, eval_sheet_overview, range_norm, result_dict, dynamic_class, static_class)
        print(
            f'{"":11s}: {"Metric":14s}: {"mIoU":14s}:{result_dict[METRIC_IOU]:9.2f} [ % ]') 
        print(
            f'{"":11s}: {"":14s}: {"MAE":14s}:{result_dict[METRIC_MAE]:9.4f} [ m ]')
        print(
            f'{"":11s}: {"":14s}: {"RMSE":14s}:{result_dict[METRIC_RMSE]:9.4f} [ m ]')
        print(
            f'{"":11s}: {"":14s}: {"MAPE":14s}:{result_dict[METRIC_MAPE]:9.4f}')
        filename = f"{run.cfg.name}.xlsx"
        eval_book.save(os.path.join(dir_path, filename))


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        test(run, mode=cfg.data.testset.mode, save=OmegaConf.select(cfg, 'save', default=False))



if __name__ == '__main__':
    main()
