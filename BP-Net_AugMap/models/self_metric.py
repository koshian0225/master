import torch
from typing import List, Dict , Tuple
import torch.nn as nn
from torch import Tensor
from typing import Any, NamedTuple
import torch.nn.functional as F
import numpy as np


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

class MetricIntersection(nn.Module):
    def __init__(self):
        super(MetricIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = gt.to(torch.int64)
        gt = F.one_hot(gt, num_classes=num_classes)

        intersection = torch.logical_and(pred, gt)

        return torch.sum(intersection, dim=(1, 2))

class MetricSumIntersection(MetricIntersection):
    def __init__(self):
        super(MetricSumIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt), dim=0)

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

        union = torch.logical_or(pred, gt)

        return torch.sum(union, (1, 2)) + self.smooth

class MetricSumUnion(MetricUnion):
    def __init__(self, smooth: float = 1e-10):
        super().__init__(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt) - self.smooth, dim=0) + self.smooth

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
        if self.num_classes > 0:
            seg_gt = seg_gt.long()
            _one_hot: Tensor = F.one_hot(seg_gt, num_classes=self.num_classes).bool()
        for th_idx in range(self.thresholds.shape[0]):
            _th: Tensor = self.thresholds[th_idx]
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
    
class LabelConverter:
    """
    ラベル変換
    """
    def __init__(self, version: str):
        if version == 'carla':
            self.convert_map = {
                "0": 0, "1": 7, "10": 3, "11": 13, "12": 4, "13": 4, "14": 10, "15": 10,
                "16": 10, "17": 10, "18": 10, "19": 10, "2": 8, "20": 1, "21": 3, "22": 3,
                "3": 1, "4": 11, "5": 2, "6": 5, "7": 14, "8": 12, "9": 9, "23": 3, "24": 6,
                "25": 3, "26": 1, "27": 7, "28": 11
            }
        elif version == 'kitti360':
            self.convert_map = {
                "-1": 3,"0": 0, "1": 0, "2": 0, "3": 0,"4": 2,"5": 3,"6": 1,"7": 1,"8": 1,"9": 1,
                "10": 2,"11": 2,"12": 2,"13": 2,"14": 2,"15": 2,"16": 2,"17": 2,"18": 2,"19": 2,
                "20": 2,"21": 2,"22": 2,"23": 0,"24": 4,"25": 4,"26": 3,"27": 3,"28": 3,"29": 3,
                "30": 3,"31": 3,"32": 3,"33": 3,"34": 2,"35": 2,"36": 2,"37": 2,"38": 2,"39": 2,
                "40": 2,"41": 2,"42": 2,"43": 3,"44": 2
            }
        else:
            raise ValueError(f"Unknown version: {version}")

        self.keys = np.array([int(k) for k in self.convert_map.keys()])
        self.values = np.array([v for v in self.convert_map.values()])

    def convert(self, label: np.ndarray) -> np.ndarray:
        converted_label = np.zeros_like(label, dtype=np.uint8)
        for key, value in zip(self.keys, self.values):
            converted_label = np.where(label == key, value, converted_label)
        return converted_label

