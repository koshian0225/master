from types import MethodType
import os
import numpy as np
import cv2
from pointsmap import invertTransform, combineTransforms
from h5dataloader.common.structure import *
from h5dataloader.common.create_funcs import NORMALIZE_INF
from h5dataloader.pytorch import HDF5Dataset
from h5dataloader.pytorch.structure import CONVERT_TORCH, DTYPE_TORCH
import torch

BASE_CHANNELS = 64
SKIP_CHANNELS = 24

# Train: Training
ARG_TAG = 'tag'
ARG_TRAIN_DL_CONFIG = 'train_dl_config'
ARG_VAL_DL_CONFIG = 'val_dl_config'
ARG_BLOCK_SIZE = 'block_size'
ARG_TRAIN_DATA = 'train_data'
ARG_VAL_DATA = 'val_data'
ARG_EPOCHS = 'epochs'
ARG_EPOCH_START_COUNT = 'epoch_start_count'
ARG_STEPS_PER_EPOCH = 'steps_per_epoch'
ARG_PROJECTED_POSITION_AUGMENTATION = 'projected_position_augmentation'
ARG_TR_ERROR_RANGE = 'tr_error_range'
ARG_ROT_ERROR_RANGE = 'rot_error_range'
ARG_AUTO_EVALUATION = 'auto_evaluation'
# Train: Network
ARG_BATCH_SIZE = 'batch_size'
ARG_RESUME = 'resume'
ARG_AMP = 'amp'
# Train: Optimizer
ARG_CLIP_MAX_NORM = 'clip_max_norm'
ARG_OPTIM_PARAMS = 'optim_params'
ARG_OPTIMIZER = 'optimizer'
ARG_LR_POLICY = 'lr_policy'
# Train: Loss
ARG_L1 = 'l1'
ARG_SEG_CE = 'seg_ce'
ARG_SEG_CE_AUX1 = 'seg_ce_aux1'
ARG_SEG_CE_AUX2 = 'seg_ce_aux2'
# Train: Debug
ARG_DETECT_ANOMALY = 'detect_anomaly'
# Train: Other
ARG_NUM_CLASSES = 'num_classes'
ARG_RANGE_NORM = 'range_norm'
ARG_CAMERA_SHAPE = 'camera_shape'
ARG_PARAMS = 'params'
ARG_DATE = 'date'
# Eval: Evaluate
ARG_EVAL_DATA = 'eval_data'
ARG_EVAL_DL_CONFIG = 'eval_dl_config'
ARG_CHECK_POINT = 'check_point'
ARG_TRAIN_CONFIG = 'train_config'
ARG_THRESHOLDS = 'thresholds'
ARG_NOMAP = 'nomap'
# Eval: Other
ARG_LABEL_TAGS = 'label_tags'
# Optuna
ARG_SAMPLER = 'sampler'
ARG_SEED = 'seed'
ARG_FUNC = 'func'
ARG_N_TRIALS = 'n_trials'
ARG_HOST = 'host'
# TF-Check
ARG_DATA = 'data'
ARG_DL_CONFIG = 'dl_config'
# CKPT2TSM
ARG_HEIGHT = 'height'
ARG_WIDTH = 'width'

DEFAULT_THRESHOLDS = [
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0
]

DATASET_CAMERA = 'camera'
DATASET_MAP = 'map'
DATASET_LABEL = 'label'
DATASET_DEPTH = 'depth'
DATASET_POSE_ERR = 'pose_err'

DIR_CHECKPOINTS = 'checkpoints'
DIR_LOGS = 'logs'
DIR_RESULTS = 'results'
DIR_OPTUNA = 'optuna'

SINGLE_SAMPLER_TPE = 'tpe'
SINGLE_SAMPLER_GRID = 'grid'
SINGLE_SAMPLER_RANDOM = 'random'
SINGLE_SAMPLER_CMAES = 'cmaes'
SINGLE_SAMPLER_PARTIALFIXED = 'partialfixed'
SINGLE_SAMPLERS = [
    SINGLE_SAMPLER_TPE,
    SINGLE_SAMPLER_GRID,
    SINGLE_SAMPLER_RANDOM,
    SINGLE_SAMPLER_CMAES,
]

MULTI_SAMPLER_NSGA2 = 'nsga2'
MULTI_SAMPLER_MOTPE = 'motpe'
MULTI_SAMPLERS = [
    MULTI_SAMPLER_NSGA2,
    MULTI_SAMPLER_MOTPE,
]

CONFIG_ENCODER = 'encoder'
CONFIG_DECODER = 'decoder'
CONFIG_DEPTH = 'depth'
CONFIG_SEGMENTATION = 'segmentation'
CONFIG_OPTIMIZER = 'optimizer'
CONFIG_SCHEDULER = 'scheduler'

CONFIG_LR = 'lr'
CONFIG_NITER = 'niter'
CONFIG_BETA1 = 'beta1'
CONFIG_BETA2 = 'beta2'
CONFIG_MOMENTUM = 'momentum'
CONFIG_EPOCH_COUNT = 'epoch_count'
CONFIG_NITER = 'niter'
CONFIG_NITER_DECAY = 'niter_decay'
CONFIG_GAMMA = 'gamma'
CONFIG_PATIENCE = 'patience'
CONFIG_FACTOR = 'factor'
CONFIG_THRESHOLD = 'threshold'
CONFIG_ETA_MIN = 'eta_min'
CONFIG_BASE_LR = 'base_lr'
CONFIG_MAX_LR = 'max_lr'

OPTIM_TYPE_ADAM = 'adam'
OPTIM_TYPE_SGD = 'sgd'
OPTIM_TYPE_ADABELIEF = 'adabelief'
OPTIM_TYPES = [
    OPTIM_TYPE_ADAM,
    OPTIM_TYPE_SGD,
    OPTIM_TYPE_ADABELIEF,
]

LR_POLICY_LAMBDA = 'lambda'
LR_POLICY_STEP = 'step'
LR_POLICY_PLATEAU = 'plateau'
LR_POLICY_COS = 'cos'
LR_POLICY_CLR = 'clr'
LR_POLICIES = [
    LR_POLICY_LAMBDA, LR_POLICY_STEP, LR_POLICY_PLATEAU,
    LR_POLICY_COS, LR_POLICY_CLR,
]

INIT_TYPE_NORMAL = 'normal'
INIT_TYPE_XAVIER = 'xavier'
INIT_TYPE_KAIMING = 'kaiming'
INIT_TYPE_ORTHOGONAL = 'orthogonal'

METRIC_IOU = 'iou'
METRIC_MAE = 'mae'
METRIC_RMSE = 'rmse'
METRIC_MAPE = 'mape'

PADDING_ZEROS = 'zeros'
PADDING_REFLECT = 'reflect'
PADDING_REPLICATE = 'replicate'
PADDING_CIRCULAR = 'circular'

UPSAMPLE_BILINEAR = 'bilinear'

class PMOD_Train_Dataset(HDF5Dataset):
    def __init__(self, h5_paths: List[str], config: str, quiet: bool = True, block_size: int = 0, use_mods: Tuple[int, int] = None,
                 visibility_filter_radius: int = 0, visibility_filter_threshold: float = 3.0, tr_err_range: float = 2.0, rot_err_range: float = 10.0) -> None:

        super(PMOD_Train_Dataset, self).__init__(h5_paths, config, quiet, block_size,
                                                 use_mods, visibility_filter_radius, visibility_filter_threshold)

        # Random Pose Error
        self.tr_err_range: float = np.array(tr_err_range, dtype=np.float32)
        self.rot_err_range: float = np.deg2rad(rot_err_range)
        self.minibatch[DATASET_MAP][CONFIG_TAG_TF].append(('', False))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Random Pose Error
        rot_vec: np.ndarray = (np.random.rand(3) * 2.0 - 1.0)
        rot_vec /= np.linalg.norm(rot_vec)
        rot_abs: float = np.random.rand() * self.rot_err_range

        self.q_err: np.ndarray = self.__vec2quat(rot_vec, rot_abs)
        self.tr_err: np.ndarray = (np.random.rand(
            3) * 2.0 - 1.0) * self.tr_err_range

        # Get Items.
        items: Dict[str, torch.Tensor] = super().__getitem__(index)
        tr_norm: float = 1.0
        if self.minibatch[DATASET_MAP][CONFIG_TAG_NORMALIZE] is True:
            tr_norm *= self.minibatch[DATASET_MAP][CONFIG_TAG_RANGE][1]
        items[DATASET_POSE_ERR] = torch.from_numpy(CONVERT_TORCH[TYPE_POSE](
            DTYPE_TORCH[TYPE_POSE](np.concatenate([self.tr_err / tr_norm, self.q_err]))))

        return items

    def __vec2quat(self, vec: np.ndarray, abs: float) -> np.ndarray:
        # Rotation vector to Quaternion.
        xyz: np.ndarray = vec * np.sin(abs * 0.5)
        return np.append(xyz, np.cos(abs * 0.5))

    def depth_common(self, src: np.ndarray, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        dst = src
        if minibatch_config[CONFIG_TAG_NORMALIZE] is True:
            range_min, range_max = minibatch_config[CONFIG_TAG_RANGE][:2]
            dst = np.where(range_max < dst, NORMALIZE_INF,
                           (dst - range_min) / (range_max - range_min))

        shape = minibatch_config[CONFIG_TAG_SHAPE]
        if shape != dst.shape[:2]:
            dst = cv2.resize(dst, dsize=(
                shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        return dst

    def create_pose_from_pose(self, key: str, link_idx: int, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_pose_from_pose
        "pose"を生成する
        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定
        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]
        """
        translations = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        quaternions = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)]

        for child_frame_id, invert in minibatch_config[CONFIG_TAG_TF]:
            tf_data: Dict[str, str] = self.tf[CONFIG_TAG_DATA].get(
                child_frame_id)
            if tf_data is None:
                # Random Pose Error
                trns: np.ndarray = self.tr_err
                qtrn: np.ndarray = self.q_err
            else:
                h5_key = tf_data[CONFIG_TAG_KEY]
                if h5_key[0] == '/':
                    h5_key = str(link_idx) + h5_key
                else:
                    h5_key = os.path.join(key, h5_key)
                trns: np.ndarray = self.h5links[h5_key][SUBTYPE_TRANSLATION][(
                )]
                qtrn: np.ndarray = self.h5links[h5_key][SUBTYPE_ROTATION][()]
            if invert is True:
                trns, qtrn = invertTransform(translation=trns, quaternion=qtrn)
            translations.append(trns)
            quaternions.append(qtrn)

        translation, quaternion = combineTransforms(
            translations=translations, quaternions=quaternions)

        return np.concatenate([translation, quaternion])


class PMOD_Test_Dataset(HDF5Dataset):
    def __init__(self, h5_paths: List[str], config: str, quiet: bool = True, block_size: int = 0, use_mods: Tuple[int, int] = None,
                 visibility_filter_radius: int = 0, visibility_filter_threshold: float = 3.0, tr_err_range: float = 2.0, rot_err_range: float = 10.0) -> None:

        super(PMOD_Test_Dataset, self).__init__(h5_paths, config, quiet, block_size,
                                                use_mods, visibility_filter_radius, visibility_filter_threshold)

        if self.minibatch.get(DATASET_POSE_ERR) is None:
            self.random_pose: bool = True
            # Random Pose Error
            self.tr_err_range: float = np.array(tr_err_range, dtype=np.float32)
            self.rot_err_range: float = np.deg2rad(rot_err_range)
            self.minibatch[DATASET_MAP][CONFIG_TAG_TF].append(('', False))

            rot_vec: np.ndarray = (np.random.rand(self.length, 3) * 2.0 - 1.0)
            rot_vec /= np.linalg.norm(rot_vec, axis=1, keepdims=True)
            rot_abs: float = np.random.rand(self.length) * self.rot_err_range

            self.q_err_list: np.ndarray = self.__vec2quat(rot_vec, rot_abs)
            self.tr_err_list: np.ndarray = (np.random.rand(
                self.length, 3) * 2.0 - 1.0) * self.tr_err_range
        else:
            self.random_pose: bool = False

    def __vec2quat(self, vec: np.ndarray, abs: float) -> np.ndarray:
        # Rotation vector to Quaternion.
        xyz: np.ndarray = vec * \
            np.sin(np.repeat(abs[:, np.newaxis], 3, axis=1) * 0.5)
        return np.append(xyz, np.cos(abs * 0.5)[:, np.newaxis], axis=1)

    def create_pose_from_pose(self, key: str, link_idx: int, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_pose_from_pose
        "pose"を生成する
        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定
        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]
        """
        translations = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        quaternions = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)]

        for child_frame_id, invert in minibatch_config[CONFIG_TAG_TF]:
            tf_data: Dict[str, str] = self.tf[CONFIG_TAG_DATA].get(
                child_frame_id)
            if tf_data is None:
                # Random Pose Error
                trns: np.ndarray = self.tr_err
                qtrn: np.ndarray = self.q_err
            else:
                h5_key = tf_data[CONFIG_TAG_KEY]
                if h5_key[0] == '/':
                    h5_key = str(link_idx) + h5_key
                else:
                    h5_key = os.path.join(key, h5_key)
                trns: np.ndarray = self.h5links[h5_key][SUBTYPE_TRANSLATION][(
                )]
                qtrn: np.ndarray = self.h5links[h5_key][SUBTYPE_ROTATION][()]
            if invert is True:
                trns, qtrn = invertTransform(translation=trns, quaternion=qtrn)
            translations.append(trns)
            quaternions.append(qtrn)

        translation, quaternion = combineTransforms(
            translations=translations, quaternions=quaternions)

        return np.concatenate([translation, quaternion])

    def __getitem__(self, index: int) -> dict:
        if self.random_pose is True:
            self.tr_err = self.tr_err_list[index]
            self.q_err = self.q_err_list[index]
        items: Dict[str, torch.Tensor] = super().__getitem__(index)
        tr_norm: float = 1.0
        if self.minibatch[DATASET_MAP][CONFIG_TAG_NORMALIZE] is True:
            tr_norm *= self.minibatch[DATASET_MAP][CONFIG_TAG_RANGE][1]
        if self.random_pose is True:
            items[DATASET_POSE_ERR] = torch.from_numpy(CONVERT_TORCH[TYPE_POSE](
                DTYPE_TORCH[TYPE_POSE](np.concatenate([self.tr_err / tr_norm, self.q_err]))))
        else:
            items[DATASET_POSE_ERR][:3] = items[DATASET_POSE_ERR][:3] / tr_norm
        return items
