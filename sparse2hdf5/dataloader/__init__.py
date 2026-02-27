from typing import Dict, List, Tuple, Union
from h5dataloader.pytorch.hdf5dataset import HDF5Dataset
from torch.utils.data.dataloader import DataLoader
from h5dataloader.common.structure import CONFIG_TAG_LABEL, CONFIG_TAG_LABELTAG, CONFIG_TAG_RANGE, CONFIG_TAG_SHAPE, CONFIG_TAG_TAG
from .pmod_dataset import PMOD_Train_Dataset, PMOD_Test_Dataset

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

def init_train_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    train_use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        train_use_mods = (0, block_size - 2)

    tr_err_range = args[ARG_TR_ERROR_RANGE] if args[ARG_PROJECTED_POSITION_AUGMENTATION] is True else 0.0
    rot_err_range = args[ARG_ROT_ERROR_RANGE] if args[ARG_PROJECTED_POSITION_AUGMENTATION] is True else 0.0

    train_dataset = PMOD_Train_Dataset(
        h5_paths=args[ARG_TRAIN_DATA], config=args[ARG_TRAIN_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=train_use_mods,
        tr_err_range=tr_err_range, rot_err_range=rot_err_range
    )

    camera_shape: Tuple[int, int,
                        int] = train_dataset.minibatch[DATASET_CAMERA][CONFIG_TAG_SHAPE]
    args[ARG_CAMERA_SHAPE] = [camera_shape[2], camera_shape[0], camera_shape[1]]
    args[ARG_NUM_CLASSES] = len(
        train_dataset.label_color_configs[train_dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]])
    args[ARG_RANGE_NORM] = train_dataset.minibatch[DATASET_DEPTH][CONFIG_TAG_RANGE][1]

    return DataLoader(train_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)


def init_val_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    val_use_mods: Tuple[int, int] = None
    block_size: int = args[ARG_BLOCK_SIZE]

    if block_size > 2:
        val_use_mods = (block_size - 2, block_size - 1)

    val_dataset = PMOD_Test_Dataset(
        h5_paths=args[ARG_VAL_DATA], config=args[ARG_VAL_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=val_use_mods,
        tr_err_range=args.get(ARG_TR_ERROR_RANGE), rot_err_range=args.get(ARG_ROT_ERROR_RANGE)
    )
    return DataLoader(val_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)


def init_eval_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    eval_use_mods: Tuple[int, int] = None
    block_size: int = args.get(ARG_BLOCK_SIZE, 0)

    if block_size > 2:
        eval_use_mods = (block_size - 1, block_size)

    eval_dataset = PMOD_Test_Dataset(
        h5_paths=args[ARG_EVAL_DATA], config=args[ARG_EVAL_DL_CONFIG], quiet=True,
        block_size=block_size, use_mods=eval_use_mods,
        tr_err_range=args.get(ARG_TR_ERROR_RANGE), rot_err_range=args.get(ARG_ROT_ERROR_RANGE)
    )

    label_color_configs: List[Dict[str, Union[str, int]]
                              ] = eval_dataset.label_color_configs[eval_dataset.minibatch[DATASET_LABEL][CONFIG_TAG_LABELTAG]]

    args[ARG_LABEL_TAGS] = {
        str(lcc[CONFIG_TAG_LABEL]): lcc[CONFIG_TAG_TAG] for lcc in label_color_configs}
    args[ARG_NUM_CLASSES] = len(label_color_configs)
    args[ARG_RANGE_NORM] = eval_dataset.minibatch[DATASET_DEPTH][CONFIG_TAG_RANGE][1]

    return DataLoader(eval_dataset, batch_size=args[ARG_BATCH_SIZE], shuffle=False)


def init_tfcheck_dataloader(args: Dict[str, Union[int, float, str, dict]]) -> DataLoader:
    tfcheck_dataset = HDF5Dataset(
        h5_paths=args[ARG_DATA], config=args[ARG_DL_CONFIG], quiet=True)
    return DataLoader(tfcheck_dataset, batch_size=None, shuffle=False)
