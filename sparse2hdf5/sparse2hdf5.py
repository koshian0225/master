import argparse
import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.progress import track
import h5py
from dataloader import *
from h5datacreator import *

ARG_EVAL_DATA = 'eval_data'
ARG_EVAL_DL_CONFIG = 'eval_dl_config'
ARG_TR_ERROR_RANGE = 'tr_error_range'
ARG_ROT_ERROR_RANGE = 'rot_error_range'
ARG_THRESHOLDS = 'thresholds'
ARG_VERSION = 'version'

DEFAULT_THRESHOLDS = [
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 
    30.0, 40.0, 50.0, 60.0, 70.0, 80.0
]
SUPPORTED_VERSIONS = ['carla', 'kitti', 'real']
VERSION_CONFIG = {
    'carla': {
        'rgb_sensor': 'sensor.camera.rgb-front',
        'in_map_sensor': 'sensor.in_map.depth-front',
        'norm_depth_sensor': 'sensor.camera_norm.depth-front',
        'aug_map_sensor': 'sensor.in_aug_map.depth-front',
        'frame_id': 'front',
        'file_ext': '.h5'
    },
    'kitti': {
        'rgb_sensor': 'image_00',
        'in_map_sensor': 'image_00_in_map',
        'norm_depth_sensor': 'image_00_norm_depth',
        'aug_map_sensor': 'image_00_in_aug_map',
        'frame_id': 'front',
        'file_ext': '.hdf5'
    },
    'real': {
        'rgb_sensor': 'zed_rgb',
        'in_map_sensor': 'sparse_depth',
        'norm_depth_sensor': 'zed_norm_depth',
        'frame_id': 'zed_left_camera_optical_frame',
        'file_ext': '.hdf5'
    }
}

def arg_hyphen(arg: str) -> str:
    return arg.replace('_', '-')

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description='Convert sparse depth data to HDF5 format'
    )
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        '-dc', f'--{arg_hyphen(ARG_EVAL_DL_CONFIG)}',
        type=str, metavar='PATH', required=True,
        help='Path to JSON file of dataloader config'
    )
    data_group.add_argument(
        '-d', f'--{arg_hyphen(ARG_EVAL_DATA)}',
        type=str, metavar='PATH', nargs='+', required=True,
        help='Path to HDF5 datasets'
    )
    data_group.add_argument(
        '-v', f'--{arg_hyphen(ARG_VERSION)}',
        type=str, choices=SUPPORTED_VERSIONS, default='kitti',
        help='Dataset version'
    )
    error_group = parser.add_argument_group('Error Range Options')
    error_group.add_argument(
        '-ter', f'--{arg_hyphen(ARG_TR_ERROR_RANGE)}',
        type=float, nargs=3, default=[0.6, 1.3, 0.7],
        help='Translation error range [m]'
    )
    error_group.add_argument(
        '-rer', f'--{arg_hyphen(ARG_ROT_ERROR_RANGE)}',
        type=float, default=3.0,
        help='Rotation error range [deg]'
    )
    misc_group = parser.add_argument_group('Miscellaneous Options')
    misc_group.add_argument(
        f'--{arg_hyphen(ARG_THRESHOLDS)}',
        type=float, nargs='+', default=DEFAULT_THRESHOLDS,
        help='Depth thresholds'
    )
    misc_group.add_argument(
        '-bs', f'--{arg_hyphen(ARG_BLOCK_SIZE)}',
        type=int, default=0,
        help='Block size of dataset'
    )
    misc_group.add_argument(
        '-b', f'--{arg_hyphen(ARG_BATCH_SIZE)}',
        type=int, default=1,
        help='Batch size'
    )

    return vars(parser.parse_args())


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f'{"Device":11s}: {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        print(f'{"Device":11s}: CPU')
    
    return device


def extract_timestamps(file_path: str, version: str) -> Tuple[list, list, int]:
    stamp_sec_list = []
    stamp_nsec_list = []
    config = VERSION_CONFIG[version]
    
    with h5py.File(file_path, 'r') as f:
        header = f['header/length'][()]
        
        for i in range(header):
            data = f['data'][str(i)][config['rgb_sensor']].attrs
            stamp_sec_list.append(data['stamp.sec'])
            stamp_nsec_list.append(data['stamp.nsec'])
    
    return stamp_sec_list, stamp_nsec_list, header


def process_depth_data(tensor_data: torch.Tensor, device: torch.device) -> np.ndarray:
    data_np = tensor_data.to(device, non_blocking=True).cpu().numpy().astype(np.float32)
    data_np = np.squeeze(data_np)
    data_np[data_np == 2] = 0
    data_np = data_np * 255.0

    return data_np


def process_dataset(args: dict, dataloader: DataLoader, stamp_sec_list: list, stamp_nsec_list: list, header: int, device: torch.device) -> None:
    h5file = H5Dataset(args[ARG_EVAL_DATA][0])
    data_group: h5py.Group = h5file.get_current_data_group()
    config = VERSION_CONFIG[args[ARG_VERSION]]
    
    for i, batch in enumerate(track(dataloader, description=f'{"Processing":11s}')):
        in_map: torch.Tensor = batch[DATASET_MAP]
        in_map_np = process_depth_data(in_map, device)
        set_depth(data_group, config['in_map_sensor'], in_map_np, 
                 config['frame_id'], int(stamp_sec_list[i]), int(stamp_nsec_list[i]))
        
        # aug_map: torch.Tensor = batch[DATASET_MAP]
        # aug_map_np = process_depth_data(aug_map, device)
        # set_depth(data_group, config.get('aug_map_sensor', config['in_map_sensor']), aug_map_np, 
        #          config['frame_id'], int(stamp_sec_list[i]), int(stamp_nsec_list[i]))
        
        gt_depth: torch.Tensor = batch[DATASET_DEPTH]
        gt_depth_np = process_depth_data(gt_depth, device)
        set_depth(data_group, config['norm_depth_sensor'], gt_depth_np, 
                 config['frame_id'], int(stamp_sec_list[i]), int(stamp_nsec_list[i]))
        
        if i != header - 1:
            data_group = h5file.get_next_data_group()

def rename_output_file(file_path: str, version: str) -> None:
    config = VERSION_CONFIG[version]
    file_ext = config['file_ext']
    new_path = file_path.replace(file_ext, f"_wm{file_ext}")
    os.rename(file_path, new_path)

def main(args: dict) -> None:
    device = setup_device()
    dataloader = init_eval_dataloader(args)
    file_path = args[ARG_EVAL_DATA][0]
    version = args[ARG_VERSION]
    stamp_sec_list, stamp_nsec_list, header = extract_timestamps(file_path, version)
    process_dataset(args, dataloader, stamp_sec_list, stamp_nsec_list, header, device)
    rename_output_file(file_path, version)
    
    print("おけ")

if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)