import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs
import h5py

__all__ = [
    'LabelConverter',
    'DATASET',
]

class LabelConverter:
    def __init__(self, version: str):
        if version == 'carla':
            self.convert_map = {
                "0": 0, "1": 1, "10": 2, "11": 0, "12": 4, "13": 4, "14": 3, "15": 3,
                "16": 3, "17": 3, "18": 3, "19": 3, "2": 1, "20": 2, "21": 2, "22": 2,
                "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2, "9": 2, "23": 2, "24": 1,
                "25": 2, "26": 2, "27": 1, "28": 2
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
    
class DATASET(torch.utils.data.Dataset):

    def __init__(self, version, ppa, data_list, mode, path=f'datas/version', height=256, width=1216, mean=(90.9950, 96.2278, 94.3213),
                 std=(79.2382, 80.5267, 82.1483), RandCrop=False, tp_min=20, *args, **kwargs):
        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode
        self.RandCrop = RandCrop and mode == 'train'
        self.ppa = ppa
        self.tp_min = tp_min
        self.version = version
        if mode == 'train':
            self.transform = augs.Compose([
                augs.Jitter(),
                augs.Flip(),
                augs.Norm(mean=mean, std=std),
            ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])

        mode_dir = os.path.join(self.base_dir, mode)
        if data_list:
            self.hdf5_files = [os.path.join(mode_dir, f) for f in data_list]
        else:
            if version == 'carla':
                self.hdf5_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith('.h5')]
            elif version == 'kitti360':
                self.hdf5_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith('.hdf5')]

        self.file_indices = []
        for file_idx, hdf5_file in enumerate(self.hdf5_files):
            with h5py.File(hdf5_file, 'r') as f:
                num_samples = len(f['data'])
                self.file_indices.extend([(file_idx, i) for i in range(num_samples)])
        self.num_samples = len(self.file_indices)

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1)
        self.xy = xy

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):
        file_idx, data_idx = self.file_indices[index]
        hdf5_file = self.hdf5_files[file_idx]

        with h5py.File(hdf5_file, 'r') as f:
            sample = f['data'][str(data_idx)]
            if self.version == 'carla':
                depth = sample['sensor.camera_norm.depth-front'][()]
                depth = np.expand_dims(depth, axis=2)
                if self.ppa == True:
                    in_map = sample['sensor.in_aug_map.depth-front'][()]
                else:
                    in_map = sample['sensor.in_map.depth-front'][()]
                in_map = np.expand_dims(in_map, axis=2)
                rgb = sample['sensor.camera.rgb-front'][()]
                gt_label = sample['sensor.camera.semantic_segmentation-front'][()]
                gt_label = np.expand_dims(gt_label, axis=2)
                Cx = f['intrinsic']['sensor.camera.rgb-front/Cx'][()]
                Cy = f['intrinsic']['sensor.camera.rgb-front/Cy'][()]
                Fx = f['intrinsic']['sensor.camera.rgb-front/Fx'][()]
                Fy = f['intrinsic']['sensor.camera.rgb-front/Fy'][()]
            elif self.version == 'kitti360':
                depth = sample['image_00_norm_depth'][()]
                depth = np.expand_dims(depth, axis=2)
                if self.mode == 'train' or self.mode == 'selval':
                    if self.ppa == True:
                        in_map = sample['image_00_in_map'][()]
                    else:
                        in_map = sample['image_00_in_aug_map'][()]
                else:
                    if self.ppa == True:
                        in_map = sample['image_00_in_aug_map'][()]
                    else:
                        in_map = sample['image_00_in_map'][()]
                in_map = np.expand_dims(in_map, axis=2)
                rgb = sample['image_00'][()]
                gt_label = sample['semantic'][()]
                gt_label = np.expand_dims(gt_label, axis=2)
                Cx = f['intrinsic']['image_00/Cx'][()]
                Cy = f['intrinsic']['image_00/Cy'][()]
                Fx = f['intrinsic']['image_00/Fx'][()]
                Fy = f['intrinsic']['image_00/Fy'][()]
            K_cam = np.array([[Fx, 0.0, Cx], [0.0, Fy, Cy], [0.0, 0.0, 1.0]])
        rgb = rgb.astype(np.float32)
        in_map = in_map.astype(np.float32)
        depth = depth.astype(np.float32)
        gt_label = gt_label.astype(np.uint8)
        K_cam = K_cam.astype(np.float32)

        rgb_nd = rgb.transpose(2, 0, 1).astype(np.float32)
        rgb_hdf5 =rgb.transpose(2, 0, 1).astype(np.uint8)

        if self.transform:
            rgb, in_map, depth, K_cam, gt_label = self.transform(rgb, in_map, depth, K_cam, gt_label)

        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        in_map = in_map.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        gt_label = gt_label.transpose(2, 0, 1).astype(np.uint8)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop:
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)
        crop_slice = (slice(None), slice(tp, tp + self.height), slice(lp, lp + self.width))
        rgb = rgb[crop_slice]
        rgb_hdf5 = rgb_hdf5[crop_slice]
        rgb_nd = rgb_nd[crop_slice]
        in_map = in_map[crop_slice]
        depth = depth[crop_slice]
        gt_label = gt_label[crop_slice]

        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp

        # 352*1216 -> 256*1216
        crop_slice_96 = crop_slice_96 = (slice(None), slice(96, 96 + 256), slice(0, self.width))
        rgb = rgb[crop_slice_96]
        rgb_hdf5 = rgb_hdf5[crop_slice_96].transpose(1, 2, 0)
        in_map = in_map[crop_slice_96]
        depth = depth[crop_slice_96]
        gt_label = gt_label[crop_slice_96]

        K_cam[1, 2] -= 96

        converter = LabelConverter(self.version)
        gt_label = converter.convert(gt_label)

        if self.mode == 'train' or self.mode == 'selval':
            return rgb, in_map, K_cam, depth, gt_label, rgb_nd
        elif self.mode == 'test':
            return rgb, in_map, K_cam, depth, gt_label, rgb_nd, rgb_hdf5