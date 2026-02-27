import argparse
from typing import Dict, Tuple
import os
from h5dataloader.common.structure import *
import numpy as np
import h5py
import cv2
from rich.progress import track
import open3d as o3d
import json
from scipy.spatial.transform import Rotation
import scipy

"""
-v {carla_data,
    kitti_data,
    real_data}

-l {carla,
    kitti,
    real}
"""

class DataDump:
    def __init__(self, args: Dict[str, str], result_dir: str):
        self.args = args
        self.output_dir = result_dir

    def create_output_dir(self, *dirs: str) -> None:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def save_image(self, path: str, image) -> None:
        cv2.imwrite(path, image)

    @staticmethod
    def convert_label(src: h5py.Dataset, label_group) -> np.ndarray:
        label_tag: str = list(label_group.keys())[0]
        labels: h5py.Group = label_group[label_tag]

        color_label: np.ndarray = np.zeros(
            (src.shape[0], src.shape[1], 3), dtype=np.uint8)

        for label_key, label_values in labels.items():
            color_label[np.where(src[()] == int(label_key))
                        ] = label_values[TYPE_COLOR][()]
        return color_label

    @staticmethod
    def depth2colormap(src: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        out_range = np.where((src < range_min) | (range_max < src))
        src_norm = np.uint8(
            (1.0 - (src - range_min) / (range_max - range_min)) * 255.0)
        colormap = cv2.applyColorMap(src_norm, cv2.COLORMAP_JET)
        colormap[out_range] = [0, 0, 0]
        return colormap

    def point_cloud_to_image(self, point_cloud: np.ndarray, Cx: float, Cy: float, Fx: float, Fy: float, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Pred_point_cloudを画像座標に逆投影し、画像上に点を描画した1ch距離画像を返す（最前面優先、超高速）
        """
        h, w = image_shape
        depth_img = np.zeros((h, w), dtype=np.float32)
        if point_cloud.size == 0:
            return depth_img

        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        valid = x > 0
        x = x[valid]
        y = y[valid]
        z = z[valid]

        px = -y * Fx / x
        py = z * Fy / x
        u = (px + Cx).astype(np.int32)
        v = (-py + Cy).astype(np.int32)

        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[mask]
        v = v[mask]
        x = x[mask]

        # 高速化: (v, u)ごとに最小x（最前面）をセット
        idx_flat = v * w + u
        order = np.argsort(x)  # xが小さい順（手前優先）
        idx_flat = idx_flat[order]
        x = x[order]

        # 重複座標は最初（最前面）のみ残す
        _, unique_indices = np.unique(idx_flat, return_index=True)
        idx_flat = idx_flat[unique_indices]
        x = x[unique_indices]

        depth_img_flat = depth_img.ravel()
        depth_img_flat[idx_flat] = x
        depth_img = depth_img_flat.reshape((h, w))

        return depth_img

    def process_velodyne_points(self, velodyne_points: np.ndarray, homogenous_trans: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
        pseudo_velodyne_points = np.dot(homogenous_trans, np.vstack((velodyne_points.T, np.ones(velodyne_points.shape[0])))).T[:, :3]
        camera_points = pseudo_velodyne_points[pseudo_velodyne_points[:, 2] > 0]
        pixel_x = camera_points[:, 0] / camera_points[:, 2]
        pixel_y = camera_points[:, 1] / camera_points[:, 2]

        pixel = np.ones(camera_points.shape)
        pixel[:, 0] = pixel_x
        pixel[:, 1] = pixel_y

        pixel = np.dot(intrinsic, pixel.T).T
        
        condition = (pixel[:, 0] >= 0.0) & (pixel[:, 0] <= width) & (pixel[:, 1] >= 0.0) & (pixel[:, 1] <= height)
        pixel = pixel[condition]
        camera_points = camera_points[condition]

        depth_picture = np.zeros((height, width), dtype=np.float32)
        depth_picture[pixel[:, 1].astype(int), pixel[:, 0].astype(int)] = camera_points[:, 2]

        return depth_picture

    def save_images_mde(self, i: int, in_camera: np.ndarray, gt_depth: np.ndarray, gt_label: np.ndarray, mde_depth: np.ndarray, input_basename: str):
        in_camera_output_dir = os.path.join(self.output_dir, input_basename, 'in_camera')
        gt_depth_output_dir = os.path.join(self.output_dir, input_basename, 'gt_depth')
        gt_label_output_dir = os.path.join(self.output_dir, input_basename, 'gt_label')
        mde_depth_output_dir = os.path.join(self.output_dir, input_basename, 'mde_depth')
        self.create_output_dir(in_camera_output_dir, gt_depth_output_dir, gt_label_output_dir, mde_depth_output_dir)

        in_camera_path = os.path.join(in_camera_output_dir, f'in_camera_{i:06d}.png')
        gt_depth_path = os.path.join(gt_depth_output_dir, f'gt_depth_{i:06d}.png')
        gt_label_path = os.path.join(gt_label_output_dir, f'gt_label_{i:06d}.png')
        mde_depth_path = os.path.join(mde_depth_output_dir, f'mde_depth_{i:06d}.png')

        self.save_image(in_camera_path, in_camera)
        self.save_image(gt_depth_path, gt_depth)
        self.save_image(gt_label_path, gt_label)
        self.save_image(mde_depth_path, mde_depth)
   
   #ND用に修正
    # def save_images(self, i: int, in_camera: np.ndarray, gt_depth: np.ndarray, gt_label: np.ndarray, input_basename: str):
    def save_images(self, i: int, in_camera: np.ndarray, in_camera2: np.ndarray, gt_depth: np.ndarray, gt_label: np.ndarray, input_basename: str):
        in_camera_output_dir = os.path.join(self.output_dir, input_basename, 'in_camera')
        in_camera_output_dir2 = os.path.join(self.output_dir, input_basename, 'in_camera2')
        gt_depth_output_dir = os.path.join(self.output_dir, input_basename, 'gt_depth')
        gt_label_output_dir = os.path.join(self.output_dir, input_basename, 'gt_label')
        self.create_output_dir(in_camera_output_dir, gt_depth_output_dir, gt_label_output_dir)
        self.create_output_dir(in_camera_output_dir, gt_depth_output_dir, in_camera_output_dir2)

        in_camera_path = os.path.join(in_camera_output_dir, f'in_camera_{i:06d}.png')
        in_camera_path2 = os.path.join(in_camera_output_dir2, f'in_camera2_{i:06d}.png')
        gt_depth_path = os.path.join(gt_depth_output_dir, f'gt_depth_{i:06d}.png')
        gt_label_path = os.path.join(gt_label_output_dir, f'gt_label_{i:06d}.png')

        self.save_image(in_camera_path, in_camera)
        self.save_image(gt_depth_path, gt_depth)
        self.save_image(in_camera_path2, in_camera2)
        self.save_image(gt_label_path, gt_label)

    def save_images_result(self, i: int, in_camera: np.ndarray, in_map: np.ndarray, pr_seg: np.ndarray, pr_depth: np.ndarray, gt_depth: np.ndarray, gt_label: np.ndarray, input_basename: str):
    # def save_images_result(self, i: int, in_camera: np.ndarray, in_map: np.ndarray, pr_depth: np.ndarray, gt_depth: np.ndarray, gt_label: np.ndarray, input_basename: str):
        in_camera_output_dir = os.path.join(self.output_dir, input_basename, 'in_camera')
        in_map_output_dir = os.path.join(self.output_dir, input_basename, 'in_map')
        pr_seg_output_dir = os.path.join(self.output_dir, input_basename, 'pr_seg')
        pr_depth_output_dir = os.path.join(self.output_dir, input_basename, 'pr_depth')
        gt_depth_output_dir = os.path.join(self.output_dir, input_basename, 'gt_depth')
        gt_label_output_dir = os.path.join(self.output_dir, input_basename, 'gt_label')
        self.create_output_dir(in_camera_output_dir, in_map_output_dir, pr_depth_output_dir, gt_depth_output_dir, gt_label_output_dir)
        self.create_output_dir(in_camera_output_dir, in_map_output_dir, pr_seg_output_dir, pr_depth_output_dir, gt_depth_output_dir, gt_label_output_dir)

        in_camera_path = os.path.join(in_camera_output_dir, f'in_camera_{i:06d}.png')
        in_map_path = os.path.join(in_map_output_dir, f'in_map_{i:06d}.png')
        pr_seg_path = os.path.join(pr_seg_output_dir, f'pr_seg_{i:06d}.png')
        pr_depth_path = os.path.join(pr_depth_output_dir, f'pr_depth_{i:06d}.png')
        gt_depth_path = os.path.join(gt_depth_output_dir, f'gt_depth_{i:06d}.png')
        gt_label_path = os.path.join(gt_label_output_dir, f'gt_label_{i:06d}.png')

        self.save_image(in_camera_path, in_camera)
        self.save_image(in_map_path, in_map)
        self.save_image(pr_seg_path, pr_seg)
        self.save_image(pr_depth_path, pr_depth)
        self.save_image(gt_depth_path, gt_depth)
        self.save_image(gt_label_path, gt_label)

    def generate_point_cloud(self, depth: np.ndarray, Cx: float, Cy: float, Fx: float, Fy: float, max_depth: float = 80.0) -> np.ndarray:
        # u_grid, v_grid = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        h, w = depth.shape
        start_row = h // 3
        depth_crop = depth[start_row:, :]
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(start_row, h))
        px = u_grid - Cx
        py = -v_grid + Cy

        # x = depth
        # y = -px * depth / Fx
        # z = py * depth / Fy

        # valid_mask = depth < max_depth

        x = depth_crop
        y = -px * depth_crop / Fx
        z = py * depth_crop / Fy

        valid_mask = depth_crop < max_depth

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]

        point_cloud = np.stack((x, y, z), axis=-1)
        return point_cloud

    def save_point_cloud(self, point_cloud: np.ndarray, filename: str) -> None:
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(point_cloud)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        o3d.io.write_point_cloud(filename, ply, write_ascii=True)

    def save_point_cloud2(self, point_cloud: np.ndarray, color_cloud: np.ndarray, filename: str) -> None:
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(point_cloud)
        ply.colors = o3d.utility.Vector3dVector(color_cloud)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        o3d.io.write_point_cloud(filename, ply, write_ascii=True)

    def dump(self):
        with h5py.File(self.args['input'], mode='r') as h5file:
            # header = h5file['header/length'][()]
            header = len(h5file['data'])
            data_group: h5py.Group = h5file['data']
            label_group: h5py.Group = h5file['label']

            if self.args["version"] == "carla_data":
                Cx = h5file['intrinsic/sensor.camera.rgb-front/Cx'][()]
                Cy = h5file['intrinsic/sensor.camera.rgb-front/Cy'][()]
                Fx = h5file['intrinsic/sensor.camera.rgb-front/Fx'][()]
                Fy = h5file['intrinsic/sensor.camera.rgb-front/Fy'][()]
                width = h5file['intrinsic/sensor.camera.rgb-front/width'][()]
                height = h5file['intrinsic/sensor.camera.rgb-front/height'][()]

            if self.args["version"] == "kitti_data":
                Cx = h5file['intrinsic/image_00/Cx'][()]
                Cy = h5file['intrinsic/image_00/Cy'][()]
                Fx = h5file['intrinsic/image_00/Fx'][()]
                Fy = h5file['intrinsic/image_00/Fy'][()]
                width = h5file['intrinsic/image_00/width'][()]
                height = h5file['intrinsic/image_00/height'][()]

                intrinsic = np.array([[Fx, 0.0, Cx],
                                    [0.0, Fy, Cy],
                                    [0.0, 0.0, 1.0]])
                
                cam02velo_translation = h5file['tf_static/cam0_to_velo/translation'][()]
                cam02velo_quaternion = h5file['tf_static/cam0_to_velo/rotation'][()]

                quat = Rotation.from_quat(cam02velo_quaternion)
                cam02velo_mat = quat.as_matrix()
                
                homogenous_trans = np.identity(4)
                homogenous_trans[:3, :3] = cam02velo_mat
                homogenous_trans[:3, 3] = cam02velo_translation
            
            if self.args["version"] == "result_data":
                color_data = []
                if self.args["label"] == "carla":
                    # depth_front_Fx = 704.0000000000001
                    # depth_front_Fy = 704.0000000000001
                    # depth_front_Cx = 704.0
                    # depth_front_Cy = 188.0
                    depth_front_Fx = 704.0000000000001
                    depth_front_Fy = 704.0000000000001
                    depth_front_Cx = 608.0
                    depth_front_Cy = 68.0
                    scale_x = 512 / 1408
                    scale_y = 256 / 376
                    depth_front_Cx *= scale_x
                    depth_front_Cy *= scale_y
                    depth_front_Fx *= scale_x
                    depth_front_Fy *= scale_y
                    with open('./config/carla-5class.json') as f:
                        config_data = json.load(f)
                    for l in range(15) :
                        colors = config_data["label"]["config"]["carla-0.9.14"]["dst"][f'{l}']["color"]
                        color_data.extend(np.array([[colors[0], colors[1], colors[2]]]))
                elif self.args["label"] == "kitti":
                    depth_front_Fx = 552.554261
                    depth_front_Fy = 552.554261
                    depth_front_Cx = 682.049453
                    depth_front_Cy = 238.769549
                    # depth_front_Cx = 586.04944
                    # depth_front_Cy = 214.76955
                    scale_x = 512 / 1408
                    scale_y = 256 / 376
                    depth_front_Cx *= scale_x
                    depth_front_Cy *= scale_y
                    depth_front_Fx *= scale_x
                    depth_front_Fy *= scale_y
                    # Fx = 552.554261
                    # Fy = 552.554261
                    # Cx = 586.04944
                    # Cy = 214.76955
                    Fx = depth_front_Fx
                    Fy = depth_front_Fy
                    Cx = depth_front_Cx
                    Cy = depth_front_Cy
                    with open('./config/kitti360-5class.json') as f:
                        config_data = json.load(f)
                    for m in range(5) :
                        colors = config_data["label"]["config"]["5class"]["dst"][f'{m}']["color"]
                        color_data.extend(np.array([[colors[0], colors[1], colors[2]]]))
                elif self.args["label"] == "real":
                    depth_front_Fx = 535.0951697601366
                    depth_front_Fy = 535.0951697601366
                    depth_front_Cx = 492.62023162841797
                    depth_front_Cy = 271.74023056030273
                    scale_x = 1216 / 960
                    scale_y = 352 / 540
                    depth_front_Cx *= scale_x
                    depth_front_Cy *= scale_y
                    depth_front_Fx *= scale_x
                    depth_front_Fy *= scale_y
                    # Fx = 535.0951697601366
                    # Fy = 535.0951697601366
                    # Cx = 492.62023162841797
                    # Cy = 271.74023056030273
                    depth_front_Cy -=96
                    Fx = depth_front_Fx
                    Fy = depth_front_Fy
                    Cx = depth_front_Cx
                    Cy = depth_front_Cy
                    print(depth_front_Cx, depth_front_Cy, depth_front_Fx, depth_front_Fy)
                    with open('./config/real_fukuta_80m.json') as f:
                        config_data = json.load(f)
                    for n in range(5) :
                        colors = config_data["label"]["config"]["label"]["dst"][f'{n}']["color"]
                        color_data.extend(np.array([[colors[0], colors[1], colors[2]]]))
            for i in track(range(header)):
                if self.args["version"] == "carla_data":
                    src_group: h5py.Group = data_group[str(i)]

                    in_camera: np.ndarray = src_group['sensor.camera.rgb-front'][()]
                    gt_depth: np.ndarray = self.depth2colormap(src_group['sensor.camera.depth-front'][()], 0.0, 100.0)
                    gt_depth_row: np.ndarray = src_group['sensor.camera.depth-front'][()]
                    gt_point_cloud = self.generate_point_cloud(gt_depth_row, Cx, Cy, Fx, Fy)
                    gt_label: np.ndarray = self.convert_label(src_group['sensor.camera.semantic_segmentation-front'][()], label_group)

                    input_basename = os.path.basename(self.args['input'])
                    point_cloud_path = os.path.join(self.output_dir, input_basename, 'gt_depth_ply', f'{i:06d}.ply')
                    self.save_point_cloud(gt_point_cloud, point_cloud_path)
                    self.save_images(i, in_camera, gt_depth, gt_label, input_basename)

                if self.args["version"] == "kitti_data":
                    src_group: h5py.Group = data_group[str(i)]

                    in_camera: np.ndarray = src_group['image_00'][()]
                    in_camera2: np.ndarray = src_group['image_01'][()]
                    gt_label: np.ndarray = self.convert_label(src_group['semantic'][()], label_group)

                    velodyne_points: np.ndarray = src_group['velodyne_points'][()]
                    depth_picture = self.process_velodyne_points(velodyne_points, homogenous_trans, intrinsic, width, height)
                    # gt_point_cloud = self.generate_point_cloud(depth_picture, Cx, Cy, Fx, Fy)

                    input_basename = os.path.basename(self.args['input'])
                    # point_cloud_path = os.path.join(self.output_dir, input_basename, 'gt_depth_ply', f'{i:06d}.ply')
                    # self.save_point_cloud(gt_point_cloud, point_cloud_path)
                    # self.save_images(i, in_camera, depth_picture, gt_label, input_basename)
                    self.save_images(i, in_camera, in_camera2, depth_picture, gt_label, input_basename)

                if self.args["version"] == "real_data":
                    src_group: h5py.Group = data_group[str(i)]

                    in_camera: np.ndarray = src_group['zed_rgb'][()]
                    gt_depth: np.ndarray = self.depth2colormap(src_group['zed_depth'][()], 0.0, 100.0)
                    gt_label: np.ndarray = self.convert_label(src_group['label'][()], label_group)

                    input_basename = os.path.basename(self.args['input'])
                    self.save_images(i, in_camera, gt_depth, gt_label, input_basename)

                if self.args["version"] == "result_data":
                    src_group: h5py.Group = data_group[str(i)]

                    Pred_Depth = src_group['Pred-Depth'][()]
                    # Pred_Label = src_group['Pred-Label'][()]
                    Pred_Label = src_group['GT-Label'][()]
                    GT_Depth = src_group['GT-Depth'][()]
                    GT_Label = src_group['GT-Label'][()]

                    u_grid, v_grid = np.meshgrid(np.arange(Pred_Depth.shape[1]), np.arange(Pred_Depth.shape[0] // 3, Pred_Depth.shape[0]))
                    # u_grid, v_grid = np.meshgrid(np.arange(Pred_Depth.shape[1]), np.arange(Pred_Depth.shape[0]))
                    px = u_grid - depth_front_Cx
                    py = -v_grid + depth_front_Cy
                    depth = Pred_Depth[v_grid, u_grid]

                    x = depth
                    y = -px * depth / depth_front_Fx
                    z = py * depth / depth_front_Fy

                    valid_mask = (Pred_Label[v_grid, u_grid] != 13) & (depth < 80)
                    x = x[valid_mask]
                    y = y[valid_mask]
                    z = z[valid_mask]

                    valid_labels = Pred_Label[v_grid, u_grid][valid_mask]
                    valid_labels = valid_labels.astype(int)
                    bgr = np.take(color_data, valid_labels, axis=0)

                    r = bgr[:, 2] / 255.0
                    g = bgr[:, 1] / 255.0
                    b = bgr[:, 0] / 255.0

                    Pred_point_cloud = np.stack((x, y, z), axis=-1)
                    Pred_color_cloud = np.stack((r, g, b), axis=-1)

                    u_grid, v_grid = np.meshgrid(np.arange(GT_Depth.shape[1]), np.arange(GT_Depth.shape[0]))
                    px = u_grid - depth_front_Cx
                    py = -v_grid + depth_front_Cy
                    depth = GT_Depth[v_grid, u_grid]

                    x = depth
                    y = -px * depth / depth_front_Fx
                    z = py * depth / depth_front_Fy

                    valid_mask = (GT_Label[v_grid, u_grid] != 13) & (depth < 80)
                    x = x[valid_mask]
                    y = y[valid_mask]
                    z = z[valid_mask]

                    valid_labels = GT_Label[v_grid, u_grid][valid_mask]
                    valid_labels = valid_labels.astype(int)
                    bgr = np.take(color_data, valid_labels, axis=0)

                    r = bgr[:, 2] / 255.0
                    g = bgr[:, 1] / 255.0
                    b = bgr[:, 0] / 255.0

                    GT_point_cloud = np.stack((x, y, z), axis=-1)
                    GT_color_cloud = np.stack((r, g, b), axis=-1)
                    
                    input_basename = os.path.basename(self.args['input'])

                    pr_point_cloud_path = os.path.join(self.output_dir, input_basename, 'pr_depth_ply', f'{i:06d}.ply')
                    gt_point_cloud_path = os.path.join(self.output_dir, input_basename, 'gt_depth_ply', f'{i:06d}.ply')
                    # pred_mask_img = self.point_cloud_to_image(Pred_point_cloud, depth_front_Cx, depth_front_Cy, depth_front_Fx, depth_front_Fy, (Pred_Depth.shape[0], Pred_Depth.shape[1]))
                    # print(np.max(pred_mask_img), np.min(pred_mask_img))
                    # pr_point_cloud_npy_path = os.path.join(self.output_dir, input_basename, 'pr_depth_ply', f'{i:06d}.npy')
                    # os.makedirs(os.path.dirname(pr_point_cloud_npy_path), exist_ok=True)
                    # np.save(pr_point_cloud_npy_path, pred_mask_img)
                    
                    
                    self.save_point_cloud2(Pred_point_cloud, Pred_color_cloud, pr_point_cloud_path)
                    self.save_point_cloud2(GT_point_cloud, GT_color_cloud, gt_point_cloud_path)
                    # gt_point_cloud = self.generate_point_cloud(GT_Depth, Cx, Cy, Fx, Fy)
                    # self.save_point_cloud(gt_point_cloud, gt_point_cloud_path)
                    # pr_point_cloud = self.generate_point_cloud(Pred_Depth, Cx, Cy, Fx, Fy)
                    # self.save_point_cloud(pr_point_cloud, pr_point_cloud_path)
                    src_group: h5py.Group = data_group[str(i)]

                    in_camera: np.ndarray = src_group['Input-Camera'][()]
                    # gt_depth: np.ndarray = src_group['GT-Depth'][()]
                    # pr_depth: np.ndarray = src_group['Pred-Depth'][()]
                    in_map: np.ndarray = self.depth2colormap(src_group['Input-Map'][()], 0.0, 100.0)
                    pr_seg: np.ndarray = self.convert_label(src_group['Pred-Label'], label_group)
                    pr_depth: np.ndarray = self.depth2colormap(src_group['Pred-Depth'][()], 0.0, 100.0)
                    gt_depth: np.ndarray = self.depth2colormap(src_group['GT-Depth'][()], 0.0, 100.0)
                    gt_label: np.ndarray = self.convert_label(src_group['GT-Label'][()], label_group)

                    self.save_images_result(i, in_camera, in_map, pr_seg, pr_depth, gt_depth, gt_label, input_basename)
                    # self.save_images_result(i, in_camera, in_map, pr_depth, gt_depth, gt_label, input_basename)


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str, metavar='PATH', required=True,
        help='Input path.'
    )
    parser.add_argument(
        '-v', '--version',
        type=str, metavar='STR', required=True,
        help='Version string.'
    )
    parser.add_argument(
        '-l', '--label',
        type=str, metavar='STR',
        help='label string.'
    )
    args = vars(parser.parse_args())
    return args

def main(args: Dict[str, str]):
    result_dir = "./dump_data"
    os.makedirs(result_dir, exist_ok=True)
    data_dump = DataDump(args, result_dir)
    data_dump.dump()

if __name__ == '__main__':
    args = parse_args()
    main(args)