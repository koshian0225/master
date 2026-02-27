import torch
import torch.backends.cudnn as cudnn

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors,inv_normalize
from networks.NewCRFDepth import NewCRFDepth

from collections import OrderedDict
import h5py
from scipy.spatial.transform import Rotation

import cv2
import open3d as o3d    
import re
import json
from visualize import *    
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from metric.metric import Metric



def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='nddepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')

parser.add_argument('--hdf5',                      type=str,   help='if set will use hdf5 file for evaluation',required=True)
parser.add_argument('--config',                      type=str,   help='if set will use config file for evaluation',required=True)
parser.add_argument('--save_points',                action='store_true',   help='if set will save point cloud',required=False)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader

class nddepth_pointcloud:
    def __init__(self,args):
        self.sequence_name = re.search(r'seq\d+',args.hdf5).group()
        
        # save dir
        self.save_dir = os.path.join('./models',self.sequence_name)
        self.pc_dir = os.path.join(self.save_dir,'point_cloud')
        self.scale_img_dir = os.path.join(self.save_dir,'trapezoid_visualization')
        if not os.path.exists(self.pc_dir):
            os.makedirs(self.pc_dir)
        if not os.path.exists(self.scale_img_dir):
            os.makedirs(self.scale_img_dir)
            
        # json list setting
        config_json_open = open(args.config,'r')
        config_json_load = json.load(config_json_open)
        self.dst_list:list = list(config_json_load['label']['config']['5class']['dst'].items())
        self.label_dict:dict = list(config_json_load['label']['config']['5class']['convert'].items())
        self.config_json_list:list = list()
        for (key, item) in self.label_dict:
            convert_dict:dict = {}
            convert_dict['src'] = int(key)
            convert_dict['dst'] = item
            self.config_json_list.append(convert_dict)
        
        # original image size
        self.img_height = None
        self.img_width = None
        # prediction depth image size
        self.depth_img_height = None
        self.depth_img_width = None
        
        self.cam_K = None
        self.inv_cam_K = None
        
        # label information
        self.label_info = None
        
        # camera transformation
        self.camera_transformation = None
        
        
    def hdf5_loader(self,args,pred_depth,frame_num):
        # hdf5 file
        with h5py.File(args.hdf5, 'r') as h5_data:
            # camera intrinsic
            Cx_ = h5_data['intrinsic']['image_00']['Cx'][()]
            Cy_ = h5_data['intrinsic']['image_00']['Cy'][()]
            Fx_ = h5_data['intrinsic']['image_00']['Fx'][()]
            Fy_ = h5_data['intrinsic']['image_00']['Fy'][()]
            # original image size
            self.img_height = h5_data['intrinsic']['image_00']['height'][()]
            self.img_width = h5_data['intrinsic']['image_00']['width'][()]

            # depth image size
            (self.depth_img_height, self.depth_img_width) = pred_depth.shape

            # calculate intrinsic threshold
            height_threshold = self.depth_img_height / self.img_height
            width_threshold = self.depth_img_width / self.img_width
            
            Fx = Fx_ * width_threshold
            Cx = Cx_ * width_threshold
            Fy = Fy_ * height_threshold
            Cy = Cy_ * height_threshold
            
            # define camera intrinsic
            self.cam_K = np.array([[Fx,0,Cx],
                                [0,Fy,Cy],
                                [0, 0, 1]])
            
            self.inv_cam_K = np.linalg.inv(self.cam_K)
            pose2cam0_transformation = np.identity(4)
            # tf_static
            pose2cam0_translation = h5_data['tf_static']['pose_to_cam0']['translation'][()]
            # quaternion
            pose2cam0_rotation = h5_data['tf_static']['pose_to_cam0']['rotation'][()]
            # rotation transform
            cam0_quat = Rotation.from_quat(pose2cam0_rotation)
            pose2cam0_mat = cam0_quat.as_matrix()
            pose2cam0_transformation[:3,:3] = pose2cam0_mat
            pose2cam0_transformation[:3,3] = pose2cam0_translation

            # tf world to pose
            world_translation:np.ndarray = h5_data['data'][f'{frame_num}']['world_to_pose']['translation'][()]
            # quaternion
            world_rotation:np.ndarray = h5_data['data'][f'{frame_num}']['world_to_pose']['rotation'][()]
            
            # rotation transform
            world_quat = Rotation.from_quat(world_rotation)
            world_mat = world_quat.as_matrix()
            
            world_transformation = np.identity(4)
            world_transformation[:3,:3] = world_mat
            world_transformation[:3,3] = world_translation
            
            # calculate camera pose
            self.camera_transformation:np.ndarray = np.dot(world_transformation,pose2cam0_transformation)
            
            # color information
            label_img:np.ndarray = h5_data['data'][f'{frame_num}']['semantic'][()]
        # crop
            if args.do_kb_crop:
                height = label_img.shape[0]
                width = label_img.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                label_img = label_img[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                
            self.label_info = label_img.flatten()

            # convert label
            for convert_config in self.config_json_list:
                self.label_info[np.where(self.label_info == convert_config['src'])] = convert_config['dst']
                
            # class数を5classにする
            self.convert_label_image = self.label_info.reshape(label_img.shape)
            
    def project_3dpoints(self,pred_depth,image,frame_num,show=False):
        # define condition ground label is 1
        ground_condition = (self.convert_label_image == 1)
        # 台形の任意のピクセルを定義
        bottom_pixels = 100
        trapezoid_height = 50
        # 画像の下部分の任意のピクセル抽出
        bottom_pixels_mask = np.zeros_like(ground_condition)
        bottom_pixels_mask[352-bottom_pixels:352,:] = True

        # 台形定義の前に台形の幅や高さを定義する
        ground_and_bottom_pixels = np.logical_and(ground_condition,bottom_pixels_mask)  
        ground_and_botom_img = np.where(ground_and_bottom_pixels==True,np.array(self.convert_label_image),0)
        
        # 一番下のピクセルかつground labelのピクセルのインデックスを取得
        bottom_yind = np.max(np.where(np.any(ground_and_botom_img,axis=1)))
        min_xind = np.min(np.where(np.all(ground_and_botom_img[bottom_yind,:].reshape(-1,1),axis=1)))
        max_xind = np.max(np.where(np.all(ground_and_botom_img[bottom_yind,:].reshape(-1,1),axis=1)))

        # 台形の形でラベル情報抽出
        trapezoid_mask = np.zeros_like(ground_condition)
        vertices = np.array([[min_xind+(trapezoid_height/2), bottom_yind-trapezoid_height], [min_xind, bottom_yind], [max_xind, bottom_yind], [max_xind - (trapezoid_height/2), bottom_yind-trapezoid_height]], dtype=np.int32)
        # vertices = vertices.reshape((-1, 1, 2))
        mask = np.zeros_like(self.convert_label_image)
        cv2.fillPoly(mask, [vertices], (255, 255, 255))
        # 台形として抽出された部分をboolean配列として取り出す
        trapezoid_mask = np.where(mask == 255,True,False)
        
        # mask情報を統合
        mask_condition = np.logical_and(trapezoid_mask,np.logical_and(ground_condition,bottom_pixels_mask))
        # maskをもとに台形型にdepth情報抽出
        trapezoid_depth = np.where(mask_condition==True,np.array(pred_depth),0)

        # visualize
        plt.style.use('dark_background')
        fig, ax = plt.subplots(3,figsize=(12,15),constrained_layout=True)
        visualize_image(image,ax=ax[0])
        # visualize_label(trapezoid_image,ax=ax[1])
        # visualize_depth(trapezoid_depth,ax=ax[2])
        visualize_label_alphamask(self.convert_label_image,mask_condition,ax=ax[1])
        visualize_depth_alphamask(pred_depth,mask_condition,ax=ax[2])
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.scale_img_dir,f'{frame_num:06d}.png'))
            plt.close('all')
        
        # 逆投影の計算を行う pred_depthはdepth画像 inv_cam_Kはカメラの内部パラメータの逆行列　２つを使ってカメラ軸の点群を計算する
        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        # depth = pred_depth.reshape(-1) # 1次元に変換
        depth = trapezoid_depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        cam_points = uvw_normalized * depth[:, None]
        # make label points
        label_points:np.ndarray = np.hstack([cam_points,self.label_info.reshape(-1,1)])
        
        semantic1d_color = np.empty((self.label_info.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(self.label_info == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0

        scale = self.scale_calculator(nd_label_points=label_points)
        
        # scale_cam_points = uvw_normalized * (depth[:, None] * scale)
        # scale_cam_points = np.stack([scale_cam_points[:,0],scale_cam_points[:,1],scale_cam_points[:,2],np.ones_like(scale_cam_points[:,0])],axis=-1)
        # print(np.mean(scale_cam_points[:,1]))

        # # world_points = np.dot(camera_transformation,scale_cam_points.T).T[:,:3]
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(world_points)   
        # pcd.colors = o3d.utility.Vector3dVector(semantic1d_color)  
        # if args.save_points:
        #     o3d.io.write_point_cloud(os.path.join(pc_dir,f'{frame_num:06d}.ply'),pcd,write_ascii=True)
        return scale
    
    def project_evaldepth(self,pred_depth,gt_depth,frame_num):
        # 逆投影の計算を行う pred_depthはdepth画像 inv_cam_Kはカメラの内部パラメータの逆行列　２つを使ってカメラ軸の点群を計算する
        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        # depth = pred_depth.reshape(-1) # 1次元に変換
        depth = pred_depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        # cam_points = uvw_normalized * depth[:, None]
        # make label points
        # label_points:np.ndarray = np.hstack([cam_points,self.label_info.reshape(-1,1)])
        
        semantic1d_color = np.empty((self.label_info.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(self.label_info == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0
        
        scale_cam_points = uvw_normalized * depth[:, None]
        scale_cam_points = np.stack([scale_cam_points[:,0],scale_cam_points[:,1],scale_cam_points[:,2],np.ones_like(scale_cam_points[:,0])],axis=-1)
        # print(np.mean(scale_cam_points[:,1]))

        # world_points = np.dot(self.camera_transformation,scale_cam_points.T).T[:,:3]
        
        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        gt_depth = gt_depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        # gt_cam_points = uvw_normalized * gt_depth[:, None]
        # make label points
        # label_points:np.ndarray = np.hstack([gt_cam_points,self.label_info.reshape(-1,1)])
        
        semantic1d_color = np.empty((self.label_info.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(self.label_info == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0
        
        gt_scale_cam_points = uvw_normalized * gt_depth[:, None]
        gt_scale_cam_points = np.stack([gt_scale_cam_points[:,0],gt_scale_cam_points[:,1],gt_scale_cam_points[:,2],np.ones_like(gt_scale_cam_points[:,0])],axis=-1)
        # print(np.mean(gt_scale_cam_points[:,1]))

        # gt_world_points = np.dot(self.camera_transformation,gt_scale_cam_points.T).T[:,:3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scale_cam_points[:,:3])   
        pcd.colors = o3d.utility.Vector3dVector(semantic1d_color)  
        
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_scale_cam_points[:,:3])   
        gt_pcd.colors = o3d.utility.Vector3dVector(semantic1d_color)  

        if args.save_points:
            o3d.io.write_point_cloud(os.path.join(self.pc_dir,f'pred_{frame_num:06d}.ply'),pcd,write_ascii=True) 
            o3d.io.write_point_cloud(os.path.join(self.pc_dir,f'gt_{frame_num:06d}.ply'),gt_pcd,write_ascii=True) 

    # cameraの高さとラベル付き点群からスケールを計算する関数
    def scale_calculator(self,nd_label_points:np.ndarray,translation:np.ndarray=None,camera_translation:np.ndarray=None):
        # extract ground points ground label is 1 y軸方向が高さ　負方向がプラスになる
        ground_points = nd_label_points[:,1][nd_label_points[:,1].nonzero()]
        ground_height = np.median(ground_points)
        # ground_condition = (nd_label_points[:,3] == 1)    
        # ground_points = nd_label_points[ground_condition][:,:3]
        # ground_height = np.mean(ground_points[:,1])
        # print(f'ground_height:{ground_height}')
        
        # define camera height in camera axis
        camera_height = 1.55

        # define camera height in world axis
        # world_camera_height = camera_translation[2]
        scale = abs(ground_height) / abs(camera_height)
        print(f'scale:{scale}')
        return scale
    
    def scale_calculator_new(self,pred_depth:np.ndarray):
        """
        self.convert_label_image:np.ndarray
            5classのラベル画像
        pred_depth:np.ndarray
            推論したdepth画像(抽出なし)
        
        """
        camera_height = 1.55
        
        # 地面点群のみを抽出する
        ground_condition = (self.convert_label_image == 1)
        ground_depth = np.where(ground_condition==True,pred_depth,0)
        
        # 逆投影の計算を行う pred_depthはdepth画像 inv_cam_Kはカメラの内部パラメータの逆行列　２つを使ってカメラ軸の点群を計算する
        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        # depth = pred_depth.reshape(-1) # 1次元に変換
        depth = ground_depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        
        semantic1d_color = np.empty((self.label_info.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(self.label_info == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0

        scale_cam_points = uvw_normalized * depth[:, None]
        scale_cam_points = np.stack([scale_cam_points[:,0],scale_cam_points[:,1],scale_cam_points[:,2],np.ones_like(scale_cam_points[:,0])],axis=-1)
        # 最小二乗法にて求めた切片
        ground_height = self.calculate_ground_height(scale_cam_points)
        print(f'ground_height:{ground_height}')
        scale = abs(camera_height) /abs(ground_height)  
        print(f'scale:{scale}')

    def calculate_ground_height(self,scale_cam_points):
        # y, z座標を取得
        y = scale_cam_points[:, 1].reshape(-1, 1)
        z = scale_cam_points[:, 2]

        # 地面の傾きと切片を求める
        model = LinearRegression()
        model.fit(y, z)

        # z = 0 のときの y の値（切片）を求める
        ground_height = -model.intercept_ / model.coef_

        return ground_height

def output_scalecsv(scale_list:list,save_dir:str):
    scale_list = np.array(scale_list)
    scale_list = scale_list.reshape(-1,1)
    np.savetxt(os.path.join(save_dir,'scale.csv'),scale_list,delimiter=',')
    
def output_csvfromdict(eval_dict:dict,save_dir:str):
    """
    eval_dict: dict
        key: metric name (MAE,RMSE,MAPE) -> all,car,person
        value: metric value
    """
    first_idx = list()
    column_idx = list()
    data_list = list()
    
    for key,item in eval_dict.items():
        first_idx.append(key)
        column_idx = [s for s in item.keys() if s not in column_idx]
        for key2,item2 in item.items():
            data_list.append(item2)
            
    """
    data_list:list [all:mae,car:mae,person:mae,all:rmse,car:rmse,person:rmse,all:mape,car:mape,person:mape]
    data_listをnumpy arrayに変換してreshape
    """
    data_array = np.array(data_list).reshape((len(first_idx),len(column_idx)))
            
    df = pd.DataFrame(data = data_array,index = first_idx,columns = column_idx)
    df.to_csv(os.path.join(save_dir,'eval.csv'))



def eval_pmod(model, dataloader_eval, epoch=5, post_process=False,args=None):
    # scale_calculator(args=args,pred_depth=None)
    # eval_measures = torch.zeros(10).cuda()
    
    # pmod-netの評価指標を定義
    eval_dict = OrderedDict()
    eval_dict['MAE'] = {'car':0,'person':0,'all':0}
    eval_dict['RMSE'] = {'car':0,'person':0,'all':0}
    eval_dict['MAPE'] = {'car':0,'person':0,'all':0}
    
    label_dict = OrderedDict()
    label_dict['car'] = 3
    label_dict['person'] = 4
    
    # 評価のためのクラスをインスタンス化
    metric = Metric(debug=True)
    
    scale_list = list()
    
    for frame_num, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue
            
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            color_image = inv_normalize(image[0, :, :, :]).permute(1, 2, 0).cpu().numpy()
            inv_K = torch.autograd.Variable(eval_sample_batched['inv_K'].cuda())
            depth1_list, uncer1_list, depth2_list, uncer2_list, _, _ = model(image, inv_K, epoch)
            if epoch < 5:
                pred_depth = 0.5 * (depth1_list + depth2_list)
            else:
                pred_depth = 0.5 * (depth1_list[-1] + depth2_list[-1])
            if post_process:
                image_flipped = flip_lr(image)
                depth1_list_flipped, uncer1_list_flipped, depth2_list_flipped, uncer2_list_flipped, _, _ = model(image_flipped, inv_K, epoch)
                if epoch < 5:
                    pred_depth_flipped = 0.5 * (depth1_list_flipped + depth2_list_flipped)
                else:
                    pred_depth_flipped = 0.5 * (depth1_list_flipped[-1] + depth2_list_flipped[-1])
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        # 評価に使われるpred_depthを点群化して評価します
        predict_scale = nddepth_pointcloud(args=args)
        predict_scale.hdf5_loader(args=args,pred_depth=pred_depth,frame_num=frame_num)
        # predict_scale.scale_calculator_new(pred_depth=pred_depth)
        # scale = predict_scale.project_3dpoints(pred_depth=pred_depth,image=color_image,frame_num=frame_num,show=False)

        # scale_list.append(scale)
        # # scale倍してpred_depthを評価します
        # pred_depth = pred_depth * scale

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
        # measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        # print('rms',measures[3])
        # mimにあるオリジナルのgt_depthを読み込む -> 結果256で割られているため256倍する必要がある
        # original_gt_depth = cv2.imread(os.path.join(args.gt_path_eval,f'{frame_num:06d}_gt.png'),cv2.IMREAD_UNCHANGED)
        # height, width = original_gt_depth.shape
        # top_margin = int(height - 352)
        # left_margin = int((width - 1216) / 2)
        # crop_orig_gtdepth = np.zeros((352, 1216), dtype=np.float32)
        # crop_orig_gtdepth = original_gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        # print('0',crop_orig_gtdepth - gt_depth)
        # print('0',np.unique(crop_orig_gtdepth - gt_depth))
        # print(crop_orig_gtdepth.max(),gt_depth.max())
        # exit()
        
        # 点群逆投影のためのプログラム
        # p_depth = np.where(valid_mask==True,pred_depth,0)
        # g_depth = np.where(valid_mask==True,gt_depth,0) * 256.0
        # predict_scale.project_evaldepth(pred_depth=p_depth,gt_depth=g_depth,frame_num=frame_num)

        metric.evaluate_allmetric(pred_depth[valid_mask],gt_depth[valid_mask] * 256.0)
        # 人、車の動的物体に対してのみ評価する
        for key in label_dict.keys():
            condition = (predict_scale.convert_label_image == label_dict[key])
            dynamic_condition = np.logical_and(condition,valid_mask)
            metric.evaluate_dynamicmetric(pred_depth[dynamic_condition],gt_depth[dynamic_condition] * 256.0,key=key)
    
    # eval_dict['MAE']['all'] = metric.eval_dict['AE']['all'] / metric.count
    # eval_dict['RMSE']['all'] = metric.eval_dict['MSE']['all'] / metric.count
    # eval_dict['MAPE']['all'] = metric.eval_dict['APE']['all'] / metric.count  
    metric.get_final_metrics()
    output_scalecsv(scale_list=scale_list,save_dir=predict_scale.save_dir)
    output_csvfromdict(eval_dict=metric.eval_dict,save_dir=predict_scale.save_dir)
    return None


def main_worker(args):

    # CRF model
    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model.cuda()

    print("== Model Initialized")
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
            model = torch.nn.DataParallel(model)

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            # del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        eval_pmod(model, dataloader_eval, post_process=True,args=args)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
