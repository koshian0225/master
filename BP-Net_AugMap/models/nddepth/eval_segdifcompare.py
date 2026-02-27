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

if args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
if args.dataset == 'kitti':
    from dataloaders.dataloader_kitti import NewDataLoader
if args.dataset == 'carla':
    from dataloaders.dataloader_carla import NewDataLoader
if args.dataset == 'real':
    from dataloaders.dataloader_real import NewDataLoader

class nddepth_pointcloud:
    def __init__(self,args):
        if args.dataset == 'kitti' or args.dataset == 'nyu':
            self.sequence_name = re.search(r'seq\d+',args.hdf5).group()
        elif args.dataset == 'carla':
            self.sequence_name = re.search(r'Town\d+',args.hdf5).group()
        elif args.dataset == 'real':
            self.sequence_name = args.hdf5.split('/')[-1].split('.')[0]
        
        # save dir
        self.save_dir = os.path.join('./models',args.model_name + self.sequence_name)
        self.pc_dir = os.path.join(self.save_dir,'point_cloud')
        self.pred_depthdir = os.path.join(self.save_dir,'pred_depth')
        self.gt_depthdir = os.path.join(self.save_dir,'gt_depth')

        # self.scale_img_dir = os.path.join(self.save_dir,'trapezoid_visualization')
        if not os.path.exists(self.pc_dir):
            os.makedirs(self.pc_dir)
        if not os.path.exists(self.pred_depthdir):
            os.makedirs(self.pred_depthdir)
        if not os.path.exists(self.gt_depthdir):
            os.makedirs(self.gt_depthdir)
        # if not os.path.exists(self.scale_img_dir):
            # os.makedirs(self.scale_img_dir)
        os.makedirs(os.path.join(self.pc_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(self.pc_dir, 'pred'), exist_ok=True)
            
        # json list setting
        config_json_open = open(args.config,'r')
        config_json_load = json.load(config_json_open)
        if args.dataset == 'kitti' or args.dataset == 'nyu':
            self.dst_list:list = list(config_json_load['label']['config']['5class']['dst'].items())
            self.label_dict:dict = list(config_json_load['label']['config']['5class']['convert'].items())
        elif args.dataset == 'carla':
            self.dst_list:list = list(config_json_load['label']['src']['carla-0.9.14'].items())
            self.label_dict:dict = list(config_json_load['label']['config']['carla-0.9.14']['convert'].items())
        elif args.dataset == 'real':
            self.dst_list:list = list(config_json_load['label']['config']['label']['dst'].items())
            self.label_dict:dict = list(config_json_load['label']['config']['label']['convert'].items())

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
            if args.dataset == 'kitti' or args.dataset == 'nyu':
                # camera intrinsic
                Cx_ = h5_data['intrinsic']['image_00']['Cx'][()]
                Cy_ = h5_data['intrinsic']['image_00']['Cy'][()]
                Fx_ = h5_data['intrinsic']['image_00']['Fx'][()]
                Fy_ = h5_data['intrinsic']['image_00']['Fy'][()]
                # original image size
                self.img_height = h5_data['intrinsic']['image_00']['height'][()]
                self.img_width = h5_data['intrinsic']['image_00']['width'][()]
            elif args.dataset == 'carla':
                # camera intrinsic
                # tmp load for intrinsic Cx, Cy, Fx, Fy, height, width
                with open('/home/NDdepth/Estimation/datasets/intrinsic_carla.txt', 'r')as f:
                    data = f.readline().split(' ')
                    Cx_ = float(data[0])
                    Cy_ = float(data[1])
                    Fx_ = float(data[2])
                    Fy_ = float(data[3])
                    self.img_height = int(data[4])
                    self.img_width = int(data[5])
                    
                # Cx_ = h5_data['intrinsic']['sensor.camera.depth-front']['Cx'][()]
                # Cy_ = h5_data['intrinsic']['sensor.camera.depth-front']['Cy'][()]
                # Fx_ = h5_data['intrinsic']['sensor.camera.depth-front']['Fx'][()]
                # Fy_ = h5_data['intrinsic']['sensor.camera.depth-front']['Fy'][()]
                # # original sensor.camera.depth-front size
                # self.img_height = h5_data['intrinsic']['sensor.camera.depth-front']['height'][()]
                # self.img_width = h5_data['intrinsic']['sensor.camera.depth-front']['width'][()]

            elif args.dataset == 'real':
                # camera intrinsic
                Cx_ = h5_data['K']['camera']['Cx'][()]
                Cy_ = h5_data['K']['camera']['Cy'][()]
                Fx_ = h5_data['K']['camera']['Fx'][()]
                Fy_ = h5_data['K']['camera']['Fy'][()]
                # original camera size
                self.img_height = h5_data['K']['camera']['height'][()]
                self.img_width = h5_data['K']['camera']['width'][()]

            # depth image size
            (self.depth_img_height, self.depth_img_width) = pred_depth.shape

            # calculate intrinsic threshold
            if args.dataset == 'kitti' or args.dataset == 'nyu':
                self.depth_img_height = 376
                self.depth_img_width = 1408
                height_threshold = self.depth_img_height / self.img_height
                width_threshold = self.depth_img_width / self.img_width
            else:
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
            # color information
            if args.dataset == 'kitti' or args.dataset == 'nyu':
                label_img:np.ndarray = h5_data['data'][f'{frame_num}']['semantic'][()]
            elif args.dataset == 'carla':
                label_img:np.ndarray = h5_data['data'][f'{frame_num}']['sensor.camera.semantic_segmentation-front_nd'][()]
            elif args.dataset == 'real':
                label_img:np.ndarray = h5_data['data'][f'{frame_num}']['label'][()]
                label_img = cv2.resize(label_img,(self.depth_img_width,self.depth_img_height), interpolation=cv2.INTER_NEAREST)
        # crop
            if args.do_kb_crop:
                if args.dataset == 'kitti':
                    pass
                else:
                    height = label_img.shape[0]
                    width = label_img.shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    label_img = label_img[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                
            self.label_info = label_img.flatten()
            label_info = np.zeros_like(self.label_info)

            # convert label
            for convert_config in self.config_json_list:
                label_info[np.where(self.label_info == convert_config['src'])] = convert_config['dst']
            self.label_info = label_info
                
            # class数を5classにする
            self.convert_label_image = label_info.reshape(label_img.shape)
            
    def save_pred_depth(self, pred_depth, frame_num):
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255
        cv2.imwrite(os.path.join(self.pred_depthdir, f'{frame_num:06d}_pred.png'), pred_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    def save_gt_depth(self, gt_depth, frame_num):
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min()) * 255
        cv2.imwrite(os.path.join(self.gt_depthdir, f'{frame_num:06d}_gt.png'), gt_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
    def project_evaldepth(self,depth,gt_depth,frame_num, effectiveness:bool = False):
        if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'carla':
            depth[depth > 80.0] = 0.0
        elif args.dataset == 'real':
            depth[depth > 20.0] = 0.0
            
        if effectiveness:
            gt_depth_condition = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            depth[~gt_depth_condition] = 0.0
        # 逆投影の計算を行う pred_depthはdepth画像 inv_cam_Kはカメラの内部パラメータの逆行列　２つを使ってカメラ軸の点群を計算する
        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        depth = depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        
        scale_cam_points = uvw_normalized * depth[:, None]
        scale_cam_points = np.stack([scale_cam_points[:,0],scale_cam_points[:,1],scale_cam_points[:,2],np.ones_like(scale_cam_points[:,0])],axis=-1)

        u,v = np.meshgrid(np.arange(self.depth_img_width),np.arange(self.depth_img_height))
        uvw = np.stack([u,v,np.ones_like(u)],axis=-1)
        gt_depth = gt_depth.reshape(-1) # 1次元に変換
        uvw_normalized = np.dot(uvw.reshape(-1, 3), self.inv_cam_K.T)
        
        semantic1d_color = np.zeros((self.label_info.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(self.label_info == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0
        
        gt_scale_cam_points = uvw_normalized * gt_depth[:, None]
        gt_scale_cam_points = np.stack([gt_scale_cam_points[:,0],gt_scale_cam_points[:,1],gt_scale_cam_points[:,2],np.ones_like(gt_scale_cam_points[:,0])],axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scale_cam_points[:,:3])   
        pcd.colors = o3d.utility.Vector3dVector(semantic1d_color)  
        
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_scale_cam_points[:,:3])   
        gt_pcd.colors = o3d.utility.Vector3dVector(semantic1d_color)  

        if args.save_points:
            o3d.io.write_point_cloud(os.path.join(self.pc_dir, 'pred',f'pred_{frame_num:06d}.ply'),pcd,write_ascii=True) 
            o3d.io.write_point_cloud(os.path.join(self.pc_dir, 'gt',f'gt_{frame_num:06d}.ply'),gt_pcd,write_ascii=True) 
    
def output_csvfromdict(eval_dict:dict,save_dir:str):
    """
    eval_dict: dict
        key: metric name (MAE,RMSE,MAPE) -> all,car,person
        value: metric value
    """
    first_idx = list()
    column_idx = list()
    data_list = list()
    print("!!!!!!!!!!!!!!!!!!!!!")
    
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

def output_csvfromdict_crop(eval_dict:dict,save_dir:str):
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
    df.to_csv(os.path.join(save_dir,'eval_crop.csv'))



def eval_pmod(model, dataloader_eval, epoch=5, post_process=False,args=None):
    # scale_calculator(args=args,pred_depth=None)
    # eval_measures = torch.zeros(10).cuda()
    
    label_dict = OrderedDict()
    if args.dataset == 'carla':
        label_dict['car'] = 10
        label_dict['person'] = 4
    elif args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'real':
        label_dict['car'] = 3
        label_dict['person'] = 4
    
    # 評価のためのクラスをインスタンス化
    metric = Metric(debug=True)
    crop_metric = Metric(debug=True)
    
    for frame_num, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            #GTのdepthを取得　有効な深度がない場合はスキップ
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue
            
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            #可視化用に正規化を解除
            color_image = inv_normalize(image[0, :, :, :]).permute(1, 2, 0).cpu().numpy()
            #カメラの内部パラメータを取得
            inv_K = torch.autograd.Variable(eval_sample_batched['inv_K'].cuda())
            #モデルに画像と逆行列を入力し、2本の推論結果と不確かさ??????を得る
            depth1_list, uncer1_list, depth2_list, uncer2_list, _, _ = model(image, inv_K, epoch)
            if epoch < 5: #２つの推論結果の平均を取る
                pred_depth = 0.5 * (depth1_list + depth2_list)
            else: # 5エポック以降は最後の推論結果を使用
                pred_depth = 0.5 * (depth1_list[-1] + depth2_list[-1])
            # 画像を左右反転させて、モデルに入力し、推論結果を得る
            if post_process:
                image_flipped = flip_lr(image)
                depth1_list_flipped, uncer1_list_flipped, depth2_list_flipped, uncer2_list_flipped, _, _ = model(image_flipped, inv_K, epoch)
                if epoch < 5:
                    pred_depth_flipped = 0.5 * (depth1_list_flipped + depth2_list_flipped)
                else:
                    pred_depth_flipped = 0.5 * (depth1_list_flipped[-1] + depth2_list_flipped[-1])
                #左右両方の結果を組み合わせて最終の深度を補正
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        # do_kb_cropが有効ならKITTIのトリミングサイズに合わせて深度マップを復元
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped
            
        if args.dataset == 'kitti':
            pad_rows = 376 - pred_depth.shape[0]
            pad_cols = (1408 - pred_depth.shape[1]) // 2
            pred_depth = np.pad(pred_depth, ((pad_rows, 0), (pad_cols, pad_cols)), 'constant', constant_values=0)
            gt_depth = np.pad(gt_depth, ((pad_rows, 0), (pad_cols, pad_cols)), 'constant', constant_values=0)
        
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        # 評価に使われるpred_depthを点群化して評価します
        predict_scale = nddepth_pointcloud(args=args)
        predict_scale.hdf5_loader(args=args,pred_depth=pred_depth,frame_num=frame_num)
        # for rapid evaluate town07
        # predict_scale.project_evaldepth(depth=pred_depth,gt_depth=gt_depth,frame_num=frame_num, effectiveness=True)

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

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        metric.evaluate_allmetric(pred_depth[valid_mask],gt_depth[valid_mask])
        if args.dataset == 'carla':
            static_condition = np.logical_and(predict_scale.convert_label_image != 4, predict_scale.convert_label_image != 10, predict_scale.convert_label_image != 0)
        else:
            static_condition = np.logical_and(predict_scale.convert_label_image != 3, predict_scale.convert_label_image != 4)
        static_valid_mask = np.logical_and(valid_mask,static_condition)
        # metric.evaluate_staticmetric(pred_depth[static_valid_mask],gt_depth[static_valid_mask],save_dir=predict_scale.save_dir)
        metric.evaluate_staticmetric(pred_depth[static_valid_mask],gt_depth[static_valid_mask])
        # 人、車の動的物体に対してのみ評価する
        for key in label_dict.keys():
            condition = (predict_scale.convert_label_image == label_dict[key])
            dynamic_condition = np.logical_and(condition,valid_mask)
            # metric.evaluate_dynamicmetric(pred_depth[dynamic_condition],gt_depth[dynamic_condition],key=key, save_dir=predict_scale.save_dir)
            metric.evaluate_dynamicmetric(pred_depth[dynamic_condition],gt_depth[dynamic_condition],key=key)

            
        
        # # # crop evaluation
        # original_shape:tuple = (600, 800) # (H,W)
        # nd_shape:tuple = pred_depth.shape # (H,W)
        # left_margin:int = int((nd_shape[1] - original_shape[1]) // 2)
        # right_margin:int = int(nd_shape[1] - left_margin)
        # crop_pred_depth:np.ndarray = pred_depth[:,left_margin:right_margin]
        # crop_gt_depth:np.ndarray = gt_depth[:,left_margin:right_margin]
        # crop_label_img:np.ndarray = predict_scale.convert_label_image[:,left_margin:right_margin]
        
        # valid_crop_mask = np.logical_and(crop_gt_depth > args.min_depth_eval, crop_gt_depth < args.max_depth_eval)
        # crop_metric.evaluate_allmetric(crop_pred_depth[valid_crop_mask],crop_gt_depth[valid_crop_mask])

        # if args.dataset == 'carla':
        #     static_condition = np.logical_and(crop_label_img != 4, crop_label_img != 10, crop_label_img != 0)
        # else:
        #     static_condition = np.logical_and(crop_label_img != 3, crop_label_img != 4)
        # static_valid_crop_mask = np.logical_and(valid_crop_mask,static_condition)
        # # crop_metric.evaluate_staticmetric(crop_pred_depth[static_valid_crop_mask],crop_gt_depth[static_valid_crop_mask],save_dir=predict_scale.save_dir)
        # # crop_metric.evaluate_staticmetric(crop_pred_depth[static_valid_crop_mask],crop_gt_depth[static_valid_crop_mask])
        # # 人、車の動的物体に対してのみ評価する
        # for key in label_dict.keys():
        #     condition = (crop_label_img == label_dict[key])
        #     dynamic_condition = np.logical_and(condition,valid_crop_mask)
        #     # crop_metric.evaluate_dynamicmetric(crop_pred_depth[dynamic_condition],crop_gt_depth[dynamic_condition],key=key, save_dir=predict_scale.save_dir)
        #     # crop_metric.evaluate_dynamicmetric(crop_pred_depth[dynamic_condition],crop_gt_depth[dynamic_condition],key=key)


        pred_depth = pred_depth.astype(np.uint8)
        gt_depth = gt_depth.astype(np.uint8)
        # normalization 0~1 
        predict_scale.save_pred_depth(pred_depth, frame_num)
        print(pred_depth)
        predict_scale.save_gt_depth(gt_depth, frame_num)

    
    metric.get_final_metrics()
    # crop_metric.get_final_metrics()
    output_csvfromdict(eval_dict=metric.eval_dict,save_dir=predict_scale.save_dir)
    # output_csvfromdict_crop(eval_dict=crop_metric.eval_dict,save_dir=predict_scale.save_dir)
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
