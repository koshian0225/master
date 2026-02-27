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

class SAVE_NPY:
    def __init__(self,args):
        self.sequence_name = re.search(r'seq\d+',args.hdf5).group()
        
        # save dir
        self.save_dir = os.path.join('./models',self.sequence_name)
        self.result_imgdir = os.path.join(self.save_dir,'result_npy')
        if not os.path.exists(self.result_imgdir):
            os.makedirs(self.result_imgdir)

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
        if re.search(r'seq\d+',args.hdf5).group() == 'seq09': 
            if frame_num <= 9648:
                continue
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

        save_img__ = SAVE_NPY(args)
        np.save(os.path.join(save_img__.result_imgdir, f'{frame_num:06d}.npy'), pred_depth)
        # cv2.imwrite(os.path.join(save_img__.result_imgdir,f'{frame_num:06d}.png'),pred_depth.astype(np.float32),[cv2.IMWRITE_PNG_COMPRESSION, 0])

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
