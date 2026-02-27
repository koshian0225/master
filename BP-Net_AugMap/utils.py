#!/usr/bin/env
# -*- coding: utf-8 -*-
# @Filename : utils
# @Date : 2022-05-06
# @Project: BP-Net
# @AUTHOR : jie

import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
import cv2
import augs
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from collections import OrderedDict
import math
import datetime
import dill

# NDDepth
from models.nddepth.networks.NewCRFDepth import NewCRFDepth

__all__ = [
    'AverageMeter',
    'Trainer',
]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        self.cfg.gpu_id = self.cfg.gpus[self.rank]
        self.init_gpu()
        self.ddp = len(self.cfg.gpus) > 1
        self.iter = 0
        self.epoch = 0
        self.best_metric_ema = float('inf')
        #####################################################################################
        now = datetime.datetime.now().strftime("%m-%d-%H%M")
        self.save_path = os.path.join('checkpoints', f"{self.cfg.name}-{now}")
        self.log = self.init_log()
        self.init_device()
        self.init_seed()
        self.writer = self.init_viz()
        self.trainloader, self.testloader = self.init_dataset()
        net = self.init_net()
        criterion = self.init_loss()
        metric = self.init_metric()
        self.net, self.criterion, self.metric = self.init_cuda(net, criterion, metric)
        self.net_ema = self.init_ema()
        if self.ddp:
            self.net = DDP(self.net, find_unused_parameters=False)
        self.optimizer = self.init_optim()
        self.lr_scheduler = self.init_sched_lr()
        self.lr_iter = OmegaConf.select(self.cfg.sched.lr, 'iter', default=False)
        self.clip = self.init_clip()

    def init_log(self):
        return Blank() if self.rank else logging.getLogger(f'{self.cfg.name}')

    def init_device(self):
        torch.cuda.set_device(f'cuda:{self.cfg.gpu_id}')
        if self.ddp:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        self.ddp_log(f'device is {self.cfg.gpu_id}', always=True)

    def init_seed(self):
        manual_seed = self.cfg.manual_seed
        self.ddp_log(f"Random Seed: {manual_seed:04d}")
        torch.initial_seed()
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    def init_viz(self):
        if self.rank:
            return Blank()
        else:
            writer_name = os.path.join('runs', f'{self.cfg.name}')
            return SummaryWriter(writer_name)

    def init_dataset(self):
        trainset = instantiate(self.cfg.data.trainset)
        testset = instantiate(self.cfg.data.testset)
        if self.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset,
                num_replicas=len(self.cfg.gpus),
                rank=self.rank,
                shuffle=True,
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset,
                num_replicas=len(self.cfg.gpus),
                rank=self.rank,
                shuffle=False,
            )
        else:
            train_sampler = None
            test_sampler = None
        self.train_sampler = train_sampler
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.cfg.train_batch_size,
                                                  num_workers=self.cfg.num_workers, shuffle=(train_sampler is None),
                                                  sampler=train_sampler,
                                                  drop_last=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.cfg.test_batch_size,
                                                 num_workers=self.cfg.num_workers, shuffle=False,
                                                 sampler=test_sampler,
                                                 drop_last=True, pin_memory=True)
        self.ddp_log(f'num_train = {len(trainloader)}, num_test = {len(testloader)}')
        return trainloader, testloader

    # checkpointsの扱い変更に伴う修正
    def init_net(self):
        model = instantiate(self.cfg.net.model)
        nd_in_ckpt = False

        if 'chpt' in self.cfg:
            self.ddp_log(f'resume CHECKPOINTS')
            save_path = os.path.join('checkpoints', self.cfg.chpt)
            ckpt_path = os.path.join(save_path, 'result_ema.pth')
            cp = torch.load(ckpt_path, map_location=torch.device('cpu'), pickle_module=dill)
            state = cp['net']

            nd_in_ckpt = any(k.startswith('nddepth_model.') for k in state.keys())
            try:
                if nd_in_ckpt:
                    from models.nddepth.networks.NewCRFDepth import NewCRFDepth
                    nddepth_model = NewCRFDepth(version='large07', inv_depth=False, max_depth=80)
                    model.set_nddepth_model(nddepth_model, trainable=False)  # CPU上でOK
                    model.load_state_dict(state, strict=True)
                    self.ddp_log('Loaded BPNet + NDDepth weights from checkpoint')
                else:
                    net_state_dict = {k: v for k, v in state.items() if not k.startswith('nddepth_model.')}
                    model.load_state_dict(net_state_dict, strict=True)
                    self.ddp_log('Loaded BPNet weights (NDDepth not in checkpoint)')
            except Exception as e:
                self.ddp_log(f'Failed to load full checkpoint ({e}), fallback to BPNet-only')
                net_state_dict = {k: v for k, v in state.items() if not k.startswith('nddepth_model.')}
                model.load_state_dict(net_state_dict, strict=True)

            self.best_metric_ema = cp.get('best_metric_ema', self.best_metric_ema)
            del cp

        if (not nd_in_ckpt) and ('nddepth_chpt' in self.cfg):
            # print(f"Loading NDDepth checkpoint: {self.cfg.nddepth_chpt}")
            nd_ckpt_path = os.path.join('models/nd_weights', self.cfg.nddepth_chpt)
            nddepth_model = self.load_nddepth_model(
                checkpoint_path=nd_ckpt_path,
                encoder='large07',
                max_depth=80,
                device=f'cuda:{self.cfg.gpu_id}',
                strict=True,
            )
            nddepth_model = nddepth_model.to(f'cuda:{self.cfg.gpu_id}')
            model.set_nddepth_model(nddepth_model, trainable=False)
        elif not nd_in_ckpt:
            model.set_nddepth_model(None)

        model = model.to(f'cuda:{self.cfg.gpu_id}')
        if self.ddp and OmegaConf.select(self.cfg.net, 'sbn', default=False):
            self.ddp_log('sbn')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            """
            SBN is not compatible with torch.compile
            """
            if OmegaConf.select(self.cfg.net, 'compile', default=False):
                self.ddp_log('compile')
                model = torch.compile(model)
        return model
    
    @staticmethod
    def load_nddepth_model(checkpoint_path, encoder='large07', max_depth=80, device='cuda', strict=True):
        model = NewCRFDepth(version=encoder, inv_depth=False, max_depth=max_depth)
        checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
        state_dict = checkpoint['model']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=strict)
        model = model.to(device)
        model.eval()
        return model

    def init_ema(self):
        ema = instantiate(self.cfg.net.ema, model=self.net, ddp=self.ddp)
        try:
            device = next(self.net.parameters()).device
            ema.module.to(device)
        except Exception as e:
            self.ddp_log(f'Warning: failed to move EMA to device: {e}')
        return ema

    def init_loss(self):
        return instantiate(self.cfg.loss)

    def init_metric(self):
        return instantiate(self.cfg.metric)

    def init_cuda(self, *modules):
        modules = [module.to(f'cuda:{self.cfg.gpu_id}') for module in modules]
        return modules

    def init_gpu(self):
        with torch.cuda.device(f'cuda:{self.cfg.gpu_id}'):
            torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    def init_optim(self):
        optim = instantiate(self.cfg.optim, _partial_=True)
        return optim(params=config_param(self.net))

    def init_clip(self):
        if 'clip' in self.cfg.net:
            return instantiate(self.cfg.net.clip, _partial_=True)
        else:
            return None

    def init_sched_lr(self):
        sched = instantiate(self.cfg.sched.lr.policy, _partial_=True)
        return sched(optimizer=self.optimizer)


    def ddp_log(self, content, always=False):
        # self.log.info(f'{content}')
        if (not self.rank) or always:
            self.log.info(f'{content}')

    def ddp_cout(self, content, always=False):
        # tqdm.write(f'{content}')
        if (not self.rank) or always:
            tqdm.write(f'{content}')

    # 実機実験のためのモデル一括保存 -> 単眼深度推定モデルのcheckpoint不要
    def save_state(self, save_path=None):
        if self.rank:
            return
        if save_path is None:
            save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        model = self.net_ema.module
        if hasattr(model, 'module'):
            model = model.module
        model_state_dict = model.state_dict()
        state_dict = {
            'net': model_state_dict,
            'epoch': self.epoch,
            'best_metric_ema': self.best_metric_ema,
        }
        torch.save(state_dict, os.path.join(save_path, 'result_ema.pth'))
        try:
            model.eval()
            model.cpu()
            torch.save(model, os.path.join(save_path, 'model.pth'), pickle_module=dill)
            self.ddp_log(f'Model saved to {os.path.join(save_path, "model.pth")}')
            model.to(f'cuda:{self.cfg.gpu_id}')
        except Exception as e:
            self.ddp_log(f'Failed to save model: {e}')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.writer.close()
        self.ddp_log(f'best_metric_ema={self.best_metric_ema:.4f}')


def config_param(model):
    param_groups = []
    other_params = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            g = {'params': [param], 'weight_decay': 0.0}
            param_groups.append(g)
        else:
            other_params.append(param)
    param_groups.append({'params': other_params})
    return param_groups


def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad



class Blank(object):
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            return None
        return wrapper

FloorDiv = lambda a, b: a // b

CeilDiv = lambda a, b: math.ceil(a / b)

Div = lambda a, b: a / b

Mul = lambda a, b: a * b


