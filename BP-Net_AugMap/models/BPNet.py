# -*- coding: utf-8 -*-
# @File : BPNet.py
# @Project: BP-Net
# @Author : jie
# @Time : 4/8/23 12:43 PM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import Conv1x1, Basic2d, BasicBlock, weights_init, inplace_relu, GenKernel, PMP, AdapnetSkipLayer, TransposeWeightLayer, WeightLayer

# NDDepth
from models.nddepth.utils import post_process_depth, flip_lr, compute_errors, inv_normalize

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    'Pre_MF_Post',
    'Segmentation_Decoder',
]


class Net(nn.Module):
    """
    BPNet with integrated NDDepth as internal module
    """

    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2, 2],
                 drop_path=0.1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', 
                 drift=1e6, num_classes=5, nddepth_model=None, nddepth_trainable=False):
        super().__init__()
        self.drift = drift
        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, sum(img_layers))

        self.nddepth_model = None
        self.nddepth_trainable = nddepth_trainable
        self._cached_inv_K = None
        self._cached_norm_params = None
        self._last_device = None
        self._last_dtype = None

        self.inplanes = bc * 2
        self.conv_img = nn.Sequential(
            Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1),
            self._make_layer(block, bc * 2, 2, stride=1)
        )
        
        self.seg_decoder = Segmentation_Decoder(num_classes=num_classes)

        self.layer1_img = self._make_layer(block, bc * 4, img_layers[1], stride=2)

        self.inplanes = bc * 4
        self.layer2_img = self._make_layer(block, bc * 8, img_layers[2], stride=2)

        self.inplanes = bc * 8
        self.layer3_img = self._make_layer(block, bc * 16, img_layers[3], stride=2)

        self.inplanes = bc * 16
        self.layer4_img = self._make_layer(block, bc * 16, img_layers[4], stride=2)

        self.inplanes = bc * 16
        self.layer5_img = self._make_layer(block, bc * 16, img_layers[5], stride=2)

        self.pred0 = PMP(level=0, in_ch=bc * 4, out_ch=bc * 2, drop_path=drop_path, pool=False)
        self.pred1 = PMP(level=1, in_ch=bc * 8, out_ch=bc * 4, drop_path=drop_path)
        self.pred2 = PMP(level=2, in_ch=bc * 16, out_ch=bc * 8, drop_path=drop_path)
        self.pred3 = PMP(level=3, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred4 = PMP(level=4, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path)
        self.pred5 = PMP(level=5, in_ch=bc * 16, out_ch=bc * 16, drop_path=drop_path, up=False)

    def set_nddepth_model(self, nddepth_model, trainable=False):
        if nddepth_model is not None:
            # print("Integrating NDDepth model into BPNet...")
            device = next(self.parameters()).device
            self.nddepth_model = nddepth_model.to(device)
            self.nddepth_trainable = trainable
            
            if trainable:
                # print("NDDepth model set to trainable.")
                self.nddepth_model.train()
                for param in self.nddepth_model.parameters():
                    param.requires_grad = True
            else:
                # print("NDDepth model set to fixed (not trainable).")
                self.nddepth_model.eval()
                for param in self.nddepth_model.parameters():
                    param.requires_grad = False
        else:
            # print("No NDDepth model provided; proceeding without it.")
            self.nddepth_model = None
            self.nddepth_trainable = False
    
    def _get_cached_inv_K(self, device, dtype, batch_size):
        if (self._cached_inv_K is None or 
            self._last_device != device or 
            self._last_dtype != dtype or
            self._cached_inv_K.shape[0] != batch_size):
            
            self._cached_inv_K = torch.tensor([
                [7.2391e-10, 7.2391e-03, -1.0606e+00],
                [-1.2930e-10, 7.2391e-03, -3.8869e-01],
                [-4.3221e-11, 3.8488e-11, 1.0000e+00]
            ], device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            self._last_device = device
            self._last_dtype = dtype
        
        return self._cached_inv_K

    def _get_cached_norm_params(self, device, dtype):
        if (self._cached_norm_params is None or 
            self._last_device != device or 
            self._last_dtype != dtype):
            
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
            self._cached_norm_params = (mean, std)
            self._last_device = device
            self._last_dtype = dtype
        
        return self._cached_norm_params

    def forward(self, I, S, K, I_ND, L=None, epoch=None, use_nddepth=True):
        """
        I: Bx3xHxW (RGB image)
        S: Bx1xHxW (LiDAR sparse depth)
        K: Bx3x3 (Camera intrinsics)
        I_ND: Bx3xHxW (RGB image for NDDepth, optional)
        L: Bx1xHxW (ground truth label, optional)
        epoch: 学習用スケジューリング
        use_nddepth: NDDepthを使用するか
        """
        output = []
        pred_segmentation = []

        # BPNetエンコーダ部分
        XI0 = self.conv_img(I)  # Bx32x256x1216
        XI1 = self.layer1_img(XI0) # Bx64x128x608
        XI2 = self.layer2_img(XI1) # Bx128x64x304
        XI3 = self.layer3_img(XI2) # Bx256x32x152
        XI4 = self.layer4_img(XI3) # Bx256x16x76
        XI5 = self.layer5_img(XI4) # Bx256x8x38

        # セグメンテーション部分
        seg_out, aux1, aux2 = self.seg_decoder(XI5, XI3, XI2)
        pred_segmentation.append(seg_out)
        pred_segmentation.append(aux1)
        pred_segmentation.append(aux2)

        # 動的マスクの計算
        if L is not None:
            # 学習時/評価時: Ground Truthを使用してカリキュラム学習
            L = L.squeeze(1)  # BxHxW
            pred_label = torch.argmax(seg_out, dim=1)  # BxHxW
            
            dynamic_class_ids = torch.tensor([3, 4], device=L.device)
            mask_gt = torch.isin(L, dynamic_class_ids)
            mask_pred = torch.isin(pred_label, dynamic_class_ids)

            total_epoch = 50
            warmup_epoch = 30
            if epoch is not None:
                alpha = 1.0 - min(max(epoch - warmup_epoch, 0) / (total_epoch - warmup_epoch), 1.0)
            else:
                alpha = 0.0

            rand_mask = torch.rand_like(pred_label, dtype=torch.float)
            mask_dynamic = torch.where(rand_mask < alpha, mask_gt, mask_pred)
        else:
            # 推論時: 予測結果のみを使用
            pred_label = torch.argmax(seg_out, dim=1)  # BxHxW
            dynamic_class_ids = torch.tensor([3, 4], device=pred_label.device)
            mask_dynamic = torch.isin(pred_label, dynamic_class_ids)

        # NDDepthによる動的深度推定
        dynamic_depth = None
        if use_nddepth and self.nddepth_model is not None and I_ND is not None:
            dynamic_depth = self._internal_nddepth_inference(I_ND, epoch=5)
            
            # マスクを適用して動的オブジェクトの深度のみ抽出
            mask_dynamic_expanded = mask_dynamic.unsqueeze(1)  # Bx1xHxW
            masked_dynamic_depth = torch.zeros_like(dynamic_depth)
            masked_dynamic_depth[mask_dynamic_expanded] = dynamic_depth[mask_dynamic_expanded]
            dynamic_depth = masked_dynamic_depth

        # PMP階層による深度推定
        fout, dout = self.pred5(fout=None, dout=None, XI=XI5, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(F.interpolate(dout, scale_factor=2 ** 5, mode='bilinear', align_corners=True))

        fout, dout = self.pred4(fout=fout, dout=dout, XI=XI4, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(F.interpolate(dout, scale_factor=2 ** 4, mode='bilinear', align_corners=True))

        fout, dout = self.pred3(fout=fout, dout=dout, XI=XI3, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(F.interpolate(dout, scale_factor=2 ** 3, mode='bilinear', align_corners=True))

        fout, dout = self.pred2(fout=fout, dout=dout, XI=XI2, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(F.interpolate(dout, scale_factor=2 ** 2, mode='bilinear', align_corners=True))

        fout, dout = self.pred1(fout=fout, dout=dout, XI=XI1, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(F.interpolate(dout, scale_factor=2 ** 1, mode='bilinear', align_corners=True))

        fout, dout = self.pred0(fout=fout, dout=dout, XI=XI0, S=S, K=K, dynamic_depth=dynamic_depth)
        output.append(dout)

        return output, pred_segmentation

    def _internal_nddepth_inference(self, image, epoch=5):
        grad_enabled = self.training and self.nddepth_trainable
        with torch.set_grad_enabled(grad_enabled):
            if image.max() > 1.0:
                image = image / 255.0
            mean, std = self._get_cached_norm_params(image.device, image.dtype)
            image = (image - mean) / std
            batch_size = image.shape[0]
            inv_K = self._get_cached_inv_K(image.device, image.dtype, batch_size)

            try:
                depth1_list, uncer1_list, depth2_list, uncer2_list, _, _ = self.nddepth_model(image, inv_K, epoch)
                pred_depth = (depth1_list[-1] + depth2_list[-1]) * 0.5
                image_flipped = flip_lr(image)
                depth1_list_flipped, _, depth2_list_flipped, _, _, _ = self.nddepth_model(image_flipped, inv_K, epoch)
                pred_depth_flipped = (depth1_list_flipped[-1] + depth2_list_flipped[-1]) * 0.5
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
            
                pred_depth = pred_depth[:, :, 96:352, :]
                pred_depth = torch.clamp(pred_depth * (255.0 / 80.0), 0, 255)
                
                return pred_depth
                
            except Exception as e:
                print(f"NDDepth inference failed: {e}")
                return torch.zeros(
                    (batch_size, 1, 256, image.shape[3]), 
                    device=image.device, 
                    dtype=image.dtype
                )

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)
    
def Pre_MF_Post(num_classes=5, nddepth_trainable=False):
    """
    Pre.+MF.+Post. (NDDepthは後からutils.pyで設定される)
    """
    net = Net(num_classes=num_classes, nddepth_trainable=nddepth_trainable)
    net.apply(functools.partial(weights_init, mode='trunc'))
    for m in net.modules():
        if isinstance(m, GenKernel) and m.conv[1].conv.bn.weight is not None:
            nn.init.constant_(m.conv[1].conv.bn.weight, 0)
    net.apply(inplace_relu)
    return net


def save_image(img, path, cmap='viridis'):
    """
    画像保存用グローバル関数
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 2:
        plt.imsave(path, img, cmap=cmap)
    elif img.ndim == 3 and img.shape[2] == 3:
        plt.imsave(path, img)
    else:
        raise ValueError("画像のshapeが不正です")


class Segmentation_Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_aux: bool = True,
        bias: bool = True,
        padding_mode: str = 'zeros',
        inplace: bool = True
    ):
        super().__init__()
        in_channels, in_height, in_width = 256, 8, 38  # XI5: Bx256x8x38
        out_height, out_width = in_height*2**5, in_width*2**5  # 最終アップサンプル後のサイズ
        skip_channels = 24  # スキップ接続のチャンネル数
        out_channels = num_classes  # 出力のクラス数
        d_channels = 256
        self.use_aux = use_aux

        # Block 1 (XI5)
        self.b1_1_tw = TransposeWeightLayer(
            in_channels, out_channels=d_channels, kernel_size=4, stride=2, padding=1, bias=bias
        )
        self.b1_2_tw = TransposeWeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=4, stride=2, padding=1, bias=bias
        )

        # Aux 1
        if self.use_aux is True:
            self.a1_1_wl: nn.Module = WeightLayer(
                in_channels=d_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace)
            self.a1_2_us: nn.Module = nn.Upsample(
                size=(out_height, out_width), mode='bilinear', align_corners=True)

        # Skip 1 (XI3)
        self.s1_1_sl = AdapnetSkipLayer(
            in_channels=256, out_channels=skip_channels, bias=bias, padding_mode=padding_mode
        )

        # Block 2
        self.b2_1_wl = WeightLayer(
            in_channels=d_channels + skip_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b2_2_wl = WeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b2_3_tw = TransposeWeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=4, stride=2, padding=1, bias=bias
        )

        # Aux 2
        if self.use_aux is True:
            self.a2_1_wl: nn.Module = WeightLayer(
                in_channels=d_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace)
            self.a2_2_us: nn.Module = nn.Upsample(
                size=(out_height, out_width), mode='bilinear', align_corners=True)

        # Skip 2 (XI2)
        self.s2_1_sl = AdapnetSkipLayer(
            in_channels=int(d_channels / 2), out_channels=skip_channels, bias=bias, padding_mode=padding_mode
        )

        # Block 3
        self.b3_1_wl = WeightLayer(
            in_channels=d_channels + skip_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_2_wl = WeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_3_wl = WeightLayer(
            in_channels=d_channels, out_channels=num_classes, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_4_tw = TransposeWeightLayer(
            in_channels=num_classes, out_channels=num_classes, kernel_size=8, stride=4, padding=2, bias=bias
        )

    def forward(self, XI5, XI3, XI2):
        # XI5: Bx256x8x38, XI3: Bx256x32x152, XI2: Bx128x64x304
        tx1 = self.b1_1_tw(XI5)         # Bx256x16x76
        tx1 = self.b1_2_tw(tx1)         # Bx256x32x152
        sk1 = self.s1_1_sl(XI3)         # Bx24x32x152
        tx2 = torch.cat((tx1, sk1), dim=1)  # Bx280x32x152

        tx3 = self.b2_1_wl(tx2)         # Bx256x32x152
        tx4 = self.b2_2_wl(tx3)         # Bx256x32x152
        tx5 = self.b2_3_tw(tx4)         # Bx256x64x304
        sk2 = self.s2_1_sl(XI2)         # Bx24x64x304
        tx6 = torch.cat((tx5, sk2), dim=1)  # Bx280x64x304

        tx7 = self.b3_1_wl(tx6)         # Bx256x64x304
        tx8 = self.b3_2_wl(tx7)         # Bx256x64x304
        tx9 = self.b3_3_wl(tx8)         # Bx(num_classes)x64x304
        tx10 = self.b3_4_tw(tx9)        # Bx(num_classes)x256x1216

        if self.use_aux:
            a1x1 = self.a1_1_wl(tx1)
            a1x2 = self.a1_2_us(a1x1)
            a2x1 = self.a2_1_wl(tx5)
            a2x2 = self.a2_2_us(a2x1)
            return tx10, a1x2, a2x2
        else:
            return tx10, None, None
