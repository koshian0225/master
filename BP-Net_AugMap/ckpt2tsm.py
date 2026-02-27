import argparse
from typing import Dict
import os

import torch
from torch import Tensor
import torch.nn as nn
import dill


ARG_HEIGHT_ND = 'width_nd'
ARG_CHECK_POINT = 'check_point'
ARG_WIDTH = 'width'
ARG_HEIGHT = 'height'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', f'--{(ARG_CHECK_POINT)}', type=str, metavar='PATH',
        help='Path of checkpoint file.'
    )
    parser.add_argument(
        '-x', f'--{(ARG_WIDTH)}', type=int, default=1216,
        help='Width of input images.'
    )
    parser.add_argument(
        '-nd', f'--{(ARG_HEIGHT_ND)}', type=int, default=352,
        help='Width of input images.'
    )
    parser.add_argument(
        '-y', f'--{(ARG_HEIGHT)}', type=int, default=256,
        help='Height of input images.'
    )
    return vars(parser.parse_args())


def main(args: Dict[str, str]):
    netG: nn.Module = torch.load(args[ARG_CHECK_POINT], pickle_module=dill)
    netG.eval()
    import models.utils as utils
    # 推論用のスクリプト可能な代替関数を用意して utils.BpDist を置き換える
    def _bpdist_infer(xy, idx, Valid, num, H: int, W: int):
        # forward 相当の処理（autograd 不要・単純化）
        # ensure sizes are plain Python ints for TorchScript
        B = int(Valid.size(0))
        Cc = int(xy.size(1))
        num_i = int(num)
        N = int(H * W)
        args = torch.zeros([B, num_i, N], dtype=torch.long, device=xy.device)
        IPCnum = torch.zeros([B, Cc, num_i, N], dtype=xy.dtype, device=xy.device)
        for b in range(B):
            Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1)
            utils.BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N))
            idx_inds = args[b:b + 1].reshape(-1)
            if idx_valid.numel() == 0:
                 args[b:b + 1] = torch.zeros_like(args[b:b + 1])
            else:

                max_i = int(idx_valid.size(0) - 1)
                idx_safe = idx_inds.clamp(0, max_i).long()
                args[b:b + 1] = torch.index_select(idx_valid, 0, idx_safe).reshape(1, num_i, N)
        return IPCnum, args

    try:
        # Do not script (BpOps.Dist is a native extension). Replace with plain Python
        # inference function so tracing executes it directly.
        utils.BpDist = _bpdist_infer
        print("BpDist replaced with pure-Python inference function (for tracing)")
    except Exception:
        # keep original if replacement somehow fails
        print("BpDist replacement failed; keeping original")
    netG.cpu()

    # , device=torch.device('cuda'))
    # sample_rgb: Tensor = torch.empty(1, 3, args[ARG_HEIGHT], args[ARG_WIDTH])
    # # , device=torch.device('cuda'))
    # sample_depth: Tensor = torch.empty(1, 1, args[ARG_HEIGHT], args[ARG_WIDTH])
    # sample_K: Tensor = torch.empty(1, 3, 3)
    # sample_rgb_nd: Tensor = torch.empty(1, 3, args[ARG_HEIGHT_ND], args[ARG_WIDTH])
    sample_rgb = torch.zeros(1, 3, 256, 1216)  # 学習時サイズに合わせる
    sample_depth = torch.zeros(1, 1, 256, 1216)
    sample_K = torch.eye(3).unsqueeze(0)
    sample_rgb_nd = torch.zeros(1, 3, 352, 1216)


    traced_ts = torch.jit.trace(netG, (sample_rgb, sample_depth, sample_K, sample_rgb_nd), strict=False)
    out_path = f'{os.path.splitext(args[ARG_CHECK_POINT])[0]}.pt'
    traced_ts.save(out_path)
    print(f'Saved TorchScript: {out_path}')

if __name__ == '__main__':
    args = parse_args()
    if not args.get(ARG_CHECK_POINT):
        raise SystemExit('Error: -c/--check_point を指定してください。')
    if not os.path.exists(args[ARG_CHECK_POINT]):
        raise SystemExit(f'Error: ファイルが見つかりません: {args[ARG_CHECK_POINT]}')
    main(args)