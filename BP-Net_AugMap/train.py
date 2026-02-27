# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
import os
from tqdm import tqdm
import hydra
import torch.distributed as dist
from utils import *
from models.self_metric import *
import warnings
from models.self_metric import *
import openpyxl as xl
from loss_weight import LOSS_WEIGHT


"""
datas ->  0:rgb, 1:in_map, 2:K_cam, 3:depth, 4:gt_label, 5:rgb_nd, 6:rgb_hdf5
"""
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def train(run):
    lossCE: LOSS_WEIGHT = LOSS_WEIGHT(nn.CrossEntropyLoss(), 0.5)
    for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                test(run, iter=True)
        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)
        output, pred_segmentation = run.net(datas[0],datas[1],datas[2], datas[5], datas[4], epoch = run.epoch)  # 0:rgb, 1:in_map, 2:K_cam, 3:depth, 4:gt_label, 5:rgb_nd, 6:rgb_hdf5
        loss = run.criterion(output, datas[3])  # 3:depth

        gt_label = datas[4].squeeze(1).long()
        for pred_seg in pred_segmentation:
            loss_seg_ce = lossCE(pred_seg, gt_label)
            loss.append(loss_seg_ce)

        sum(loss).backward()
        if run.clip:
            grad_norm = run.clip(run.net.parameters())
        run.optimizer.step()
        run.net_ema.update(run.net)
        if run.lr_iter:
            run.lr_scheduler.step()
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            run.writer.add_scalars("Loss", {f"{idx}": l.item() for idx, l in enumerate(loss)}, run.iter)
            if run.clip and (grad_norm is not None):
                run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)
        run.iter += 1
    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()

def test(run, iter=False):
    metricSumAE: nn.Module = MetricSumAE()
    metricSumSE: nn.Module = MetricSumSE()
    metricSumAPE: nn.Module = MetricSumAPE()
    metricSumIntersection: nn.Module = MetricSumIntersection()
    metricSumUnion: nn.Module = MetricSumUnion()
    
    top1 = AverageMeter()
    net = run.net_ema.module
    best_metric_name = "best_metric_ema"
    legand = 'net_ema'
    net.eval()
    with torch.no_grad():
        sum_intersection: float = 0.0
        sum_union: float = 0.0
        sum_ae = 0.0
        sum_se = 0.0
        sum_ape = 0.0
        count_in_range = 0.0
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            datas = run.init_cuda(*datas)
            output, pred_segmentation= net(datas[0],datas[1],datas[2], datas[5], datas[4] ,epoch = run.epoch)  # 0:rgb, 1:in_map, 2:K_cam, 3:depth, 4:gt_label, 5:rgb_nd, 6:rgb_hdf5
            if isinstance(output, (list, tuple)):
                output = output[-1]
            if isinstance(pred_segmentation, (list, tuple)):
                pred_segmentation = pred_segmentation[0]
            gt_depth = datas[3] /255.0 # 3:depth
            gt_label = datas[4]
            pred_depth = output /255.0

            ae: METRIC_SUM_OUTPUT = metricSumAE(pred_depth, gt_depth)
            sum_ae += ae.sum.sum().item()
            se: METRIC_SUM_OUTPUT = metricSumSE(pred_depth, gt_depth)
            sum_se += se.sum.sum().item()
            ape: METRIC_SUM_OUTPUT = metricSumAPE(pred_depth, gt_depth)
            sum_ape += ape.sum.sum().item()
            count_in_range += ape.count.sum().item()
            intersection: Tensor = metricSumIntersection(pred_segmentation, gt_label)
            sum_intersection += intersection.cpu().detach().numpy()

            union: Tensor = metricSumUnion(pred_segmentation, gt_label)
            sum_union += union.cpu().detach().numpy()

            sum_ae_tensor = torch.tensor(sum_ae, device='cuda')
            sum_se_tensor = torch.tensor(sum_se, device='cuda')
            sum_ape_tensor = torch.tensor(sum_ape, device='cuda')
            count_in_range_tensor = torch.tensor(count_in_range, device='cuda')
            sum_intersection_tensor = torch.tensor(sum_intersection, device='cuda')
            sum_union_tensor = torch.tensor(sum_union, device='cuda')

            if run.ddp:
                torch.distributed.all_reduce(sum_ae_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sum_se_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sum_ape_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count_in_range_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sum_intersection_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sum_union_tensor, op=torch.distributed.ReduceOp.SUM)


            prec1 = run.metric(output, datas[3])  # 3:depth
            if isinstance(prec1, (list, tuple)):
                prec1 = prec1[0]
            if run.ddp:
                dist.reduce(prec1, 0, dist.ReduceOp.AVG)
            top1.update(prec1.item())
            
    if run.rank == 0:
        tmp_mae: float = sum_ae_tensor.item() / count_in_range_tensor.item() * 80.0
        tmp_rmse: float = np.sqrt(sum_se_tensor.item() / count_in_range_tensor.item()) * 80.0
        tmp_mape: float = sum_ape_tensor.item() / count_in_range_tensor.item()
        tmp_miou: float = np.mean(sum_intersection_tensor.cpu().numpy() / sum_union_tensor.cpu().numpy())

        val_metrics: Dict[str, float] = {}
        val_metrics['mIoU'] = tmp_miou * 100.0
        val_metrics['MAE'] = tmp_mae
        val_metrics['RMSE'] = tmp_rmse
        val_metrics['MAPE'] = tmp_mape

    if iter:
        run.writer.add_scalars("RMSE_Iter", {legand: top1.avg}, run.iter)
    else:
        run.writer.add_scalars("RMSE", {legand: top1.avg}, run.epoch)
    if top1.avg < getattr(run, best_metric_name):
        setattr(run, best_metric_name, top1.avg)
        run.save_state()
        run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')

    save_dir: str = run.save_path
    val_book_path: str = os.path.join(save_dir, 'validation.xlsx')
    if run.rank == 0:
        if run.epoch == 0:
            workbook = xl.Workbook()
            worksheet = workbook.worksheets[0]
            worksheet.title = 'validation'
            worksheet.cell(row=1, column=1, value='Epoch')
            for col, title in enumerate(val_metrics.keys(), 2):
                worksheet.cell(row=1, column=col, value=title)
            worksheet.cell(row=2, column=1, value=1)
            for col, value in enumerate(val_metrics.values(), 2):
                worksheet.cell(row=2, column=col, value=value)
            workbook.save(val_book_path)
        else:
            workbook = xl.load_workbook(val_book_path)
            worksheet = workbook.worksheets[0]
            worksheet.cell(row=run.epoch+2, column=1, value=run.epoch+1)
            for col, value in enumerate(val_metrics.values(), 2):
                worksheet.cell(row=run.epoch+2, column=col, value=value)
            workbook.save(val_book_path)

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)
            train(run)
            torch.cuda.synchronize()
            test(run)
            torch.cuda.synchronize()


if __name__ == '__main__':
    main()
