import torch
import torch.nn as nn
import numpy as np
import math,random


eps = 1e-10

# 设定程序的随机数种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getMAE_numpy_2D(pred, gt):
    dot_product = (pred * gt).sum(axis=1)  # (65536,3)
    pred_length = np.linalg.norm(pred, axis=1, ord=2)
    gt_length = np.linalg.norm(gt, axis=1, ord=2)
    result = dot_product / ((pred_length * gt_length) + eps)
    clipped_result = np.clip(result, -1, 1)
    radian = np.arccos(clipped_result)
    angle = (radian / math.pi) * 180
    return angle


def getMAE_numpy_3D(pred, gt):
    product = (pred * gt).sum(axis=2)  # (256,256,3)
    pred_length = np.linalg.norm(pred, axis=2, ord=2)
    gt_length = np.linalg.norm(gt, axis=2, ord=2)
    result = product / ((pred_length * gt_length) + eps)
    clipped_result = np.clip(result, -1, 1)
    radian = np.arccos(clipped_result)
    angle = (radian / math.pi) * 180
    return angle


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # x 应该是一个形状为 [batch_size, channels, height, width] 的张量
        # batch_size, channels, height, width = x.size()

        # 计算水平方向的 TV Loss
        horizontal_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
        horizontal_tv_loss = torch.mean(torch.abs(horizontal_diff))

        # 计算垂直方向的 TV Loss
        vertical_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        vertical_tv_loss = torch.mean(torch.abs(vertical_diff))

        # 总 TV Loss 是水平和垂直方向 TV Loss 的和
        total_tv_loss = horizontal_tv_loss + vertical_tv_loss

        return total_tv_loss


def MAE_loss(pred, gt):  # 一批的MAE,而非每张图片的
    dot_product = torch.sum(pred * gt, dim=1)
    pred_length = torch.norm(pred, p=2, dim=1)
    gt_length = torch.norm(gt, p=2, dim=1)
    divisin = torch.clamp(dot_product / (pred_length * gt_length + eps), -1, 1)
    angle = torch.arccos(divisin) / math.pi * 180
    angle_mean = torch.mean(angle, dim=[0, 1, 2])
    return angle_mean


def getMAE_tensor(pred,gt):  # 每张图片的MAE
    dot_product = torch.sum(pred * gt, dim=1)
    pred_length = torch.norm(pred, p=2, dim=1)
    gt_length = torch.norm(gt, p=2, dim=1)
    divisin = torch.clamp(dot_product / (pred_length * gt_length + eps), -1, 1)
    angle = torch.arccos(divisin) / math.pi * 180
    angle_mean = torch.mean(angle, dim=[1, 2])
    return angle_mean


def getMAE_single_tensor(pred,gt):  # 每张图片的MAE
    dot_product = torch.sum(pred * gt, dim=1)
    pred_length = torch.norm(pred, p=2, dim=1)
    gt_length = torch.norm(gt, p=2, dim=1)
    divisin = torch.clamp(dot_product / (pred_length * gt_length + eps), -1, 1)
    angle = torch.arccos(divisin) / math.pi * 180
    # angle_mean = torch.mean(angle, dim=[1, 2])
    return angle


