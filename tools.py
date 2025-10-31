import torch
import numpy as np
import random
import os
import cv2
import math
from torchvision.utils import make_grid, save_image
from PIL import Image

eps = 1e-7
# 设定整体程序的随机数种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # 避免选择卷积算法，会降低运算性能
    torch.backends.cudnn.deterministic = True # 保证所选用的卷积算法是非随机的

# 创建文件夹
def make_dir(path):
    os.makedirs(path,exist_ok=True)

# 展示一张16位PNG图像
def show_16png(img_path):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32)
    img_nor = img/np.max(img)
    img_nor_8bit = (img_nor*255).astype(np.uint8)
    cv2.imshow("example image", img_nor_8bit)
    cv2.waitKey(0) # 等待按键

# MSE误差
def MSE(pred, gt):
    return np.mean((pred - gt) ** 2, axis = 1)

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

# 计算统计值
def errors_statistics(errors):
    mean = np.mean(errors, axis=0) # mean
    q1 = np.percentile(errors, 25, axis=0) # q1
    median = np.percentile(errors, 50, axis=0) # q2(median)
    q3 = np.percentile(errors, 75, axis=0) # q3
    trimean = (q1 + 2 * median + q3) / 4
    return [mean, q1, median, q3, trimean]

# 将一个batch的图片拼接成一张图片，并存储下来。
def save_batch(batch, idx):
    images = make_grid(batch, 4, 2)
    save_image(images, f'batch_{idx}_show.png')

def process_fft_difference(tensor1, tensor2):
    """
    计算两个张量的傅里叶变换差异并重建

    参数:
    tensor1: 第一个输入张量
    tensor2: 第二个输入张量

    返回:
    reconstructed: 重建后的差异结果
    fft_shifted: 中心化的频域差异（可选）
    """
    # 正向变换
    fft_result1 = torch.fft.fft2(tensor1.float())
    fft_result2 = torch.fft.fft2(tensor2.float())

    # 计算频域差异
    fft_diff = fft_result1 - fft_result2

    # 零频率移到中心
    fft_shifted = torch.fft.fftshift(fft_diff, dim=(-2, -1))

    # 逆向变换
    fft_unshifted = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
    reconstructed = torch.fft.ifft2(fft_unshifted).real

    return reconstructed, fft_shifted


# if __name__ == '__main__':
#     pass
