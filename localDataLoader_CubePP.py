import colorsys
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import *
from tool import *
import torchvision.transforms.v2 as v2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------
FILE_TYPE = 'tiff'
ESP = 1e-6
# -------------------------------------

class RandomColor:
    def __init__(self, sat_min, sat_max, val_min, val_max, hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self, h, s, v):  # 返回rgb元组
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def threshold_test(self, hue_list, hue):  # 列表里放的是已经随机生成的h，新生成的h要和列表中的和进行比较，若新生成的h与已生成的h过于接近，则返回false
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 光源1，2，3的rgb向量
        for i in illum_count:  # 随机生成几种光源,字符串也可以循环
            while True:
                hue = np.random.uniform(0, 1)  # 随机生成一个数字作为h
                saturation = np.random.uniform(self.sat_min, self.sat_max)  # 随机生成一个数字作为s
                value = np.random.uniform(self.val_min, self.val_max)  # 随机生成一个数字作为v
                chroma_rgb = np.array(self.hsv2rgb(hue, saturation, value), dtype='float32')  # 将随机的hsv转为rgb
                chroma_rgb /= chroma_rgb[1]  # 对绿通道归一化

                if self.threshold_test(hue_list, hue):  # hue_list为空，hue_list与hue不过分接近为true，若hue_list不为空且hue_list里的值与hue过分接近则不执行if里的代码，意味着需要重新生成一种颜色向量。
                    hue_list.append(hue)
                    ret_chroma[int(i) - 1] = chroma_rgb
                    break  # 跳出while，处理完第i个光源

        return np.array(ret_chroma)  # 返回新的光源色度矩阵

class CubePPDataset(Dataset):
    def __init__(self, root_dir, file_type = 'None'):
        self.root_dir = root_dir
        self.file_type = file_type
        self.file_list = [f for f in os.listdir(self.root_dir + self.file_type) if f.endswith(FILE_TYPE) and 'gt' not in f and 'mask' not in f]
        self.file_list.sort()
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.file_type) if f.endswith(FILE_TYPE) and 'gt' in f]

        self.transform = transforms.ToTensor()
        self.ESP = 1e-6

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_type, self.file_list[idx])  # 偏色图片路径
        img_file_name = os.path.splitext(self.file_list[idx])[0]

        # gt_image_path = os.path.join(self.root_dir, self.file_type, self.gt_image_list[idx])  # 正常图片路径
        # gt_image_file_name = os.path.splitext(self.gt_image_list[idx])[0]

        gt_path = os.path.join(self.root_dir + '/illu_map', img_file_name + '.npy')  # 光照图文件路径

        # 处理输入图像
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        h, w = image.shape[:2]
        # 自动判断裁剪方向
        if h < w:
            crop_x = (w - h) // 2
            cropped_image = image[:, crop_x:crop_x + h, :]
        else:
            crop_y = (h - w) // 2
            cropped_image = image[crop_y:crop_y + w, :, :]
        # 缩放到256x256
        resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_LINEAR).astype('uint16')
        image_out = torch.from_numpy(resized_image).permute(2, 0, 1).float()
        image_out = image_out / (image_out.max() + self.ESP)

        # # 处理GT图像
        # gt_image = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        # h_gt, w_gt = gt_image.shape[:2]
        # # 自动判断裁剪方向
        # if h_gt < w_gt:
        #     crop_x_gt = (w_gt - h_gt) // 2
        #     cropped_gt_image = gt_image[:, crop_x_gt:crop_x_gt + h_gt, :]
        # else:
        #     crop_y_gt = (h_gt - w_gt) // 2
        #     cropped_gt_image = gt_image[crop_y_gt:crop_y_gt + w_gt, :, :]
        # # 缩放到256x256
        # resized_gt_image = cv2.resize(cropped_gt_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        # gt_image_out = torch.from_numpy(resized_gt_image).permute(2, 0, 1).float()
        # gt_image_out = gt_image_out / (gt_image_out.max() + self.EPS)

        # 处理光照图（保持原始处理方式）
        gt = np.load(gt_path)
        gt = torch.tensor(gt)
        gt = gt.permute(2, 0, 1)  # 转换维度为(3, H, W)
        stop = 0

        return {
            "image": image_out,
            # "input_gt_image": gt_image_out,
            "gt": gt,
            "image_name": img_file_name
        }

