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

FILE_TYPE = 'tiff'
EPS = 1e-6

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

# 加载训练集
class CustomDataset(Dataset):
    def __init__(self, root_dir, data_type = 'train'):
        self.root_dir = root_dir  # str类型
        self.data_type = data_type
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' not in f and 'mask' not in f]
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' in f]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_type, self.file_list[idx])
        img_file_name = os.path.splitext(self.file_list[idx])[0]
        # gt_image_path = os.path.join(self.root_dir, self.data_type, self.gt_image_list[idx])
        # gt_image_file_name = os.path.splitext(self.gt_image_list[idx])[0]
        gt_path = os.path.join(self.root_dir + '/illu_map', img_file_name + '.npy')

        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        # gt_image = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        gt = np.load(gt_path)

        gt = torch.tensor(gt)
        gt = gt.permute(2, 0, 1)  # 转成3，255，255

        # image = self.transform(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # 转成3，255，255
        image = image / (image.max() + EPS)

        # gt_image = self.transform(gt_image)
        # gt_image = torch.from_numpy(gt_image)
        # gt_image = gt_image.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image,
                # "gt_image": gt_image,
                "gt": gt,
                "image_name": img_file_name}


# 加载验证集
class CustomValDataset(Dataset):
    def __init__(self, root_dir, data_type = 'val'):
        self.root_dir = root_dir
        self.data_type = data_type
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' not in f and 'mask' not in f]
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' in f]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_type, self.file_list[idx])
        img_file_name = os.path.splitext(self.file_list[idx])[0]
        # gt_image_path = os.path.join(self.root_dir, self.data_type, self.gt_image_list[idx])
        # gt_image_file_name = os.path.splitext(self.gt_image_list[idx])[0]
        # mask_path = os.path.join(self.root_dir, img_file_name.split("_")[0] + "_mask.png")
        gt_path = os.path.join(self.root_dir + '/illu_map', img_file_name + '.npy')

        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        # gt_image = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        gt = np.load(gt_path)
        gt = torch.tensor(gt)  # （256，256，3）
        gt = gt.permute(2, 0, 1)

        # image = self.transform(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # 转成3，255，255
        image = image / (image.max() + EPS)

        # gt_image = self.transform(gt_image)
        # gt_image = torch.from_numpy(gt_image)
        # gt_image = gt_image.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image,
                "gt": gt,
                # "gt_image": gt_image
                }


# 加载测试集
class CustomTestDataset(Dataset):
    def __init__(self, root_dir, data_type = 'test'):
        self.root_dir = root_dir
        self.data_type = data_type
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' not in f and 'mask' not in f]
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(FILE_TYPE) and 'gt' in f]  # 原色文件
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_type, self.file_list[idx])
        img_file_name = os.path.splitext(self.file_list[idx])[0]
        # gt_image_path = os.path.join(self.root_dir, self.data_type, self.gt_image_list[idx])
        # gt_image_file_name = os.path.splitext(self.gt_image_list[idx])[0]
        # mask_path = os.path.join(self.root_dir, img_file_name.split("_")[0] + "_mask.png")
        gt_path = os.path.join(self.root_dir + '/illu_map', img_file_name + '.npy')

        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        # gt_image = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
        gt = np.load(gt_path)
        gt = torch.tensor(gt)  # （256，256，3）
        gt = gt.permute(2, 0, 1)

        # image = self.transform(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # 转成3，255，255
        image = image / (image.max() + EPS)

        # gt_image = self.transform(gt_image)
        # gt_image = torch.from_numpy(gt_image)
        # gt_image = gt_image.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image,
                # "gt_image": gt_image,
                "gt": gt}
