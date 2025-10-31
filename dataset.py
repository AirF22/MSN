import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import tools
import pandas as pd
import cv2
import yaml

EPS = 1e-7

# 加载训练集
class CustomDataset(Dataset):
    def __init__(self, hyper):
        self.root_dir = os.path.expanduser(hyper['TRAIN']['ROOT_PATH']) # str类型
        self.data_type = 'train'
        self.file_type = 'tiff'
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(self.file_type) and 'gt' not in f and 'mask' not in f]  # 文件名列表 and '123' in f
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(self.file_type) and 'gt' in f]
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

        gt = np.load(gt_path).astype('float32')
        gt = torch.tensor(gt)
        gt = gt.permute(2, 0, 1)  # 转成3，255，255

        # image_out = self.transform(image)
        image_out = torch.from_numpy(image)
        image_out = image_out.permute(2, 0, 1)  # 转成3，255，255
        image_out = image_out / (image_out.max() + EPS)

        # gt_image_out = self.transform(gt_image)
        # gt_image_out = torch.from_numpy(gt_image)
        # gt_image_out = gt_image_out.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image_out,
                # "gt_image": gt_image_out,
                "gt": gt,
                "image_name": img_file_name}


# 加载验证集
class CustomValDataset(Dataset):
    def __init__(self, hyper):
        self.root_dir = os.path.expanduser(hyper['TRAIN']['ROOT_PATH']) # str类型
        self.data_type = 'val'
        self.file_type = 'tiff'
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(self.file_type) and 'gt' not in f and 'mask' not in f]  # 文件名列表
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(file_type) and 'gt' in f]
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
        gt = np.load(gt_path).astype('float32')
        gt = torch.tensor(gt)  # （256，256，3）
        gt = gt.permute(2, 0, 1)

        # image_out = self.transform(image)
        image_out = torch.from_numpy(image)
        image_out = image_out.permute(2, 0, 1)  # 转成3，255，255
        image_out = image_out / (image_out.max() + EPS)

        # gt_image_out = self.transform(gt_image)
        # gt_image_out = torch.from_numpy(gt_image)
        # gt_image_out = gt_image_out.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image_out,
                "gt": gt
                # "gt_image": gt_image_out
                }


# 加载测试集
class CustomTestDataset(Dataset):
    def __init__(self, hyper):
        self.root_dir = os.path.expanduser(hyper['TRAIN']['ROOT_PATH']) # str类型
        self.data_type = 'test'
        self.file_type = 'tiff'
        self.file_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(self.file_type) and 'gt' not in f and 'mask' not in f]  # 偏色文件名列表
        # self.gt_image_list = [f for f in os.listdir(self.root_dir + self.data_type) if f.endswith(self.file_type) and 'gt' in f]  # 原色文件
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
        gt = np.load(gt_path).astype('float32')
        gt = torch.tensor(gt)  # （256，256，3）
        gt = gt.permute(2, 0, 1)

        # image_out = self.transform(image)
        image_out = torch.from_numpy(image)
        image_out = image_out.permute(2, 0, 1)  # 转成3，255，255
        image_out = image_out / (image_out.max() + EPS)

        # gt_image_out = self.transform(gt_image)
        # gt_image_out = torch.from_numpy(gt_image)
        # gt_image_out = gt_image_out.permute(2, 0, 1)  # 转成3，255，255
        return {"image": image_out,
                # "gt_image": gt_image_out,
                "gt": gt}



if __name__ == '__main__':
    with open('hyper_parameters.yaml', 'r') as f:
        hyper = yaml.safe_load(f)
    ds = CustomDataset(hyper)
    tr_loader = DataLoader(dataset = ds, batch_size = 16, shuffle = False, num_workers = 1, pin_memory=True)
    tr_batch = next(iter(tr_loader)) # 拿出部分数据做验证