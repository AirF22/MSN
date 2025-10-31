import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from PIL import Image
import numpy as np

class MyLoss(nn.Module):
    def __init__(self, hyper):
        super(MyLoss, self).__init__()
        self.hyper = hyper

    def angular_error(self, pred, gt):
        eps = 1e-6

        # 确保输入维度一致
        assert pred.shape == gt.shape, "Pred and GT must have same shape"

        # 计算点积和向量长度
        dot_product = torch.sum(pred * gt, dim=1)
        pred_length = torch.norm(pred, p=2, dim=1) + eps  # 避免除零
        gt_length = torch.norm(gt, p=2, dim=1) + eps

        # 计算余弦相似度，并进行安全裁剪
        cosine_sim = dot_product / (pred_length * gt_length)

        # 更安全的裁剪范围，确保在[-1, 1]范围内
        # 由于浮点精度问题，可能需要稍微缩小范围
        cosine_sim = torch.clamp(cosine_sim, -1 + eps, 1 - eps)

        # 计算角度（弧度）
        angle_rad = torch.acos(cosine_sim)

        # 转换为角度并计算均值
        angle_deg = angle_rad * (180.0 / math.pi)
        angle_mean = torch.mean(angle_deg)
        angle_std = torch.std(angle_deg)
        return angle_mean, angle_std

    # region
    # def decompose_gt(self, gt):
    #     gt = gt[:, [0, 2], :, :] # 不考虑绿通道

    #     # 高斯金字塔
    #     gt_gaussian_1 = trff.gaussian_blur(gt, [5, 5])
    #     gt_downsample_1 = gt_gaussian_1[:, :, ::2, ::2]

    #     gt_gaussian_2 = trff.gaussian_blur(gt_downsample_1, [5, 5])
    #     gt_downsample_2 = gt_gaussian_2[:, :, ::2, ::2]

    #     # 上采样
    #     gt_upsample_1 = torch.zeros(gt.shape[0], gt.shape[1], gt.shape[2], gt.shape[3]).to(self.hyper['TRAIN']['DEVICE'])
    #     gt_upsample_1[:, :, ::2, ::2] = gt_downsample_1
    #     gt_upsample_gaussian_1 = trff.gaussian_blur(gt_upsample_1, [5, 5])
    #     gt_res_1 = gt - gt_upsample_gaussian_1

    #     gt_upsample_2 = torch.zeros(gt_downsample_1.shape[0], gt_downsample_1.shape[1], \
    #                                 gt_downsample_1.shape[2], gt_downsample_1.shape[3]).to(self.hyper['TRAIN']['DEVICE'])
    #     gt_upsample_2[:, :, ::2, ::2] = gt_downsample_2
    #     gt_upsample_gaussian_2 = trff.gaussian_blur(gt_upsample_2, [5, 5])
    #     gt_res_2 = gt_downsample_1 - gt_upsample_gaussian_2
    #     gt_res_2 = f.interpolate(gt_res_2, scale_factor = 2) # 将其恢复到原始大小

    #     all_one = torch.zeros(gt.shape[0], 1, gt.shape[2], gt.shape[3]).to(self.hyper['TRAIN']['DEVICE'])
    #     gt_res_1 = torch.cat((gt_res_1[:, 0:1, :, :], all_one, gt_res_1[:, 1:2, :, :]), dim = 1)
    #     gt_res_2 = torch.cat((gt_res_2[:, 0:1, :, :], all_one, gt_res_2[:, 1:2, :, :]), dim = 1)
    #     return gt_res_1, gt_res_2

    # # 本卷积核对噪声是敏感的，恰好对本任务是有利的
    # def tv_term(self, pred, gt):
    #     # 预测值的一阶梯度
    #     pred_horizontal_diff = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    #     pred_verttical_diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    #     pred_grad = torch.abs(pred_horizontal_diff) + torch.abs(pred_verttical_diff)

    #     # 真实值的一阶梯度
    #     gt_horizontal_diff = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    #     gt_verttical_diff = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    #     gt_grad = torch.abs(gt_horizontal_diff) + torch.abs(gt_verttical_diff)

    #     return torch.mean(torch.abs(pred_grad - gt_grad))
    # endregion

    # # 低频保持正则化项
    # def tv_loss(self, x):
    #     # 计算水平方向的 TV Loss
    #     horizontal_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
    #     horizontal_tv_loss = torch.mean(torch.abs(horizontal_diff))

    #     # 计算垂直方向的 TV Loss
    #     vertical_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
    #     vertical_tv_loss = torch.mean(torch.abs(vertical_diff))

    #     return horizontal_tv_loss + vertical_tv_loss

    # 
    # sobel正则化项
    def sobel_term(self, pred, gt):
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ], dtype=torch.float32)

        # 扩展维度以适应卷积层要求 (out_channels, in_channels, H, W)
        sobel_x = sobel_x.view(1, 1, 3, 3).to(self.hyper['TRAIN']['DEVICE'])
        sobel_y = sobel_y.view(1, 1, 3, 3).to(self.hyper['TRAIN']['DEVICE'])
        # # 复制核以适应输入通道数
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)

        # 使用sobel算子计算预测图与真实图的梯度，并计算两者间的误差
        pred_gx = F.conv2d(pred, sobel_x, groups=3)
        pred_gy = F.conv2d(pred, sobel_y, groups=3)
        pred_grad = torch.abs(pred_gx) + torch.abs(pred_gy)

        gt_gx = F.conv2d(gt, sobel_x, groups=3)
        gt_gy = F.conv2d(gt, sobel_y, groups=3)
        gt_grad = torch.abs(gt_gx) + torch.abs(gt_gy)

        return torch.mean(torch.abs(pred_grad - gt_grad))
    
    # 高斯正则化项
    def gaussian_term(self, pred, gt, kernel_size = 5, sigma = 1):
         # 创建坐标网格
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # 计算高斯分布
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # 归一化
        
        # 扩展为多通道
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(self.hyper['TRAIN']['DEVICE'])
        kernel = kernel.repeat(3, 1, 1, 1)

        # 使用高斯算子对预测图与真实图作模糊操作
        pred_gaussian = F.conv2d(pred, kernel, groups = 3)
        gt_gaussian = F.conv2d(gt, kernel, groups = 3)

        return torch.mean(torch.abs(pred_gaussian - gt_gaussian))
    
    # 均值正则化项
    def average_term(self, pred, gt, kernel_size = 5):
         # 创建均值核
        kernel = torch.ones(kernel_size, kernel_size, dtype=torch.float32)
        kernel = kernel / kernel.sum()  # 归一化
        
        # 扩展为多通道
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(self.hyper['TRAIN']['DEVICE'])
        kernel = kernel.repeat(3, 1, 1, 1)

        # 使用高斯算子对预测图与真实图作模糊操作
        pred_box = F.conv2d(pred, kernel, groups = 3)
        gt_box = F.conv2d(gt, kernel, groups = 3)

        return torch.mean(torch.abs(pred_box - gt_box))

    def forward(self, pred, gt, illu_branch):
        total_error = self.hyper['TRAIN']['LOSS']['ANGULAR_ERROR'] * self.angular_error(pred, gt)[0]
        # 对大尺寸分支的输出，施加sobel正则化项
        for illu in illu_branch[1:]:
            total_error += self.hyper['TRAIN']['LOSS']['HIGH_FRE_REG'] * self.sobel_term(pred, illu)
        return total_error

# class VanillaGANLoss(nn.Module):
#     def __init__(self, target_real_label=1.0, target_fake_label=0.0):
#         super(VanillaGANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         self.loss = nn.BCEWithLogitsLoss()  # 更稳定，包含sigmoid+BCE

#     def get_target_tensor(self, prediction, target_is_real):
#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(prediction)

#     def __call__(self, prediction, target_is_real):
#         target_tensor = self.get_target_tensor(prediction, target_is_real)
#         loss = self.loss(prediction, target_tensor)
#         return loss

if __name__ == '__main__':
    with open('hyper_parameters.yaml', 'r', encoding='utf-8') as f:
        hyper = yaml.safe_load(f)
    loss = MyLoss(hyper)
    img = torch.from_numpy(np.array(Image.open('LSMI.png'), dtype=np.float32)[:, :, :3])
    img = torch.permute(img, (2, 0, 1))
    img = torch.unsqueeze(img, 0)
    img_blur = loss.average_terms_test(img)
    img_blur = img_blur[0, :, :, :]
    img_blur = torch.permute(img_blur, (1, 2, 0))
    img_blur = img_blur.numpy().astype(np.uint8)
    img_blur_pil = Image.fromarray(img_blur)
    img_blur_pil.save('LSMI_blur.png')