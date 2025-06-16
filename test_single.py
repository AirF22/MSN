import os
import numpy as np
import torch
import numpy
import pandas as pd
import time
from tqdm import tqdm
from tool import *
from localDataLoader import *
from model import *
from torch.utils.data import DataLoader
from openpyxl import load_workbook, Workbook

# =============================== Hyper Parameter ===================================
EXP_NAME = 'pretrained_galaxy'  # 实验名称
CAMERA = 'galaxy'
FILE_TYPE = 'tiff'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
eps = 1e-10
MODEL_NAME = 'best'  # 模型名称
DATASET = 'train'
MODEL_PATH = './Models/' + CAMERA + '/'+ EXP_NAME  # 模型路径
DATA_PATH = os.path.expanduser('~/work/python/Public_DataSets/' + 'LSMI/' + FILE_TYPE + '/' + CAMERA + '_256/')  # LSMI数据集
# DATA_PATH = os.path.expanduser('~/work/python/Public_DataSets/' + 'CubePP/' + FILE_TYPE + '/' + CAMERA + '/')  # Cube++数据集
ANGLE_path = './Result/check_result.xlsx'
MSE_path = './Result/check_result_MSE.xlsx'
ILLU_NUMBER = 'all_illu'  # 测试集的光源数量

picture_mae_path = './Result/'+CAMERA+'/'+EXP_NAME+'/'
# excel_file_path = picture_mae_path+'picture_mae.xlsx'
# if not os.path.isdir(picture_mae_path):
#     os.makedirs(picture_mae_path)
# if not os.path.exists(excel_file_path):
#     wb = Workbook()
#     ws = wb.active
#     ws.title = 'Sheet1'
#     ws.append(['file_name', 'MAE'])  # 写入表头
#     wb.save(excel_file_path)
#
# # 加载现有的工作簿和工作表
# wb = load_workbook(excel_file_path)
# ws = wb['Sheet1']
# ===================================================================================
print('Using：', DEVICE)
# model = torch.load(MODEL_PATH + '/' + MODEL_NAME + '.pkl', map_location=torch.device(DEVICE))
model = FinalNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH + '/' + MODEL_NAME + '.pt', map_location=torch.device(DEVICE)))

torch.set_grad_enabled(False)
model.eval()

file_name = [os.path.splitext(f)[0] for f in os.listdir(DATA_PATH + DATASET) if f.endswith('.' + FILE_TYPE) and 'gt' not in f]

angle_list = []
MSE_list = []
time_list = []
for filename_prefix in tqdm(file_name):
    # part = filename_prefix.split('_')

    # if len(part[1]) != 1:  # 只要单光源
    #     continue

    # if len(part[1])==1 or len(part[1])==3:  # 放弃单光源和3光源(只要2光源)
    #     continue

    # if len(part[1]) != 3:  # 只要3光源
    #     continue

    # 获取路径
    img_path = os.path.join(DATA_PATH, DATASET, filename_prefix + '.' + FILE_TYPE)
    # img_file_name = os.path.splitext(self.file_list[idx])[0]
    # gt_image_path = os.path.join(DATA_PATH, DATASET, filename_prefix + '_gt.'+FILE_TYPE)
    # gt_image_file_name = os.path.splitext(self.gt_image_list[idx])[0]
    mask_path = os.path.join(DATA_PATH, 'train', filename_prefix + '_mask.png')
    gt_path = os.path.join(DATA_PATH, 'illu_map', filename_prefix + ".npy")

    # 加载数据
    image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
    mask = cv2.cvtColor(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
    # gt_image = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype('float32')
    gt = np.load(gt_path)

    # 转为张量，送入模型
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2, 0, 1)  # 转成3，255，255
    image_tensor = image_tensor / (image_tensor.max() + eps)
    image_tensor = image_tensor.unsqueeze(dim=0).to(DEVICE)

    start = time.perf_counter()
    pred_illu, pixel_illu, patch_illu, overall_illu, weight_map = model(image_tensor)
    end = time.perf_counter()
    elapsed = end - start
    time_list.append(elapsed)
    # ---------------------- 将光照图取平均并广播(cube++) --------------------
    # illu = pred_illu.detach()
    # illu = illu.squeeze()
    # illu = illu.permute(1, 2, 0).cpu().numpy()
    # mean_values = np.mean(illu, axis=(0, 1), keepdims=True)  # (1, 1, 3)
    # pred_illu = np.broadcast_to(mean_values, illu.shape)  # (256, 256, 3)
    # ---------------------------------------------------------------------


    # ---------------------------- 检查中间光照图 ----------------------------
    # if len(filename_prefix.split('_')[-1]) > 1:  # 多光源
    #
    #     gt_np = gt * 255
    #     gt_np = gt_np.astype(np.uint8)  # 转换为8位无符号整数
    #     gt_np[:, :, 1] = 0
    #     gt_illu_image = Image.fromarray(gt_np)
    #     gt_illu_image.save('./Result/' + CAMERA + '/' + filename_prefix + '_gt_illu_image.png')
    #
    #     pixel_illu_tensor = pixel_illu * 255
    #     pixel_illu_tensor = pixel_illu_tensor.squeeze()
    #     pixel_illu_tensor = pixel_illu_tensor.permute(1, 2, 0)
    #     pixel_illu_np = pixel_illu_tensor.cpu().numpy().astype(np.uint8)
    #     pixel_illu_np[:, :, 1] = 0
    #     pixel_illu_image = Image.fromarray(pixel_illu_np)
    #     pixel_illu_image.save('./Result/'+CAMERA+'/'+filename_prefix+'_pixel_illu_image.png')
    #
    #     patch_illu_tensor = patch_illu * 255
    #     patch_illu_tensor = patch_illu_tensor.squeeze()
    #     patch_illu_tensor = patch_illu_tensor.permute(1, 2, 0)
    #     patch_illu_np = patch_illu_tensor.cpu().numpy().astype(np.uint8)
    #     patch_illu_np[:, :, 1] = 0
    #     patch_illu_image = Image.fromarray(patch_illu_np)
    #     patch_illu_image.save('./Result/'+CAMERA+'/' + filename_prefix + '_patch_illu_image.png')
    #
    #     overall_illu_tensor = overall_illu * 255
    #     overall_illu_tensor = overall_illu_tensor.squeeze()
    #     overall_illu_tensor = overall_illu_tensor.permute(1, 2, 0)
    #     overall_illu_np = overall_illu_tensor.cpu().numpy().astype(np.uint8)
    #     overall_illu_np[:, :, 1] = 0
    #     overall_illu_image = Image.fromarray(overall_illu_np)
    #     overall_illu_image.save('./Result/'+CAMERA+'/' + filename_prefix + '_overall_illu_image.png')
    #
    #     weight_map_tensor = weight_map * 255
    #     weight_map_tensor = weight_map_tensor.squeeze()
    #     weight_map_tensor = weight_map_tensor.permute(1, 2, 0)
    #     weight_map_np = weight_map_tensor.cpu().numpy().astype(np.uint8)
    #     weight_map_image = Image.fromarray(weight_map_np)
    #     weight_map_image.save('./Result/'+CAMERA+'/' + filename_prefix + '_weight_map_image.png')
    #
    #     pred_tensor = pred_illu * 255
    #     pred_tensor = pred_tensor.squeeze()
    #     pred_tensor = pred_tensor.permute(1, 2, 0)
    #     pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
    #     pred_np[:, :, 1] = 0
    #     pred_image = Image.fromarray(pred_np)
    #     pred_image.save('./Result/'+CAMERA+'/' + filename_prefix + '_pred_image.png')
    # ---------------------------------------------------------------------


    pred_illu = pred_illu.squeeze().permute(1, 2, 0)
    pred_illu = pred_illu.cpu().numpy()

    pred_illu = pred_illu.reshape(pred_illu.shape[0] * pred_illu.shape[1], 3)
    gt = gt.reshape(gt.shape[0] * gt.shape[1], 3)

    mask = mask.reshape(mask.shape[0] * mask.shape[1], 3)
    row_mask = np.any(mask != 0, axis=1)
    filtered_pred_illu = pred_illu[row_mask]
    filtered_gt = gt[row_mask]
    # ---------------------- statistic -----------------------
    # 计算每张图片的MSE
    MSE = np.mean((filtered_gt - filtered_pred_illu) ** 2)
    MSE_list.append(MSE)

    # 计算每张图片的角度差
    angle = getMAE_numpy_2D(filtered_pred_illu, filtered_gt)
    a_pic_mean = angle.mean()
    angle_list.append(a_pic_mean)
    # ws.append([filename_prefix, a_pic_mean])  # 记录每张图片的MAE到文件
    # --------------------------------------------------------
# wb.save(excel_file_path)  # 保存
# ---------------- FPS --------------------
total_time = sum(time_list)
total_number = len(time_list)
fps = total_number / total_time
print('fps:', fps, '\n')
# -----------------------------------------
def statistic(alist, save_path):
    alist = sorted(alist)
    total = sum(alist)
    n = len(alist)
    avg = total / n  # 平均值

    squared_diffs = sum((x - avg) ** 2 for x in alist)  # 计算每个数据点与平均值的差的平方，并求和
    variance = squared_diffs / len(alist)  # 计算方差（标准差的平方）
    std = variance ** 0.5  # 计算标准差

    if n % 2 == 1:
        median = alist[n // 2]
    else:
        median = (alist[n // 2 - 1] + alist[n // 2]) / 2

    # 四分位数
    q1_index = int(n * 0.25)
    q3_index = int(n * 0.75)
    q1 = alist[q1_index]
    q3 = alist[q3_index]

    trimean = (2 * median + q1 + q3) / 4  # 三均值

    best25_avg = sum(alist[:q1_index + 1]) / (q1_index + 1)  # 前25%的平均值

    # 后75%的平均值
    last_75_percent = alist[q1_index:]
    worst25_avg = sum(last_75_percent) / len(last_75_percent)

    # 第95%位的值
    ninety_fifth_percentile_index = int((n - 1) * 0.95 + 0.5)  # 四舍五入到最接近的整数
    per95 = alist[ninety_fifth_percentile_index]
    print("Mean:", avg,
          "\nStd:", std,
          "\nMedian:", median,
          "\nTrimean:", trimean,
          "\nBest25:", best25_avg,
          "\nWorst25:", worst25_avg,
          "\nPercentile95:", per95)
    # ---------------------- 保存excel ----------------------
    if not os.path.exists(save_path):
        wb = Workbook()
        ws = wb.active
        ws.title = 'Sheet1'
        ws.append(['Name', 'Mean', 'Std', 'Median', 'Trimean', 'Best25', 'Worst25', '95Per', ])  # 写入表头
        wb.save(save_path)

    # 加载现有的工作簿和工作表
    wb = load_workbook(save_path)
    ws = wb['Sheet1']
    ws.append([ILLU_NUMBER + '>>>' + CAMERA + '>>>' + EXP_NAME + '>>>' + MODEL_NAME + '>>>' + DATASET,
               avg, std, median, trimean, best25_avg, worst25_avg, per95])
    wb.save(save_path)
    # -----------------------------------------------------
    print(f'\n新数据已追加到 {save_path}')
    return 0
statistic(angle_list, ANGLE_path)  # 统计角度差
statistic(MSE_list, MSE_path)  # 统计均方差


