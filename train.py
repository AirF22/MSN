import torch
import os
import datetime
import shutil
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import model
from localDataLoader import *
# from localDataLoader_CubePP import CubePPDataset
from tool import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# ============================== Hyper Parameter ==============================
DESCRIBE = "预训练的nikon模型"
EXP_NAME = 'pretrain_nikon'
DATASET = 'LSMI'
CAMERA = 'nikon'
FILE_TYPE = 'tiff'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCH = 600
VALIDATION_INTERVAL = 5
eps = 1e-10
set_seed(35)
# =============================================================================
# 定义数据和模型的路径
MODEL_PATH = './Models/'+CAMERA+'/'+EXP_NAME  # 模型路径
writer = SummaryWriter('./logs/'+CAMERA+'/'+EXP_NAME)  # 设置TensorBoard的日志路径
DATA_PATH = os.path.expanduser('~/work/python/Public_DataSets/'+DATASET+'/'+FILE_TYPE+'/'+CAMERA+'_256/')  # LSMI训练集
VAL_DATA_PATH = os.path.expanduser('~/work/python/Public_DataSets/'+DATASET+'/'+FILE_TYPE+'/'+CAMERA+'_256/')  # LSMI验证集

print('Loading train set...')
# 加载训练集
dataset = CustomDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

# 加载验证集
val_dateset = CustomValDataset(VAL_DATA_PATH)
val_loader = DataLoader(val_dateset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1)

print('Using：', DEVICE)

# 建立模型并载入设备
model = model.FinalNet().to(DEVICE)

# 定义优化器及调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, 600)

print('\n-----------------\n'
      'Num of epoch: {}\n'
      'Batch size: {}\n'
      'Num of batch: {}'.format(EPOCH, BATCH_SIZE, len(loader)))
print('-----------------\n')
print('Start training...')

# ------------------------------- 保存相关源码文件 -------------------------------------
source_files = ['./model.py',
                './train.py',
                './tool.py',
                './test_single.py',
                './localDataLoader.py'
                ]
destination_dir = MODEL_PATH + '/'
if not os.path.exists(destination_dir):
    print(f"没有保存文件的文件夹 {destination_dir}！")
    os.makedirs(destination_dir)
    print("已自动创建！")

for source_file in source_files:
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_dir)
        print(f"已复制文件: {source_file} -> {destination_dir}")
    else:
        print(f"警告: 源文件 {source_file} 不存在，跳过复制")

file_path = destination_dir+'describe.txt'
if os.path.exists(file_path):
    # 若文件存在，则以追加模式（'a'）打开文件
    mode = 'a'  # 'a'代表追加模式
else:
    # 若文件不存在，则以写入模式（'w'）打开文件
    mode = 'w'  # 'w'代表写入模式，会覆盖已有内容（但在此场景中，若文件不存在，则创建新文件）

# 打开（如果不存在则创建）一个名为describe.txt的文件，以写入模式（'w'）
with open(file_path, mode, encoding='utf-8') as file:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_content = f"{current_time}\n   {DESCRIBE}\n"
    file.write(full_content)  # 将字符串写入文件
# 文件会在with块结束时自动关闭
# ----------------------------------------------------------------------------

# 训练
count = 0  # 用于验证过程中的计数
min_MAE = 999  # 记录最小角度差
for epoch in tqdm(range(EPOCH), desc="训练进度：", leave=False):
    print('\nTraining epoch {}/{}'.format(epoch + 1, EPOCH))

    train_loss_epoch = 0.0  # 用于记录每个epoch中的总损失。
    for batch_index, batch in enumerate(loader):  # 每次处理一批图像
        #  加载数据集成为张量
        a_batch_images = batch['image']
        a_batch_gts = batch['gt']
        # a_batch_names = batch['image_name']

        a_batch_images = a_batch_images.to(DEVICE)
        a_batch_gts = a_batch_gts.to(DEVICE)

        output, _, _, _, _ = model(a_batch_images)
        loss = MAE_loss(output, a_batch_gts)
        print('Batch{}/{}Loss: {:.6f}'.format(batch_index + 1, len(loader), loss))
        train_loss_epoch += loss  # 累加成一个epoch的损失（每次加一个批次的损失）。

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_epoch_avg = train_loss_epoch/len(loader)  # 一个epoch的平均损失
    print('Epoch{}/{}Loss_Avg: {:.6f}'.format(epoch+1, EPOCH, train_loss_epoch_avg))
    writer.add_scalar("train_loss_epoch_avg", train_loss_epoch_avg, epoch+1)

    # ------------------------- Validation -------------------------
    if (epoch+1) % VALIDATION_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            val_loss_all = 0
            angle_list = []
            for batch_index_val, batch_val in enumerate(val_loader):
                print("···Validating", batch_index_val+1, "···")
                a_batch_images_val = batch_val['image']
                a_batch_gts_val = batch_val['gt']
                a_batch_images_val = a_batch_images_val.to(DEVICE)
                a_batch_gts_val = a_batch_gts_val.to(DEVICE)

                output_val, pixel_illu, patch_illu, overall_illu, weight_map = model(a_batch_images_val)

                # ----------------------- LSMI -----------------------
                val_MAE = getMAE_tensor(output_val, a_batch_gts_val)
                angle_list.append(val_MAE)
                # ----------------------------------------------------

                # ----------------------- cube++ ---------------------
                # pred_val = output_val.mean(dim=-1).mean(dim=-1)
                # gt_val = a_batch_gts_val.mean(dim=-1).mean(dim=-1)
                # val_MAE = getMAE_single_tensor(pred_val, gt_val)
                # angle_list.append(val_MAE)
                # ----------------------------------------------------

                if batch_index_val == 0:  # 记录第0批
                    count += 1
                    writer.add_image("pred_illu_val", make_grid(output_val, nrow=4, padding=1), count)
                    writer.add_image("gt_illu_val", make_grid(a_batch_gts_val, nrow=4, padding=1), count)
                    # writer.add_image("a_batch_pixel_illu_val", make_grid(pixel_illu, nrow=4, padding=1), count)  # pixel_illu
                    # writer.add_image("a_batch_patch_illu_val", make_grid(patch_illu, nrow=4, padding=1), count)  # patch_illu
                    # writer.add_image("a_batch_overall_illu_val", make_grid(overall_illu, nrow=4, padding=1), count)  # overall_illu
                    # writer.add_image("a_batch_weight_map_val", make_grid(weight_map, nrow=4, padding=1), count)

                val_loss = MAE_loss(output_val, a_batch_gts_val)
                print('Batch{}/{}Loss_avg: {:.6f}'.format(batch_index_val + 1, len(val_loader), val_loss))

                val_loss_all += val_loss  # 累加成一个epoch的损失

            val_loss_all_avg = val_loss_all/len(val_loader)  # 一个epoch的平均损失
            writer.add_scalar("val_loss_all_avg", val_loss_all_avg, epoch+1)

            angle_tensor_epoch = torch.cat(angle_list, dim=0)
            Mean = angle_tensor_epoch.mean()
            writer.add_scalar("Mean", Mean, epoch+1)

            if Mean < min_MAE:
                min_MAE = Mean
                print('Saving the Best Model...')
                if not os.path.exists(MODEL_PATH):
                    os.makedirs(MODEL_PATH)
                torch.save(model.state_dict(), '{}/best.pt'.format(MODEL_PATH))
                print('OK')
            torch.save(model.state_dict(), '{}/new.pt'.format(MODEL_PATH))
        model.train()  # 继续训练
    # --------------------------------------------------------------
    if (epoch+1) <= 400:
        scheduler.step()  # 学习率调度器，用于更新学习率
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch+1)
writer.close()

# 保存整个网络
print('Saving the model...')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH + '/final.pt')
print('Complete')
