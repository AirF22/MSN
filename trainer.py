import torch
from datetime import datetime
import os
import model
from torch.optim import lr_scheduler
from dataset import CustomDataset, CustomValDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tools
from loss import MyLoss
import shutil
import yaml
import argparse
import sys
from test_single import Tester


class Trainer():
    def __init__(self, hyper):
        self.hyper = hyper
        self.count = 0  # 用于验证过程中的计数

        # 创建训练模型、优化器及调度器
        if hyper['MODEL']['NAME'] == 'FinalNet':
            self.model = getattr(model, hyper['MODEL']['NAME'])(hyper).to(self.hyper['TRAIN']['DEVICE'])

        # 创建优化器和调度器
        self.optimizer = self.define_optimizer()
        self.scheduler = self.define_scheduler()

        # 创建数据加载器
        self.train_ld = self.define_train_loader()
        self.val_ld = self.define_validation_loader()

        # 设定损失函数
        self.loss_f = MyLoss(hyper)

        # 日志目录：在存放实验结果的目录下，用时间戳创建一个新目录来保存本次实验的日志
        self.log_dir = os.path.join('Result', hyper['DESCRIBE'], datetime.now().strftime("%m%d%H%M"))
        tools.make_dir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)  # 日志写入器

        # 记录一些训练数据，便于接续训练
        self.start_epoch = 0  # 起始训练轮次
        self.global_step = 0  # 起始迭代索引
        self.best_loss = 10 ** 6  # 记录最好的验证损失

    def train(self):
        start_time = datetime.now()  # 记录训练开始时间
        for epoch in range(self.start_epoch, self.hyper['TRAIN']['EPOCH']):
            epoch_loss = 0  # 每一轮的平均损失
            batch_num = len(self.train_ld)  # 一个epoch中batch的数量
            with tqdm(total=batch_num, leave=True,
                      desc=f'Epoch {epoch + 1}/{self.hyper["TRAIN"]["EPOCH"]}',
                      unit='it') as process_bar:
                for _, batch in enumerate(self.train_ld):
                    image = batch['image'].to(self.hyper['TRAIN']['DEVICE'])
                    gt = batch['gt'].to(self.hyper['TRAIN']['DEVICE'])
                    pre, illu_branch = self.model(image)

                    batch_loss = self.loss_f(pre, gt, illu_branch)  # 一批的损失

                    # total_loss = batch_loss

                    # tools.save_batch(batch_loss[0], 0)
                    # tools.save_batch(batch_loss[1], 1)
                    # gt[:, 1, :, :] = 0
                    # tools.save_batch(gt, 2)

                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    epoch_loss += batch_loss

                    # 计算时间信息
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    avg_time_per_epoch = elapsed_time / (epoch - self.start_epoch + 1) if epoch > self.start_epoch else 0
                    remaining_epochs = self.hyper['TRAIN']['EPOCH'] - epoch - 1
                    estimated_remaining_time = avg_time_per_epoch * remaining_epochs

                    # 更新进度条信息
                    process_bar.set_postfix({
                        'batch_loss': f'{batch_loss.item():.4f}',
                        'elapsed': f'{elapsed_time / 3600:.1f}h',
                        'remaining': f'{estimated_remaining_time / 3600:.1f}h' if estimated_remaining_time > 0 else 'N/A'
                    })
                    process_bar.update(1)
                    self.global_step += 1

            # 更新学习率
            # if (epoch + 1) <= int(self.hyper['TRAIN']['EPOCH'] * (2 / 3)):
            self.scheduler.step()

            # 记录日志
            epoch_avg_loss = epoch_loss / batch_num
            self.writer.add_scalar('epoch_loss', epoch_avg_loss, epoch + 1)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch + 1)

            # 存储训练状态
            if (epoch + 1) % self.hyper['TRAIN']['SAVE_INTERVAL'] == 0:
                self.save(epoch)
                print(f'epoch: {epoch + 1} model saved')

            # 进行模型的验证
            if (epoch + 1) % self.hyper['TRAIN']['VALIDATION_INTERVAL'] == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val_loss', val_loss, epoch + 1)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save(epoch, best=True)
                    print(f'New best model saved with val_loss: {val_loss:.4f}')
        return self.log_dir


    def validate(self):
        self.model.eval()  #
        n_val = len(self.val_ld)
        mae = 0
        with tqdm(total=n_val, desc='valdiation', unit='it', leave=False) as pbar:
            for batch_index_val, batch in enumerate(self.val_ld):
                image = batch['image'].to(self.hyper['TRAIN']['DEVICE'])
                gt = batch['gt'].to(self.hyper['TRAIN']['DEVICE'])

                with torch.no_grad():  # 不追踪参数的梯度
                    pre, illu_branch = self.model(image)

                    if self.hyper['TRAIN']['ONLY_SINGLE_ILLU']:  # 纯单光源场景
                        pre = pre.mean(dim=-1).mean(dim=-1)
                        gt = gt.mean(dim=-1).mean(dim=-1)

                    loss, _ = self.loss_f.angular_error(pre, gt)
                    if batch_index_val == 0:  # 显示第0批的结果
                        self.count += 1
                        self.writer.add_image("pre_val", make_grid(pre, nrow=4, padding=1), self.count)  # pre
                        self.writer.add_image("gts_val", make_grid(gt, nrow=4, padding=1), self.count)  # GT
                        for idx, illu in enumerate(illu_branch):
                            self.writer.add_image(f"illu_{idx}", make_grid(illu, nrow=4, padding=1), self.count)  # 每个分支（不同频率）

                mae += loss.item()

                pbar.update(1)
                pbar.set_postfix({'vali_loss': loss.item()})
        self.model.train()
        return mae / n_val

    # 根据超参数创建优化器
    def define_optimizer(self):
        if self.hyper['TRAIN']['OPTIMIZER']['ADAMW']:  # 创建线性调度器
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          self.hyper['TRAIN']['LR'])
        else:
            raise NotImplementedError('optimizer is not implemented')
        return optimizer

    # 根据超参数创建调度器
    def define_scheduler(self):
        if 'LINEARLR' in self.hyper['TRAIN']['SCHEDULER']:  # 创建线性调度器
            scheduler = lr_scheduler.LinearLR(self.optimizer,
                                              self.hyper['TRAIN']['SCHEDULER']['LINEARLR']['START_FACTOR'],
                                              self.hyper['TRAIN']['SCHEDULER']['LINEARLR']['END_FACTOR'],
                                              self.hyper['TRAIN']['SCHEDULER']['LINEARLR']['TOTAL_ITERS'])
        else:
            raise NotImplementedError('scheduler is not implemented')
        return scheduler

    # 根据超参数创建训练数据加载器
    def define_train_loader(self):
        dataset = CustomDataset(self.hyper)
        return DataLoader(dataset, batch_size=self.hyper['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=self.hyper['TRAIN']['NUM_WORKERS'], pin_memory=True)

    # 根据超参数创建验证数据加载器
    def define_validation_loader(self):
        dataset = CustomValDataset(self.hyper)
        return DataLoader(dataset, batch_size=self.hyper['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=self.hyper['TRAIN']['NUM_WORKERS'], pin_memory=True)

    # 存储模型
    def save(self, epoch, best=False):
        base_name = os.path.join(self.log_dir, 'model')

        if best:
            base_name += '_best'
        else:
            base_name += '_latest'

        check = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'global_step': self.global_step,
                 'epoch': epoch,
                 'best_loss': self.best_loss}
        torch.save(check, base_name + '.pt')  # 存储节点数据

    # 加载模型
    def load(self, check_path):
        check_point = torch.load(check_path)
        self.model.load_state_dict(check_point['model'])
        self.optimizer.load_state_dict(check_point['optimizer'])
        self.scheduler.load_state_dict(check_point['scheduler'])
        self.start_epoch = check_point['epoch'] + 1
        self.global_step = check_point['global_step'] + 1
        self.best_loss = check_point['best_loss']

    # 将超参数文件存到日志目录中
    def save_hyper(self, hyper_file_path):
        shutil.copy(hyper_file_path, self.log_dir)
        code_path = os.path.join(self.log_dir, 'code')
        os.makedirs(code_path, exist_ok=True)
        file_list = ['./dataset.py',
                     './model.py',
                     './main_train.py',
                     './trainer.py',
                     './loss.py',
                     './test_single.py',
                     './tools.py'
                     ]
        for file in file_list:
            if os.path.exists(file):
                shutil.copy(file, code_path)


if __name__ == '__main__':
    sys.argv = ['trainer.py', '--hyper_parameter_path', 'hyper_parameters.yaml']
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyper_parameter_path', type=str, required=True, help="超参数文件路径")
    config = parser.parse_args()

    # 读取设置文件
    with open(config.hyper_parameter_path, 'r') as f:
        hyper = yaml.safe_load(f)
    tools.set_seed(hyper['TRAIN']['SEED'])  # 设置随机数种子

    SCRATCH = True  # 是否为首次训练
    if SCRATCH:  # 首次训练
        trainer = Trainer(hyper)
        trainer.save_hyper(hyper_file_path='hyper_parameters.yaml')
        MODEL_PATH = trainer.train()
        # auto_test = Tester(log_path=MODEL_PATH)  # 自动完成测试
        # auto_test.test()
    else:  # 接续训练
        trainer = Trainer(hyper)
        trainer.load(check_path=None)  # 加载路径
        MODEL_PATH = trainer.train()
        auto_test = Tester(log_path=MODEL_PATH)  # 自动完成测试
        auto_test.test()