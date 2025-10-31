import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import sys
from base.wave import DWT_2D, IDWT_2D


class DCB(nn.Module):  # 双卷积模块（基本单元）
    def __init__(self, ch_in, ch_out, kernel=3, padding=1):  # 高频分支核的大小改为1， padding改为0
        super(DCB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down_blk(nn.Module):  # Unet下采样块（池化＋DCB）
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DCB(ch_in, ch_out)
        )

    def forward(self, x):
        return self.main(x)

class UCB(nn.Module):  # 上采样卷积模块（基本单元）
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# 高频门控网络
class HighFreGate(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.k = int(k * in_ch)  # 门控排名靠前的k个通道
        self.high_pass = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        # 初始化一个拉普拉斯核
        laplace_kernel = torch.tensor([[-1., -1., -1.],
                                       [-1., 8., -1.],
                                       [-1., -1., -1.]]).expand(in_ch, 1, 3, 3)
        self.high_pass.weight = nn.Parameter(laplace_kernel, requires_grad=True)  # 允许微调

    def grad(self, x):  # 计算张量的一阶梯度
        x_grad = self.high_pass(x)
        return torch.mean(x_grad, dim=(2, 3))  # 求张量的一阶梯度在空间维度的均值

    def forward(self, x):
        x_grad = self.grad(x)  # 求张量x的梯度
        top_k_values, top_k_indices = torch.topk(x_grad, k=self.k, dim=-1)
        weights = F.softmax(top_k_values, dim=-1)
        sparse_weights = torch.zeros_like(x_grad)
        sparse_weights.scatter_(-1, top_k_indices, weights)
        sparse_weights = torch.unsqueeze(sparse_weights, 2)
        sparse_weights = torch.unsqueeze(sparse_weights, 3)
        return sparse_weights * x

# 特征比较门
# 将下级特征与上级特征作比较，选出最不相似的k个特征（空间特征，k个维度）
class CompareGate(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.k = int(k * in_ch)  # 门控排名靠前的k个通道

    def forward(self, x, y):
        # 根据求特征的余弦相似度来衡量特征间的相似度，并且将相似的特征去掉
        x1 = torch.flatten(x, 2, 3)
        y = torch.flatten(y, 2, 3)
        fea_sim = F.cosine_similarity(x1, y, dim=-1)  # 计算张量的余弦相似度，空间信息作为向量
        top_k_values, top_k_indices = torch.topk(fea_sim, k=self.k, dim=-1, largest=False)  # 这里取得是最不相关的特征
        weights = F.softmax(top_k_values, dim=-1)  # 通道维度
        sparse_weights = torch.zeros_like(fea_sim)
        sparse_weights.scatter_(-1, top_k_indices, weights)
        sparse_weights = torch.unsqueeze(sparse_weights, 2)
        sparse_weights = torch.unsqueeze(sparse_weights, 3)
        return sparse_weights * x

# 上采样模块
class Up_blk(nn.Module):  # Unet上采样模块
    def __init__(self, ch_in, ch_out, high_gate=False, high_k=0.2, compare_gate=False, compare_k=0.2):
        super().__init__()
        self.ucb = UCB(ch_in, ch_out)
        self.dcb = DCB(ch_in, ch_out)
        # 对特征作过滤，只保留高频特征
        if high_gate:
            self.high_gate = HighFreGate(ch_out, high_k)

        # 将下级特征与上级特征作比对，仅保留差别比较大的特征
        if compare_gate:
            self.compare_gate = CompareGate(ch_out, compare_k)

    def forward(self, x_former, skip_x, up_x=None):
        # x_former为上一层传递出来的光源特征向量
        # skip_x为对应的跳跃连接的张量
        # up_x为上一分支传递出来的光源特征向量
        x_upsample = self.ucb(x_former)
        if hasattr(self, 'high_gate'):
            skip_x = self.high_gate(skip_x)
        x_cat = torch.cat((skip_x, x_upsample), dim=1)
        rlt = self.dcb(x_cat)
        if up_x != None and hasattr(self, 'compare_gate'):  # 若将上一分支的光源特征向量传递过来了，则将其作上采样，并从rlt向量中减去up_x（低频）
            h, w = rlt.shape[2:]
            up_x = F.interpolate(up_x, size=(h, w))  # 放大上层传递而来的特征图，保证尺寸的一致性
            rlt = self.compare_gate(rlt, up_x)
        return rlt

class U_Net(nn.Module):
    def __init__(self, BASE_CHANNEL, DEPTH, high_gate, high_k, compare_gate, compare_k, img_ch=3, output_ch=2):
        super(U_Net, self).__init__()
        self.DEPTH = DEPTH  # Unet深度，采用下采样块的数量来度量
        self.down_path = nn.ModuleList()  # 下采样路径
        self.up_path = nn.ModuleList()  # 上采样路径

        self.head = DCB(ch_in=img_ch, ch_out=BASE_CHANNEL)  # 用于对图像作初始特征提取

        # 下采样路径
        for i in range(DEPTH):
            self.down_path.append(Down_blk(BASE_CHANNEL * (2 ** i), BASE_CHANNEL * (2 ** (i + 1))))

        # 上采样路径
        for i in range(DEPTH):
            self.up_path.append(Up_blk(BASE_CHANNEL * (2 ** (DEPTH - i)), BASE_CHANNEL * (2 ** (DEPTH - i - 1)),
                                       high_gate, high_k, compare_gate, compare_k))

        self.tail = nn.Conv2d(BASE_CHANNEL, output_ch, kernel_size=1, stride=1, padding=0)  # 将特征通道维度的长度调整为2

    def forward(self, x, up_feature=None):
        # encoding path
        x = self.head(x)

        skip_tensor = [x]  # 存储从下采样路径跳跃到上采样路径上的张量
        up_out = []  # 存放上采样路径的输出张量

        for i in range(self.DEPTH):  # 下采样路径
            x = self.down_path[i](x)
            if i != self.DEPTH - 1:  # 最后一个下采样块的输出，不进行跳连
                skip_tensor.append(x)

        for i in range(self.DEPTH):  # 上采样路径,Up_blk
            if up_feature != None:  # 有上层分支传递特征张量
                x = self.up_path[i](x, skip_tensor[self.DEPTH - i - 1], up_feature[i])
            else:
                x = self.up_path[i](x, skip_tensor[self.DEPTH - i - 1])
            up_out.append(x)

        d1 = self.tail(x)
        return d1, up_out

# region 特征融合模块
# 1 直接叠加
class ADD(nn.Module):
    def __init__(self, hyper):
        super().__init__()
        self.hyper = hyper

    def forward(self, x):
        x = torch.stack((x))
        return torch.sum(x, dim=0)

# 2 逐像素加权
class AIFM(nn.Module):
    def __init__(self, hyper):
        super().__init__()
        self.hyper = hyper
        self.num = len(hyper['MODEL']['SCALE_FACTOR'])
        # 融合权重生成模块
        self.main = nn.Sequential(
            nn.Conv2d(self.num * 3, self.num, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_cat = torch.cat(tuple(x), dim=1)  # 将光照图在通道维度上拼接
        weight_map = self.main(x_cat)  # 求权重
        weigth_map = torch.repeat_interleave(weight_map, 3, dim=1)  # 逐元素复制
        weighted_x = x_cat * weigth_map  # 权重相乘
        weighted_x_chunks = torch.chunk(weighted_x, chunks=self.num, dim=1)  # 划分光照图
        rlt = torch.sum(torch.stack((weighted_x_chunks)), dim=0)  # 光照图叠加
        return rlt

# 3 小波变换融合模块
class DWTFM(nn.Module):
    def __init__(self, hyper):
        super().__init__()
        self.hyper = hyper
        self.dwt = DWT_2D('haar') # 作离散小波变换
        self.idwt = IDWT_2D('haar')
    
    def forward(self, x): # 接收相邻两个分支的光源估计结果
        low_dwt = self.dwt(x[0])
        high_dwt = self.dwt(x[1])
        low_ll, low_lh, low_hl, low_hh = low_dwt.split(3, 1)
        high_ll, high_lh, high_hl, high_hh = high_dwt.split(3, 1)
        out = torch.cat((low_ll, high_lh, high_hl, high_hh), 1)
        out_idwt = self.idwt(out)
        return out_idwt

# 4 快速傅里叶融合模块，在傅里叶域作融合
class FFTFM(nn.Module):
    def __init__(self, hyper):
        pass

    def forward(self, x): # 接收相邻两个分支的光源估计结果
        pass


# 最终的网络结构
class FinalNet(nn.Module):
    def __init__(self, hyper):
        super().__init__()
        self.hyper = hyper
        # 分支
        self.branches = nn.ModuleList()
        for idx, _ in enumerate(hyper['MODEL']['SCALE_FACTOR']):
            self.branches.append(U_Net(hyper['MODEL']['BASE_CHANNEL'],
                                       hyper['MODEL']['DEPTH'][idx],
                                       hyper['MODEL']['HIGH_FRE_GATE'][idx],
                                       hyper['MODEL']['HIGH_TOP_K'],
                                       hyper['MODEL']['COMPARE_GATE'][idx],
                                       hyper['MODEL']['COMPARE_TOP_K']))

        # 创建光照图融合模块
        if len(self.hyper['MODEL']['SCALE_FACTOR']) > 1:  # 当分支的数量大于的时候，才需要作融合
            if hyper['MODEL']['FUSION'] == 'ADD':
                self.fusion_module = getattr(sys.modules[__name__], hyper['MODEL']['FUSION'])(hyper)
            elif hyper['MODEL']['FUSION'] == 'AIFM':
                self.fusion_module = getattr(sys.modules[__name__], hyper['MODEL']['FUSION'])(hyper)
            elif hyper['MODEL']['FUSION'] == 'DWTFM':
                self.fusion_module = getattr(sys.modules[__name__], hyper['MODEL']['FUSION'])(hyper)
            else:
                pass

        self.device = hyper['TRAIN']['DEVICE']  # 运算设备

    def forward(self, x):
        # 计算各分支的估计值
        illu_map_all = []
        up_features = None  # 指示上一层中的光源分布特征
        for idx, branch in enumerate(self.branches):
            scale_factor = self.hyper['MODEL']['SCALE_FACTOR'][idx]  # 拿出缩放因子

            # 根据缩放因子对输入作缩放处理；如果缩放因子为1，则不处理
            if scale_factor == 1:
                rb_channel, up_features = branch(x, up_features)
            else:
                x_down_sample = F.avg_pool2d(x, kernel_size=int(1 / scale_factor), stride=int(1 / scale_factor))
                rb_channel, up_features = branch(x_down_sample, up_features)
                rb_channel = F.interpolate(rb_channel, size=self.hyper['TRAIN']['INPUT_SIZE'], mode='bilinear', align_corners=False)  # 将输出光照图上采样到原始尺寸
            illu_map_all.append(rb_channel)

        # 光照图补全绿通道
        if len(self.hyper['MODEL']['SCALE_FACTOR']) == 1:  # 若模型中只有一个分支的话，则绿通道设置为1
            g_channel = torch.ones(x.size(0), 1, self.hyper['TRAIN']['INPUT_SIZE'],
                                   self.hyper['TRAIN']['INPUT_SIZE']).to(self.device)  # 全1绿通道
            illu_map_all[0] = torch.cat((illu_map_all[0][:, 0:1, :, :], g_channel, illu_map_all[0][:, 1:2, :, :]),
                                        dim=1)  # 为分支的输出添加绿通道
        else:  # 否则，根据尺寸来设定绿通道，仅最小尺寸的绿通道为1，其余为0
            for idx, scale in enumerate(self.hyper['MODEL']['SCALE_FACTOR']):
                if scale > min(self.hyper['MODEL']['SCALE_FACTOR']):  # 大于最小尺寸的分支，绿通道都设置为0
                    g_channel = torch.zeros(x.size(0), 1, self.hyper['TRAIN']['INPUT_SIZE'],
                                            self.hyper['TRAIN']['INPUT_SIZE']).to(self.device)  # 全0绿通道
                else:  # 尺寸最小的分支，绿通道设置为1
                    g_channel = torch.ones(x.size(0), 1, self.hyper['TRAIN']['INPUT_SIZE'],
                                           self.hyper['TRAIN']['INPUT_SIZE']).to(self.device)  # 全1绿通道
                illu_map_all[idx] = torch.cat(
                    (illu_map_all[idx][:, 0:1, :, :], g_channel, illu_map_all[idx][:, 1:2, :, :]), dim=1)  # 分支的输出

        # 若存在多个分支，则进行光照融合
        if hasattr(self, "fusion_module"):
            out = self.fusion_module(illu_map_all)  # 光照图融合
        else:
            out = illu_map_all[0]  # 单分支时，就用这个分支的输出作为最终输出

        return out, illu_map_all

if __name__ == '__main__':
    with open('hyper_parameters.yaml', 'r', encoding='utf-8') as f:
        hyper = yaml.safe_load(f)
    model = FinalNet(hyper).to(hyper['TRAIN']['DEVICE'])
    # print(model.branches[0])
    # 生成一个大小为 256x256x3 的全1张量
    tensor1 = torch.ones((2, 3, 256, 256)).to(hyper['TRAIN']['DEVICE'])
    rlt = model(tensor1)
    print('over')
