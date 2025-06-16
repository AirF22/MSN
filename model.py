import torch
import torch.nn as nn
import torch.nn.functional as F
import tool

# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练时
# DEVICE = torch.device("cpu")  # 调试时
MAX_CHANNEL = 512  # Unet的最大通道数
BASE_CHANNEL = int(MAX_CHANNEL/8)
# -----------------------------------------------------------------------------

# Unet部分

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=BASE_CHANNEL)
        self.Conv2 = conv_block(ch_in=BASE_CHANNEL, ch_out=BASE_CHANNEL*2)
        self.Conv3 = conv_block(ch_in=BASE_CHANNEL*2, ch_out=BASE_CHANNEL*4)
        self.Conv4 = conv_block(ch_in=BASE_CHANNEL*4, ch_out=BASE_CHANNEL*8)
        # self.Conv5 = conv_block(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*8)
        # self.Conv6 = conv_block(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*8)
        # self.Conv7 = conv_block(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*8)
        # self.Conv8 = conv_block(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*8)
        #
        # self.Up8 = up_conv(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)
        # self.Up_conv8 = conv_block(ch_in=BASE_CHANNEL*12, ch_out=BASE_CHANNEL*8)
        #
        # self.Up7 = up_conv(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)
        # self.Up_conv7 = conv_block(ch_in=BASE_CHANNEL*12, ch_out=BASE_CHANNEL*8)
        #
        # self.Up6 = up_conv(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)
        # self.Up_conv6 = conv_block(ch_in=BASE_CHANNEL*12, ch_out=BASE_CHANNEL*8)
        #
        # self.Up5 = up_conv(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)
        # self.Up_conv5 = conv_block(ch_in=BASE_CHANNEL*12, ch_out=BASE_CHANNEL*8)

        self.Up4 = up_conv(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)
        self.Up_conv4 = conv_block(ch_in=BASE_CHANNEL*8, ch_out=BASE_CHANNEL*4)

        self.Up3 = up_conv(ch_in=BASE_CHANNEL*4, ch_out=BASE_CHANNEL*2)
        self.Up_conv3 = conv_block(ch_in=BASE_CHANNEL*4, ch_out=BASE_CHANNEL*2)

        self.Up2 = up_conv(ch_in=BASE_CHANNEL*2, ch_out=BASE_CHANNEL)
        self.Up_conv2 = conv_block(ch_in=BASE_CHANNEL*2, ch_out=BASE_CHANNEL)

        self.Conv_1x1 = nn.Conv2d(BASE_CHANNEL, output_ch, kernel_size=1, stride=1, padding=0)

    def print_grad(self):
        print(self.Conv1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # x6 = self.Maxpool(x5)
        # x6 = self.Conv6(x6)

        # x7 = self.Maxpool(x6)
        # x7 = self.Conv7(x7)
        #
        # x8 = self.Maxpool(x7)
        # x8 = self.Conv8(x8)
        #
        # # decoding + concat path
        # d8 = self.Up8(x8)
        # d8 = torch.cat((x7, d8), dim=1)
        # d8 = self.Up_conv8(d8)
        #
        # d7 = self.Up7(d8)
        # d7 = torch.cat((x6, d7), dim=1)
        # d7 = self.Up_conv7(d7)

        # d6 = self.Up6(x6)
        # d6 = torch.cat((x5, d6), dim=1)
        # d6 = self.Up_conv6(d6)
        #
        # d5 = self.Up5(d6)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class FinalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.U256 = U_Net()
        self.U128 = U_Net()
        self.U64 = U_Net()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(4, 4)
        self.tvloss = tool.TotalVariationLoss()
        self.struct_weight = nn.Sequential(
            nn.Conv2d(6, 3, 1, padding=0),
            nn.ReLU(inplace=False),
        )


    def forward(self, x):
        # 像素特征(256)
        pixel_level = self.U256(x)  # 2通道
        pixel_green_channel = torch.ones(pixel_level.size(0), 1, pixel_level.size(2), pixel_level.size(3)).to(DEVICE)
        pixel_red_channel = pixel_level[:, 0:1, :, :]
        pixel_blue_channel = pixel_level[:, 1:2, :, :]
        pixel_illu = torch.cat((pixel_red_channel, pixel_green_channel, pixel_blue_channel), dim=1)

        # 块级特征(128)
        patch_x = self.pool1(x)  # 下采样
        patch_y = self.U128(patch_x)  # 2通道
        patch_level = F.interpolate(patch_y, size=(256, 256), mode='bilinear', align_corners=False)  # 上采样
        patch_green_channel = torch.ones(patch_level.size(0), 1, patch_level.size(2), patch_level.size(3)).to(DEVICE)
        patch_red_channel = patch_level[:, 0:1, :, :]
        patch_blue_channel = patch_level[:, 1:2, :, :]
        patch_illu = torch.cat((patch_red_channel, patch_green_channel, patch_blue_channel), dim=1)

        # 全局特征(64)
        overall_x = self.pool2(x)
        overall_y = self.U64(overall_x)  # 2通道
        overall_level = F.interpolate(overall_y, size=(256, 256), mode='bilinear', align_corners=False)  # 上采样
        overall_green_channel = torch.ones(overall_level.size(0), 1, overall_level.size(2), overall_level.size(3)).to(DEVICE)
        overall_red_channel = overall_level[:, 0:1, :, :]
        overall_blue_channel = overall_level[:, 1:2, :, :]
        overall_illu = torch.cat((overall_red_channel, overall_green_channel, overall_blue_channel), dim=1)

        # 计算权重图
        all_tensor = torch.cat((overall_level, patch_level, pixel_level), dim=1)
        weight_map = self.struct_weight(all_tensor)
        real_weight_map = F.softmax(weight_map, dim=1)

        # 对张量赋权重
        overall_illu_weighted = overall_illu * real_weight_map[:, 0:1, :, :]
        patch_illu_weighted = patch_illu * real_weight_map[:, 1:2, :, :]
        pixel_illu_weighted = pixel_illu * real_weight_map[:, 2:3, :, :]

        out_recon = overall_illu_weighted + patch_illu_weighted + pixel_illu_weighted
        return out_recon, pixel_illu, patch_illu, overall_illu, weight_map


if __name__ == "__main__":
    tensor1 = torch.ones((5, 3, 256, 256))
    model = FinalNet()
    rlt = model(tensor1)
    print("Tensor 1 shape:", tensor1.shape)
    print("rlt:", rlt)
