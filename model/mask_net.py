import torch
import torch.nn.functional as F
from torch import nn, Tensor
from thop import profile
from typing import Optional, List
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

## 3*3的卷积核
class Conv3_3(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Conv3_3, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

## 空洞卷积
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, num=None, dilation=None):
        super(Conv, self).__init__(
            # 卷积后面有BN,会被BN把bias抵消所以不设置
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            DropBlock(7, num),
            nn.LeakyReLU(0.1, inplace=True)
        )

class DoubleConv(nn.Module):  # 576 288
    def __init__(self, in_channels, out_channels, num=None, dilation=None):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv(in_channels, out_channels, num, dilation)
        self.conv1 = Conv(out_channels, out_channels, num, dilation)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv1(x1)
        out = x2 + x1
        out = self.relu(out)
        return out


class JRDM(nn.Module):
    def __init__(self, C, size):  # C -> channel, size -> patch size
        super(JRDM, self).__init__()
        self.ker_size = size
        # Unfold 操作：将图像分割为 patch
        self.unfold1 = nn.Unfold(kernel_size=(self.ker_size, self.ker_size), stride=(self.ker_size, self.ker_size))
        # 第一个卷积层：将展开的 patch 映射回原始通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=C * size * size, out_channels=C, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(num_features=C),
            # nn.ReLU(inplace=True),
        )
        # 第二个卷积层：融合特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=C * 2, out_channels=C, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=C),
            nn.ReLU(inplace=True),
        )
        # 最大池化层：下采样
        self.Maxpool = nn.Conv2d(C, C, 3, stride=2, padding=1)

    def forward(self, x):
        # 最大池化下采样
        x1 = self.Maxpool(x)
        size = self.ker_size
        # 获取输入特征图的形状
        [B, C, H, W] = x.shape
        # Unfold 操作：将图像分割为 patch
        xu1 = self.unfold1(x)
        # 调整 Unfold 结果的形状
        xu1 = xu1.reshape(B, C * size * size, H // size, W // size)
        # 第一个卷积操作
        xu1 = self.conv1(xu1)

        # 因为特征不匹配
        # 调整 xu1 的尺寸以匹配 x1
        H1, W1 = x1.shape[2:]
        if xu1.shape[2:] != (H1, W1):
            xu1 = nn.functional.interpolate(xu1, size=(H1, W1), mode='bilinear', align_corners=False)

        # 拼接特征图
        xu1 = torch.cat((xu1, x1), dim=1)
        # 第二个卷积操作
        xu1 = self.conv2(xu1)
        return xu1

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )

class simple_attn(nn.Module):
    # midc=128  heads=16
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads  # 8
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 2 * midc, 1)
        self.o_proj1 = nn.Linear(midc, midc)
        self.o_proj2 = nn.Linear(midc, midc)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, q, x):
        B, C, H, W = x.shape
        Bq, N, Cq = q.shape
        assert B == Bq and C == Cq, "Channel mismatch between x and q"

        kv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 2 * self.headc)
        kv = kv.permute(0, 2, 1, 3)  # (B, heads, HW, 2*headc)
        k, v = kv.chunk(2, dim=-1)  # (B, heads, HW, headc)

        k = self.kln(k)
        v = self.vln(v)

        q = q.reshape(B, N, self.heads, self.headc).permute(0, 2, 1, 3)  # (B, heads, N, headc)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.headc ** 0.5)  # (B, heads, N, HW)
        v = torch.matmul(attn, v)  # (B, heads, N, headc)

        v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        out = v + q.reshape(B, N, C)  # 用 reshape 恢复原始 query 做残差

        z = self.o_proj1(out)
        out = self.o_proj2(self.act(z)) + out
        return out

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # print(h.shape)
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h
class FRD_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 base_c: int = 64,
                 num=0.9):
        super(FRD_Net, self).__init__()

        self.conv1 = Conv3_3(in_channels, base_c)
        self.conv1_1 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.conv1_2 = DoubleConv(base_c, base_c, num=num, dilation=1)

        self.conv2_1 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.conv2_2 = DoubleConv(base_c, base_c, num=num, dilation=1)

        self.conv3_1 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.conv3_2 = DoubleConv(base_c, base_c, num=num, dilation=1)

        self.conv4_1 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.conv4_2 = DoubleConv(base_c, base_c, num=num, dilation=1)

        self.decoder3 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.decoder2 = DoubleConv(base_c, base_c, num=num, dilation=1)
        self.decoder1_1 = DoubleConv(base_c*2, base_c, num=num, dilation=1)
        self.decoder1_2 = DoubleConv(base_c, base_c, num=num, dilation=1)

        self.GRU3 = SepConvGRU(base_c, base_c)
        self.GRU2 = SepConvGRU(base_c, base_c)
        # self.GRU1 = SepConvGRU(base_c, base_c)

        self.jrdm1 = JRDM(base_c, 2)
        self.jrdm2 = JRDM(base_c, 2)
        self.jrdm3 = JRDM(base_c, 2)

        self.outconv = OutConv(base_c, num_classes)


    def forward(self, x):
        b, c, h, w = x.shape

        # 第一层
        x = self.conv1(x)
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        t1 = out
        out = self.jrdm1(out)

        # 第二层
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        t2 = out  # b, c1, H/2, W/2
        out = self.jrdm2(out)

        # 第三层
        out = self.conv3_1(out)  # b, c5, H/4, W4
        out = self.conv3_2(out)  # b, c4, H/4, W/4
        t3 = out  # b, c1, H/2, W/2
        out = self.jrdm3(out)

        # 第四层
        out = self.conv4_1(out)  # b, c5, H/4, W4
        out = self.conv4_2(out)  # b, c4, H/4, W/4
        out = F.interpolate(out, size=t3.shape[2:], mode='bilinear', align_corners=False)

        # 第三层
        out=self.GRU3(out,t3)
        out = self.decoder3(out)
        out = F.interpolate(out, size=t2.shape[2:], mode='bilinear', align_corners=False)

        # 第二层
        out=self.GRU2(out,t2)
        out = self.decoder2(out)
        out = F.interpolate(out, size=t1.shape[2:], mode='bilinear', align_corners=False)

        # 第一层
        out = torch.cat((out, t1), dim=1)  # 示例：按通道维度拼接
        out=self.decoder1_1(out)
        out = self.decoder1_2(out)
        out = self.outconv(out)
        return out
# 3层

if __name__ == "__main__":
    model = FRD_Net()
    input = torch.randn(4, 3, 400, 400)  # .to(device)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
