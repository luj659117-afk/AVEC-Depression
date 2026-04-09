import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CoordAtt(nn.Module):
    """Coordinate Attention 模块的简化实现。"""

    def __init__(self, inp: int, reduction: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class ConvBlock(nn.Module):
    """带 Coordinate Attention 的残差块，对应论文中的 Conv Block。

    结构：
    - 3x3 Conv -> BN -> ReLU
    - 3x3 Conv -> BN
    - CoordAtt
    - 残差连接（如通道/步幅不一致则使用 1x1 Conv 对齐）
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.att = CoordAtt(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class InBlock(nn.Module):
    """内部残差块，对应论文中的 In Block，保持分辨率和通道数不变。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.att = CoordAtt(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out)

        out = out + identity
        out = self.act(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
            channels += growth_rate

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class FEMBackbone(nn.Module):
    """DNet 风格的 FEM 主干网络，对齐论文 Table 1。

    结构概览（单路）：
    - Stem: 7x7 Conv, 32 -> MaxPool (输出 32x64x64)
    - Stage1: ConvBlock(32->64, stride=1) + InBlock×2 + DenseBlock(64->320) + Transition(320->64)
    - Stage2: ConvBlock(64->96, stride=2) + InBlock×3 + DenseBlock(96->352) + Transition(352->96)
    - Stage3: ConvBlock(96->128, stride=2) + InBlock×5 + DenseBlock(128->384) + Transition(384->128)
    - Stage4: ConvBlock(128->160, stride=2) + InBlock×2 + DenseBlock(160->416) + Transition(416->160)

    返回四个尺度的特征图：[64, 96, 128, 160] 通道，对应 64x64, 32x32, 16x16, 8x8。
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        base_channels = 32

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage1: 64x64, 通道 64
        self.stage1_conv = ConvBlock(base_channels, 64, stride=1)
        self.stage1_in = nn.Sequential(InBlock(64), InBlock(64))
        self.stage1_dense = DenseBlock(64, growth_rate=32, num_layers=8)
        self.stage1_trans = nn.Conv2d(self.stage1_dense.out_channels, 64, kernel_size=1, stride=1, padding=0)

        # Stage2: 32x32, 通道 96
        self.stage2_conv = ConvBlock(64, 96, stride=2)
        self.stage2_in = nn.Sequential(InBlock(96), InBlock(96), InBlock(96))
        self.stage2_dense = DenseBlock(96, growth_rate=32, num_layers=8)
        self.stage2_trans = nn.Conv2d(self.stage2_dense.out_channels, 96, kernel_size=1, stride=1, padding=0)

        # Stage3: 16x16, 通道 128
        self.stage3_conv = ConvBlock(96, 128, stride=2)
        self.stage3_in = nn.Sequential(
            InBlock(128),
            InBlock(128),
            InBlock(128),
            InBlock(128),
            InBlock(128),
        )
        self.stage3_dense = DenseBlock(128, growth_rate=32, num_layers=8)
        self.stage3_trans = nn.Conv2d(self.stage3_dense.out_channels, 128, kernel_size=1, stride=1, padding=0)

        # Stage4: 8x8, 通道 160
        self.stage4_conv = ConvBlock(128, 160, stride=2)
        self.stage4_in = nn.Sequential(InBlock(160), InBlock(160))
        self.stage4_dense = DenseBlock(160, growth_rate=32, num_layers=8)
        self.stage4_trans = nn.Conv2d(self.stage4_dense.out_channels, 160, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)

        # Stage1
        s1 = self.stage1_conv(x)
        s1 = self.stage1_in(s1)
        d1 = self.stage1_dense(s1)
        t1 = self.stage1_trans(d1)  # (B, 64, 64, 64)

        # Stage2
        s2 = self.stage2_conv(t1)
        s2 = self.stage2_in(s2)
        d2 = self.stage2_dense(s2)
        t2 = self.stage2_trans(d2)  # (B, 96, 32, 32)

        # Stage3
        s3 = self.stage3_conv(t2)
        s3 = self.stage3_in(s3)
        d3 = self.stage3_dense(s3)
        t3 = self.stage3_trans(d3)  # (B, 128, 16, 16)

        # Stage4
        s4 = self.stage4_conv(t3)
        s4 = self.stage4_in(s4)
        d4 = self.stage4_dense(s4)
        t4 = self.stage4_trans(d4)  # (B, 160, 8, 8)

        # 返回多尺度特征用于 FPN（按从高到低分辨率排序）
        return [t1, t2, t3, t4]


class FPN(nn.Module):
    """FPN：融合四个尺度的特征，对齐论文中 FPN(channels)=[128,192,256,320] 的设定。

    输入：来自双路 FEM 拼接后的四个尺度特征，通道数分别为 128, 192, 256, 320。
    输出：单尺度 320 通道特征图（最高分辨率，约 64x64），供后续时序建模与 ViT Block 使用。
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 320) -> None:
        super().__init__()
        assert len(in_channels_list) == 4
        c1_in, c2_in, c3_in, c4_in = in_channels_list

        self.lateral4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2_in, out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(c1_in, out_channels, kernel_size=1)

        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode="nearest") + y

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = feats

        # 自顶向下融合
        p4 = self.lateral4(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p1 = self._upsample_add(p2, self.lateral1(c1))

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # 按论文描述，选用最后一层（最高分辨率）的融合特征作为输出
        return p1


class DualFEMWithFPN(nn.Module):
    """双路 FEM + FPN：
    - 分别处理全脸和局部脸
    - 在每一尺度上通道拼接后送入 FPN
    - 最终输出 320 通道特征，并在外部做池化与时序建模
    """

    def __init__(self) -> None:
        super().__init__()
        self.full_fem = FEMBackbone()
        self.local_fem = FEMBackbone()

        # DNet 结构：单路 FEM 输出 [64, 96, 128, 160]，
        # 双路拼接后为 [128, 192, 256, 320]，对应论文 Table 1 中 FPN(channels)
        self.fpn = FPN(in_channels_list=[128, 192, 256, 320], out_channels=320)

    def forward(self, full_imgs: torch.Tensor, local_imgs: torch.Tensor) -> torch.Tensor:
        """输入形状：
        - full_imgs: (B*T, 3, 256, 256)
        - local_imgs: (B*T, 3, 256, 256)
        输出：
        - fused: (B*T, 320, H, W)
        """
        full_feats = self.full_fem(full_imgs)
        local_feats = self.local_fem(local_imgs)

        fused_feats: List[torch.Tensor] = []
        for f, l in zip(full_feats, local_feats):
            fused_feats.append(torch.cat([f, l], dim=1))

        fused = self.fpn(fused_feats)
        return fused
