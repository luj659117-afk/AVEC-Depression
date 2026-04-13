import torch
import torch.nn as nn
import torch.nn.functional as F

from .fem_fpn import DualFEMWithFPN


class SpatialViTBlock(nn.Module):
    """基于 2D 特征图的 ViT Block，实现 Local 分支 + Global 分支 (Unfold-Transformer-Fold)。

    设计对齐论文中的图像块建模思路：
    - Local 分支：3x3 Conv + BN + ReLU + 1x1 Conv；
    - Global 分支：将特征图按 patch_size 分块，线性降维到 token_dim，经 Transformer 编码，再映射回原始 patch 维度并 Fold 回 2D；
    - 融合：local_feat 与 global_feat 相加后与 identity 在通道维拼接，经 1x1 Conv 降回原通道，并与 identity 做残差相加。
    """

    def __init__(
        self,
        in_channels: int = 320,
        num_heads: int = 4,
        ff_dim: int = 480,
        token_dim: int = 64,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Local branch
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

        # Global branch
        patch_dim = in_channels * patch_size * patch_size
        self.unfold_proj = nn.Linear(patch_dim, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fold_proj = nn.Linear(token_dim, patch_dim)
        self.global_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Fusion
        self.fusion_conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: (B, C, H, W)，输出同尺寸特征 (B, C, H, W)。"""
        b, c, h, w = x.shape
        identity = x

        # 1. Local Branch
        local_feat = self.local_conv(x)

        # 2. Global Branch: Unfold -> Proj -> Transformer -> Proj -> Fold
        # 保证 H 和 W 能被 patch_size 整除；若不能，使用 interpolate 进行对齐。
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            new_h = (h // self.patch_size) * self.patch_size
            new_w = (w // self.patch_size) * self.patch_size
            new_h = max(self.patch_size, new_h)
            new_w = max(self.patch_size, new_w)
            x_resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            x_resized = x
            new_h, new_w = h, w

        unfolded = F.unfold(x_resized, kernel_size=self.patch_size, stride=self.patch_size)  # (B, C*P*P, L)
        tokens = unfolded.transpose(1, 2)  # (B, L, C*P*P)

        tokens = self.unfold_proj(tokens)  # (B, L, token_dim)
        tokens = self.transformer(tokens)  # (B, L, token_dim)
        tokens = self.fold_proj(tokens)    # (B, L, C*P*P)

        folded = tokens.transpose(1, 2)  # (B, C*P*P, L)
        global_feat = F.fold(
            folded,
            output_size=(new_h, new_w),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # (B, C, new_h, new_w)
        global_feat = self.global_proj(global_feat)

        if new_h != h or new_w != w:
            global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        # 3. Fusion & Shortcut
        fused_lg = local_feat + global_feat
        concat_feat = torch.cat([identity, fused_lg], dim=1)  # (B, 2C, H, W)
        out = self.fusion_conv1(concat_feat)  # (B, C, H, W)
        out = out + identity
        return out


class SpatialDNet(nn.Module):
    """仅进行空间建模的 DNet 变体：

    输入：单帧全脸 / 局部脸图像对 (B, 3, H, W)
    流程：DualFEMWithFPN -> SpatialViTBlock -> GAP -> 回归头
    输出：每帧一个抑郁评分 (B, 1)
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # 现有 DualFEMWithFPN 不接受 in_channels 参数，默认以 3 通道 RGB 输入。
        # 这里保留 in_channels 形参只是为了接口对齐，暂不透传。
        self.spatial_backbone = DualFEMWithFPN()
        self.vit_block = SpatialViTBlock(in_channels=320)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(320, 1),
        )

    def forward(self, full_face: torch.Tensor, local_face: torch.Tensor) -> torch.Tensor:
        """full_face/local_face: (B, 3, H, W)"""
        feats = self.spatial_backbone(full_face, local_face)  # (B, 320, H', W')
        feats = self.vit_block(feats)                         # (B, 320, H', W')
        pooled = self.gap(feats)                              # (B, 320, 1, 1)
        preds = self.head(pooled)                             # (B, 1)
        return preds
