from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fem_fpn import DualFEMWithFPN


class TemporalEncoder(nn.Module):
    """2 层 Transformer Encoder 进行时序建模，并做时间池化。

    支持可选的梯度检查点，用于降低长序列时的激活显存占用。
    """

    def __init__(self, d_model: int = 320, nhead: int = 8, num_layers: int = 2, use_checkpoint: bool = False) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 320), mask: (B, T), True 表示 padding。"""

        if self.use_checkpoint and self.training and x.requires_grad:
            def _forward(inp: torch.Tensor) -> torch.Tensor:
                # mask 通过闭包捕获
                return self.transformer(inp, src_key_padding_mask=mask)

            # 显式使用非 reentrant 版本，避免未来版本报错，并配合新版 autocast 实现。
            out = checkpoint(_forward, x, use_reentrant=False)
        else:
            out = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        summed = (out * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return pooled


class ViTBlock(nn.Module):
    """简化版 ViT Block，对齐论文的关键超参数。

    依据论文描述：
    - token 序列长度约为 240
    - Multi-head Attention 使用 4 个头
    - FFN 中间层维度为 480

    这里将输入的 320 维特征投影为 num_tokens×token_dim 的 token 序列，
    经过若干层 TransformerEncoder 后再聚合回 320 维，用于局部-全局语义提纯。
    """

    def __init__(
        self,
        in_dim: int = 320,
        num_tokens: int = 240,
        token_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 480,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.use_checkpoint = use_checkpoint

        self.unfold_fc = nn.Linear(in_dim, num_tokens * token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fold_fc = nn.Linear(token_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C=in_dim) -> (B, C)"""
        b, c = x.shape
        tokens = self.unfold_fc(x)  # (B, N*D)
        tokens = tokens.view(b, self.num_tokens, self.token_dim)
        if self.use_checkpoint and self.training and tokens.requires_grad:
            def _forward(inp: torch.Tensor) -> torch.Tensor:
                return self.transformer(inp)

            # 显式使用非 reentrant 版本，避免未来版本报错，并配合新版 autocast 实现。
            tokens = checkpoint(_forward, tokens, use_reentrant=False)
        else:
            tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        out = self.fold_fc(pooled)
        return out


class RegressionHead(nn.Module):
    """1x1 卷积 + 全连接输出单一分数。"""

    def __init__(self, in_dim: int = 320, hidden_dim: int = 320) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C)"""
        b, c = x.shape
        feat = x.view(b, c, 1, 1)
        feat = self.conv1x1(feat)
        feat = F.relu(feat, inplace=True)
        feat = feat.view(b, -1)
        out = self.fc(feat)
        return out


class End2EndDepressionModel(nn.Module):
    """端到端抑郁回归模型：
    - DualFEMWithFPN: 空间特征抽取与多尺度融合（全脸 + 局部脸）
    - TemporalEncoder: 时序 Transformer
    - ViTBlock: 局部-全局语义提纯
    - RegressionHead: 输出单一分数
    """

    def __init__(
        self,
        temporal_chunks: int = 4,
        use_checkpoint_temporal: bool = True,
        use_checkpoint_vit: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_backbone = DualFEMWithFPN()
        # 将时序长度按 temporal_chunks 分段，仅对空间主干做分块前向，
        # 时序 Transformer 仍在完整序列上工作，保证与原始模型数学等价。
        self.temporal_chunks = max(1, temporal_chunks)
        self.temporal_encoder = TemporalEncoder(
            d_model=320,
            nhead=8,
            num_layers=2,
            use_checkpoint=use_checkpoint_temporal,
        )
        # ViT Block：使用论文中提到的 240 token、4 个头和 FFN 维度 480。
        self.vit_block = ViTBlock(
            in_dim=320,
            num_tokens=240,
            token_dim=64,
            num_heads=4,
            num_layers=1,
            ff_dim=480,
            use_checkpoint=use_checkpoint_vit,
        )
        self.head = RegressionHead(in_dim=320, hidden_dim=320)

    def forward(self, full_faces: torch.Tensor, local_faces: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """输入：
        - full_faces: (B, T, 3, H, W)
        - local_faces: (B, T, 3, H, W)
        - mask: (B, T)
        输出：
        - preds: (B, 1)
        """
        b, t, c, h, w = full_faces.shape

        # 按时间维对空间主干做分段前向：
        # full_faces/local_faces: (B, T, C, H, W) -> 若干段 (B, T_i, C, H, W)
        # 每段独立通过 DualFEMWithFPN 和池化，再在时间维拼回 (B, T, 320)。
        # 由于 FEM+FPN 是逐帧独立处理的，这样与一次性处理 (B*T, C, H, W) 数学等价，
        # 但峰值显存按 temporal_chunks 成比例降低。
        full_chunks = torch.chunk(full_faces, chunks=self.temporal_chunks, dim=1)
        local_chunks = torch.chunk(local_faces, chunks=self.temporal_chunks, dim=1)

        seq_feats = []
        for f_chunk, l_chunk in zip(full_chunks, local_chunks):
            bc, tc, _, _, _ = f_chunk.shape
            f_flat = f_chunk.reshape(bc * tc, c, h, w)
            l_flat = l_chunk.reshape(bc * tc, c, h, w)
            spatial_feats = self.spatial_backbone(f_flat, l_flat)  # (B*T_i, 320, H', W')
            pooled = F.adaptive_avg_pool2d(spatial_feats, output_size=(1, 1)).view(bc, tc, -1)
            seq_feats.append(pooled)

        spatial_pooled = torch.cat(seq_feats, dim=1)  # (B, T, 320)

        temporal_feat = self.temporal_encoder(spatial_pooled, mask)  # (B, 320)
        refined = self.vit_block(temporal_feat)  # (B, 320)
        preds = self.head(refined)  # (B, 1)
        return preds
