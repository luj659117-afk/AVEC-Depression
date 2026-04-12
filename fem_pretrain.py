import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.amp as amp

from data.affectnet_dataset import AffectNetExpressionDataset
from models.fem_fpn import FEMBackbone


class FEMExpressionNet(nn.Module):
    """使用 FEMBackbone 做特征提取的表情分类网络。

    - Backbone: FEMBackbone（与 DNet 中单路 FEM 结构一致）
    - Head: 全局平均池化 + 全连接分类

    预训练完成后，只需将 backbone.state_dict() 加载到 DNet 的 FEM 上。
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.backbone = FEMBackbone(in_channels=3)
        # 使用最后一层特征图 (B, 160, H, W) 做全局池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # feats: [t1, t2, t3, t4]，选用 t4 作为高层特征
        x = feats[-1]  # (B, 160, H, W)
        x = self.gap(x).view(x.size(0), -1)  # (B, 160)
        logits = self.fc(x)
        return logits


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()
            steps += 1

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if steps == 0 or len(all_preds) == 0:
        return float("inf"), 0.0

    avg_loss = total_loss / steps
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def main() -> None:
    """在 AffectNet 上预训练 FEMBackbone 做表情分类。

    假设数据已解压到 data/AffectNet 下，且存在：
    - data/AffectNet/training.csv
    - data/AffectNet/validation.csv

    预训练完成后，将在 checkpoints/fem_backbone_affectnet.pth 中保存 backbone 的权重。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    data_root = os.path.join("data", "AffectNet")
    train_csv = os.path.join(data_root, "training.csv")
    val_csv = os.path.join(data_root, "validation.csv")

    image_size = 256
    num_classes = 7  # 使用 expression 0~6

    # 数据集与 DataLoader
    train_dataset = AffectNetExpressionDataset(
        root_dir=data_root,
        csv_path=train_csv,
        image_size=image_size,
        num_classes=num_classes,
        is_train=True,
    )
    val_dataset = AffectNetExpressionDataset(
        root_dir=data_root,
        csv_path=val_csv,
        image_size=image_size,
        num_classes=num_classes,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # 模型与优化器
    model = FEMExpressionNet(num_classes=num_classes).to(device)

    # 更激进一点的学习率，有助于从随机初始化尽快收敛
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    # 根据训练集类别分布构造 class weight，缓解严重类不平衡
    class_counts = torch.tensor(train_dataset.class_counts, dtype=torch.float32)
    # 避免除零
    class_counts = torch.clamp(class_counts, min=1.0)
    inv_freq = 1.0 / class_counts
    class_weights = inv_freq / inv_freq.sum() * num_classes
    print(f"[FEM pretrain] Class counts: {train_dataset.class_counts}")
    print(f"[FEM pretrain] Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    # 10 轮对于这么大的 FEM 来说偏少，适当增加到 30 轮
    num_epochs = 30
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    backbone_ckpt_path = os.path.join("checkpoints", "fem_backbone_affectnet.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            steps += 1

            # 记录训练集预测用于估算 train accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy().tolist())
                train_labels.extend(labels.cpu().numpy().tolist())

        if steps == 0:
            print(f"[FEM pretrain] Epoch {epoch:02d} | no valid training steps, skipping validation.")
            continue

        train_loss = running_loss / steps
        train_acc = accuracy_score(train_labels, train_preds) if train_preds else 0.0
        val_loss, val_acc = validate(model, val_loader, device)

        print(
            f"[FEM pretrain] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            # 只保存 backbone 参数，方便在 DNet 中复用
            backbone_state = model.backbone.state_dict()
            torch.save(backbone_state, backbone_ckpt_path)
            print(
                f"[FEM pretrain]  >>> New best backbone saved to {backbone_ckpt_path} "
                f"(Val Acc: {val_acc*100:.2f}%)"
            )


if __name__ == "__main__":
    main()
