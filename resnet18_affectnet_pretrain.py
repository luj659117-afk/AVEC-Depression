import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.amp as amp
from torchvision import models

from data.affectnet_dataset import AffectNetExpressionDataset


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
    """使用 torchvision ResNet18 在 AffectNet 7 类子集上做表情分类 sanity check。

    目的：验证数据加载/裁剪/标签是否正常，以及在标准 backbone 下 Val Acc 大致能到什么水平。
    这不会改动你 DNet 的 FEM，只是一个独立检查脚本。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    data_root = os.path.join("data", "AffectNet")
    train_csv = os.path.join(data_root, "training.csv")
    val_csv = os.path.join(data_root, "validation.csv")

    image_size = 256
    num_classes = 7

    # 数据集（重用之前的 AffectNetExpressionDataset，带有 bbox 裁剪和增强）
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
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # 使用 ImageNet 预训练的 ResNet18
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    model = backbone.to(device)

    # 类不平衡权重（与 FEM 预训练一致）
    class_counts = torch.tensor(train_dataset.class_counts, dtype=torch.float32)
    class_counts = torch.clamp(class_counts, min=1.0)
    inv_freq = 1.0 / class_counts
    class_weights = inv_freq / inv_freq.sum() * num_classes
    print(f"[ResNet18] Class counts: {train_dataset.class_counts}")
    print(f"[ResNet18] Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    num_epochs = 5  # sanity check 用，先跑 5 轮看趋势
    best_acc = 0.0

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

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy().tolist())
                train_labels.extend(labels.cpu().numpy().tolist())

        if steps == 0:
            print(f"[ResNet18] Epoch {epoch:02d} | no valid training steps, skipping validation.")
            continue

        train_loss = running_loss / steps
        train_acc = accuracy_score(train_labels, train_preds) if train_preds else 0.0
        val_loss, val_acc = validate(model, val_loader, device)

        print(
            f"[ResNet18] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join("checkpoints", "resnet18_affectnet_best.pth"))
            print(f"[ResNet18]  >>> New best model saved (Val Acc: {val_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
