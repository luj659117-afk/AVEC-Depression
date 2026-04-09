import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.amp as amp

from data.end2end_dataset import End2EndAVEC2014Dataset, PreprocessedAVEC2014Dataset
from models.temporal_vit import End2EndDepressionModel


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for full_faces, local_faces, mask, labels in loader:
            full_faces = full_faces.to(device, non_blocking=True)
            local_faces = local_faces.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 验证阶段也使用 autocast 以降低显存占用
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(full_faces, local_faces, mask)
            loss = criterion(preds.view(-1), labels.view(-1))

            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()
            steps += 1
            all_preds.extend(preds.view(-1).cpu().numpy().tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    if steps == 0 or len(all_preds) == 0:
        return float("inf"), float("inf"), float("inf")

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    return total_loss / steps, mae, rmse


def main() -> None:
    """从已有 best_dnet_model.pth 出发做小学习率微调。

    - 单卡 GPU 训练
    - 使用预处理好的 AVEC2014 数据（若存在）
    - 学习率固定为 1e-5，只训练 20 个 epoch
    - 将更优的模型保存为 checkpoints/best_dnet_model_ft.pth
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0

    # 数据：优先使用预处理版本
    max_frames = 48
    image_size = 256
    preprocessed_root = os.path.join("data", "AVEC2014_preprocessed")

    use_preprocessed = False
    preprocessed_train_dir = os.path.join(preprocessed_root, "train")
    if os.path.isdir(preprocessed_train_dir) and any(os.scandir(preprocessed_train_dir)):
        use_preprocessed = True

    if use_preprocessed:
        print(f"[finetune] Using preprocessed AVEC2014 tensors from '{preprocessed_root}'")
        train_dataset = PreprocessedAVEC2014Dataset(split="train", preprocessed_root=preprocessed_root)
        dev_dataset = PreprocessedAVEC2014Dataset(split="dev", preprocessed_root=preprocessed_root)
    else:
        print("[finetune] Preprocessed data not found, falling back to online MTCNN preprocessing from raw videos.")
        train_dataset = End2EndAVEC2014Dataset(split="train", device="cpu", max_frames=max_frames, image_size=image_size)
        dev_dataset = End2EndAVEC2014Dataset(split="dev", device="cpu", max_frames=max_frames, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 模型：启用 checkpoint，保持与训练脚本一致
    visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_ckpt = (device.type == "cuda") and (visible_gpus >= 1)
    model = End2EndDepressionModel(
        temporal_chunks=4,
        use_checkpoint_temporal=use_ckpt,
        use_checkpoint_vit=use_ckpt,
    ).to(device)

    ckpt_path = os.path.join("checkpoints", "best_dnet_model.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"[finetune] Cannot find base checkpoint: {ckpt_path}")

    print(f"[finetune] Loading base weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # 小学习率微调
    lr = 1e-5
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    criterion = nn.MSELoss()
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    num_epochs = 20
    best_mae = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    out_path = os.path.join("checkpoints", "best_dnet_model_ft.pth")

    print(f"[finetune] Start finetuning for {num_epochs} epochs on {device}, lr={lr}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        optimizer.zero_grad()
        accumulation_steps = 4
        accum_count = 0

        for batch_idx, (full_faces, local_faces, mask, labels) in enumerate(train_loader):
            full_faces = full_faces.to(device, non_blocking=True)
            local_faces = local_faces.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(full_faces, local_faces, mask)
                loss = criterion(preds.view(-1), labels.view(-1))

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss / accumulation_steps).backward()
            accum_count += 1

            running_loss += loss.item()
            steps += 1

            if accum_count % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

        if accum_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if steps == 0:
            print(f"[finetune] Epoch {epoch:02d} | no valid training steps, skipping validation.")
            continue

        train_loss = running_loss / steps
        val_loss, val_mae, val_rmse = validate(model, dev_loader, criterion, device)

        print(
            f"[finetune] Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), out_path)
            print(f"[finetune]  >>> New best finetuned model saved to {out_path} (MAE: {val_mae:.4f})")


if __name__ == "__main__":
    main()
