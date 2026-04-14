import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.amp as amp

from data.frame_level_avec2014 import FrameLevelAVEC2014TrainDataset, FrameLevelAVEC2014EvalDataset
from models.spatial_dnet import SpatialDNet


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_batch_size: int = 64,
) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for full_seq, local_seq, mask, labels in loader:
            full_seq = full_seq.squeeze(0)
            local_seq = local_seq.squeeze(0)
            mask = mask.squeeze(0).bool()
            labels = labels.to(device, non_blocking=True)

            valid_idx = (~mask).nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue

            full_valid = full_seq[valid_idx]
            local_valid = local_seq[valid_idx]

            frame_preds = []
            for i in range(0, full_valid.size(0), eval_batch_size):
                f = full_valid[i : i + eval_batch_size].to(device, non_blocking=True)
                l = local_valid[i : i + eval_batch_size].to(device, non_blocking=True)
                with amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(f, l)
                frame_preds.append(logits.view(-1).cpu())

            if not frame_preds:
                continue

            frame_preds = torch.cat(frame_preds, dim=0)
            video_pred = frame_preds.mean().item()
            video_label = labels.view(-1).item()

            loss = criterion(torch.tensor([video_pred]), torch.tensor([video_label]))
            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()
            steps += 1

            all_preds.append(video_pred)
            all_labels.append(video_label)

    if steps == 0 or not all_preds:
        return float("inf"), float("inf"), float("inf")

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    return total_loss / steps, mae, rmse


def main() -> None:
    """从 best_spatial_dnet.pth 出发进行小学习率 fine-tune。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    preprocessed_root = os.path.join("data", "AVEC2014_preprocessed")
    train_dataset = FrameLevelAVEC2014TrainDataset(
        split="train",
        preprocessed_root=preprocessed_root,
        frames_per_video=16,
    )
    dev_dataset = FrameLevelAVEC2014EvalDataset(
        split="dev",
        preprocessed_root=preprocessed_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
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

    model = SpatialDNet(in_channels=3).to(device)
    base_ckpt = os.path.join("checkpoints", "best_spatial_dnet.pth")
    if not os.path.exists(base_ckpt):
        raise RuntimeError(f"Base checkpoint not found: {base_ckpt}. Please run frame_level_train.py first.")

    state = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"[FrameLevelFinetune] Loaded base model from {base_ckpt}")

    # 小学习率微调
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    criterion = nn.MSELoss()
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=5,
        verbose=True,
        min_lr=1e-6,
    )

    num_epochs = 40
    best_mae = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "best_spatial_dnet_ft.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        for full_frame, local_frame, labels in train_loader:
            full_frame = full_frame.to(device, non_blocking=True)
            local_frame = local_frame.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(full_frame, local_frame)
                loss = criterion(preds.view(-1), labels.view(-1))

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

        if steps == 0:
            print(f"[FrameLevelFinetune] Epoch {epoch:02d} | no valid training steps, skipping validation.")
            continue

        train_loss = running_loss / steps
        val_loss, val_mae, val_rmse = validate(model, dev_loader, device)

        print(
            f"[FrameLevelFinetune] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"[FrameLevelFinetune]  >>> New best SpatialDNet (ft) saved (MAE: {val_mae:.4f})")


if __name__ == "__main__":
    main()
