import argparse
import math
import os
from typing import List, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF

from data.end2end_dataset import PreprocessedAVEC2014Dataset
from data.frame_level_avec2014 import FrameLevelAVEC2014EvalDataset
from models.spatial_dnet import SpatialDNet


class VideoFrameBagDataset(Dataset):
    """Video-level training dataset backed by preprocessed frame tensors.

    Each item samples K valid frames from one video. The model predicts each frame,
    averages the K frame scores, and optimizes the video-level label.
    """

    def __init__(
        self,
        split: str = "train",
        preprocessed_root: str = os.path.join("data", "AVEC2014_preprocessed_uniform96"),
        frames_per_video: int = 8,
        augment: bool = True,
    ) -> None:
        assert split in {"train", "dev"}
        self.split = split
        self.base = PreprocessedAVEC2014Dataset(split=split, preprocessed_root=preprocessed_root)
        self.frames_per_video = max(1, frames_per_video)
        self.augment = augment and split == "train"

        self.labels: List[float] = []
        for idx in range(len(self.base)):
            _, _, _, label = self.base[idx]
            self.labels.append(float(label.view(-1).item()))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_seq, local_seq, mask, label = self.base[idx]
        mask = mask.bool()

        valid_indices = (~mask).nonzero(as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            chosen = torch.zeros(self.frames_per_video, dtype=torch.long)
        elif valid_indices.numel() >= self.frames_per_video:
            perm = torch.randperm(valid_indices.numel())[: self.frames_per_video]
            chosen = valid_indices[perm].sort().values
        else:
            extra = torch.randint(0, valid_indices.numel(), (self.frames_per_video - valid_indices.numel(),))
            chosen = torch.cat([valid_indices, valid_indices[extra]], dim=0).sort().values

        full_frames = full_seq[chosen].clone()
        local_frames = local_seq[chosen].clone()

        if self.augment:
            full_frames, local_frames = self._augment(full_frames, local_frames)

        return full_frames, local_frames, label.view(-1)

    def _augment(self, full_frames: torch.Tensor, local_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply paired transforms to full/local frames so both views stay aligned.
        if torch.rand(1).item() < 0.5:
            full_frames = TF.hflip(full_frames)
            local_frames = TF.hflip(local_frames)

        if torch.rand(1).item() < 0.3:
            factor = 1.0 + 0.1 * (2.0 * torch.rand(1).item() - 1.0)
            full_frames = full_frames * factor
            local_frames = local_frames * factor

        return full_frames, local_frames


def make_balanced_sampler(labels: List[float]) -> WeightedRandomSampler:
    """Build a coarse BDI-bin balanced sampler."""

    bins = []
    for label in labels:
        if label <= 10:
            bins.append(0)
        elif label <= 16:
            bins.append(1)
        elif label <= 20:
            bins.append(2)
        elif label <= 30:
            bins.append(3)
        else:
            bins.append(4)

    counts = torch.bincount(torch.tensor(bins, dtype=torch.long), minlength=5).float()
    weights = torch.tensor([1.0 / counts[b].clamp(min=1.0).item() for b in bins], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


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

    for full_seq, local_seq, mask, labels in loader:
        full_seq = full_seq.squeeze(0)
        local_seq = local_seq.squeeze(0)
        mask = mask.squeeze(0).bool()
        labels = labels.to(device, non_blocking=True)

        valid_idx = (~mask).nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            continue

        frame_preds = []
        for i in range(0, valid_idx.numel(), eval_batch_size):
            idx = valid_idx[i : i + eval_batch_size]
            full_batch = full_seq[idx].to(device, non_blocking=True)
            local_batch = local_seq[idx].to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(full_batch, local_batch)
            frame_preds.append(logits.view(-1).float().cpu())

        if not frame_preds:
            continue

        video_pred = torch.cat(frame_preds, dim=0).mean().item()
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    forward_batch_size: int,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    accum_count = 0
    optimizer.zero_grad()

    for full_frames, local_frames, labels in loader:
        # full/local: (B, K, 3, H, W), labels: (B, 1)
        bsz, num_frames, channels, height, width = full_frames.shape
        labels = labels.to(device, non_blocking=True).view(-1)
        full_flat = full_frames.reshape(bsz * num_frames, channels, height, width)
        local_flat = local_frames.reshape(bsz * num_frames, channels, height, width)

        pred_chunks = []
        for i in range(0, full_flat.size(0), forward_batch_size):
            full_batch = full_flat[i : i + forward_batch_size].to(device, non_blocking=True)
            local_batch = local_flat[i : i + forward_batch_size].to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                pred_chunks.append(model(full_batch, local_batch).view(-1).float())

        frame_preds = torch.cat(pred_chunks, dim=0).view(bsz, num_frames)
        video_preds = frame_preds.mean(dim=1)
        loss = criterion(video_preds, labels)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss / grad_accum_steps).backward()
        accum_count += 1
        running_loss += loss.item()
        steps += 1

        if accum_count % grad_accum_steps == 0:
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

    return running_loss / max(1, steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-level fine-tuning for SpatialDNet.")
    parser.add_argument("--preprocessed-root", default=os.path.join("data", "AVEC2014_preprocessed_uniform96"))
    parser.add_argument("--base-ckpt", default=os.path.join("checkpoints", "best_spatial_dnet_ft.pth"))
    parser.add_argument("--out-ckpt", default=os.path.join("checkpoints", "best_spatial_dnet_uniform96_video_ft.pth"))
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_dataset = VideoFrameBagDataset(
        split="train",
        preprocessed_root=args.preprocessed_root,
        frames_per_video=args.frames_per_video,
        augment=not args.no_augment,
    )
    dev_dataset = FrameLevelAVEC2014EvalDataset(split="dev", preprocessed_root=args.preprocessed_root)

    sampler = None if args.no_balanced_sampling else make_balanced_sampler(train_dataset.labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SpatialDNet(in_channels=3).to(device)
    if not os.path.exists(args.base_ckpt):
        raise RuntimeError(f"Base checkpoint not found: {args.base_ckpt}")

    state = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"[VideoLevelFinetune] Loaded base checkpoint: {args.base_ckpt}")
    print(
        f"[VideoLevelFinetune] train_videos={len(train_dataset)} | "
        f"frames_per_video={args.frames_per_video} | batch_size={args.batch_size} | "
        f"grad_accum_steps={args.grad_accum_steps} | balanced_sampling={not args.no_balanced_sampling}"
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=6,
        verbose=True,
        min_lr=1e-6,
    )

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    base_val_loss, best_mae, base_rmse = validate(model, dev_loader, device, args.eval_batch_size)
    torch.save(model.state_dict(), args.out_ckpt)
    print(
        f"[VideoLevelFinetune] Base Dev Loss: {base_val_loss:.4f} | "
        f"Base Dev MAE: {best_mae:.4f} | Base Dev RMSE: {base_rmse:.4f}"
    )
    print(f"[VideoLevelFinetune] Initial checkpoint copied to: {args.out_ckpt}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            forward_batch_size=args.forward_batch_size,
        )
        val_loss, val_mae, val_rmse = validate(model, dev_loader, device, args.eval_batch_size)
        scheduler.step(val_mae)

        print(
            f"[VideoLevelFinetune] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Dev Loss: {val_loss:.4f} | "
            f"Dev MAE: {val_mae:.4f} | Dev RMSE: {val_rmse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"[VideoLevelFinetune]  >>> New best video-level checkpoint saved (MAE: {val_mae:.4f})")


if __name__ == "__main__":
    main()
