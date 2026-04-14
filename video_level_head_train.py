import argparse
import math
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data.end2end_dataset import PreprocessedAVEC2014Dataset
from models.spatial_dnet import SpatialDNet
from models.video_aggregator import MaskedAttentionStatsRegressor, ResidualAttentionStatsRegressor


DEFAULT_PREPROCESSED_ROOT = os.path.join("data", "AVEC2014_preprocessed_uniform96")
DEFAULT_BACKBONE_CKPT = os.path.join("checkpoints", "best_spatial_dnet_ft.pth")
DEFAULT_OUT_CKPT = os.path.join("checkpoints", "best_video_head_uniform96.pth")
FEATURE_SCHEMA = "spatial320_frame_scores_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight video-level head on frozen SpatialDNet features."
    )
    parser.add_argument("--preprocessed-root", default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-ckpt", default=DEFAULT_OUT_CKPT)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-type", choices=["attention", "residual"], default="attention")
    parser.add_argument("--residual-scale", type=float, default=8.0)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--loss", choices=["mse", "mae", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--smooth-l1-beta", type=float, default=5.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_cache_dir(preprocessed_root: str, backbone_ckpt: str) -> str:
    root_name = os.path.basename(os.path.normpath(preprocessed_root))
    ckpt_name = os.path.splitext(os.path.basename(backbone_ckpt))[0]
    return os.path.join("data", f"{root_name}_spatial_features", ckpt_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(checkpoint: str, device: torch.device) -> SpatialDNet:
    if not os.path.exists(checkpoint):
        raise RuntimeError(f"Backbone checkpoint not found: {checkpoint}")

    model = SpatialDNet(in_channels=3).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[VideoHead] Loaded frozen backbone: {checkpoint}")
    return model


def cache_metadata(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "preprocessed_root": os.path.abspath(args.preprocessed_root),
        "backbone_ckpt": os.path.abspath(args.backbone_ckpt),
        "feature_schema": FEATURE_SCHEMA,
    }


def cache_matches(cache: Dict, args: argparse.Namespace) -> bool:
    metadata = cache.get("metadata", {})
    expected = cache_metadata(args)
    return "frame_scores" in cache and all(metadata.get(key) == value for key, value in expected.items())


@torch.no_grad()
def extract_split_features(
    split: str,
    backbone: SpatialDNet,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    dataset = PreprocessedAVEC2014Dataset(split=split, preprocessed_root=args.preprocessed_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    features: List[torch.Tensor] = []
    frame_scores: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    labels: List[float] = []
    files: List[str] = []

    for video_idx, (full_seq, local_seq, mask, label) in enumerate(loader):
        full_seq = full_seq.squeeze(0)
        local_seq = local_seq.squeeze(0)
        mask = mask.squeeze(0).bool()
        valid_idx = (~mask).nonzero(as_tuple=False).view(-1)
        seq_features = torch.zeros(mask.numel(), 320, dtype=torch.float32)
        seq_scores = torch.zeros(mask.numel(), dtype=torch.float32)

        for start in range(0, valid_idx.numel(), args.extract_batch_size):
            idx = valid_idx[start : start + args.extract_batch_size]
            full_batch = full_seq[idx].to(device, non_blocking=True)
            local_batch = local_seq[idx].to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=device.type == "cuda"):
                batch_features = backbone.extract_features(full_batch, local_batch)
                batch_scores = backbone.head(batch_features.view(-1, 320, 1, 1)).view(-1)
            seq_features[idx] = batch_features.detach().float().cpu()
            seq_scores[idx] = batch_scores.detach().float().cpu()

        features.append(seq_features)
        frame_scores.append(seq_scores)
        masks.append(mask.cpu())
        labels.append(float(label.view(-1).item()))
        files.append(dataset.files[video_idx])

        if (video_idx + 1) % 10 == 0 or video_idx + 1 == len(dataset):
            print(f"[VideoHead] Extracted {split}: {video_idx + 1}/{len(dataset)} videos")

    return {
        "features": torch.stack(features, dim=0),
        "frame_scores": torch.stack(frame_scores, dim=0),
        "mask": torch.stack(masks, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "files": files,
        "metadata": cache_metadata(args),
    }


def load_or_build_cache(
    split: str,
    backbone: SpatialDNet,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{split}.pt")

    if os.path.exists(cache_path) and not args.rebuild_cache:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_matches(cache, args):
            print(f"[VideoHead] Loaded cached {split} features: {cache_path}")
            return cache
        print(f"[VideoHead] Cache metadata mismatch, rebuilding: {cache_path}")

    cache = extract_split_features(split, backbone, args, device)
    torch.save(cache, cache_path)
    print(f"[VideoHead] Saved cached {split} features: {cache_path}")
    return cache


class CachedVideoFeatureDataset(Dataset):
    def __init__(self, cache: Dict) -> None:
        self.features = cache["features"]
        self.frame_scores = cache["frame_scores"]
        self.mask = cache["mask"].bool()
        self.labels = cache["labels"].float()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.mask[idx], self.frame_scores[idx], self.labels[idx]


def make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    bins = []
    for label in labels.tolist():
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


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "mae":
        return nn.L1Loss()
    return nn.SmoothL1Loss(beta=args.smooth_l1_beta)


def apply_temporal_dropout(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0.0:
        return mask

    new_mask = mask.clone()
    valid = ~new_mask
    drop = (torch.rand_like(new_mask.float()) < probability) & valid
    new_mask = new_mask | drop

    all_masked = new_mask.all(dim=1)
    if all_masked.any():
        for row in all_masked.nonzero(as_tuple=False).view(-1):
            valid_idx = valid[row].nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() > 0:
                keep = valid_idx[torch.randint(valid_idx.numel(), (1,), device=valid_idx.device)]
                new_mask[row, keep] = False

    return new_mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    temporal_dropout: float,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for features, mask, frame_scores, labels in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        frame_scores = frame_scores.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1)
        train_mask = apply_temporal_dropout(mask, temporal_dropout)

        preds = model(features, train_mask, frame_scores).view(-1)
        loss = criterion(preds, labels)
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        steps += 1

    return running_loss / max(1, steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    all_preds: List[float] = []
    all_labels: List[float] = []
    total_loss = 0.0
    steps = 0
    mse = nn.MSELoss()

    for features, mask, frame_scores, labels in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        frame_scores = frame_scores.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1)

        preds = model(features, mask, frame_scores).view(-1)
        loss = mse(preds, labels)
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        steps += 1
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    if steps == 0 or not all_preds:
        return float("inf"), float("inf"), float("inf")

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    return total_loss / steps, mae, rmse


def save_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    dev_mae: float,
    dev_rmse: float,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": 320,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "head_type": args.head_type,
            "residual_scale": args.residual_scale,
            "dev_mae": dev_mae,
            "dev_rmse": dev_rmse,
            "preprocessed_root": args.preprocessed_root,
            "backbone_ckpt": args.backbone_ckpt,
            "cache_dir": args.cache_dir,
        },
        path,
    )


def build_head(args: argparse.Namespace) -> nn.Module:
    if args.head_type == "attention":
        return MaskedAttentionStatsRegressor(
            input_dim=320,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    return ResidualAttentionStatsRegressor(
        input_dim=320,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    )


def load_head_checkpoint(path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(path):
        raise RuntimeError(f"Video head checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head_type = payload.get("head_type", "attention")
    if head_type == "attention":
        model = MaskedAttentionStatsRegressor(
            input_dim=int(payload.get("input_dim", 320)),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(device)
    else:
        model = ResidualAttentionStatsRegressor(
            input_dim=int(payload.get("input_dim", 320)),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            dropout=float(payload.get("dropout", 0.0)),
            residual_scale=float(payload.get("residual_scale", 8.0)),
        ).to(device)
    model.load_state_dict(payload["model_state"])
    return model


def main() -> None:
    args = parse_args()
    if args.cache_dir is None:
        args.cache_dir = default_cache_dir(args.preprocessed_root, args.backbone_ckpt)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(
        f"[VideoHead] preprocessed_root={args.preprocessed_root} | "
        f"backbone_ckpt={args.backbone_ckpt} | cache_dir={args.cache_dir} | "
        f"out_ckpt={args.out_ckpt}"
    )

    backbone = load_backbone(args.backbone_ckpt, device)
    train_cache = load_or_build_cache("train", backbone, args, device)
    dev_cache = load_or_build_cache("dev", backbone, args, device)
    test_cache = None
    if not args.skip_test:
        test_cache = load_or_build_cache("test", backbone, args, device)

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.epochs <= 0:
        print("[VideoHead] epochs <= 0, exiting after feature-cache setup.")
        return

    train_dataset = CachedVideoFeatureDataset(train_cache)
    dev_dataset = CachedVideoFeatureDataset(dev_cache)
    test_dataset = CachedVideoFeatureDataset(test_cache) if test_cache is not None else None

    sampler = make_balanced_sampler(train_dataset.labels) if args.balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = build_head(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    criterion = build_criterion(args)

    best_mae = float("inf")
    best_rmse = float("inf")
    bad_epochs = 0

    print(
        f"[VideoHead] train_videos={len(train_dataset)} | dev_videos={len(dev_dataset)} | "
        f"batch_size={args.batch_size} | head_type={args.head_type} | "
        f"loss={args.loss} | balanced_sampling={args.balanced_sampling}"
    )

    if args.head_type == "residual":
        base_dev_loss, best_mae, best_rmse = evaluate(model, dev_loader, device)
        save_checkpoint(args.out_ckpt, model, args, best_mae, best_rmse)
        print(
            f"[VideoHead] Initial residual baseline | Dev Loss: {base_dev_loss:.4f} | "
            f"Dev MAE: {best_mae:.4f} | Dev RMSE: {best_rmse:.4f}"
        )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            temporal_dropout=args.temporal_dropout,
        )
        dev_loss, dev_mae, dev_rmse = evaluate(model, dev_loader, device)
        scheduler.step(dev_mae)

        improved = dev_mae < best_mae
        if improved:
            best_mae = dev_mae
            best_rmse = dev_rmse
            bad_epochs = 0
            save_checkpoint(args.out_ckpt, model, args, best_mae, best_rmse)
        else:
            bad_epochs += 1

        marker = " *" if improved else ""
        print(
            f"[VideoHead] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | Dev MAE: {dev_mae:.4f} | Dev RMSE: {dev_rmse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}"
        )

        if bad_epochs >= args.early_stop_patience:
            print(
                f"[VideoHead] Early stopping after {bad_epochs} epochs without Dev MAE improvement."
            )
            break

    best_model = load_head_checkpoint(args.out_ckpt, device)
    best_dev_loss, best_dev_mae, best_dev_rmse = evaluate(best_model, dev_loader, device)
    print(
        f"[VideoHead] Best Dev Loss: {best_dev_loss:.4f} | "
        f"Best Dev MAE: {best_dev_mae:.4f} | Best Dev RMSE: {best_dev_rmse:.4f}"
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        test_loss, test_mae, test_rmse = evaluate(best_model, test_loader, device)
        print(
            f"[VideoHead] Test Loss: {test_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}"
        )


if __name__ == "__main__":
    main()
