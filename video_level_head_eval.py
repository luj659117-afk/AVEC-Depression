import argparse
import os

import torch
from torch.utils.data import DataLoader

from video_level_head_train import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_PREPROCESSED_ROOT,
    CachedVideoFeatureDataset,
    default_cache_dir,
    evaluate,
    load_backbone,
    load_head_checkpoint,
    load_or_build_cache,
)


DEFAULT_HEAD_CKPT = os.path.join("checkpoints", "best_video_head_uniform96.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen-backbone video-level head.")
    parser.add_argument("--preprocessed-root", default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE_CKPT)
    parser.add_argument("--head-ckpt", default=DEFAULT_HEAD_CKPT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_dir is None:
        args.cache_dir = default_cache_dir(args.preprocessed_root, args.backbone_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(args.backbone_ckpt, device)
    cache = load_or_build_cache(args.split, backbone, args, device)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dataset = CachedVideoFeatureDataset(cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = load_head_checkpoint(args.head_ckpt, device)
    loss, mae, rmse = evaluate(model, loader, device)
    print(
        f"[VideoHeadEval] split={args.split} | Loss: {loss:.4f} | "
        f"MAE: {mae:.4f} | RMSE: {rmse:.4f}"
    )


if __name__ == "__main__":
    main()
