import os
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.amp as amp

from data.frame_level_avec2014 import FrameLevelAVEC2014EvalDataset
from models.spatial_dnet import SpatialDNet


@torch.no_grad()
def evaluate(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_root = os.path.join("data", "AVEC2014_preprocessed")
    test_dataset = FrameLevelAVEC2014EvalDataset(
        split="test",
        preprocessed_root=preprocessed_root,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = SpatialDNet(in_channels=3).to(device)

    # 优先使用微调后的权重 best_spatial_dnet_ft.pth，若不存在则退回初始训练得到的 best_spatial_dnet.pth。
    ckpt_ft = os.path.join("checkpoints", "best_spatial_dnet_ft.pth")
    ckpt_base = os.path.join("checkpoints", "best_spatial_dnet.pth")
    if os.path.exists(ckpt_ft):
        ckpt_path = ckpt_ft
        print(f"[FrameLevelEval] Using fine-tuned checkpoint: {ckpt_path}")
    else:
        ckpt_path = ckpt_base
        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"Checkpoint not found: {ckpt_path}. Please run frame_level_train.py (and optionally frame_level_finetune.py) first."
            )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    loss, mae, rmse = evaluate(model, test_loader, device)

    print(
        f"[FrameLevelEval] Test Loss: {loss:.4f} | Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}"
    )


if __name__ == "__main__":
    main()
