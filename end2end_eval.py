import os
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.amp as amp

from data.end2end_dataset import End2EndAVEC2014Dataset, PreprocessedAVEC2014Dataset
from models.temporal_vit import End2EndDepressionModel


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for full_faces, local_faces, mask, labels in loader:
            full_faces = full_faces.to(device, non_blocking=True)
            local_faces = local_faces.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
    """在 AVEC2014 test 集上评估最终 DNet 模型。

    优先使用微调后的 best_dnet_model_ft.pth，若不存在则退回 best_dnet_model.pth。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据：优先使用预处理后的 AVEC2014
    max_frames = 48
    image_size = 256
    preprocessed_root = os.path.join("data", "AVEC2014_preprocessed")

    use_preprocessed = False
    preprocessed_test_dir = os.path.join(preprocessed_root, "test")
    if os.path.isdir(preprocessed_test_dir) and any(os.scandir(preprocessed_test_dir)):
        use_preprocessed = True

    if use_preprocessed:
        print(f"[eval] Using preprocessed AVEC2014 tensors from '{preprocessed_root}' (split=test)")
        test_dataset = PreprocessedAVEC2014Dataset(split="test", preprocessed_root=preprocessed_root)
    else:
        print("[eval] Preprocessed data not found, falling back to online MTCNN preprocessing from raw videos (test split).")
        test_dataset = End2EndAVEC2014Dataset(split="test", device="cpu", max_frames=max_frames, image_size=image_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 模型：与训练/微调保持一致
    visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_ckpt = (device.type == "cuda") and (visible_gpus >= 1)
    model = End2EndDepressionModel(
        temporal_chunks=4,
        use_checkpoint_temporal=use_ckpt,
        use_checkpoint_vit=use_ckpt,
    ).to(device)

    # 选择权重：优先使用微调后的 ft 模型
    ckpt_ft = os.path.join("checkpoints", "best_dnet_model_ft.pth")
    ckpt_base = os.path.join("checkpoints", "best_dnet_model.pth")

    if os.path.exists(ckpt_ft):
        ckpt_path = ckpt_ft
        print(f"[eval] Loading finetuned weights from {ckpt_path}")
    elif os.path.exists(ckpt_base):
        ckpt_path = ckpt_base
        print(f"[eval] Loading base best weights from {ckpt_path}")
    else:
        raise RuntimeError("[eval] No checkpoint found: neither best_dnet_model_ft.pth nor best_dnet_model.pth exists.")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    print(f"[eval] Start evaluation on test split using device={device}")
    val_loss, val_mae, val_rmse = evaluate(model, test_loader, device)

    print(
        f"[eval] Test Loss (MSE): {val_loss:.4f} | "
        f"Test MAE: {val_mae:.4f} | Test RMSE: {val_rmse:.4f}"
    )


if __name__ == "__main__":
    main()
