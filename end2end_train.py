import os
import math
from typing import Tuple

# 尽早配置 CUDA 分配器，降低显存碎片风险
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    # 仅使用单机单卡训练，简化分布式逻辑，避免多进程通信带来的不稳定性。
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 单卡：batch=1、累积 4 次，对齐论文有效 batch≈4。
    train_batch_size = 1
    accumulation_steps = 4

    torch.backends.cudnn.benchmark = True
    # 数据部分：优先使用离线预处理好的人脸特征（与论文设置一致），
    # 若预处理目录不存在或为空，则退化为在线从原始视频抽帧 + MTCNN 方式。
    max_frames = 48
    image_size = 256
    preprocessed_root = os.path.join("data", "AVEC2014_preprocessed")

    use_preprocessed = False
    preprocessed_train_dir = os.path.join(preprocessed_root, "train")
    if os.path.isdir(preprocessed_train_dir) and any(os.scandir(preprocessed_train_dir)):
        use_preprocessed = True

    if use_preprocessed:
        if rank == 0:
            print(f"Using preprocessed AVEC2014 tensors from '{preprocessed_root}'")
        train_dataset = PreprocessedAVEC2014Dataset(split="train", preprocessed_root=preprocessed_root)
        dev_dataset = PreprocessedAVEC2014Dataset(split="dev", preprocessed_root=preprocessed_root)
    else:
        if rank == 0:
            print("Preprocessed data not found, falling back to online MTCNN preprocessing from raw videos.")
        # Dataset 内部的 MTCNN 固定在 CPU 上运行，避免在 DataLoader 多进程中初始化 CUDA。
        # 对齐论文设置：
        # - 输入分辨率 image_size = 256
        # - batch_size = 4（见下方 DataLoader 设置）
        # 为了在 24G 显存下稳定训练，同时仍保持较长的时序信息，
        # 将每段视频使用的最大帧数设为 48 帧，并在数据集中对训练样本做随机时间采样。
        train_dataset = End2EndAVEC2014Dataset(split="train", device="cpu", max_frames=max_frames, image_size=image_size)
        dev_dataset = End2EndAVEC2014Dataset(split="dev", device="cpu", max_frames=max_frames, image_size=image_size)

    # 单卡场景下使用普通 DataLoader
    train_sampler = None
    dev_sampler = None

    # 使用多进程 DataLoader 提升吞吐（MTCNN 在 CPU 上，通常不会触发 CUDA 多进程问题）
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 构建当前 DNet 结构；若存在与该结构匹配的最新 checkpoint，则优先加载
    # temporal_chunks=4：将 48 帧拆成 4 段，每段 12 帧，仅在 FEM+FPN 中分段前向，
    # 时序 Transformer 仍对完整 48 帧序列建模，保证与论文结构等价。
    # 单卡 GPU 训练时启用梯度检查点以节省显存。
    visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_ckpt = (device.type == "cuda") and (visible_gpus >= 1)
    model = End2EndDepressionModel(
        temporal_chunks=4,
        use_checkpoint_temporal=use_ckpt,
        use_checkpoint_vit=use_ckpt,
    ).to(device)
    ckpt_path = os.path.join("checkpoints", "best_dnet_model.pth")
    resumed = False
    if os.path.exists(ckpt_path):
        print(f"Loading DNet weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        resumed = True
    else:
        # 若尚无端到端 DNet checkpoint，但存在 FEMBackbone 在 AffectNet 上的预训练权重，
        # 则优先加载该预训练权重到 DualFEMWithFPN 的全脸和局部分支，作为更好的初始化。
        fem_ckpt = os.path.join("checkpoints", "fem_backbone_affectnet.pth")
        if os.path.exists(fem_ckpt):
            print(f"Loading pretrained FEM backbone from {fem_ckpt}")
            fem_state = torch.load(fem_ckpt, map_location="cpu", weights_only=True)
            # spatial_backbone 是 DualFEMWithFPN 实例
            model.spatial_backbone.load_pretrained_single_fem(fem_state, strict=False)

    # 超参数：Adam, lr=1e-4, weight_decay=5e-5, MSELoss
    # 若是从已有最优模型继续训练，则适当减小学习率，避免发散
    base_lr = 1e-4
    if resumed:
        # 继续训练时进一步减小学习率，缓和 loss/MAE 的震荡
        base_lr = 2e-5

    optimizer = optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    criterion = nn.MSELoss()

    # 使用 AMP 混合精度训练以降低显存占用（新版 torch.amp 接口）
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    # 学习率调度器：当验证 MAE 连续若干 epoch 无明显改善时自动降低 lr，减小后期抖动
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )

    # 论文中训练轮数约为 50，这里适当增加到 150，结合早停可灵活控制
    num_epochs = 150
    best_mae = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        optimizer.zero_grad()
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

            # 梯度累积：将 loss 按 accumulation_steps 缩放后再反传
            scaler.scale(loss / accumulation_steps).backward()
            accum_count += 1

            running_loss += loss.item()
            steps += 1

            if accum_count % accumulation_steps == 0:
                # 先反缩放，再做梯度裁剪和优化步
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

        # 处理最后不足 accumulation_steps 的残余梯度
        if accum_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if steps == 0:
            print(f"Epoch {epoch:02d} | no valid training steps, skipping validation.")
            continue

        train_loss = running_loss / steps
        val_loss, val_mae, val_rmse = validate(model, dev_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 使用验证 MAE 作为调度依据，MAE 无法持续下降时自动减小学习率
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_dnet_model.pth"))
            print(f"  >>> New best DNet model saved (MAE: {val_mae:.4f})")


if __name__ == "__main__":
    main()
