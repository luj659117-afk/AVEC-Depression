# AVEC2014 Depression Recognition

本项目用于基于 AVEC2014 视频数据集进行抑郁程度回归预测，当前推荐的主流程是**帧级 DNet 风格网络**：FEM + FPN + 空间 ViT Block（无显式时序 Transformer），并配合 AffectNet 上的表情预训练。

当前推荐（帧级 DNet 风格）流程概览：
- 数据预处理：从原始 AVEC2014 视频中每 3 帧抽 1 帧作为候选帧，再默认均匀采样 96 帧覆盖整段视频；使用 MTCNN + OpenCV 仿射变换做人脸对齐，对每帧进行双尺度裁剪（全脸 / 局部面部），并在缩放到 256×256 时采用黑边 Padding 保持宽高比。预处理可离线一次性完成，保存为张量文件。  
- 空间骨干（FEM + FPN）：双路 FEM（交替 ConvBlock + InBlock + Coordinate Attention + DenseBlock）分别处理全脸和局部脸特征，通道维拼接后送入 FPN 做多尺度融合，输出 320 通道特征图。  
- 空间 ViT Block（SpatialViTBlock）：在 2D 特征图上，同时通过 3×3 卷积分支（局部）与 Unfold→Transformer→Fold 分支（全局）建模不同面部区域的局部/全局语义，并与输入特征残差融合。  
- 帧级回归头：对 ViT Block 输出做全局平均池化（GAP），经 1×1 卷积 + 全连接层输出单帧的抑郁分数；视频级预测为该视频所有帧预测分数的平均值。  

> 说明：仓库中仍保留早期的“时序 Transformer + ViT Block”的端到端实现（`end2end_*.py` + `models/temporal_vit.py`），但当前推荐的实验与指标均基于帧级 SpatialDNet 流程。

## 项目结构

- `data/end2end_dataset.py`：端到端 AVEC2014 数据集，包含抽帧、MTCNN 检测与对齐、双尺度裁剪和 Padding；同时提供 `PreprocessedAVEC2014Dataset` 读取离线预处理好的张量（帧级流程复用该预处理结果）。  
- `data/affectnet_dataset.py`：AffectNet 表情数据集封装（csv 解析 + bbox 裁剪 + 增强），用于 FEM / ResNet18 的表情预训练与 sanity check。  
- `data/frame_level_avec2014.py`：帧级 AVEC2014 数据集封装，包含：
	- `FrameLevelAVEC2014TrainDataset`：frame-level 训练集，每个样本为单帧 `(full, local, label)`，支持按视频分层采样与轻量数据增强；
	- `FrameLevelAVEC2014EvalDataset`：验证/测试集，返回整段视频的帧序列与 mask，用于视频级评估。  
- `models/fem_fpn.py`：FEM（含 Coordinate Attention + DenseBlock）与 FPN 的双路空间骨干实现。  
- `models/spatial_dnet.py`：SpatialDNet 帧级回归模型（DualFEMWithFPN + SpatialViTBlock + GAP + 回归头）。  
- `models/video_aggregator.py`：冻结 SpatialDNet 帧特征后的轻量视频级聚合头，支持 attention+统计聚合与残差校正两种形式。  
- `preprocess_avec2014.py`：从原始 AVEC2014 视频离线抽帧 + MTCNN + 对齐 + 裁剪 + Padding，将每个样本保存为 `.pt`（full/local 人脸序列 + mask + label）。  
- `fem_pretrain.py`：在 AffectNet 上用 FEMBackbone 做 7 类表情分类预训练，训练得到的 FEM 权重可选地加载到 DNet 的双路 FEM 中。  
- `resnet18_affectnet_pretrain.py`：使用 torchvision ResNet18+ImageNet 预训练在 AffectNet 上做表情分类，用于 sanity check 数据与预处理管道。  
- `frame_level_train.py`：帧级 SpatialDNet 训练脚本，从 `uniform-96` 预处理 `.pt` 中按帧采样训练，并在 dev 上以视频平均分评估 MAE / RMSE，最佳模型默认保存到 `checkpoints/best_spatial_dnet_uniform96.pth`。  
- `frame_level_finetune.py`：在 `best_spatial_dnet_uniform96.pth`（若不存在则回退旧 checkpoint）基础上，以更小学习率进一步微调，最佳模型默认保存到 `checkpoints/best_spatial_dnet_uniform96_ft.pth`。  
- `frame_level_eval.py`：帧级 SpatialDNet 测试脚本，默认读取 `uniform-96` 预处理数据，并优先加载 `best_spatial_dnet_uniform96_ft.pth` / `best_spatial_dnet_uniform96.pth`，在 AVEC2014 test 集上评估 MAE / RMSE。  
- `video_level_head_train.py`：冻结帧级 SpatialDNet，缓存每帧 320 维特征，然后只训练轻量视频级聚合头；默认输出 `checkpoints/best_video_head_uniform96.pth`。  
- `video_level_head_eval.py`：评估已训练好的视频级聚合头，复用缓存特征并输出 dev/test MAE / RMSE。  
- `data/AVEC2014/`：原始 AVEC2014 视频与标签根目录。  
- `data/AVEC2014_preprocessed_uniform96/`：当前默认的 `uniform-96` 离线预处理人脸张量根目录（按 `train/dev/test` 划分）。  
- `data/AVEC2014_preprocessed/`：旧版 48 帧预处理目录，仅作为历史对比保留。  
- `data/AffectNet/`：AffectNet 表情数据集根目录（含 `training.csv` / `validation.csv` 及解压后的图片子目录），仅用于 FEM / ResNet18 表情预训练实验，不参与 AVEC2014 主流程。  
- `checkpoints/`：保存端到端/帧级模型和预训练权重的目录（如 `fem_backbone_affectnet.pth`、`best_spatial_dnet*.pth` 等）。

## 数据约定

数据集按如下方式组织：

- `data/AVEC2014/train/`
- `data/AVEC2014/dev/`
- `data/AVEC2014/test/`

每个视频文件名示例：

- `203_1_Freeform_video.mp4`
- `203_2_Freeform_video.mp4`

其中：

- `203` 表示第 203 个受试者
- `203_1` 表示该受试者第 1 个时期的数据
- `203_2` 表示该受试者第 2 个时期的数据

标签文件位于：

- `data/AVEC2014/label/DepressionLabels/`

文件名示例：

- `203_1_Depression.csv`
- `203_2_Depression.csv`

文件内容为对应样本的抑郁分数。

## 环境依赖

建议使用 `depress_cv` 虚拟环境。

主要依赖：

- Python 3.10+
- PyTorch
- torchvision
- numpy
- scikit-learn
- facenet-pytorch
- opencv-python
- tqdm

## 运行流程

### 0.（推荐）AffectNet 上预训练 FEM / sanity check

当前实验证明：在修正后的 AffectNet 数据管线与帧级 SpatialDNet 架构下，对 FEMBackbone 在 AffectNet 上进行 7 类表情预训练**能够显著提升 AVEC2014 上的 MAE / RMSE**（例如在严格 subject-level 划分下，test MAE 由约 9.8 降至约 7.7）。

- 在 AffectNet 上预训练 FEMBackbone：  

```bash
CUDA_VISIBLE_DEVICES=1 python fem_pretrain.py
```

- 使用 ResNet18 在 AffectNet 上做 7 类表情分类 sanity check（验证数据与预处理逻辑）：  

```bash
CUDA_VISIBLE_DEVICES=1 python resnet18_affectnet_pretrain.py
```

若 `checkpoints/fem_backbone_affectnet.pth` 存在，`frame_level_train.py` / `frame_level_finetune.py` 会自动将其加载到 SpatialDNet 的双路 FEM 中作为初始化；否则退回随机初始化。

### 1. 激活环境

```bash
source /data/hcf/miniconda3/etc/profile.d/conda.sh
conda activate depress_cv
```

### 2. 离线预处理 AVEC2014（推荐）

```bash
cd /data/hcf/projects/avec_depression
CUDA_VISIBLE_DEVICES=  # 可选，预处理主要跑在 CPU 上
python preprocess_avec2014.py
```

在 24G 显存约束下，推荐使用 `uniform` 预处理：每个视频保存固定帧数，但这些帧均匀覆盖整段视频，而不是只取视频开头的若干帧。当前实测中，`uniform-96` 是更好的折中点：比 `uniform-48` 指标更好，而评估显存仍远低于 24G。

```bash
CUDA_VISIBLE_DEVICES=2 python preprocess_avec2014.py \
  --out-root data/AVEC2014_preprocessed_uniform96 \
  --temporal-sample uniform \
  --device cuda \
  --max-frames 96
```

完成后，应在 `data/AVEC2014_preprocessed_uniform96/train|dev|test` 下看到若干 `.pt` 文件，每个文件包含：

- `full_faces`: `(T, 3, 256, 256)` 全脸序列张量（已归一化）；
- `local_faces`: `(T, 3, 256, 256)` 局部脸序列张量；
- `mask`: `(T,)` 帧掩码，True 表示 padding 帧；
- `label`: 标量抑郁分数。

### 3. 帧级 SpatialDNet 训练（单卡）

```bash
CUDA_VISIBLE_DEVICES=1 python frame_level_train.py
```

脚本会：

- 默认使用 `data/AVEC2014_preprocessed_uniform96/` 中的预处理数据，通过 `FrameLevelAVEC2014TrainDataset` 按视频分层采样单帧 `(full, local, label)` 进行训练；
- 训练时使用 `batch_size=64`，带轻量空间增广（水平翻转 + 亮度扰动）；
- 在 dev 集上通过 `FrameLevelAVEC2014EvalDataset` 以“整段视频所有有效帧的预测均值”作为视频级分数，评估 MAE / RMSE；
- 在 dev 集 MAE 最优时保存模型到 `checkpoints/best_spatial_dnet_uniform96.pth`。

### 4. 帧级 SpatialDNet 小学习率微调

在得到初始最佳帧级模型后，可用较小学习率做额外微调：

```bash
CUDA_VISIBLE_DEVICES=1 python frame_level_finetune.py
```

脚本会优先从 `checkpoints/best_spatial_dnet_uniform96.pth` 加载权重，若不存在则回退到旧版 `best_spatial_dnet_ft.pth` / `best_spatial_dnet.pth`；使用学习率 `2e-5` 再训练若干 epoch，并在 dev 集 MAE 进一步下降时将模型保存到：

- `checkpoints/best_spatial_dnet_uniform96_ft.pth`

### 5. 在 test 集上评估（帧级 SpatialDNet）

```bash
CUDA_VISIBLE_DEVICES=1 python frame_level_eval.py
```

脚本默认读取 `data/AVEC2014_preprocessed_uniform96/`，并优先加载 `best_spatial_dnet_uniform96_ft.pth` / `best_spatial_dnet_uniform96.pth`，若不存在则回退旧版 checkpoint，并在 AVEC2014 test 集上输出：

- `Test Loss (MSE)`
- `Test MAE`
- `Test RMSE`

若要评估指定 checkpoint 或指定预处理根目录，可使用：

```bash
CUDA_VISIBLE_DEVICES=2 python frame_level_eval.py \
  --preprocessed-root data/AVEC2014_preprocessed_uniform96 \
  --checkpoint checkpoints/best_spatial_dnet_ft.pth
```

当前实测中，在不重新训练模型的情况下，仅切换评估预处理即可提升指标：

- 原始 48 帧策略：`Test MAE: 7.7153 | Test RMSE: 9.6147`
- `uniform-48`：`Test MAE: 7.6070 | Test RMSE: 9.4392`
- `uniform-96`：`Test MAE: 7.5670 | Test RMSE: 9.3476`
- `uniform-144` test-only 快速评估：`Test MAE: 7.5649 | Test RMSE: 9.3461`

`uniform-144` 相对 `uniform-96` 收益已经很小；一次性前向 144 帧时 PyTorch 记录的 2 号 4090 峰值显存约为 `peak_alloc=3.84GB`、`peak_reserved=6.63GB`，评估阶段不存在 24G 显存瓶颈。

端到端训练显存 smoke test（batch=1、AMP、gradient checkpoint、按时间分段前向）显示：`uniform-96` 的一次 forward+backward 峰值约 `peak_reserved=11.81GB`，`uniform-144` 约 `peak_reserved=17.50GB`，二者在 24G 4090 上均未 OOM。考虑指标收益和安全边际，当前优先推荐 `uniform-96`。

### 6. 推荐增强：冻结特征 + 轻量视频级聚合头

在当前 96 帧版本上，更推荐冻结已有帧级 SpatialDNet，只训练一个轻量视频级 head，而不是继续微调整个空间骨干。该脚本会先把每个视频的 96 帧抽成 `(T, 320)` 特征缓存到磁盘，然后训练 attention+mean/std 统计聚合头：

```bash
CUDA_VISIBLE_DEVICES=2 python video_level_head_train.py \
  --preprocessed-root data/AVEC2014_preprocessed_uniform96 \
  --backbone-ckpt checkpoints/best_spatial_dnet_ft.pth \
  --out-ckpt checkpoints/best_video_head_uniform96.pth \
  --head-type attention \
  --epochs 300 \
  --batch-size 16 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --temporal-dropout 0.1 \
  --dropout 0.2 \
  --loss smooth_l1
```

训练完成后可单独复测：

```bash
CUDA_VISIBLE_DEVICES=2 python video_level_head_eval.py \
  --preprocessed-root data/AVEC2014_preprocessed_uniform96 \
  --backbone-ckpt checkpoints/best_spatial_dnet_ft.pth \
  --head-ckpt checkpoints/best_video_head_uniform96.pth \
  --split test
```

当前实测结果：

- 96 帧帧均值基线（`best_spatial_dnet_ft.pth`）：`Test MAE: 7.5670 | Test RMSE: 9.3476`
- 冻结特征 + attention 视频级聚合头：`Test MAE: 7.4362 | Test RMSE: 9.2224`

这个路线没有更新 FEM/FPN/SpatialViT 骨干，显存压力很小；特征缓存完成后，head 的调参迭代速度也明显更快。已额外尝试残差校正头和 MSE 版本，当前都没有超过 attention+SmoothL1 组合。

### 7. 实验性：视频级 SpatialDNet 微调

`video_level_finetune.py` 会从已有帧级 checkpoint 出发，每个视频采样 K 帧，对帧级预测取均值后使用视频级标签计算 loss。该路线用于验证“训练目标与视频级评估目标对齐”是否有效，会保存到独立 checkpoint，不覆盖当前最佳模型：

```bash
CUDA_VISIBLE_DEVICES=2 python video_level_finetune.py \
  --preprocessed-root data/AVEC2014_preprocessed_uniform96 \
  --frames-per-video 8 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 2e-6 \
  --no-balanced-sampling \
  --out-ckpt checkpoints/best_spatial_dnet_uniform_video_ft.pth
```

初步实验显示，全模型 video-level fine-tune 容易扰动已学好的空间表征，当前主要有效收益来自 `uniform-96` 预处理；若继续优化，建议优先尝试冻结 FEM/FPN/ViT 后只训练轻量视频聚合头。

## 训练说明

当前帧级 SpatialDNet 训练 / 微调脚本已经内置了以下稳定性策略：

- 训练和验证阶段都会跳过非有限 loss（NaN / Inf）；
- 使用 AMP 混合精度降低显存占用；
- 反向传播时使用梯度裁剪，减少数值爆炸风险；
- 单卡训练，避免多进程通信导致的不稳定因素；

## 输出

训练 / 微调过程中会输出：

- `Train Loss`
- `Val Loss (MSE)`
- `Val MAE`
- `Val RMSE`

最终模型文件：

- 初始最佳帧级 SpatialDNet 模型：`checkpoints/best_spatial_dnet_uniform96.pth`
- 小学习率微调后最佳帧级模型：`checkpoints/best_spatial_dnet_uniform96_ft.pth`
- 冻结特征视频级聚合头：`checkpoints/best_video_head_uniform96.pth`

评估脚本会在终端打印 test 集的 MAE / RMSE，可直接与 DNet 论文中的 AVEC2014 指标进行对比。

> 早期的端到端时序模型相关脚本（`end2end_train.py` / `end2end_finetune.py` / `end2end_eval.py` / `models/temporal_vit.py`）目前已不再作为推荐流程的一部分，仅作为对比和参考代码保留。
