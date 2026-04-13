# AVEC2014 Depression Recognition

本项目用于基于 AVEC2014 视频数据集进行抑郁程度回归预测，当前仅保留**端到端特征学习网络**（DNet 风格：FEM + FPN + 时序 Transformer + ViT Block），不再依赖 legacy 特征提取代码。

端到端（DNet 风格）流程概览：
- 数据预处理：从原始 AVEC2014 视频中每 3 帧抽 1 帧，使用 MTCNN + OpenCV 仿射变换做人脸对齐；对每帧进行双尺度裁剪（全脸 / 局部面部），并在缩放到 256×256 时采用黑边 Padding 保持宽高比。预处理可离线一次性完成，保存为张量文件。  
- 空间骨干（FEM + FPN）：双路 FEM（交替 ConvBlock + InBlock + Coordinate Attention + DenseBlock）分别处理全脸和局部脸特征，通道维拼接后送入 FPN 做多尺度融合，输出 320 通道特征图。  
- 时序与局部-全局提纯：对每帧 320 维特征序列使用 2 层 Transformer Encoder 建模帧间动态，并做时序池化得到视频级表示；再送入 ViT Block（Unfold→Transformer→Fold）进一步提炼局部与全局高级语义特征。  
- 回归头：在视频级特征后接 1×1 卷积 + 全连接层输出单一抑郁分数，训练中使用 Adam、MSELoss，支持混合精度与梯度累积。

## 项目结构

- `data/end2end_dataset.py`：端到端 AVEC2014 数据集，包含抽帧、MTCNN 检测与对齐、双尺度裁剪和 Padding；同时提供 `PreprocessedAVEC2014Dataset` 读取离线预处理好的张量。  
- `data/affectnet_dataset.py`：AffectNet 表情数据集封装（csv 解析 + bbox 裁剪 + 增强），用于 FEM / ResNet18 的表情预训练与 sanity check。  
- `models/fem_fpn.py`：FEM（含 Coordinate Attention + DenseBlock）与 FPN 的双路空间骨干实现。  
- `models/temporal_vit.py`：时序 Transformer + ViT Block + 回归头的端到端回归模型。  
- `preprocess_avec2014.py`：从原始 AVEC2014 视频离线抽帧 + MTCNN + 对齐 + 裁剪 + Padding，将每个样本保存为 `.pt`（full/local 人脸序列 + mask + label）。  
- `end2end_train.py`：端到端训练脚本（单卡），优先使用离线预处理数据，无则退回在线 MTCNN 预处理。  
- `end2end_finetune.py`：从 `checkpoints/best_dnet_model.pth` 以较小学习率继续微调若干 epoch，最佳模型保存到 `checkpoints/best_dnet_model_ft.pth`。  
- `end2end_eval.py`：加载最终模型（优先微调后的 `best_dnet_model_ft.pth`），在 AVEC2014 test 集上评估 MAE / RMSE。  
- `fem_pretrain.py`：在 AffectNet 上用 FEMBackbone 做 7 类表情分类预训练，训练得到的 FEM 权重可选地加载到 DNet 的双路 FEM 中。  
- `resnet18_affectnet_pretrain.py`：使用 torchvision ResNet18+ImageNet 预训练在 AffectNet 上做表情分类，用于 sanity check 数据与预处理管道。  
- `data/AVEC2014/`：原始 AVEC2014 视频与标签根目录。  
- `data/AVEC2014_preprocessed/`：离线预处理后的人脸张量文件根目录（按 `train/dev/test` 划分）。  
- `data/AffectNet/`：AffectNet 表情数据集根目录（含 `training.csv` / `validation.csv` 及解压后的图片子目录），仅用于 FEM / ResNet18 表情预训练实验，不参与 AVEC2014 主流程。  
- `checkpoints/`：保存端到端模型权重的目录。

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

### 0.（可选）AffectNet 上预训练 FEM / sanity check

该步骤用于对齐论文中“先在 AffectNet 上预训练 FEM”的设定，但在当前实现下，对 AVEC2014 最终指标提升有限，因此完全可以跳过。只在你有兴趣复现实验时使用。

- 在 AffectNet 上预训练 FEMBackbone：  

```bash
CUDA_VISIBLE_DEVICES=1 python fem_pretrain.py
```

- 使用 ResNet18 在 AffectNet 上做 7 类表情分类 sanity check（验证数据与预处理逻辑）：  

```bash
CUDA_VISIBLE_DEVICES=1 python resnet18_affectnet_pretrain.py
```

若 `checkpoints/fem_backbone_affectnet.pth` 存在且当前没有端到端 checkpoint，`end2end_train.py` 会自动将其加载到 DNet 的双路 FEM 中作为初始化；否则退回随机初始化。

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

完成后，应在 `data/AVEC2014_preprocessed/train|dev|test` 下看到若干 `.pt` 文件，每个文件包含：

- `full_faces`: `(T, 3, 256, 256)` 全脸序列张量（已归一化）；
- `local_faces`: `(T, 3, 256, 256)` 局部脸序列张量；
- `mask`: `(T,)` 帧掩码，True 表示 padding 帧；
- `label`: 标量抑郁分数。

### 3. 端到端训练（单卡）

```bash
CUDA_VISIBLE_DEVICES=1 python end2end_train.py
```

脚本会：

- 优先使用 `data/AVEC2014_preprocessed/` 中的预处理数据；若不存在则在线从原始视频抽帧 + MTCNN；
- 使用单卡训练，物理 `batch_size=1`，通过 `accumulation_steps=4` 实现等效 batch≈4；
- 使用 AMP 混合精度与梯度裁剪，自动跳过非有限 loss；
- 在 dev 集 MAE 最优时保存模型到 `checkpoints/best_dnet_model.pth`。

### 4. 小学习率微调

在得到初始最佳模型后，可用较小学习率做额外微调：

```bash
CUDA_VISIBLE_DEVICES=1 python end2end_finetune.py
```

脚本会从 `checkpoints/best_dnet_model.pth` 加载权重，使用学习率 `1e-5` 再训练若干 epoch，并在 dev 集 MAE 进一步下降时将模型保存到：

- `checkpoints/best_dnet_model_ft.pth`

### 5. 在 test 集上评估

```bash
CUDA_VISIBLE_DEVICES=1 python end2end_eval.py
```

脚本将优先加载 `best_dnet_model_ft.pth`，若不存在则退回 `best_dnet_model.pth`，并在 AVEC2014 test 集上输出：

- `Test Loss (MSE)`
- `Test MAE`
- `Test RMSE`

## 训练说明

当前端到端训练 / 微调脚本已经内置了以下稳定性策略：

- 训练和验证阶段都会跳过非有限 loss（NaN / Inf）；
- 使用 AMP 混合精度与 `torch.utils.checkpoint` 降低显存占用；
- 反向传播时使用梯度裁剪，减少数值爆炸风险；
- 单卡训练，避免多进程通信导致的不稳定因素；
- 使用梯度累积，在较小显存下实现等效较大 batch。

## 输出

训练 / 微调过程中会输出：

- `Train Loss`
- `Val Loss (MSE)`
- `Val MAE`
- `Val RMSE`

最终模型文件：

- 初始最佳端到端模型：`checkpoints/best_dnet_model.pth`
- 小学习率微调后最佳模型：`checkpoints/best_dnet_model_ft.pth`

评估脚本会在终端打印 test 集的 MAE / RMSE，可直接与 DNet 论文中的 AVEC2014 指标进行对比。
