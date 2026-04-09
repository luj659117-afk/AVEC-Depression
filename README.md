# AVEC2014 Depression Recognition

本项目用于基于 AVEC2014 视频数据集进行抑郁程度回归预测。目前包含两套流程：

1. 旧版：离线提取视频特征 + 时序回归模型（保存在 `legacy/` 目录，仅作对比基线）。
2. 新版：端到端特征学习网络（FEM + FPN + 时序 Transformer + ViT Block）。

旧版（legacy）流程：
- 特征提取：使用 facenet-pytorch 的 MTCNN 检测人脸，VGGFace2 预训练 InceptionResnetV1 提取 512 维人脸特征，光流帧同样提特征，全部离线保存为 .npy。
- 模型结构：四路 TemporalStream（基于 Transformer Encoder），分别处理 Freeform/Northwind 两种任务下的 RGB/Flow 特征，每路输出 256 维，四路输出拼接（1024 维）后接回归头。

新版（端到端）流程：
- 数据预处理：直接从原始 AVEC2014 视频中每 3 帧抽 1 帧，使用 MTCNN + OpenCV 仿射变换做人脸对齐；对每帧进行双尺度裁剪（全脸 / 局部面部），并在缩放到 256×256 时采用黑边 Padding 保持宽高比。
- 空间骨干：双路 FEM（交替 ConvBlock + InBlock + Coordinate Attention + DenseBlock）分别处理全脸和局部脸特征，特征在通道维拼接后送入 FPN 做多尺度融合，输出 320 通道特征。
- 时序与局部-全局提纯：将每帧 320 维特征序列送入 2 层 Transformer Encoder 建模帧间动态，并使用 Temporal Pooling 聚合为视频级表示；再送入自定义 ViT Block（Unfold→Transformer→Fold）进一步提炼局部与全局高级语义特征。
- 回归头：在视频级特征后接 1×1 卷积 + 全连接层输出单一抑郁分数，训练中使用 Adam（β1=0.9, β2=0.999）、学习率 1e-4、Batch Size=4、weight_decay=5e-5，损失函数为 MSELoss。

## 项目结构

- `legacy/extract_vggface_features.py`：旧版，使用 MTCNN + VGGFace2 提取 RGB / Flow 离线特征，保存为 `.npy`
- `legacy/train_depression.py`：旧版训练脚本，自动解析 `train` / `dev` 划分并读取标签
- `legacy/dataset.py`：旧版通用数据集读取逻辑（基于预提取特征）
- `legacy/main.py`：旧版主训练脚本（四路 TemporalStream + 回归头）
- `data/end2end_dataset.py`：新版端到端 AVEC2014 数据集，包含抽帧、MTCNN 检测与对齐、双尺度裁剪和 Padding
- `models/fem_fpn.py`：FEM（含 Coordinate Attention + DenseBlock）与 FPN 的双路空间骨干实现
- `models/temporal_vit.py`：时序 Transformer + ViT Block + 回归头的端到端回归模型
- `end2end_train.py`：新版端到端训练脚本
- `data/AVEC2014/`：数据集根目录

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

## 特征文件格式

离线提取后的特征默认保存到：

- `features_vggface/`

文件命名格式：

- `[SubjectID]_[Task]_RGB.npy`
- `[SubjectID]_[Task]_Flow.npy`

示例：

- `203_1_Freeform_RGB.npy`
- `203_1_Freeform_Flow.npy`
- `203_1_Northwind_RGB.npy`
- `203_1_Northwind_Flow.npy`

每个 `.npy` 文件应为二维数组，形状类似：

- `RGB`: `(T, 512)`
- `Flow`: `(T, 512)`

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

### 1. 激活环境

```bash
source /data/hcf/miniconda3/etc/profile.d/conda.sh
conda activate depress_cv
```

### 2. 提取特征

```bash
python extract_vggface_features.py
```

提取完成后，检查 `features_vggface/` 下是否已经生成 `.npy` 文件。

### 3. 训练模型

```bash
python train_depression.py
```

该脚本会：

- 自动扫描 `train` 和 `dev` 目录中的视频样本
- 从 `DepressionLabels` 中读取对应分数
- 使用前 4 张 GPU 进行训练（如果可用）
- 保存验证集 MAE 最优的模型到 `checkpoints/best_vggface_model.pth`

## 训练说明

当前训练脚本已经内置了以下稳定性策略：

- 特征中的 `NaN / Inf` 会被清理为 0
- 回归头使用 `LayerNorm`，避免多卡小 batch 下 `BatchNorm` 报错
- 训练和验证阶段都会跳过非有限 loss
- 反向传播时使用梯度裁剪，减少数值爆炸风险
- 默认只使用前 4 张 GPU，避免 8 卡切分导致单卡 batch 过小

## 输出

训练过程中会输出：

- `Train Loss`
- `Val MAE`
- `Val RMSE`

最优模型会保存到：

- `checkpoints/best_vggface_model.pth`

## 备注

如果你后续切换了特征提取器，需要同步检查：

- 特征维度是否仍为 512
- `train_depression.py` 中的数据加载维度是否匹配
- 特征输出目录是否仍为 `features_vggface/`
