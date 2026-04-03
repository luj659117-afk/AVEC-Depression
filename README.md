# AVEC2014 Depression Recognition

本项目用于基于 AVEC2014 视频数据集进行抑郁程度回归预测。当前流程分为两部分：

1. 离线提取视频特征
2. 使用时序模型进行训练与验证
具体如下：

特征提取：使用 facenet-pytorch 的 MTCNN 检测人脸，VGGFace2 预训练 InceptionResnetV1 提取 512 维人脸特征，光流帧同样提特征，全部离线保存为 .npy。
模型结构：
四路 TemporalStream（基于 Transformer Encoder），分别处理 Freeform/Northwind 两种任务下的 RGB/Flow 特征，每路输出 256 维。
四路输出拼接（共 1024 维），送入回归头（两层全连接+LayerNorm+ReLU+Dropout），输出单一抑郁分数。
训练细节：
损失函数：MSELoss，评估指标为 MAE、RMSE。
优化器：AdamW，weight_decay=1e-3。
正则化：回归头用 LayerNorm 和 Dropout(p=0.6)。
多卡训练：默认用前 4 张 GPU，支持 DataParallel。
梯度裁剪，自动跳过 NaN/Inf loss，特征中的 NaN/Inf 会被清零。
如需查看完整模型代码，请参考 main.py 中 UltimateDepressionModel 和 TemporalStream 的实现。

## 项目结构

- `extract_vggface_features.py`：使用 MTCNN + VGGFace2 预训练的 `InceptionResnetV1` 提取 RGB / Flow 特征，保存为 `.npy`
- `train_depression.py`：独立训练脚本，自动解析 `train` / `dev` 划分并读取标签
- `dataset.py`：通用数据集读取逻辑
- `main.py`：主训练脚本，保留了当前实验使用的模型结构与训练流程
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
