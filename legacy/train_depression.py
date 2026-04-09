import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import glob

# ================= 1. 数据集加载与防御逻辑 =================
class AVEC2014Dataset(Dataset):
    def __init__(self, metadata_dict, feature_dir, max_seq_len=1800):
        self.subjects = list(metadata_dict.keys())
        self.labels = metadata_dict
        self.feature_dir = feature_dir
        self.max_len = max_seq_len

    def __len__(self):
        return len(self.subjects)

    def _load_and_pad(self, subject_id, task, modality):
        filename = f"{subject_id}_{task}_{modality}.npy"
        path = os.path.join(self.feature_dir, filename)
        
        try:
            feat = np.load(path)
            # 逻辑防线：光流计算在遇到纯黑帧或剧烈抖动时可能产生 NaN
            # 必须在输入网络前将其抹平为 0，否则整个模型的权重会瞬间崩溃
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"警告: 无法读取 {path}, 使用零矩阵填充。")
            feat = np.zeros((1, 512)) # 注意这里改成了 512
            
        T = feat.shape[0]
        
        if T > self.max_len:
            feat = feat[:self.max_len, :]
            mask = np.zeros(self.max_len, dtype=bool)
        else:
            pad_len = self.max_len - T
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
            mask = np.concatenate([np.zeros(T, dtype=bool), np.ones(pad_len, dtype=bool)])
            
        return torch.FloatTensor(feat), torch.BoolTensor(mask)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        
        ff_rgb, ff_mask = self._load_and_pad(sub, 'Freeform', 'RGB')
        ff_flow, _ = self._load_and_pad(sub, 'Freeform', 'Flow')
        nw_rgb, nw_mask = self._load_and_pad(sub, 'Northwind', 'RGB')
        nw_flow, _ = self._load_and_pad(sub, 'Northwind', 'Flow')
        
        label = torch.FloatTensor([self.labels[sub]])
        return ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, label

def parse_labels(data_root='data/AVEC2014', label_dir='data/AVEC2014/label/DepressionLabels', split='train'):
    split_path = os.path.join(data_root, split)
    video_files = glob.glob(os.path.join(split_path, '**', '*.mp4'), recursive=True)

    sample_ids = set()
    for video_path in video_files:
        basename = os.path.basename(video_path)
        parts = basename.replace('_video.mp4', '').split('_')
        if len(parts) >= 2:
            sample_ids.add('_'.join(parts[:2]))

    metadata = {}
    for sample_id in sorted(sample_ids):
        csv_path = os.path.join(label_dir, f'{sample_id}_Depression.csv')
        if not os.path.exists(csv_path):
            print(f'警告: 找不到标签文件 {csv_path}，跳过。')
            continue

        try:
            with open(csv_path, 'r') as f:
                score = float(f.read().strip())
            metadata[sample_id] = score
        except Exception:
            print(f'警告: 读取标签失败 {csv_path}，跳过。')

    return metadata

# ================= 2. 四路网络架构 (降维修正版) =================
class TemporalStream(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x, padding_mask):
        x = self.projector(x)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        mask_float = (~padding_mask).unsqueeze(-1).float()
        pooled = (out * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        return pooled

class CrossModalFusion(nn.Module):
    def __init__(self, feature_dim=256, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.3,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
        )

    def forward(self, f1, f2, f3, f4):
        stacked_features = torch.stack([f1, f2, f3, f4], dim=1)
        interacted_features = self.fusion_transformer(stacked_features)
        flattened = interacted_features.reshape(interacted_features.size(0), -1)
        return self.regressor(flattened)


class UltimateDepressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_rgb_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.ff_flow_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.nw_rgb_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.nw_flow_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.fusion_layer = CrossModalFusion(feature_dim=256)

    def forward(self, ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask):
        f1 = self.ff_rgb_stream(ff_rgb, ff_mask)
        f2 = self.ff_flow_stream(ff_flow, ff_mask)
        f3 = self.nw_rgb_stream(nw_rgb, nw_mask)
        f4 = self.nw_flow_stream(nw_flow, nw_mask)
        return self.fusion_layer(f1, f2, f3, f4)


class CostSensitiveSmoothL1Loss(nn.Module):
    def __init__(self, alpha=2.0, max_score=63.0):
        super().__init__()
        self.alpha = alpha
        self.max_score = max_score

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        base_loss = F.smooth_l1_loss(preds, targets, reduction='none')
        weights = 1.0 + self.alpha * (targets / self.max_score)
        weighted_loss = torch.mean(weights * base_loss)
        return weighted_loss

# ================= 3. 训练与验证核心逻辑 =================
def validate(model, loader, eval_criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    valid_steps = 0
    
    with torch.no_grad():
        for batch in loader:
            ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, labels = [b.to(device) for b in batch]
            preds = model(ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask)
            loss = eval_criterion(preds.view(-1), labels.view(-1))

            if not torch.isfinite(loss):
                continue
            
            total_loss += loss.item()
            valid_steps += 1
            all_preds.extend(preds.squeeze().cpu().numpy().flatten())
            all_labels.extend(labels.squeeze().cpu().numpy().flatten())
            
    if valid_steps == 0 or len(all_preds) == 0:
        return float('inf'), float('inf'), float('inf')

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    return total_loss / valid_steps, mae, rmse

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 直接从数据目录和 DepressionLabels 中解析 train/dev 样本及其分数。
    train_metadata = parse_labels(split='train')
    dev_metadata = parse_labels(split='dev')

    feature_dir = 'features_vggface'

    if not train_metadata or not dev_metadata:
        print('致命错误: 未解析到有效的 train/dev 标签，请检查数据目录与标签文件。')
        return

    train_dataset = AVEC2014Dataset(train_metadata, feature_dir)
    dev_dataset = AVEC2014Dataset(dev_metadata, feature_dir)
    
    # 限制 Batch Size，避免极端的小批量梯度方差
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    model = UltimateDepressionModelV2()
    
    # 优先只用前 4 张卡，避免 8 卡切分后单卡 batch 过小。
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 4:
        print('Using first 4 GPUs')
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
    elif gpu_count > 1:
        print(f'Using {gpu_count} GPUs')
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = CostSensitiveSmoothL1Loss(alpha=2.0, max_score=63.0)
    eval_criterion = nn.SmoothL1Loss()
    
    best_mae = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    print("开始训练...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask)
            loss = criterion(preds, labels)

            if not torch.isfinite(loss):
                print('警告: 检测到非有限 loss，跳过该 batch。')
                continue

            loss.backward()
            
            # 逻辑防线：光流梯度裁剪，防止偶尔的突刺导致 Loss 变成 NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) == 0:
            print(f'Epoch {epoch:02d} | 训练集没有有效 batch，跳过。')
            continue
            
        val_loss, val_mae, val_rmse = validate(model, dev_loader, eval_criterion, device)
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, 'checkpoints/best_vggface_model.pth')
            print(f"  >>> 新的最优模型已保存 (MAE: {val_mae:.4f})")

if __name__ == '__main__':
    main()