import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from legacy.dataset import AVEC2014Dataset
import os
import glob
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import scipy.io

class TemporalStream(nn.Module):
# ... (existing code for TemporalStream is unchanged)
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x, padding_mask):
        x = self.projector(x)
        # 核心逻辑：传入 src_key_padding_mask 忽略全 0 补齐的部分
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # 使用掩码进行平均池化，排除 padding 部分的影响
        mask_float = (~padding_mask).unsqueeze(-1).float()
        pooled = (out * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        return pooled

class UltimateDepressionModel(nn.Module):
# ... (existing code for UltimateDepressionModel is unchanged)
    def __init__(self):
        super().__init__()
        # 为 4 种不同的模态/任务实例化独立的时序编码器
        self.ff_rgb_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.ff_flow_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.nw_rgb_stream = TemporalStream(input_dim=512, hidden_dim=256)
        self.nw_flow_stream = TemporalStream(input_dim=512, hidden_dim=256)
        
        # 融合层：4 * 256 = 1024 维
        self.regressor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)
        )

    def forward(self, ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask):
        f1 = self.ff_rgb_stream(ff_rgb, ff_mask)
        f2 = self.ff_flow_stream(ff_flow, ff_mask) # 共用 mask
        f3 = self.nw_rgb_stream(nw_rgb, nw_mask)
        f4 = self.nw_flow_stream(nw_flow, nw_mask)
        
        # 暴力特征拼接
        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        return self.regressor(fused_features)

def parse_labels(data_root='data/AVEC2014', label_dir='data/AVEC2014/label/DepressionLabels', split='train'):
    """
    根据目录结构确定指定分集（train/dev/test）的受试者，并从 DepressionLabels 文件夹读取分数。
    """
    split_path = os.path.join(data_root, split)
    # 查找该分集下的所有视频文件以确定受试者
    video_files = glob.glob(os.path.join(split_path, '**', '*.mp4'), recursive=True)
    
    subject_ids = set()
    for f in video_files:
        basename = os.path.basename(f)
        # 从文件名（如 '203_1_Freeform_video.mp4'）中提取 '203_1'
        subject_id = '_'.join(basename.split('_')[:2])
        subject_ids.add(subject_id)
        
    metadata = {}
    # 遍历找到的受试者ID
    for subject_id in sorted(list(subject_ids)): # 排序以保证顺序确定性
        csv_path = os.path.join(label_dir, f"{subject_id}_Depression.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as fin:
                try:
                    bdi_score = int(fin.read().strip())
                    metadata[subject_id] = float(bdi_score)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse score from {csv_path}, skipping.")
        else:
            # 如果一个受试者在分集目录中存在，但没有对应的标签文件，则打印警告
            print(f"Warning: Label file not found for subject {subject_id} in split '{split}', skipping.")
            
    return metadata

def validate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, labels = [b.to(device) for b in batch]
            
            preds = model(ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask)
            loss = criterion(preds.squeeze(), labels.squeeze())
            
            total_loss += loss.item()
            all_preds.extend(preds.squeeze().cpu().numpy().flatten())
            all_labels.extend(labels.squeeze().cpu().numpy().flatten())
            
    # 计算 MAE 和 RMSE
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    avg_loss = total_loss / len(loader)
    
    return avg_loss, mae, rmse

def train_and_validate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 数据准备 ---
    feature_dir = 'features_vggface'
    train_metadata = parse_labels(split='train')
    dev_metadata = parse_labels(split='dev')
    
    train_dataset = AVEC2014Dataset(train_metadata, feature_dir)
    dev_dataset = AVEC2014Dataset(dev_metadata, feature_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    # --- 模型初始化 ---
    model = UltimateDepressionModel()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) # 增加 weight_decay 抑制过拟合
    criterion = nn.SmoothL1Loss() 
    
    best_mae = float('inf')
    num_epochs = 100
    save_path = 'checkpoints'
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask)
            loss = criterion(preds.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- 验证阶段 ---
        val_loss, val_mae, val_rmse = validate(model, dev_loader, criterion, device)
        
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"           | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # --- 保存最优模型权重 ---
        if val_mae < best_mae:
            best_mae = val_mae
            # 处理 DataParallel 包装后的模型权重保存
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': val_mae,
                'rmse': val_rmse,
            }, os.path.join(save_path, 'best_depression_model.pth'))
            print(f"*** New Best Model Saved with MAE: {val_mae:.4f} ***")

if __name__ == '__main__':
    train_and_validate()
