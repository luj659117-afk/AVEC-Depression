import torch
from torch.utils.data import Dataset
import numpy as np
import os

class AVEC2014Dataset(Dataset):
    def __init__(self, metadata_dict, feature_dir, max_seq_len=1800):
        # metadata_dict: { 'Subject_ID': BDI_Score }
        self.subjects = list(metadata_dict.keys())
        self.labels = metadata_dict
        self.feature_dir = feature_dir
        self.max_len = max_seq_len # 假设 1 分半的视频按 20fps 采样，最多约 1800 帧

    def __len__(self):
        return len(self.subjects)

    def _load_and_pad(self, subject_id, task, modality):
        # 读取特征，如 "205_1_Freeform_RGB.npy"
        filename = f"{subject_id}_{task}_{modality}.npy"
        path = os.path.join(self.feature_dir, filename)
        
        try:
            feat = np.load(path)
        except Exception:
            feat = np.zeros((1, 2048)) # 文件缺失时的防御
            
        T = feat.shape[0]
        
        # 截断或填充
        if T > self.max_len:
            feat = feat[:self.max_len, :]
            mask = np.zeros(self.max_len, dtype=bool)
        else:
            pad_len = self.max_len - T
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
            # Mask: True 表示是填充的废数据，Transformer 计算时会忽略
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