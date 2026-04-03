import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import glob
from tqdm import tqdm

class TwoStreamExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=False, device=self.device, margin=20)
        
        # 共享特征提取器 (使用同一套 ResNet50 权重分别提取 RGB 和 Flow)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.backbone.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_optical_flow(self, prev_gray, curr_gray):
        # 使用 Farneback 算法计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 将光流 (dx, dy) 映射为 HSV 颜色空间，再转为 RGB，以便输入 ResNet
        hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb_flow

    @torch.no_grad()
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
            
        ret, prev_frame = cap.read()
        if not ret: return None, None
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        rgb_features, flow_features = [], []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in range(frame_count - 1):
            ret, curr_frame = cap.read()
            if not ret: break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 裁剪人脸 (基于 RGB 帧)
            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            
            # MTCNN 在 BGR 格式上训练，但 facenet-pytorch 内部会转换
            # 为了安全起见，我们传入 RGB
            boxes, _ = self.mtcnn.detect(curr_rgb)
            
            if boxes is not None:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(curr_rgb.shape[1], box[2]), min(curr_rgb.shape[0], box[3])
                
                if y2 - y1 < 20 or x2 - x1 < 20: continue
                
                # 2. 截取 RGB 面部并提特征
                face_rgb = curr_rgb[y1:y2, x1:x2]
                tensor_rgb = self.transform(face_rgb).unsqueeze(0).to(self.device)
                rgb_feat = self.backbone(tensor_rgb).squeeze().cpu().numpy()
                
                # 3. 计算光流，截取面部并提特征
                flow_img = self.compute_optical_flow(prev_gray, curr_gray)
                face_flow = flow_img[y1:y2, x1:x2]
                tensor_flow = self.transform(face_flow).unsqueeze(0).to(self.device)
                flow_feat = self.backbone(tensor_flow).squeeze().cpu().numpy()
                
                rgb_features.append(rgb_feat)
                flow_features.append(flow_feat)
                
            prev_gray = curr_gray
            
        cap.release()
        if len(rgb_features) == 0: return None, None
        # 返回形状均为 (T, 2048)
        return np.array(rgb_features), np.array(flow_features)

def extract_all_features(data_root='data/AVEC2014', feature_out_dir='features'):
    """
    遍历数据集所有视频，提取 RGB 和 Flow 特征并保存。
    """
    if not os.path.exists(feature_out_dir):
        os.makedirs(feature_out_dir)
        
    extractor = TwoStreamExtractor()
    
    # 遍历 train, dev, test
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(data_root, split)
        
        # 查找所有 .mp4 文件
        video_files = glob.glob(os.path.join(split_path, '**', '*.mp4'), recursive=True)
        
        print(f"Processing {split} set, found {len(video_files)} videos...")
        
        for video_path in tqdm(video_files):
            basename = os.path.basename(video_path)
            
            # e.g., 203_1_Freeform_video.mp4
            parts = basename.replace('_video.mp4', '').split('_')
            subject_id = f"{parts[0]}_{parts[1]}" # "203_1"
            task = parts[2] # "Freeform" or "Northwind"
            
            # 定义输出文件名
            rgb_out_path = os.path.join(feature_out_dir, f"{subject_id}_{task}_RGB.npy")
            flow_out_path = os.path.join(feature_out_dir, f"{subject_id}_{task}_Flow.npy")
            
            # 如果文件已存在，则跳过
            if os.path.exists(rgb_out_path) and os.path.exists(flow_out_path):
                continue

            # 提取特征
            rgb_feats, flow_feats = extractor.process_video(video_path)
            
            if rgb_feats is not None and flow_feats is not None:
                np.save(rgb_out_path, rgb_feats)
                np.save(flow_out_path, flow_feats)
            else:
                print(f"Warning: Failed to extract features for {video_path}")


if __name__ == '__main__':
    # 调用主函数开始提取
    extract_all_features()