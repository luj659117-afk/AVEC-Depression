import os
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm


class DualStreamVGGFaceExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Initializing feature extractor on device: {self.device}...')

        self.mtcnn = MTCNN(keep_all=False, device=self.device, margin=20)
        self.feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compute_optical_flow(self, prev_gray, curr_gray):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @torch.no_grad()
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Warning: could not open video {video_path}')
            return None, None

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None, None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        rgb_features = []
        flow_features = []

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

            boxes, _ = self.mtcnn.detect(curr_rgb)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(curr_rgb.shape[1], box[2]), min(curr_rgb.shape[0], box[3])

                if y2 - y1 >= 20 and x2 - x1 >= 20:
                    face_rgb = curr_rgb[y1:y2, x1:x2]
                    tensor_rgb = self.transform(face_rgb).unsqueeze(0).to(self.device)
                    rgb_feat = self.feature_extractor(tensor_rgb).squeeze(0).cpu().numpy()

                    flow_img = self.compute_optical_flow(prev_gray, curr_gray)
                    face_flow = flow_img[y1:y2, x1:x2]
                    tensor_flow = self.transform(face_flow).unsqueeze(0).to(self.device)
                    flow_feat = self.feature_extractor(tensor_flow).squeeze(0).cpu().numpy()

                    rgb_features.append(rgb_feat)
                    flow_features.append(flow_feat)

            prev_gray = curr_gray

        cap.release()

        if len(rgb_features) == 0:
            return None, None

        return np.asarray(rgb_features), np.asarray(flow_features)


def extract_all_features(data_root='data/AVEC2014', feature_out_dir='features_vggface'):
    os.makedirs(feature_out_dir, exist_ok=True)
    extractor = DualStreamVGGFaceExtractor()

    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(data_root, split)
        video_files = glob.glob(os.path.join(split_path, '**', '*.mp4'), recursive=True)
        print(f'Processing {split}: found {len(video_files)} videos')

        for video_path in tqdm(video_files, desc=f'Extracting {split}'):
            basename = os.path.basename(video_path)
            parts = basename.replace('_video.mp4', '').split('_')
            if len(parts) < 3:
                print(f'Warning: unexpected filename format, skipping {basename}')
                continue

            sample_id = '_'.join(parts[:2])
            task = parts[2]

            rgb_out_path = os.path.join(feature_out_dir, f'{sample_id}_{task}_RGB.npy')
            flow_out_path = os.path.join(feature_out_dir, f'{sample_id}_{task}_Flow.npy')

            if os.path.exists(rgb_out_path) and os.path.exists(flow_out_path):
                continue

            rgb_feats, flow_feats = extractor.process_video(video_path)
            if rgb_feats is None or flow_feats is None:
                print(f'Warning: failed to extract features for {video_path}')
                continue

            np.save(rgb_out_path, rgb_feats)
            np.save(flow_out_path, flow_feats)


if __name__ == '__main__':
    extract_all_features()
