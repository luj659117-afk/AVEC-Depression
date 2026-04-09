import os
import torch

from data.end2end_dataset import End2EndAVEC2014Dataset


@torch.no_grad()
def preprocess_split(split: str, out_root: str = "data/AVEC2014_preprocessed", max_frames: int = 48, image_size: int = 256) -> None:
    """离线预处理 AVEC2014：

    - 按论文设置从原始视频中抽帧、MTCNN 检测 + 仿射对齐
    - 裁剪全脸 / 局部脸，并 resize+padding 到 256x256
    - 进行时间采样和 padding，得到 (T, 3, 256, 256) 的 full/local 与 mask
    - 将每个样本保存为一个 .pt 文件，供训练阶段直接加载
    """

    dataset = End2EndAVEC2014Dataset(
        split=split,
        device="cpu",  # 预处理统一在 CPU 上跑 MTCNN
        max_frames=max_frames,
        image_size=image_size,
    )

    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing split='{split}' to '{out_dir}' ({len(dataset)} samples)...")

    for video_path, sid, task in dataset.samples:
        key = f"{sid}_{task}"
        out_path = os.path.join(out_dir, f"{key}.pt")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        label = float(dataset.labels[key])
        full_faces, local_faces, mask = dataset._process_video(video_path)
        if full_faces is None:
            print(f"[warn] failed to process video '{video_path}', skipping")
            continue

        data = {
            "full_faces": full_faces,
            "local_faces": local_faces,
            "mask": mask,
            "label": label,
        }
        torch.save(data, out_path)
        print(f"[ok] saved {out_path} | frames={full_faces.shape[0]}")


def main() -> None:
    for split in ["train", "dev", "test"]:
        preprocess_split(split)


if __name__ == "__main__":
    main()
