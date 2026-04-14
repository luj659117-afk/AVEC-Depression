import os
import argparse
import torch

from data.end2end_dataset import End2EndAVEC2014Dataset


@torch.no_grad()
def preprocess_split(
    split: str,
    out_root: str = "data/AVEC2014_preprocessed",
    max_frames: int = 48,
    image_size: int = 256,
    temporal_sample: str = "legacy",
    device: str = "cpu",
    overwrite: bool = False,
) -> None:
    """离线预处理 AVEC2014：

    - 按论文设置从原始视频中抽帧、MTCNN 检测 + 仿射对齐
    - 裁剪全脸 / 局部脸，并 resize+padding 到 256x256
    - 进行时间采样和 padding，得到 (T, 3, 256, 256) 的 full/local 与 mask
    - 将每个样本保存为一个 .pt 文件，供训练阶段直接加载
    """

    dataset = End2EndAVEC2014Dataset(
        split=split,
        device=device,
        max_frames=max_frames,
        image_size=image_size,
        temporal_sample=temporal_sample,
    )

    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preprocessing split='{split}' to '{out_dir}' ({len(dataset)} samples)...")

    for video_path, sid, task in dataset.samples:
        key = f"{sid}_{task}"
        out_path = os.path.join(out_dir, f"{key}.pt")
        if os.path.exists(out_path) and not overwrite:
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
    parser = argparse.ArgumentParser(description="Preprocess AVEC2014 videos into full/local face tensors.")
    parser.add_argument("--out-root", default="data/AVEC2014_preprocessed")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--temporal-sample", choices=["legacy", "uniform", "random"], default="legacy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    args = parser.parse_args()

    for split in args.splits:
        preprocess_split(
            split=split,
            out_root=args.out_root,
            max_frames=args.max_frames,
            image_size=args.image_size,
            temporal_sample=args.temporal_sample,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
