import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


def save_vis_png(
    depth: np.ndarray,
    out_path: Path,
    vis_max: float,
    vis_mode: str,
    vis_min_percentile: float,
    vis_max_percentile: float,
) -> None:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if vis_mode == "normalize":
        valid = depth[np.isfinite(depth) & (depth > 0.0)]
        if valid.size > 16:
            dmin = float(np.percentile(valid, vis_min_percentile))
            dmax = float(np.percentile(valid, vis_max_percentile))
            if dmax <= dmin:
                dmax = dmin + 1e-6
            depth_norm = np.clip((depth - dmin) / (dmax - dmin), 0.0, 1.0)
            vis = (depth_norm * 255.0).astype(np.uint8)
        else:
            depth_clip = np.clip(depth, 0.0, vis_max)
            vis = (depth_clip / max(vis_max, 1e-6) * 255.0).astype(np.uint8)
    else:
        depth_clip = np.clip(depth, 0.0, vis_max)
        vis = (depth_clip / max(vis_max, 1e-6) * 255.0).astype(np.uint8)

    Image.fromarray(vis).save(out_path)


def build_intrinsics(
    fx: float, fy: float, cx: float, cy: float, device: torch.device
) -> Pinhole:
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    return Pinhole(K=k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch infer depth from extracted frames using UniDepthV2."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input frames dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root dir")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="model/unidepth-v2-vitl14",
        help="HF model id or local model directory.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vis_max", type=float, default=20.0)
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["fixed", "normalize"],
        default="normalize",
        help="Visualization mode: fixed (0~vis_max) or per-frame normalization.",
    )
    parser.add_argument(
        "--vis_min_percentile",
        type=float,
        default=2.0,
        help="Lower percentile used in normalize mode.",
    )
    parser.add_argument(
        "--vis_max_percentile",
        type=float,
        default=98.0,
        help="Upper percentile used in normalize mode.",
    )
    parser.add_argument("--fx", type=float, default=505.038)
    parser.add_argument("--fy", type=float, default=504.937)
    parser.add_argument("--cx", type=float, default=1080.938)
    parser.add_argument("--cy", type=float, default=1080.776)
    parser.add_argument("--print_every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "depth_npy"
    vis_dir = output_dir / "depth_vis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    frame_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {input_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, fallback to CPU.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_source = str(Path(args.model_name_or_path)) if Path(args.model_name_or_path).exists() else args.model_name_or_path
    print(f"Loading UniDepthV2 from: {model_source}")
    model = UniDepthV2.from_pretrained(model_source).to(device).eval()

    camera = build_intrinsics(args.fx, args.fy, args.cx, args.cy, device)
    print(
        "Using Cam1 intrinsics: "
        f"fx={args.fx:.3f}, fy={args.fy:.3f}, cx={args.cx:.3f}, cy={args.cy:.3f}"
    )
    print(
        f"Visualization mode: {args.vis_mode} "
        f"(vis_max={args.vis_max}, pmin={args.vis_min_percentile}, pmax={args.vis_max_percentile})"
    )
    print(f"Found {len(frame_paths)} frames in {input_dir}")

    with torch.inference_mode():
        for i, frame_path in enumerate(frame_paths, 1):
            rgb_np = np.array(Image.open(frame_path).convert("RGB"))
            rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous().to(device)

            preds = model.infer(rgb, camera)
            depth = preds["depth"].squeeze().float().cpu().numpy()

            stem = frame_path.stem
            np.save(raw_dir / f"{stem}.npy", depth)
            save_vis_png(
                depth=depth,
                out_path=vis_dir / f"{stem}.png",
                vis_max=args.vis_max,
                vis_mode=args.vis_mode,
                vis_min_percentile=args.vis_min_percentile,
                vis_max_percentile=args.vis_max_percentile,
            )

            if (i % args.print_every == 0) or (i == len(frame_paths)):
                print(f"[{i}/{len(frame_paths)}] done")

    print("Depth inference completed.")
    print(f"Depth npy directory: {raw_dir}")
    print(f"Depth visualization directory: {vis_dir}")


if __name__ == "__main__":
    main()
