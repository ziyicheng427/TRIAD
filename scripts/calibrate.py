#!/usr/bin/env python3
"""TRIAD-Cal CLI — targetless calibration with covariance output (Sec. 5.1).

Example
-------
    python scripts/calibrate.py \
        --cloud scans_accumulated.ply \
        --frames frames/ --intrinsics intrinsics.yaml \
        --nominal T_nominal.npy \
        --config configs/cal_outdoor.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triad.cal import CalConfig, TriadCalPipeline
from triad.cal.pipeline import CalFrame
from triad.cal.sam_adapter import MaskSet


def load_cloud(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    if path.endswith((".ply", ".pcd")):
        try:
            from open3d import io as o3d_io  # optional dependency
        except ImportError as exc:
            raise SystemExit("open3d is required for .ply/.pcd clouds") from exc
        cloud = o3d_io.read_point_cloud(path)
        return np.asarray(cloud.points, dtype=np.float64)
    raise SystemExit(f"unsupported cloud format: {path}")


def load_image(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to read images") from exc
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", required=True, help="accumulated point cloud")
    parser.add_argument("--frames", required=True,
                        help="directory of camera frames (image files)")
    parser.add_argument("--masks", default=None,
                        help="optional directory of precomputed SAM masks (.npy)")
    parser.add_argument("--intrinsics", required=True,
                        help="YAML with fx, fy, cx, cy")
    parser.add_argument("--nominal", required=True,
                        help=".npy holding the 4x4 CAD nominal extrinsic")
    parser.add_argument("--config", default=None, help="CalConfig YAML override")
    parser.add_argument("--out", default="outputs/calibration.npz")
    args = parser.parse_args()

    with open(args.intrinsics, "r", encoding="utf-8") as f:
        intr = yaml.safe_load(f)
    K = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    )

    config = CalConfig.from_yaml(args.config) if args.config else CalConfig()
    pipeline = TriadCalPipeline(config=config, K=K)

    frames = []
    names = sorted(os.listdir(args.frames))
    for name in names:
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image = load_image(os.path.join(args.frames, name))
        masks = None
        if args.masks is not None:
            mask_path = os.path.join(args.masks, os.path.splitext(name)[0] + ".npy")
            if os.path.exists(mask_path):
                data = np.load(mask_path, allow_pickle=True).item()
                masks = MaskSet(masks=data["masks"], scores=data["scores"])
        frames.append(CalFrame(image_rgb=image, masks=masks))

    cloud = load_cloud(args.cloud)
    T_nominal = np.load(args.nominal)

    result = pipeline.run(cloud, frames, T_nominal)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        T_lidar_cam=result.T_lidar_cam,
        covariance=result.covariance,
        coverage_warning=result.coverage_warning,
        correspondence_rate=result.correspondence_rate,
    )
    print(f"extrinsic saved to {args.out}")
    print(f"converged={result.converged} iterations={result.iterations} "
          f"correspondence_rate={result.correspondence_rate:.3f} "
          f"coverage_warning={result.coverage_warning}")


if __name__ == "__main__":
    main()
