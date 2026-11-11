#!/usr/bin/env python3
"""TRIAD-Reg evaluation CLI — KITTI protocol of Sec. 5.2.

Evaluates a checkpoint on the KITTI odometry test split (sequences 08–10)
and reports RTE / RRE / RR, optionally bucketed by ground-truth overlap
(Table 7: 10–30 %, 30–50 %, > 50 %).

    python scripts/evaluate_registration.py \
        --kitti-root /path/to/kitti --sequences 08 09 10 \
        --checkpoint checkpoints/triad_reg.pt --overlap-buckets
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TRIAD-Reg evaluation requires torch: pip install -e '.[reg]'") from exc

from triad.common.se3 import SE3
from triad.evaluation import evaluate_pairs
from triad.reg import RegConfig, TRIADRegModel, load_kitti_poses, load_kitti_scan
from triad.reg.train import TEST_SEQUENCES, overlap_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", required=True)
    parser.add_argument("--sequences", nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--pair-gap", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--overlap-buckets", action="store_true")
    args = parser.parse_args()

    config = RegConfig.from_yaml(args.config) if args.config else RegConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TRIADRegModel(config).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    bucket_names = ["10-30", "30-50", ">50"]
    buckets: dict = {name: ([], []) for name in bucket_names}
    all_pred, all_gt = [], []

    with torch.no_grad():
        for sequence in args.sequences:
            poses = load_kitti_poses(args.kitti_root, sequence)
            for i in range(0, len(poses) - args.pair_gap, args.stride):
                j = i + args.pair_gap
                src = load_kitti_scan(args.kitti_root, sequence, i)
                tgt = load_kitti_scan(args.kitti_root, sequence, j)
                T_gt = SE3.inverse(poses[j]) @ poses[i]

                src_t = torch.as_tensor(src, dtype=torch.float32, device=device)
                tgt_t = torch.as_tensor(tgt, dtype=torch.float32, device=device)
                T_pred = model(src_t, tgt_t).transform.cpu().numpy()

                all_pred.append(T_pred)
                all_gt.append(T_gt)
                if args.overlap_buckets:
                    ov = overlap_ratio(src, tgt)
                    if ov < 0.10:
                        continue
                    bucket = "10-30" if ov < 0.30 else "30-50" if ov < 0.50 else ">50"
                    buckets[bucket][0].append(T_pred)
                    buckets[bucket][1].append(T_gt)

    print("overall:", evaluate_pairs(all_pred, all_gt,
                                     config.rte_thresh_m, config.rre_thresh_deg))
    if args.overlap_buckets:
        for name in bucket_names:
            preds, gts = buckets[name]
            if preds:
                print(f"overlap {name}%:",
                      evaluate_pairs(preds, gts, config.rte_thresh_m,
                                     config.rre_thresh_deg))


if __name__ == "__main__":
    main()
