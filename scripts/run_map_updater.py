#!/usr/bin/env python3
"""TRIAD-Map CLI — replay scans through the updater (Sec. 5.3).

    python scripts/run_map_updater.py --config configs/map.yaml \
        --scans scans/ --mode pgc_tbb --sigma-t 0.04 --out outputs/map.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triad.link import l2_gains
from triad.map import LogOddsGrid, MapConfig, TriadMapUpdater, UpdateMode


def load_scan(path: str) -> np.ndarray:
    return np.load(path).astype(np.float64)[:, :2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--scans", required=True,
                        help="directory of .npy scans (N,2) in sensor frame")
    parser.add_argument("--mode", default="pgc_tbb",
                        choices=[m.value for m in UpdateMode])
    parser.add_argument("--origin", nargs=2, type=float, default=(-10.0, -10.0))
    parser.add_argument("--shape", nargs=2, type=int, default=(400, 400))
    parser.add_argument("--sigma-t", type=float, default=None,
                        help="constant registration pose std for L2 damping")
    parser.add_argument("--out", default="outputs/map.npz")
    args = parser.parse_args()

    config = MapConfig.from_yaml(args.config) if args.config else MapConfig()
    grid = LogOddsGrid(config, width=args.shape[0], height=args.shape[1],
                       origin=tuple(args.origin))
    updater = TriadMapUpdater(grid, config, mode=UpdateMode(args.mode))

    hit_gain = miss_gain = 1.0
    if args.sigma_t is not None:
        hit_gain, miss_gain = l2_gains(args.sigma_t)

    origin_sensor = np.zeros(2)
    for name in sorted(os.listdir(args.scans)):
        if not name.endswith(".npy"):
            continue
        endpoints = load_scan(os.path.join(args.scans, name))
        updater.update(endpoints, origin_sensor,
                       hit_gain=hit_gain, miss_gain=miss_gain)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, probability=grid.probability, log_odds=grid.log_odds)
    print(f"map saved to {args.out}; occupied cells: {int(grid.occupied_mask().sum())}")


if __name__ == "__main__":
    main()
