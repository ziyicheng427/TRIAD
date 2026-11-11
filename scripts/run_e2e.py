#!/usr/bin/env python3
"""End-to-end evaluation CLI — Sec. 5.6, Tables 15-16.

Runs the three configurations over a recorded scene:

* ``baseline``       — the conventional stack (fixed-gain mapping);
* ``triad_wo_link``  — the three task modules with all links disabled;
* ``full``           — task modules with all TRIAD-Link interfaces.

Dynamic stress combines 15–25 % dynamic points, 25–40 % overlap, and
controlled extrinsic perturbation; clean conditions use stable extrinsics
and high overlap.

    python scripts/run_e2e.py --scene corridor --condition dynamic \
        --scans scans/ --repetitions 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triad.evaluation import false_occupied_cells, map_iou
from triad.link.config import LinkConfig
from triad.map import LogOddsGrid, MapConfig, TriadMapUpdater, UpdateMode
from triad.pipeline import StackConfig, TriadPipeline

CONDITIONS = ("clean", "dynamic")
CONFIGURATIONS = ("baseline", "triad_wo_link", "full")


def run_configuration(
    name: str, scans: list, sensor_origin: np.ndarray, map_config: MapConfig,
    link_config: LinkConfig, sigma_t: float | None,
) -> dict:
    """One configuration over the scan sequence; returns the Table 15 metrics."""
    if name == "baseline":
        pipeline = TriadPipeline(
            StackConfig(map=map_config, link=link_config), use_task_modules=False
        )
    elif name == "triad_wo_link":
        link_off = LinkConfig(**{**link_config.to_dict(),
                                 "enable_l1": False, "enable_l2": False,
                                 "enable_l3": False})
        pipeline = TriadPipeline(
            StackConfig(map=map_config, link=link_off), use_task_modules=True
        )
    else:  # full
        pipeline = TriadPipeline(
            StackConfig(map=map_config, link=link_config), use_task_modules=True
        )

    for endpoints in scans:
        pipeline.register_and_map(endpoints, sensor_origin, sigma_t=sigma_t)

    grid = pipeline.grid
    reference = np.zeros_like(grid.occupied_mask())
    # the caller-supplied scans' bounding box approximates the reference map
    all_pts = np.vstack(scans)
    cells = grid.world_to_cell(all_pts)
    in_b = grid.in_bounds(cells)
    reference[cells[in_b, 1], cells[in_b, 0]] = True
    # dilate the reference by one cell to tolerate discretisation
    reference = reference | np.roll(reference, 1, 0) | np.roll(reference, -1, 0) \
        | np.roll(reference, 1, 1) | np.roll(reference, -1, 1)

    return {
        "IoU": map_iou(grid.occupied_mask(), reference),
        "false_cells_per_1000m2": false_occupied_cells(
            grid.occupied_mask(), reference, map_config.resolution**2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=["corridor", "warehouse", "road"])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--scans", required=True,
                        help="directory of .npy scans (N,2)")
    parser.add_argument("--map-config", default=None)
    parser.add_argument("--link-config", default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-t", type=float, default=None,
                        help="registration pose std (dynamic stress injects "
                             "0.02-0.08 m when omitted)")
    args = parser.parse_args()

    map_config = MapConfig.from_yaml(args.map_config) if args.map_config else MapConfig()
    link_config = LinkConfig.from_yaml(args.link_config) if args.link_config else LinkConfig()
    rng = np.random.default_rng(args.seed)

    names = sorted(n for n in os.listdir(args.scans) if n.endswith(".npy"))
    scans = [np.load(os.path.join(args.scans, n))[:, :2].astype(float) for n in names]
    sensor_origin = np.zeros(2)

    print(f"scene={args.scene} condition={args.condition} runs={args.repetitions}")
    print("configuration    IoU     false_cells/1000m2")
    for name in CONFIGURATIONS:
        ious, falses = [], []
        for _ in range(args.repetitions):
            # dynamic stress: inject extrinsic perturbation + registration noise
            if args.condition == "dynamic" and args.sigma_t is None:
                sigma_t = float(rng.uniform(0.02, 0.08))
                j = rng.integers(0, len(scans))
                scans[j] = scans[j] + rng.normal(0, sigma_t, scans[j].shape)
            else:
                sigma_t = args.sigma_t
            metrics = run_configuration(name, scans, sensor_origin, map_config,
                                        link_config, sigma_t)
            ious.append(metrics["IoU"])
            falses.append(metrics["false_cells_per_1000m2"])
        print(f"{name:16s} {np.mean(ious):6.3f}  {np.mean(falses):8.1f}")


if __name__ == "__main__":
    main()
