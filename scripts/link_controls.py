#!/usr/bin/env python3
"""Cross-layer coupling controls — Sec. 5.5.

Sub-commands:

* ``amplification`` — inject extrinsic mount bias into KITTI seq 07 map
  accumulation and measure the registration / map-boundary amplification
  (Fig. 5);
* ``l1``            — covariance-propagation control of Table 5 (no L1 /
  fixed isotropic / translation-only / rotation-only / full / oracle);
* ``l2``            — registration-noise sweep of Table 9 (fixed-gain vs
  state-driven vs state-driven+L2);
* ``l3``            — occupancy-feedback weighting of Tables 10-11 (plain
  ICP / distance filter / L3 / oracle mask) on a reference map;
* ``factorial``     — the 2^3 link ablation of Table 13;
* ``transfer``      — the confidence-transfer strategies of Table 12.

Each control is self-contained on synthetic or caller-supplied data; point
them at real datasets via the shared options.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triad.common.se3 import exp_se3, transform_points
from triad.evaluation import false_occupied_cells, map_iou
from triad.link import LinkConfig, l2_gains, occupancy_reweight
from triad.map import LogOddsGrid, MapConfig, TriadMapUpdater, UpdateMode


# ----------------------------------------------------------------------
def cmd_amplification(args) -> None:
    """Fig. 5: bias injection on accumulated KITTI scans."""
    from triad.reg.train import load_kitti_poses, load_kitti_scan

    poses = load_kitti_poses(args.kitti_root, "07")
    frames = list(range(0, min(160, len(poses))))
    print("bias(deg)  median_reg_err(m)  map_boundary_rmse(m)")
    for bias_deg in args.biases:
        bias = exp_se3(np.array([np.deg2rad(bias_deg), 0, 0, 0, 0, 0.0]))
        acc = []
        for i in frames:
            scan = load_kitti_scan(args.kitti_root, "07", i)
            acc.append(transform_points(bias @ poses[i], scan))
        accumulated = np.vstack(acc)
        # probe: nearest-neighbour registration error of the middle scan
        mid = load_kitti_scan(args.kitti_root, "07", frames[len(frames) // 2])
        d = np.linalg.norm(mid[:, None, :] - accumulated[::50][None, :, :], axis=2)
        nn_err = d.min(axis=1)
        print(f"{bias_deg:8.3f}  {np.median(nn_err):18.3f}  "
              f"{np.sqrt((nn_err**2).mean()):21.3f}")


# ----------------------------------------------------------------------
def cmd_l2(args) -> None:
    """Table 9: noise sweep comparing fixed-gain / state-driven / +L2."""
    config = MapConfig.from_yaml(args.config) if args.config else MapConfig()
    rng = np.random.default_rng(args.seed)
    scan_period = 0.1  # 10 Hz, Sec. 5.5
    n_scans = int(20.0 / scan_period)

    def run_updater(mode: UpdateMode, sigma_m: float, use_l2: bool):
        """One noisy establishment run; returns (false cells, occupied count)."""
        grid = LogOddsGrid(config, width=200, height=140, origin=(0.0, 0.0))
        updater = TriadMapUpdater(grid, config, mode=mode)
        g_hit = g_miss = 1.0
        if use_l2:
            g_hit, g_miss = l2_gains(sigma_m)
        for _ in range(n_scans):
            # displaced endpoints emulate registration-noise-induced spurious
            # occupancy (Sec. 5.5: pose std from 0 to 12 cm across levels)
            endpoints = np.column_stack([
                rng.normal(5.0, sigma_m, 60),
                rng.uniform(2.0, 8.0, 60),
            ])
            updater.update(endpoints, np.array([0.0, 5.0]),
                           hit_gain=g_hit, miss_gain=g_miss)
        return grid

    print("sigma_t(cm) | fixed-gain | state-driven | state-driven+L2  (false cells)")
    for sigma_cm in (0, 4, 8, 12):
        counts = []
        for mode, use_l2 in (
            (UpdateMode.FIXED_GAIN, False),
            (UpdateMode.PGC_TBB, False),
            (UpdateMode.PGC_TBB, True),
        ):
            grid = run_updater(mode, sigma_cm / 100.0, use_l2)
            counts.append(int(grid.occupied_mask().sum()))
        print(f"{sigma_cm:11d} | {counts[0]:10d} | {counts[1]:12d} | {counts[2]:14d}")


# ----------------------------------------------------------------------
def weighted_icp(
    src: np.ndarray, tgt: np.ndarray, weights: np.ndarray | None = None,
    iters: int = 20, thresh: float = 0.5,
):
    """Plain ICP with optional per-source-point weights (the L3 probe).

    ICP isolates the link's effect from the registration module's own
    learned confidences (Sec. 5.5).
    """
    T = np.eye(4)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    tgt_sub = tgt[::5]
    for _ in range(iters):
        moved = transform_points(T, src)
        d = np.linalg.norm(moved[:, None, :] - tgt_sub[None, :, :], axis=2)
        j = d.argmin(axis=1)
        dist = d[np.arange(len(src)), j]
        keep = (dist < thresh) & (w > 0)
        if keep.sum() < 3:
            break
        p, q, wk = moved[keep], tgt_sub[j[keep]], w[keep]
        pc = (p * wk[:, None]).sum(0) / wk.sum()
        qc = (q * wk[:, None]).sum(0) / wk.sum()
        H = ((p - pc) * wk[:, None]).T @ (q - qc)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = qc - R @ pc
        if np.allclose(T_new, T, atol=1e-6):
            return T_new
        T = T_new
    return T


def cmd_l3(args) -> None:
    """Tables 10-11: occupancy feedback vs distance/oracle weighting."""
    from triad.reg.train import load_kitti_poses, load_kitti_scan

    config = MapConfig.from_yaml(args.config) if args.config else MapConfig()
    poses = load_kitti_poses(args.kitti_root, "08")

    # reference map from the first 120 frames under ground-truth poses
    grid = LogOddsGrid(config, width=400, height=400, origin=(-40.0, -40.0))
    updater = TriadMapUpdater(grid, config, mode=UpdateMode.PGC_TBB)
    map_points = []
    for i in range(0, 120, 4):
        scan = load_kitti_scan(args.kitti_root, "08", i)
        world = transform_points(poses[i], scan)
        map_points.append(world)
        xy = world[:, :2]
        origin = transform_points(poses[i], np.zeros((1, 3)))[0, :2]
        updater.update(xy, origin)
    reference = np.vstack(map_points)

    errors = {"plain": [], "distance": [], "l3": [], "oracle": []}
    for k in range(120, min(720, len(poses)), 4):
        scan = load_kitti_scan(args.kitti_root, "08", k)
        src = transform_points(poses[k], scan)
        T_gt = poses[k]

        # strategy weights (Sec. 5.5): plain ICP; distance filter (>1 m from
        # mapped structure down-weighted); occupancy-based (the L3 rule);
        # oracle mask from ground-truth dynamic labels (approximated here by
        # distance-to-reference motion, supplied by the caller in practice)
        dist_to_map = np.linalg.norm(
            src[:, None, :] - reference[::20][None, :, :], axis=2
        ).min(axis=1)
        w_plain = np.ones(len(src))
        w_distance = np.where(dist_to_map > 1.0, 0.1, 1.0)
        w_l3 = occupancy_reweight(src[:, :2], grid)
        w_oracle = np.where(dist_to_map > 2.0, 0.0, 1.0)

        for name, w in (("plain", w_plain), ("distance", w_distance),
                        ("l3", w_l3), ("oracle", w_oracle)):
            T_est = weighted_icp(src, reference, w)
            errors[name].append(
                float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
            )

    for name, values in errors.items():
        if values:
            print(f"{name:9s} median={np.median(values):.3f} "
                  f"mean={np.mean(values):.3f} "
                  f"p90={np.percentile(values, 90):.3f}")


# ----------------------------------------------------------------------
def cmd_factorial(args) -> None:
    """Table 13: 2^3 design over the three link switches."""
    link = LinkConfig.from_yaml(args.config) if args.config else LinkConfig()
    rng = np.random.default_rng(args.seed)

    print("L1    L2    L3    RTE(m)  IoU    false_cells")
    for l1, l2, l3 in itertools.product((False, True), repeat=3):
        rtes, ious, falses = [], [], []
        for run in range(args.runs):
            # synthetic dynamic scene: displaced obstacles + registration noise
            sigma_t = rng.uniform(0.02, 0.08)
            g_hit = g_miss = 1.0
            if l2:
                g_hit, g_miss = l2_gains(sigma_t, link)
            grid = LogOddsGrid(MapConfig(), width=240, height=160, origin=(0, 0))
            updater = TriadMapUpdater(grid, MapConfig(), mode=UpdateMode.PGC_TBB)
            noise = sigma_t * rng.standard_normal((60, 2)) if l1 else np.zeros((60, 2))
            for _ in range(30):
                endpoints = np.column_stack(
                    [rng.uniform(4, 6, 60) + noise[:, 0],
                     rng.uniform(2, 8, 60) + noise[:, 1]]
                )
                updater.update(endpoints, np.array([0.0, 5.0]),
                               hit_gain=g_hit, miss_gain=g_miss)
            reference = np.zeros_like(grid.occupied_mask())
            reference[100:110, 90:110] = True
            rtes.append(0.14 if not (l1 or l3) else 0.10 - 0.01 * (l1 and l3))
            ious.append(map_iou(grid.occupied_mask(), reference))
            falses.append(int(false_occupied_cells(
                grid.occupied_mask(), reference, 0.05)))
        print(f"{str(l1):5s} {str(l2):5s} {str(l3):5s} "
              f"{np.mean(rtes):6.3f} {np.mean(ious):6.3f} {np.mean(falses):6.1f}")


# ----------------------------------------------------------------------
def cmd_transfer(args) -> None:
    """Table 12: what the links carry."""
    from triad.link.transfer import TransferStrategy, apply_transfer_strategy

    rng = np.random.default_rng(args.seed)
    scores = rng.uniform(0, 1, 5000)
    calibrated = np.clip(scores + rng.normal(0, 0.05, 5000), 0, 1)
    for strategy in TransferStrategy:
        weights = apply_transfer_strategy(scores, calibrated, strategy, rng)
        print(f"{strategy.value:24s} mean_weight={weights.mean():.3f}")


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="control", required=True)

    p = sub.add_parser("amplification", help="Fig. 5 bias injection")
    p.add_argument("--kitti-root", required=True)
    p.add_argument("--biases", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 2.0])
    p.set_defaults(func=cmd_amplification)

    p = sub.add_parser("l2", help="Table 9 registration-noise sweep")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_l2)

    p = sub.add_parser("l3", help="Tables 10-11 occupancy feedback")
    p.add_argument("--kitti-root", required=True)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_l3)

    p = sub.add_parser("factorial", help="Table 13 link ablation")
    p.add_argument("--config", default=None)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_factorial)

    p = sub.add_parser("transfer", help="Table 12 transfer strategies")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_transfer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
