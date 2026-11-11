# TRIAD

**Consistency-Guided LiDAR-Camera Calibration, Point Cloud Registration, and
Dynamic Occupancy Mapping with Cross-Layer Confidence Propagation**

Reference implementation of the TRIAD framework. TRIAD is organised as four
complementary components at two technical levels:

- **TRIAD-Cal** — targetless LiDAR-camera calibration from depth-continuous
  LiDAR edges and SAM-derived image boundaries, returning the extrinsic
  transformation *and its covariance*.
- **TRIAD-Reg** — hierarchical large-scale point cloud registration with
  distance-weighted sampling (DW-FPS), geometric consistency-guided attention
  (CGA), and iterative consistency propagation.
- **TRIAD-Map** — dynamic occupancy updates with persistence gating (PGC) and
  transition-band boosting (TBB) on top of standard log-odds mapping.
- **TRIAD-Link** — three explicit cross-layer interfaces (L1/L2/L3) that
  propagate calibration covariance, registration uncertainty, and occupancy
  belief between the task modules.

The three task modules instantiate a shared **consistency-to-confidence (C2C)**
principle: a consistency probe measures agreement between an observation and
its governing context, a calibrated confidence map converts probe outputs into
weights, and the module's estimator adapts accordingly (Eq. 1 of the paper).

## Repository layout ↔ paper modules

| Paper component | Section | Code |
| --- | --- | --- |
| C2C operator (Eq. 1) | §3.1 | `triad/common/c2c.py` |
| Isotonic confidence calibration, ECE | §5.4 | `triad/common/confidence_calibration.py` |
| SE(3) Lie-group utilities | §3.2 | `triad/common/se3.py` |
| Depth-continuous LiDAR edges (voxels, RANSAC planes, dihedral-filtered intersections) | §3.2 | `triad/cal/lidar_edges.py` |
| SAM prompting, NMS/Sobel thinning, adaptive RGB filter, attraction field | §3.2 | `triad/cal/image_edges.py`, `triad/cal/sam_adapter.py` |
| Directional + multi-frame reliability weighting | §3.2 | `triad/cal/association.py` |
| Noise-aware Huber estimation, Σ_ξ covariance, grid-search init | §3.2 | `triad/cal/estimator.py` |
| Full calibration pipeline | §3.2, Fig. 2 | `triad/cal/pipeline.py` |
| Distance-weighted FPS | §3.3 | `triad/reg/sampling.py` |
| Shared hierarchical encoder | §3.3 | `triad/reg/encoder.py` |
| Dual-softmax matching + seed selection | §3.3 | `triad/reg/matching.py` |
| Geometric compatibility matrix + CGA | §3.3 | `triad/reg/cga.py` |
| Consistency propagation (Eq. 2) | §3.3 | `triad/reg/propagation.py` |
| Weighted Kabsch, coarse-to-fine refinement | §3.3 | `triad/reg/pose.py` |
| InfoNCE + wBCE + L1 transform losses | §3.3 | `triad/reg/losses.py` |
| End-to-end registration network + training (AdamW, 50 epochs, KITTI) | §3.3, §5.2 | `triad/reg/model.py`, `triad/reg/train.py` |
| Log-odds grid + clamped LUT updates | §3.4 | `triad/map/grid.py` |
| Persistence probe (radius-R, consecutive scans, reset) | §3.4 | `triad/map/persistence.py` |
| PGC + TBB occupancy-state update rule, raycasting | §3.4, Fig. 4 | `triad/map/raycasting.py`, `triad/map/updater.py` |
| 2D-3D bridge (ground-plane removal, column projection) | §3.4 | `triad/map/bridge_3d.py` |
| Reimplemented dynamic-mapping baselines | §5.3, Table 6 | `triad/map/baselines.py` |
| L1: Σ_ξ → observation variance | §3.5 | `triad/link/l1_covariance.py` |
| L2: σ_t → hit/miss gain damping (Eq. 3) | §3.5 | `triad/link/l2_gains.py` |
| L3: occupancy → correspondence weights | §3.5 | `triad/link/l3_reweight.py` |
| Confidence-transfer strategies (uniform/shuffled/…) | §5.5, Table 12 | `triad/link/transfer.py` |
| Full stack with link switches (factorial ablation) | §5.5 | `triad/pipeline.py` |
| RTE / RRE / RR, map delays, IoU, error budget | §5 | `evaluation/` |
| Experiment CLIs | §5 | `scripts/` |

A finer-grained mapping (every equation, figure, and table) is in
[`docs/paper_map.md`](docs/paper_map.md).

## Architecture

```
                 ┌──────────────────────────── TRIAD-Link ────────────────────────────┐
                 │  L1: Σ_ξ → observation variance        (cal  → reg, map)           │
                 │  L2: σ_t → hit/miss gain damping       (reg  → map)                │
                 │  L3: occupancy → correspondence weight (map  → reg)                │
                 └───────▲──────────────────────────▲─────────────────────▲──────────┘
                         │ Σ_ξ                      │ σ_t, gains          │ P(occ)
      ┌──────────────────┴────────┐   ┌─────────────┴───────────┐   ┌─────┴────────────┐
      │ TRIAD-Cal                 │   │ TRIAD-Reg               │   │ TRIAD-Map        │
      │ plane-intersection edges  │   │ DW-FPS sampling         │   │ persistence gate │
      │ SAM boundary attraction   │   │ CGA attention           │◄──┤ transition boost │
      │ Huber + noise-aware ξ,Σ_ξ │   │ consistency propagation │   │ log-odds LUTs    │
      └───────────────────────────┘   └─────────────────────────┘   └──────────────────┘
                     every task module instantiates the C2C operator (Eq. 1)
```

## Installation

```bash
pip install -e .            # core: numpy, scipy, pyyaml
pip install -e ".[reg]"     # + torch, for TRIAD-Reg
pip install -e ".[sam]"     # + segment-anything, for SAM-derived boundaries
```

## Usage

```bash
# TRIAD-Cal: targetless calibration with covariance output
python scripts/calibrate.py --cloud scans.ply --frames frames/ --intrinsics K.yaml

# TRIAD-Reg: KITTI-style pair registration evaluation (RTE/RRE/RR)
python scripts/evaluate_registration.py --kitti-root /path/to/kitti/odometry --sequences 08 09 10

# TRIAD-Map: occupancy updater with PGC+TBB (+L2 damping)
python scripts/run_map_updater.py --config configs/map.yaml --scan-dir scans/

# Isotonic confidence calibration of the three C2C maps (≥5000 labelled probes)
python scripts/calibrate_confidence.py --module cal --scores cal_scores.npy --labels cal_labels.npy

# Cross-layer controls: amplification, L1, L2, L3, 2^3 factorial, transfer strategies
python scripts/link_controls.py l2 --config configs/link.yaml

# End-to-end stack (Baseline / TRIAD w/o Link / Full TRIAD)
python scripts/run_e2e.py --scene corridor --condition dynamic
```

All constants named in the paper (voxel sizes, dihedral window, N=3, bands
0.49/0.55, p_hit^TBB=0.85, σ₀=3 cm, L3 thresholds 0.2/0.1, …) are defaults in
`triad/{cal,reg,map,link}/config.py` and overridable via the YAML files in
`configs/`.

## Tests

```bash
pytest tests/
```

Unit tests cover the SE(3) algebra round-trips, the C2C weighting identity,
the map update bands (PGC/TBB), the L2 clip bounds, the L3 rule, and DW-FPS
behaviour. They do not require datasets or GPU.

## Citation

```bibtex
@article{cheng2026triad,
  title   = {TRIAD: Consistency-Guided LiDAR-Camera Calibration, Point Cloud
             Registration, and Dynamic Occupancy Mapping with Cross-Layer
             Confidence Propagation},
  author  = {Cheng, Ziyi and Zhang, Yue},
  journal = {Robotics and Autonomous Systems},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
