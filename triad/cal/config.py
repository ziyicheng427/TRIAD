"""TRIAD-Cal configuration (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config import BaseConfig


@dataclass
class CalConfig(BaseConfig):
    # -- LiDAR edge construction ----------------------------------------
    # voxel size for the plane-intersection edge extraction: 1 m outdoors,
    # 0.5 m indoors (Sec. 3.2)
    voxel_size: float = 1.0
    # RANSAC plane fitting per voxel
    ransac_iterations: int = 200
    ransac_distance_thresh: float = 0.03  # m
    plane_min_inliers: int = 8
    # adjacent planes contribute an intersection line only when their dihedral
    # angle lies in [30°, 150°] (Sec. 3.2)
    dihedral_min_deg: float = 30.0
    dihedral_max_deg: float = 150.0
    # two planes in neighbouring voxels must be near-coincident to intersect
    plane_offset_tolerance: float = 0.10  # m
    # sampling density along each intersection line
    edge_point_spacing: float = 0.05      # m

    # -- image boundary extraction ---------------------------------------
    # a 16x16 point grid prompts SAM (Sec. 3.2)
    sam_grid: int = 16
    # adaptive filter keeps edge pixels whose local RGB intensity standard
    # deviation exceeds 0.7 x the mean over instances (Sec. 3.2)
    rgb_std_ratio: float = 0.7
    # SAM edge localisation error entering the noise model (Sec. 3.2)
    sam_edge_sigma_px: float = 1.2

    # -- LiDAR observation noise -----------------------------------------
    # sigma_i combines the LiDAR angular disturbance and additive range noise
    # (Sec. 3.2); both are sensor specs (Livox Mid-70 class defaults)
    lidar_angular_sigma_deg: float = 0.05  # deg
    lidar_range_sigma: float = 0.02        # m

    # -- association reliability ------------------------------------------
    # multi-frame stability window of five frames (Sec. 3.2)
    multi_frame_window: int = 5
    # decay of the temporal-consistency weight (px)
    temporal_tau_px: float = 5.0
    # direction weight suppresses matches whose projected line direction
    # contradicts the image edge normal (Sec. 3.2); cos^2 damping below
    direction_pow: float = 2.0

    # -- estimator ---------------------------------------------------------
    huber_delta: float = 1.345
    max_iterations: int = 30
    convergence_tol: float = 1e-8
    # initialisation: correspondence-rate-driven grid search (Sec. 3.2);
    # randomised within ±5° / ±10 cm of nominal in the evaluation (Sec. 5.1)
    init_rot_range_deg: float = 5.0
    init_trans_range_m: float = 0.10
    init_grid_rot_step_deg: float = 1.0
    init_grid_trans_step_m: float = 0.04
    # a hit counts as a correspondence when the attraction-field residual is
    # below this threshold (px)
    correspondence_thresh_px: float = 10.0

    # -- quality warning ----------------------------------------------------
    # mask coverage ratio below 0.6 triggers a warning that inflates the
    # reported covariance by 50 percent (Sec. 3.2)
    coverage_warn_threshold: float = 0.6
    coverage_inflation: float = 1.5
