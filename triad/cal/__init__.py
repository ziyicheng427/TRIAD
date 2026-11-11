"""TRIAD-Cal — consistency-guided targetless LiDAR-camera calibration (Sec. 3.2).

Pipeline (Fig. 2):

1. depth-continuous LiDAR edges from voxel-wise RANSAC plane intersections
   (:mod:`.lidar_edges`);
2. SAM-derived image boundaries thinned and filtered into an attraction field
   (:mod:`.sam_adapter`, :mod:`.image_edges`);
3. directional + multi-frame reliability weighting of the cross-modal
   association (:mod:`.association`);
4. noise-aware Huber estimation of the extrinsic ``xi ∈ SE(3)`` and its
   covariance ``Sigma_xi`` (:mod:`.estimator`), orchestrated by
   :class:`.pipeline.TriadCalPipeline`.
"""

from .config import CalConfig
from .lidar_edges import extract_lidar_edges, VoxelPlaneFitter, LidarEdge
from .image_edges import build_attraction_field, filter_boundary_pixels
from .association import CrossModalAssociation
from .estimator import CalibResult, ExtrinsicEstimator, grid_search_init
from .pipeline import TriadCalPipeline

__all__ = [
    "CalConfig",
    "extract_lidar_edges",
    "VoxelPlaneFitter",
    "LidarEdge",
    "build_attraction_field",
    "filter_boundary_pixels",
    "CrossModalAssociation",
    "CalibResult",
    "ExtrinsicEstimator",
    "grid_search_init",
    "TriadCalPipeline",
]
