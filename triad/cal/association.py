"""Cross-modal association and reliability weighting — Sec. 3.2.

"Directional compatibility and multi-frame stability distinguish persistent
structural associations from ambiguous matches."

For each LiDAR edge point projected through the current extrinsic estimate:

* the residual is the attraction-field value ``r_i = G(p_i)``;
* a **direction weight** suppresses matches whose projected line direction
  contradicts the image edge normal — resolving fence-and-background
  ambiguities.  A LiDAR edge lying on an image boundary runs *along* it, so
  its direction is perpendicular to the boundary normal; the weight therefore
  penalises large ``|cos|`` between line direction and normal;
* a **multi-frame weight** adds temporal stability from position and
  projection consistency over a five-frame window.

The two weights are the module's C2C "extra weights": they multiply the
confidence weights of Eq. (1) inside the noise-aware estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..common.se3 import transform_points
from .config import CalConfig
from .image_edges import AttractionField
from .lidar_edges import LidarEdge


@dataclass
class ProjectedEdgePoint:
    """One LiDAR edge point with its projection and per-point quantities."""

    point_lidar: np.ndarray   # (3,)
    point_cam: np.ndarray     # (3,) transformed into the camera frame
    uv: np.ndarray            # (2,) pixel coordinates
    residual: float           # G(u, v) in pixels
    direction_cam: np.ndarray  # (3,) unit edge-line direction in camera frame
    edge_index: int


@dataclass
class AssociationResult:
    """Weighted cross-modal association for one frame."""

    points: List[ProjectedEdgePoint]
    residuals: np.ndarray      # (N,) attraction-field values
    direction_weights: np.ndarray  # (N,)
    temporal_weights: np.ndarray   # (N,)
    consistency_scores: np.ndarray  # (N,) raw C2C probe scores in [0, 1]


def project_points(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of (N, 3) camera-frame points to (N, 2) pixels."""
    z = np.maximum(pts_cam[:, 2], 1e-6)
    uv_h = (pts_cam / z[:, None]) @ K.T
    return uv_h[:, :2]


def projection_jacobian(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """d(u, v) / d(p_cam) — (N, 2, 3) pinhole projection Jacobian."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    z2 = np.maximum(z**2, 1e-12)
    J = np.empty((len(pts_cam), 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 1] = 0.0
    J[:, 0, 2] = -fx * x / z2
    J[:, 1, 0] = 0.0
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z2
    del cx, cy  # principal point drops out of the Jacobian
    return J


class CrossModalAssociation:
    """Projects LiDAR edge points, evaluates residuals and reliability weights."""

    def __init__(self, config: CalConfig, K: np.ndarray):
        self.config = config
        self.K = np.asarray(K, dtype=float)

    # ------------------------------------------------------------------
    def associate(
        self,
        edges: List[LidarEdge],
        field: AttractionField,
        T_lidar_cam: np.ndarray,
        extra_temporal: Optional[np.ndarray] = None,
    ) -> AssociationResult:
        """Project all edge points and compute the reliability weights.

        ``T_lidar_cam`` maps LiDAR-frame points into the camera frame.
        ``extra_temporal`` optionally overrides the multi-frame weights (the
        pipeline passes the ones computed over the five-frame window).
        """
        points: List[ProjectedEdgePoint] = []
        for edge_index, edge in enumerate(edges):
            pts_cam = transform_points(T_lidar_cam, edge.points)
            uv = project_points(self.K, pts_cam)
            in_view = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < field.field.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < field.field.shape[0])
                & (pts_cam[:, 2] > 0)
            )
            direction_cam = (T_lidar_cam[:3, :3] @ edge.direction) / max(
                np.linalg.norm(T_lidar_cam[:3, :3] @ edge.direction), 1e-12
            )
            for i in np.nonzero(in_view)[0]:
                u, v = uv[i]
                residual = float(field.value_at(np.array([u]), np.array([v]))[0])
                points.append(
                    ProjectedEdgePoint(
                        point_lidar=edge.points[i],
                        point_cam=pts_cam[i],
                        uv=uv[i],
                        residual=residual,
                        direction_cam=direction_cam,
                        edge_index=edge_index,
                    )
                )

        residuals = np.array([p.residual for p in points]) if points else np.zeros(0)

        # -- direction weight ------------------------------------------------
        direction_weights = self._direction_weights(points, field)

        # -- multi-frame temporal weight --------------------------------------
        if extra_temporal is not None:
            temporal_weights = np.asarray(extra_temporal, dtype=float)
        else:
            temporal_weights = np.ones_like(residuals)

        # -- raw C2C consistency score ----------------------------------------
        # closeness to the boundary, damped by the two reliability weights;
        # the confidence map Phi (isotonic fit, Sec. 5.4) turns this into the
        # calibrated weight of Eq. (1)
        closeness = np.exp(-residuals / max(self.config.correspondence_thresh_px, 1e-6))
        consistency = closeness * direction_weights * temporal_weights
        return AssociationResult(
            points=points,
            residuals=residuals,
            direction_weights=direction_weights,
            temporal_weights=temporal_weights,
            consistency_scores=np.clip(consistency, 0.0, 1.0),
        )

    # ------------------------------------------------------------------
    def _direction_weights(
        self, points: List[ProjectedEdgePoint], field: AttractionField
    ) -> np.ndarray:
        """Penalise projected line directions parallel to the edge normal.

        ``w = 1 - |cos|^p`` between the camera-frame line direction and the
        image-edge normal ``n = ∇G / |∇G|``; the paper resolves
        fence-and-background ambiguities with exactly this compatibility.
        """
        if not points:
            return np.zeros(0)
        uv = np.stack([p.uv for p in points])
        dirs = np.stack([p.direction_cam for p in points])
        normals = field.normals_at(uv[:, 0], uv[:, 1])  # (N, 2) in image plane

        # project the 3D line direction onto the image plane (drop the z axis
        # of the camera frame); only its image-plane component can oppose n
        dirs_uv = dirs[:, :2]
        dirs_norm = np.maximum(np.linalg.norm(dirs_uv, axis=1), 1e-12)
        cos = np.abs(np.sum(dirs_uv / dirs_norm[:, None] * normals, axis=1))
        return np.clip(1.0 - cos**self.config.direction_pow, 0.0, 1.0)


def temporal_weights_over_window(
    residual_history: List[np.ndarray], config: CalConfig
) -> np.ndarray:
    """Multi-frame stability weight over the five-frame window (Sec. 3.2).

    ``residual_history`` holds the per-point residuals of the ``W`` most
    recent frames (aligned by point index).  A point is temporally stable when
    its projection stayed close to the image boundary across the window:
    ``w = exp(-std(r) / tau)`` with the window's std; a point never observed
    in earlier frames starts from the neutral weight of an empty history.
    """
    window = min(len(residual_history), config.multi_frame_window)
    if window == 0:
        raise ValueError("residual_history must contain at least one frame")
    stacked = np.stack(residual_history[-window:], axis=0)  # (W, N)
    std = stacked.std(axis=0)
    return np.exp(-std / max(config.temporal_tau_px, 1e-6))
