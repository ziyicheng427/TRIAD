"""Depth-continuous LiDAR edges — Sec. 3.2, "Edge extraction".

Depth-discontinuity edges suffer boundary-localisation artifacts from the
finite beam footprint, so TRIAD-Cal extracts edges directly on the point
cloud instead:

1. the cloud is partitioned into voxels (1 m outdoors / 0.5 m indoors);
2. each voxel is fitted with RANSAC planes;
3. adjacent planes whose dihedral angle lies in [30°, 150°] contribute their
   intersection line.

The fitted planes are exposed separately because the 2D-3D bridge of
TRIAD-Map "reuses the calibration module's voxel and plane machinery"
(Sec. 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import CalConfig


@dataclass
class Plane:
    """A fitted plane ``n . x + d = 0`` with unit normal ``n``."""

    normal: np.ndarray  # (3,)
    d: float
    inlier_count: int
    voxel_index: Tuple[int, int, int]

    def distance_to(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.d


@dataclass
class LidarEdge:
    """A depth-continuous edge: sampled points plus the line direction."""

    points: np.ndarray   # (M, 3) in LiDAR frame
    direction: np.ndarray  # (3,) unit line direction in LiDAR frame
    voxel_pair: Tuple[Tuple[int, int, int], Tuple[int, int, int]]


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int,
    distance_thresh: float,
    rng: np.random.Generator,
) -> Plane | None:
    """Minimal RANSAC plane fit over one voxel's points (inlier count only)."""
    n = len(points)
    if n < 3:
        return None
    best_count, best_plane = -1, None
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:  # degenerate sample
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        count = int(np.sum(np.abs(points @ normal + d) < distance_thresh))
        if count > best_count:
            best_count, best_plane = count, (normal, d)
    if best_plane is None or best_count < 3:
        return None
    normal, d = best_plane
    return Plane(normal=normal, d=d, inlier_count=best_count, voxel_index=(0, 0, 0))


def _dihedral_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Dihedral angle between plane normals, folded to [0°, 180°]."""
    cos = abs(float(np.dot(n1, n2)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _intersect_planes(p1: Plane, p2: Plane) -> Tuple[np.ndarray, np.ndarray] | None:
    """Intersection line of two near-coincident planes: (point, direction)."""
    direction = np.cross(p1.normal, p2.normal)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:  # parallel planes
        return None
    direction = direction / norm
    # solve the 3x3 system [n1; n2; direction] x = [-d1, -d2, 0]
    A = np.vstack([p1.normal, p2.normal, direction])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    point = np.linalg.solve(A, np.array([-p1.d, -p2.d, 0.0]))
    return point, direction


def _clip_segment_to_voxels(
    point: np.ndarray,
    direction: np.ndarray,
    voxels: Tuple[np.ndarray, np.ndarray],
    spacing: float,
) -> np.ndarray:
    """Sample points along the line inside the union of the two voxel boxes."""
    lo, hi = voxels
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    # parameter range where the line is inside the axis-aligned box
    t_lo, t_hi = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not (lo[axis] - 1e-9 <= point[axis] <= hi[axis] + 1e-9):
                return np.empty((0, 3))
            continue
        t1 = (lo[axis] - point[axis]) / direction[axis]
        t2 = (hi[axis] - point[axis]) / direction[axis]
        t_lo = max(t_lo, min(t1, t2))
        t_hi = min(t_hi, max(t1, t2))
    if t_hi <= t_lo:
        return np.empty((0, 3))
    length = t_hi - t_lo
    n_pts = max(int(length / spacing) + 1, 2)
    ts = np.linspace(t_lo, t_hi, n_pts)
    return point[None, :] + ts[:, None] * direction[None, :]


class VoxelPlaneFitter:
    """Voxel partition + per-voxel RANSAC planes (shared with TRIAD-Map).

    The 2D-3D bridge of Sec. 3.4 reuses this machinery for ground-plane
    removal on solid-state scans.
    """

    def __init__(self, config: CalConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def fit(self, cloud: np.ndarray) -> Dict[Tuple[int, int, int], List[Plane]]:
        """Return the per-voxel plane lists for a point cloud."""
        cloud = np.asarray(cloud, dtype=float)
        s = self.config.voxel_size
        indices = np.floor(cloud / s).astype(int)
        planes: Dict[Tuple[int, int, int], List[Plane]] = {}
        for voxel_index in np.unique(indices, axis=0):
            key = tuple(int(v) for v in voxel_index)
            pts = cloud[np.all(indices == voxel_index, axis=1)]
            if len(pts) < self.config.plane_min_inliers:
                continue
            plane = fit_plane_ransac(
                pts,
                iterations=self.config.ransac_iterations,
                distance_thresh=self.config.ransac_distance_thresh,
                rng=self.rng,
            )
            if plane is not None and plane.inlier_count >= self.config.plane_min_inliers:
                plane.voxel_index = key
                planes[key] = [plane]
        return planes

    def voxel_bounds(self, voxel_index: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        s = self.config.voxel_size
        lo = np.asarray(voxel_index, dtype=float) * s
        return lo, lo + s


def extract_lidar_edges(cloud: np.ndarray, config: CalConfig, seed: int = 0) -> List[LidarEdge]:
    """Depth-continuous LiDAR edges from plane intersections (Sec. 3.2).

    Adjacent planes (voxels within the 26-neighbourhood) whose dihedral angle
    lies in ``[dihedral_min_deg, dihedral_max_deg]`` and whose offsets agree
    within ``plane_offset_tolerance`` contribute their intersection line,
    sampled inside the union of the two voxels.
    """
    fitter = VoxelPlaneFitter(config, seed=seed)
    planes = fitter.fit(cloud)

    edges: List[LidarEdge] = []
    seen_pairs = set()
    for key, plane_list in planes.items():
        for neighbor in _neighborhood26(key):
            if neighbor not in planes or (neighbor, key) in seen_pairs:
                continue
            seen_pairs.add((key, neighbor))
            p1, p2 = plane_list[0], planes[neighbor][0]
            dihedral = _dihedral_deg(p1.normal, p2.normal)
            if not (config.dihedral_min_deg <= dihedral <= config.dihedral_max_deg):
                continue
            # planes must be near-coincident in offset to describe one structure
            if abs(p1.d - p2.d) > config.plane_offset_tolerance:
                continue
            intersection = _intersect_planes(p1, p2)
            if intersection is None:
                continue
            point, direction = intersection
            lo1, hi1 = fitter.voxel_bounds(key)
            lo2, hi2 = fitter.voxel_bounds(neighbor)
            pts = _clip_segment_to_voxels(
                point, direction, (lo1, hi1), config.edge_point_spacing
            )
            pts = np.vstack(
                [
                    pts,
                    _clip_segment_to_voxels(
                        point, direction, (lo2, hi2), config.edge_point_spacing
                    ),
                ]
            )
            if len(pts) == 0:
                continue
            edges.append(
                LidarEdge(points=pts, direction=direction, voxel_pair=(key, neighbor))
            )
    return edges


def _neighborhood26(key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    return [
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
