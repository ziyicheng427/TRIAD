"""2D-3D integration for solid-state scans — Sec. 3.4, "Map update and 2D-3D
integration".

"For 3D solid-state scans the bridge reuses the calibration module's voxel
and plane machinery: fitted ground planes are removed and residual obstacle
points are column-projected to the grid."

The bridge therefore instantiates :class:`triad.cal.lidar_edges.VoxelPlaneFitter`
with the calibration module's voxel machinery, classifies fitted planes with
near-vertical normals as ground, removes their inliers, and column-projects
(x, y) of the residual obstacle points to the 2-D grid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..cal.config import CalConfig
from ..cal.lidar_edges import VoxelPlaneFitter
from .config import MapConfig


class SolidStateBridge:
    """Converts a 3-D solid-state scan into 2-D grid endpoints."""

    def __init__(
        self,
        map_config: MapConfig,
        voxel_size: float = 0.5,
        seed: int = 0,
    ):
        self.map_config = map_config
        # reuse the calibration module's voxel + plane machinery (Sec. 3.4):
        # indoor voxel size by default for solid-state range
        self.cal_config = CalConfig(voxel_size=voxel_size)
        self.fitter = VoxelPlaneFitter(self.cal_config, seed=seed)

    # ------------------------------------------------------------------
    def project(
        self, cloud: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-removing column projection.

        Returns ``(endpoints_xy, ground_points)``: the obstacle endpoints in
        the grid plane (N, 2) and the removed ground points (M, 3) for
        diagnostics.
        """
        planes = self.fitter.fit(np.asarray(cloud, dtype=float))
        ground_mask = np.zeros(len(cloud), dtype=bool)
        for plane_list in planes.values():
            plane = plane_list[0]
            if abs(plane.normal[2]) >= self.map_config.ground_normal_z:
                # near-horizontal plane: mark its inliers as ground
                dist = cloud @ plane.normal + plane.d
                ground_mask |= np.abs(dist) <= self.cal_config.ransac_distance_thresh
        obstacle = cloud[~ground_mask]
        ground = cloud[ground_mask]
        # column projection: (x, y) of residual obstacle points
        return obstacle[:, :2], ground
