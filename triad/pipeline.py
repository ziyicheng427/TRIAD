"""The full TRIAD stack with per-link switches — Sec. 3, Fig. 1.

The pipeline wires the four components so one object drives an online
perception step:

    TRIAD-Cal (offline) ──Σ_ξ──▶ L1 ──▶ observation variance ──▶ TRIAD-Reg
    TRIAD-Reg ──σ_t──▶ L2 ──▶ hit/miss gains ──▶ TRIAD-Map
    TRIAD-Map ──P(occ)──▶ L3 ──▶ correspondence weights ──▶ TRIAD-Reg

The three link switches support the 2³ factorial ablation of Sec. 5.5 and
the three end-to-end configurations of Sec. 5.6: Baseline (default stack
without TRIAD modules), TRIAD w/o Link (modules with all links disabled),
and Full TRIAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cal.config import CalConfig
from .cal.estimator import CalibResult
from .link.config import LinkConfig
from .link.l1_covariance import map_gain_from_variance, propagate_to_observation_variance
from .link.l2_gains import l2_gains
from .link.l3_reweight import occupancy_reweight
from .map.config import MapConfig
from .map.grid import LogOddsGrid
from .map.updater import TriadMapUpdater, UpdateMode


@dataclass
class StackConfig:
    cal: CalConfig = field(default_factory=CalConfig)
    map: MapConfig = field(default_factory=MapConfig)
    link: LinkConfig = field(default_factory=LinkConfig)


class TriadPipeline:
    """One online step of the full stack; the calibration runs once offline."""

    def __init__(
        self,
        config: Optional[StackConfig] = None,
        grid: Optional[LogOddsGrid] = None,
        grid_shape: tuple = (400, 400),
        grid_origin: tuple = (-10.0, -10.0),
        use_task_modules: bool = True,
    ):
        """``use_task_modules=False`` yields the Baseline stack (fixed-gain
        updater) of Sec. 5.6; the link switches then have no effect."""
        self.config = config or StackConfig()
        self.grid = grid or LogOddsGrid(
            self.config.map, width=grid_shape[0], height=grid_shape[1],
            origin=grid_origin,
        )
        self.use_task_modules = use_task_modules
        mode = UpdateMode.PGC_TBB if use_task_modules else UpdateMode.FIXED_GAIN
        self.updater = TriadMapUpdater(self.grid, self.config.map, mode=mode)
        self.calibration: Optional[CalibResult] = None

    # ------------------------------------------------------------------
    # offline stage
    # ------------------------------------------------------------------
    def set_calibration(self, result: CalibResult) -> None:
        """Install TRIAD-Cal's output; L1 propagates its covariance."""
        self.calibration = result

    # ------------------------------------------------------------------
    # online stages
    # ------------------------------------------------------------------
    def observe_variance(self, points_lidar: np.ndarray) -> Optional[np.ndarray]:
        """L1: variance inflation of the registration/map observations.

        Disabled (``enable_l1=False``, or no calibration installed) →
        ``None``, i.e. the observations keep their own noise model.
        """
        if not self.config.link.enable_l1 or self.calibration is None:
            return None
        return propagate_to_observation_variance(
            points_lidar,
            self.calibration.T_lidar_cam,
            self.calibration.covariance,
        )

    def register_and_map(
        self,
        endpoints_xy: np.ndarray,
        sensor_origin_xy: np.ndarray,
        pose: Optional[np.ndarray] = None,
        sigma_t: Optional[float] = None,
        source_points_xy: Optional[np.ndarray] = None,
        correspondence_weights: Optional[np.ndarray] = None,
    ) -> dict:
        """One registration → mapping step with the enabled links.

        Parameters
        ----------
        endpoints_xy, sensor_origin_xy, pose : the scan in sensor frame and
            its pose in the map frame (from TRIAD-Reg or the odometry front
            end feeding it).
        sigma_t : registration pose std from TRIAD-Reg (L2 input).
        source_points_xy, correspondence_weights : registration source
            points and their weights (L3 reweights them against the map).

        Returns a small dict of the step's reliability quantities.
        """
        report: dict = {}

        # L3 (map → registration): occupancy-informed correspondence weights
        if (
            self.config.link.enable_l3
            and self.use_task_modules
            and source_points_xy is not None
        ):
            correspondence_weights = occupancy_reweight(
                source_points_xy, self.grid, correspondence_weights,
                self.config.link,
            )
            report["l3"] = correspondence_weights

        # L1 (cal → reg/map): observation variance and the map gains it implies
        endpoints = np.asarray(endpoints_xy, dtype=float)
        extra_variance = self.observe_variance(endpoints)
        hit_gain = miss_gain = 1.0
        if extra_variance is not None and len(extra_variance) > 0:
            gains = map_gain_from_variance(
                extra_variance, self.config.link.sigma_0
            ).mean()
            hit_gain = miss_gain = float(gains)
            report["l1_extra_variance"] = extra_variance

        # L2 (reg → map): pose-dependent gain damping (Eq. 3)
        if sigma_t is not None and self.config.link.enable_l2:
            g_hit, g_miss = l2_gains(sigma_t, self.config.link)
            hit_gain *= g_hit
            miss_gain *= g_miss
            report["l2_gains"] = (g_hit, g_miss)

        self.updater.update(
            endpoints, sensor_origin_xy, pose=pose,
            hit_gain=hit_gain, miss_gain=miss_gain,
        )
        return report
