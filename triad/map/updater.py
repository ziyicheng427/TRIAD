"""TRIAD-Map occupancy-state update rule — Sec. 3.4, Fig. 4.

Occupancy-state adaptation partitions *known* cells by pre-update
probability into three bands:

* ``p < 0.49`` (free): a hit is accepted only if persistence has reached
  ``N = 3`` frames — the **persistence gate** (PGC), discarding endpoints of
  passing objects while admitting genuinely placed obstacles;
* ``0.49 <= p < 0.55`` (transition): a boosted hit with hit-model
  probability 0.85 — the **transition-band boost** (TBB), shortening the
  establishment of new obstacles;
* ``p >= 0.55`` (occupied): base hit updates.

Ray-traversed cells without an endpoint receive miss updates, including
previously unknown cells.  L2's gain modulation (Eq. 3) enters here
multiplicatively through ``hit_gain``/``miss_gain``, separately from these
occupancy-state rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..common.se3 import transform_points
from .config import MapConfig
from .grid import LogOddsGrid
from .persistence import PersistenceTracker
from .raycasting import cast_rays


class UpdateMode(Enum):
    """Updater variants of the matched-parameter ablation (Table 8)."""

    FIXED_GAIN = "fixed_gain"  # the Cartographer-style default
    PGC_ONLY = "pgc_only"      # persistence gate, base hits elsewhere
    TBB_ONLY = "tbb_only"      # transition-band boost, no gate
    PGC_TBB = "pgc_tbb"        # full TRIAD-Map rule


class OccupancyUpdater:
    """Common interface of TRIAD-Map and the reimplemented baselines."""

    def __init__(self, grid: LogOddsGrid, config: Optional[MapConfig] = None):
        self.grid = grid
        self.config = config or grid.config

    def update(
        self,
        endpoints_xy: np.ndarray,
        sensor_origin_xy: np.ndarray,
        pose: Optional[np.ndarray] = None,
        sigma_t: Optional[float] = None,
    ) -> None:
        raise NotImplementedError


class TriadMapUpdater(OccupancyUpdater):
    """PGC + TBB occupancy updates with clamped log-odds (Fig. 4)."""

    def __init__(
        self,
        grid: LogOddsGrid,
        config: Optional[MapConfig] = None,
        mode: UpdateMode = UpdateMode.PGC_TBB,
    ):
        super().__init__(grid, config)
        self.mode = mode
        self.persistence = PersistenceTracker(self.config.neighborhood_radius)

    # ------------------------------------------------------------------
    def update(
        self,
        endpoints_xy: np.ndarray,
        sensor_origin_xy: np.ndarray,
        pose: Optional[np.ndarray] = None,
        sigma_t: Optional[float] = None,
        hit_gain: float = 1.0,
        miss_gain: float = 1.0,
    ) -> None:
        """Integrate one scan.

        ``pose`` (4x4) places the scan in the map frame; ``sigma_t`` is the
        registration pose standard deviation consumed by L2 (Eq. 3) — the
        gains it produces may be passed directly via ``hit_gain``/``miss_gain``
        (computed by :mod:`triad.link.l2_gains`) or derived here from
        ``sigma_t`` for convenience.
        """
        endpoints = np.asarray(endpoints_xy, dtype=float)
        origin = np.asarray(sensor_origin_xy, dtype=float)
        if pose is not None:
            endpoints = transform_points(pose, endpoints)
            origin = transform_points(pose, origin[None, :])[0]

        # L2 gains from Eq. (3) when sigma_t is given and gains not overridden
        if sigma_t is not None and hit_gain == 1.0 and miss_gain == 1.0:
            from ..link.l2_gains import l2_gains  # local import: avoids cycle

            hit_gain, miss_gain = l2_gains(sigma_t)

        miss_cells, endpoint_cells = cast_rays(self.grid, origin, endpoints)

        # persistence probe advances on this scan's endpoint support (Sec. 3.4)
        endpoint_cells_arr = np.asarray(endpoint_cells, dtype=int)
        if len(endpoint_cells_arr) > 0:
            self.persistence.observe(endpoint_cells_arr, self.grid)

        # --- miss updates: every traversed cell without an endpoint ----------
        hit_set = set(map(tuple, endpoint_cells))
        miss_only = np.array(
            [c for c in miss_cells if c not in hit_set], dtype=int
        ).reshape(-1, 2)
        if len(miss_only) > 0:
            increments = np.full(len(miss_only), self.grid.table.miss)
            self.grid.apply_updates(miss_only, increments, hit=False,
                                    gains=miss_gain)

        # --- hit updates: occupancy-state bands on pre-update probability -----
        if len(endpoint_cells_arr) == 0:
            return
        p_pre = self.grid.p_at(endpoint_cells_arr)
        counts = self.persistence.counts_array(endpoint_cells_arr)
        table = self.grid.table

        increments = np.zeros(len(endpoint_cells_arr))
        for i, (p, c) in enumerate(zip(p_pre, counts)):
            increments[i] = self._hit_increment(p, int(c), table)

        admissible = increments != 0.0
        if np.any(admissible):
            self.grid.apply_updates(
                endpoint_cells_arr[admissible],
                increments[admissible],
                hit=True,
                gains=hit_gain,
            )

    # ------------------------------------------------------------------
    def _hit_increment(self, p_pre: float, persistence: int, table) -> float:
        """The band rule of Fig. 4 for one endpoint cell (0 = gated out)."""
        cfg = self.config
        below_free = p_pre < cfg.p_free_threshold
        in_transition = (
            cfg.p_free_threshold <= p_pre < cfg.p_transition_threshold
        )

        if below_free:
            # PGC: free cells accept a hit only after N = 3 supporting scans
            gate_open = (
                persistence >= cfg.persistence_frames
                or self.mode in (UpdateMode.TBB_ONLY, UpdateMode.FIXED_GAIN)
            )
            return table.base_hit if gate_open else 0.0
        if in_transition:
            # TBB: boosted hit with hit-model probability 0.85
            if self.mode in (UpdateMode.TBB_ONLY, UpdateMode.PGC_TBB):
                return table.boosted_hit
            return table.base_hit
        # occupied band: base hit
        return table.base_hit
