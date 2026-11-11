"""Clamped log-odds grid and precomputed update tables — Sec. 3.4.

"Cells keep log-odds occupancy with clamped updates. ... Updates are
precomputed as lookup tables inside the point cloud insertion stage of a
standard SLAM front end."

The :class:`UpdateTable` precomputes the log-odds increments of every update
kind the updater emits (base hit, boosted hit, miss) so the per-scan update
is a table lookup plus a multiplicative gain (L2) — no trigonometry or
logits on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import MapConfig


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def sigmoid(x) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class UpdateTable:
    """Precomputed log-odds increments for every update kind (Sec. 3.4)."""

    base_hit: float      # base hit increment: logit(p_hit)
    boosted_hit: float   # TBB increment: logit(p_hit_transition = 0.85)
    miss: float          # miss increment: logit(p_miss)

    @classmethod
    def from_config(cls, config: MapConfig) -> "UpdateTable":
        return cls(
            base_hit=logit(config.p_hit),
            boosted_hit=logit(config.p_hit_transition),
            miss=logit(config.p_miss),
        )


class LogOddsGrid:
    """A 2-D occupancy grid storing clamped log-odds.

    ``known`` marks cells that have received at least one update; untouched
    cells stay at log-odds 0 (p = 0.5, unknown) and receive miss updates as
    soon as a ray crosses them (Sec. 3.4).
    """

    def __init__(
        self,
        config: MapConfig,
        width: int,
        height: int,
        origin: Tuple[float, float] = (0.0, 0.0),
    ):
        self.config = config
        self.width = width
        self.height = height
        self.origin = np.asarray(origin, dtype=float)
        self.log_odds = np.zeros((height, width), dtype=float)
        self.known = np.zeros((height, width), dtype=bool)
        self.lo_min = logit(config.p_min)
        self.lo_max = logit(config.p_max)
        self.table = UpdateTable.from_config(config)

    # -- coordinate helpers ------------------------------------------------
    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) world metres → (N, 2) integer cell indices (col, row)."""
        xy = np.asarray(xy, dtype=float)
        cells = np.floor((xy - self.origin) / self.config.resolution).astype(int)
        return cells

    def cell_center(self, cell: np.ndarray) -> np.ndarray:
        return (np.asarray(cell, dtype=float) + 0.5) * self.config.resolution + self.origin

    def in_bounds(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells)
        return (
            (cells[:, 0] >= 0) & (cells[:, 0] < self.width)
            & (cells[:, 1] >= 0) & (cells[:, 1] < self.height)
        )

    # -- beliefs ---------------------------------------------------------------
    @property
    def probability(self) -> np.ndarray:
        return sigmoid(self.log_odds)

    def p_at(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells)
        return sigmoid(self.log_odds[cells[:, 1], cells[:, 0]])

    def occupied_mask(self) -> np.ndarray:
        return self.probability >= self.config.p_occupied_threshold

    def free_mask(self) -> np.ndarray:
        return self.probability < self.config.p_free_l3_threshold

    # -- updates ------------------------------------------------------------------
    def apply_updates(
        self,
        cells: np.ndarray,
        increments: np.ndarray,
        hit: bool,
        gains: np.ndarray | float = 1.0,
    ) -> None:
        """Clamped log-odds update for a batch of cells.

        ``increments`` come from the update table; ``gains`` are the
        multiplicative L2 hit/miss gains (1.0 when the link is disabled).
        """
        cells = np.asarray(cells)
        inc = np.asarray(increments) * gains
        rows, cols = cells[:, 1], cells[:, 0]
        self.log_odds[rows, cols] = np.clip(
            self.log_odds[rows, cols] + inc, self.lo_min, self.lo_max
        )
        self.known[rows, cols] = True
        del hit  # increments already encode hit/miss; kept for API clarity
