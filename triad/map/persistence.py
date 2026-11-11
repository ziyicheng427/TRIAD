"""The persistence probe — Sec. 3.4, "Temporal persistence measurement".

"The persistence probe counts consecutive scans with at least one endpoint
inside the cell's radius-R neighborhood and resets on any scan without one."

This is TRIAD-Map's instantiation of the C2C consistency probe (Sec. 3.1):
the probe's raw score is the consecutive-support count, calibrated by the
module's isotonic confidence map into a probability of reliable cell update
(Sec. 5.4) when labels are available.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class PersistenceTracker:
    """Per-cell consecutive-support counters over scans."""

    def __init__(self, radius: float = 0.15):
        self.radius = radius
        self._counts: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    def support_cells(self, endpoint_cells: np.ndarray, grid) -> set:
        """All cells within radius R of any endpoint cell (the support set).

        A scan supports cell ``c`` iff at least one endpoint lands in the
        radius-R neighbourhood of ``c`` (Sec. 3.4).
        """
        support: set = set()
        r_cells = int(np.ceil(self.radius / grid.config.resolution))
        for col, row in np.asarray(endpoint_cells):
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        support.add((int(col + dx), int(row + dy)))
        return support

    # ------------------------------------------------------------------
    def observe(self, endpoint_cells: np.ndarray, grid) -> None:
        """Advance the counters by one scan: increment supported cells,
        reset every unsupported cell to zero (Sec. 3.4)."""
        support = self.support_cells(endpoint_cells, grid)
        for cell in support:
            self._counts[cell] = self._counts.get(cell, 0) + 1
        # reset any tracked cell the current scan does not support
        for cell in list(self._counts):
            if cell not in support:
                self._counts[cell] = 0

    # ------------------------------------------------------------------
    def count(self, cell: Tuple[int, int]) -> int:
        return self._counts.get(cell, 0)

    def counts_array(self, cells: np.ndarray) -> np.ndarray:
        return np.array([self._counts.get((int(c), int(r)), 0) for c, r in cells])
