"""L3: occupancy-informed correspondence reweighting — Sec. 3.5.

"The occupancy probability of the cell under a source point becomes a
dynamic flag.  Points in confidently free cells (P < 0.2) carry
correspondence weight 0.1.  Points in occupied and unknown cells retain
their weights.  This rule suppresses points inconsistent with observed free
space while preserving observations in unmapped regions."

The reweighted weights feed TRIAD-Reg's matching (as extra weights of the
C2C estimator) or, as in the L3 probe of Sec. 5.5, a weighted ICP.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import LinkConfig


def occupancy_reweight(
    points_xy: np.ndarray,
    grid,
    weights: Optional[np.ndarray] = None,
    config: Optional[LinkConfig] = None,
) -> np.ndarray:
    """Apply the L3 rule to source-point correspondence weights.

    Parameters
    ----------
    points_xy : (N, 2) source points in map coordinates.
    grid : a :class:`triad.map.grid.LogOddsGrid` (the map's occupancy belief).
    weights : optional existing weights; free-cell points are down-weighted
        to ``0.1 ×`` their own weight when supplied, else set to 0.1.
    config : link config (free threshold and weight).

    Returns
    -------
    (N,) reweighted correspondence weights.
    """
    config = config or LinkConfig()
    points = np.asarray(points_xy, dtype=float)
    cells = grid.world_to_cell(points)
    in_bounds = grid.in_bounds(cells)

    if weights is None:
        out = np.ones(len(points))
    else:
        out = np.asarray(weights, dtype=float).copy()

    if np.any(in_bounds):
        p = grid.p_at(cells[in_bounds])
        confidently_free = p < config.l3_free_threshold
        free_indices = np.nonzero(in_bounds)[0][confidently_free]
        # free-cell points carry correspondence weight 0.1; occupied and
        # unknown cells retain their weights (Sec. 3.5)
        out[free_indices] = config.l3_free_weight
    return out
