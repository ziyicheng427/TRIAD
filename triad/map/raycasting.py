"""Ray traversal — Sec. 3.4.

"Ray-traversed cells without an endpoint receive miss updates, including
previously unknown cells."

Integer DDA (Amanatides–Woo) walks each beam from the sensor origin to its
endpoint; the endpoint cell is excluded from the miss set (it receives the
hit, subject to the occupancy-state rule).
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


def cast_rays(
    grid,
    sensor_origin_xy: np.ndarray,
    endpoints_xy: np.ndarray,
    max_range: float = 30.0,
) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int]]]:
    """Cast all beams of one scan.

    Returns
    -------
    miss_cells :
        set of traversed cells eligible for miss updates.
    endpoint_cells :
        list of endpoint cells (ordered like the endpoints).
    """
    origin = np.asarray(sensor_origin_xy, dtype=float)
    endpoints = np.asarray(endpoints_xy, dtype=float)

    miss_cells: Set[Tuple[int, int]] = set()
    endpoint_cells: List[Tuple[int, int]] = []

    o_cell = np.floor((origin - grid.origin) / grid.config.resolution).astype(int)
    for endpoint in endpoints:
        d = endpoint - origin
        dist = float(np.linalg.norm(d))
        if dist > max_range:
            # clip the beam at max range; the clipped endpoint still yields
            # misses but no hit
            endpoint = origin + d / max(dist, 1e-9) * max_range
            is_hit = False
        else:
            is_hit = True
        e_cell = np.floor((endpoint - grid.origin) / grid.config.resolution).astype(int)

        if is_hit:
            endpoint_cells.append((int(e_cell[0]), int(e_cell[1])))
        miss_cells.update(_walk(o_cell, e_cell, grid))

    return miss_cells, endpoint_cells


def _walk(start: np.ndarray, end: np.ndarray, grid) -> List[Tuple[int, int]]:
    """Traversed cells from ``start`` to ``end`` (end exclusive), Bresenham.

    Integer cell endpoints make the beam a discrete line; Bresenham's
    error-accumulation walk visits exactly the crossed cells without
    floating-point tie issues.
    """
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    step_x = 1 if x1 >= x0 else -1
    step_y = 1 if y1 >= y0 else -1

    cells: List[Tuple[int, int]] = []
    col, row = x0, y0
    err = dx - dy
    while (col, row) != (x1, y1):
        if 0 <= col < grid.width and 0 <= row < grid.height:
            cells.append((col, row))
        elif cells:
            break  # left the grid after having entered it; stop walking
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            col += step_x
        if err2 < dx:
            err += dx
            row += step_y
    return cells
