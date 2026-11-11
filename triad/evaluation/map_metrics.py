"""Occupancy-map metrics — Sec. 5.3 and 5.5.

* **clearing delay** ``T_clr``: time from removing an obstacle until its
  cell falls (and stays) below the free threshold (Tables 5, 6, 8);
* **establishment delay** ``T_bld``: time from placing a new obstacle until
  its cell crosses the occupied threshold (Tables 5, 6, 8);
* **false occupied cells**: occupied cells where the reference map is free
  or unknown (Tables 9, 13; per 1000 m² in Table 15);
* **map IoU**: intersection over union of occupied cells against the
  reference map (Tables 13-15).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _threshold_crossing_time(
    probabilities: np.ndarray, threshold: float, direction: int,
    scan_period: float = 0.1,
) -> Optional[float]:
    """First time a cell's probability crosses ``threshold`` and stays for
    three consecutive scans; ``direction=-1`` for clearing, ``+1`` for
    establishment.  Returns seconds or None if never crossed."""
    target = probabilities >= threshold if direction > 0 else probabilities < threshold
    for i in range(len(target) - 2):
        if target[i] and target[i + 1] and target[i + 2]:
            return i * scan_period
    return None


def clearing_delay(
    cell_probability_history: np.ndarray,
    free_threshold: float,
    scan_period: float = 0.1,
) -> float:
    """``T_clr`` for one cell whose obstacle was removed at t = 0."""
    t = _threshold_crossing_time(
        cell_probability_history, free_threshold, direction=-1,
        scan_period=scan_period,
    )
    return float("inf") if t is None else t


def establishment_delay(
    cell_probability_history: np.ndarray,
    occupied_threshold: float,
    scan_period: float = 0.1,
) -> float:
    """``T_bld`` for one cell whose obstacle was placed at t = 0."""
    t = _threshold_crossing_time(
        cell_probability_history, occupied_threshold, direction=+1,
        scan_period=scan_period,
    )
    return float("inf") if t is None else t


def false_occupied_cells(
    predicted_occupied: np.ndarray,
    reference_occupied: np.ndarray,
    area_per_cell_m2: float,
) -> float:
    """False occupied cells normalised per 1000 m² (Table 15)."""
    false_cells = int(np.sum(predicted_occupied & ~reference_occupied))
    total_area = predicted_occupied.size * area_per_cell_m2
    return false_cells / max(total_area, 1e-9) * 1000.0


def map_iou(predicted_occupied: np.ndarray, reference_occupied: np.ndarray) -> float:
    """Intersection over union of the occupied sets (Tables 13-15)."""
    intersection = np.sum(predicted_occupied & reference_occupied)
    union = np.sum(predicted_occupied | reference_occupied)
    return float(intersection / union) if union > 0 else 1.0
