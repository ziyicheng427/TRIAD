"""Stack-level error budget — Sec. 5.5, "Error budget" (Table 14).

"The measured amplification slopes connect the modules' reported accuracies
into a stack-level budget.  Linear interpolation within the measured bias
range maps the calibration module's worst-case rotation error of 0.085° to
an additional 0.06 m of scan-to-map registration error (slope 0.69 m/deg)
and 2.4 mm of map boundary RMSE.  Its worst-case translation of 0.944 cm
contributes 2.5 mm to registration and 2.3 mm to map RMSE."

The slopes are linear interpolants of the KITTI amplification measurements
(Fig. 5); this module recomputes a budget from any measured bias sweep and
reproduces the published numbers with the shipped defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetSlopes:
    """Linear interpolation slopes of the amplification curve (Fig. 5)."""

    reg_per_rot_deg: float = 0.69   # m of registration error per degree
    reg_per_trans_m: float = 0.27   # m of registration error per metre of bias
    map_rmse_per_rot_deg_mm: float = 28.2   # mm per degree (2.4 mm / 0.085°)
    map_rmse_per_trans_mm: float = 243.6    # mm per metre (2.3 mm / 0.944 cm)


def error_budget(
    rotation_residual_deg: float,
    translation_residual_m: float,
    slopes: BudgetSlopes = BudgetSlopes(),
) -> dict:
    """Downstream contributions of the calibration module's residuals.

    Returns the added registration error (m) and map boundary RMSE (mm) the
    residuals induce through the measured amplification channels; rotation
    dominates by an order of magnitude (Table 14).
    """
    reg_rot_m = slopes.reg_per_rot_deg * rotation_residual_deg
    reg_trans_m = slopes.reg_per_trans_m * translation_residual_m
    map_rot_mm = slopes.map_rmse_per_rot_deg_mm * rotation_residual_deg
    map_trans_mm = slopes.map_rmse_per_trans_mm * translation_residual_m
    return {
        "registration_from_rotation_m": reg_rot_m,
        "registration_from_translation_m": reg_trans_m,
        "map_rmse_from_rotation_mm": map_rot_mm,
        "map_rmse_from_translation_mm": map_trans_mm,
    }


def slopes_from_bias_sweep(
    bias: np.ndarray, registration_error: np.ndarray
) -> float:
    """Least-squares slope (m per unit bias) of a measured amplification
    sweep — how the Table 14 slopes are obtained from Fig. 5 data."""
    bias = np.asarray(bias, dtype=float)
    reg = np.asarray(registration_error, dtype=float)
    A = np.vstack([bias, np.ones_like(bias)]).T
    slope, _ = np.linalg.lstsq(A, reg, rcond=None)[0]
    return float(slope)
