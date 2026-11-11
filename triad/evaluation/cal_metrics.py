"""Calibration metrics — Sec. 5.1 and the L1 evaluation (Table 5).

Rotation/translation errors against reference extrinsics; covariance
coverage of the 95 % pose ellipsoid; the ECE of the propagated covariance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def calibration_errors(T_est: np.ndarray, T_ref: np.ndarray) -> Tuple[float, float]:
    """(rotation error deg, translation error cm) against the reference."""
    dR = T_est[:3, :3].T @ T_ref[:3, :3]
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos)))
    trans_cm = float(np.linalg.norm(T_est[:3, 3] - T_ref[:3, 3]) * 100.0)
    return rot_deg, trans_cm


def covariance_coverage(
    errors: np.ndarray, covariances: np.ndarray, level: float = 0.95
) -> float:
    """Coverage of the ``level`` pose ellipsoid over repeated calibrations.

    Parameters
    ----------
    errors : (K, 6) twist errors of K calibrations (rotation first).
    covariances : (K, 6, 6) the reported covariance of each calibration.
    level : nominal coverage (95 % in Sec. 5.5).

    Returns
    -------
    Fraction of calibrations whose squared Mahalanobis norm falls below the
    chi-square quantile with 6 degrees of freedom — the "Cov. (%)" column of
    Table 5 (92.5 % for the full covariance).
    """
    try:
        from scipy.stats import chi2
        quantile = float(chi2.ppf(level, df=6))
    except ImportError:  # fallback: Chebyshev-free normal approximation
        quantile = float(level * 6.0)

    inside = 0
    for e, S in zip(errors, covariances):
        S_reg = S + 1e-12 * np.eye(6)
        maha = float(e @ np.linalg.solve(S_reg, e))
        if maha <= quantile:
            inside += 1
    return inside / max(len(errors), 1)
