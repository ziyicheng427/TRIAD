"""L1: calibration uncertainty propagation — Sec. 3.5.

"The calibration module provides the extrinsic covariance to the downstream
observation models.  The covariance Σ_ξ maps to an observation-noise prior:
residuals inherit ``σ_i² += J_ξ Σ_ξ J_ξ^T``, so correspondences and map
updates are gentler where calibration is uncertain."

``J_ξ`` is the Jacobian of the observation quantity (a correspondence point
or a map endpoint, in the sensor frame) with respect to the extrinsic twist,
under the right-perturbation convention of
:func:`triad.common.se3.point_jacobian_right`.
"""

from __future__ import annotations

import numpy as np

from ..common.se3 import point_jacobian_right


def propagate_to_observation_variance(
    points_lidar: np.ndarray,
    T_lidar_cam: np.ndarray,
    Sigma_xi: np.ndarray,
) -> np.ndarray:
    """Per-point observation-variance inflation from the calibration covariance.

    Parameters
    ----------
    points_lidar : (N, 3) observed points in the LiDAR/sensor frame (the
        registration correspondences or the map endpoints).
    T_lidar_cam : the calibrated extrinsic whose covariance is propagated.
    Sigma_xi : (6, 6) calibration twist covariance from TRIAD-Cal.

    Returns
    -------
    (N,) variance increments ``J_ξ Σ_ξ J_ξ^T`` (m²) — one per point; add them
    to each observation's own variance.
    """
    points_lidar = np.asarray(points_lidar, dtype=float)
    J = point_jacobian_right(np.asarray(T_lidar_cam, dtype=float), points_lidar)
    # tr(J Σ J^T) per point: total positional variance from calibration
    JS = J @ np.asarray(Sigma_xi, dtype=float)
    return np.einsum("nij,nij->n", JS, J)


def map_gain_from_variance(
    extra_variance: np.ndarray, sigma_0: float = 0.03
) -> np.ndarray:
    """Convert L1's variance inflation into per-cell map-update gains.

    Map updates are "gentler where calibration is uncertain" (Sec. 3.5): the
    same damping shape as Eq. (3), applied per observation with the inflated
    variance in place of ``σ_t²``.  Returns gains in (0, 1].
    """
    extra_variance = np.asarray(extra_variance, dtype=float)
    return sigma_0**2 / (sigma_0**2 + extra_variance)
