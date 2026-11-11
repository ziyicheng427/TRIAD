"""Noise-aware extrinsic estimation — Sec. 3.2, "Noise-aware extrinsic estimation".

The extrinsic ``xi ∈ SE(3)`` minimises a Huber-kernel sum of noise-normalised
residuals.  Per projected edge point ``i``:

* residual  ``r_i = G(p_i)`` — the attraction-field value;
* noise     ``sigma_i`` combines the LiDAR angular disturbance, additive
  range noise, and the SAM edge localisation error of 1.2 px;
* Jacobian  ``J_i = ∇G^T · d(u,v)/d(p_cam) · d(p_cam)/d(xi)`` — "the Jacobian
  acquires the attraction-field gradient".

The estimator is a Gauss-Newton / IRLS solve on the right-perturbed twist
``T ← T · exp(dxi)`` and returns ``xi_hat`` together with its covariance

    Sigma_xi = (J^T Sigma_r^{-1} J)^{-1},

the quantity TRIAD-Link L1 forwards downstream.  Initialisation uses a
correspondence-rate-driven grid search; a mask coverage ratio below 0.6
triggers a warning that inflates the reported covariance by 50 percent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.c2c import C2COperator, ConsistencyProbe, identity_map
from ..common.se3 import exp_se3, point_jacobian_right, transform_points
from .association import AssociationResult, projection_jacobian, project_points
from .config import CalConfig


@dataclass
class CalibResult:
    """Output of TRIAD-Cal: the transformation *and* its covariance."""

    T_lidar_cam: np.ndarray   # 4x4, LiDAR frame → camera frame
    covariance: np.ndarray    # 6x6 twist covariance of xi (rotation first)
    converged: bool
    iterations: int
    correspondence_rate: float  # fraction of points within the threshold
    mask_coverage: float
    coverage_warning: bool
    final_cost: float


class ExtrinsicEstimator:
    """Weighted, noise-aware Huber estimation of the extrinsic (Eq. 1)."""

    def __init__(self, config: CalConfig, c2c: Optional[C2COperator] = None):
        self.config = config
        self.c2c = c2c or C2COperator(identity_map, huber_delta=config.huber_delta)
        self._K: Optional[np.ndarray] = None
        self._attraction = None

    # ------------------------------------------------------------------
    def attach(self, K: np.ndarray, attraction) -> None:
        """Provide the intrinsics and the attraction field of the frame.

        The field's gradients enter the Jacobian ("the Jacobian acquires the
        attraction-field gradient", Sec. 3.2).
        """
        self._K = np.asarray(K, dtype=float)
        self._attraction = attraction

    # ------------------------------------------------------------------
    def point_sigmas(self, assoc: AssociationResult) -> np.ndarray:
        """Per-point noise ``sigma_i`` in pixels (Sec. 3.2).

        ``sigma_i^2 = sigma_SAM^2 + |P_J · (sigma_range + ||p|| sigma_ang)|^2``
        with the pinhole projection Jacobian ``P_J`` carrying the 3-D LiDAR
        noise into pixel units.
        """
        if len(assoc.points) == 0:
            return np.zeros(0)
        pts_cam = np.stack([p.point_cam for p in assoc.points])
        P_J = projection_jacobian(self._K, pts_cam)  # (N, 2, 3)
        sigma_3d = self.config.lidar_range_sigma + np.linalg.norm(pts_cam, axis=1) * (
            np.deg2rad(self.config.lidar_angular_sigma_deg)
        )
        lidar_px = np.linalg.norm(P_J * sigma_3d[:, None, None], axis=(1, 2))
        return np.sqrt(self.config.sam_edge_sigma_px**2 + lidar_px**2)

    # ------------------------------------------------------------------
    def estimate(
        self,
        assoc: AssociationResult,
        T_init: np.ndarray,
        mask_coverage: float = 1.0,
    ) -> CalibResult:
        """Refine ``T_init`` by noise-aware IRLS; return the full result."""
        if self._K is None or self._attraction is None:
            raise RuntimeError("call attach(K, attraction) before estimate()")
        if len(assoc.points) == 0:
            raise ValueError("no projected edge points; cannot estimate")
        cfg = self.config

        pts_lidar = np.stack([p.point_lidar for p in assoc.points])
        uv = np.stack([p.uv for p in assoc.points])
        residuals = assoc.residuals  # r_i = G(p_i): pixel lookup, frame-fixed
        extra = assoc.direction_weights * assoc.temporal_weights
        probe = ConsistencyProbe(scores=assoc.consistency_scores)
        sigma_i = self.point_sigmas(assoc)
        coverage_warning = mask_coverage < cfg.coverage_warn_threshold

        T = np.asarray(T_init, dtype=float).copy()
        r_bar = residuals / sigma_i
        w = self.c2c.irls_weights(r_bar, probe, extra_weights=extra)

        converged = False
        cost_prev = np.inf
        iterations = 0
        for iterations in range(1, cfg.max_iterations + 1):
            pts_cam = transform_points(T, pts_lidar)

            # noise-normalised residuals and C2C weights (Eq. 1)
            r_bar = residuals / sigma_i
            w = self.c2c.irls_weights(r_bar, probe, extra_weights=extra)

            # Jacobian: attraction-field gradient ← projection ← twist
            P_J = projection_jacobian(self._K, pts_cam)            # (N,2,3)
            Q_J = point_jacobian_right(T, pts_lidar)               # (N,3,6)
            grad_uv = self._attraction.gradient_at(uv[:, 0], uv[:, 1])  # (N,2)
            J = np.einsum("ni,nik,njl->nl", grad_uv, P_J, Q_J)     # (N,6)
            J_bar = J / sigma_i[:, None]

            # weighted normal equations for the twist step
            H = (J_bar * w[:, None]).T @ J_bar
            g = (J_bar * w[:, None]).T @ r_bar
            try:
                dxi = -np.linalg.solve(H + 1e-10 * np.eye(6), g)
            except np.linalg.LinAlgError:
                dxi = -np.linalg.lstsq(H + 1e-6 * np.eye(6), g, rcond=None)[0]
            T = T @ exp_se3(dxi)

            cost = self.c2c.objective(r_bar, probe)
            if abs(cost_prev - cost) < cfg.convergence_tol:
                converged = True
                break
            cost_prev = cost

        # final covariance: Sigma_xi = (J^T Sigma_r^{-1} J)^{-1}, with the
        # final robust weights folded in (standard IRLS covariance)
        pts_cam = transform_points(T, pts_lidar)
        P_J = projection_jacobian(self._K, pts_cam)
        Q_J = point_jacobian_right(T, pts_lidar)
        grad_uv = self._attraction.gradient_at(uv[:, 0], uv[:, 1])
        J = np.einsum("ni,nik,njl->nl", grad_uv, P_J, Q_J)
        J_w = J * np.sqrt(w)[:, None]
        covariance = np.linalg.inv(J_w.T @ J_w + 1e-10 * np.eye(6))
        if coverage_warning:
            covariance = cfg.coverage_inflation * covariance

        rate = float(np.mean(residuals < cfg.correspondence_thresh_px))
        return CalibResult(
            T_lidar_cam=T,
            covariance=covariance,
            converged=converged,
            iterations=iterations,
            correspondence_rate=rate,
            mask_coverage=float(mask_coverage),
            coverage_warning=coverage_warning,
            final_cost=float(cost_prev),
        )


# ----------------------------------------------------------------------
def grid_search_init(
    edges_points_lidar: np.ndarray,
    K: np.ndarray,
    attraction,
    T_nominal: np.ndarray,
    config: CalConfig,
) -> tuple[np.ndarray, float]:
    """Correspondence-rate-driven grid-search initialisation (Sec. 3.2).

    The search runs over the same ranges over which the evaluation randomises
    initial values (±5° / ±10 cm of nominal, Sec. 5.1) in two stages: a
    rotation grid at zero translation offset, then a translation grid at the
    best rotation.  Each candidate is scored by the correspondence rate — the
    fraction of projected edge points whose attraction-field residual falls
    below the threshold.  Returns the best transform and its rate.
    """

    def correspondence_rate(T_cand: np.ndarray) -> float:
        pts_cam = transform_points(T_cand, edges_points_lidar)
        uv = project_points(K, pts_cam)
        h, w = attraction.field.shape
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w)
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            & (pts_cam[:, 2] > 0)
        )
        if not np.any(inside):
            return -1.0
        r = attraction.value_at(uv[inside, 0], uv[inside, 1])
        return float(np.mean(r < config.correspondence_thresh_px))

    angles = np.linspace(
        -config.init_rot_range_deg, config.init_rot_range_deg,
        int(2 * config.init_rot_range_deg / config.init_grid_rot_step_deg) + 1,
    )
    offsets = np.linspace(
        -config.init_trans_range_m, config.init_trans_range_m,
        int(2 * config.init_trans_range_m / config.init_grid_trans_step_m) + 1,
    )

    # stage 1: rotation grid at zero translation offset
    best_T, best_rate = T_nominal.copy(), correspondence_rate(T_nominal)
    for rx in angles:
        for ry in angles:
            for rz in angles:
                xi = np.array([np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz), 0, 0, 0])
                T_cand = T_nominal @ exp_se3(xi)
                rate = correspondence_rate(T_cand)
                if rate > best_rate:
                    best_T, best_rate = T_cand, rate

    # stage 2: translation grid at the best rotation
    for tx in offsets:
        for ty in offsets:
            for tz in offsets:
                xi = np.array([0.0, 0.0, 0.0, tx, ty, tz])
                T_cand = best_T @ exp_se3(xi)
                rate = correspondence_rate(T_cand)
                if rate > best_rate:
                    best_T, best_rate = T_cand, rate
    return best_T, best_rate
