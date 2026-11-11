"""SE(3) Lie-group utilities shared by TRIAD-Cal and TRIAD-Link L1.

Conventions
-----------
* A rigid transform is a 4x4 homogeneous matrix ``T`` with rotation block
  ``R ∈ SO(3)`` and translation ``t ∈ R^3``.
* A twist ``xi ∈ R^6`` stacks the rotation vector first:
  ``xi = [omega (3), v (3)]`` such that ``exp_se3(xi)`` is the matrix
  exponential of ``[[hat(omega), v], [0, 0]]``.
* Perturbations use the *right* convention ``T ← T · exp(hat(dxi))``. With
  this convention the Jacobian of a transformed point ``q = T·p`` with respect
  to ``dxi`` is ``[R, -hat(q)·R]`` (:func:`point_jacobian_right`), which is the
  quantity TRIAD-Link L1 combines with the calibration covariance
  ``Sigma_xi`` to inflate observation variance.
"""

from __future__ import annotations

import numpy as np


def hat3(v: np.ndarray) -> np.ndarray:
    """3-vector → skew-symmetric matrix."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula: rotation vector → rotation matrix."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + hat3(w)
    k = w / theta
    K = hat3(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (principal branch)."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # near pi: recover the axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        k = int(np.argmax(axis))
        if axis[k] > 1e-8:
            axis = A[:, k] / axis[k]
        axis = axis / np.linalg.norm(axis)
        # disambiguate the sign
        if np.trace(hat3(axis) @ R) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Twist [omega, v] → 4x4 SE(3) matrix (closed form, small-angle safe)."""
    xi = np.asarray(xi, dtype=float)
    w, v = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        R, t = np.eye(3), v.copy()
    else:
        k = w / theta
        K = hat3(k)
        R = exp_so3(w)
        # left Jacobian of SO(3)
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * K
            + (theta - np.sin(theta)) / theta**3 * (K @ K)
        )
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) matrix → twist [omega, v]."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    w = log_so3(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.concatenate([w, t])
    k = w / theta
    K = hat3(k)
    # inverse of the left Jacobian
    half = 0.5 * theta
    V_inv = (
        np.eye(3)
        - 0.5 * K
        + (1.0 / theta**2)
        * (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
        * (K @ K)
    )
    v = V_inv @ t
    return np.concatenate([w, v])


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) point array."""
    pts = np.asarray(pts, dtype=float)
    return pts @ np.asarray(T[:3, :3], dtype=float).T + np.asarray(T[:3, 3], dtype=float)


def point_jacobian_right(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Jacobian of ``T·p`` w.r.t. a right perturbation ``T ← T·exp(hat(dxi))``.

    Returns an (N, 3, 6) stack: ``d(T p) / d(dxi) = [R, -hat(T p) R]``.
    Used by the TRIAD-Cal estimator and by TRIAD-Link L1
    (``sigma_i^2 += J_xi Sigma_xi J_xi^T``).
    """
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float)
    R = T[:3, :3]
    q = transform_points(T, pts)  # (N, 3)
    N = pts.shape[0]
    J = np.empty((N, 3, 6))
    J[:, :, :3] = R[None, :, :]
    J[:, :, 3:] = -np.einsum("nij,jk->nik", _skew_batched(q), R)
    return J


def _skew_batched(v: np.ndarray) -> np.ndarray:
    """(N, 3) → (N, 3, 3) skew matrices."""
    N = v.shape[0]
    S = np.zeros((N, 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S


class SE3:
    """Small namespace of SE(3) helpers operating on 4x4 matrices."""

    exp = staticmethod(exp_se3)
    log = staticmethod(log_se3)

    @staticmethod
    def inverse(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        R, t = T[:3, :3], T[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = R.T
        inv[:3, 3] = -R.T @ t
        return inv

    @staticmethod
    def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        return np.asarray(T1, dtype=float) @ np.asarray(T2, dtype=float)

    @staticmethod
    def rotation_error_deg(T_est: np.ndarray, T_ref: np.ndarray) -> float:
        """Relative rotation error in degrees (TRIAD-Reg RRE metric, Sec. 5.2)."""
        dR = np.asarray(T_est)[:3, :3].T @ np.asarray(T_ref)[:3, :3]
        cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def translation_error(T_est: np.ndarray, T_ref: np.ndarray) -> float:
        """Relative translation error norm in metres (RTE metric, Sec. 5.2)."""
        return float(
            np.linalg.norm(
                np.asarray(T_est)[:3, 3] - np.asarray(T_ref)[:3, 3]
            )
        )
