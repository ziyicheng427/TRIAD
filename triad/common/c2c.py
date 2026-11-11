"""The consistency-to-confidence (C2C) operator — Sec. 3.1, Eq. (1).

Each stage ``s ∈ {cal, reg, map}`` maintains a state ``theta_s`` and receives
an observation ``z_s`` of unknown reliability.  The C2C operator assigns
reliability in three steps:

1. a **consistency probe**  ``kappa_s(z_s; C_s) ∈ R^k`` measures agreement
   between the observation and its governing context ``C_s``;
2. a **confidence map**     ``Phi: R^k -> [0, 1]^m`` converts probe outputs
   into weights, decreasing in disagreement;
3. the module's estimator adapts to the weights.  For residual-based
   estimation (TRIAD-Cal, TRIAD-Reg) the principle takes the weighted form

       theta_hat = argmin_theta  sum_i  w_i * rho(r_i(z_s, theta)),
       w_i = Phi_i(kappa_s)                                        (Eq. 1)

   with a robust kernel ``rho``.  TRIAD-Map instantiates the same principle
   through persistence-dependent log-odds updates instead of weighted
   residuals (Sec. 3.4).

Module-specific probes (Sec. 3.1):

* calibration: cross-modal consistency between projected LiDAR edges and
  SAM-derived image boundaries (:mod:`triad.cal.association`);
* registration: pairwise geometric compatibility between correspondences
  (:mod:`triad.reg.cga`);
* mapping: endpoint persistence across consecutive frames
  (:mod:`triad.map.persistence`).

The confidence maps are fitted per module by isotonic regression against
module-specific reliability labels (Sec. 5.4); see
:mod:`triad.common.confidence_calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


def huber_weight(residuals: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights of the Huber kernel ``rho`` used in Eq. (1).

    ``w = 1`` inside ``|r| <= delta`` and ``delta / |r|`` outside — the
    standard robust kernel of the noise-aware estimator in Sec. 3.2.
    """
    r = np.asarray(residuals, dtype=float)
    return np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))


@dataclass
class ConsistencyProbe:
    """Result of probing one observation set.

    ``scores`` are raw consistency scores (higher = more consistent) and
    ``labels`` optionally carry the module's reliability labels used to fit
    the confidence map (Sec. 5.4): valid edge matches for calibration,
    registration inliers for registration, reliable cell updates for mapping.
    """

    scores: np.ndarray
    labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


#: A confidence map converts raw probe scores into weights in [0, 1].
ConfidenceMap = Callable[[np.ndarray], np.ndarray]


class C2COperator:
    """Weighted residual estimator implementing Eq. (1).

    Parameters
    ----------
    confidence_map:
        ``Phi`` — maps probe scores to weights in ``[0, 1]``.  Pass an
        :class:`~triad.common.confidence_calibration.IsotonicConfidenceMap`
        fitted on labelled probes, or the identity-like
        :func:`identity_map` before calibration.
    kernel:
        robust kernel ``rho``; ``"huber"`` by default (``delta`` in the same
        units as the noise-normalised residuals).
    """

    def __init__(
        self,
        confidence_map: ConfidenceMap,
        kernel: str = "huber",
        huber_delta: float = 1.345,
    ):
        if kernel != "huber":
            raise ValueError(f"unsupported robust kernel: {kernel!r}")
        self.confidence_map = confidence_map
        self.kernel = kernel
        self.huber_delta = huber_delta

    # ------------------------------------------------------------------
    # Step 1-2: probe → confidence
    # ------------------------------------------------------------------
    def weights_from_probe(self, probe: ConsistencyProbe) -> np.ndarray:
        """``w_i = Phi_i(kappa_s)`` — confidence weights for all observations."""
        return self.confidence_map(np.asarray(probe.scores, dtype=float))

    # ------------------------------------------------------------------
    # Step 3: weighted residual objective (Eq. 1)
    # ------------------------------------------------------------------
    def objective(
        self,
        residuals: np.ndarray,
        probe: ConsistencyProbe,
    ) -> float:
        """Value of ``sum_i w_i * rho(r_i)`` for noise-normalised residuals."""
        w = self.weights_from_probe(probe)
        r = np.asarray(residuals, dtype=float)
        d = self.huber_delta
        rho = np.where(
            np.abs(r) <= d, 0.5 * r**2, d * (np.abs(r) - 0.5 * d)
        )
        return float(np.sum(w * rho))

    def irls_weights(
        self,
        residuals: np.ndarray,
        probe: ConsistencyProbe,
        extra_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combined per-observation weights: confidence × robust-kernel × extra.

        The product is what enters the normal equations of the weighted
        estimators in TRIAD-Cal (Sec. 3.2) and TRIAD-Reg (Sec. 3.3).
        ``extra_weights`` accommodates the reliability factors that are not
        C2C quantities themselves, e.g. the direction and multi-frame weights
        of TRIAD-Cal or the L3 occupancy feedback of TRIAD-Link.
        """
        w = self.weights_from_probe(probe) * huber_weight(
            residuals, self.huber_delta
        )
        if extra_weights is not None:
            w = w * np.asarray(extra_weights, dtype=float)
        return w

    # ------------------------------------------------------------------
    # Utility used by every weighted solver
    # ------------------------------------------------------------------
    @staticmethod
    def weighted_normal_equation(
        J: np.ndarray, r: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """Solve ``min_sum w_i ||J_i dtheta + r_i||^2`` for ``dtheta``.

        ``J`` is (N, d, k), ``r`` is (N, d), ``w`` is (N,).  Returns the k-vector
        step of the (weighted) Gauss-Newton / Kabsch-style solve used by the
        Cal estimator and, in batched form, by Reg's pose head.
        """
        sqrt_w = np.sqrt(np.asarray(w, dtype=float))[:, None, None]
        Jw = J * sqrt_w
        rw = np.asarray(r, dtype=float) * np.sqrt(np.asarray(w, dtype=float))[:, None]
        H = np.einsum("ndk,ndl->kl", Jw, Jw)
        g = np.einsum("ndk,nd->k", Jw, rw)
        return np.linalg.solve(H + 1e-12 * np.eye(H.shape[0]), -g)


def identity_map(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Uncalibrated confidence map: clip raw scores into [0, 1].

    Used before the isotonic fit of Sec. 5.4 is available.
    """
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
