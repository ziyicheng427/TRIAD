"""Confidence-transfer strategies — Sec. 5.5, "Confidence transfer strategies"
(Table 12).

"The factorial design varies which links exist; a complementary control
varies *what they carry": uniform weights (discard the confidence),
module-specific confidences without calibration or transfer, calibrated C2C
confidences without cross-layer transfer, shuffled confidences (preserving
the distribution but breaking the observation association), and full
transfer.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class TransferStrategy(Enum):
    UNIFORM = "uniform"                    # discard the confidence entirely
    MODULE_SPECIFIC = "module_specific"    # raw per-module scores, no calibration
    CALIBRATED_NO_TRANSFER = "calibrated_no_transfer"  # Φ applied, links cut
    SHUFFLED = "shuffled"                  # Φ applied, then permuted across obs
    FULL = "full"                          # calibrated confidence, full transfer


def apply_transfer_strategy(
    raw_scores: np.ndarray,
    calibrated: np.ndarray | None,
    strategy: TransferStrategy,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Weights a module ships across its links under a transfer strategy.

    Parameters
    ----------
    raw_scores : per-observation raw consistency-probe scores of the module.
    calibrated : per-observation calibrated confidence (Φ of the scores), or
        ``None`` when no isotonic fit is available.
    strategy : the strategy of Table 12.
    rng : source of the shuffle permutation.

    Returns
    -------
    (N,) weights propagated through the link under the strategy.
    """
    raw_scores = np.asarray(raw_scores, dtype=float)
    rng = rng or np.random.default_rng(0)

    if strategy == TransferStrategy.UNIFORM:
        return np.ones_like(raw_scores)

    if strategy == TransferStrategy.MODULE_SPECIFIC:
        # module-specific confidences without calibration or transfer
        return np.clip(raw_scores, 0.0, 1.0)

    if calibrated is None:
        raise ValueError(f"{strategy} requires calibrated confidences")

    if strategy == TransferStrategy.CALIBRATED_NO_TRANSFER:
        # calibrated C2C confidences, but the link carries no cross-layer
        # information: observations keep their own confidence
        return np.asarray(calibrated, dtype=float)

    if strategy == TransferStrategy.SHUFFLED:
        # preserves the confidence distribution, breaks its correspondence
        # with the observations
        perm = rng.permutation(len(calibrated))
        return np.asarray(calibrated, dtype=float)[perm]

    return np.asarray(calibrated, dtype=float)  # FULL
