"""Isotonic confidence-map calibration and expected calibration error (Sec. 5.4).

The C2C confidence maps ``Phi`` are fitted per module by isotonic regression
with at least 5000 labelled observations per module.  Calibration labels
identify:

* TRIAD-Cal — valid edge matches against reference extrinsics;
* TRIAD-Reg — inliers against ground-truth transformations;
* TRIAD-Map — reliable cell updates against a reference map.

Isotonic regression is implemented here directly (pool-adjacent-violators,
PAV) so the calibration logic is self-contained and inspectable.  Expected
calibration error (ECE) is computed with equal-width bins over the predicted
probability.

Reported in the paper: ECE falls from 0.114 → 0.034 (Cal), 0.095 → 0.030
(Reg), 0.126 → 0.037 (Map) after isotonic fitting (Table 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Minimum number of labelled probe observations recommended per module (Sec. 5.4).
MIN_CALIBRATION_OBSERVATIONS = 5000


def _pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    """Fit non-decreasing piecewise-constant values to ``y`` (unit weights).

    ``y`` are responses ordered by ascending score.  Returns the fitted values
    at the same positions, non-decreasing, minimising squared error (PAV).
    """
    y = np.asarray(y, dtype=float)
    blocks: list = []  # each block: [count, response_sum]
    for yi in y:
        blocks.append([1, yi])
        # pool backwards while the previous block's mean exceeds this one's
        while len(blocks) >= 2:
            prev_mean = blocks[-2][1] / blocks[-2][0]
            cur_mean = blocks[-1][1] / blocks[-1][0]
            if prev_mean <= cur_mean:
                break
            cnt, total = blocks.pop()
            blocks[-1][0] += cnt
            blocks[-1][1] += total

    fitted = np.concatenate(
        [np.full(cnt, total / cnt) for cnt, total in blocks]
    )
    return fitted


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> "IsotonicConfidenceMap":
    """Fit the confidence map ``Phi`` of one module (Sec. 5.4).

    Parameters
    ----------
    scores:
        raw consistency-probe scores (any monotone scale).
    labels:
        binary reliability labels of the module's reliability event.

    Returns
    -------
    IsotonicConfidenceMap
        callable map producing calibrated probabilities in [0, 1].
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be 1-D arrays of equal length")
    if len(scores) < MIN_CALIBRATION_OBSERVATIONS:
        import warnings

        warnings.warn(
            f"only {len(scores)} labelled probes; the paper fits Phi with at "
            f"least {MIN_CALIBRATION_OBSERVATIONS} per module (Sec. 5.4)",
            stacklevel=2,
        )

    order = np.argsort(scores, kind="stable")
    fitted = _pool_adjacent_violators(labels[order])
    return IsotonicConfidenceMap(scores[order], np.clip(fitted, 0.0, 1.0))


@dataclass
class IsotonicConfidenceMap:
    """Piecewise-constant, non-decreasing confidence map ``Phi``."""

    knots: np.ndarray  # sorted raw scores
    values: np.ndarray  # calibrated probabilities at the knots

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        # right-continuous step interpolation between knots
        idx = np.searchsorted(self.knots, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]

    @property
    def is_calibrated(self) -> bool:
        return len(self.knots) > 0


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error: sum of bin-wise |accuracy − confidence|.

    ``probs`` are predicted probabilities in [0, 1], ``labels`` binary
    outcomes.  Used for the raw-vs-calibrated ECE comparison of Table 4 and
    the covariance-ECE column of Table 5.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        ece += n / len(probs) * abs(labels[in_bin].mean() - probs[in_bin].mean())
    return float(ece)
