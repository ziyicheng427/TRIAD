"""Registration metrics — Sec. 5.2, Table 2.

RTE: relative translation error (m); RRE: relative rotation error (deg);
RR: registration recall, the fraction of pairs with RTE < 2 m *and*
RRE < 5° (Table 2 caption).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def rte(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))


def rre(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    dR = T_pred[:3, :3].T @ T_gt[:3, :3]
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def registration_recall(
    rtes: Sequence[float], rres: Sequence[float],
    rte_thresh: float = 2.0, rre_thresh: float = 5.0,
) -> float:
    ok = [
        e < rte_thresh and a < rre_thresh for e, a in zip(rtes, rres)
    ]
    return float(np.mean(ok)) if ok else 0.0


def evaluate_pairs(
    predictions: Iterable[np.ndarray],
    ground_truths: Iterable[np.ndarray],
    rte_thresh: float = 2.0,
    rre_thresh: float = 5.0,
) -> Dict[str, float]:
    """RTE / RRE / RR over an evaluated pair set (KITTI 08–10 protocol)."""
    rtes: List[float] = []
    rres: List[float] = []
    for T_pred, T_gt in zip(predictions, ground_truths):
        rtes.append(rte(np.asarray(T_pred), np.asarray(T_gt)))
        rres.append(rre(np.asarray(T_pred), np.asarray(T_gt)))
    return {
        "RTE": float(np.mean(rtes)) if rtes else float("nan"),
        "RRE": float(np.mean(rres)) if rres else float("nan"),
        "RR": registration_recall(rtes, rres, rte_thresh, rre_thresh),
    }
