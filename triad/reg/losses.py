"""Training losses — Sec. 3.3.

"Training uses InfoNCE, weighted binary cross-entropy, and an L1 transform
loss over all hierarchy levels."

The three terms target the three stages of the network: InfoNCE on the
descriptor space, weighted BCE on the correspondence inliers (whose weights
are the C2C confidences), and the L1 transform loss on the final poses of
every hierarchy level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    feats_src: torch.Tensor,
    feats_tgt: torch.Tensor,
    positive_src: torch.Tensor,
    positive_tgt: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE between descriptors of matched keypoint pairs.

    ``positive_src``/``positive_tgt`` are index arrays of the ground-truth
    overlapping keypoint pairs; the negatives are all other keypoints of the
    other cloud.
    """
    if len(positive_src) == 0:
        return feats_src.sum() * 0.0  # keep the graph connected

    logits = feats_src @ feats_tgt.T / temperature          # (N_s, N_t)
    target = positive_tgt                                    # class index per row
    loss_src = F.cross_entropy(logits[positive_src], target)
    loss_tgt = F.cross_entropy(logits.T[positive_tgt], positive_src)
    return 0.5 * (loss_src + loss_tgt)


def weighted_bce_loss(
    confidence: torch.Tensor,
    inlier_labels: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Weighted binary cross-entropy on correspondence inlier labels.

    ``weights`` are the C2C confidence weights (Eq. 1 instantiated on the
    registration probe), so uncertain correspondences contribute less.
    """
    confidence = confidence.clamp(eps, 1.0 - eps)
    bce = -(
        inlier_labels * confidence.log()
        + (1.0 - inlier_labels) * (1.0 - confidence).log()
    )
    return (weights * bce).sum() / weights.sum().clamp_min(1e-12)


def transform_l1_loss(T_pred: torch.Tensor, T_gt: torch.Tensor) -> torch.Tensor:
    """L1 transform loss: mean absolute difference of the 4x4 transforms."""
    return (T_pred - T_gt).abs().mean()


def registration_loss(
    infonce_terms: list,
    wbce_terms: list,
    transform_terms: list,
    weights: tuple = (0.2, 1.0, 1.0),
) -> torch.Tensor:
    """Sum the three loss families "over all hierarchy levels" (Sec. 3.3)."""
    families = (
        (infonce_terms, weights[0]),
        (wbce_terms, weights[1]),
        (transform_terms, weights[2]),
    )
    non_empty = [terms for terms, _ in families if len(terms) > 0]
    if not non_empty:
        return torch.zeros(1, requires_grad=True).squeeze()
    total = non_empty[0][0] * 0.0  # zero on the right device, graph-connected
    for terms, weight in families:
        if len(terms) == 0:
            continue
        total = total + weight * torch.stack(terms).mean()
    return total
