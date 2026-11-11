"""Weighted Kabsch and coarse-to-fine pose estimation — Sec. 3.3.

"Converged scores join feature similarity in regressing per-correspondence
weights, and a weighted Kabsch solve produces the coarse transform at the
deepest level; upper levels re-align and refine with local searches."

Hierarchy levels are ordered coarse → fine (level 0 = deepest): the weighted
Kabsch solve runs at the deepest level on the propagated correspondence
weights, and each finer level re-aligns with a local (radius) search before
re-solving.  ``estimate_pose_std`` exports the pose dispersion that TRIAD-Link
L2 consumes as ``σ_t``.
"""

from __future__ import annotations

from typing import Optional

import torch


def weighted_kabsch(
    src: torch.Tensor, tgt: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Weighted rigid alignment ``tgt ≈ R @ src + t``; returns the 4x4.

    ``src``/``tgt`` are (N, 3) matched points; ``weights`` (N,) are the
    per-correspondence weights (propagated scores × similarity, Sec. 3.3).
    """
    if weights is None:
        weights = torch.ones(len(src), device=src.device)
    w = weights / weights.sum().clamp_min(1e-12)

    src_mean = (src * w[:, None]).sum(dim=0)
    tgt_mean = (tgt * w[:, None]).sum(dim=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = (w[:, None] * src_c).T @ tgt_c          # (3, 3)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=src.device, dtype=src.dtype))
    R = Vt.T @ D @ U.T

    t = tgt_mean - R @ src_mean
    T = torch.eye(4, device=src.device, dtype=src.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _local_rematch(
    src: torch.Tensor, tgt_points: torch.Tensor, T: torch.Tensor, radius: float
):
    """Nearest-target re-matching of transformed source points within ``radius``."""
    src_t = src @ T[:3, :3].T + T[:3, 3]
    dist = torch.cdist(src_t, tgt_points)
    nn_dist, nn_idx = dist.min(dim=1)
    keep = nn_dist < radius
    return keep, nn_idx


def hierarchical_refine(
    levels_src,          # list of (N_l, 3), coarse → fine
    levels_tgt,          # list of (N_l, 3), coarse → fine
    corr_src_idx,        # (M,) source indices at the deepest (coarsest) level
    corr_tgt_idx,        # (M,) target indices at the deepest level
    weights: torch.Tensor,  # (M,) propagated correspondence weights
    refine_radii,        # per finer level, local search radius (m)
) -> torch.Tensor:
    """Coarse-to-fine: weighted Kabsch at the deepest level, then re-align."""
    src_coarse = levels_src[0][corr_src_idx]
    tgt_coarse = levels_tgt[0][corr_tgt_idx]
    T = weighted_kabsch(src_coarse, tgt_coarse, weights)

    for level, radius in zip(range(1, len(levels_src)), refine_radii):
        src_pts = levels_src[level]
        tgt_pts = levels_tgt[level]
        keep, nn_idx = _local_rematch(src_pts, tgt_pts, T, radius)
        if int(keep.sum()) < 3:
            continue
        # carry the weights through the refinement: the inlier weights of the
        # previous solve are unknown for re-matched pairs, so use uniform
        # weights inside the surviving radius (local search, Sec. 3.3)
        T = weighted_kabsch(src_pts[keep], tgt_pts[nn_idx[keep]])
    return T


def estimate_pose_std(
    src: torch.Tensor, tgt: torch.Tensor, T: torch.Tensor, inlier_mask: torch.Tensor
) -> float:
    """Pose dispersion ``σ_t`` consumed by TRIAD-Link L2 (Eq. 3).

    The transform is fixed; ``σ_t`` is estimated as the RMS residual of the
    inlier correspondences — the dispersion the next map update should expect
    from this registration.  Returns metres.
    """
    src_t = src @ T[:3, :3].T + T[:3, 3]
    residuals = (src_t - tgt).norm(dim=1)[inlier_mask]
    if len(residuals) == 0:
        return float("inf")
    return float(torch.sqrt((residuals**2).mean()))
