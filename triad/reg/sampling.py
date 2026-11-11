"""Distance-weighted farthest point sampling (DW-FPS) — Sec. 3.3.

Standard farthest point sampling over-selects in the near field of outdoor
scans, whose density decays non-uniformly with range.  DW-FPS compensates:

    "the next key point maximizes w_i d_i with
     w_i = ln(1 + d_i / d_max) + β,"

where ``d_i`` is the distance from candidate ``i`` to the already-selected
set and ``d_max`` is the scan's range scale (max distance to the centroid,
computed once per cloud).  The log weight preserves distant anchors that
constrain global rotation without over-attending extreme-range noise.
"""

from __future__ import annotations

import torch


def distance_weighted_fps(
    points: torch.Tensor,
    n_samples: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Batched DW-FPS.

    Parameters
    ----------
    points : (B, N, 3) point cloud batch.
    n_samples : number of keypoints to select.
    beta : ``β`` of the weight formula.

    Returns
    -------
    (B, n_samples) LongTensor of selected point indices.
    """
    B, N, _ = points.shape
    device = points.device
    n_samples = min(n_samples, N)

    # range scale of the scan: max distance to the centroid, per batch item
    centroid = points.mean(dim=1, keepdim=True)
    d_max = (points - centroid).norm(dim=-1).max(dim=1).values  # (B,)
    d_max = d_max.clamp_min(1e-6)

    # seed with the point farthest from the centroid
    dist_to_centroid = (points - centroid).norm(dim=-1)  # (B, N)
    first = dist_to_centroid.argmax(dim=1)  # (B,)

    min_dist = torch.full((B, N), float("inf"), device=device)
    selected = torch.empty(B, n_samples, dtype=torch.long, device=device)

    idx = first
    for k in range(n_samples):
        selected[:, k] = idx
        chosen = points[torch.arange(B, device=device), idx].unsqueeze(1)  # (B,1,3)
        dist = (points - chosen).norm(dim=-1)  # (B, N)
        min_dist = torch.minimum(min_dist, dist)

        w = torch.log1p(min_dist / d_max.unsqueeze(1)) + beta  # w_i = ln(1 + d_i/d_max) + β
        score = w * min_dist                                 # maximise w_i d_i
        score.scatter_(1, selected[:, : k + 1], -float("inf"))
        idx = score.argmax(dim=1)
    return selected
