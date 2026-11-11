"""Shared three-level MLP encoder — Sec. 3.3.

"A shared three-level MLP encoder produces the features used for descriptor
matching and coarse-to-fine registration."

The same encoder (a three-layer MLP) is applied at every hierarchy level and
shared between source and target clouds.  Hierarchy levels are formed by
successive DW-FPS downsampling; each level's input features pool the previous
level's features of the discarded neighbourhood, so coarse levels summarise
larger regions.  ``point_features`` optionally injects inputs such as the L3
occupancy-derived weights (TRIAD-Link), which are features of the *source*
points only.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import RegConfig
from .sampling import distance_weighted_fps


class ThreeLevelMLP(nn.Module):
    """The shared encoder: a three-layer MLP (level 3 in the paper's name)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm1d expects (B, C); operate on point lists then re-stack
        shape = x.shape
        out = self.net(x.reshape(-1, shape[-1]))
        return out.reshape(*shape[:-1], -1)


def _pool_previous(
    points: torch.Tensor, prev_points: torch.Tensor, prev_feats: torch.Tensor
) -> torch.Tensor:
    """For each point, mean-pool the previous level's features of its k=1
    nearest neighbours (exact nearest here; levels differ by one stage)."""
    dist = torch.cdist(points, prev_points)  # (N_cur, N_prev)
    idx = dist.argmin(dim=1)                 # nearest previous-level point
    return prev_feats[idx]


class HierarchicalEncoder(nn.Module):
    """Builds the three-level hierarchy and encodes every level.

    ``forward(points)`` returns ``(points_l, feats_l)`` for
    ``l = 0 .. num_levels-1``, ordered coarse → fine: level 0 is the most
    downsampled (the deepest hierarchy level whose weighted Kabsch yields the
    coarse transform, Sec. 3.3), the last level is the finest.
    """

    def __init__(self, config: RegConfig, in_channels: int = 3):
        super().__init__()
        self.config = config
        self.levels = nn.ModuleList()
        counts = list(config.keypoint_counts)
        if len(counts) != config.num_levels:
            raise ValueError("keypoint_counts must have num_levels entries")
        # coarse → fine: counts descend from the raw cloud upward; the config
        # lists fine→coarse counts, so reverse
        for level in range(config.num_levels):
            in_dim = in_channels + (config.feature_dim if level > 0 else 0)
            self.levels.append(
                ThreeLevelMLP(in_dim, config.mlp_hidden, config.feature_dim)
            )
        self._counts_fine_to_coarse = counts[::-1]

    @property
    def counts_coarse_to_fine(self) -> List[int]:
        return self._counts_fine_to_coarse

    def forward(
        self,
        points: torch.Tensor,
        point_features: Optional[torch.Tensor] = None,
    ) -> List[tuple]:
        """Encode one cloud (unbatched: (N, 3) tensors).

        Returns a list of ``(points_l, feats_l)`` coarse → fine.
        """
        if points.dim() != 2:
            raise ValueError("HierarchicalEncoder encodes a single (N, 3) cloud")

        prev_points = points
        prev_feats = None
        outputs = []
        for level, mlp in enumerate(self.levels):
            n_keypoints = min(self._counts_fine_to_coarse[level], len(points))
            if n_keypoints < len(prev_points):
                idx = distance_weighted_fps(
                    prev_points.unsqueeze(0), n_keypoints, self.config.dw_fps_beta
                ).squeeze(0)
                pts_l = prev_points[idx]
                feats_at_l = point_features[idx] if point_features is not None else None
            else:
                pts_l = prev_points
                feats_at_l = point_features

            if level == 0:
                x = pts_l if feats_at_l is None else torch.cat(
                    [pts_l, feats_at_l], dim=-1
                )
            else:
                # each level pools the previous (finer) level's features of
                # its nearest neighbours before re-encoding
                agg = _pool_previous(pts_l, prev_points, prev_feats)
                x = torch.cat([pts_l, agg], dim=-1)
            feats_l = mlp(x)
            outputs.append((pts_l, feats_l))
            prev_points, prev_feats = pts_l, feats_l
        return outputs
