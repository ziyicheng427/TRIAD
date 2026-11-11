"""Geometric consistency-guided attention (CGA) — Sec. 3.3.

The consistency probe of TRIAD-Reg is the **geometric compatibility matrix**

    B_ij = ReLU(1 − |d_pair_ij| / σ_c)

over correspondence *pairs*: ``d_pair_ij`` compares the intra-source distance
``‖s_i − s_j‖`` with the intra-target distance ``‖t_i − t_j‖`` of the two
correspondences — compatible pairs preserve pairwise distances.

"Seeds are selected by combined confidence, and each source key point
attends only over the target neighborhoods of its reliable seeds,
restricting cross-attention to geometrically admissible regions."
"""

from __future__ import annotations

import torch
import torch.nn as nn


def geometric_compatibility(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    sigma_c: float,
) -> torch.Tensor:
    """(M, M) compatibility matrix ``B`` over M candidate correspondences.

    Parameters
    ----------
    src_points : (M, 3) source points of the correspondences.
    tgt_points : (M, 3) target points of the correspondences.
    sigma_c : compatibility scale σ_c (Sec. 3.3).
    """
    d_src = torch.cdist(src_points, src_points)  # ‖s_i − s_j‖
    d_tgt = torch.cdist(tgt_points, tgt_points)  # ‖t_i − t_j‖
    d_pair = (d_src - d_tgt).abs()               # |d_pair_ij|
    return torch.relu(1.0 - d_pair / sigma_c)


class ConsistencyGuidedAttention(nn.Module):
    """Masked multi-head cross-attention restricted to seed neighbourhoods.

    The attention mask marks, for every source keypoint, the target points
    inside the k-NN neighbourhoods of the reliable seeds' target anchors;
    logits outside the admissible region are −∞.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.attention_dim
        self.num_heads = config.attention_heads
        self.q_proj = nn.Linear(config.feature_dim, self.embed_dim)
        self.k_proj = nn.Linear(config.feature_dim, self.embed_dim)
        self.v_proj = nn.Linear(config.feature_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, config.feature_dim)

    def forward(
        self,
        feats_src: torch.Tensor,      # (N_s, D)
        feats_tgt: torch.Tensor,      # (N_t, D)
        tgt_points: torch.Tensor,     # (N_t, 3)
        seed_tgt_anchors: torch.Tensor,  # (S,) target indices of the seeds
    ) -> torch.Tensor:
        """Refined source features; attention restricted to admissible regions."""
        # admissible target set: union of k-NN neighbourhoods of seed anchors
        S = len(seed_tgt_anchors)
        if S == 0:
            # no seeds survive: plain (unmasked) cross-attention fallback
            mask = None
        else:
            anchor_pts = tgt_points[seed_tgt_anchors]              # (S, 3)
            dist = torch.cdist(tgt_points, anchor_pts)             # (N_t, S)
            knn = dist.topk(
                k=min(self.config.seed_neighborhood_k, len(tgt_points)),
                dim=0, largest=False,
            ).indices                                                 # (k, S)
            admissible = torch.zeros(len(tgt_points), dtype=torch.bool,
                                     device=tgt_points.device)
            admissible[knn.reshape(-1)] = True
            admissible[seed_tgt_anchors] = True
            mask = admissible.unsqueeze(0)                           # (1, N_t)

        q = self._split_heads(self.q_proj(feats_src))   # (h, N_s, d)
        k = self._split_heads(self.k_proj(feats_tgt))
        v = self._split_heads(self.v_proj(feats_tgt))

        logits = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)     # (h, N_s, N_t)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        out = attn @ v                                # (h, N_s, d)
        out = out.transpose(0, 1).reshape(len(feats_src), self.embed_dim)
        return self.out_proj(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.shape
        d = self.embed_dim // self.num_heads
        return x.view(N, self.num_heads, d).permute(1, 0, 2)
