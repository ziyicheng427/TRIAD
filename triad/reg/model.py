"""End-to-end TRIAD-Reg network — Sec. 3.3, Fig. 3.

forward(src_points, tgt_points) chains the pipeline:

DW-FPS + shared encoder → dual-softmax candidates → geometric compatibility
→ seed selection → CGA-refined descriptors → consistency propagation (Eq. 2)
→ weighted Kabsch + coarse-to-fine refinement.

TRIAD-Link hooks (Sec. 3.5) enter as optional inputs and leave as outputs:

* ``src_point_features`` — per-source-point extras (e.g. the L3
  occupancy-derived weights concatenated as input features);
* ``extra_sigma`` — per-correspondence observation standard deviations added
  in quadrature by L1's covariance propagation;
* outputs expose ``pose_std`` (the σ_t L2 consumes) and the calibrated
  correspondence confidences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .cga import ConsistencyGuidedAttention, geometric_compatibility
from .config import RegConfig
from .encoder import HierarchicalEncoder
from .matching import candidate_correspondences, dual_softmax_similarity, select_seeds
from .pose import estimate_pose_std, hierarchical_refine
from .propagation import ConsistencyPropagation


@dataclass
class RegOutput:
    transform: torch.Tensor        # (4, 4)
    src_idx: torch.Tensor          # (M,) correspondence source indices (coarse level)
    tgt_idx: torch.Tensor          # (M,)
    confidences: torch.Tensor      # (M,) propagated scores in [0, 1]
    weights: torch.Tensor          # (M,) final per-correspondence weights
    pose_std: float                # σ_t for TRIAD-Link L2


class TRIADRegModel(nn.Module):
    """The full registration network of Fig. 3."""

    def __init__(self, config: Optional[RegConfig] = None):
        super().__init__()
        self.config = config or RegConfig()
        self.encoder = HierarchicalEncoder(self.config)
        self.cga = ConsistencyGuidedAttention(self.config)
        self.propagation = ConsistencyPropagation(self.config.propagation_iterations)
        # regresses the final per-correspondence weights from the converged
        # propagation scores joined with feature similarity (Sec. 3.3)
        self.weight_head = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        src_points: torch.Tensor,   # (N, 3)
        tgt_points: torch.Tensor,   # (N, 3)
        src_point_features: Optional[torch.Tensor] = None,  # L3 input
        extra_sigma: Optional[torch.Tensor] = None,         # L1 input (M,)
    ) -> RegOutput:
        cfg = self.config

        levels_src = self.encoder(src_points, src_point_features)
        levels_tgt = self.encoder(tgt_points)

        # deepest (coarsest) level drives correspondence selection
        pts_src, feats_src = levels_src[0]
        pts_tgt, feats_tgt = levels_tgt[0]

        similarity = dual_softmax_similarity(feats_src, feats_tgt, cfg.match_temperature)
        src_idx, tgt_idx, sim_conf = candidate_correspondences(similarity, cfg.mutual_topk)
        if len(src_idx) < 3:
            identity = torch.eye(4, device=src_points.device)
            return RegOutput(
                transform=identity,
                src_idx=src_idx, tgt_idx=tgt_idx,
                confidences=sim_conf, weights=sim_conf,
                pose_std=float("inf"),
            )

        # consistency probe: pairwise geometric compatibility (Sec. 3.3)
        B = geometric_compatibility(pts_src[src_idx], pts_tgt[tgt_idx],
                                    cfg.compatibility_sigma)
        compat_score = B.mean(dim=1)  # mean compatibility with all candidates
        seeds = select_seeds(sim_conf, compat_score, cfg.num_seeds)

        # CGA restricts matching to the seed neighbourhoods, then the refined
        # descriptors re-score the candidates
        feats_src_ref = self.cga(feats_src, feats_tgt, pts_tgt, tgt_idx[seeds])
        sim_ref = dual_softmax_similarity(feats_src_ref, feats_tgt,
                                          cfg.match_temperature)
        s0 = sim_ref[src_idx, tgt_idx]

        # iterative consistency propagation (Eq. 2)
        scores = self.propagation(B, s0)

        # converged scores join feature similarity in regressing weights
        weights = self.weight_head(
            torch.stack([scores, s0], dim=1)
        ).squeeze(-1)

        if extra_sigma is not None:
            # L1: elevated observation variance → down-weight correspondences
            # where calibration is uncertain (Sec. 3.5)
            inflation = 1.0 / (1.0 + extra_sigma.clamp_min(0.0))
            weights = weights * inflation

        T = hierarchical_refine(
            [p for p, _ in levels_src],
            [p for p, _ in levels_tgt],
            src_idx, tgt_idx, weights, cfg.refine_radii,
        )

        # σ_t: RMS inlier residual under the final transform (L2 input)
        src_t = pts_src[src_idx] @ T[:3, :3].T + T[:3, 3]
        residuals = (src_t - pts_tgt[tgt_idx]).norm(dim=1)
        inliers = residuals < cfg.compatibility_sigma
        pose_std = estimate_pose_std(pts_src[src_idx], pts_tgt[tgt_idx], T, inliers)

        return RegOutput(
            transform=T,
            src_idx=src_idx, tgt_idx=tgt_idx,
            confidences=scores, weights=weights,
            pose_std=pose_std,
        )
