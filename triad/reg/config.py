"""TRIAD-Reg configuration.

Values named in the paper: DW-FPS weight ``w = ln(1 + d/d_max) + β``
(Sec. 3.3), compatibility scale ``σ_c`` of ``B_ij = ReLU(1 − |d_pair|/σ_c)``
(Sec. 3.3), three-level hierarchy, AdamW for 50 epochs on KITTI 00–05 with
06–07 validation and 08–10 test (Sec. 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config import BaseConfig


@dataclass
class RegConfig(BaseConfig):
    # -- DW-FPS (Sec. 3.3) --------------------------------------------------
    dw_fps_beta: float = 1.0        # β in w = ln(1 + d/d_max) + β
    keypoint_counts: tuple = (768, 1024, 1280)  # per hierarchy level, fine→coarse use reversed

    # -- encoder --------------------------------------------------------------
    num_levels: int = 3             # shared three-level MLP encoder (Sec. 3.3)
    feature_dim: int = 128
    mlp_hidden: int = 256

    # -- matching ---------------------------------------------------------------
    match_temperature: float = 0.1  # softmax temperature of dual-softmax (τ)
    num_seeds: int = 256            # correspondence seeds selected by combined confidence
    mutual_topk: int = 1            # mutual-nearest candidate correspondences

    # -- geometric compatibility (Sec. 3.3) ---------------------------------------
    compatibility_sigma: float = 0.1  # σ_c, on metric distances (m)
    seed_neighborhood_k: int = 32     # kNN neighbourhood of seed anchors for CGA

    # -- CGA -------------------------------------------------------------------
    attention_heads: int = 4
    attention_dim: int = 128

    # -- consistency propagation (Eq. 2) -------------------------------------------
    propagation_iterations: int = 3

    # -- coarse-to-fine -----------------------------------------------------------
    refine_radii: tuple = (1.0, 0.5, 0.25)  # local search radii per level (m)

    # -- training (Sec. 5.2) ---------------------------------------------------------
    epochs: int = 50
    lr: float = 1e-4              # AdamW learning rate
    weight_decay: float = 0.05    # AdamW weight decay
    batch_size: int = 1           # point-cloud pairs are batched by padding in collate
    train_overlap_thresh: float = 0.3
    val_overlap_thresh: float = 0.1
    loss_infonce_weight: float = 0.2
    loss_wbce_weight: float = 1.0
    loss_transform_weight: float = 1.0

    # -- evaluation ---------------------------------------------------------------------
    rte_thresh_m: float = 2.0     # RR = RTE < 2 m and RRE < 5° (Table 2 caption)
    rre_thresh_deg: float = 5.0
