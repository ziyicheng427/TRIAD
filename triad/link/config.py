"""TRIAD-Link configuration — Sec. 3.5.

Paper-named values: ``σ₀ = 3 cm`` and the asymmetric clip bounds of Eq. (3)
(0.25 for hits, 0.6 for misses); L3's free-cell threshold ``P < 0.2`` and
down-weight ``0.1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config import BaseConfig


@dataclass
class LinkConfig(BaseConfig):
    # -- L2 (Eq. 3) ---------------------------------------------------------
    sigma_0: float = 0.03            # σ₀ = 3 cm
    hit_gain_clip_min: float = 0.25  # g_hit = clip(σ₀²/(σ₀²+σ_t²), 0.25, 1)
    miss_gain_clip_min: float = 0.6  # g_miss = clip(σ₀²/(σ₀²+σ_t²), 0.6, 1)

    # -- L3 ---------------------------------------------------------------------
    l3_free_threshold: float = 0.2   # confidently free cells (P < 0.2)
    l3_free_weight: float = 0.1      # carry correspondence weight 0.1

    # -- factorial ablation switches (Sec. 5.5, Table 13) ---------------------------
    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True
