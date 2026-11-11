"""L2: registration uncertainty-aware map updates — Sec. 3.5, Eq. (3).

    g_hit  = clip( σ₀² / (σ₀² + σ_t²), 0.25, 1 )
    g_miss = clip( σ₀² / (σ₀² + σ_t²), 0.6,  1 )                    (Eq. 3)

with ``σ₀ = 3 cm``.  "The asymmetry is deliberate: a hit is one displaced
endpoint whose artifact persists until many misses erase it, whereas a miss
is backed by the many rays crossing a cell per scan, whose pose-induced
errors partially cancel."
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import LinkConfig


def l2_gains(
    sigma_t: float,
    config: Optional[LinkConfig] = None,
) -> Tuple[float, float]:
    """Hit and miss gain multipliers from the registration pose std ``σ_t``.

    Returns ``(g_hit, g_miss)``; both are 1.0 at zero registration noise and
    floor at the asymmetric clip bounds as ``σ_t`` grows.
    """
    config = config or LinkConfig()
    sigma_t = max(float(sigma_t), 0.0)
    ratio = config.sigma_0**2 / (config.sigma_0**2 + sigma_t**2)
    g_hit = float(np.clip(ratio, config.hit_gain_clip_min, 1.0))
    g_miss = float(np.clip(ratio, config.miss_gain_clip_min, 1.0))
    return g_hit, g_miss
