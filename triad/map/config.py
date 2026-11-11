"""TRIAD-Map configuration — Sec. 3.4.

Paper-named values: the three occupancy-state bands (``p < 0.49``,
``[0.49, 0.55)``, ``p ≥ 0.55``), the persistence gate ``N = 3`` frames, and
the transition-band hit-model probability 0.85.  The base hit/miss model,
clamping bounds, resolution, and neighbourhood radius are fully configurable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config import BaseConfig


@dataclass
class MapConfig(BaseConfig):
    # -- grid ---------------------------------------------------------------
    resolution: float = 0.05     # m per cell (Cartographer 2D default)
    p_min: float = 0.10          # clamp bounds of the log-odds grid
    p_max: float = 0.90

    # -- base hit/miss model (fixed-gain default of Tables 5-6) ----------------
    p_hit: float = 0.55
    p_miss: float = 0.49

    # -- occupancy-state bands (Sec. 3.4, Fig. 4) --------------------------------
    p_free_threshold: float = 0.49     # below: persistence gate (PGC)
    p_transition_threshold: float = 0.55  # [0.49, 0.55): transition-band boost (TBB)
    persistence_frames: int = 3        # N = 3: hits admitted after 3 supporting scans
    p_hit_transition: float = 0.85     # TBB hit-model probability (Sec. 3.4)

    # -- persistence probe ---------------------------------------------------------
    neighborhood_radius: float = 0.15  # R: endpoint support radius (m)

    # -- occupancy threshold used by metrics and L3 ----------------------------------
    p_occupied_threshold: float = 0.55
    p_free_l3_threshold: float = 0.20  # L3: confidently free cells (Sec. 3.5)

    # -- 2D-3D bridge ------------------------------------------------------------------
    ground_normal_z: float = 0.90      # plane with |n·ẑ| above this is ground
    bridge_plane_distance: float = 0.10

    # -- baselines (Table 6) --------------------------------------------------------------
    region_window: int = 3             # RegionStatistics window (cells)
    change_decay: float = 0.85         # ChangeDecay decay factor per scan
    freq_harmonics: int = 4            # FrequencyDomain Fourier harmonics
