"""TRIAD: consistency-guided calibration, registration, and occupancy mapping.

Package layout mirrors the paper's four components:

- ``triad.common`` — the C2C operator (Eq. 1), isotonic confidence calibration,
  and SE(3) Lie-group utilities shared by all modules.
- ``triad.cal``    — TRIAD-Cal, targetless LiDAR-camera calibration (Sec. 3.2).
- ``triad.reg``    — TRIAD-Reg, hierarchical point cloud registration (Sec. 3.3).
- ``triad.map``    — TRIAD-Map, persistence-aware dynamic occupancy updates (Sec. 3.4).
- ``triad.link``   — TRIAD-Link, cross-layer confidence propagation (Sec. 3.5).
- ``triad.pipeline`` — the full stack with per-link switches.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
