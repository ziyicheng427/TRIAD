"""TRIAD-Map — occupancy update with cross-frame persistence (Sec. 3.4).

The updater runs inside the point-cloud insertion stage of a standard SLAM
front end (Cartographer in the evaluation), with scan matching and the back
end untouched.  Components:

* :mod:`.grid`       — clamped log-odds grid with precomputed update tables;
* :mod:`.persistence` — the persistence probe: consecutive scans with an
  endpoint in the cell's radius-R neighbourhood, resetting on any scan
  without one;
* :mod:`.raycasting` — ray traversal for miss updates;
* :mod:`.updater`    — the occupancy-state-adaptive update rule: persistence
  gating (PGC) below ``p < 0.49``, transition-band boosting (TBB) in
  ``[0.49, 0.55)`` with hit-model probability 0.85, base hits at
  ``p >= 0.55`` (Fig. 4);
* :mod:`.bridge_3d`  — 2D-3D integration reusing the calibration module's
  voxel and plane machinery;
* :mod:`.baselines`  — the reimplemented dynamic-mapping baselines of
  Table 6.
"""

from .config import MapConfig
from .grid import LogOddsGrid, UpdateTable
from .persistence import PersistenceTracker
from .raycasting import cast_rays
from .updater import OccupancyUpdater, TriadMapUpdater, UpdateMode
from .bridge_3d import SolidStateBridge
from .baselines import (
    RegionStatisticsUpdater,
    ChangeDecayUpdater,
    FrequencyDomainUpdater,
    BeamContributionUpdater,
)

__all__ = [
    "MapConfig",
    "LogOddsGrid",
    "UpdateTable",
    "PersistenceTracker",
    "cast_rays",
    "OccupancyUpdater",
    "TriadMapUpdater",
    "UpdateMode",
    "SolidStateBridge",
    "RegionStatisticsUpdater",
    "ChangeDecayUpdater",
    "FrequencyDomainUpdater",
    "BeamContributionUpdater",
]
