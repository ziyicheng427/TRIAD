"""Evaluation metrics of Sec. 5.

* :mod:`.registration_metrics` — RTE / RRE / RR (Tables 2, 7) and the
  KITTI/nuScenes evaluation protocol;
* :mod:`.map_metrics` — clearing/establishment delay (Tables 5-8), false
  occupied cells (Table 9), map IoU;
* :mod:`.cal_metrics` — calibration rotation/translation error and the
  covariance coverage / ECE of the L1 evaluation (Table 5);
* :mod:`.budget` — the stack-level error budget from measured amplification
  slopes (Table 14).
"""

from .registration_metrics import rte, rre, registration_recall, evaluate_pairs
from .map_metrics import clearing_delay, establishment_delay, false_occupied_cells, map_iou
from .cal_metrics import calibration_errors, covariance_coverage
from .budget import error_budget

__all__ = [
    "rte",
    "rre",
    "registration_recall",
    "evaluate_pairs",
    "clearing_delay",
    "establishment_delay",
    "false_occupied_cells",
    "map_iou",
    "calibration_errors",
    "covariance_coverage",
    "error_budget",
]
