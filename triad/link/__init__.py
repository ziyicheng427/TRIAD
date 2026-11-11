"""TRIAD-Link — propagating confidence across layers (Sec. 3.5).

Three explicit interfaces exchange reliability between the task modules:

* **L1** calibration covariance → observation variance
  (``sigma_i^2 += J_xi Sigma_xi J_xi^T``) — :mod:`.l1_covariance`;
* **L2** registration pose standard deviation → hit/miss gain damping
  (Eq. 3, σ₀ = 3 cm, asymmetric clips) — :mod:`.l2_gains`;
* **L3** occupancy belief → correspondence weights (free cells P < 0.2 carry
  weight 0.1; occupied and unknown keep theirs) — :mod:`.l3_reweight`.

:mod:`.transfer` implements the confidence-transfer strategies of the
control in Table 12 (uniform / module-specific / calibrated-no-transfer /
shuffled / full).
"""

from .config import LinkConfig
from .l1_covariance import propagate_to_observation_variance, map_gain_from_variance
from .l2_gains import l2_gains
from .l3_reweight import occupancy_reweight
from .transfer import TransferStrategy, apply_transfer_strategy

__all__ = [
    "LinkConfig",
    "propagate_to_observation_variance",
    "map_gain_from_variance",
    "l2_gains",
    "occupancy_reweight",
    "TransferStrategy",
    "apply_transfer_strategy",
]
