"""TRIAD-Reg — consistency-guided hierarchical registration (Sec. 3.3).

Pipeline (Fig. 3):

1. distance-weighted farthest point sampling (DW-FPS) retains far-field
   geometric support (:mod:`.sampling`);
2. a shared three-level MLP encoder produces hierarchical features
   (:mod:`.encoder`);
3. dual-softmax descriptor similarity + geometric compatibility select
   correspondence seeds (:mod:`.matching`);
4. consistency-guided attention (CGA) restricts matching to geometrically
   admissible regions (:mod:`.cga`);
5. iterative consistency propagation (Eq. 2) refines correspondence weights
   (:mod:`.propagation`);
6. weighted Kabsch + coarse-to-fine refinement produce the transform
   (:mod:`.pose`); training losses in :mod:`.losses`, full network in
   :mod:`.model`, KITTI training loop in :mod:`.train`.

The learned components require ``torch`` (``pip install -e ".[reg]"``).
"""

from .config import RegConfig  # torch-free

try:
    from .sampling import distance_weighted_fps
    from .encoder import HierarchicalEncoder
    from .matching import dual_softmax_similarity, select_seeds
    from .cga import geometric_compatibility, ConsistencyGuidedAttention
    from .propagation import ConsistencyPropagation
    from .pose import weighted_kabsch, hierarchical_refine, estimate_pose_std
    from .losses import registration_loss
    from .model import TRIADRegModel
    from .train import train, KittiOdometryDataset
    _TORCH_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - torch not installed
    _TORCH_AVAILABLE = False
    _IMPORT_ERROR = _exc

    def _require_torch(*_args, **_kwargs):
        raise ImportError(
            "TRIAD-Reg requires torch: pip install -e '.[reg]' "
            f"(original error: {_exc})"
        )

    distance_weighted_fps = HierarchicalEncoder = _require_torch
    dual_softmax_similarity = select_seeds = _require_torch
    geometric_compatibility = ConsistencyGuidedAttention = _require_torch
    ConsistencyPropagation = _require_torch
    weighted_kabsch = hierarchical_refine = estimate_pose_std = _require_torch
    registration_loss = TRIADRegModel = _require_torch
    train = KittiOdometryDataset = _require_torch

__all__ = [
    "RegConfig",
    "distance_weighted_fps",
    "HierarchicalEncoder",
    "dual_softmax_similarity",
    "select_seeds",
    "geometric_compatibility",
    "ConsistencyGuidedAttention",
    "ConsistencyPropagation",
    "weighted_kabsch",
    "hierarchical_refine",
    "estimate_pose_std",
    "registration_loss",
    "TRIADRegModel",
    "train",
    "KittiOdometryDataset",
]
