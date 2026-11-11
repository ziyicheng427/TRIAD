"""Shared machinery: SE(3) algebra, the C2C operator, confidence calibration."""

from .se3 import (
    SE3,
    exp_se3,
    log_se3,
    exp_so3,
    log_so3,
    hat3,
    transform_points,
    point_jacobian_right,
)
from .c2c import C2COperator, ConsistencyProbe, ConfidenceMap, huber_weight
from .confidence_calibration import (
    IsotonicConfidenceMap,
    fit_isotonic,
    expected_calibration_error,
)

__all__ = [
    "SE3",
    "exp_se3",
    "log_se3",
    "exp_so3",
    "log_so3",
    "hat3",
    "transform_points",
    "point_jacobian_right",
    "C2COperator",
    "ConsistencyProbe",
    "ConfidenceMap",
    "huber_weight",
    "IsotonicConfidenceMap",
    "fit_isotonic",
    "expected_calibration_error",
]
