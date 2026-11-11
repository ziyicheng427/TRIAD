"""Image-boundary extraction and the attraction field — Sec. 3.2.

SAM instance masks are converted into thinned boundary pixels:

1. per-instance boundaries are extracted and thinned with non-maximum
   suppression over Sobel responses;
2. an adaptive filter keeps edge pixels whose local RGB intensity standard
   deviation exceeds ``0.7 x`` the mean over instances, targeting clutter
   from grass, foliage, and pavement cracks;
3. a distance transform on the surviving boundaries yields the attraction
   field ``G(u, v)`` whose value at a projected LiDAR edge point is that
   point's residual (Sec. 3.2, "Reliability weighting").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .config import CalConfig
from .sam_adapter import MaskSet


_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


@dataclass
class AttractionField:
    """Distance-transform attraction field ``G`` over the image plane.

    ``field``    : (H, W) Euclidean distance to the nearest retained boundary
                   pixel, in pixels — the residual of a projected LiDAR edge
                   point.
    ``grad_u/grad_v`` : finite-difference gradients of ``G``, used by the
                   estimator Jacobian ("the Jacobian acquires the
                   attraction-field gradient", Sec. 3.2).
    """

    field: np.ndarray
    grad_u: np.ndarray
    grad_v: np.ndarray

    def value_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Bilinear-free nearest-pixel lookup of ``G`` at pixel coords (u, v)."""
        ui = np.clip(np.round(u).astype(int), 0, self.field.shape[1] - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.field.shape[0] - 1)
        return self.field[vi, ui]

    def gradient_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(N, 2) gradient (dG/du, dG/dv) at pixel coords (u, v)."""
        ui = np.clip(np.round(u).astype(int), 0, self.field.shape[1] - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.field.shape[0] - 1)
        return np.stack([self.grad_u[vi, ui], self.grad_v[vi, ui]], axis=1)

    def normals_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Unit image-edge normals ``∇G / |∇G|`` at (u, v).

        Used by the direction weight of Sec. 3.2: a projected LiDAR edge line
        should run *along* the boundary, i.e. perpendicular to the normal.
        """
        g = self.gradient_at(u, v)
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(norm, 1e-12)


def _instance_boundaries(mask: np.ndarray) -> np.ndarray:
    """Boolean boundary of one instance mask (mask ∧ ¬ eroded-mask)."""
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~eroded


def _thin_with_nms(boundary: np.ndarray) -> np.ndarray:
    """Non-maximum suppression over Sobel responses thins the edge map."""
    gx = ndimage.convolve(boundary.astype(float), _SOBEL_X, mode="nearest")
    gy = ndimage.convolve(boundary.astype(float), _SOBEL_Y, mode="nearest")
    magnitude = np.hypot(gx, gy)
    local_max = magnitude >= ndimage.maximum_filter(magnitude, size=3)
    return boundary & local_max & (magnitude > 0)


def filter_boundary_pixels(
    boundary: np.ndarray,
    image_rgb: np.ndarray,
    masks: Optional[MaskSet] = None,
    config: Optional[CalConfig] = None,
) -> np.ndarray:
    """Adaptive clutter filter (Sec. 3.2).

    Retains edge pixels whose local RGB intensity standard deviation exceeds
    ``0.7 x`` the mean local standard deviation over the instance set —
    "the filter targets clutter from grass, foliage, and pavement cracks".
    """
    config = config or CalConfig()
    gray = image_rgb.astype(float).mean(axis=2)
    local_std = ndimage.generic_filter(
        gray, np.std, size=5, mode="nearest"
    )
    if masks is not None and len(masks.masks) > 0:
        # mean local std over all instance-boundary pixels of the frame
        instance_boundary = np.zeros_like(boundary)
        for mask in masks.masks:
            instance_boundary |= _thin_with_nms(_instance_boundaries(mask))
        reference = float(local_std[instance_boundary].mean()) if instance_boundary.any() else 0.0
    else:
        reference = float(local_std[boundary].mean()) if boundary.any() else 0.0
    return boundary & (local_std > config.rgb_std_ratio * reference)


def build_attraction_field(
    image_rgb: np.ndarray,
    masks: MaskSet,
    config: Optional[CalConfig] = None,
) -> tuple[AttractionField, float]:
    """Distance-transform attraction field ``G`` from SAM masks (Sec. 3.2).

    Returns the field and the retained-boundary ratio (kept pixels over
    unfiltered boundary pixels) for diagnostics.
    """
    config = config or CalConfig()
    boundary = np.zeros(image_rgb.shape[:2], dtype=bool)
    for mask in masks.masks:
        boundary |= _thin_with_nms(_instance_boundaries(mask))
    n_unfiltered = int(boundary.sum())
    boundary = filter_boundary_pixels(boundary, image_rgb, masks, config)
    field = ndimage.distance_transform_edt(~boundary)
    grad_v, grad_u = np.gradient(field)  # array indexing: rows = v, cols = u
    kept_ratio = float(boundary.sum() / n_unfiltered) if n_unfiltered > 0 else 0.0
    return AttractionField(field=field, grad_u=grad_u, grad_v=grad_v), kept_ratio
