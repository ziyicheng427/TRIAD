"""SAM adapter — 16x16 grid prompting for image boundaries (Sec. 3.2).

"On the camera side, a 16 by 16 point grid prompts SAM to produce masks."

The adapter wraps Meta's ``segment-anything`` package when installed and
falls back to caller-supplied masks otherwise, so the calibration pipeline
also runs on precomputed segmentations.  Thinning, the adaptive RGB filter,
and the distance transform live in :mod:`.image_edges`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CalConfig


@dataclass
class MaskSet:
    """Instance masks produced by SAM (or supplied by the caller).

    ``masks``  : (M, H, W) boolean array
    ``scores`` : (M,) predicted mask quality
    """

    masks: np.ndarray
    scores: np.ndarray

    @property
    def coverage_ratio(self) -> float:
        """Fraction of image pixels covered by at least one mask.

        The mask coverage ratio feeds the Sec. 3.2 warning rule: below 0.6,
        the reported covariance is inflated by 50 percent.
        """
        if len(self.masks) == 0:
            return 0.0
        return float(np.any(self.masks, axis=0).mean())


class SamMaskExtractor:
    """Produce :class:`MaskSet` from an image via a 16x16 point-grid prompt."""

    def __init__(self, config: CalConfig, model_type: str = "vit_h",
                 checkpoint: Optional[str] = None, device: str = "cpu"):
        self.config = config
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None
        self._checked_backend = False

    # ------------------------------------------------------------------
    def _ensure_backend(self) -> bool:
        """Lazily import ``segment_anything``; return whether it is available."""
        if self._checked_backend:
            return self._predictor is not None
        self._checked_backend = True
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except ImportError:
            return False
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        return True

    # ------------------------------------------------------------------
    def grid_points(self, width: int, height: int) -> np.ndarray:
        """The 16x16 prompt grid in pixel coordinates (Sec. 3.2)."""
        g = self.config.sam_grid
        us = np.linspace(0.0, width - 1.0, g)
        vs = np.linspace(0.0, height - 1.0, g)
        uu, vv = np.meshgrid(us, vs)
        return np.stack([uu.ravel(), vv.ravel()], axis=1)

    # ------------------------------------------------------------------
    def extract(self, image_rgb: np.ndarray, precomputed: Optional[MaskSet] = None) -> MaskSet:
        """Masks for one frame.

        With ``segment-anything`` installed, every grid point prompts the
        predictor (label 1 = foreground) and the returned masks are collected
        with their scores.  Without it, ``precomputed`` masks are passed
        through, keeping the rest of the pipeline runnable.
        """
        if precomputed is not None:
            return precomputed

        h, w = image_rgb.shape[:2]
        if not self._ensure_backend():
            raise RuntimeError(
                "segment-anything is not installed and no precomputed masks "
                "were supplied; see requirements.txt ([sam] extra)"
            )
        self._predictor.set_image(image_rgb)
        points = self.grid_points(w, h)
        labels = np.ones(len(points), dtype=int)

        masks: List[np.ndarray] = []
        scores: List[float] = []
        # one prompt per grid point; SAM predicts a single mask per prompt
        for point in points:
            mask, score, _ = self._predictor.predict(
                point_coords=point[None, :].astype(float),
                point_labels=labels[:1],
                multimask_output=False,
            )
            masks.append(mask[0])
            scores.append(float(score[0]))
        return MaskSet(
            masks=np.asarray(masks, dtype=bool), scores=np.asarray(scores)
        )
