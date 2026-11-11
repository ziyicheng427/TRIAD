"""TRIAD-Cal pipeline — Sec. 3.2, Fig. 2.

One run accumulates point cloud over 25 s (Sec. 5.1): the caller supplies the
accumulated cloud plus the camera frames of the window.  The pipeline then

1. extracts depth-continuous LiDAR edges (plane intersections);
2. builds SAM-derived image boundaries and the attraction field per frame;
3. associates LiDAR edges with the field, weighting them by directional
   compatibility and multi-frame stability over the five-frame window;
4. initialises with the correspondence-rate-driven grid search and refines
   with the noise-aware Huber estimator, returning ``xi_hat`` and
   ``Sigma_xi`` (the L1 input).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..common.c2c import C2COperator, identity_map
from .association import CrossModalAssociation, temporal_weights_over_window
from .config import CalConfig
from .estimator import CalibResult, ExtrinsicEstimator, grid_search_init
from .image_edges import build_attraction_field
from .lidar_edges import extract_lidar_edges
from .sam_adapter import MaskSet, SamMaskExtractor


@dataclass
class CalFrame:
    """One camera frame: RGB image plus optional precomputed SAM masks."""

    image_rgb: np.ndarray
    masks: Optional[MaskSet] = None


class TriadCalPipeline:
    """End-to-end TRIAD-Cal over one accumulated cloud + frame window."""

    def __init__(
        self,
        config: Optional[CalConfig] = None,
        K: Optional[np.ndarray] = None,
        c2c: Optional[C2COperator] = None,
        sam: Optional[SamMaskExtractor] = None,
        seed: int = 0,
    ):
        self.config = config or CalConfig()
        self.K = np.asarray(K, dtype=float) if K is not None else None
        self.c2c = c2c or C2COperator(identity_map, huber_delta=self.config.huber_delta)
        self.sam = sam or SamMaskExtractor(self.config)
        self.seed = seed

    # ------------------------------------------------------------------
    def run(
        self,
        cloud: np.ndarray,
        frames: Sequence[CalFrame],
        T_nominal: np.ndarray,
    ) -> CalibResult:
        """Calibrate; ``T_nominal`` is the CAD-mount initial value."""
        if self.K is None:
            raise ValueError("camera intrinsics K are required")
        if len(frames) == 0:
            raise ValueError("at least one camera frame is required")

        # 1. depth-continuous LiDAR edges on the accumulated cloud
        edges = extract_lidar_edges(cloud, self.config, seed=self.seed)
        if not edges:
            raise RuntimeError("no plane-intersection edges found in the cloud")

        # 2. attraction field per frame (SAM grid prompts → thin → filter → DT)
        mask_sets = [
            self.sam.extract(f.image_rgb, precomputed=f.masks) for f in frames
        ]
        fields = [build_attraction_field(f.image_rgb, m, self.config)[0]
                  for f, m in zip(frames, mask_sets)]
        coverage = float(np.mean([m.coverage_ratio for m in mask_sets]))

        association = CrossModalAssociation(self.config, self.K)
        estimator = ExtrinsicEstimator(self.config, c2c=self.c2c)

        # 3. association on the latest frame; temporal weight over the window
        #    (the residual histories of the window's frames)
        history: List[np.ndarray] = []
        for field in fields:
            result = association.associate(edges, field, T_nominal)
            history.append(result.residuals)
        temporal = temporal_weights_over_window(history, self.config)

        assoc = association.associate(edges, fields[-1], T_nominal,
                                      extra_temporal=temporal)

        # 4. grid-search initialisation, then noise-aware refinement
        edge_pts = np.vstack([e.points for e in edges])
        T_init, _ = grid_search_init(edge_pts, self.K, fields[-1], T_nominal,
                                     self.config)
        estimator.attach(self.K, fields[-1])
        return estimator.estimate(assoc, T_init, mask_coverage=coverage)
