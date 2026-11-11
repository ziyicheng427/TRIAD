"""Iterative consistency propagation — Sec. 3.3, Eq. (2).

"Consistency propagation iterates

    s_i^(t) = f( Σ_k B_ik s_k^(t−1) )                          (Eq. 2)

so that correspondences embedded in a globally compatible set gain score and
outliers are suppressed."

``B`` is the geometric compatibility matrix (a symmetric, non-negative
affinity), ``s^(0)`` the initial matching confidence, and ``f`` a bounded
non-linearity — here a sigmoid centred so that isolated (row-sum ≈ 0)
correspondences decay rather than persist.
"""

from __future__ import annotations

import torch


class ConsistencyPropagation(torch.nn.Module):
    """Iterates Eq. (2) over the compatibility matrix ``B``."""

    def __init__(self, iterations: int = 3):
        super().__init__()
        self.iterations = iterations

    def forward(self, B: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        B : (M, M) geometric compatibility matrix (rows need not sum to 1).
        s0 : (M,) initial correspondence confidences in [0, 1].

        Returns
        -------
        (M,) propagated scores in [0, 1].
        """
        s = s0.clone()
        row_norm = B.sum(dim=1).clamp_min(1e-8)  # degree of each correspondence
        for _ in range(self.iterations):
            agg = (B @ s) / row_norm             # mean neighbour score
            # f: sigmoid centred at 0.5 — compatible neighbourhoods reinforce,
            # incompatible ones pull the score below its previous value
            s = torch.sigmoid(4.0 * (agg - 0.5) + 2.0 * (s - 0.5))
        return s.clamp(0.0, 1.0)
