"""Candidate correspondences and seed selection — Sec. 3.3.

"Candidate correspondences are scored by descriptor similarity with dual
softmax normalization. ... Seeds are selected by combined confidence."

Dual-softmax normalisation of the similarity matrix (softmax over the source
axis, then over the target axis) produces a softly mutual matching matrix
from which candidate correspondences and their confidences are read off.
"""

from __future__ import annotations

from typing import Tuple

import torch


def dual_softmax_similarity(
    feats_src: torch.Tensor,
    feats_tgt: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """(N_s, N_t) dual-softmax-normalised similarity matrix.

    ``S = softmax(f_src f_tgt^T / τ, dim=1) ⊙ softmax(..., dim=0)`` — values
    in [0, 1], sharply peaked at softly-mutual matches.
    """
    logits = feats_src @ feats_tgt.T / temperature
    return torch.softmax(logits, dim=1) * torch.softmax(logits, dim=0)


def candidate_correspondences(
    similarity: torch.Tensor,
    topk: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual top-k candidates from a dual-softmax matrix.

    Returns ``(src_idx, tgt_idx, confidence)`` sorted by descending
    confidence, keeping pairs that are top-k in *both* directions.
    """
    n_src, n_tgt = similarity.shape
    top_from_tgt = similarity.topk(k=min(topk, n_src), dim=0).indices  # (k, N_t)

    src_best = similarity.argmax(dim=1)  # (N_s,)
    tgt_best = similarity.argmax(dim=0)  # (N_t,)

    pairs = []
    for i in range(n_src):
        j = int(src_best[i])
        if int(tgt_best[j]) == i:
            pairs.append((i, j, float(similarity[i, j])))
        elif topk > 1 and i in top_from_tgt[:, j]:
            pairs.append((i, j, float(similarity[i, j])))
    if not pairs:
        empty_l = torch.zeros(0, dtype=torch.long, device=similarity.device)
        empty_f = torch.zeros(0, device=similarity.device)
        return empty_l, empty_l, empty_f
    pairs.sort(key=lambda p: -p[2])
    src_idx = torch.tensor([p[0] for p in pairs], device=similarity.device)
    tgt_idx = torch.tensor([p[1] for p in pairs], device=similarity.device)
    conf = torch.tensor([p[2] for p in pairs], device=similarity.device)
    return src_idx, tgt_idx, conf


def select_seeds(
    confidence: torch.Tensor,
    compatibility_score: torch.Tensor,
    num_seeds: int,
) -> torch.Tensor:
    """Seed selection by combined confidence (Sec. 3.3).

    Combined confidence = descriptor-similarity confidence × mean geometric
    compatibility with the other candidates; the top-``num_seeds`` candidates
    become the seeds whose target neighbourhoods CGA restricts attention to.
    """
    combined = confidence * compatibility_score
    k = min(num_seeds, len(combined))
    return combined.topk(k=k).indices
