"""Adaptive selection of the number of sub-prototypes per class.

For each class the routine inspects how crowded its neighbourhood is in
the text-prototype space and assigns a number of sub-prototypes that
scales with the level of inter-class confusion. Classes whose top
neighbours sit far away keep a single prototype; classes that sit close
to several neighbours receive ``K > 1`` so the downstream label
propagation can split them with KMeans.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_boundary_stress(
    class_features: torch.Tensor,
    top_n: int = 5,
    open_threshold: float = 0.5,
    k_max: int = 3,
) -> Dict[int, int]:
    """Map each class to a suggested number of sub-prototypes.

    The score per class is the average cosine similarity to its ``top_n``
    nearest neighbours. Classes whose score exceeds ``open_threshold``
    receive ``k = k_max``; classes that are mildly confused receive
    intermediate values; well-isolated classes get ``k = 1``.

    Args:
        class_features: ``[C, D]`` per-class anchors (typically the source
            text prototypes after CLIP encoding).
        top_n: Number of neighbours used to score crowdedness.
        open_threshold: Similarity at or above which a class is considered
            fully open and gets ``k_max`` sub-prototypes.
        k_max: Maximum number of sub-prototypes returned.

    Returns:
        Dict mapping ``class_id -> k`` for every class with ``k > 1``.
        Classes with a single prototype are omitted.
    """
    feats = F.normalize(class_features.float(), p=2, dim=-1)
    sim = feats @ feats.t()
    num_classes = sim.size(0)
    sim.fill_diagonal_(-1.0)

    top_n = min(top_n, num_classes - 1)
    top_sim, _ = sim.topk(top_n, dim=-1)
    crowdedness = top_sim.mean(dim=-1)

    result: Dict[int, int] = {}
    # Linear ramp between [open_threshold/2, open_threshold] -> [2, k_max].
    low = open_threshold * 0.5
    span = max(open_threshold - low, 1e-6)
    for c in range(num_classes):
        score = float(crowdedness[c].item())
        if score >= open_threshold:
            k = k_max
        elif score > low:
            k = 1 + int(round((score - low) / span * (k_max - 1)))
            k = max(2, min(k, k_max))
        else:
            k = 1
        if k > 1:
            result[c] = k
    return result
