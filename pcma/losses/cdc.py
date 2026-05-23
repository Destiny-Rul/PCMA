"""Cross-Domain Contrastive (CDC) loss anchoring target features to source text."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDomainContrastiveLoss(nn.Module):
    """InfoNCE-style loss that pulls target image features toward frozen
    source text prototypes.

    The "anchors" are the text features produced by the source-trained
    prompt at the start of adaptation. They stay fixed throughout target
    training, so this loss penalises large drifts of the target image
    features away from the source semantic geometry.

    Args:
        temperature: Softmax temperature applied to cosine similarities.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        source_text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            image_features: ``[B, D]`` L2-normalised target image features.
            source_text_features: ``[C, D]`` frozen source text features.
            labels: ``[B]`` pseudo-labels in ``[0, C)``.

        Returns:
            Scalar loss (cross-entropy over the similarity logits).
        """
        if source_text_features is None:
            return torch.tensor(0.0, device=image_features.device)
        sim = image_features @ source_text_features.t()
        return F.cross_entropy(sim / self.temperature, labels)
