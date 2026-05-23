"""Class-wise prototype bank with momentum updates.

During source training the bank tracks one prototype per class, refreshed
from the running mean of normalised image features. It feeds the keys and
values of :class:`pcma.modules.CrossDomainAttention` once every class has
been observed at least once.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class MomentumPrototypeBank:
    """Tracks one normalised prototype per class with momentum updates.

    The first time a class is seen the prototype is set directly from the
    batch mean; subsequent updates blend it with momentum ``m``::

        proto[c] <- m * proto[c] + (1 - m) * mean(features[labels == c])

    Args:
        num_classes: Total number of classes the bank covers.
        embed_dim: Feature dimensionality.
        momentum: Mixing coefficient for the momentum update.
        device: Device on which to allocate the prototype tensors.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        momentum: float = 0.9,
        device: str = "cuda",
    ) -> None:
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.device = device

        self.prototypes = torch.zeros(num_classes, embed_dim, device=device)
        self.initialised = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.update_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Refresh prototypes from a mini-batch of features.

        Args:
            features: L2-normalised image features of shape ``[B, D]``.
            labels: Integer class labels of shape ``[B]``.
        """
        features = features.detach()

        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            class_mean = features[mask].mean(dim=0)
            if not self.initialised[c]:
                self.prototypes[c] = class_mean
                self.initialised[c] = True
            else:
                m = self.momentum
                self.prototypes[c] = m * self.prototypes[c] + (1.0 - m) * class_mean
            self.update_counts[c] += 1

        self.prototypes = F.normalize(self.prototypes, p=2, dim=-1)

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.clone()

    def get_valid_mask(self) -> torch.Tensor:
        return self.initialised.clone()

    def get_stats(self) -> Dict[str, float]:
        valid = self.prototypes[self.initialised]
        if len(valid) > 0:
            proto_norm = valid.norm(dim=-1).mean().item()
            proto_std = valid.std().item()
        else:
            proto_norm = 0.0
            proto_std = 0.0
        return {
            "initialised_classes": int(self.initialised.sum().item()),
            "total_updates": int(self.update_counts.sum().item()),
            "proto_norm": proto_norm,
            "proto_std": proto_std,
        }
