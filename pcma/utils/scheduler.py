"""Warmup-then-cosine learning-rate scheduler used during source training."""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    """Linear warmup followed by a cosine decay schedule.

    Args:
        optimizer: The optimizer whose ``param_groups`` should have their
            learning rates updated in-place.
        warmup_epochs: Number of epochs spent at ``warmup_lr`` before decay.
        total_epochs: Total number of training epochs.
        warmup_lr: Constant LR applied during warmup.
        base_lr: Peak LR at the end of warmup; cosine decay starts here.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-5,
        base_lr: float = 2e-3,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self) -> float:
        """Advance one epoch and apply the new learning rate."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.warmup_lr
        else:
            decay_span = max(self.total_epochs - self.warmup_epochs, 1)
            progress = (self.current_epoch - self.warmup_epochs) / decay_span
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
