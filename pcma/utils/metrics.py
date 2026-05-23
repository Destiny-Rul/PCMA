"""Accuracy metrics used by the source and target stages."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)) -> List[float]:
    """Top-k accuracy over a mini-batch.

    Args:
        output: Logits of shape ``[B, C]``.
        target: Ground-truth labels of shape ``[B]``.
        topk: Iterable of ``k`` values to evaluate.

    Returns:
        A list of correct-count floats, one per entry in ``topk``.
    """
    maxk = max(topk)
    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def per_class_accuracy(preds, labels, num_classes: int) -> Dict[str, object]:
    """Per-class accuracy and its unweighted mean.

    Args:
        preds: Predicted labels (tensor or array).
        labels: Ground-truth labels (tensor or array).
        num_classes: Total number of classes.

    Returns:
        Dict with ``"per_class"`` (list of floats) and ``"mean"`` (float).
    """
    preds_arr = preds.cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    labels_arr = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    per_class: List[float] = []
    for c in range(num_classes):
        mask = labels_arr == c
        if mask.sum() > 0:
            per_class.append(float((preds_arr[mask] == labels_arr[mask]).sum() / mask.sum()))
        else:
            per_class.append(0.0)

    return {"per_class": per_class, "mean": float(np.mean(per_class))}
