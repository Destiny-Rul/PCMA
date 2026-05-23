"""Cross-domain attention block used by the PCMA text branch (ViL-T).

The module refines text prototypes by attending over class-wise visual
anchors and adding the residual through a learned sigmoid gate, which
keeps the update small at the start of training and lets it grow as the
visual anchors become reliable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionStats:
    """Diagnostics returned by ``CrossDomainAttention`` when requested."""

    gate_mean: float
    gate_std: float
    gate_min: float
    gate_max: float
    attn_entropy: float
    attn_diag: float
    feat_delta: float
    feat_sim: float


class CrossDomainAttention(nn.Module):
    """Cross-domain attention with a sigmoid-gated residual.

    Args:
        embed_dim: Dimensionality of the text/visual embedding.
        init_gate_bias: Bias of the gate projection at initialisation. A
            negative value (default ``-2.0``) keeps the residual small early
            in training.
        scale_factor: Sharpens the softmax temperature; the effective scale
            applied before softmax is ``sqrt(embed_dim) / scale_factor``.

    Shapes:
        - text_query: ``[C, D]`` text prototypes.
        - visual_anchors: ``[C, D]`` class-wise visual anchors.
        - output: ``[C, D]`` refined text prototypes (L2-normalised).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        init_gate_bias: float = -2.0,
        scale_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate_proj = nn.Linear(embed_dim, 1, bias=True)

        self._reset_parameters(init_gate_bias)

    def _reset_parameters(self, init_gate_bias: float) -> None:
        # Identity init keeps queries and keys aligned with the input space at
        # step 0, so the attention output starts as a class-wise mean of the
        # visual anchors before training begins to deform it.
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)

    def forward(
        self,
        text_query: torch.Tensor,
        visual_anchors: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor:
        text_query = text_query.float()
        visual_anchors = visual_anchors.float()

        q = self.q_proj(text_query)
        k = self.k_proj(visual_anchors)
        v = visual_anchors

        scale = math.sqrt(self.embed_dim) / self.scale_factor
        attn_weights = torch.softmax(q @ k.t() / scale, dim=-1)
        attended = attn_weights @ v

        gate = torch.sigmoid(self.gate_proj(text_query))
        delta = attended - text_query
        refined = text_query + gate * delta
        refined = F.normalize(refined, p=2, dim=-1)

        if not return_stats:
            return refined

        with torch.no_grad():
            text_norm = F.normalize(text_query, p=2, dim=-1)
            stats = AttentionStats(
                gate_mean=gate.mean().item(),
                gate_std=gate.std().item(),
                gate_min=gate.min().item(),
                gate_max=gate.max().item(),
                attn_entropy=-(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean().item(),
                attn_diag=attn_weights.diag().mean().item(),
                feat_delta=(refined - text_norm).norm(dim=-1).mean().item(),
                feat_sim=(refined * text_norm).sum(dim=-1).mean().item(),
            )
        return refined, stats
