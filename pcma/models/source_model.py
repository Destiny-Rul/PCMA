"""Source-domain model: CLIP backbone with a learnable context prompt.

The source stage trains a small set of context vectors prepended to the
class names (CoOp-style) together with an optional :class:`CrossDomainAttention`
that consumes class-wise visual prototypes maintained by a
:class:`MomentumPrototypeBank`. The resulting checkpoint provides three
artifacts for the target stage: the learned context vectors, the CA
weights, and the bank of source visual anchors.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from clip import clip
except ImportError as exc:  # pragma: no cover - import-time message
    raise ImportError(
        "PCMA requires the OpenAI CLIP package. Install with\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from exc

from ..modules.cross_attention import CrossDomainAttention
from ..modules.prototype_bank import MomentumPrototypeBank


class ContextPromptLearner(nn.Module):
    """Learnable context vectors shared across all classes.

    Each forward pass assembles ``[SOS] + [context] + [class tokens] + [.]``
    and pushes it through the frozen CLIP text transformer, returning the
    L2-normalised class prototypes.

    Args:
        class_names: List of textual class names.
        clip_model: A loaded ``clip`` model whose text encoder will be used.
        n_ctx: Number of learnable context tokens (default 16).
        init_text: Text used to initialise the leading context tokens
            (defaults to ``"a photo of a"``).
        device: Target device.
    """

    def __init__(
        self,
        class_names: List[str],
        clip_model,
        n_ctx: int = 16,
        init_text: str = "a photo of a",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_cls = len(class_names)
        dtype = clip_model.dtype
        embed_dim = clip_model.ln_final.weight.shape[0]

        tokenised = clip.tokenize([init_text]).to(device)
        with torch.no_grad():
            init_embedding = clip_model.token_embedding(tokenised).type(dtype)
        prefix_tokens = init_embedding[0, 1:5, :]  # the four content tokens

        if n_ctx <= 4:
            context_vectors = prefix_tokens[:n_ctx, :].clone()
        else:
            extra = torch.empty(n_ctx - 4, embed_dim, dtype=dtype, device=device)
            nn.init.normal_(extra, std=0.02)
            context_vectors = torch.cat([prefix_tokens, extra], dim=0)
        self.context_vectors = nn.Parameter(context_vectors)

        prompts = [f"{name}." for name in class_names]
        tokenised_prompts = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            token_embedding = clip_model.token_embedding(tokenised_prompts).type(dtype)

        self.register_buffer("tokenised_prompts", tokenised_prompts)
        self.register_buffer("token_embedding", token_embedding)
        self.class_token_lens = tokenised_prompts.argmax(dim=-1)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim

    def forward(self, clip_model) -> torch.Tensor:
        dtype = self.context_vectors.dtype
        embedding = self.token_embedding.clone()
        prefix = self.context_vectors.unsqueeze(0).expand(self.n_cls, -1, -1)

        new_embedding = torch.cat(
            [
                embedding[:, :1, :],
                prefix,
                embedding[:, 1 : 77 - self.n_ctx, :],
            ],
            dim=1,
        )

        x = new_embedding + clip_model.positional_embedding.to(dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x)

        eos_indices = torch.clamp(1 + self.n_ctx + self.class_token_lens - 1, max=76)
        text_features = x[torch.arange(self.n_cls, device=x.device), eos_indices]
        text_features = text_features @ clip_model.text_projection
        return F.normalize(text_features, p=2, dim=-1)


class SourceModel(nn.Module):
    """Frozen CLIP backbone with learnable prompt, optional CA, and momentum bank.

    Args:
        class_names: Source-domain class names.
        architecture: CLIP architecture name (default ``ViT-B/16``).
        n_ctx: Number of learnable context tokens.
        use_ca: Enable :class:`CrossDomainAttention`.
        use_prototype_bank: Enable :class:`MomentumPrototypeBank`.
        prototype_momentum: Momentum for the prototype bank.
        ca_init_gate_bias: Initial bias of the CA gate.
        ckpt_dir: Directory used by CLIP to cache downloaded weights.
        device: Target device.
    """

    def __init__(
        self,
        class_names: List[str],
        architecture: str = "ViT-B/16",
        n_ctx: int = 16,
        use_ca: bool = True,
        use_prototype_bank: bool = True,
        prototype_momentum: float = 0.9,
        ca_init_gate_bias: float = -2.0,
        ckpt_dir: str = "./checkpoints",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.n_ctx = n_ctx
        self.use_ca = use_ca
        self.use_prototype_bank = use_prototype_bank

        clip_model, self.preprocess = clip.load(
            download_root=ckpt_dir, name=architecture, device=self.device
        )
        clip_model = clip_model.float()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.embed_dim = clip_model.ln_final.weight.shape[0]

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.prompt_learner = ContextPromptLearner(
            class_names, clip_model, n_ctx=n_ctx, device=self.device
        )

        self.ca_adapter: Optional[CrossDomainAttention] = None
        if use_ca:
            self.ca_adapter = CrossDomainAttention(
                embed_dim=self.embed_dim, init_gate_bias=ca_init_gate_bias
            )

        self.prototype_bank: Optional[MomentumPrototypeBank] = None
        if use_prototype_bank:
            self.prototype_bank = MomentumPrototypeBank(
                num_classes=len(class_names),
                embed_dim=self.embed_dim,
                momentum=prototype_momentum,
                device=self.device,
            )

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        with torch.no_grad():
            self._zeroshot_text_features = self._compute_zeroshot_text_features()

    def _compute_zeroshot_text_features(self) -> torch.Tensor:
        prompts = [f"a photo of a {name}." for name in self.class_names]
        tokenised = clip.tokenize(prompts).to(self.device)
        feats = self.clip_model.encode_text(tokenised)
        return F.normalize(feats, p=2, dim=-1).detach()

    def encode_text(self, return_ca_stats: bool = False):
        text_features = self.prompt_learner(self.clip_model)
        if (
            self.use_ca
            and self.ca_adapter is not None
            and self.prototype_bank is not None
            and self.prototype_bank.get_valid_mask().all()
        ):
            prototypes = self.prototype_bank.get_prototypes()
            if return_ca_stats:
                text_features, stats = self.ca_adapter(
                    text_features, prototypes, return_stats=True
                )
                return text_features, stats
            text_features = self.ca_adapter(text_features, prototypes)
        if return_ca_stats:
            return text_features, None
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image.type(self.dtype))
        return F.normalize(features, p=2, dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_prototypes: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        if (
            self.prototype_bank is not None
            and labels is not None
            and update_prototypes
        ):
            self.prototype_bank.update(image_features, labels)

        text_features = self.encode_text()
        scale = self.logit_scale.exp()
        logits = scale * image_features @ text_features.t()
        return logits, image_features, text_features

    def get_zeroshot_text_features(self) -> torch.Tensor:
        return self._zeroshot_text_features

    def export_checkpoint(self) -> dict:
        """Bundle the trainable artefacts for the target stage."""
        payload = {
            "context_vectors": self.prompt_learner.context_vectors.data.cpu(),
            "n_ctx": self.n_ctx,
            "embed_dim": self.embed_dim,
            "class_names": list(self.class_names),
        }
        if self.ca_adapter is not None:
            payload["ca_state_dict"] = self.ca_adapter.state_dict()
        if self.prototype_bank is not None:
            payload["source_visual_anchors"] = self.prototype_bank.get_prototypes().cpu()
            payload["source_anchor_valid"] = self.prototype_bank.get_valid_mask().cpu()
        return payload


def compute_anchor_loss(
    learned_text_features: torch.Tensor,
    zeroshot_text_features: torch.Tensor,
) -> torch.Tensor:
    """Cosine-anchor penalty keeping the learned text features close to CLIP."""
    sim = (learned_text_features * zeroshot_text_features).sum(dim=-1)
    return (1.0 - sim).mean()
