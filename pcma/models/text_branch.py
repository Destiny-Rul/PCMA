"""ViL-T (text branch) for target adaptation.

The text branch fine-tunes only the LayerNorm parameters of the frozen
CLIP text encoder, optionally refines its zero-shot weights with the
:class:`CrossDomainAttention` block, and supplies the source-anchored
:class:`CrossDomainContrastiveLoss` for target image features.
"""

from __future__ import annotations

from typing import List, Optional

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

from ..losses.cdc import CrossDomainContrastiveLoss
from ..modules.cross_attention import CrossDomainAttention
from ..utils.prompts import CLIP_TEMPLATES


class TextBranch(nn.Module):
    """Target-domain text branch.

    Args:
        class_names: Target-domain class names.
        architecture: CLIP architecture name.
        device: Target device.
        use_ca: Enable :class:`CrossDomainAttention`.
        scale_factor: Sharpening factor of the CA softmax.
        ca_init_gate_bias: Initial bias of the CA gate.
        use_cdc: Enable :class:`CrossDomainContrastiveLoss`.
        cdc_weight: Weight applied to the CDC loss in the total objective.
        cdc_temperature: Temperature for the CDC similarity logits.
        ckpt_dir: Directory used by CLIP to cache downloaded weights.
    """

    def __init__(
        self,
        class_names: List[str],
        architecture: str = "ViT-B/16",
        device: Optional[torch.device] = None,
        use_ca: bool = True,
        scale_factor: float = 4.0,
        ca_init_gate_bias: float = -2.0,
        use_cdc: bool = True,
        cdc_weight: float = 0.5,
        cdc_temperature: float = 0.07,
        ckpt_dir: str = "./checkpoints",
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ca = use_ca
        self.scale_factor = scale_factor
        self.use_cdc = use_cdc
        self.cdc_weight = cdc_weight

        self.source_context_vectors: Optional[torch.Tensor] = None
        self.source_text_features: Optional[torch.Tensor] = None

        self.base_model, self.preprocess = clip.load(
            download_root=ckpt_dir, name=architecture, device=self.device
        )
        self.embed_dim = self.base_model.ln_final.weight.shape[0]

        self.trainable_params: list = []
        self.ca_params: list = []
        self.setup_trainable_params()

        self.ca_adapter: Optional[CrossDomainAttention] = None
        if self.use_ca:
            self.ca_adapter = CrossDomainAttention(
                embed_dim=self.embed_dim,
                init_gate_bias=ca_init_gate_bias,
                scale_factor=self.scale_factor,
            ).to(self.device)
            self.ca_params = list(self.ca_adapter.parameters())

        self.cdc_loss: Optional[CrossDomainContrastiveLoss] = None
        if self.use_cdc:
            self.cdc_loss = CrossDomainContrastiveLoss(temperature=cdc_temperature)

        self.visual_anchors: Optional[torch.Tensor] = None

    def setup_trainable_params(self) -> None:
        """Freeze CLIP and unlock LayerNorm weights of the text transformer."""
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        for module in self.base_model.transformer.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
                self.trainable_params.append(module.weight)
                self.trainable_params.append(module.bias)

        self.base_model.ln_final.requires_grad_(True)
        self.trainable_params.append(self.base_model.ln_final.weight)
        self.trainable_params.append(self.base_model.ln_final.bias)

    def load_source_weights(self, ckpt_path: str) -> dict:
        """Load context vectors, CA weights, and source text features."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        loaded = []

        if "context_vectors" in ckpt:
            self.source_context_vectors = ckpt["context_vectors"].to(self.device)
            loaded.append("context_vectors")

        if self.use_ca and self.ca_adapter is not None and "ca_state_dict" in ckpt:
            self.ca_adapter.load_state_dict(ckpt["ca_state_dict"])
            loaded.append("CA")

        if self.source_context_vectors is not None:
            with torch.no_grad():
                self.source_text_features = self._encode_with_source_prompt()
                if self.source_text_features is not None:
                    loaded.append("source_text_features")

        if loaded:
            print(f"  Loaded: {', '.join(loaded)}")
        return ckpt

    def set_visual_anchors(self, anchors: torch.Tensor) -> None:
        """Provide the per-class visual anchors consumed by CA."""
        self.visual_anchors = anchors.to(self.device)

    def _encode_with_source_prompt(self) -> Optional[torch.Tensor]:
        if self.source_context_vectors is None or len(self.class_names) == 0:
            return None

        token_embedding = self.base_model.token_embedding
        features = []

        for classname in self.class_names:
            text = f"a photo of a {classname}"
            tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                x = token_embedding(tokens).type(self.base_model.dtype)
                n_ctx = self.source_context_vectors.shape[0]
                prefix = self.source_context_vectors.unsqueeze(0).type(x.dtype)
                x = torch.cat([x[:, :1], prefix, x[:, 1 + n_ctx :]], dim=1)
                x = x + self.base_model.positional_embedding.type(x.dtype)
                x = x.permute(1, 0, 2)
                x = self.base_model.transformer(x)
                x = x.permute(1, 0, 2)
                x = self.base_model.ln_final(x).type(x.dtype)
                eot_idx = tokens.argmax(dim=-1)
                feat = x[0, eot_idx[0]] @ self.base_model.text_projection
                features.append(F.normalize(feat, p=2, dim=-1))

        return torch.stack(features, dim=0)

    def encode_text_zeroshot(self, full_templates: bool = False) -> torch.Tensor:
        """Zero-shot CLIP text features (no CA, no source prompt)."""
        if full_templates:
            outputs = []
            for classname in self.class_names:
                prompts = [t.format(classname) for t in CLIP_TEMPLATES]
                tokens = clip.tokenize(prompts).to(self.device)
                embeddings = self.base_model.encode_text(tokens)
                embeddings = F.normalize(embeddings, p=2, dim=-1).mean(dim=0)
                outputs.append(F.normalize(embeddings, p=2, dim=-1))
            return torch.stack(outputs, dim=0)

        prompts = [f"a photo of a {name}" for name in self.class_names]
        tokens = clip.tokenize(prompts).to(self.device)
        embeddings = self.base_model.encode_text(tokens)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_text(self, full_templates: bool = False, apply_ca: bool = True) -> torch.Tensor:
        """Text features with optional CA refinement."""
        text_features = self.encode_text_zeroshot(full_templates=full_templates)
        if (
            apply_ca
            and self.use_ca
            and self.ca_adapter is not None
            and self.visual_anchors is not None
        ):
            original_dtype = text_features.dtype
            text_features = self.ca_adapter(text_features, self.visual_anchors)
            text_features = text_features.to(dtype=original_dtype)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encode_image(image)
        return F.normalize(features, p=2, dim=1)

    def forward(self, image: torch.Tensor):
        image_features = self.encode_image(image)
        if self.training:
            text_features = self.encode_text(full_templates=False)
        else:
            with torch.no_grad():
                text_features = self.encode_text(full_templates=False)
        logits = 100.0 * image_features @ text_features.t()
        return logits, image_features

    def compute_cdc_loss(self, image_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.use_cdc or self.cdc_loss is None or self.source_text_features is None:
            return torch.tensor(0.0, device=self.device)
        return self.cdc_loss(image_features, self.source_text_features, labels)
