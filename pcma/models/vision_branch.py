"""ViL-V (vision branch) for target adaptation.

The vision branch fine-tunes the LayerNorm parameters of the CLIP visual
encoder while keeping a frozen template-averaged classification head that
is later refreshed by graph-clustered centroids.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

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

from ..utils.prompts import CLIP_TEMPLATES


class VisionBranch(nn.Module):
    """Target-domain vision branch with template-averaged classifier weights.

    Args:
        class_names: Target-domain class names.
        architecture: CLIP architecture name.
        templates: Optional iterable of prompt templates. Defaults to
            :data:`pcma.utils.prompts.CLIP_TEMPLATES`.
        ckpt_dir: Directory used by CLIP to cache downloaded weights.
    """

    def __init__(
        self,
        class_names: List[str],
        architecture: str = "ViT-B/16",
        templates: Optional[Sequence[str]] = None,
        ckpt_dir: str = "./checkpoints",
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self.templates = list(templates) if templates is not None else list(CLIP_TEMPLATES)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model, self.preprocess = clip.load(
            download_root=ckpt_dir, name=architecture, device=device
        )

        with torch.no_grad():
            self.classifier_weights = self._template_averaged_weights(self.class_names).detach()

        self.trainable_params: list = []
        self.setup_trainable_params()

    def setup_trainable_params(self) -> None:
        """Freeze CLIP and unlock LayerNorm/BatchNorm of the visual encoder."""
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        for module in self.base_model.visual.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                module.requires_grad_(True)
                self.trainable_params.append(module.weight)
                self.trainable_params.append(module.bias)

    def _template_averaged_weights(self, class_names: Sequence[str]) -> torch.Tensor:
        device = self.base_model.visual.conv1.weight.device
        columns = []
        for name in class_names:
            if isinstance(name, list):
                prompts = [t.format(c) for t in self.templates for c in name]
            else:
                prompts = [t.format(name) for t in self.templates]
            tokens = clip.tokenize(prompts).to(device)
            embeddings = self.base_model.encode_text(tokens)
            embeddings = F.normalize(embeddings, p=2, dim=-1).mean(dim=0)
            columns.append(F.normalize(embeddings, p=2, dim=-1))
        return torch.stack(columns, dim=1).to(device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encode_image(image)
        return F.normalize(features, p=2, dim=1)

    def forward(self, image: torch.Tensor):
        image_features = self.encode_image(image)
        weights = F.normalize(self.classifier_weights, p=2, dim=0)
        weights = weights.to(dtype=image_features.dtype, device=image_features.device)
        logits = 100.0 * image_features @ weights
        return logits, image_features

    def update_classifier_weights(self, weights: torch.Tensor) -> None:
        """Replace the classifier weights with new (e.g. graph-clustered) prototypes."""
        self.classifier_weights = weights.to(dtype=self.classifier_weights.dtype, device=self.classifier_weights.device)
