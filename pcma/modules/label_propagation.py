"""Graph-based label propagation with optional multi-prototype splitting.

The propagator builds a symmetric nearest-neighbour graph over class
anchors, sampled multi-prototypes (if any), and unlabeled target features,
then runs a normalised diffusion solved with conjugate gradient. Confident
samples are aggregated into per-class visual anchors that feed back into
the cross-domain attention block.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse import linalg as s_linalg
from sklearn.cluster import KMeans


class GraphLabelPropagation(nn.Module):
    """Diffuse pseudo-labels over a kNN graph built from target features.

    Args:
        class_anchors: ``[C, D]`` initial per-class anchors (e.g. CLIP text
            features or zero-shot weights).
        dataset_size: Number of unlabeled target samples; pre-allocates
            pseudo-label storage.
        num_neighbors: ``k`` for the kNN graph (default 10).
        alpha: Diffusion coefficient in ``(I - alpha W)^-1`` (default 0.99).
        cut_dim: Number of leading SVD components removed from the
            projection (the top component carries domain energy and is
            discarded; the next ``cut_dim - 1`` components are retained).
        topk_per_class: Cap on the number of high-confidence samples used
            when refreshing each class anchor.
        proto_momentum: Momentum for the running update of class anchors.
        multi_proto_classes: Optional mapping ``{class_id: K}`` that
            requests ``K`` sub-prototypes for the given class (via KMeans).
    """

    def __init__(
        self,
        class_anchors: torch.Tensor,
        dataset_size: int,
        num_neighbors: int = 10,
        alpha: float = 0.99,
        cut_dim: int = 512,
        topk_per_class: int = 50,
        proto_momentum: float = 0.9,
        multi_proto_classes: Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.class_anchors = class_anchors
        self.device = class_anchors.device
        self.num_classes = class_anchors.size(0)
        self.feat_dim = class_anchors.size(1)
        self.num_neighbors = num_neighbors
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.cut_dim = cut_dim
        self.topk_per_class = topk_per_class
        self.proto_momentum = proto_momentum
        self.multi_proto_classes = dict(multi_proto_classes or {})

        self.feature_buffer = []
        self.index_buffer = []
        self.label_buffer: Dict[int, int] = {}
        self.pseudo_labels: Dict[int, int] = {i: 0 for i in range(dataset_size)}
        self.confidence: Dict[int, float] = {i: 0.0 for i in range(dataset_size)}

        self.visual_anchors = class_anchors.clone().to(self.device)
        self.anchor_valid_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)

        # Per-class sub-prototypes; default to the single class anchor.
        self.sub_prototypes: Dict[int, torch.Tensor] = {
            c: class_anchors[c : c + 1, :].clone().to(self.device)
            for c in range(self.num_classes)
        }

        self.centroids: Optional[torch.Tensor] = None
        self.projection_matrix: Optional[nn.Parameter] = None
        self.update_projection(class_anchors.t())
        self.update_centroids(class_anchors)

    def forward(self, features: torch.Tensor, idx: torch.Tensor, label: torch.Tensor) -> None:
        """Stage a mini-batch of features for the next propagation pass."""
        idx_list = list(idx.cpu().numpy())
        label_list = list(label.cpu().numpy())
        self.feature_buffer.append(features.detach())
        self.index_buffer.extend(idx_list)
        for i, sample_idx in enumerate(idx_list):
            self.label_buffer[sample_idx] = label_list[i]

    def update_projection(self, class_weight: Optional[torch.Tensor] = None) -> None:
        if class_weight is None:
            class_weight = self.centroids.t()
        u, _, _ = torch.svd(class_weight.to(torch.float32))
        proj = u[:, 1 : self.cut_dim] @ u[:, 1 : self.cut_dim].t()
        self.projection_matrix = nn.Parameter(proj.to(torch.float32), requires_grad=False)

    def update_centroids(self, centroids: torch.Tensor) -> None:
        self.centroids = centroids

    def update_visual_anchors(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        features: torch.Tensor,
        labeled_count: int,
    ) -> None:
        """Refresh class anchors (and sub-prototypes) from high-confidence samples."""
        if prediction.shape[0] <= labeled_count:
            return
        data_pred = prediction[labeled_count:]
        data_conf = confidence[labeled_count:]
        data_feat = features[labeled_count:]

        for class_id in range(self.num_classes):
            class_mask = data_pred == class_id
            if not np.any(class_mask):
                continue
            class_indices = np.where(class_mask)[0]
            class_conf = data_conf[class_indices]
            if class_indices.shape[0] > self.topk_per_class:
                top = np.argsort(-class_conf)[: self.topk_per_class]
                class_indices = class_indices[top]
            indices_tensor = torch.from_numpy(class_indices).long()
            selected = data_feat[indices_tensor]

            target_k = self.multi_proto_classes.get(class_id, 1)
            if target_k > 1 and selected.shape[0] >= target_k * 2:
                feat_np = selected.cpu().numpy()
                kmeans = KMeans(n_clusters=target_k, n_init=10, random_state=42).fit(feat_np)
                centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
                self.sub_prototypes[class_id] = centers
                new_proto = centers.mean(dim=0)
            else:
                new_proto = selected.mean(dim=0).to(self.device)
                self.sub_prototypes[class_id] = new_proto.unsqueeze(0)

            old = self.visual_anchors[class_id]
            m = self.proto_momentum
            self.visual_anchors[class_id] = m * old + (1.0 - m) * new_proto
            self.anchor_valid_mask[class_id] = True

    def get_pseudo_label(self, idx: torch.Tensor) -> Tuple[list, list]:
        idx_list = list(idx.cpu().numpy())
        labels = [self.pseudo_labels[i] for i in idx_list]
        scores = [self.confidence[i] for i in idx_list]
        return labels, scores

    def solve_diffusion(
        self,
        sources: np.ndarray,
        graph: csr_matrix,
        alpha: float = 0.99,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        operator = eye(graph.shape[0]) - alpha * graph
        out_columns = []
        for i in range(sources.shape[0]):
            f, _ = s_linalg.cg(operator, sources[i, :], rtol=tol, atol=0.0, maxiter=max_iter)
            out_columns.append(f.reshape(-1, 1))
        out_sims = np.concatenate(out_columns, axis=1)
        ranks = np.argsort(-out_sims, axis=0)
        return ranks, out_sims

    def propagate(self, clear_cache: bool = True, cluster_centroids: bool = False):
        """Run one round of label propagation over the buffered features."""
        stacked = torch.cat(self.feature_buffer, dim=0)

        need_update_projection = False
        if self.centroids.shape[1] != stacked.shape[1]:
            if self.centroids.shape[0] == stacked.shape[1]:
                self.centroids = self.centroids.t()
                need_update_projection = True

        # Order: class anchors -> sampled visual anchors (if valid) -> data points.
        points = [self.centroids]
        num_proto = 0
        valid_idx: Optional[torch.Tensor] = None
        if self.anchor_valid_mask.any():
            valid_idx = torch.nonzero(self.anchor_valid_mask, as_tuple=False).view(-1)
            points.append(self.visual_anchors[valid_idx].to(self.device))
            num_proto = valid_idx.numel()
        points.append(stacked.to(self.device))
        all_points = torch.cat(points, dim=0).detach()
        all_points_original = all_points.cpu().to(torch.float32)

        if need_update_projection:
            self.update_projection(self.centroids.t())

        projected = all_points.to(torch.float32) @ self.projection_matrix.to(torch.float32)
        projected = F.normalize(projected, p=2, dim=-1).to(torch.float32)
        n_nodes = projected.size(0)

        feats_np = projected.cpu().numpy()
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(feats_np.shape[1]))
        index.add(feats_np)
        distances, neighbors = index.search(feats_np, self.num_neighbors + 1)

        rows = np.arange(n_nodes).repeat(self.num_neighbors)
        cols = neighbors[:, 1:].flatten()
        data = distances[:, 1:].flatten()
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        graph = (graph + graph.T) / 2
        row_sum = np.array(graph.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1.0
        d_inv_sqrt = diags(1.0 / np.sqrt(row_sum))
        normalised = d_inv_sqrt @ graph @ d_inv_sqrt

        sources = np.zeros((self.num_classes, n_nodes))
        for c in range(self.num_classes):
            if c < n_nodes:
                sources[c, c] = 1.0

        if num_proto > 0 and valid_idx is not None:
            valid_idx_np = valid_idx.cpu().numpy()
            for j, class_id in enumerate(valid_idx_np):
                node = self.num_classes + j
                if node < n_nodes:
                    sources[int(class_id), node] = 1.0

        _, scores = self.solve_diffusion(sources, normalised, self.alpha)
        prediction = np.argmax(scores, axis=1)

        eps = 1e-9
        row_sums = scores.sum(axis=1, keepdims=True)
        probs = scores / (row_sums + eps)
        entropy = -probs * np.log(np.maximum(probs, eps))
        confidence = 1 - entropy.sum(axis=1) / np.log(self.num_classes)

        labeled_count = self.num_classes + num_proto
        self.update_visual_anchors(prediction, confidence, all_points_original, labeled_count)

        new_centroids = None
        if cluster_centroids:
            new_centroids = self.centroids.clone()
            for c in range(self.num_classes):
                mask = prediction == c
                if mask.sum() > 0:
                    centroid = torch.mean(all_points_original[mask], dim=0)
                    centroid = F.normalize(centroid.to(torch.float32), p=2, dim=0)
                    new_centroids[c, :] = centroid

        prediction = prediction[labeled_count:]
        confidence = confidence[labeled_count:]
        for i, sample_idx in enumerate(self.index_buffer):
            self.pseudo_labels[sample_idx] = int(prediction[i])
            self.confidence[sample_idx] = float(confidence[i])

        pseudo_acc = float(
            np.mean([self.pseudo_labels[i] == self.label_buffer[i] for i in self.index_buffer])
        )

        if clear_cache:
            self.feature_buffer = []
            self.index_buffer = []
            self.label_buffer = {}

        if cluster_centroids:
            return pseudo_acc, new_centroids
        return pseudo_acc
