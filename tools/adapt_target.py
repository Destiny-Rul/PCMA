"""Target-domain adaptation entry point for Office-Home.

Example:
    python -m tools.adapt_target \
        --data-root /path/to/office_home \
        --source art --target clipart \
        --source-ckpt ./checkpoints/source/source_art.pth
"""

from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcma.adaptive import compute_boundary_stress
from pcma.data.office_home import (
    OFFICE_HOME_DOMAINS,
    build_office_home,
    get_class_prompts,
)
from pcma.models.text_branch import TextBranch
from pcma.models.vision_branch import VisionBranch
from pcma.modules.label_propagation import GraphLabelPropagation
from pcma.utils.metrics import accuracy
from pcma.utils.seeds import set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCMA target adaptation (Office-Home)")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, choices=OFFICE_HOME_DOMAINS)
    parser.add_argument("--target", type=str, required=True, choices=OFFICE_HOME_DOMAINS)
    parser.add_argument("--source-ckpt", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="ViT-B/16")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr-text", type=float, default=1e-2)
    parser.add_argument("--lr-vision", type=float, default=5e-3)
    parser.add_argument("--ca-lr", type=float, default=1e-4)
    parser.add_argument("--scale-factor", type=float, default=4.0)

    parser.add_argument("--cdc-weight", type=float, default=0.5,
                        help="Weight of the cross-domain contrastive loss")

    parser.add_argument("--knn", type=int, default=10, help="Neighbours for the LP graph")
    parser.add_argument("--alpha", type=float, default=0.99, help="Diffusion coefficient")
    parser.add_argument("--cut-dim", type=int, default=512)
    parser.add_argument("--proto-topk", type=int, default=50)
    parser.add_argument("--proto-momentum", type=float, default=0.9)

    parser.add_argument("--use-dynamic-fusion", action="store_true",
                        help="Confidence-weighted fusion of branches at evaluation time")
    parser.add_argument("--fusion-temperature", type=float, default=1.0)

    parser.add_argument("--use-multi-proto", action="store_true",
                        help="Enable adaptive K sub-prototypes")
    parser.add_argument("--multi-proto-k-max", type=int, default=3)
    parser.add_argument("--multi-proto-threshold", type=float, default=0.5)
    return parser.parse_args()


def evaluation_phase(
    text_branch: TextBranch,
    vision_branch: VisionBranch,
    test_loader: DataLoader,
    text_lp: GraphLabelPropagation,
    vision_lp: GraphLabelPropagation,
    device: torch.device,
    use_dynamic_fusion: bool,
    fusion_temperature: float,
):
    text_branch.eval()
    vision_branch.eval()
    top1_text = top1_vision = top1_fused = 0.0
    n_total = 0

    with torch.no_grad():
        for images, idx, labels in tqdm(test_loader, desc="Eval", leave=False):
            images = images.to(device)
            labels = labels.to(device).view(-1)

            text_logits, text_features = text_branch(images)
            vision_logits, vision_features = vision_branch(images)

            text_lp(text_features, idx, labels)
            vision_lp(vision_features, idx, labels)

            if use_dynamic_fusion:
                t_conf = F.softmax(text_logits / fusion_temperature, dim=-1).max(dim=-1)[0]
                v_conf = F.softmax(vision_logits / fusion_temperature, dim=-1).max(dim=-1)[0]
                weights = F.softmax(torch.stack([t_conf, v_conf], dim=-1), dim=-1)
                fused = weights[:, 0:1] * text_logits + weights[:, 1:2] * vision_logits
            else:
                fused = 0.5 * (text_logits + vision_logits)

            top1_text += accuracy(text_logits, labels)[0]
            top1_vision += accuracy(vision_logits, labels)[0]
            top1_fused += accuracy(fused, labels)[0]
            n_total += labels.size(0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    text_lp.propagate(clear_cache=True)
    _, new_centroids = vision_lp.propagate(clear_cache=True, cluster_centroids=True)
    vision_branch.update_classifier_weights(new_centroids.t())

    return {
        "acc_text": top1_text / n_total,
        "acc_vision": top1_vision / n_total,
        "acc_fused": top1_fused / n_total,
    }


def training_phase(
    text_branch: TextBranch,
    vision_branch: VisionBranch,
    train_loader: DataLoader,
    text_lp: GraphLabelPropagation,
    vision_lp: GraphLabelPropagation,
    text_optimizer: torch.optim.Optimizer,
    vision_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    text_branch.train()
    vision_branch.train()
    total_t = total_v = total_cdc = 0.0
    agree_count = sample_count = update_count = 0

    for images, idx, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device).view(-1)

        text_logits, text_features = text_branch(images)
        vision_logits, _ = vision_branch(images)

        text_pseudo, _ = text_lp.get_pseudo_label(idx)
        vision_pseudo, _ = vision_lp.get_pseudo_label(idx)
        text_pseudo = torch.LongTensor(text_pseudo).to(device)
        vision_pseudo = torch.LongTensor(vision_pseudo).to(device)

        consensus_mask = vision_pseudo == text_pseudo
        agree_count += consensus_mask.sum().item()
        sample_count += labels.numel()

        if consensus_mask.sum() == 0:
            continue

        confident_labels = text_pseudo[consensus_mask]
        confident_t_logits = text_logits[consensus_mask]
        confident_v_logits = vision_logits[consensus_mask]
        confident_t_features = text_features[consensus_mask]

        t_loss = criterion(confident_t_logits, confident_labels).mean()
        v_loss = criterion(confident_v_logits, confident_labels).mean()

        if text_branch.use_cdc:
            cdc = text_branch.compute_cdc_loss(confident_t_features, confident_labels)
            t_loss = t_loss + text_branch.cdc_weight * cdc
            total_cdc += cdc.item()

        total_t += t_loss.item()
        total_v += v_loss.item()
        text_optimizer.zero_grad()
        vision_optimizer.zero_grad()
        t_loss.backward()
        v_loss.backward()
        text_optimizer.step()
        vision_optimizer.step()
        update_count += 1

    denom = max(update_count, 1)
    return {
        "t_loss": total_t / denom,
        "v_loss": total_v / denom,
        "cdc_loss": total_cdc / denom,
        "consensus_rate": agree_count / max(sample_count, 1),
    }


def adapt(args: argparse.Namespace) -> float:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = get_class_prompts()
    print(f"{args.source.upper()} -> {args.target.upper()} | num_classes={len(class_names)}")

    text_branch = TextBranch(
        class_names=class_names,
        architecture=args.architecture,
        device=device,
        use_ca=True,
        scale_factor=args.scale_factor,
        use_cdc=True,
        cdc_weight=args.cdc_weight,
    ).to(device)

    if not os.path.exists(args.source_ckpt):
        raise FileNotFoundError(f"Source checkpoint not found: {args.source_ckpt}")
    print(f"Loading source weights: {args.source_ckpt}")
    text_branch.load_source_weights(args.source_ckpt)

    vision_branch = VisionBranch(class_names=class_names, architecture=args.architecture).to(device)

    target_dataset = build_office_home(domain=args.target, root_dir=args.data_root, transform=text_branch.preprocess)
    generator = torch.Generator().manual_seed(args.seed)
    test_loader = DataLoader(
        target_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, worker_init_fn=worker_init_fn, generator=generator,
    )
    train_loader = DataLoader(
        target_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=worker_init_fn, generator=generator,
    )

    multi_proto_classes = None
    if args.use_multi_proto:
        with torch.no_grad():
            text_features_for_k = text_branch.encode_text_zeroshot(full_templates=False)
        multi_proto_classes = compute_boundary_stress(
            text_features_for_k,
            top_n=5,
            open_threshold=args.multi_proto_threshold,
            k_max=args.multi_proto_k_max,
        )
        print(f"Adaptive K mapping: {len(multi_proto_classes)} classes assigned k>1")

    vision_lp = GraphLabelPropagation(
        class_anchors=vision_branch.classifier_weights.t(),
        dataset_size=len(target_dataset),
        num_neighbors=args.knn,
        alpha=args.alpha,
        cut_dim=args.cut_dim,
        topk_per_class=args.proto_topk,
        proto_momentum=args.proto_momentum,
        multi_proto_classes=multi_proto_classes,
    )
    with torch.no_grad():
        text_zeroshot = text_branch.encode_text_zeroshot(full_templates=False)
    text_lp = GraphLabelPropagation(
        class_anchors=text_zeroshot,
        dataset_size=len(target_dataset),
        num_neighbors=args.knn,
        alpha=args.alpha,
        cut_dim=args.cut_dim,
        topk_per_class=args.proto_topk,
        proto_momentum=args.proto_momentum,
        multi_proto_classes=multi_proto_classes,
    )

    text_param_groups = [{"params": text_branch.trainable_params, "lr": args.lr_text}]
    if text_branch.ca_params:
        text_param_groups.append({"params": text_branch.ca_params, "lr": args.ca_lr})
    text_optimizer = torch.optim.SGD(text_param_groups, momentum=0.9, weight_decay=1e-4)
    vision_optimizer = torch.optim.SGD(
        [{"params": vision_branch.trainable_params, "lr": args.lr_vision}],
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        print(f"[Epoch {epoch:2d}/{args.epochs}]")
        eval_metrics = evaluation_phase(
            text_branch, vision_branch, test_loader, text_lp, vision_lp,
            device, args.use_dynamic_fusion, args.fusion_temperature,
        )

        if text_branch.ca_adapter is not None:
            anchors = F.normalize(vision_lp.visual_anchors.clone(), p=2, dim=-1)
            text_branch.set_visual_anchors(anchors)

        train_metrics = training_phase(
            text_branch, vision_branch, train_loader, text_lp, vision_lp,
            text_optimizer, vision_optimizer, criterion, device,
        )

        if eval_metrics["acc_fused"] > best_acc:
            best_acc = eval_metrics["acc_fused"]
            best_epoch = epoch
            best_marker = " (*)"
        else:
            best_marker = ""

        print(
            f"  Loss T={train_metrics['t_loss']:.4f}, V={train_metrics['v_loss']:.4f}, "
            f"CDC={train_metrics['cdc_loss']:.4f} | "
            f"Consensus={train_metrics['consensus_rate'] * 100:.1f}%"
        )
        print(
            f"  Acc T={eval_metrics['acc_text'] * 100:.2f}%, "
            f"V={eval_metrics['acc_vision'] * 100:.2f}%, "
            f"F={eval_metrics['acc_fused'] * 100:.2f}%{best_marker}"
        )

        with torch.no_grad():
            class_weight = text_branch.encode_text_zeroshot(full_templates=True)
            text_lp.update_projection(class_weight)
            text_lp.update_centroids(class_weight.t())

    print(f"Best fused accuracy: {best_acc * 100:.2f}% (epoch {best_epoch})")
    return best_acc


def main() -> None:
    args = parse_args()
    adapt(args)


if __name__ == "__main__":
    main()
