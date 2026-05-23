"""Source-domain training entry point for Office-Home.

Example:
    python -m tools.train_source --data-root /path/to/office_home --source art
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcma.data.office_home import (
    OFFICE_HOME_DOMAINS,
    build_office_home,
    get_class_prompts,
)
from pcma.models.source_model import SourceModel, compute_anchor_loss
from pcma.utils.metrics import accuracy
from pcma.utils.scheduler import WarmupCosineScheduler
from pcma.utils.seeds import set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCMA source-domain training (Office-Home)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing the Office-Home domain folders")
    parser.add_argument("--source", type=str, required=True, choices=OFFICE_HOME_DOMAINS,
                        help="Source domain")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/source")
    parser.add_argument("--architecture", type=str, default="ViT-B/16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="Prompt learning rate")
    parser.add_argument("--ca-lr", type=float, default=1e-4, help="CA learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--ca-init-gate-bias", type=float, default=-2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="Optional cosine-anchor regulariser keeping prompt close to CLIP")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: SourceModel, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc="Eval", leave=False)
    for images, _, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).view(-1)
        logits, _, _ = model(images, labels=None, update_prototypes=False)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})
    return 100.0 * correct / total, total_loss / len(loader)


def train(args: argparse.Namespace) -> float:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = get_class_prompts()
    print(f"Source domain: {args.source} | num_classes: {len(class_names)}")

    model = SourceModel(
        class_names=class_names,
        architecture=args.architecture,
        n_ctx=args.n_ctx,
        use_ca=True,
        use_prototype_bank=True,
        prototype_momentum=args.prototype_momentum,
        ca_init_gate_bias=args.ca_init_gate_bias,
        device=device,
    ).to(device)

    dataset = build_office_home(domain=args.source, root_dir=args.data_root, transform=model.preprocess)
    print(f"Number of source samples: {len(dataset)}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=worker_init_fn, generator=generator,
    )
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )

    param_groups = [{"params": model.prompt_learner.parameters(), "lr": args.lr}]
    if model.ca_adapter is not None:
        param_groups.append({"params": model.ca_adapter.parameters(), "lr": args.ca_lr})
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        warmup_lr=args.warmup_lr,
        base_lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_eval, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_anchor = 0.0
        correct, total = 0, 0
        lr = scheduler.step()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, _, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).view(-1)
            logits, _, text_features = model(images, labels=labels)
            loss = criterion(logits, labels)
            if args.anchor_weight > 0:
                anchor = compute_anchor_loss(text_features, model.get_zeroshot_text_features())
                loss = loss + args.anchor_weight * anchor
                running_anchor += anchor.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = 100.0 * correct / total
        anchor_str = f", Anchor={running_anchor / len(train_loader):.4f}" if args.anchor_weight > 0 else ""
        print(f"[Epoch {epoch:2d}] LR={lr:.2e} | Loss={running_loss / len(train_loader):.4f}{anchor_str} | Train Acc={train_acc:.2f}%")

        if epoch % 5 == 0 or epoch == args.epochs:
            eval_acc, _ = evaluate(model, eval_loader, device)
            if eval_acc > best_eval:
                best_eval = eval_acc
                best_epoch = epoch
            print(f"             Eval Acc={eval_acc:.2f}% (best={best_eval:.2f}% @ epoch {best_epoch})")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"source_{args.source}.pth")
    payload = model.export_checkpoint()
    payload.update({
        "source_domain": args.source,
        "best_eval_acc": best_eval,
        "best_epoch": best_epoch,
        "config": vars(args),
    })
    torch.save(payload, save_path)
    print(f"Saved checkpoint to {save_path}")
    print(f"Best eval accuracy: {best_eval:.2f}% (epoch {best_epoch})")
    return best_eval


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
