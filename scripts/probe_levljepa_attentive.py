#!/usr/bin/env python
"""Distributed attentive ImageNet probe for a frozen LeVLJEPA vision encoder."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.attentive import AttentiveClassifier
from utils.eval_utils import ImageNetTrain, ImageNetVal


class NoPadDistributedSampler(Sampler[int]):
    """Shard sequential indices across ranks without padding duplicates."""

    def __init__(self, dataset_size: int, rank: int, world_size: int):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self):
        if self.rank >= self.dataset_size:
            return 0
        return (self.dataset_size - 1 - self.rank) // self.world_size + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a frozen-LeVLJEPA attentive pooling ImageNet probe."
    )
    parser.add_argument("--vision-ckpt", required=True)
    parser.add_argument("--vit", default="vit_base_patch16_224")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CACHE_DIR", "./data"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Autocast precision for the frozen vision tower.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path. Defaults under analysis_results/levljepa_attentive_probe/.",
    )
    return parser.parse_args()


def slurm_master_addr() -> str:
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        return (
            subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
            .decode()
            .splitlines()[0]
        )
    except Exception:
        return socket.gethostname()


def setup_distributed():
    if "SLURM_NTASKS" in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", slurm_master_addr())
        os.environ.setdefault("MASTER_PORT", "29581")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )

    return rank, local_rank, world_size


def device_for_rank(local_rank: int):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible = torch.cuda.device_count()
    device_index = 0 if visible == 1 else local_rank
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)


def autocast_for(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_vision_encoder(path: str, vit: str, device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
    model = timm.create_model(vit, pretrained=False, num_classes=0, dynamic_img_size=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load vision checkpoint cleanly: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def patch_tokens(vision_encoder, images, device, precision):
    with autocast_for(device, precision):
        tokens = vision_encoder.forward_features(images)
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected NLC token tensor, got shape={tuple(tokens.shape)}")
    return tokens[:, 1:].float().detach()


def reduce_sum(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def build_imagenet_datasets(cache_dir: str, rank: int, world_size: int):
    # Avoid concurrent HuggingFace cache lock creation across many SLURM tasks.
    if world_size <= 1:
        return ImageNetTrain(cache_dir=cache_dir), ImageNetVal(cache_dir=cache_dir)

    train_dataset = None
    val_dataset = None
    for owner in range(world_size):
        if rank == owner:
            train_dataset = ImageNetTrain(cache_dir=cache_dir)
            val_dataset = ImageNetVal(cache_dir=cache_dir)
        dist.barrier()

    return train_dataset, val_dataset


def train_one_epoch(
    vision_encoder,
    classifier,
    loader,
    optimizer,
    scheduler,
    device,
    precision,
):
    classifier.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tokens = patch_tokens(vision_encoder, images, device, precision)
        logits = classifier(tokens)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch = labels.numel()
        loss_sum += loss.item() * batch
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += batch

    loss_sum, correct, total = reduce_sum([loss_sum, correct, total], device)
    return {
        "train_loss": loss_sum / max(total, 1),
        "train_top1": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(vision_encoder, classifier, loader, device, precision):
    classifier.eval()
    top1 = 0
    top5 = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tokens = patch_tokens(vision_encoder, images, device, precision)
        logits = classifier(tokens)
        pred = logits.topk(5, dim=-1).indices
        top1 += (pred[:, 0] == labels).sum().item()
        top5 += (pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()

    top1, top5, total = reduce_sum([top1, top5, total], device)
    return {
        "top1": 100.0 * top1 / max(total, 1),
        "top5": 100.0 * top5 / max(total, 1),
        "top1_correct": int(top1),
        "top5_correct": int(top5),
        "num_samples": int(total),
    }


def output_path_for(vision_ckpt: str) -> Path:
    stem = Path(vision_ckpt).stem.replace("_vision_", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("analysis_results") / "levljepa_attentive_probe" / f"{stem}_{stamp}.json"


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = device_for_rank(local_rank)
    is_main = rank == 0
    torch.set_float32_matmul_precision("high")

    if is_main:
        print(f"[probe] vision_ckpt={args.vision_ckpt}", flush=True)
        print(
            f"[probe] world_size={world_size} device={device} "
            f"epochs={args.epochs} batch_size={args.batch_size}",
            flush=True,
        )

    vision_encoder = load_vision_encoder(args.vision_ckpt, args.vit, device)

    train_dataset, val_dataset = build_imagenet_datasets(
        cache_dir=args.cache_dir,
        rank=rank,
        world_size=world_size,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = NoPadDistributedSampler(len(val_dataset), rank, world_size)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    classifier = AttentiveClassifier(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        depth=args.depth,
    ).to(device)
    if world_size > 1:
        classifier = DDP(
            classifier,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    output_path = Path(args.output) if args.output else output_path_for(args.vision_ckpt)
    history_path = output_path.with_suffix(".jsonl")
    if is_main:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            history_path.unlink()
        print(
            f"[probe] train={len(train_dataset):,} val={len(val_dataset):,} "
            f"steps_per_epoch={len(train_loader)} output={output_path}",
            flush=True,
        )

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            vision_encoder,
            classifier,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.precision,
        )
        val_metrics = evaluate(
            vision_encoder,
            classifier,
            val_loader,
            device,
            args.precision,
        )
        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        history.append(record)
        if best is None or record["val_top1"] > best["val_top1"]:
            best = record.copy()

        if is_main:
            print(f"[probe] epoch={epoch} metrics={record}", flush=True)
            with history_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    result = {
        "vision_ckpt": args.vision_ckpt,
        "vit": args.vit,
        "dataset": "ILSVRC/imagenet-1k",
        "probe": "AttentiveClassifier on frozen LeVLJEPA patch tokens",
        "epochs": args.epochs,
        "batch_size_per_rank": args.batch_size,
        "val_batch_size_per_rank": args.val_batch_size,
        "world_size": world_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "depth": args.depth,
        "precision": args.precision,
        "cache_dir": args.cache_dir,
        "best": best,
        "final": history[-1],
        "history": history,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if is_main:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["best"], indent=2, sort_keys=True), flush=True)
        print(f"[probe] wrote {output_path}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
