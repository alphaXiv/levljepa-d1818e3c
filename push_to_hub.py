#!/usr/bin/env python3
"""
Upload a LeVLJEPA checkpoint to a HuggingFace model repository.

Downloads the vision and text .pt checkpoint files from an HF bucket (or uses
local files), converts weights to safetensors, writes a config.json, generates
a model card, and pushes everything to a new or existing HF model repo.

Usage — from bucket:
    python push_to_hub.py \\
        --bucket your-hf-org/your-bucket \\
        --run_name my_levljepa_run \\
        --step 50000 \\
        --repo_id your-hf-org/LeVLJEPA-ViT-B-CC12M

Usage — from local .pt files:
    python push_to_hub.py \\
        --vision_ckpt /path/to/vision.pt \\
        --text_ckpt /path/to/text.pt \\
        --repo_id your-hf-org/LeVLJEPA-ViT-B-CC12M

Required:
    --repo_id        Target HF model repo  (e.g. org/model-name)

Source (one of):
    --bucket + --run_name + --step    Download from HF bucket
    --vision_ckpt + --text_ckpt      Use local .pt files

Optional:
    --model_size     tiny | small | base  (default: base)
    --embed_dim      Projection embedding dimension (default: 768 for base)
    --projector_width   MLP hidden width (default: 2048)
    --projector_depth   MLP depth (default: 4)
    --total_steps    Training steps (default: 50000)
    --batch_size     Training batch size (shown in card, default: 256)
    --private        Make the repo private
    --tmp_dir        Local directory for downloaded files (default: /tmp/levljepa_push)
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi, create_repo, get_token
from safetensors.torch import save_file


MODEL_CONFIGS = {
    "tiny":  {"vit": "vit_tiny_patch16_224",  "hidden_size": 192, "num_layers": 12, "num_heads": 3},
    "small": {"vit": "vit_small_patch16_224", "hidden_size": 384, "num_layers": 12, "num_heads": 6},
    "base":  {"vit": "vit_base_patch16_224",  "hidden_size": 768, "num_layers": 12, "num_heads": 12},
}


def flatten_state_dict(state_dict: dict, prefix: str) -> dict:
    """Prefix all keys so nested component dicts become a flat safetensors-compatible dict."""
    return {f"{prefix}.{k}": v.contiguous() for k, v in state_dict.items()}


def _http_download(url: str, dest: Path, headers: dict) -> None:
    """Stream a file from `url` to `dest`, showing MB progress."""
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def load_checkpoint_from_bucket(bucket: str, run_name: str, step: int, tmp_dir: Path) -> tuple[dict, dict]:
    """Download vision and text .pt files from an HF bucket via direct HTTP.

    Buckets are accessible at:
        https://huggingface.co/buckets/{org}/{bucket}/resolve/{run_name}/{filename}
    mirroring the 'tree' browse URL shown in the web UI.
    """
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base = f"https://huggingface.co/buckets/{bucket}/resolve/{run_name}"

    vision_local = tmp_dir / f"vision_step{step}.pt"
    text_local   = tmp_dir / f"text_step{step}.pt"

    for filename, local_path in [
        (f"{run_name}_vision_step{step}.pt", vision_local),
        (f"{run_name}_text_step{step}.pt",   text_local),
    ]:
        url = f"{base}/{filename}"
        print(f"[download] {url}")
        _http_download(url, local_path, headers)

    vision_ckpt = torch.load(vision_local, map_location="cpu", weights_only=True)
    text_ckpt   = torch.load(text_local,   map_location="cpu", weights_only=True)
    return vision_ckpt, text_ckpt


def load_checkpoint_from_local(vision_path: str, text_path: str) -> tuple[dict, dict]:
    vision_ckpt = torch.load(vision_path, map_location="cpu", weights_only=True)
    text_ckpt   = torch.load(text_path,   map_location="cpu", weights_only=True)
    return vision_ckpt, text_ckpt


def build_config(args, model_cfg: dict) -> dict:
    cfg = {
        "model_type": args.model_type,
        "model_size": args.model_size,
        "vision_encoder": {
            "architecture": model_cfg["vit"],
            "hidden_size": model_cfg["hidden_size"],
            "num_layers": model_cfg["num_layers"],
            "num_heads": model_cfg["num_heads"],
        },
        "text_encoder": {
            "architecture": "gpt2",
            "hidden_size": model_cfg["hidden_size"],
            "num_layers": model_cfg["num_layers"],
            "num_heads": model_cfg["num_heads"],
            "vocab_size": 50257,
        },
        "embed_dim": args.embed_dim,
        "projector_width": args.projector_width,
        "projector_depth": args.projector_depth,
        "training": {
            "dataset": "cc12m",
            "total_steps": args.total_steps,
            "batch_size": args.batch_size,
        },
    }
    return cfg


def build_model_card(args, model_cfg: dict) -> str:
    variant = "LeVLJEPA"

    return f"""\
---
license: apache-2.0
language:
- en
tags:
- vision-language
- self-supervised-learning
- jepa
- non-contrastive
- image-text
- vit
- gpt2
datasets:
- pixparse/cc12m-wds
---

# {variant} — {model_cfg['vit']} / CC12M

Official checkpoint for **{variant}**, a non-contrastive vision-language pretraining method
based on joint-embedding prediction and SIGReg regularisation.
Trained on [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds) for {args.total_steps:,} steps
with batch size {args.batch_size:,}.

## Model summary

| Property | Value |
|---|---|
| Vision encoder | `{model_cfg['vit']}` (timm) |
| Text encoder | GPT-2 ({model_cfg['num_layers']}L / {model_cfg['num_heads']}H / {model_cfg['hidden_size']}D) |
| Embedding dim | {args.embed_dim} |
| Projector | {args.projector_depth}-layer MLP, width {args.projector_width} |
| Training objective | Cross-modal prediction + SIGReg |
| Training data | CC12M (~12 M image-caption pairs) |
| Training steps | {args.total_steps:,} |

## Method

{variant} aligns image and text embeddings through **predictive losses** rather than
contrastive classification:

1. **Cross-modal prediction** — image embeddings predict stop-gradient text embeddings and
   vice versa via modality-specific MLP predictors.
2. **SIGReg regularisation** — each modality's marginal embedding distribution is independently
   regularised toward an isotropic Gaussian, preventing representation collapse without
   needing negative pairs.

The objective has **no negatives, no temperature parameter, no momentum encoder**.

## Usage

```python
import torch
import timm
from transformers import GPT2Config, GPT2Model, AutoTokenizer
from safetensors.torch import load_file
from torchvision.ops import MLP
import torch.nn as nn

HIDDEN = {model_cfg['hidden_size']}
EMBED  = {args.embed_dim}

# ── Vision encoder ───────────────────────────────────────────────────────────
vision_encoder = timm.create_model(
    "{model_cfg['vit']}", pretrained=False, num_classes=0, dynamic_img_size=True
)
vision_pre_proj = nn.Sequential(
    nn.Linear(HIDDEN, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, EMBED)
)

# ── Text encoder ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

text_encoder = GPT2Model(GPT2Config(
    n_embd=HIDDEN, n_layer={model_cfg['num_layers']}, n_head={model_cfg['num_heads']},
    n_inner=HIDDEN * 4, vocab_size=tokenizer.vocab_size,
    attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
))
text_pre_proj = nn.Sequential(
    nn.Linear(HIDDEN, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, EMBED)
)

# ── Load weights ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download

vision_weights = load_file(hf_hub_download("{args.repo_id}", "vision_encoder.safetensors"))
text_weights   = load_file(hf_hub_download("{args.repo_id}", "text_encoder.safetensors"))

vision_encoder.load_state_dict({{k[len("encoder."):]: v for k, v in vision_weights.items() if k.startswith("encoder.")}})
vision_pre_proj.load_state_dict({{k[len("pre_proj."):]: v for k, v in vision_weights.items() if k.startswith("pre_proj.")}})
text_encoder.load_state_dict({{k[len("encoder."):]: v for k, v in text_weights.items() if k.startswith("encoder.")}})
text_pre_proj.load_state_dict({{k[len("pre_proj."):]: v for k, v in text_weights.items() if k.startswith("pre_proj.")}})

vision_encoder.eval()
text_encoder.eval()

# ── Encode an image ──────────────────────────────────────────────────────────
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("image.jpg").convert("RGB")
pixel_values = transform(image).unsqueeze(0)

with torch.no_grad():
    image_features = vision_pre_proj(vision_encoder(pixel_values))  # (1, {args.embed_dim})

# ── Encode a caption ─────────────────────────────────────────────────────────
inputs = tokenizer("a photo of a cat", return_tensors="pt", padding=True)
with torch.no_grad():
    text_hidden = text_encoder(**inputs).last_hidden_state[:, -1, :]
    text_features = text_pre_proj(text_hidden)  # (1, {args.embed_dim})
```

## Files

| File | Contents |
|---|---|
| `vision_encoder.safetensors` | Vision encoder (`encoder.*`), pre-projection head (`pre_proj.*`), and cross-modal projector MLP (`projector.*`) |
| `text_encoder.safetensors`   | Text encoder (`encoder.*`), pre-projection head (`pre_proj.*`), and cross-modal projector MLP (`projector.*`) |
| `config.json` | Architecture and training hyperparameters |

## Citation

```bibtex
@inproceedings{{anonymous2026levljepa,
  title     = {{LeVLJEPA: Non-Contrastive Joint-Embedding Prediction for Vision-Language Pretraining}},
  author    = {{Anonymous Authors}},
  booktitle = {{NeurIPS}},
  year      = {{2026}},
}}
```
"""


def push_to_hub(args):
    model_cfg = MODEL_CONFIGS[args.model_size]
    # Default embed_dim to hidden_size if not explicitly set
    if args.embed_dim is None:
        args.embed_dim = model_cfg["hidden_size"]

    # ── Load checkpoints ──────────────────────────────────────────────────────
    if args.bucket:
        if not args.run_name or args.step is None:
            raise ValueError("--run_name and --step are required when using --bucket")
        tmp_dir = Path(args.tmp_dir)
        vision_ckpt, text_ckpt = load_checkpoint_from_bucket(
            args.bucket, args.run_name, args.step, tmp_dir
        )
    else:
        if not args.vision_ckpt or not args.text_ckpt:
            raise ValueError("Provide either --bucket or both --vision_ckpt and --text_ckpt")
        vision_ckpt, text_ckpt = load_checkpoint_from_local(args.vision_ckpt, args.text_ckpt)

    # ── Convert to safetensors ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)

        vision_flat = {}
        for component in ("encoder", "pre_proj", "projector"):
            vision_flat.update(flatten_state_dict(vision_ckpt[component], component))
        save_file(vision_flat, staging / "vision_encoder.safetensors")
        print(f"[convert] vision — {len(vision_flat)} tensors")

        text_flat = {}
        for component in ("encoder", "pre_proj", "projector"):
            text_flat.update(flatten_state_dict(text_ckpt[component], component))
        save_file(text_flat, staging / "text_encoder.safetensors")
        print(f"[convert] text   — {len(text_flat)} tensors")

        # ── Write config.json ─────────────────────────────────────────────────
        config = build_config(args, model_cfg)
        with open(staging / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ── Write model card ──────────────────────────────────────────────────
        model_card = build_model_card(args, model_cfg)
        with open(staging / "README.md", "w") as f:
            f.write(model_card)

        # ── Push to Hub ───────────────────────────────────────────────────────
        api = HfApi()
        create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
        print(f"[push] uploading to https://huggingface.co/{args.repo_id}")

        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload {args.model_type} {args.model_size} checkpoint",
        )
        print(f"[done] https://huggingface.co/{args.repo_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Source
    src = p.add_argument_group("source (bucket)")
    src.add_argument("--bucket",   help="HF bucket path, e.g. your-hf-org/your-bucket")
    src.add_argument("--run_name", help="Run name used during training (subdirectory in bucket)")
    src.add_argument("--step",     type=int, help="Checkpoint step number")
    src.add_argument("--tmp_dir",  default="/tmp/levljepa_push", help="Local temp directory for downloads")

    src2 = p.add_argument_group("source (local files)")
    src2.add_argument("--vision_ckpt", help="Local path to *_vision_step*.pt")
    src2.add_argument("--text_ckpt",   help="Local path to *_text_step*.pt")

    # Destination
    dst = p.add_argument_group("destination")
    dst.add_argument("--repo_id",  required=True, help="HF model repo, e.g. org/model-name")
    dst.add_argument("--private",  action="store_true", help="Create a private repo")

    # Architecture
    arch = p.add_argument_group("architecture")
    arch.add_argument("--model_type",       default="levljepa", choices=["levljepa"])
    arch.add_argument("--model_size",       default="base", choices=["tiny", "small", "base"])
    arch.add_argument("--embed_dim",        type=int, default=None,
                      help="Projection embedding dim (defaults to hidden_size)")
    arch.add_argument("--projector_width",  type=int, default=2048)
    arch.add_argument("--projector_depth",  type=int, default=4)

    # Training metadata (for model card)
    train = p.add_argument_group("training metadata")
    train.add_argument("--total_steps",  type=int, default=50000)
    train.add_argument("--batch_size",   type=int, default=256)

    args = p.parse_args()
    push_to_hub(args)


if __name__ == "__main__":
    main()
