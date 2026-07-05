#!/usr/bin/env python
"""Single-GPU CLIP-style zero-shot classification for a LeVLJEPA/InfoNCE/SigLIP model.

Loads one VL checkpoint (vision + text encoders, each with a BN-MLP `pre_proj`
head and a cross-modal `projector`) and evaluates zero-shot top-1/top-5 on
ImageNet, Places365, FGVC-Aircraft, and Oxford-IIIT Pets.

For each dataset we report four alignment variants (class text embeddings are the
text `pre_proj` space, pad77 readout):
  - zeroshot : image pre_proj   vs text pre_proj   (contrastive alignment; read this for InfoNCE/SigLIP)
  - proj     : image projector  vs text pre_proj   (image->text prediction)
  - proj_text: image pre_proj   vs text projector  (text->image prediction)
  - avg      : mean of proj + proj_text            (predictive alignment; read this for LeVLJEPA)

Single GPU, no DDP -> schedules into any free GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MLP
from torchvision.transforms import v2 as T
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config, GPT2Model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# repo, split, image_field, label_field, prompt template
DATASETS = {
    "imagenet": ("ILSVRC/imagenet-1k", "validation", "image", "label", "a photo of a {}."),
    "places365": ("dpdl-benchmark/Places365-Validation", "train", "image", "label", "a photo of a {}."),
    "aircraft": ("Multimodal-Fatima/FGVC_Aircraft_test", "test", "image", "label", "a photo of a {}, a type of aircraft."),
    "pets": ("timm/oxford-iiit-pet", "test", "image", "label", "a photo of a {}, a type of pet."),
}
ALIGN_VARIANTS = ("zeroshot", "proj", "proj_text", "avg")


def autocast_for(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def clip_transform():
    return T.Compose([
        T.ToImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_pre_proj(hidden, width, embed):
    return nn.Sequential(
        nn.Linear(hidden, width), nn.BatchNorm1d(width), nn.GELU(), nn.Linear(width, embed)
    )


def build_projector(embed, width, depth):
    return MLP(embed, [width] * depth + [embed], norm_layer=nn.BatchNorm1d,
               activation_layer=nn.GELU, dropout=0.0)


def load_model(args, tokenizer, device):
    vc = torch.load(args.vision_ckpt, map_location="cpu", weights_only=True)
    tc = torch.load(args.text_ckpt, map_location="cpu", weights_only=True)
    vis_enc = timm.create_model(args.vit, pretrained=False, num_classes=0, dynamic_img_size=True)
    vis_pre = build_pre_proj(args.hidden_size, args.pre_proj_width, args.embed_dim)
    txt_enc = GPT2Model(GPT2Config(
        n_embd=args.hidden_size, n_layer=args.num_layers, n_head=args.num_heads,
        n_inner=args.hidden_size * 4, vocab_size=tokenizer.vocab_size,
        attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0))
    txt_pre = build_pre_proj(args.hidden_size, args.pre_proj_width, args.embed_dim)

    vis_enc.load_state_dict(vc["encoder"]); vis_pre.load_state_dict(vc["pre_proj"])
    txt_enc.load_state_dict(tc["encoder"]); txt_pre.load_state_dict(tc["pre_proj"])

    # The cross-modal predictor ("projector") is LeVLJEPA-specific; contrastive
    # baselines (InfoNCE/SigLIP) save it empty and align directly in pre_proj space.
    has_proj = bool(vc.get("projector")) and bool(tc.get("projector"))
    if has_proj:
        vis_proj = build_projector(args.embed_dim, args.projector_width, args.projector_depth)
        txt_proj = build_projector(args.embed_dim, args.projector_width, args.projector_depth)
        vis_proj.load_state_dict(vc["projector"]); txt_proj.load_state_dict(tc["projector"])
    else:
        vis_proj = txt_proj = None

    for m in (vis_enc, vis_pre, vis_proj, txt_enc, txt_pre, txt_proj):
        if m is None:
            continue
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
    return vis_enc, vis_pre, vis_proj, txt_enc, txt_pre, txt_proj


class HFZeroShot(Dataset):
    def __init__(self, repo, split, image_field, label_field, cache_dir, limit=None):
        ds = load_dataset(repo, split=split, cache_dir=cache_dir)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds, self.image_field, self.label_field = ds, image_field, label_field
        self.class_names = ds.features[label_field].names
        self.transform = clip_transform()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        return self.transform(ex[self.image_field].convert("RGB")), int(ex[self.label_field])


@torch.no_grad()
def encode_classes(txt_enc, txt_pre, txt_proj, tokenizer, class_names, template,
                   device, precision, batch_size=64):
    prompts = [template.format(c.replace("_", " ").strip()) for c in class_names]
    feats, projs = [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=77,
                           return_tensors="pt").to(device)
        with autocast_for(device, precision):
            hidden = txt_enc(**inputs).last_hidden_state
            pooled = hidden[:, -1, :]            # pad77 readout
            tf = txt_pre(pooled)
            tp = txt_proj(tf) if txt_proj is not None else None
        feats.append(F.normalize(tf.float(), dim=-1))
        if tp is not None:
            projs.append(F.normalize(tp.float(), dim=-1))
    return torch.cat(feats), (torch.cat(projs) if projs else None)


@torch.no_grad()
def eval_dataset(mods, tokenizer, name, args, device):
    vis_enc, vis_pre, vis_proj, txt_enc, txt_pre, txt_proj = mods
    repo, split, img_f, lbl_f, template = DATASETS[name]
    if name == "imagenet" and args.cache_dir:
        ds = HFZeroShot(repo, split, img_f, lbl_f, args.cache_dir, args.limit_samples)
    else:
        ds = HFZeroShot(repo, split, img_f, lbl_f, args.cache_dir, args.limit_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    txt_feat, txt_projd = encode_classes(
        txt_enc, txt_pre, txt_proj, tokenizer, ds.class_names, template, device, args.precision)

    has_proj = vis_proj is not None and txt_projd is not None
    variants = list(ALIGN_VARIANTS) if has_proj else ["zeroshot"]
    counts = {v: [0, 0] for v in variants}
    total = 0
    for images, labels in tqdm(loader, desc=f"{name}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_for(device, args.precision):
            img_feat = vis_pre(vis_enc(images))
            img_projd = vis_proj(img_feat) if has_proj else None
        i_norm = F.normalize(img_feat.float(), dim=-1)
        logits = {"zeroshot": i_norm @ txt_feat.T}
        if has_proj:
            ip_norm = F.normalize(img_projd.float(), dim=-1)
            logits["proj"] = ip_norm @ txt_feat.T
            logits["proj_text"] = i_norm @ txt_projd.T
            logits["avg"] = (logits["proj"] + logits["proj_text"]) / 2
        for v in variants:
            pred = logits[v].topk(5, dim=-1).indices
            counts[v][0] += (pred[:, 0] == labels).sum().item()
            counts[v][1] += (pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()

    return {
        "num_classes": len(ds.class_names), "num_samples": total,
        "template": template, "variants": variants,
        **{f"{v}_top1": 100.0 * counts[v][0] / max(total, 1) for v in variants},
        **{f"{v}_top5": 100.0 * counts[v][1] / max(total, 1) for v in variants},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vision-ckpt", required=True)
    p.add_argument("--text-ckpt", required=True)
    p.add_argument("--label", default="model", help="short model name for the output")
    p.add_argument("--embed-dim", type=int, required=True)
    p.add_argument("--vit", default="vit_base_patch16_224")
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--pre-proj-width", type=int, default=2048)
    p.add_argument("--projector-width", type=int, default=2048)
    p.add_argument("--projector-depth", type=int, default=4)
    p.add_argument("--datasets", default="imagenet,places365,aircraft,pets")
    p.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "./data"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise ValueError(f"unknown dataset {d!r}; choices {list(DATASETS)}")

    print(f"[zs] model={args.label} embed_dim={args.embed_dim} datasets={datasets}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    mods = load_model(args, tokenizer, device)

    results = {}
    for name in datasets:
        m = eval_dataset(mods, tokenizer, name, args, device)
        results[name] = m
        extra = f" avg_top1={m['avg_top1']:.2f}" if "avg_top1" in m else ""
        print(f"[zs] {args.label}/{name}: zeroshot_top1={m['zeroshot_top1']:.2f}{extra} "
              f"(n={m['num_samples']}, {m['num_classes']} cls)", flush=True)

    out = {
        "label": args.label, "vision_ckpt": args.vision_ckpt, "text_ckpt": args.text_ckpt,
        "embed_dim": args.embed_dim, "precision": args.precision,
        "results": results, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output) if args.output else Path(
        f"analysis_results/zeroshot/{args.label}_zeroshot.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[zs] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
