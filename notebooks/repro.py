# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "timm",
#     "transformers",
#     "datasets",
#     "safetensors",
#     "huggingface_hub>=1.0",
#     "numpy",
#     "pillow",
#     "tqdm",
#     "marimo",
#     "matplotlib",
# ]
#
# [tool.uv]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
# torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
# ///
import marimo

__generated_with__ = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        """
        # LeVLJEPA — minimal reproduction

        Reproduces two claims from **LeVLJEPA: End-to-End Vision-Language
        Pretraining Without Negatives** (arXiv:2607.00784) using the released
        ViT-B/16 Datacomp-200k checkpoint against public contrastive baselines:

        1. **Dense Feature Advantage** — linear semantic segmentation on ADE20K
           (frozen patch tokens, mIoU).
        2. **Object-Centricity Validation** — ImageNet-9 background-shift
           robustness (linear probe on frozen CLS features).

        Self-contained: marimo does not clone the repo, so checkpoints and a
        small slice of the eval data are downloaded into the notebook. Sliders
        default to small subsets so it finishes in minutes; run
        `bash run_repro.sh` in the repo for full-scale numbers.
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import random
    import tarfile
    import urllib.request as urlreq
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    torch.set_float32_matmul_precision("high")
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEV
    return (
        DEV, Dataset, DataLoader, F, Image, Path, nn, np, os, random, tarfile,
        torch, transforms, urlreq,
    )


@app.cell
def _():
    # Subset sizes (edit here or in `marimo edit`). Kept small so the notebook
    # runs in minutes; `bash run_repro.sh` in the repo runs the full datasets.
    n_ade_train = 600
    n_ade_val = 200
    n_in9_per_class = 60
    return n_ade_train, n_ade_val, n_in9_per_class


@app.cell
def _(mo):
    mo.md("## Load the three frozen vision encoders (ViT-B/16, native norm)")
    return


@app.cell
def _():
    LEVLJEPA_HF = "lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k"
    SIGLIP_HF = "google/siglip-base-patch16-224"
    CLIP_HF = "openai/clip-vit-base-patch16"
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    MODELS = ["levljepa", "siglip", "clip_openai"]
    return LEVLJEPA_HF, SIGLIP_HF, CLIP_HF, CLIP_MEAN, CLIP_STD, MODELS


@app.cell
def _(DEV, LEVLJEPA_HF, torch):
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    enc = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
    )
    vw = load_file(hf_hub_download(LEVLJEPA_HF, "vision_encoder.safetensors"))
    enc.load_state_dict(
        {k[8:]: v for k, v in vw.items() if k.startswith("encoder.")}, strict=True
    )
    enc.to(DEV).eval()
    for _p in enc.parameters():
        _p.requires_grad = False
    "LeVLJEPA loaded"
    return enc, timm, load_file, hf_hub_download


@app.cell
def _(DEV, SIGLIP_HF, torch):
    from transformers import SiglipVisionModel

    senc = SiglipVisionModel.from_pretrained(SIGLIP_HF)
    senc.to(DEV).eval()
    for _p in senc.parameters():
        _p.requires_grad = False
    "SigLIP loaded"
    return SiglipVisionModel, senc


@app.cell
def _(CLIP_HF, DEV, torch):
    from transformers import CLIPVisionModel

    cenc = CLIPVisionModel.from_pretrained(CLIP_HF)
    cenc.to(DEV).eval()
    for _p in cenc.parameters():
        _p.requires_grad = False
    "OpenAI CLIP loaded"
    return CLIPVisionModel, cenc


@app.cell
def _(cenc, enc, np, senc, torch):
    @torch.no_grad()
    def feats(name, images):
        """Return (cls [B,D], patches [B,196,D]) for a 224x224 batch."""
        if name == "levljepa":
            f = enc.forward_features(images)  # (B,197,D)
            return f[:, 0].float(), f[:, 1:].float()
        if name == "clip_openai":
            out = cenc(pixel_values=images).last_hidden_state  # (B,197,D)
            return out[:, 0].float(), out[:, 1:].float()
        out = senc(pixel_values=images).last_hidden_state  # (B,196,D), no CLS
        return out.mean(dim=1).float(), out.float()


    def dim_of(name):
        return {
            "levljepa": enc.embed_dim,
            "clip_openai": cenc.config.hidden_size,
            "siglip": senc.config.hidden_size,
        }[name]


    def norm_for(name):
        return (CLIP_MEAN, CLIP_STD) if name != "siglip" else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return feats, dim_of, norm_for


@app.cell
def _(mo):
    mo.md("## 1) Dense Feature Advantage — ADE20K linear segmentation")
    return


@app.cell
def _():
    from datasets import load_dataset

    ade = load_dataset("merve/scene_parse_150")
    ade_train, ade_val = ade["train"], ade["validation"]
    NUM_ADE, VOID, ONE_IDX = 150, 0, True  # verified: 0=void, 1..150 classes
    NUM_ADE
    return NUM_ADE, VOID, ONE_IDX, ade_train, ade_val, load_dataset


@app.cell
def _(Dataset, Image, NUM_ADE, ONE_IDX, VOID, np, transforms):
    def seg_tf_for(norm):
        mean, std = norm
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    lab_tf = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
        ]
    )


    class ADESeg(Dataset):
        def __init__(self, split, n, img_tf):
            self.split, self.n, self.img_tf = split, min(n, len(split)), img_tf

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            ex = self.split[i]
            img = self.img_tf(ex["image"].convert("RGB"))
            lab = lab_tf(ex["annotation"].convert("L"))
            lab = torch.as_tensor(np.array(lab, dtype=np.int64))
            lab = torch.where(lab == 0, torch.tensor(255), lab - 1) if ONE_IDX else torch.where(lab == VOID, torch.tensor(255), lab)
            return img, lab
    return ADESeg, lab_tf, seg_tf_for


@app.cell
def _(ADESeg, DEV, F, DataLoader, NUM_ADE, ade_train, ade_val, dim_of, feats, norm_for, n_ade_train, n_ade_val, np, seg_tf_for, torch):
    def run_seg(name, ntr, nva, epochs=20):
        tf = seg_tf_for(norm_for(name))
        tr, va = ADESeg(ade_train, ntr, tf), ADESeg(ade_val, nva, tf)
        D = dim_of(name)

        def cache(ds):
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            fs, ls = [], []
            for img, lab in loader:
                _, p = feats(name, img.to(DEV))
                fs.append(p.cpu()); ls.append(lab)
            return torch.cat(fs, 0), torch.cat(ls, 0)

        trf, trl = cache(tr)
        vaf, val = cache(va)
        flat = trf.reshape(-1, D).to(DEV)
        mean, std = flat.mean(0).cpu(), flat.std(0).clamp(min=1e-8).cpu()
        del flat; torch.cuda.empty_cache()
        trl14 = F.interpolate(trl.unsqueeze(1).float(), size=(14, 14), mode="nearest")[:, 0].long()
        lin = nn.Linear(D, NUM_ADE, bias=True).to(DEV)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
        lf = nn.CrossEntropyLoss(ignore_index=255)
        idx = torch.arange(trf.shape[0]); rng = random.Random(0)
        for _ in range(epochs):
            lin.train()
            perm = idx[rng.sample(range(trf.shape[0]), trf.shape[0])]
            for s in range(0, trf.shape[0], 128):
                b = perm[s:s + 128]
                f = ((trf[b] - mean) / std).to(DEV)
                l = trl14[b].to(DEV)
                logits = lin(f).permute(0, 2, 1).contiguous().view(-1, NUM_ADE, 14, 14)
                loss = lf(logits, l)
                opt.zero_grad(); loss.backward(); opt.step()
        lin.eval()
        cm = torch.zeros(NUM_ADE, NUM_ADE, dtype=torch.long)
        with torch.no_grad():
            for j in range(vaf.shape[0]):
                f = ((vaf[j:j + 1] - mean) / std).to(DEV)
                g = val[j].to(DEV)
                logits = lin(f).permute(0, 2, 1).contiguous().view(1, NUM_ADE, 14, 14)
                up = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False).argmax(1)[0]
                v = g != 255
                p, gg = up[v], g[v]
                cm += torch.bincount(p * NUM_ADE + gg, minlength=NUM_ADE * NUM_ADE).reshape(NUM_ADE, NUM_ADE).cpu()
        inter = torch.diag(cm).float()
        union = (cm.sum(0) + cm.sum(1) - torch.diag(cm)).float().clamp(min=1)
        present = (cm.sum(0) + cm.sum(1) - torch.diag(cm)) > 0
        miou = (inter / union)[present].mean().item() * 100
        return {"mIoU": miou, "lin": lin, "mean": mean, "std": std,
                "tf": tf, "val_split": va}
    return (run_seg,)


@app.cell
def _(MODELS, mo, n_ade_train, n_ade_val, run_seg):
    mo.md("Running ADE20K linear segmentation for all encoders...")
    seg = {m: run_seg(m, n_ade_train, n_ade_val) for m in MODELS}
    _seg_rows = "\n".join(f"| {m} | {seg[m]['mIoU']:.2f} |" for m in MODELS)
    mo.md(
        f"""
        **ADE20K linear mIoU** (frozen patch tokens, subset
        {n_ade_train} train / {n_ade_val} val):

        | Encoder | ADE20K mIoU |
        |---|---|
        {_seg_rows}

        Paper (full Datacomp-L): LeVLJEPA 23.15, InfoNCE 20.90, SigLIP 19.24.
        """
    )
    return (seg,)


@app.cell
def _(mo):
    mo.md("### Predicted segmentation masks — a few ADE20K val samples")
    return


@app.cell
def _(ADESeg, DEV, F, Image, MODELS, NUM_ADE, ade_val, feats, mo, n_ade_val, np, seg, torch):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(0)
    idxs = rng.choice(n_ade_val, size=4, replace=False)
    raw = [ade_val[int(i)] for i in idxs]

    cmap = ListedColormap(np.random.default_rng(1).random((NUM_ADE, 3)))
    names = ["GT"] + MODELS
    fig, axes = plt.subplots(len(raw), 1 + len(MODELS), figsize=(2.4 * (1 + len(MODELS)), 2.4 * len(raw)))
    if len(raw) == 1:
        axes = axes[None, :]
    for r, ex in enumerate(raw):
        gt = ex["annotation"].convert("L").resize((224, 224), Image.NEAREST)
        gt_arr = np.array(gt, dtype=np.int64)
        gt_arr = np.where(gt_arr == 0, 255, gt_arr - 1)
        axes[r, 0].imshow(gt_arr, cmap=cmap, vmin=0, vmax=NUM_ADE - 1, interpolation="nearest")
        axes[r, 0].set_title("GT", fontsize=9)
        axes[r, 0].axis("off")
        for ci, m in enumerate(MODELS):
            tf = seg[m]["tf"]
            img = tf(ex["image"].convert("RGB")).unsqueeze(0).to(DEV)
            with torch.no_grad():
                _, p = feats(m, img)
            f = ((p - seg[m]["mean"]) / seg[m]["std"]).to(DEV)
            logits = seg[m]["lin"](f).permute(0, 2, 1).contiguous().view(1, NUM_ADE, 14, 14)
            up = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            axes[r, ci + 1].imshow(up, cmap=cmap, vmin=0, vmax=NUM_ADE - 1, interpolation="nearest")
            axes[r, ci + 1].set_title(f"{m}\n{seg[m]['mIoU']:.1f}", fontsize=8)
            axes[r, ci + 1].axis("off")
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("## 2) Object-Centricity — ImageNet-9 background robustness")
    return


@app.cell
def _(Path, os, tarfile, urlreq):
    DATA = Path(os.environ.get("REPRO_DATA", "./data_repro"))
    DATA.mkdir(parents=True, exist_ok=True)
    root = DATA / "bg_challenge"
    if not (root / "original" / "val" / "00_dog").exists():
        url = ("https://github.com/MadryLab/backgrounds_challenge/releases/download/data/"
               "backgrounds_challenge_data.tar.gz")
        tar = DATA / "bg.tar.gz"
        if not tar.exists():
            urlreq.urlretrieve(url, tar)
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(DATA)
    "IN-9 release ready"
    return DATA, root


@app.cell
def _():
    IN9_CLASSES = ["00_dog", "01_bird", "02_wheeled vehicle", "03_reptile", "04_carnivore",
           "05_insect", "06_musical instrument", "07_primate", "08_fish"]
    return (IN9,)


@app.cell
def _(Dataset, Image, IN9, transforms):
    def cls_tf_for(norm):
        mean, std = norm
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


    class IN9(Dataset):
        def __init__(self, root, split, img_tf, limit_per_class=None):
            base = root / split / "val"
            self.items = []
            for ci, c in enumerate(IN9_CLASSES):
                d = base / c
                if not d.exists():
                    continue
                files = sorted(d.iterdir())
                if limit_per_class:
                    files = files[:limit_per_class]
                for f in files:
                    self.items.append((str(f), ci))
            self.img_tf = img_tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, c = self.items[i]
            return self.img_tf(Image.open(p).convert("RGB")), c
    return IN9, cls_tf_for


@app.cell
def _(DEV, F, IN9, DataLoader, cls_tf_for, feats, n_in9_per_class, nn, norm_for, random, root, torch):
    def run_in9(name, npc):
        tf = cls_tf_for(norm_for(name))
        orig = IN9(root, "original", tf, npc)
        msame = IN9(root, "mixed_same", tf, npc)
        mrand = IN9(root, "mixed_rand", tf, npc)
        byc = {}
        for i in range(len(orig)):
            byc.setdefault(orig.items[i][1], []).append(i)
        rng = random.Random(0)
        tr, ev = [], []
        for ci, idxs in byc.items():
            idxs = idxs[:]; rng.shuffle(idxs); cut = len(idxs) // 2
            tr.extend(idxs[:cut]); ev.extend(idxs[cut:])

        def sub(base, idxs):
            d = IN9.__new__(IN9); d.items = [base.items[i] for i in idxs]; d.img_tf = base.img_tf
            return d

        def cache(ds):
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
            fs, ls = [], []
            for img, lab in loader:
                c, _ = feats(name, img.to(DEV))
                fs.append(c.cpu()); ls.append(lab)
            return F.normalize(torch.cat(fs, 0).float(), dim=-1), torch.cat(ls, 0)

        trf, trl = cache(sub(orig, tr))
        evf, evl = cache(sub(orig, ev))
        msf, msl = cache(msame)
        mrf, mrl = cache(mrand)
        mean, std = trf.mean(0), trf.std(0).clamp(min=1e-8)
        nC = 9
        lin = nn.Linear(trf.shape[1], nC, bias=False).to(DEV)
        opt = torch.optim.AdamW(lin.parameters(), lr=0.1, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss()
        n = trf.shape[0]; idx = torch.arange(n)
        for _ in range(50):
            for s in range(0, n, min(4096, n)):
                b = idx[rng.sample(range(n), n)][s:s + min(4096, n)]
                f = ((trf[b] - mean) / std).to(DEV); y = trl[b].to(DEV)
                loss = lf(lin(f), y)
                opt.zero_grad(); loss.backward(); opt.step()
        lin.eval()

        def acc(f, y):
            with torch.no_grad():
                p = lin(((f - mean) / std).to(DEV)).argmax(1).cpu()
            return 100.0 * (p == y).float().mean().item()

        ao, am, ar = acc(evf, evl), acc(msf, msl), acc(mrf, mrl)
        return {"Original": ao, "Mixed-Same": am, "Mixed-Rand": ar,
                "drop_ms": ao - am, "drop_mr": ao - ar}
    return (run_in9,)


@app.cell
def _(MODELS, mo, n_in9_per_class, run_in9):
    mo.md("Running ImageNet-9 linear-probe robustness for all encoders...")
    in9 = {m: run_in9(m, n_in9_per_class) for m in MODELS}
    _in9_rows = "\n".join(
        f"| {m} | {in9[m]['Original']:.2f} | {in9[m]['Mixed-Same']:.2f} | "
        f"{in9[m]['Mixed-Rand']:.2f} | {in9[m]['drop_ms']:.2f} | {in9[m]['drop_mr']:.2f} |"
        for m in MODELS
    )
    mo.md(
        f"""
        **ImageNet-9** (linear probe on frozen CLS, subset ~{n_in9_per_class}/class):

        | Encoder | Original | Mixed-Same | Mixed-Rand | drop MS | drop MR |
        |---|---|---|---|---|---|
        {_in9_rows}

        Paper (full): LeVLJEPA 96.96/91.01/79.75 (drops 5.95/17.21),
        SigLIP 96.44/89.41/78.35 (7.03/18.09).
        """
    )
    return (in9,)


@app.cell
def _(mo):
    mo.md("### Background-shift robustness — drops per encoder")
    return


@app.cell
def _(IN9, MODELS, Image, in9, mo, root):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(MODELS))
    w = 0.38
    ms = [in9[m]["drop_ms"] for m in MODELS]
    mr = [in9[m]["drop_mr"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w / 2, ms, w, label="drop Mixed-Same", color="#4C72B0")
    ax.bar(x + w / 2, mr, w, label="drop Mixed-Rand", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel("Accuracy drop (pp)"); ax.legend(fontsize=8)
    ax.set_title("IN-9 background-shift drops (larger = more object-centric on MR)")
    for i, v in enumerate(ms):
        ax.text(i - w / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
    for i, v in enumerate(mr):
        ax.text(i + w / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("### What the background shift looks like — one class, three splits")
    return


@app.cell
def _(Image, IN9, MODELS, mo, np, root, transforms):
    import matplotlib.pyplot as plt

    cls_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    splits = ["original", "mixed_same", "mixed_rand"]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for c, sp in enumerate(splits):
        ds = IN9(root, sp, cls_tf, limit_per_class=1)
        if len(ds) == 0:
            axes[c].axis("off"); continue
        img, lab = ds[0]
        axes[c].imshow(img.permute(1, 2, 0).numpy())
        axes[c].set_title(f"{sp}\n(cls {lab})", fontsize=9)
        axes[c].axis("off")
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(in9, mo, seg):
    mo.md(
        f"""
        ## Verdict

        - **LeVLJEPA's own numbers** (ADE20K {seg['levljepa']['mIoU']:.2f} mIoU; IN-9
          drops {in9['levljepa']['drop_ms']:.2f}/{in9['levljepa']['drop_mr']:.2f})
          track the paper (23.15; 5.95/17.21) once each encoder is evaluated
          under its **native** normalization and the linear-head output is
          permuted to `(B,C,14,14)` before the loss.
        - Against public SigLIP (WebLI) and OpenAI CLIP (WIT) the dense-feature
          and robustness *advantages* reverse: those baselines are trained on
          stronger datasets than Datacomp-L. The paper's matched-data baselines
          are not public, so the objective-level comparison needs same-data
          checkpoints. Full-scale numbers: `bash run_repro.sh` in the repo.
        """
    )
    return


if __name__ == "__main__":
    app.run()
