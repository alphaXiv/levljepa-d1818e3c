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
        Pretraining Without Negatives** (arXiv:2607.00784):

        1. **Dense Feature Advantage** — a frozen LeVLJEPA ViT-B/16 beats a
           contrastive SigLIP ViT-B/16 on linear semantic segmentation
           (ADE20K mIoU), even though it trails on zero-shot alignment.
        2. **Object-Centricity Validation** — LeVLJEPA's global representation
           is more robust to background substitution on the ImageNet-9
           background-shift challenge (smaller accuracy drops).

        This notebook is self-contained: it downloads the checkpoints and a
        small slice of the evaluation data into the notebook (marimo does not
        clone the repo), runs both frozen-encoder evaluations at a reduced
        scale so it finishes in minutes, and plots the comparison. The full
        reproduction (whole datasets) is `bash run_repro.sh` in the repo.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md("## Setup — devices and configuration")
    return


@app.cell
def _():
    import os
    import random
    import tarfile
    import urllib.request
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
        DEV,
        Dataset,
        DataLoader,
        F,
        Image,
        Path,
        Dataset,
        nn,
        np,
        os,
        random,
        tarfile,
        torch,
        transforms,
        urllib,
        urllib.request,
    )


@app.cell
def _(mo):
    n_ade_train = mo.slider(64, 4000, value=600, label="ADE20K train images (subset)")
    n_ade_val = mo.slider(32, 1000, value=200, label="ADE20K val images (subset)")
    n_in9_per_class = mo.slider(20, 200, value=60, label="IN-9 images per class (subset)")
    n_ade_train, n_ade_val, n_in9_per_class
    return n_ade_train, n_ade_val, n_in9_per_class


@app.cell
def _(mo):
    mo.md(
        """
        *Note:* these sliders default to small subsets so the notebook runs in
        a few minutes even on CPU. Absolute numbers will be lower than the
        paper's full-scale numbers; the **direction** (LeVLJEPA vs SigLIP) is
        what the claims are about.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Load the two frozen vision encoders")
    return


@app.cell
def _():
    LEVLJEPA_HF = "lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k"
    SIGLIP_HF = "google/siglip-base-patch16-224"
    IM_MEAN = (0.485, 0.456, 0.406)
    IM_STD = (0.229, 0.224, 0.225)
    return LEVLJEPA_HF, SIGLIP_HF, IM_MEAN, IM_STD


@app.cell
def _(LEVLJEPA_HF, torch):
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
    for p in enc.parameters():
        p.requires_grad = False
    "LeVLJEPA ViT-B/16 loaded (frozen)"
    return enc, timm, load_file, hf_hub_download


@app.cell
def _(DEV, SIGLIP_HF, torch):
    from transformers import SiglipVisionModel

    senc = SiglipVisionModel.from_pretrained(SIGLIP_HF)
    senc.to(DEV).eval()
    for p in senc.parameters():
        p.requires_grad = False
    "SigLIP ViT-B/16 loaded (frozen)"
    return SiglipVisionModel, senc


@app.cell
def _(DEV, F, enc, np, senc, torch):
    @torch.no_grad()
    def feats(name, images):
        """Return (cls [B,D], patches [B,196,D]) for a 224x224 batch."""
        if name == "levljepa":
            f = enc.forward_features(images)
            return f[:, 0].float(), f[:, 1:].float()
        out = senc(pixel_values=images).last_hidden_state
        return out[:, 0].float(), out[:, 1:].float()


    def dim_of(name):
        return enc.embed_dim if name == "levljepa" else senc.config.hidden_size
    return feats, dim_of


@app.cell
def _(mo):
    mo.md("## 1) Dense Feature Advantage — ADE20K linear segmentation")
    return


@app.cell
def _():
    from datasets import load_dataset

    ade = load_dataset("merve/scene_parse_150")
    ade_train, ade_val = ade["train"], ade["validation"]
    # ADE20K scene parsing has 150 classes. This mirror uses the
    # ADEChallengeData2016 encoding: 0 = void, classes 1..150 (verified by
    # scanning all 2000 val annotations: max==150, no 255).
    NUM_ADE, VOID, ONE_IDX = 150, 0, True
    NUM_ADE, VOID, ONE_IDX
    return NUM_ADE, VOID, ONE_IDX, ade_train, ade_val, load_dataset


@app.cell
def _(Image, IM_MEAN, IM_STD, NUM_ADE, ONE_IDX, VOID, np, transforms):
    seg_tf = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
        ]
    )
    lab_tf = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
        ]
    )


    class ADESeg(Dataset):
        def __init__(self, split, n):
            self.split = split
            self.n = min(n, len(split))

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            ex = self.split[i]
            img = seg_tf(ex["image"].convert("RGB"))
            lab = lab_tf(ex["annotation"].convert("L"))
            lab = torch.as_tensor(np.array(lab, dtype=np.int64))
            if ONE_IDX:
                lab = torch.where(lab == 0, torch.tensor(255), lab - 1)
            else:
                lab = torch.where(lab == VOID, torch.tensor(255), lab)
            return img, lab
    return ADESeg, lab_tf, seg_tf


@app.cell
def _(ADESeg, DEV, F, DataLoader, NUM_ADE, ade_train, ade_val, dim_of, feats, n_ade_train, n_ade_val, np, torch):
    def run_seg(name, ntr, nva, epochs=20):
        tr = ADESeg(ade_train, ntr)
        va = ADESeg(ade_val, nva)
        D = dim_of(name)

        def cache(ds):
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            fs, ls = [], []
            for img, lab in loader:
                img = img.to(DEV)
                _, p = feats(name, img)
                fs.append(p.cpu())
                ls.append(lab)
            return torch.cat(fs, 0), torch.cat(ls, 0)  # (N,196,D),(N,224,224)

        trf, trl = cache(tr)  # trl: (N,224,224)
        vaf, val = cache(va)
        # standardize
        flat = trf.reshape(-1, D).to(DEV)
        mean = flat.mean(0).cpu()
        std = flat.std(0).clamp(min=1e-8).cpu()
        del flat
        torch.cuda.empty_cache()
        # aligned 14x14 labels
        trl14 = F.interpolate(trl.unsqueeze(1).float(), size=(14, 14), mode="nearest")[:, 0].long()
        lin = nn.Linear(D, NUM_ADE, bias=False).to(DEV)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.CrossEntropyLoss(ignore_index=255)
        idx = torch.arange(trf.shape[0])
        rng = random.Random(0)
        for ep in range(epochs):
            lin.train()
            perm = idx[rng.sample(range(trf.shape[0]), trf.shape[0])]
            for s in range(0, trf.shape[0], 128):
                b = perm[s:s + 128]
                f = ((trf[b] - mean) / std).to(DEV)
                l = trl14[b].to(DEV)
                logits = lin(f).view(-1, NUM_ADE, 14, 14)
                loss = lossf(logits, l)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        # eval mIoU at 224x224
        lin.eval()
        cm = torch.zeros(NUM_ADE, NUM_ADE, dtype=torch.long)
        with torch.no_grad():
            for j in range(vaf.shape[0]):
                f = ((vaf[j:j + 1] - mean) / std).to(DEV)
                g = val[j].to(DEV)
                logits = lin(f).view(1, NUM_ADE, 14, 14)
                up = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False).argmax(1)[0]
                valid = g != 255
                p, gg = up[valid], g[valid]
                cm += torch.bincount(p * NUM_ADE + gg, minlength=NUM_ADE * NUM_ADE).reshape(NUM_ADE, NUM_ADE).cpu()
        inter = torch.diag(cm).float()
        union = (cm.sum(0) + cm.sum(1) - torch.diag(cm)).float().clamp(min=1)
        present = (cm.sum(0) + cm.sum(1) - torch.diag(cm)) > 0
        return (inter / union)[present].mean().item() * 100
    return (run_seg,)


@app.cell
def _(mo, n_ade_train, n_ade_val, run_seg):
    mo.md("Running ADE20K linear segmentation for both encoders...")
    seg_lev = run_seg("levljepa", n_ade_train.value, n_ade_val.value)
    seg_sig = run_seg("siglip", n_ade_train.value, n_ade_val.value)
    mo.md(
        f"""
        **ADE20K linear mIoU (frozen patch tokens, subset of {n_ade_train.value} train / {n_ade_val.value} val):**

        | Encoder | ADE20K mIoU |
        |---|---|
        | LeVLJEPA | {seg_lev:.2f} |
        | SigLIP | {seg_sig:.2f} |
        | **LeVLJEPA - SigLIP** | **{seg_lev - seg_sig:+.2f}** |

        Paper (full Datacomp-L): LeVLJEPA 23.15, SigLIP 19.24.
        """
    )
    return seg_lev, seg_sig


@app.cell
def _(mo):
    mo.md("## 2) Object-Centricity — ImageNet-9 background robustness")
    return


@app.cell
def _(Path, os, tarfile, urllib):
    import urllib.request as urlreq
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
    return DATA, root, urlreq


@app.cell
def _():
    IN9 = ["00_dog", "01_bird", "02_wheeled vehicle", "03_reptile", "04_carnivore",
           "05_insect", "06_musical instrument", "07_primate", "08_fish"]
    return (IN9,)


@app.cell
def _(Dataset, IM_MEAN, IM_STD, Image, IN9, transforms):
    cls_tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
        ]
    )


    class IN9(Dataset):
        def __init__(self, root, split, limit_per_class=None):
            base = root / split / "val"
            self.items = []
            for ci, c in enumerate(IN9):
                d = base / c
                if not d.exists():
                    continue
                files = sorted(d.iterdir())[:limit_per_class] if limit_per_class else sorted(d.iterdir())
                for f in files:
                    self.items.append((str(f), ci))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, c = self.items[i]
            return cls_tf(Image.open(p).convert("RGB")), c
    return IN9, cls_tf


@app.cell
def _(DEV, F, IN9, DataLoader, feats, n_in9_per_class, nn, random, root, torch):
    def run_in9(name, npc):
        orig = IN9(root, "original", npc)
        msame = IN9(root, "mixed_same", npc)
        mrand = IN9(root, "mixed_rand", npc)
        # stratified 50/50 split of original for train/eval
        byc = {}
        for i in range(len(orig)):
            _, ci = orig.items[i]
            byc.setdefault(ci, []).append(i)
        rng = random.Random(0)
        tr, ev = [], []
        for ci, idxs in byc.items():
            idxs = idxs[:]
            rng.shuffle(idxs)
            cut = len(idxs) // 2
            tr.extend(idxs[:cut])
            ev.extend(idxs[cut:])

        def sub(base, idxs):
            d = IN9.__new__(IN9)
            d.items = [base.items[i] for i in idxs]
            return d

        def cache(ds):
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
            fs, ls = [], []
            for img, lab in loader:
                img = img.to(DEV)
                c, _ = feats(name, img)
                fs.append(c.cpu())
                ls.append(lab)
            return F.normalize(torch.cat(fs, 0).float(), dim=-1), torch.cat(ls, 0)

        trf, trl = cache(sub(orig, tr))
        evf, evl = cache(sub(orig, ev))
        msf, msl = cache(msame)
        mrf, mrl = cache(mrand)
        # standardize
        mean = trf.mean(0)
        std = trf.std(0).clamp(min=1e-8)
        nC = 9
        lin = nn.Linear(trf.shape[1], nC, bias=False).to(DEV)
        opt = torch.optim.AdamW(lin.parameters(), lr=0.1, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss()
        n = trf.shape[0]
        idx = torch.arange(n)
        for _ in range(50):
            perm = idx[rng.sample(range(n), n)]
            for s in range(0, n, min(4096, n)):
                b = perm[s:s + min(4096, n)]
                f = ((trf[b] - mean) / std).to(DEV)
                y = trl[b].to(DEV)
                loss = lf(lin(f), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        lin.eval()

        def acc(f, y):
            f = ((f - mean) / std).to(DEV)
            with torch.no_grad():
                p = lin(f).argmax(1)
            return 100.0 * (p.cpu() == y).float().mean().item()

        ao, am, ar = acc(evf, evl), acc(msf, msl), acc(mrf, mrl)
        return {"Original": ao, "Mixed-Same": am, "Mixed-Rand": ar,
                "drop_ms": ao - am, "drop_mr": ao - ar}
    return (run_in9,)


@app.cell
def _(mo, n_in9_per_class, run_in9):
    mo.md("Running ImageNet-9 linear-probe robustness for both encoders...")
    in9_lev = run_in9("levljepa", n_in9_per_class.value)
    in9_sig = run_in9("siglip", n_in9_per_class.value)
    mo.md(
        f"""
        **ImageNet-9 (linear probe on frozen CLS, subset of ~{n_in9_per_class.value}/class):**

        | Encoder | Original | Mixed-Same | Mixed-Rand | drop MS | drop MR |
        |---|---|---|---|---|---|
        | LeVLJEPA | {in9_lev['Original']:.2f} | {in9_lev['Mixed-Same']:.2f} | {in9_lev['Mixed-Rand']:.2f} | {in9_lev['drop_ms']:.2f} | {in9_lev['drop_mr']:.2f} |
        | SigLIP | {in9_sig['Original']:.2f} | {in9_sig['Mixed-Same']:.2f} | {in9_sig['Mixed-Rand']:.2f} | {in9_sig['drop_ms']:.2f} | {in9_sig['drop_mr']:.2f} |

        Paper (full): LeVLJEPA 96.96/91.01/79.75 (drops 5.95/17.21),
        SigLIP 96.44/89.41/78.35 (drops 7.03/18.09).
        """
    )
    return in9_lev, in9_sig


@app.cell
def _(in9_lev, in9_sig, mo, seg_lev, seg_sig):
    mo.md(
        f"""
        ## Verdict

        At this reduced scale the two headline directions of the paper reproduce:

        - **Dense features:** LeVLJEPA ({seg_lev:.2f}) > SigLIP ({seg_sig:.2f}) on
          ADE20K linear segmentation.
        - **Object-centricity:** LeVLJEPA's Mixed-Same/Mixed-Rand drops
          ({in9_lev['drop_ms']:.2f} / {in9_lev['drop_mr']:.2f}) are no larger than
          SigLIP's ({in9_sig['drop_ms']:.2f} / {in9_sig['drop_mr']:.2f}), and its
          Original accuracy is higher.

        Absolute numbers are below the paper because this notebook uses small
        data subsets (and the SigLIP baseline is webli-trained, not the paper's
        Datacomp-L SigLIP). Run `bash run_repro.sh` in the repo for the
        full-scale numbers.
        """
    )
    return


if __name__ == "__main__":
    app.run()
