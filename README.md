# LeVLJEPA

**LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives**

Official implementation of **LeVLJEPA**, the first fully non-contrastive,
end-to-end method for vision-language pretraining. LeVLJEPA aligns image and
text through cross-modal prediction with stop-gradient targets and per-modality
distributional regularization (SIGReg) — with **no negatives, no temperature, no
momentum encoder, and no teacher-student schedule**.

[Project page](https://levljepa.github.io) ·
[Checkpoints](https://huggingface.co/lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k)

## Overview

Vision-language pretraining remains dominated by contrastive objectives (CLIP,
SigLIP), which rely on negative pairs and large batches. Vision-only
self-supervised learning, by contrast, has largely moved to non-contrastive
joint-embedding prediction. LeVLJEPA brings that paradigm to paired
image-caption data.

The central finding is that **non-contrastive pretraining yields a vision
encoder with substantially stronger dense semantic features than contrastive
pretraining**. LeVLJEPA trades a small amount of zero-shot retrieval accuracy
for markedly better dense prediction (semantic segmentation), background
robustness, and — most notably — performance as a *frozen vision backbone for
multimodal LLMs*.

## Method

An image encoder (ViT) and a text encoder (GPT-2) are trained with two
components:

1. **Cross-modal prediction.** A modality-specific MLP predictor maps the image
   embedding onto the stop-gradient text embedding, and another maps the text
   embedding onto the stop-gradient image embedding. The predictions are matched
   to their targets with an MSE (or BYOL-style cosine) loss.
2. **SIGReg regularization.** Each modality's marginal embedding distribution is
   independently regularized toward an isotropic Gaussian using random
   one-dimensional projections and a characteristic-function normality test.
   This prevents representation collapse without negative pairs.

The total objective is

```
loss = (1 - λv - λt) · prediction  +  λv · SIGReg(vision)  +  λt · SIGReg(text)
```

where `λv = lambda_vision` and `λt = lambda_text`.

## Framework

Training is built on
[stable-pretraining](https://github.com/rbalestr-lab/stable-pretraining) and
PyTorch Lightning. A single entry point (`main.py`) drives training; Lightning
handles distributed training, mixed precision and SyncBatchNorm.

Repository layout:

- `main.py` — Hydra entry point; builds the dataset, `spt.Module`, `Trainer` and runs `spt.Manager`
- `forwards.py` — the LeVLJEPA loss bound to the module
- `callbacks.py` — training-metric logging, periodic ImageNet zero-shot eval, online attentive probe, gradient clipping, checkpoint sync
- `configs/` — Hydra configs (`levljepa.yaml` and the `model/{tiny,small,base}` sizes)
- `utils/` — SIGReg, the CC12M/DataComp Lance dataset, attentive-probe heads, ImageNet evaluation helpers, and text tokenization
- `scripts/` — dataset builder and standalone evaluation scripts
- `slurm/` — example SLURM launchers
- `push_to_hub.py` — convert and upload a trained checkpoint to the HuggingFace Hub

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) for dependency
management.

```bash
# Install uv if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Data

The released ViT-B checkpoint is trained on
[DataComp](https://www.datacomp.ai/). The training pipeline reads any paired
image-caption corpus materialized as a [Lance](https://lancedb.github.io/lance/)
dataset, and a builder for
[CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds) is included:

```bash
python scripts/build_lance_cc12m.py --output ./data/cc12m/train.lance
```

Zero-shot evaluation uses
[ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), streamed from
HuggingFace and cached under `cache_dir`. Point `lance_path` at your built
dataset and set `lance_image_column` / `lance_text_column` accordingly.

## Usage

Edit the paths in `configs/levljepa.yaml` before launching:

- `lance_path`: the Lance training dataset
- `output_dir`: where checkpoints are written
- `cache_dir`: local dataset/cache directory
- `hf_bucket`: optional HuggingFace checkpoint upload target

Launch training:

```bash
python main.py
```

Multi-GPU training is handled by Lightning — set `devices` instead of launching
with `torchrun`:

```bash
python main.py devices=8 batch_size=256
```

Override config values from the command line:

```bash
python main.py run_name=my_run batch_size=256 model=small total_steps=200000
```

On a SLURM cluster, use the example launchers (one task per GPU; scale by
raising `--nodes`):

```bash
sbatch slurm/train.slurm
TOTAL_STEPS=200000 sbatch --nodes=2 slurm/train.slurm
```

## Configuration

All hyperparameters are managed with [Hydra](https://hydra.cc/). The single
training config is `configs/levljepa.yaml`. Model size is selected with
`model=tiny`, `model=small`, or `model=base` (default `base`).

Key hyperparameters:

| Hyperparameter | Description |
|---------------|-------------|
| `lambda_vision` | SIGReg weight for vision embeddings |
| `lambda_text` | SIGReg weight for text embeddings |
| `align_loss` | `mse` (default) or `cosine` (BYOL-style normalized) prediction loss |
| `pre_proj_width` / `pre_proj_depth` | Pre-projection MLP hidden width / number of hidden layers |
| `pre_proj_hidden_dims` | Optional explicit list of pre-projection hidden widths, e.g. `[4096, 2048]` |
| `projector_width` / `projector_depth` | Cross-modal predictor MLP hidden width / number of hidden layers |
| `projector_hidden_dims` | Optional explicit list of predictor hidden widths |
| `predictor_dropout` | Dropout inside the predictor heads |
| `lr_schedule` | `cosine` or `wsd` (warmup-stable-decay) |
| `text_readout` | `eot` (end-of-text token) or `pad77` (last position) |
| `online_attentive_probe` | Train an online ImageNet attentive probe during pretraining |
| `devices` | Number of GPUs per node (Lightning) |

## Results

LeVLJEPA, InfoNCE and SigLIP below all use the same ViT-B encoder, trained and
evaluated under identical protocols. **Higher is better** unless noted.

**Frozen vision backbone for multimodal LLMs** (accuracy gain over a no-vision
baseline; lightweight MLP bridge trained, ViT and LLM frozen):

| Backbone | Benchmark | LeVLJEPA | InfoNCE | SigLIP |
|---|---|---|---|---|
| Llama-1B  | GQA   | **+8.2**  | +6.3 | +6.0 |
| Llama-1B  | VQAv2 | **+11.0** | +8.4 | +6.0 |
| Llama-1B  | POPE  | **+17.3** | +16.2 | +12.4 |
| Qwen-1.5B | GQA   | **+6.7**  | +5.2 | +4.6 |
| Qwen-1.5B | VQAv2 | **+10.5** | +5.8 | +4.1 |
| Qwen-1.5B | POPE  | **+22.6** | +19.1 | +18.0 |

**Semantic segmentation** (linear head on frozen patch tokens, mIoU):

| Dataset | LeVLJEPA | InfoNCE | SigLIP |
|---|---|---|---|
| ADE20K     | **23.15** | 20.90 | 19.24 |
| COCO-Stuff | **31.10** | 29.02 | 28.88 |

**Background robustness** (accuracy drop, **lower is better**):

| Split | LeVLJEPA | InfoNCE | SigLIP |
|---|---|---|---|
| Mixed-Same | **5.95**  | 6.57  | 7.03 |
| Mixed-Rand | **17.21** | 18.67 | 18.09 |

**Linear probing** (top-1):

| Dataset | LeVLJEPA | InfoNCE | SigLIP |
|---|---|---|---|
| ImageNet   | 65.42 | 65.75 | **66.34** |
| Places365  | 36.07 | **37.11** | 36.81 |
| Aircraft   | 46.38 | 44.10 | **47.46** |
| Pets       | 81.28 | **82.86** | 82.64 |

**Zero-shot classification** (top-1) — the one axis where contrastive
objectives, optimized directly for retrieval, remain ahead:

| Dataset | LeVLJEPA | InfoNCE | SigLIP |
|---|---|---|---|
| ImageNet   | 42.45 | 47.32 | **50.78** |
| Places365  | 29.97 | 34.46 | **33.76** |
| Aircraft   | 7.65  | 8.10  | **10.62** |
| Pets       | 59.63 | 68.98 | **77.27** |

See the [project page](https://levljepa.github.io) and paper for the full set of
experiments and protocols.

## Checkpoints

A pretrained ViT-B checkpoint is available on the HuggingFace Hub:
[`lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k`](https://huggingface.co/lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k).

`push_to_hub.py` converts a trained `*_vision_step*.pt` / `*_text_step*.pt`
checkpoint pair to safetensors, writes a config and model card, and uploads it:

```bash
python push_to_hub.py \
    --vision_ckpt run_vision_step200000.pt \
    --text_ckpt run_text_step200000.pt \
    --repo_id your-hf-org/LeVLJEPA-ViT-B
```

## Evaluation

Training periodically runs ImageNet zero-shot evaluation through the
`ImageNetEval` callback and an online attentive probe through
`OnlineAttentiveProbe`, both logged alongside training metrics.

Standalone evaluation scripts:

- `scripts/eval_zeroshot_classification.py` — zero-shot classification on ImageNet, Places365, FGVC-Aircraft and Oxford-IIIT Pets
- `scripts/probe_levljepa_attentive.py` — distributed attentive-pooling probe on a frozen vision encoder

```bash
python scripts/eval_zeroshot_classification.py \
    --vision-ckpt run_vision_step200000.pt \
    --text-ckpt run_text_step200000.pt \
    --embed-dim 768
```

The dense-prediction (segmentation), background-robustness and frozen-VLM-backbone
protocols are described in the paper.

## Authors

Lukas Kuhn¹²³, Giuseppe Serra¹, Randall Balestriero⁴\*, Florian Buettner¹²³\*

¹ German Cancer Research Center (DKFZ) · ² German Cancer Consortium (DKTK) ·
³ Goethe University Frankfurt · ⁴ Brown University · \*Joint last authors

## Citation

```bibtex
@article{kuhn2026levljepa,
  title   = {LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives},
  author  = {Kuhn, Lukas and Serra, Giuseppe and Balestriero, Randall and Buettner, Florian},
  year    = {2026}
}
```
