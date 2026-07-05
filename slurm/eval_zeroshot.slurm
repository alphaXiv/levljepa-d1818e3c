#!/bin/bash
# Single-GPU zero-shot classification eval for a trained LeVLJEPA checkpoint.
#
#   VISION_CKPT=run_vision_step50000.pt TEXT_CKPT=run_text_step50000.pt \
#     sbatch slurm/eval_zeroshot.slurm
#
#SBATCH --job-name=levljepa-zeroshot
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

VISION_CKPT=${VISION_CKPT:?set VISION_CKPT to the *_vision_step*.pt checkpoint}
TEXT_CKPT=${TEXT_CKPT:?set TEXT_CKPT to the *_text_step*.pt checkpoint}
LABEL=${LABEL:-levljepa}
EMBED_DIM=${EMBED_DIM:-768}
VIT=${VIT:-vit_base_patch16_224}
DATASETS=${DATASETS:-imagenet,places365,aircraft,pets}
export CACHE_DIR=${CACHE_DIR:-./data}

srun uv run python scripts/eval_zeroshot_classification.py \
    --vision-ckpt "${VISION_CKPT}" \
    --text-ckpt "${TEXT_CKPT}" \
    --label "${LABEL}" \
    --embed-dim "${EMBED_DIM}" \
    --vit "${VIT}" \
    --datasets "${DATASETS}" \
    "$@"
