#!/bin/bash
# Generic LeVLJEPA training launcher (single- or multi-node).
#
# One task per GPU is launched by srun; Lightning's SLURM integration maps each
# task to a rank, so we pass devices=<gpus-per-node> and num_nodes=<nodes>.
# Scale to more nodes by raising --nodes below (keep --ntasks-per-node == #GPUs).
#
#   sbatch slurm/train.slurm
#   TOTAL_STEPS=312500 BATCH_SIZE=256 sbatch --nodes=2 slurm/train.slurm
#
#SBATCH --job-name=levljepa
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

# Multi-node InfiniBand / GPUDirect RDMA (no-op on a single node).
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

# ---- Overridable run settings (export before sbatch, or edit defaults) -------
RUN_NAME=${RUN_NAME:-levljepa_${SLURM_JOB_ID}}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-2e-4}
WARMUP=${WARMUP:-1000}
EVAL_EVERY=${EVAL_EVERY:-5000}
SAVE_EVERY=${SAVE_EVERY:-5000}

LANCE_PATH=${LANCE_PATH:-/path/to/cc12m/train.lance}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
CACHE_DIR=${CACHE_DIR:-./data}
HF_BUCKET=${HF_BUCKET:-your-hf-org/your-bucket}
WANDB_PROJECT=${WANDB_PROJECT:-LeVLJEPA}
WANDB_ENABLED=${WANDB_ENABLED:-true}

srun uv run python main.py \
    devices="${SLURM_NTASKS_PER_NODE}" \
    num_nodes="${SLURM_NNODES}" \
    batch_size="${BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    total_steps="${TOTAL_STEPS}" \
    lr="${LR}" \
    warmup_steps="${WARMUP}" \
    run_name="${RUN_NAME}" \
    wandb_project="${WANDB_PROJECT}" \
    wandb_enabled="${WANDB_ENABLED}" \
    lance_path="${LANCE_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    cache_dir="${CACHE_DIR}" \
    hf_bucket="${HF_BUCKET}" \
    eval_every_n_steps="${EVAL_EVERY}" \
    save_every_n_steps="${SAVE_EVERY}" \
    "$@"
