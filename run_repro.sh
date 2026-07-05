#!/usr/bin/env bash
set -euo pipefail

# Reproduction entrypoint: runs the frozen-encoder evaluations for LeVLJEPA vs
# SigLIP (ADE20K linear segmentation + ImageNet-9 background robustness) and
# writes EVAL.md + .openresearch/artifacts/results.json. Run on a GPU instance.
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_TORCH_BACKEND=auto

uv run scripts/repro_eval.py
