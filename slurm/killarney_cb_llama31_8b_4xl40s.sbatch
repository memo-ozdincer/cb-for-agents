#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --job-name=cb_llama31_8b_4xl40s_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Training - Killarney 4×L40s (Llama-3.1-8B-Instruct)
# =============================================================================
#
# Hardware: 4x NVIDIA L40S (48GB each = 192GB total VRAM)
# Full training (150 steps) on 4 GPUs with distributed training
#
# Submit from your project directory:
#   cd /project/6105522/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/killarney_cb_llama31_8b_4xl40s.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
SCRATCH_DIR="$HOME/scratch"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

# Use --force to unload all modules deterministically
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Create it once with:"
  echo "  module load StdEnv/2023 python/3.11.5"
  echo "  python -m venv $VENV_DIR"
  echo "  source $VENV_DIR/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Preflight: fail early if key packages are missing
# =============================================================================
python - << 'PY'
import sys

def check(mod: str) -> None:
    try:
        __import__(mod)
    except Exception as e:
        print(f"ERROR: failed to import '{mod}': {e}")
        sys.exit(1)

for m in ("torch", "transformers", "peft", "accelerate"):
    check(m)

import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY

# =============================================================================
# API Tokens (export these from login node before submitting)
# =============================================================================
# Before running sbatch, export on login node:
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...
# Job will inherit these from the environment.

# =============================================================================
# CPU Threading Configuration (per GPU)
# =============================================================================
# With 32 CPUs and 4 GPUs, each GPU gets ~8 CPU threads
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# =============================================================================
# Storage and Cache Setup
# =============================================================================
# Use scratch for outputs (temporary) and project for cache (persistent)
RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# =============================================================================
# W&B Configuration
# =============================================================================
export WANDB_MODE=online
export WANDB_PROJECT=circuit-breakers
export WANDB_RUN_GROUP=killarney-cb-4xl40s-full
export WANDB_TAGS=killarney,cb,4xl40s,llama31,full-training

# =============================================================================
# Distributed Training Setup (Multi-GPU)
# =============================================================================
export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

# =============================================================================
# PyTorch Memory Optimization
# =============================================================================
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Circuit Breaker Training - Killarney"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Account: aip-rgrosse"
echo "GPUs: 4 x L40S 48GB (192GB total VRAM)"
echo "CPUs: 32 (8 per GPU)"
echo "Memory: 128GB"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Total Steps: 150 (full training)"
echo "Output: $RUN_DIR/outputs/cb_llama31_8b_instruct"
echo "========================================"

# =============================================================================
# Multi-GPU Training with Accelerate
# =============================================================================
# With 4 GPUs, use lower gradient accumulation (2 instead of 8)
# Effective batch size = 4 GPUs × 4 batch × 2 accumulation = 32
accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --use_deepspeed=false \
  scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-instruct \
  --loss-weighting dual \
  --total-steps 150 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --output-dir "$RUN_DIR/outputs/cb_llama31_8b_instruct"

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"
