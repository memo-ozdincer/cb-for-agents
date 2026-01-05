#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128GB
#SBATCH --job-name=cb_llama31_8b_2xh100_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Training - Killarney 2×H100 (Llama-3.1-8B-Instruct)
# =============================================================================
#
# Full training (150 steps) on 2×H100 GPUs (160GB total VRAM)
# Distributed training with model sharding
#
# Submit from your project directory:
#   cd /project/6105522/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/killarney_cb_llama31_8b_4xh100.sbatch
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
module load git-lfs/3.3.0 || echo "git-lfs module not available"

# Ensure Git LFS files are pulled (training data is stored via LFS)
echo "=== Git LFS check ==="
if command -v git-lfs &> /dev/null || git lfs version &> /dev/null; then
  git lfs pull --include="data/circuit_breakers/**/*.jsonl" || echo "Warning: git lfs pull had issues"
else
  echo "WARNING: git-lfs not available"
fi

# Validate training data
DATA_FILE="$REPO_DIR/data/circuit_breakers/cb_training_batches.jsonl"
if [[ -f "$DATA_FILE" ]]; then
  FIRST_CHAR=$(head -c 1 "$DATA_FILE")
  if [[ "$FIRST_CHAR" != "{" ]]; then
    echo "ERROR: $DATA_FILE is a Git LFS pointer, not actual data!"
    echo "Run 'git lfs pull' to download the file."
    exit 1
  fi
  echo "Training data validated: $(wc -l < "$DATA_FILE") lines"
else
  echo "ERROR: Training data not found at $DATA_FILE"
  exit 1
fi

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

for m in ("torch", "transformers", "peft"):
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
# CPU Threading Configuration
# =============================================================================
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

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
export WANDB_RUN_GROUP=killarney-cb-1xh100-full
export WANDB_TAGS=killarney,cb,1xh100,llama31,full-training

# =============================================================================
# Distributed Training Setup
# =============================================================================
export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500

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
echo "GPUs: 1 x H100 SXM 80GB (4x nodes down)"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Total Steps: 150 (full training)"
echo "Output: $RUN_DIR/outputs/cb_llama31_8b_instruct"
echo "========================================"

# Single GPU - no accelerate needed, use higher gradient accumulation
python scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-instruct \
  --loss-weighting dual \
  --total-steps 150 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --output-dir "$RUN_DIR/outputs/cb_llama31_8b_instruct"

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"
