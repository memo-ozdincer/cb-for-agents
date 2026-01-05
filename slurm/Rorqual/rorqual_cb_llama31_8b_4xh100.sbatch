#!/bin/bash
#SBATCH --job-name=cb_llama31_8b_4xh100_rorqual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing_gpu

# =============================================================================
# Circuit Breakers Training - Rorqual 4×H100 (Llama-3.1-8B-Instruct)
# =============================================================================
#
# Submit:
#   cd /lustre09/project/6098391/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/rorqual_cb_llama31_8b_4xh100.sbatch
#
# Notes:
# - Rorqual compute nodes have NO INTERNET ACCESS.
# - Run setup on login node first: bash slurm/setup_rorqual.sbatch
# - HF_TOKEN and WANDB_API_KEY are hardcoded for memoozd
# - GPU nodes: 64 cores, 498G memory, 4×H100 (SXM5 interconnect).
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/lustre09/project/6098391/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p logs

# On Alliance systems, some modules can remain loaded via default collections.
# Use --force to unload everything deterministically.
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run setup on LOGIN NODE first (no internet on compute):"
  echo "  bash slurm/setup_rorqual.sbatch"
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Preflight: fail early if key packages/CLIs are missing
# =============================================================================
python - << 'PY'
import sys

def check(mod: str) -> None:
  try:
    __import__(mod)
  except Exception as e:
    print(f"ERROR: failed to import '{mod}': {e}")
    sys.exit(1)

for m in ("torch", "transformers", "accelerate"):
  check(m)

import torch, transformers, accelerate
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
PY

if ! python -m accelerate.commands.launch --help >/dev/null 2>&1; then
  echo "ERROR: Accelerate is not runnable in this venv ($VENV_DIR)."
  echo "This typically means the login-node setup did not complete, or the venv is stale."
  echo "Fix on LOGIN NODE (compute has no internet):"
  echo "  cd $REPO_DIR"
  echo "  bash slurm/setup_rorqual.sbatch"
  echo "Then resubmit this job."
  exit 1
fi

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

RUN_DIR="$PROJECT_DIR/cb_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# DON'T set HF_HUB_OFFLINE=1 here; it causes Transformers' is_base_mistral check to fail.
# Instead, let the training script set local_files_only=True, which tolerates metadata checks.

# W&B defaults for HPC (offline since no internet on compute).
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-circuit-breakers}"
export WANDB_GROUP="${WANDB_GROUP:-rorqual_cb_4xh100}"
export WANDB_TAGS="${WANDB_TAGS:-rorqual,cb,4xh100,llama31}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

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
echo "Circuit Breaker Training - Rorqual"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPUs: 4 x H100 SXM5"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Output: $RUN_DIR/outputs/cb_llama31_8b_instruct"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --main_process_port $MASTER_PORT \
  scripts/train_circuit_breaker.py \
    --preset llama-3.1-8b-instruct \
    --loss-weighting dual \
    --total-steps 150 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --output-dir "$RUN_DIR/outputs/cb_llama31_8b_instruct"

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"
