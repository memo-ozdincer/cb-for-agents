#!/bin/bash
#SBATCH --job-name=cb_llama31_8b_1xh100_debug_fir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=/scratch/memoozd/harmful-agents-meta-dataset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Circuit Breakers Training - Fir 1×H100 Debug (Llama-3.1-8B-Instruct)
# =============================================================================
#
# Single-GPU version for testing/debugging. Lower batch size and fewer steps.
# Use fir_cb_llama31_8b_4xh100.sbatch for full training.
#
# Submit from your SCRATCH clone:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/fir_cb_llama31_8b_1xh100_debug.sbatch
#
# Note: HF_TOKEN and WANDB_API_KEY are hardcoded for memoozd
#
# =============================================================================

set -euo pipefail

SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$SCRATCH_DIR/harmful-agents-meta-dataset"
VENV_DIR="$SCRATCH_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p logs

module purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run once: sbatch slurm/setup_fir.sbatch"
  exit 1
fi

source "$VENV_DIR/bin/activate"

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# W&B defaults for HPC (disabled for debug run by default)
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-circuit-breakers}"

# =============================================================================
# PyTorch Memory Optimization
# =============================================================================
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "CB Training - DEBUG (Fir 1×H100)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPUs: 1 x H100 SXM 80GB"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Steps: 10 (debug)"
echo "Output: $RUN_DIR/outputs/cb_llama31_8b_debug"
echo "========================================"

# Run with single GPU, reduced batch size, minimal steps for testing
python scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-instruct \
  --loss-weighting dual \
  --total-steps 10 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --no-wandb \
  --output-dir "$RUN_DIR/outputs/cb_llama31_8b_debug"

echo "========================================"
echo "Debug training completed!"
echo "========================================"
