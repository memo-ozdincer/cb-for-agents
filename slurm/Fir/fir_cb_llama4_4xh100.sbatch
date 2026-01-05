#!/bin/bash
#SBATCH --job-name=cb_llama4_4xh100_fir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=06:00:00
#SBATCH --chdir=/scratch/memoozd/harmful-agents-meta-dataset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing 

# =============================================================================
# Circuit Breakers Training - Fir 4×H100
# =============================================================================
#
# Submit from your SCRATCH clone:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/fir_cb_llama4_4xh100.sbatch
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

# W&B defaults for HPC. Override when submitting if desired.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-circuit-breakers}"
export WANDB_GROUP="${WANDB_GROUP:-fir_cb_4xh100}"
export WANDB_TAGS="${WANDB_TAGS:-fir,cb,4xh100,llama4}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port $MASTER_PORT \
  scripts/train_circuit_breaker.py \
    --preset llama-4-scout \
    --loss-weighting dual \
    --total-steps 300 \
    --output-dir "$RUN_DIR/outputs/cb_llama4_scout"
