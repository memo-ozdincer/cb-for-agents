#!/bin/bash
#SBATCH -p ml
#SBATCH -A ml
#SBATCH -q ml
#SBATCH --job-name=cb_llama4_4xh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # TODO: Check if needs gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Training - CSLab ML Cluster 4×H100
# =============================================================================
#
# Trains Llama-4-Scout-17B-16E with Circuit Breakers using 4 H100 GPUs
# on the CSLab ML cluster (concerto node).
#
# Submit:
#   cd /path/to/harmful-agents-meta-dataset
#   sbatch slurm/cslab_cb_llama4_4xh100.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs

# Prefer envs on /mfs1 (fast scratch on CSLab)
source "/mfs1/u/$USER/.venvs/cb_env/bin/activate"

export CACHE_ROOT="/mfs1/u/$USER/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

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
    --output-dir outputs/cb_llama4_scout
