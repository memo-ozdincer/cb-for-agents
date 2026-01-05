#!/bin/bash
#SBATCH --job-name=cb_llama4_4xh100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=def-zhijing  # TODO: Set your Trillium RAP (def-/rrg-/rpp-)

# =============================================================================
# Circuit Breakers Training - Trillium 4×H100
# =============================================================================
#
# Trains Llama-4-Scout-17B-16E with Circuit Breakers using 4 H100 GPUs.
#
# Submit from $SCRATCH:
#   cd $SCRATCH/harmful-agents-meta-dataset
#   sbatch slurm/trillium_cb_llama4_4xh100.sbatch
#
# =============================================================================

set -euo pipefail

# Ensure outputs land on SCRATCH (Trillium requirement)
cd "$SCRATCH"
RUN_DIR="$SCRATCH/cb_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{logs,outputs}
cd "$SLURM_SUBMIT_DIR"  # code can live in $PROJECT/$HOME (read-only on compute)

# Load Trillium modules
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

# Activate venv built in scratch
source "$SCRATCH/.venvs/cb_env/bin/activate"

# Caches on scratch
export CACHE_ROOT="$SCRATCH/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# NCCL for multi-GPU
export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

# Redirect outputs to scratch
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port $MASTER_PORT \
  scripts/train_circuit_breaker.py \
    --preset llama-4-scout \
    --loss-weighting dual \
    --total-steps 300 \
    --output-dir "$RUN_DIR/outputs/cb_llama4_scout"
