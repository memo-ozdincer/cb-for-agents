#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=cb_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# DEBUG SCRIPT - 1 L40S, 3 training steps, catches real code errors
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv check ==="
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"
echo "Which:  $(which python)"

echo "=== Quick package check ==="
python -c 'import torch, transformers, peft; print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())'

echo "=== Environment setup ==="
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"

SCRATCH_DIR="$HOME/scratch"
RUN_DIR="$SCRATCH_DIR/cb_debug/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export WANDB_MODE=online
export WANDB_PROJECT=circuit-breakers-debug

echo "=== RUNNING ACTUAL TRAINING (3 steps) ==="
python scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-instruct \
  --loss-weighting dual \
  --total-steps 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --output-dir "$RUN_DIR/debug_output"

echo "=== Debug complete ==="
