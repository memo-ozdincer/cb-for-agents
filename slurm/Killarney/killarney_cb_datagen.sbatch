#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --job-name=cb_datagen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# CB Data Generation - Killarney 4×L40S (Tensor Parallel)
# =============================================================================
#
# Generates Circuit Breaker training data using abliterated model
# Uses vLLM with tensor_parallel=4 for high-throughput batched inference
#
# Hardware: 4x NVIDIA L40S (48GB VRAM each = 192GB total)
# Estimated time: ~1-2 hours for full dataset with TP=4
#
# Submit:
#   cd /project/6105522/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/Killarney/killarney_cb_datagen.sbatch
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
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Preflight
# =============================================================================
python - << 'PY'
import sys

def check(mod: str) -> None:
    try:
        __import__(mod)
    except Exception as e:
        print(f"ERROR: failed to import '{mod}': {e}")
        sys.exit(1)

for m in ("torch", "vllm"):
    check(m)

import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "CB Data Generation - Killarney"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 4 x L40S 48GB (tensor_parallel=4)"
echo "Model: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
echo "========================================"

# =============================================================================
# Run Data Generation Pipeline
# =============================================================================

# IMPORTANT: Use abliterated model so it actually generates harmful completions
# Standard Llama-3.1-Instruct will refuse!

# Using tensor_parallel=4 to shard model across 4 GPUs for faster inference
python scripts/cb_data_generation/run_pipeline.py \
    --all \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --backend vllm \
    --tensor-parallel 4 \
    --temperature 0.7 \
    --num-samples 3 \
    --output-dir data/circuit_breakers/

echo "========================================"
echo "Data generation complete!"
echo "========================================"
echo "Outputs:"
echo "  Ds: data/circuit_breakers/ds/circuit_breaker_set.jsonl"
echo "  Dr: data/circuit_breakers/dr/retain_set.jsonl"
echo "  Eval: data/circuit_breakers/eval/eval_set.jsonl"
echo "  Report: data/circuit_breakers/quality_report.json"
echo "========================================""
