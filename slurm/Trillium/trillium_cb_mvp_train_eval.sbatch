#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cb_mvp_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# CB MVP Training + Evaluation Pipeline - Trillium 1×H100
# =============================================================================
#
# Full MVP pipeline:
# 1. Pre-training sanity check
# 2. Train Circuit Breaker
# 3. Post-training sanity check
# 4. Evaluate baseline vs CB model
#
# Hardware: 1x NVIDIA H100 SXM (80GB VRAM)
#           24 CPU cores, 188 GiB RAM
#
# Prerequisites:
#   Run trillium_cb_mvp_datagen.sbatch first to generate data
#
# Usage:
#   export CB_MVP_DATA=/scratch/memoozd/cb_mvp_data/<DATAGEN_JOB_ID>
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_cb_mvp_train_eval.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p /scratch/memoozd/logs

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

# Load modules
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

# Preflight checks
python - << 'PY'
import sys
def check(mod):
    try: __import__(mod)
    except Exception as e: print(f"ERROR: {mod}: {e}"); sys.exit(1)
for m in ("torch", "transformers", "peft", "accelerate"): check(m)
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Environment
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# IMPORTANT: All outputs must go to SCRATCH (Trillium requirement)
RUN_DIR="$SCRATCH_DIR/cb_mvp_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export WANDB_MODE=online
export WANDB_PROJECT=circuit-breakers-mvp
export WANDB_RUN_GROUP=trillium-cb-mvp
export WANDB_TAGS=trillium,cb,mvp,h100,llama31

export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "========================================"
echo "CB MVP Training + Evaluation - Trillium"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "GPU: 1x H100 SXM (80GB)"
echo "Output: $RUN_DIR"
echo "========================================"

# =============================================================================
# Configuration: Data paths
# =============================================================================

# Check for CB_MVP_DATA env var (data from datagen job)
if [[ -z "${CB_MVP_DATA:-}" ]]; then
    echo ""
    echo "WARNING: CB_MVP_DATA not set!"
    echo "Searching for most recent data generation output..."

    LATEST_DATA=$(ls -td $SCRATCH_DIR/cb_mvp_data/*/ds_stage1.jsonl 2>/dev/null | head -1)
    if [[ -n "$LATEST_DATA" ]]; then
        CB_MVP_DATA=$(dirname "$LATEST_DATA")
        echo "  Found: $CB_MVP_DATA"
    else
        echo ""
        echo "ERROR: No data found! Run datagen first:"
        echo "  sbatch slurm/Trillium/trillium_cb_mvp_datagen.sbatch"
        exit 1
    fi
fi

DS_FILE="$CB_MVP_DATA/ds_stage1.jsonl"
DR_FILE="$CB_MVP_DATA/dr_stage1.jsonl"

echo ""
echo "Data files:"
echo "  Ds: $DS_FILE"
echo "  Dr: $DR_FILE"

# Verify data exists
if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found: $DS_FILE"
    exit 1
fi

if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr file not found: $DR_FILE"
    exit 1
fi

echo "  ✓ Ds samples: $(wc -l < $DS_FILE)"
echo "  ✓ Dr samples: $(wc -l < $DR_FILE)"

# =============================================================================
# STEP 1: Pre-training sanity check
# =============================================================================
echo ""
echo "Step 1: Pre-training sanity check..."
echo "========================================"

python scripts/circuit_breakers/sanity_check.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct

echo "✓ Pre-training sanity check passed!"

# =============================================================================
# STEP 2: Train Circuit Breaker
# =============================================================================
echo ""
echo "Step 2: Training Circuit Breaker..."
echo "========================================"

# Train with MVP data
# Using single GPU, so no accelerate needed, but keeping for consistency
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    scripts/train_circuit_breaker.py \
    --preset llama-3.1-8b-instruct \
    --ds-data "$DS_FILE" \
    --dr-data "$DR_FILE" \
    --loss-weighting dual \
    --total-steps 300 \
    --alpha-decay-multiplier 1.0 \
    --batch-size 1 \
    --max-seq-length 512 \
    --gradient-accumulation-steps 16 \
    --output-dir "$RUN_DIR/outputs/cb_mvp"

echo "✓ Training complete!"

# =============================================================================
# STEP 3: Post-training sanity check
# =============================================================================
echo ""
echo "Step 3: Post-training sanity check..."
echo "========================================"

FINAL_CKPT="$RUN_DIR/outputs/cb_mvp/final"

if [[ ! -d "$FINAL_CKPT" ]]; then
    echo "ERROR: Final checkpoint not found at $FINAL_CKPT"
    echo "Available checkpoints:"
    ls -la "$RUN_DIR/outputs/cb_mvp/" || echo "  None"
    exit 1
fi

python scripts/circuit_breakers/sanity_check.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter-path "$FINAL_CKPT" \
    --fail-on-error

echo "✓ Post-training sanity check passed!"

# =============================================================================
# STEP 4: Evaluate baseline vs CB model
# =============================================================================
echo ""
echo "Step 4: Evaluating baseline vs CB model..."
echo "========================================"

python scripts/circuit_breakers/eval_mvp.py \
    --baseline meta-llama/Llama-3.1-8B-Instruct \
    --cb-adapter "$FINAL_CKPT" \
    --output-dir "$RUN_DIR/eval_results" \
    --fail-on-gate

echo "✓ Evaluation complete!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "CB MVP Pipeline Complete!"
echo "========================================"
echo "Date: $(date)"
echo ""
echo "Training output:"
echo "  Checkpoint: $FINAL_CKPT"
echo ""
echo "Evaluation results:"
echo "  Directory: $RUN_DIR/eval_results"
echo ""

# Display evaluation summary if available
EVAL_SUMMARY="$RUN_DIR/eval_results/summary.json"
if [[ -f "$EVAL_SUMMARY" ]]; then
    echo "Evaluation Summary:"
    python -c "
import json
with open('$EVAL_SUMMARY') as f:
    data = json.load(f)
    for key, val in data.items():
        print(f'  {key}: {val}')
"
else
    echo "  (Summary file not found)"
fi

echo ""
echo "All outputs saved to: $RUN_DIR"
echo "========================================"
