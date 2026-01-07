#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --job-name=eval_cb_llama31_8b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Evaluation - Killarney 1×L40S (Llama-3.1-8B-Instruct)
# =============================================================================
#
# Evaluates:
# 1. Baseline model (no CB)
# 2. CB-trained model
#
# Metrics:
# - Refusal rate on harmful prompts
# - Capability score on benign prompts
# - Agent safety rate (action-based)
#
# Usage:
#   export CB_CHECKPOINT=/path/to/checkpoint-150  # or checkpoint-100, final
#   sbatch slurm/Killarney/killarney_eval_cb.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
SCRATCH_DIR="$HOME/scratch"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

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

# =============================================================================
# Environment
# =============================================================================

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export WANDB_MODE=online
export WANDB_PROJECT=circuit-breakers
export WANDB_RUN_GROUP=killarney-cb-eval
export WANDB_TAGS=killarney,cb,eval,llama31

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# CB checkpoint to evaluate (can be set via env var)
# Default: use the final checkpoint from most recent run
if [[ -z "${CB_CHECKPOINT:-}" ]]; then
  LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/outputs/cb_llama31_8b_instruct 2>/dev/null | head -1)
  if [[ -n "$LATEST_RUN" && -d "$LATEST_RUN/final" ]]; then
    CB_CHECKPOINT="$LATEST_RUN/final"
  else
    echo "ERROR: No CB checkpoint found. Set CB_CHECKPOINT env var or ensure training completed."
    exit 1
  fi
fi

EVAL_OUTPUT_DIR="$SCRATCH_DIR/cb_eval/$SLURM_JOB_ID"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "========================================"
echo "Circuit Breaker Evaluation - Killarney"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Base Model: $BASE_MODEL"
echo "CB Checkpoint: $CB_CHECKPOINT"
echo "Eval Output: $EVAL_OUTPUT_DIR"
echo "========================================"

# =============================================================================
# Step 1: Prepare Evaluation Data
# =============================================================================

echo ""
echo "=== Step 1: Preparing Evaluation Data ==="
echo ""

EVAL_DATA_DIR="$REPO_DIR/data/circuit_breakers/eval"

if [[ ! -f "$EVAL_DATA_DIR/harmful_eval.jsonl" ]]; then
  echo "Extracting eval samples from training data..."
  python scripts/prepare_eval_data.py \
    --training-data data/circuit_breakers/cb_training_batches.jsonl \
    --output-dir "$EVAL_DATA_DIR" \
    --n-harmful 200 \
    --n-benign 200 \
    --seed 42
else
  echo "Eval data already exists at $EVAL_DATA_DIR"
fi

HARMFUL_DATA="$EVAL_DATA_DIR/harmful_eval.jsonl"
BENIGN_DATA="$EVAL_DATA_DIR/benign_eval.jsonl"

# Verify data exists
if [[ ! -f "$HARMFUL_DATA" ]] || [[ ! -f "$BENIGN_DATA" ]]; then
  echo "ERROR: Eval data not found!"
  echo "  Harmful: $HARMFUL_DATA"
  echo "  Benign: $BENIGN_DATA"
  exit 1
fi

echo "Eval data ready:"
echo "  Harmful: $(wc -l < $HARMFUL_DATA) prompts"
echo "  Benign: $(wc -l < $BENIGN_DATA) prompts"

# =============================================================================
# Step 2: Evaluate Baseline Model (No CB)
# =============================================================================

echo ""
echo "=== Step 2: Evaluating Baseline Model ==="
echo ""

python scripts/circuit_breakers/eval.py \
  --base-model "$BASE_MODEL" \
  --harmful-data "$HARMFUL_DATA" \
  --benign-data "$BENIGN_DATA" \
  --output "$EVAL_OUTPUT_DIR/baseline_results.json" \
  --max-samples 200 \
  --wandb-project circuit-breakers \
  --wandb-run-name "eval_baseline_llama31_8b_job${SLURM_JOB_ID}" \
  --wandb-tags eval,baseline,llama31 \
  --wandb-log-artifact

echo "Baseline evaluation complete!"

# =============================================================================
# Step 3: Evaluate CB-Trained Model
# =============================================================================

echo ""
echo "=== Step 3: Evaluating CB-Trained Model ==="
echo ""

python scripts/circuit_breakers/eval.py \
  --base-model "$BASE_MODEL" \
  --adapter-path "$CB_CHECKPOINT" \
  --harmful-data "$HARMFUL_DATA" \
  --benign-data "$BENIGN_DATA" \
  --output "$EVAL_OUTPUT_DIR/cb_results.json" \
  --max-samples 200 \
  --wandb-project circuit-breakers \
  --wandb-run-name "eval_cb_llama31_8b_job${SLURM_JOB_ID}" \
  --wandb-tags eval,cb,llama31 \
  --wandb-log-artifact

echo "CB evaluation complete!"

# =============================================================================
# Step 4: Compare Results
# =============================================================================

echo ""
echo "=== Step 4: Comparison Summary ==="
echo ""

python scripts/analyze_cb_results.py \
  --baseline "$EVAL_OUTPUT_DIR/baseline_results.json" \
  --cb-model "$EVAL_OUTPUT_DIR/cb_results.json" \
  --output "$EVAL_OUTPUT_DIR/analysis.json"

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Date: $(date)"
echo "Results: $EVAL_OUTPUT_DIR"
echo "========================================"
