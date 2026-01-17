#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cb_stage2_train_quick
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16

# ============================================================================
# Stage 2 QUICK TRAIN (SMOKE TEST)
# ============================================================================
# Fast run to validate masking + training loop before long jobs.
# Uses small subset of batches + fewer steps.
# ============================================================================

set -e

# =============================================================================
# Environment Setup
# =============================================================================
PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv activation ==="
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

echo "=== W&B setup ==="
export WANDB_MODE=disabled

# Offline mode
export HF_HOME="$SCRATCH_DIR/cb_cache/hf"
export HF_HUB_CACHE="$SCRATCH_DIR/cb_cache/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"
MERGED_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"

RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_stage2_adapter_quick"
mkdir -p "$RUN_DIR"/{logs,outputs}

if [[ ! -f "$MERGED_FILE" ]]; then
  echo "ERROR: Merged Stage 2 data not found: $MERGED_FILE"
  exit 1
fi

echo ""
echo "Creating SMALL Stage 2 batches (50 harmful, 2:1 Dr:Ds)..."
QUICK_BATCHES="$STAGE2_DATA_DIR/stage2/train_batches_quick.jsonl"
python scripts/cb_data_generation/create_stage2_batches.py \
    --input "$MERGED_FILE" \
    --output "$QUICK_BATCHES" \
    --benign-per-harmful 2 \
    --max-harmful 50 \
    --require-tool-calls-harmful

echo "Quick batches: $QUICK_BATCHES ($(wc -l < "$QUICK_BATCHES") batches)"

# =============================================================================
# Run Training - QUICK
# =============================================================================
echo ""
echo "Starting Stage 2 QUICK training..."

accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    scripts/train_circuit_breaker.py \
    --preset llama-3.1-8b-stage2 \
    --base-model "$BASE_MODEL" \
    --data-path "$QUICK_BATCHES" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 0.5 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 30 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 15 \
    --no-wandb

echo ""
echo "Quick training complete."
