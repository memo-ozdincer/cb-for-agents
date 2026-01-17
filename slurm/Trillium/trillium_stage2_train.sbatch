#!/bin/bash
#SBATCH --job-name=cb_stage2_train
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=def-zhijing

# ============================================================================
# Stage 2 MVP: Circuit Breaker Training - FIXED FOR REPRESENTATION COLLAPSE
# ============================================================================
# Key changes from Stage 1:
# 1. MUCH lower alpha-max (0.5 instead of 10.0) - prevent collapse
# 2. Higher retain weight via Dr:Ds ratio (5:1 instead of 1:1)  
# 3. Single target layer (layer 15 only) - more focused intervention
# 4. Gradient clipping (0.5) - prevent explosive gradients
# 5. More training data variety
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

echo "========================================"
echo "Stage 2 MVP: Circuit Breaker Training"
echo "COLLAPSE FIX VERSION"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""
echo "KEY CHANGES FROM STAGE 1:"
echo "  - alpha-max: 0.5 (was 10.0)"
echo "  - target-layers: 15 only (was 10, 20)"
echo "  - gradient clipping: 0.5"
echo "  - Dr:Ds ratio: 5:1 (expand Dr)"
echo "========================================"

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
CB_DATA_DIR="$PROJECT_DIR/data/circuit_breakers"
STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"

# Stage 2 merged data file (created by merge_stage2_data.py)
MERGED_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"
FALLBACK_MERGED_FILE="$CB_DATA_DIR/stage2/train.jsonl"

RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_stage2_adapter"

mkdir -p "$RUN_DIR"/{logs,outputs}

if [[ -f "$MERGED_FILE" ]]; then
    COMBINED_FILE="$MERGED_FILE"
    echo "Using merged Stage 2 data: $COMBINED_FILE"
elif [[ -f "$FALLBACK_MERGED_FILE" ]]; then
    COMBINED_FILE="$FALLBACK_MERGED_FILE"
    echo "Using merged Stage 2 data: $COMBINED_FILE"
else
    echo "ERROR: Merged Stage 2 data not found."
    echo "Run: sbatch slurm/Trillium/trillium_stage2_merge.sbatch"
    exit 1
fi

# =============================================================================
# Stage 2 Fix: Create CB training batches from flat samples
# =============================================================================
echo ""
echo "Creating Stage 2 training batches (5:1 Dr:Ds)..."

STAGE2_BATCHES="$STAGE2_DATA_DIR/stage2/train_batches.jsonl"
python scripts/cb_data_generation/create_stage2_batches.py \
    --input "$COMBINED_FILE" \
    --output "$STAGE2_BATCHES" \
    --benign-per-harmful 5

echo "Batches file: $STAGE2_BATCHES ($(wc -l < "$STAGE2_BATCHES") batches)"
COMBINED_FILE="$STAGE2_BATCHES"

# =============================================================================
# Resolve Model Path
# =============================================================================
echo ""
echo "Resolving model path from cache..."
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")
echo "Using local model path: $MODEL_PATH"
BASE_MODEL="$MODEL_PATH"

# =============================================================================
# Run Training - STAGE 2 HYPERPARAMETERS
# =============================================================================
echo ""
echo "Starting Stage 2 Circuit Breaker training..."
echo "Using $SLURM_GPUS_ON_NODE GPUs"
echo ""
echo "CRITICAL HYPERPARAMETER CHANGES:"
echo "  --alpha-max 0.5         (was 10.0 - MUCH lower to prevent collapse)"
echo "  --cb-target-layers 15   (was 10 20 - single layer, more focused)"
echo "  --max-grad-norm 0.5     (was 1.0 - tighter gradient clipping)"
echo ""

accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    scripts/train_circuit_breaker.py \
    --preset llama-3.1-8b-stage2 \
    --base-model "$BASE_MODEL" \
    --data-path "$COMBINED_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 0.5 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 300 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 15 \
    --wandb-project "circuit-breakers-mvp" \
    --wandb-run-name "cb_stage2_collapse_fix_${SLURM_JOB_ID}" \
    --wandb-notes "Stage 2: alpha=0.5, single layer 15, 5:1 Dr:Ds ratio"

# =============================================================================
# Post-Training Sanity Check
# =============================================================================
echo ""
echo "Running post-training sanity check..."

python scripts/circuit_breakers/sanity_check.py \
    --base-model "$BASE_MODEL" \
    --adapter-path "$OUTPUT_DIR/final" \
    --epsilon 1e-4 \
    --fail-on-error

# =============================================================================
# Quick Sample Output Check
# =============================================================================
echo ""
echo "Generating sample outputs to verify no collapse..."
EVAL_FILE="$STAGE2_DATA_DIR/stage2/eval.jsonl"
if [[ ! -f "$EVAL_FILE" ]]; then
  EVAL_FILE="$CB_DATA_DIR/stage2/eval.jsonl"
fi

if [[ -f "$EVAL_FILE" ]]; then
  python scripts/circuit_breakers/sample_outputs.py \
      --eval-data "$EVAL_FILE" \
      --cb-adapter "$OUTPUT_DIR/final" \
      --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
      --n 5 \
      2>&1 | tee "$RUN_DIR/sample_outputs.txt"
else
  echo "WARNING: eval data not found; skipping sample_outputs.py"
fi

# =============================================================================
# Create symlink
# =============================================================================
ln -sfn "$RUN_DIR" "$SCRATCH_DIR/cb_runs/latest_stage2"

echo "========================================"
echo "Stage 2 Training complete!"
echo "========================================"
echo "Adapter saved to: $OUTPUT_DIR"
echo "Sample outputs: $RUN_DIR/sample_outputs.txt"
echo ""
echo "Next steps:"
echo "  1. Check sample_outputs.txt for coherent generation"
echo "  2. sbatch slurm/Trillium/trillium_mvp_eval.sbatch (update adapter path)"
echo "========================================"
