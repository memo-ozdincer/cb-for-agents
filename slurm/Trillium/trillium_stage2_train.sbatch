#!/bin/bash
#SBATCH --job-name=cb_stage2_train
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

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
PROJECT_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
SCRATCH_DIR="/scratch/memoozd"

source "$PROJECT_DIR/agent_adversarial/bin/activate"
cd "$PROJECT_DIR"

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
DATA_DIR="$SCRATCH_DIR/cb_mvp_data"
CB_DATA_DIR="$PROJECT_DIR/data/circuit_breakers"
STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"

# Stage 2 merged data file (created by merge_stage2_data.py)
MERGED_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"
FALLBACK_MERGED_FILE="$CB_DATA_DIR/stage2/train.jsonl"

# Fallback to individual files if merged not available
DS_FILE="$DATA_DIR/ds_stage2.jsonl"
DR_FILE="$DATA_DIR/dr_stage2.jsonl"

# Check for merged Stage 2 data first
if [[ -f "$MERGED_FILE" ]]; then
    echo "Using merged Stage 2 data: $MERGED_FILE"
    COMBINED_FILE="$MERGED_FILE"
    SKIP_MERGE=1
elif [[ -f "$FALLBACK_MERGED_FILE" ]]; then
    echo "Using merged Stage 2 data: $FALLBACK_MERGED_FILE"
    COMBINED_FILE="$FALLBACK_MERGED_FILE"
    SKIP_MERGE=1
else
    echo "Merged Stage 2 data not found, using individual files..."
    SKIP_MERGE=0
    
    # Fallback to Stage 1 data if Stage 2 not ready
    if [[ ! -f "$DS_FILE" ]]; then
        echo "Stage 2 Ds not found, using Stage 1 Ds..."
        DS_FILE="$DATA_DIR/ds_stage1.jsonl"
    fi
    if [[ ! -f "$DR_FILE" ]]; then
        echo "Stage 2 Dr not found, using Stage 1 Dr..."
        DR_FILE="$DATA_DIR/dr_stage1.jsonl"
    fi
fi

COMBINED_FILE="${COMBINED_FILE:-$DATA_DIR/cb_training_stage2.jsonl}"
RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_stage2_adapter"

mkdir -p "$RUN_DIR"/{logs,outputs}

if [[ -z "$SKIP_MERGE" ]] || [[ "$SKIP_MERGE" != "1" ]]; then
    echo "Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"
    echo "Dr file: $DR_FILE ($(wc -l < "$DR_FILE") samples)"
fi

# =============================================================================
# Stage 2 Fix: Create training batches (skip if using merged data)
# =============================================================================
if [[ "$SKIP_MERGE" == "1" ]]; then
    echo ""
    echo "Using pre-merged Stage 2 data: $COMBINED_FILE"
    echo "Samples: $(wc -l < "$COMBINED_FILE")"
else
    echo ""
    echo "Creating Stage 2 training batches with 5:1 Dr:Ds ratio..."
    python -c "
import json
from pathlib import Path
import random

ds_file = Path('$DS_FILE')
dr_file = Path('$DR_FILE')
output_file = Path('$COMBINED_FILE')

# Load samples
ds_samples = [json.loads(line) for line in ds_file.read_text().strip().split('\n') if line.strip()]
dr_samples = [json.loads(line) for line in dr_file.read_text().strip().split('\n') if line.strip()]

print(f'Loaded {len(ds_samples)} Ds samples')
print(f'Loaded {len(dr_samples)} Dr samples')

# Stage 2 FIX: Replicate Dr to achieve 5:1 ratio
target_dr_count = len(ds_samples) * 5
if len(dr_samples) < target_dr_count:
    # Replicate Dr samples
    replication_factor = (target_dr_count // len(dr_samples)) + 1
    dr_samples_expanded = (dr_samples * replication_factor)[:target_dr_count]
    print(f'Expanded Dr from {len(dr_samples)} to {len(dr_samples_expanded)} (5:1 ratio)')
    dr_samples = dr_samples_expanded

# Create batches
batches = []

# First, pair each Ds with 5 Dr samples
for i, ds in enumerate(ds_samples):
    batch = {
        'harmful': ds,
        'retain': dr_samples[i * 5:(i + 1) * 5] if i * 5 + 5 <= len(dr_samples) else [dr_samples[i % len(dr_samples)]]
    }
    batches.append(batch)

# Shuffle
random.seed(42)
random.shuffle(batches)

# Write
with open(output_file, 'w') as f:
    for batch in batches:
        f.write(json.dumps(batch) + '\n')

print(f'Created {len(batches)} training batches')
print(f'Average Dr per batch: {sum(len(b[\"retain\"]) if isinstance(b[\"retain\"], list) else 1 for b in batches) / len(batches):.1f}')
"
fi

# =============================================================================
# Rebuild with Llama 3.1 format
# =============================================================================
echo ""
echo "Rebuilding training data with proper Llama 3.1 format..."

FIXED_FILE="${COMBINED_FILE%.jsonl}_llama31.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

python scripts/cb_data_generation/rebuild_training_data_v2.py \
    --input "$COMBINED_FILE" \
    --output "$FIXED_FILE" \
    --tool-schema "$TOOL_SCHEMA" \
    --filter-tool-only

echo "Fixed file: $FIXED_FILE ($(wc -l < "$FIXED_FILE") batches)"
COMBINED_FILE="$FIXED_FILE"

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
    --max-grad-norm 0.5 \
    --rr-weight 0.7 \
    --retain-weight 1.3 \
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
python scripts/circuit_breakers/sample_outputs.py \
    --eval-data "$DATA_DIR/eval_stage1.jsonl" \
    --cb-adapter "$OUTPUT_DIR/final" \
    --tool-schema "$PROJECT_DIR/configs/tool_schemas/b4_standard_v1.json" \
    --n 5 \
    2>&1 | tee "$RUN_DIR/sample_outputs.txt"

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
