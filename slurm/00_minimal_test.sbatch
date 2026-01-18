#!/bin/bash
#SBATCH --job-name=cb_minimal_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# MINIMAL TEST: Full Pipeline Sanity Check
# Trillium 1×H100, ~5-10 minutes
# =============================================================================
#
# This is a MINIMAL test to verify:
# 1. All scripts run without errors
# 2. Model loads correctly
# 3. Trainer initializes
# 4. A few training epochs complete
# 5. KL metrics are computed
#
# NOT FOR REAL TRAINING - just a sanity check!
#
# Usage:
#   sbatch slurm/00_minimal_test.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

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

echo "========================================"
echo "MINIMAL TEST - Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,xdg_cache,xdg_config,vllm,flashinfer}

export HOME="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export VLLM_USAGE_STATS_DIR="$CACHE_DIR/vllm"
export FLASHINFER_WORKSPACE_DIR="$CACHE_DIR/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$CB_SCRATCH/data"
TEST_DIR="$CB_SCRATCH/test_$SLURM_JOB_ID"
mkdir -p "$TEST_DIR"

B4_DATA="data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

DS_FILE="$TEST_DIR/ds_test.jsonl"
DR_FILE="$TEST_DIR/dr_test.jsonl"
COMBINED_FILE="$TEST_DIR/cb_training_test.jsonl"
FORMATTED_FILE="$TEST_DIR/cb_training_test_llama31.jsonl"
OUTPUT_DIR="$TEST_DIR/adapter"

# =============================================================================
# STAGE 1: Generate Ds (Harmful Set) - MINIMAL
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 1: Generating Ds (limit=10)"
echo "========================================"

python src/data_generation/generate_ds_mvp.py \
    --b4-data "$B4_DATA" \
    --tool-schema "$TOOL_SCHEMA" \
    --model "$BASE_MODEL" \
    --output "$DS_FILE" \
    --limit 10 \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --min-yield 0.0

if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds generation failed"
    exit 1
fi

DS_COUNT=$(wc -l < "$DS_FILE")
echo "✓ Generated $DS_COUNT Ds samples"

# =============================================================================
# STAGE 2: Generate Dr (Retain Set) - MINIMAL
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 2: Generating Dr"
echo "========================================"

python src/data_generation/generate_dr_mvp.py \
    --ds-file "$DS_FILE" \
    --b4-data "$B4_DATA" \
    --tool-schema "$TOOL_SCHEMA" \
    --model "$BASE_MODEL" \
    --output "$DR_FILE" \
    --tensor-parallel-size 1 \
    --batch-size 4

if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr generation failed"
    exit 1
fi

DR_COUNT=$(wc -l < "$DR_FILE")
echo "✓ Generated $DR_COUNT Dr samples"

# =============================================================================
# STAGE 3: Combine Ds + Dr into Training Batches
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 3: Combining Ds + Dr"
echo "========================================"

python -c "
import json
from pathlib import Path

ds_samples = [json.loads(line) for line in Path('$DS_FILE').read_text().strip().split('\n') if line.strip()]
dr_samples = [json.loads(line) for line in Path('$DR_FILE').read_text().strip().split('\n') if line.strip()]
print(f'Loaded {len(ds_samples)} Ds, {len(dr_samples)} Dr')

batches = []
for i, ds in enumerate(ds_samples):
    dr = None
    for r in dr_samples:
        if r.get('metadata', {}).get('paired_with') == ds['id']:
            dr = r
            break
    if dr is None and dr_samples:
        dr = dr_samples[i % len(dr_samples)]
    batches.append({'id': f'batch_{i}', 'harmful': [ds], 'benign': [dr]})

with open('$COMBINED_FILE', 'w') as f:
    for b in batches:
        f.write(json.dumps(b) + '\n')
print(f'✓ Wrote {len(batches)} batches')
"

# =============================================================================
# STAGE 4: Rebuild with Llama 3.1 Format
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 4: Formatting for Llama 3.1"
echo "========================================"

python src/data_generation/rebuild_training_data.py \
    --input "$COMBINED_FILE" \
    --output "$FORMATTED_FILE" \
    --tool-schema "$TOOL_SCHEMA" \
    --filter-tool-only

BATCH_COUNT=$(wc -l < "$FORMATTED_FILE")
echo "✓ Formatted $BATCH_COUNT training batches"

# =============================================================================
# STAGE 5: Validate Format
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 5: Validating Format"
echo "========================================"

python src/data_generation/validate_format.py \
    --input "$FORMATTED_FILE" \
    --strict

echo "✓ Validation passed"

# =============================================================================
# STAGE 6: MINIMAL TRAINING (just 10 steps!)
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 6: Training (10 steps)"
echo "========================================"

# Resolve Model Path
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")

# Run minimal training
accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$FORMATTED_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 10 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --lora-r 8 \
    --lora-alpha 16 \
    --cb-target-layers 10 20 \
    --no-wandb

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Training failed - no output directory"
    exit 1
fi

echo "✓ Training completed"

# =============================================================================
# STAGE 7: Sanity Check
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 7: Sanity Check"
echo "========================================"

python src/evaluation/sanity_check.py \
    --base-model "$MODEL_PATH" \
    --adapter-path "$OUTPUT_DIR/final" \
    --epsilon 1e-4 \
    --fail-on-error

echo "✓ Sanity check passed"

# =============================================================================
# SUCCESS!
# =============================================================================
echo ""
echo "========================================"
echo "✓✓✓ ALL STAGES PASSED ✓✓✓"
echo "========================================"
echo ""
echo "Test artifacts in: $TEST_DIR"
echo "Adapter saved to: $OUTPUT_DIR/final"
echo ""
echo "Pipeline is working! Ready for full runs."
echo ""
