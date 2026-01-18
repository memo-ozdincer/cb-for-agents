#!/bin/bash
#SBATCH --job-name=cb_full_4gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Full Pipeline - 4 GPU Variant
# Trillium 4×H100
# =============================================================================
#
# Conservative hyperparameters:
# - alpha_max: 0.5 (prevent collapse)
# - cb_target_layers: [15] (single mid-layer)
# - max_grad_norm: 0.5 (tighter clipping)
# - total_steps: 300 (less aggressive training)
# - learning_rate: 3e-5 (gentler updates)
#
# Usage:
#   sbatch slurm/07_full_pipeline_4gpu.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
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
echo "FULL PIPELINE (4 GPU) - Job ID: $SLURM_JOB_ID"
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
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$CB_SCRATCH/data"
RUN_DIR="$CB_SCRATCH/runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

B4_DATA="data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

DS_FILE="$DATA_DIR/ds_full.jsonl"
DR_FILE="$DATA_DIR/dr_full.jsonl"
COMBINED_FILE="$DATA_DIR/cb_training_full.jsonl"
FORMATTED_FILE="$DATA_DIR/cb_training_full_llama31.jsonl"
OUTPUT_DIR="$RUN_DIR/adapter"

# =============================================================================
# STAGE 1: Generate Ds (Harmful Set) - FULL DATA
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 1: Generating Ds (ALL DATA)"
echo "========================================"

python src/data_generation/generate_ds_mvp.py \
    --b4-data "$B4_DATA" \
    --tool-schema "$TOOL_SCHEMA" \
    --model "$BASE_MODEL" \
    --backend vllm \
    --tensor-parallel 4 \
    --batch-size 64 \
    --dtype bfloat16 \
    --temperature 0.7 \
    --min-yield 0.10 \
    --fail-on-low-yield \
    --output "$DS_FILE" \
    --output-ids "${DS_FILE%.jsonl}_ids.txt"

if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds generation failed"
    exit 1
fi

DS_COUNT=$(wc -l < "$DS_FILE")
echo "✓ Generated $DS_COUNT Ds samples"

# =============================================================================
# STAGE 2: Generate Dr (Retain Set) - FULL DATA
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 2: Generating Dr"
echo "========================================"

python src/data_generation/generate_dr_mvp.py \
    --ds-data "$DS_FILE" \
    --tool-schema "$TOOL_SCHEMA" \
    --model "$BASE_MODEL" \
    --backend vllm \
    --tensor-parallel 4 \
    --batch-size 64 \
    --dtype bfloat16 \
    --temperature 0.3 \
    --output "$DR_FILE"

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

if [[ $BATCH_COUNT -eq 0 ]]; then
    echo "⚠ Skipping validation (0 batches)"
    echo "ERROR: No training data generated"
    exit 1
fi

python src/data_generation/validate_format.py \
    --data "$FORMATTED_FILE" \
    --strict

echo "✓ Validation passed"

# =============================================================================
# STAGE 6: TRAINING - Conservative Hyperparameters
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 6: Training (300 steps)"
echo "========================================"

# Resolve Model Path
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")

# Run training with conservative hyperparameters
accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$FORMATTED_FILE" \
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
    --wandb-project "circuit-breakers" \
    --wandb-run-name "cb_full_4gpu_${SLURM_JOB_ID}"

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
echo "✓✓✓ FULL PIPELINE COMPLETE ✓✓✓"
echo "========================================"
echo ""
echo "Run directory: $RUN_DIR"
echo "Adapter saved to: $OUTPUT_DIR/final"
echo "Training data: $FORMATTED_FILE"
echo ""
echo "Hyperparameters used:"
echo "  alpha_max: 0.5 (conservative)"
echo "  cb_target_layers: [15] (single mid-layer)"
echo "  total_steps: 300"
echo "  learning_rate: 3e-5"
echo ""

# Create symlink to latest run
ln -sfn "$RUN_DIR" "$CB_SCRATCH/runs/latest"
echo "Latest run symlink: $CB_SCRATCH/runs/latest"
echo ""
