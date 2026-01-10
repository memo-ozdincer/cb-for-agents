#!/bin/bash
#SBATCH --job-name=mvp_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Circuit Breaker Training
# Trillium 1×H100
# =============================================================================
#
# Trains Circuit Breaker adapter using:
# - Ds (harmful set): Successful tool-flip attacks
# - Dr (retain set): Paired benign twins
# - Dual coefficient schedule (cs, cr)
# - Llama 3.1 8B Instruct base model
#
# DEPENDENCIES: Run these first:
#   1. trillium_mvp_generate_ds.sbatch
#   2. trillium_mvp_generate_dr.sbatch
#   3. trillium_mvp_validate.sbatch
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_train.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"


# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup (MUST match prefetch_models.sh)
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/datasets,torch}

# CRITICAL FIX: Override HOME to trick flashinfer and other tools that hardcode ~/.cache
export HOME="$CACHE_ROOT"

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export WANDB_DIR="$CACHE_ROOT/wandb"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache" # Ensure this is redirected too
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"


# Enable offline mode - fail fast if model not cached
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Weights & Biases in offline mode
export WANDB_MODE=offline
export WANDB_DIR="$CACHE_ROOT/wandb"

# DO NOT SET TRANSFORMERS_CACHE - deprecated and causes cache fragmentation

echo "Python: $(python -V)"
echo ""
echo "Cache Configuration:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Circuit Breaker Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
echo "========================================"

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$SCRATCH_DIR/cb_mvp_data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"
DR_FILE="$DATA_DIR/dr_stage1.jsonl"
COMBINED_FILE="$DATA_DIR/cb_training_batches.jsonl"
RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_mvp_adapter"

mkdir -p "$RUN_DIR"/{logs,outputs}

# =============================================================================
# Check Dependencies
# =============================================================================
if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found at $DS_FILE"
    echo "Please run trillium_mvp_generate_ds.sbatch first!"
    exit 1
fi

if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr file not found at $DR_FILE"
    echo "Please run trillium_mvp_generate_dr.sbatch first!"
    exit 1
fi

echo "Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"
echo "Dr file: $DR_FILE ($(wc -l < "$DR_FILE") samples)"

# =============================================================================
# Combine Ds + Dr into Training Batches
# =============================================================================
echo "Combining Ds and Dr into training batches..."
python -c "
import json
from pathlib import Path

ds_file = Path('$DS_FILE')
dr_file = Path('$DR_FILE')
output_file = Path('$COMBINED_FILE')

# Load samples
ds_samples = [json.loads(line) for line in ds_file.read_text().strip().split('\n') if line.strip()]
dr_samples = [json.loads(line) for line in dr_file.read_text().strip().split('\n') if line.strip()]

print(f'Loaded {len(ds_samples)} Ds samples')
print(f'Loaded {len(dr_samples)} Dr samples')

# Create batches (1:1 pairing where possible)
batches = []
for i, ds in enumerate(ds_samples):
    # Find corresponding Dr (by paired_with in metadata)
    dr = None
    for r in dr_samples:
        if r.get('metadata', {}).get('paired_with') == ds['id']:
            dr = r
            break
    
    # If no paired Dr, use any Dr
    if dr is None and dr_samples:
        dr = dr_samples[i % len(dr_samples)]
    
    batch = {
        'id': f'batch_{i}',
        'harmful': [ds],
        'benign': [dr],
    }
    batches.append(batch)

# Write combined batches
with output_file.open('w') as f:
    for batch in batches:
        f.write(json.dumps(batch) + '\n')

print(f'Wrote {len(batches)} training batches to {output_file}')
"

if [[ ! -f "$COMBINED_FILE" ]]; then
    echo "ERROR: Failed to create combined training file"
    exit 1
fi
echo "Combined file: $COMBINED_FILE ($(wc -l < "$COMBINED_FILE") batches)"

# =============================================================================
# CRITICAL FIX: Rebuild Training Data with Proper Llama 3.1 Format
# =============================================================================
# The original data used simple "User: ... Assistant: ..." format which:
# 1. Doesn't match Llama 3.1 chat template (<|start_header_id|>assistant etc.)
# 2. Missing <|eot_id|>/<|eom_id|> terminators on completions
# 3. Causes assistant boundary masking to fail
# 4. Mixed tool call formats ([TOOL_CALLS] vs <|python_tag|>)
# This step rebuilds the text field with proper formatting.
echo ""
echo "Rebuilding training data with proper Llama 3.1 format..."

FIXED_FILE="${COMBINED_FILE%.jsonl}_llama31.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

# Use V2 rebuild script which:
# - Converts [TOOL_CALLS] format to <|python_tag|>
# - Adds proper Llama 3.1 headers and terminators
# - Filters to tool-routing samples only (critical for CB training)
python scripts/cb_data_generation/rebuild_training_data_v2.py \
    --input "$COMBINED_FILE" \
    --output "$FIXED_FILE" \
    --tool-schema "$TOOL_SCHEMA" \
    --filter-tool-only

if [[ ! -f "$FIXED_FILE" ]]; then
    echo "ERROR: Failed to create fixed training file"
    exit 1
fi
echo "Fixed file: $FIXED_FILE ($(wc -l < "$FIXED_FILE") batches)"

# Use the fixed file for training
COMBINED_FILE="$FIXED_FILE"

# =============================================================================
# Resolve Model Path (Offline Fix)
# =============================================================================
echo ""
echo "Resolving model path from cache..."
# We use python to find the snapshot path since we are in offline mode.
# AutoTokenizer with a generic ID triggers a hub check which fails in offline mode.
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")
echo "Using local model path: $MODEL_PATH"
# Update BASE_MODEL to the absolute path
BASE_MODEL="$MODEL_PATH"

# =============================================================================
# Pre-Training Sanity Check
# =============================================================================
echo ""
echo "Running pre-training sanity check..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Loading base model to verify GPU access...')
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True)
print('✅ Base model loaded successfully')
del model
torch.cuda.empty_cache()
"

# =============================================================================
# Run Training
# =============================================================================
echo ""
echo "Starting Circuit Breaker training..."
echo "Using $SLURM_GPUS_ON_NODE GPUs"

# STAGE 1 FIX: Improved hyperparameters
# - alpha-max 10.0: Paper default, stronger rerouting push (was 1.0)
# - cb-target-layers 10 20: Sparse mid-layer targeting is more effective (was 16-23)
# - preset llama-3.1-8b-instruct: Match the actual model
# - total-steps 500: More training with low baseline ASR

# Multi-GPU training with Accelerate
# With 4 GPUs: effective batch = 4 (batch_size) * 4 (grad_accum) * 4 (GPUs) = 64
accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    scripts/train_circuit_breaker.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$BASE_MODEL" \
    --data-path "$COMBINED_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 500 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 20 \
    --wandb-project "circuit-breakers-mvp" \
    --wandb-run-name "cb_mvp_stage1_${SLURM_JOB_ID}"

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
# Create symlink to latest run
# =============================================================================
ln -sfn "$RUN_DIR" "$SCRATCH_DIR/cb_runs/latest"

echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Adapter saved to: $OUTPUT_DIR"
echo "Symlink: $SCRATCH_DIR/cb_runs/latest"
echo ""
echo "Next steps:"
echo "  1. sbatch slurm/Trillium/trillium_mvp_sanity_check.sbatch"
echo "  2. sbatch slurm/Trillium/trillium_mvp_eval.sbatch"
echo "========================================"
