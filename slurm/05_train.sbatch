#!/bin/bash
#SBATCH --job-name=mvp_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Circuit Breaker Training
# Trillium 1×H100
# =============================================================================
#
# DEPENDENCIES: Run 01-04 first!
#
# Submit:
#   sbatch slurm/05_train.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache Setup
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,xdg_cache,xdg_config}
export HOME="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$CB_SCRATCH/data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"
DR_FILE="$DATA_DIR/dr_stage1.jsonl"
COMBINED_FILE="$DATA_DIR/cb_training_batches.jsonl"
RUN_DIR="$CB_SCRATCH/runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/adapter"

mkdir -p "$RUN_DIR"

# Check Dependencies
if [[ ! -f "$DS_FILE" ]]; then echo "ERROR: Ds file not found"; exit 1; fi
if [[ ! -f "$DR_FILE" ]]; then echo "ERROR: Dr file not found"; exit 1; fi

echo "Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"
echo "Dr file: $DR_FILE ($(wc -l < "$DR_FILE") samples)"

# Combine Ds + Dr into Training Batches
echo "Combining Ds and Dr..."
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
print(f'Wrote {len(batches)} batches')
"

# Rebuild with Llama 3.1 format
FIXED_FILE="${COMBINED_FILE%.jsonl}_llama31.jsonl"
python src/data_generation/rebuild_training_data.py \
    --input "$COMBINED_FILE" \
    --output "$FIXED_FILE" \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --filter-tool-only

COMBINED_FILE="$FIXED_FILE"

# Resolve Model Path
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")
BASE_MODEL="$MODEL_PATH"

# =============================================================================
# STAGE 6: TRAINING - Conservative Hyperparameters
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 6: Training (300 steps, 4 GPUs)"
echo "========================================"

# Run training with conservative hyperparameters
accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-4} \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
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
