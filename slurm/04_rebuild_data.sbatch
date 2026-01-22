#!/bin/bash
#SBATCH --job-name=mvp_rebuild_data
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Rebuild and Validate Training Data
# CPU-only (no GPU needed)
# =============================================================================
#
# DEPENDENCY: Run 01_generate_ds.sbatch and 02_generate_dr.sbatch first!
#
# This script:
# 1. Combines Ds + Dr into batch format
# 2. Rebuilds with proper Llama 3.1 format
# 3. Validates the final training data
#
# Submit:
#   sbatch slurm/04_rebuild_data.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load python/3.11.5

source "$VENV_DIR/bin/activate"

CACHE_DIR="$CB_SCRATCH/cache"
export HF_HOME="$CACHE_DIR/hf"

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Check input files
DATA_DIR="$CB_SCRATCH/data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"
DR_FILE="$DATA_DIR/dr_stage1.jsonl"

if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found: $DS_FILE"
    exit 1
fi

if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr file not found: $DR_FILE"
    exit 1
fi

echo "Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"
echo "Dr file: $DR_FILE ($(wc -l < "$DR_FILE") samples)"

# =============================================================================
# Step 1: Combine Ds + Dr into Training Batches
# =============================================================================
echo ""
echo "========================================"
echo "Step 1: Combining Ds and Dr..."
echo "========================================"

COMBINED_FILE="$DATA_DIR/cb_training_batches_raw.jsonl"

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
print(f'Wrote {len(batches)} batches to $COMBINED_FILE')
"

# =============================================================================
# Step 2: Rebuild with Llama 3.1 format
# =============================================================================
echo ""
echo "========================================"
echo "Step 2: Rebuilding with Llama 3.1 format..."
echo "========================================"

TRAINING_FILE="$DATA_DIR/cb_training_batches.jsonl"

python src/data_generation/rebuild_training_data.py \
    --input "$COMBINED_FILE" \
    --output "$TRAINING_FILE" \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --filter-tool-only \
    --print-examples \
    --n-filtered 5 \
    --n-rendered 3 \
    --n-failed 5

echo "Training data written to: $TRAINING_FILE"

# =============================================================================
# Step 3: Validate Training Data
# =============================================================================
echo ""
echo "========================================"
echo "Step 3: Validating training data..."
echo "========================================"

if python src/data_generation/validate_format.py \
    --data "$TRAINING_FILE" \
    --strict \
    --print-examples \
    --n-per-error-type 3; then
    echo ""
    echo "✅ VALIDATION PASSED"
    echo "Training data ready: $TRAINING_FILE"
    exit 0
else
    echo ""
    echo "❌ VALIDATION FAILED"
    exit 1
fi
