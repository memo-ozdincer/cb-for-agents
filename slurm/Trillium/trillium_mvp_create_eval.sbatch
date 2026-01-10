#!/bin/bash
#SBATCH --job-name=mvp_eval_set
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Create Held-Out Evaluation Set
# Trillium CPU-only (no GPU needed)
# =============================================================================
#
# Creates evaluation set from B4 records:
# - Excludes records used in training (Ds)
# - Stratified sampling by attack subtype
# - ~15% holdout, 100-500 samples
#
# DEPENDENCY: Run trillium_mvp_generate_ds.sbatch first!
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_create_eval.sbatch
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
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Note: Eval set creation is CPU-only, but we still set cache vars for consistency
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
export HF_HOME="$CACHE_ROOT/hf"

echo "Python: $(python -V)"
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Create Evaluation Set"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# =============================================================================
# Check Dependencies
# =============================================================================
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_data"
DS_IDS="$OUTPUT_DIR/ds_stage1_ids.txt"

if [[ ! -f "$DS_IDS" ]]; then
    # Try to extract from JSONL if txt doesn't exist
    DS_FILE="$OUTPUT_DIR/ds_stage1.jsonl"
    if [[ ! -f "$DS_FILE" ]]; then
        echo "ERROR: Neither $DS_IDS nor $DS_FILE found!"
        echo "Please run trillium_mvp_generate_ds.sbatch first!"
        exit 1
    fi
    echo "Extracting IDs from $DS_FILE..."
    python -c "
import json
with open('$DS_FILE', 'r') as f:
    ids = [json.loads(line)['id'] for line in f if line.strip()]
with open('$DS_IDS', 'w') as f:
    f.write('\n'.join(ids))
print(f'Extracted {len(ids)} IDs')
"
fi

echo "Found training IDs: $DS_IDS"
echo "Training ID count: $(wc -l < "$DS_IDS")"

# =============================================================================
# Run Eval Set Creation
# =============================================================================
python scripts/cb_data_generation/create_eval_set.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --train-ids "$DS_IDS" \
    --holdout-fraction 0.15 \
    --min-samples 100 \
    --max-samples 500 \
    --stratify-by subtype \
    --output "$OUTPUT_DIR/eval_stage1.jsonl"

echo "========================================"
echo "Eval set creation complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR/eval_stage1.jsonl"
echo "Eval sample count: $(wc -l < "$OUTPUT_DIR/eval_stage1.jsonl")"
echo "========================================"
