#!/bin/bash
#SBATCH --job-name=mvp_eval_set
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=01:18:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Create Held-Out Evaluation Set
# CPU-only (no GPU needed)
# =============================================================================
#
# DEPENDENCY: Run 01_generate_ds.sbatch first!
#
# Submit:
#   sbatch slurm/03_create_eval.sbatch
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

# Check Dependencies
DATA_DIR="$CB_SCRATCH/data"
DS_IDS="$DATA_DIR/ds_stage1_ids.txt"

if [[ ! -f "$DS_IDS" ]]; then
    DS_FILE="$DATA_DIR/ds_stage1.jsonl"
    if [[ ! -f "$DS_FILE" ]]; then
        echo "ERROR: Neither $DS_IDS nor $DS_FILE found!"
        echo "Please run 01_generate_ds.sbatch first!"
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

echo "Found training IDs: $DS_IDS ($(wc -l < "$DS_IDS") IDs)"

# Run Eval Set Creation
python src/data_generation/create_eval_set.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --train-ids "$DS_IDS" \
    --holdout-fraction 0.15 \
    --min-samples 100 \
    --max-samples 500 \
    --stratify-by subtype \
    --output "$DATA_DIR/eval_stage1.jsonl" \
    --print-examples \
    --n-per-stratum 3

echo "Eval set creation complete!"
echo "  Output: $DATA_DIR/eval_stage1.jsonl"
echo "  Stats: $DATA_DIR/eval_stage1.stats.json"
echo "  Examples: $DATA_DIR/eval_stage1.examples.json"
