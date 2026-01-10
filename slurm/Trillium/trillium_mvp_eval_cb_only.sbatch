#!/bin/bash
#SBATCH --job-name=mvp_eval_cb_only
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# EMERGENCY: CB Model Only Evaluation (Skip Baseline)
# For when the full eval times out after baseline completes
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache setup
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
export HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "EMERGENCY: CB Model Only Eval"
echo "========================================"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/ 2>/dev/null | head -1)
CB_ADAPTER="${LATEST_RUN}outputs/cb_mvp_adapter/final"

if [[ ! -d "$CB_ADAPTER" ]]; then
    CHECKPOINT=$(ls -td ${LATEST_RUN}outputs/cb_mvp_adapter/checkpoint-* 2>/dev/null | head -1)
    CB_ADAPTER="$CHECKPOINT"
fi

DATA_DIR="$SCRATCH_DIR/cb_mvp_data"
EVAL_DATA="$DATA_DIR/eval_stage1.jsonl"
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_eval"

echo "CB adapter: $CB_ADAPTER"
echo "Eval data: $(wc -l < "$EVAL_DATA") samples"

# Run ONLY CB model evaluation (no --baseline = skip baseline)
python scripts/circuit_breakers/eval_mvp.py \
    --cb-model "$CB_ADAPTER" \
    --eval-data "$EVAL_DATA" \
    --device auto \
    --dtype bfloat16 \
    --output "$OUTPUT_DIR/eval_cb_only_${SLURM_JOB_ID}.json"

echo "Done! Results: $OUTPUT_DIR/eval_cb_only_${SLURM_JOB_ID}.json"

# If you have baseline results from a previous run, merge them:
echo ""
echo "To merge with previous baseline results, run:"
echo "python -c \"import json; b=json.load(open('BASELINE_FILE.json')); c=json.load(open('$OUTPUT_DIR/eval_cb_only_${SLURM_JOB_ID}.json')); b.update(c); json.dump(b, open('merged.json','w'), indent=2)\""
