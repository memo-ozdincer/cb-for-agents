#!/bin/bash
#SBATCH --job-name=mvp_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Evaluation
# Trillium 1×H100
# =============================================================================
#
# Submit:
#   sbatch slurm/06_eval.sbatch
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
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,xdg_cache,xdg_config}
export HOME="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

LATEST_RUN=$(ls -td $CB_SCRATCH/runs/*/ 2>/dev/null | head -1)
if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: No run directory found"
    exit 1
fi

CB_ADAPTER="${LATEST_RUN}adapter/final"
DATA_DIR="$CB_SCRATCH/data"
EVAL_DATA="$DATA_DIR/eval_stage1.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"
EVAL_DIR="$CB_SCRATCH/eval"

mkdir -p "$EVAL_DIR"

echo "Latest run: $LATEST_RUN"
echo "CB adapter: $CB_ADAPTER"

# Check Dependencies
if [[ ! -f "$EVAL_DATA" ]]; then echo "ERROR: Eval data not found"; exit 1; fi
if [[ ! -d "$CB_ADAPTER" ]]; then
    CHECKPOINT=$(ls -td ${LATEST_RUN}adapter/checkpoint-* 2>/dev/null | head -1)
    if [[ -n "$CHECKPOINT" ]]; then
        CB_ADAPTER="$CHECKPOINT"
    else
        echo "ERROR: No adapter found"
        exit 1
    fi
fi

echo "Eval data: $EVAL_DATA ($(wc -l < "$EVAL_DATA") samples)"

# Run Evaluation
python src/evaluation/eval.py \
    --baseline "$BASE_MODEL" \
    --cb-adapter "$CB_ADAPTER" \
    --eval-data "$EVAL_DATA" \
    --tool-schema "$TOOL_SCHEMA" \
    --device auto \
    --num-workers 1 \
    --gpu-ids 0 \
    --dtype bfloat16 \
    --output "$EVAL_DIR/results_${SLURM_JOB_ID}.json" \
    --fail-on-gate

echo "Evaluation complete! Results: $EVAL_DIR/results_${SLURM_JOB_ID}.json"

# Display Summary
python -c "
import json
with open('$EVAL_DIR/results_${SLURM_JOB_ID}.json', 'r') as f:
    results = json.load(f)

print('=== Stage 1 MVP Results ===')
if 'baseline' in results:
    b = results['baseline']
    print(f'Baseline ASR: {b[\"tool_flip_asr\"][\"attack_success_rate\"]:.2%}')
if 'cb_model' in results:
    c = results['cb_model']
    print(f'CB Model ASR: {c[\"tool_flip_asr\"][\"attack_success_rate\"]:.2%}')
if 'stage1_passed' in results:
    status = 'PASSED' if results['stage1_passed'] else 'FAILED'
    print(f'Stage 1: {status}')
"

# Display Sample Outputs
if [[ -f "$EVAL_DIR/results_${SLURM_JOB_ID}.paired_outputs.jsonl" ]]; then
    echo ""
    echo "=== Sample Baseline vs CB Outputs (first 20) ==="
    python -c "
import json

with open('$EVAL_DIR/results_${SLURM_JOB_ID}.paired_outputs.jsonl', 'r') as f:
    for idx, line in enumerate(f, 1):
        if idx > 20:
            break
        record = json.loads(line)
        print(f'\n--- Sample {idx} (ID: {record.get(\"id\", \"unknown\")}) ---')
        print(f'Expected Tool: {record.get(\"expected_tool\", \"N/A\")}')
        print(f'Simulated Tool: {record.get(\"simulated_tool\", \"N/A\")}')
        print(f'Baseline Observed: {record.get(\"baseline_observed_tool\", \"N/A\")} ({record.get(\"baseline_outcome\", \"N/A\")})')
        print(f'CB Observed: {record.get(\"cb_observed_tool\", \"N/A\")} ({record.get(\"cb_outcome\", \"N/A\")})')
        
        baseline_resp = record.get('baseline_response', '')
        cb_resp = record.get('cb_response', '')
        
        print(f'\nBaseline Response (first 200 chars):')
        print(f'  {baseline_resp[:200]}...' if len(baseline_resp) > 200 else f'  {baseline_resp}')
        
        print(f'\nCB Response (first 200 chars):')
        print(f'  {cb_resp[:200]}...' if len(cb_resp) > 200 else f'  {cb_resp}')
"
    echo ""
    echo "✓ Sample outputs displayed"
else
    echo "⚠️  No paired outputs file found"
fi
