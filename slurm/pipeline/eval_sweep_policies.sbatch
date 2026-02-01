#!/bin/bash
#SBATCH --job-name=eval_sweep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Evaluate all models from a sweep_policies run
# Uses the same evaluation logic as 05_eval.sbatch
# Supports multiple evaluation datasets (fujitsu, agentdojo)
# =============================================================================
#
# Configuration via environment variables:
#   SWEEP_DIR         - Path to sweep directory (required)
#   EVAL_DATASETS     - Comma-separated eval datasets (default: fujitsu,agentdojo)
#   TOOL_SCHEMA       - Tool schema JSON (default: b4_standard_v1.json)
#   LIMIT             - Limit eval samples (optional, for quick tests)
#   SHOW_EXAMPLES     - Set to "true" to print example traces
#   NUM_EXAMPLES      - Number of examples to show (default: 5)
#
# Usage:
#   SWEEP_DIR=/scratch/memoozd/cb-scratch/sweeps/policy_sweep_TIMESTAMP sbatch slurm/pipeline/eval_sweep_policies.sbatch
#
#   # Eval only on agentdojo
#   SWEEP_DIR=... EVAL_DATASETS=agentdojo sbatch slurm/pipeline/eval_sweep_policies.sbatch
#
#   # Eval only on fujitsu
#   SWEEP_DIR=... EVAL_DATASETS=fujitsu sbatch slurm/pipeline/eval_sweep_policies.sbatch
#
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

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "Evaluate Sweep Policies Models"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
if [[ -z "${SWEEP_DIR:-}" ]]; then
    # Try to find the most recent sweep
    SWEEP_DIR=$(ls -dt "$CB_SCRATCH"/sweeps/policy_sweep_* 2>/dev/null | head -1 || echo "")
    if [[ -z "$SWEEP_DIR" || ! -d "$SWEEP_DIR" ]]; then
        echo "ERROR: No sweep directory found. Set SWEEP_DIR or run sweep_policies.sbatch first."
        exit 1
    fi
    echo "Auto-detected sweep directory: $SWEEP_DIR"
fi

BASELINE_MODEL="${BASELINE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EVAL_DATASETS="${EVAL_DATASETS:-fujitsu,agentdojo}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"
LIMIT="${LIMIT:-}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
DTYPE="${DTYPE:-bfloat16}"

echo "Configuration:"
echo "  SWEEP_DIR: $SWEEP_DIR"
echo "  BASELINE_MODEL: $BASELINE_MODEL"
echo "  EVAL_DATASETS: $EVAL_DATASETS"
echo "  TOOL_SCHEMA: $TOOL_SCHEMA"
echo "  LIMIT: ${LIMIT:-unlimited}"
echo "  SHOW_EXAMPLES: $SHOW_EXAMPLES"
echo ""

# Validate tool schema
if [[ ! -f "$TOOL_SCHEMA" ]]; then
    echo "ERROR: Tool schema not found: $TOOL_SCHEMA"
    exit 1
fi

# Parse datasets
IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
echo "Will evaluate on ${#DATASETS[@]} dataset(s): ${DATASETS[*]}"
echo ""

# =============================================================================
# Discover policies in sweep
# =============================================================================
POLICIES=()
for policy_dir in "$SWEEP_DIR"/*/; do
    policy_name=$(basename "$policy_dir")
    if [[ -d "$policy_dir/model/final" ]] || [[ -d "$policy_dir/model" ]]; then
        POLICIES+=("$policy_name")
    fi
done

if [[ ${#POLICIES[@]} -eq 0 ]]; then
    echo "ERROR: No trained models found in $SWEEP_DIR"
    exit 1
fi

echo "Found ${#POLICIES[@]} policies to evaluate: ${POLICIES[*]}"
echo ""

# Create evaluation output directory
EVAL_DIR="$SWEEP_DIR/evaluations"
mkdir -p "$EVAL_DIR"

# =============================================================================
# Helper function to get dataset path
# =============================================================================
get_dataset_path() {
    local dataset="$1"
    local policy="$2"

    case "$dataset" in
        fujitsu)
            echo "$CB_SCRATCH/data/traces/fujitsu_b4_ds.jsonl"
            ;;
        agentdojo)
            # Use the split harmful traces from the sweep (if available)
            local split_path="$SWEEP_DIR/$policy/agentdojo_split/agentdojo_traces_harmful.jsonl"
            if [[ -f "$split_path" ]]; then
                echo "$split_path"
            else
                # Fallback to complete agentdojo traces
                echo "$CB_SCRATCH/data/traces/agentdojo_complete.jsonl"
            fi
            ;;
        agentdojo_all)
            echo "$CB_SCRATCH/data/traces/agentdojo_complete.jsonl"
            ;;
        *)
            echo ""
            ;;
    esac
}

# =============================================================================
# Evaluate Each Policy on Each Dataset
# =============================================================================
for policy in "${POLICIES[@]}"; do
    echo "========================================"
    echo "Evaluating: $policy"
    echo "========================================"

    POLICY_DIR="$SWEEP_DIR/$policy"
    MODEL_DIR="$POLICY_DIR/model"

    # Find the model (could be final/ or just model/)
    if [[ -d "$MODEL_DIR/final" ]]; then
        ADAPTER_PATH="$MODEL_DIR/final"
    elif [[ -f "$MODEL_DIR/adapter_config.json" ]]; then
        ADAPTER_PATH="$MODEL_DIR"
    else
        echo "  SKIP: No adapter found in $MODEL_DIR"
        continue
    fi

    echo "  Adapter: $ADAPTER_PATH"
    echo ""

    # Evaluate on each dataset
    for dataset in "${DATASETS[@]}"; do
        EVAL_DATA=$(get_dataset_path "$dataset" "$policy")

        if [[ -z "$EVAL_DATA" ]]; then
            echo "  SKIP: Unknown dataset '$dataset'"
            continue
        fi

        if [[ ! -f "$EVAL_DATA" ]]; then
            echo "  SKIP: Dataset not found: $EVAL_DATA"
            continue
        fi

        OUTPUT_FILE="$EVAL_DIR/${policy}_${dataset}_eval.json"

        echo "  Dataset: $dataset"
        echo "    Data: $EVAL_DATA"
        echo "    Output: $OUTPUT_FILE"

        # Build eval command
        EVAL_ARGS=(
            --baseline "$BASELINE_MODEL"
            --cb-adapter "$ADAPTER_PATH"
            --eval-data "$EVAL_DATA"
            --tool-schema "$TOOL_SCHEMA"
            --output "$OUTPUT_FILE"
            --dtype "$DTYPE"
        )

        if [[ -n "$LIMIT" ]]; then
            EVAL_ARGS+=(--limit "$LIMIT")
        fi

        # Run evaluation
        python src/evaluation/eval.py "${EVAL_ARGS[@]}"

        # Print summary
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo ""
            echo "    Results for $policy on $dataset:"
            python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    results = json.load(f)

if 'baseline' in results:
    print('      Baseline Tool-flip ASR: {:.1%}'.format(
        results['baseline']['tool_flip_asr']['attack_success_rate']))

if 'cb_model' in results:
    print('      CB Model Tool-flip ASR: {:.1%}'.format(
        results['cb_model']['tool_flip_asr']['attack_success_rate']))
    print('      Capability Retention:   {:.1%}'.format(
        results['cb_model']['capability_retention']['capability_retention']))

if 'delta' in results:
    print('      ASR Reduction:          {:.1%}'.format(
        -results['delta']['tool_flip_asr']))

if 'stage1_passed' in results:
    passed = results['stage1_passed']
    print('      Stage 1 Gates:          {} PASSED'.format('✅' if passed else '❌ NOT'))
" 2>/dev/null || echo "    (summary extraction failed)"
        fi

        # Show example outputs if requested
        if [[ "$SHOW_EXAMPLES" == "true" ]]; then
            DETAILS_FILE="${OUTPUT_FILE%.json}.details.jsonl"
            if [[ -f "$DETAILS_FILE" ]]; then
                echo ""
                echo "    Example Traces (first $NUM_EXAMPLES):"
                python3 -c "
import json

with open('$DETAILS_FILE', 'r') as f:
    lines = [json.loads(line) for line in f]

# Filter to tool_flip_asr cb_model results
cb_flip = [l for l in lines if l['model'] == 'cb_model' and l['metric'] == 'tool_flip_asr']

for i, sample in enumerate(cb_flip[:$NUM_EXAMPLES]):
    outcome = sample.get('outcome', '?')
    emoji = {'attack_success': '❌', 'correct_behavior': '✅', 'no_tool_call': '⚠️'}.get(outcome, '❓')
    print(f'    [{i+1}] {emoji} {outcome}')
    print(f'        Expected: {sample.get(\"expected_tool\")} | Simulated: {sample.get(\"simulated_tool\")} | Observed: {sample.get(\"observed_tool\")}')
    preview = sample.get('response_preview', '')[:120]
    if preview:
        print(f'        Response: {preview}...')
    print()
" 2>/dev/null || echo "    (example extraction failed)"
            fi
        fi

        echo ""
    done
done

# =============================================================================
# Summary Comparison (per dataset)
# =============================================================================
echo "========================================"
echo "SWEEP EVALUATION SUMMARY"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $dataset"
    echo "----------------------------------------"

    python3 -c "
import json
from pathlib import Path

eval_dir = Path('$EVAL_DIR')
dataset = '$dataset'
results = []

for f in sorted(eval_dir.glob(f'*_{dataset}_eval.json')):
    policy = f.stem.replace(f'_{dataset}_eval', '')
    try:
        with open(f) as fp:
            data = json.load(fp)

        cb_asr = data.get('cb_model', {}).get('tool_flip_asr', {}).get('attack_success_rate', -1)
        baseline_asr = data.get('baseline', {}).get('tool_flip_asr', {}).get('attack_success_rate', -1)
        capability = data.get('cb_model', {}).get('capability_retention', {}).get('capability_retention', -1)
        total_samples = data.get('cb_model', {}).get('tool_flip_asr', {}).get('total_samples', 0)
        passed = data.get('stage1_passed', False)

        results.append({
            'policy': policy,
            'baseline_asr': baseline_asr,
            'cb_asr': cb_asr,
            'capability': capability,
            'samples': total_samples,
            'passed': passed,
        })
    except Exception as e:
        print(f'Error loading {f}: {e}')

if not results:
    print('  No results found')
else:
    # Sort by CB ASR (lower is better)
    results.sort(key=lambda x: x['cb_asr'] if x['cb_asr'] >= 0 else float('inf'))

    print(f\"{'Policy':<20} {'Baseline':>10} {'CB ASR':>10} {'Samples':>8} {'Capability':>12} {'Gate':>6}\")
    print('-' * 70)

    for r in results:
        baseline = f\"{r['baseline_asr']:.1%}\" if r['baseline_asr'] >= 0 else 'N/A'
        cb = f\"{r['cb_asr']:.1%}\" if r['cb_asr'] >= 0 else 'N/A'
        cap = f\"{r['capability']:.1%}\" if r['capability'] >= 0 else 'N/A'
        status = '✅' if r['passed'] else '❌'
        print(f\"{r['policy']:<20} {baseline:>10} {cb:>10} {r['samples']:>8} {cap:>12} {status:>6}\")

    print()
    print(f\"Best policy: {results[0]['policy']} ({results[0]['cb_asr']:.1%})\")
" 2>/dev/null || echo "  (summary comparison failed)"
done

echo ""
echo "========================================"
echo "Evaluation results saved to: $EVAL_DIR"
echo "Finished at: $(date)"
