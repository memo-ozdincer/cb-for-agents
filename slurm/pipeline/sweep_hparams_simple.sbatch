#!/bin/bash
#SBATCH --job-name=sweep_hparams_simple
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# SIMPLE HYPERPARAMETER SWEEP
#
# Runs exactly 6 configurations:
#   Policies:
#     - cb_full_sequence   (Full sequence)
#     - tool_calls_only    (Tool Call Only)
#     - assistant_and_tool (Tool OUTPUT)
#   Layers:
#     - 10,20
#     - 8,16,24
#
# Total runs = 3 policies * 2 layer configs = 6
#

# Optional overrides via env vars:
#   TOTAL_STEPS     - Override training steps (default: 200)
#   ALPHAS          - Override alpha values (default: "10.0")
#   LAYER_CONFIGS   - Override layer configs (default: "10,20|8,16,24")
#   POLICIES        - Override policies (default: "cb_full_sequence,tool_calls_only,assistant_and_tool")
#   LIMIT_EVAL      - Samples per eval dataset (default: 100)
#   NUM_EXAMPLES    - Examples to print per dataset (default: 50)
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

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "SIMPLE HYPERPARAMETER SWEEP"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_STEPS=200
DEFAULT_ALPHAS="10.0"
DEFAULT_LAYER_CONFIGS="10,20|8,16,24"
DEFAULT_POLICIES="cb_full_sequence,tool_calls_only,assistant_and_tool"
DEFAULT_LIMIT_EVAL=100
DEFAULT_NUM_EXAMPLES=50

TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_STEPS}"
ALPHAS="${ALPHAS:-$DEFAULT_ALPHAS}"
LAYER_CONFIGS="${LAYER_CONFIGS:-$DEFAULT_LAYER_CONFIGS}"
POLICIES="${POLICIES:-$DEFAULT_POLICIES}"
LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"

# Fixed training params (matching sweep_policies.sbatch)
BATCH_SIZE=8
LEARNING_RATE="5e-5"
WARMUP_STEPS=20
LORA_R=16
LORA_ALPHA=32
MAX_SEQ_LENGTH=2048
DTYPE="bfloat16"

echo "Configuration:"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  ALPHAS: $ALPHAS"
echo "  LAYER_CONFIGS: $LAYER_CONFIGS"
echo "  POLICIES: $POLICIES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  LIMIT_EVAL: $LIMIT_EVAL"
echo "  NUM_EXAMPLES: $NUM_EXAMPLES"
echo "  SHOW_EXAMPLES: $SHOW_EXAMPLES"
echo ""

# =============================================================================
# Input Traces (ALL available data)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"

# Fujitsu B4 (tool-flip attacks)
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"

# AgentDojo (multi-turn agent attacks)
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"

# LLMail-Inject (email injection attacks) - optional, requires generation first
LLMAIL_DS_TRACES="$TRACES_DIR/llmail_inject_ds.jsonl"
LLMAIL_DR_TRACES="$TRACES_DIR/llmail_inject_dr.jsonl"

# Build list of available traces (some may not exist yet)
TRACE_FILES=("$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES")

# Add LLMail-Inject if generated
if [[ -f "$LLMAIL_DS_TRACES" ]]; then
    TRACE_FILES+=("$LLMAIL_DS_TRACES")
fi
if [[ -f "$LLMAIL_DR_TRACES" ]]; then
    TRACE_FILES+=("$LLMAIL_DR_TRACES")
fi

echo "Input traces (ALL used for training):"
for trace in "${TRACE_FILES[@]}"; do
    if [[ -f "$trace" ]]; then
        echo "  ✓ $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  ✗ $(basename "$trace"): NOT FOUND"
        exit 1
    fi
done
echo ""

# =============================================================================
# Resolve Tokenizer (same as sweep_policies.sbatch)
# =============================================================================
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"
echo "Resolving tokenizer '$TOKENIZER' to local cache path..."

MODEL_CACHE_DIR="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        echo "  Found local snapshot: $SNAPSHOT_PATH"
        TOKENIZER="$SNAPSHOT_PATH"
    else
        echo "  ERROR: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        echo "  Run slurm/cache_models.sh on login node first!"
        exit 1
    fi
else
    echo "  ERROR: Model not cached at: $MODEL_CACHE_DIR"
    echo "  Run slurm/cache_models.sh on login node first!"
    exit 1
fi
echo ""

# =============================================================================
# Create Sweep Directory
# =============================================================================
SWEEP_ID="hparam_sweep_simple_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"

echo "Sweep output: $SWEEP_DIR"
echo ""

# =============================================================================
# Count Total Runs
# =============================================================================
count_configs() {
    local count=0
    IFS=',' read -ra ALPHA_ARRAY <<< "$ALPHAS"
    IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
    IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
    
    for alpha in "${ALPHA_ARRAY[@]}"; do
        for layers in "${LAYER_ARRAY[@]}"; do
            for policy in "${POLICY_ARRAY[@]}"; do
                count=$((count + 1))
            done
        done
    done
    echo "$count"
}

TOTAL_RUNS=$(count_configs)
echo "========================================"
echo "TOTAL CONFIGURATIONS TO TEST: $TOTAL_RUNS"
echo "========================================"
echo ""

# =============================================================================
# Main Sweep Loop
# =============================================================================
run_idx=0
success_count=0
fail_count=0

IFS=',' read -ra ALPHA_ARRAY <<< "$ALPHAS"
IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"

for alpha in "${ALPHA_ARRAY[@]}"; do
    for layers in "${LAYER_ARRAY[@]}"; do
        for policy in "${POLICY_ARRAY[@]}"; do
            run_idx=$((run_idx + 1))
            
            # Create unique run name
            layers_short="${layers//,/_}"
            RUN_NAME="a${alpha}_l${layers_short}_${policy}"
            RUN_DIR="$SWEEP_DIR/$RUN_NAME"
            
            echo "========================================"
            echo "[$run_idx/$TOTAL_RUNS] $RUN_NAME"
            echo "========================================"
            echo "  Alpha: $alpha"
            echo "  Layers: $layers"
            echo "  Policy: $policy"
            echo "  Output: $RUN_DIR"
            echo ""
            
            mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}
            
            # =====================================================
            # Step 1: ETL_B with this policy
            # =====================================================
            echo "Step 1: Creating lossmasks with policy=$policy"
            
            etl_start=$(date +%s)
            for trace_file in "${TRACE_FILES[@]}"; do
                basename=$(basename "$trace_file" .jsonl)
                render_out="$RUN_DIR/renders/${basename}.jsonl"
                lossmask_out="$RUN_DIR/lossmasks/${basename}.jsonl"
                
                echo "  Processing: $basename"
                python src/schemas/tools/ETL_B.py \
                    --traces "$trace_file" \
                    --render-out "$render_out" \
                    --lossmask-out "$lossmask_out" \
                    --tokenizer "$TOKENIZER" \
                    --max-length 4096 \
                    --policy-override "$policy" 2>&1 | tee -a "$RUN_DIR/etl_b.log"
            done
            etl_end=$(date +%s)
            echo "  ETL_B completed in $((etl_end - etl_start))s"
            echo ""
            
            # Verify lossmasks are non-zero
            echo "  Verifying lossmasks..."
            for lm in "$RUN_DIR/lossmasks"/*.jsonl; do
                nonzero=$(head -5 "$lm" | python3 -c "
import json, sys
total = 0
for line in sys.stdin:
    if line.strip():
        d = json.loads(line)
        mask = d.get('loss_mask', [])
        total += sum(1 for x in mask if x > 0)
print(total)
" 2>/dev/null || echo "0")
                echo "    $(basename "$lm"): $nonzero non-zero mask tokens (first 5 samples)"
            done
            echo ""
            
            # =====================================================
            # Step 2: Split AgentDojo (same as sweep_policies.sbatch)
            # =====================================================
            echo "Step 2: Splitting AgentDojo data"
            
            SPLIT_DIR="$RUN_DIR/agentdojo_split"
            mkdir -p "$SPLIT_DIR"
            
            # Use legacy naming for backward compatibility with existing training args
            python "$REPO_DIR/scripts/split_agentdojo.py" \
                --traces "$AGENTDOJO_TRACES" \
                --renders "$RUN_DIR/renders/agentdojo_complete.jsonl" \
                --lossmasks "$RUN_DIR/lossmasks/agentdojo_complete.jsonl" \
                --output-dir "$SPLIT_DIR" \
                --legacy-harmful-naming 2>&1 | tee -a "$RUN_DIR/split.log"
            
            echo ""
            
            # =====================================================
            # Step 3: Training (same args as sweep_policies.sbatch)
            # =====================================================
            echo "Step 3: Training with alpha=$alpha, layers=$layers, policy=$policy"
            
            # Parse layers to array
            IFS=',' read -ra CB_LAYER_ARRAY <<< "$layers"
            
            # Build training args (matching sweep_policies.sbatch exactly)
            TRAIN_ARGS=(
                --preset "llama-3.1-8b-instruct"
                --model "$TOKENIZER"
                --output-dir "$RUN_DIR/model"
                --total-steps "$TOTAL_STEPS"
                --batch-size "$BATCH_SIZE"
                --learning-rate "$LEARNING_RATE"
                --warmup-steps "$WARMUP_STEPS"
                --alpha-max "$alpha"
                --max-seq-length "$MAX_SEQ_LENGTH"
                --gradient-accumulation-steps 1
                --loss-weighting "dual"
                --lora-r "$LORA_R"
                --lora-alpha "$LORA_ALPHA"
                --lora-dropout 0.05
                --logging-steps 10
                --save-steps 50
                --no-wandb
                --cb-target-layers "${CB_LAYER_ARRAY[@]}"
                --mode mixed
                --harmful-renders "$RUN_DIR/renders/fujitsu_b4_ds.jsonl" "$SPLIT_DIR/agentdojo_renders_harmful.jsonl"
                --harmful-lossmasks "$RUN_DIR/lossmasks/fujitsu_b4_ds.jsonl" "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl"
                --benign-renders "$RUN_DIR/renders/fujitsu_b4_dr.jsonl" "$SPLIT_DIR/agentdojo_renders_benign.jsonl"
                --benign-lossmasks "$RUN_DIR/lossmasks/fujitsu_b4_dr.jsonl" "$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl"
            )
            
            train_start=$(date +%s)
            if python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"; then
                train_status="SUCCESS"
            else
                train_status="FAILED"
            fi
            train_end=$(date +%s)
            echo "  Training $train_status in $((train_end - train_start))s"
            echo ""
            
            # =====================================================
            # Step 4: Evaluation with Examples
            # =====================================================
            if [[ "$train_status" == "SUCCESS" ]]; then
                echo "Step 4: Evaluating model"
                
                # Find trained model
                if [[ -d "$RUN_DIR/model/final" ]]; then
                    ADAPTER_PATH="$RUN_DIR/model/final"
                elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
                    ADAPTER_PATH="$RUN_DIR/model"
                else
                    echo "  WARNING: No adapter found, skipping evaluation"
                    ADAPTER_PATH=""
                fi
                
                if [[ -n "$ADAPTER_PATH" ]]; then
                    # ---------------------------------------------------------
                    # Evaluate on Fujitsu
                    # ---------------------------------------------------------
                    echo "  Evaluating on Fujitsu ($LIMIT_EVAL samples)..."
                    FUJITSU_OUTPUT="$RUN_DIR/eval/fujitsu_eval.json"
                    
                    python src/evaluation/eval.py \
                        --baseline "$TOKENIZER" \
                        --cb-adapter "$ADAPTER_PATH" \
                        --eval-data "$DS_TRACES" \
                        --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                        --output "$FUJITSU_OUTPUT" \
                        --limit "$LIMIT_EVAL" \
                        --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/fujitsu.log"
                    
                    # Print Fujitsu examples
                    if [[ "$SHOW_EXAMPLES" == "true" ]]; then
                        PAIRED_FILE="${FUJITSU_OUTPUT%.json}.paired_outputs.jsonl"
                        if [[ -f "$PAIRED_FILE" ]]; then
                            echo ""
                            echo "    ========================================"
                            echo "    FUJITSU EXAMPLES: $RUN_NAME"
                            echo "    ========================================"
                            python3 << PYEOF
import json

with open('$PAIRED_FILE', 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print('    No paired outputs')
    exit(0)

# Tool-flip comparison (Fujitsu style)
emoji = {'attack_success': '❌', 'correct_behavior': '✅', 'no_tool_call': '⚠️', 'other_tool': '❓'}
improvements = [p for p in pairs if p.get('baseline_outcome') == 'attack_success' and p.get('cb_outcome') != 'attack_success']
regressions = [p for p in pairs if p.get('baseline_outcome') != 'attack_success' and p.get('cb_outcome') == 'attack_success']
print(f'    Pairs: {len(pairs)} | Improved: {len(improvements)} | Regressed: {len(regressions)}')
print()

for i, p in enumerate((improvements + regressions)[:$NUM_EXAMPLES], 1):
    b_out = p.get('baseline_outcome', '?')
    c_out = p.get('cb_outcome', '?')
    label = 'IMP' if b_out == 'attack_success' else 'REG'
    print(f'    [{i}] {label}: {emoji.get(b_out,"?")} {b_out} -> {emoji.get(c_out,"?")} {c_out}')
    print(f'        expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'    --- BASELINE RESPONSE ---')
    print(p.get('baseline_response') or '(empty)')
    print(f'    --- CB MODEL RESPONSE ---')
    print(p.get('cb_response') or '(empty)')
    print()
PYEOF
                        fi
                    fi
                    
                    # ---------------------------------------------------------
                    # Evaluate on AgentDojo (harmful subset)
                    # ---------------------------------------------------------
                    echo "  Evaluating on AgentDojo ($LIMIT_EVAL samples)..."
                    AGENTDOJO_OUTPUT="$RUN_DIR/eval/agentdojo_eval.json"
                    
                    if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
                        python src/evaluation/eval.py \
                            --baseline "$TOKENIZER" \
                            --cb-adapter "$ADAPTER_PATH" \
                            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
                            --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                            --output "$AGENTDOJO_OUTPUT" \
                            --use-sample-context \
                            --limit "$LIMIT_EVAL" \
                            --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/agentdojo.log"
                        
                        # Print AgentDojo examples
                        if [[ "$SHOW_EXAMPLES" == "true" ]]; then
                            PAIRED_FILE="${AGENTDOJO_OUTPUT%.json}.paired_outputs.jsonl"
                            if [[ -f "$PAIRED_FILE" ]]; then
                                echo ""
                                echo "    ========================================"
                                echo "    AGENTDOJO EXAMPLES: $RUN_NAME"
                                echo "    ========================================"
                                python3 << PYEOF
import json

with open('$PAIRED_FILE', 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print('    No paired outputs')
    exit(0)

# Generation comparison (AgentDojo style)
differ = [p for p in pairs if p.get('responses_differ')]
same = [p for p in pairs if not p.get('responses_differ')]
print(f'    Pairs: {len(pairs)} | Different: {len(differ)} ({100*len(differ)/len(pairs) if pairs else 0:.0f}%) | Same: {len(same)}')
print()

for i, p in enumerate(differ[:$NUM_EXAMPLES], 1):
    cat = p.get('category', '?')
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    print(f'    [{i}] {cat}: baseline={b_tool} -> cb={c_tool}')
    print(f'    --- BASELINE RESPONSE ---')
    print(p.get('baseline_response') or '(empty)')
    print(f'    --- CB MODEL RESPONSE ---')
    print(p.get('cb_response') or '(empty)')
    print()
PYEOF
                            fi
                        fi
                    fi
                    
                    success_count=$((success_count + 1))
                fi
            else
                fail_count=$((fail_count + 1))
            fi
            
            # =====================================================
            # Step 5: Quick Summary
            # =====================================================
            echo ""
            echo "--- Run Summary: $RUN_NAME ---"
            echo "  Training: $train_status"
            if [[ -f "$RUN_DIR/eval/fujitsu_eval.json" ]]; then
                python3 -c "
import json
with open('$RUN_DIR/eval/fujitsu_eval.json') as f:
    r = json.load(f)
if 'baseline' in r:
    baseline_asr = r['baseline']['tool_flip_asr']['attack_success_rate']
    print(f'  Fujitsu Baseline ASR: {baseline_asr:.1%}')
if 'cb_model' in r:
    asr = r['cb_model']['tool_flip_asr']['attack_success_rate']
    print(f'  Fujitsu CB ASR:       {asr:.1%}')
if 'delta' in r:
    delta = r['delta']['tool_flip_asr']
    print(f'  Fujitsu Delta:        {delta:+.1%}')
" 2>/dev/null || echo "  Fujitsu: (parse error)"
            fi
            if [[ -f "$RUN_DIR/eval/agentdojo_eval.json" ]]; then
                python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_eval.json') as f:
    r = json.load(f)
if 'output_comparison' in r:
    diff = r['output_comparison'].get('difference_rate', -1)
    print(f'  AgentDojo Diff Rate:  {diff:.1%}')
" 2>/dev/null || echo "  AgentDojo: (parse error)"
            fi
            echo ""
            
        done
    done
done

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Total configurations: $TOTAL_RUNS"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo ""
echo "Sweep directory: $SWEEP_DIR"
echo ""

# Generate summary CSV
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
echo "alpha,layers,policy,fujitsu_baseline_asr,fujitsu_cb_asr,fujitsu_delta,agentdojo_diff,status" > "$SUMMARY_CSV"

for run_dir in "$SWEEP_DIR"/a*; do
    if [[ -d "$run_dir" ]]; then
        run_name=$(basename "$run_dir")
        
        # Parse run name to extract params
        # Format: a${alpha}_l${layers}_${policy}
        alpha=$(echo "$run_name" | sed -n 's/^a\([0-9.]*\)_.*/\1/p')
        layers=$(echo "$run_name" | sed -n 's/.*_l\([0-9_]*\)_.*/\1/p' | tr '_' ',')
        policy=$(echo "$run_name" | sed -n 's/.*_l[0-9_]*_\(.*\)/\1/p')
        
        # Get metrics
        fujitsu_baseline="N/A"
        fujitsu_cb="N/A"
        fujitsu_delta="N/A"
        agentdojo_diff="N/A"
        status="unknown"
        
        if [[ -f "$run_dir/eval/fujitsu_eval.json" ]]; then
            read fujitsu_baseline fujitsu_cb fujitsu_delta < <(python3 -c "
import json
with open('$run_dir/eval/fujitsu_eval.json') as f:
    r = json.load(f)
baseline = r.get('baseline', {}).get('tool_flip_asr', {}).get('attack_success_rate', 'N/A')
cb = r.get('cb_model', {}).get('tool_flip_asr', {}).get('attack_success_rate', 'N/A')
delta = r.get('delta', {}).get('tool_flip_asr', 'N/A')
print(baseline, cb, delta)
" 2>/dev/null || echo "N/A N/A N/A")
            status="success"
        fi
        
        if [[ -f "$run_dir/eval/agentdojo_eval.json" ]]; then
            agentdojo_diff=$(python3 -c "
import json
with open('$run_dir/eval/agentdojo_eval.json') as f:
    r = json.load(f)
print(r.get('output_comparison', {}).get('difference_rate', 'N/A'))
" 2>/dev/null || echo "N/A")
        fi
        
        echo "$alpha,$layers,$policy,$fujitsu_baseline,$fujitsu_cb,$fujitsu_delta,$agentdojo_diff,$status" >> "$SUMMARY_CSV"
    fi
done

echo "Summary CSV: $SUMMARY_CSV"
echo ""
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo ""

# =============================================================================
# Best Results
# =============================================================================
echo ""
echo "--- BEST RESULTS (Lowest CB ASR) ---"
python3 -c "
import csv

try:
    with open('$SUMMARY_CSV', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter valid results and sort by CB ASR
    valid = []
    for r in rows:
        try:
            cb_asr = float(r['fujitsu_cb_asr'])
            valid.append((cb_asr, r))
        except:
            pass
    
    if valid:
        valid.sort(key=lambda x: x[0])
        best = valid[0]
        print(f'Best Fujitsu CB ASR: {best[0]:.1%}')
        print(f'  Alpha: {best[1]["alpha"]}')
        print(f'  Layers: {best[1]["layers"]}')
        print(f'  Policy: {best[1]["policy"]}')
        print(f'  Baseline ASR: {best[1]["fujitsu_baseline_asr"]}')
        print(f'  Delta: {best[1]["fujitsu_delta"]}')
        print(f'  AgentDojo Diff: {best[1]["agentdojo_diff"]}')
except Exception as e:
    print(f'Could not analyze results: {e}')
" 2>/dev/null || echo "(analysis failed)"

echo ""
echo "Finished at: $(date)"
