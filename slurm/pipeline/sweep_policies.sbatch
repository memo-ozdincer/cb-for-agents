#!/bin/bash
#SBATCH --job-name=sweep_policies
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Sweep Loss Mask Policies
# Tests different LMP policies with both Fujitsu and AgentDojo data
# Runs ETL_B for each policy, then trains on each
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
echo "SWEEP: LOSS MASK POLICIES"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Input Traces (already exist from stages 1-2)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"

INPUT_TRACES="$DS_TRACES,$DR_TRACES,$AGENTDOJO_TRACES"

echo "Input traces:"
echo "  DS: $(wc -l < "$DS_TRACES") lines"
echo "  DR: $(wc -l < "$DR_TRACES") lines"
echo "  AgentDojo: $(wc -l < "$AGENTDOJO_TRACES") lines"
echo ""

# Resolve tokenizer to local cache path
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"
CACHE_MODEL_NAME="models--${TOKENIZER//\//--}"
SNAPSHOT_ROOT="$HF_HUB_CACHE/$CACHE_MODEL_NAME/snapshots"
if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        TOKENIZER="$SNAPSHOT_PATH"
        echo "Using cached tokenizer: $TOKENIZER"
    fi
fi
echo ""

# =============================================================================
# Policies to Test
# =============================================================================
POLICIES=(
    "assistant_only"
    "cb_full_sequence"
    "tool_calls_only"
    "completion_only"
)

# Training configs
TOTAL_STEPS=300
BATCH_SIZE=8
ALPHA_MAX=10.0
CB_LAYERS="10,20"

# Create sweep directory
SWEEP_DIR="$CB_SCRATCH/sweeps/policy_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

echo "Policies to test: ${POLICIES[*]}"
echo "Sweep output: $SWEEP_DIR"
echo ""

# =============================================================================
# Run Sweep
# =============================================================================
run_idx=0

for policy in "${POLICIES[@]}"; do
    run_idx=$((run_idx + 1))

    echo "========================================"
    echo "Run $run_idx: Policy = $policy"
    echo "========================================"

    # Create output directories for this policy
    POLICY_DIR="$SWEEP_DIR/$policy"
    RENDERS_DIR="$POLICY_DIR/renders"
    LOSSMASKS_DIR="$POLICY_DIR/lossmasks"
    MODEL_DIR="$POLICY_DIR/model"

    mkdir -p "$RENDERS_DIR" "$LOSSMASKS_DIR" "$MODEL_DIR"

    # =========================================================================
    # Step 1: Run ETL_B with this policy
    # =========================================================================
    echo "Step 1: Creating lossmasks with policy=$policy"

    for trace_file in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
        basename=$(basename "$trace_file" .jsonl)
        render_out="$RENDERS_DIR/${basename}.jsonl"
        lossmask_out="$LOSSMASKS_DIR/${basename}.jsonl"

        echo "  Processing: $basename"

        python src/schemas/tools/ETL_B.py \
            --traces "$trace_file" \
            --render-out "$render_out" \
            --lossmask-out "$lossmask_out" \
            --tokenizer "$TOKENIZER" \
            --max-length 4096 \
            --policy-override "$policy"
    done

    echo ""
    echo "  Renders: $(ls -1 "$RENDERS_DIR"/*.jsonl 2>/dev/null | wc -l) files"
    echo "  Lossmasks: $(ls -1 "$LOSSMASKS_DIR"/*.jsonl 2>/dev/null | wc -l) files"
    echo ""

    # =========================================================================
    # Step 2: Train with this policy's lossmasks
    # =========================================================================
    echo "Step 2: Training with policy=$policy"

    # Parse CB_LAYERS
    IFS=',' read -ra CB_LAYER_ARRAY <<< "$CB_LAYERS"

    TRAIN_ARGS=(
        --preset "llama-3.1-8b-instruct"
        --output-dir "$MODEL_DIR"
        --total-steps "$TOTAL_STEPS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "5e-5"
        --warmup-steps 20
        --alpha-max "$ALPHA_MAX"
        --max-seq-length 2048
        --gradient-accumulation-steps 1
        --loss-weighting "dual"
        --lora-r 16
        --lora-alpha 32
        --lora-dropout 0.05
        --logging-steps 10
        --save-steps 50
        --wandb-project "circuit-breakers-policy-sweep"
        --wandb-run-name "policy_${policy}"
        --cb-target-layers "${CB_LAYER_ARRAY[@]}"
        --mode ds_dr
        --ds-renders "$RENDERS_DIR/fujitsu_b4_ds.jsonl"
        --ds-lossmasks "$LOSSMASKS_DIR/fujitsu_b4_ds.jsonl"
        --dr-renders "$RENDERS_DIR/fujitsu_b4_dr.jsonl"
        --dr-lossmasks "$LOSSMASKS_DIR/fujitsu_b4_dr.jsonl"
        --additional-renders "$RENDERS_DIR/agentdojo_complete.jsonl"
        --additional-lossmasks "$LOSSMASKS_DIR/agentdojo_complete.jsonl"
        --additional-traces "$AGENTDOJO_TRACES"
    )

    python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$POLICY_DIR/train.log"

    echo ""
    echo "Run $run_idx complete: $policy"
    echo "  Model: $MODEL_DIR"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "POLICY SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Total runs: $run_idx"
echo "Sweep directory: $SWEEP_DIR"
echo ""
echo "Results:"
for policy in "${POLICIES[@]}"; do
    echo "  $policy:"
    if [[ -d "$SWEEP_DIR/$policy/model/final" ]]; then
        echo "    ✓ Model trained"
    else
        echo "    ✗ No final model"
    fi
done
echo ""
echo "Finished at: $(date)"
