#!/bin/bash
#SBATCH --job-name=sweep_hparams_simple
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
# Ensure we are running from the repo root
PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Changed directory to: $(pwd)"
else
    echo "WARNING: Repo directory not found at $REPO_DIR. Assuming current directory is correct."
fi

# Locate the sweep script
SWEEP_SCRIPT="slurm/pipeline/sweep_hparams.sbatch"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    # Try absolute path
    SWEEP_SCRIPT="$REPO_DIR/slurm/pipeline/sweep_hparams.sbatch"
fi

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    echo "ERROR: Sweep script not found at: $SWEEP_SCRIPT"
    echo "Current directory: $(pwd)"
    echo "Contents of slurm/pipeline:"
    ls -l slurm/pipeline || echo "(dir not found)"
    exit 1
fi

export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export ALPHAS="${ALPHAS:-5.0,10.0,15.0}"
export LAYER_CONFIGS="${LAYER_CONFIGS:-10,20}"
export POLICIES="${POLICIES:-cb_full_sequence}"
export LIMIT_EVAL="${LIMIT_EVAL:-100}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-50}"
export SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

echo "Starting sweep_hparams_simple.sbatch..."
echo "  ALPHAS: ${ALPHAS}"
echo "  LAYER_CONFIGS: ${LAYER_CONFIGS}"
echo "  POLICIES: ${POLICIES}"
echo "  Datasets: Fujitsu, AgentDojo, LLMail (auto-detected)"
echo "  Logging: Compatible with scripts/visualize_sweep_results.py"

# =============================================================================
# Auto-Generation of Missing Data
# =============================================================================
# Check if Fujitsu DS/DR traces exist. If not, generate them.
# This ensures the sweep can run even if data prep wasn't done manually.

TRACES_DIR="/scratch/memoozd/cb-scratch/data/traces"
FUJITSU_SKELETON="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# LLMail paths
LLMAIL_SKELETON="$TRACES_DIR/llmail_inject_skeleton.jsonl"
LLMAIL_DS="$TRACES_DIR/llmail_inject_ds.jsonl"
LLMAIL_DR="$TRACES_DIR/llmail_inject_dr.jsonl"
LLMAIL_TOOL_SCHEMA="configs/tool_schemas/llmail_inject_v1.json"
LLMAIL_RAW="$REPO_DIR/data/llmail_inject/raw_submissions_phase1.jsonl"

# Helper to generate DS/DR
generate_data() {
    local name="$1"
    local skeleton="$2"
    local ds="$3"
    local dr="$4"
    local schema="$5"
    local raw_source="${6:-}"  # Optional raw source for ETL_A

    # 1. Check for Skeleton (ETL_A)
    if [[ ! -f "$skeleton" ]]; then
        if [[ -n "$raw_source" && -f "$raw_source" ]]; then
            echo "Generating missing skeleton for $name (ETL_A)..."
            
            # Special handling for LLMail vs generic
            if [[ "$name" == "LLMail-Inject" ]]; then
                python src/schemas/tools/ETL_A.py \
                    --llmail-inject "$raw_source" \
                    --output "$skeleton" \
                    --split train \
                    --tool-schema "$schema"
            else
                echo "WARNING: No ETL_A logic defined for $name in simple script."
            fi
        fi
    fi

    # 2. Check for DS/DR (Generation)
    if [[ ! -f "$ds" || ! -f "$dr" ]]; then
        if [[ -f "$skeleton" ]]; then
            echo "Generating missing $name data (DS/DR)..."
            
            # Generate DS (Harmful)
            if [[ ! -f "$ds" ]]; then
                echo "  Generating DS (Harmful)..."
                python src/data_generation/generate_completions.py \
                    --traces "$skeleton" \
                    --output "$ds" \
                    --model "$MODEL" \
                    --tool-schema "$schema" \
                    --mode ds \
                    --temperature-ds 0.7 \
                    --use-vllm
            fi
            
            # Generate DR (Benign)
            if [[ ! -f "$dr" ]]; then
                echo "  Generating DR (Benign)..."
                python src/data_generation/generate_completions.py \
                    --ds-data "$ds" \
                    --output "$dr" \
                    --model "$MODEL" \
                    --tool-schema "$schema" \
                    --mode dr \
                    --temperature-dr 0.3 \
                    --use-vllm
            fi
        else
            echo "WARNING: Skeleton file for $name not found at $skeleton. Cannot generate data."
        fi
    else
        echo "$name data already exists. Skipping generation."
    fi
}

# Run generation
generate_data "Fujitsu B4" "$FUJITSU_SKELETON" "$FUJITSU_DS" "$FUJITSU_DR" "$TOOL_SCHEMA"
generate_data "LLMail-Inject" "$LLMAIL_SKELETON" "$LLMAIL_DS" "$LLMAIL_DR" "$LLMAIL_TOOL_SCHEMA" "$LLMAIL_RAW"

echo ""
echo "Data preparation complete. Launching sweep..."
echo ""

exec bash "$SWEEP_SCRIPT"
