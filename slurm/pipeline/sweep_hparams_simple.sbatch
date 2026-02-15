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
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Changed directory to: $(pwd)"
else
    echo "WARNING: Repo directory not found at $REPO_DIR. Assuming current directory is correct."
fi

# Load modules and activate venv (Required for data generation)
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "WARNING: Venv not found at $VENV_DIR"
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
export ALPHAS="${ALPHAS:-20.0}"
export LAYER_CONFIGS="${LAYER_CONFIGS:-10,20}"
export POLICIES="${POLICIES:-cb_full_sequence}"
export LIMIT_EVAL="${LIMIT_EVAL:-50}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-50}"
export SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export LLMAIL_LIMIT="${LLMAIL_LIMIT:-}"  # e.g. LLMAIL_LIMIT=3000 to cap LLMail samples

echo "Starting sweep_hparams_simple.sbatch..."
echo "  ALPHAS: ${ALPHAS}"
echo "  LAYER_CONFIGS: ${LAYER_CONFIGS}"
echo "  POLICIES: ${POLICIES}"
echo "  BATCH_SIZE: ${BATCH_SIZE} (effective: $((BATCH_SIZE * GRAD_ACCUM)) with grad_accum=${GRAD_ACCUM})"
echo "  Datasets: Fujitsu, AgentDojo, LLMail (auto-detected)"
echo "  Logging: Compatible with scripts/visualize_sweep_results.py"

# =============================================================================
# Cache Setup (Offline Mode)
# =============================================================================
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve Tokenizer to Local Cache (for generate_completions.py)
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export PRESET="${PRESET:-llama-3.1-8b-instruct}"
MODEL_NAME="$MODEL_ID"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | tr '/' '--')
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        echo "Found local snapshot: $SNAPSHOT_PATH"
        MODEL="$SNAPSHOT_PATH"
    else
        echo "WARNING: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        # Fallback to name, will fail if offline
        MODEL="$MODEL_NAME"
    fi
else
    echo "WARNING: Model not cached at: $MODEL_CACHE_DIR"
    MODEL="$MODEL_NAME"
fi

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
# MODEL is now set above via cache resolution

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
                LIMIT_ARGS=()
                if [[ -n "$LLMAIL_LIMIT" ]]; then
                    LIMIT_ARGS+=(--llmail-inject-limit "$LLMAIL_LIMIT")
                fi
                python src/schemas/tools/ETL_A.py \
                    --llmail-inject "$raw_source" \
                    --output "$skeleton" \
                    --split train \
                    --tool-schema "$schema" \
                    "${LIMIT_ARGS[@]}"
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
                    --max-model-len "$MAX_MODEL_LEN" \
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
                    --max-model-len "$MAX_MODEL_LEN" \
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
