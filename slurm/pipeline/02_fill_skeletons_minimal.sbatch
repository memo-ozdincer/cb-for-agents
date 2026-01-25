#!/bin/bash
#SBATCH --job-name=fill_skeletons_minimal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 2 MINIMAL: Fill Skeletons (generate_completions.py)
# Test version - processes only 2000 samples
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

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,xdg_cache,xdg_config,flashinfer,vllm}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_DIR/flashinfer"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CACHE_ROOT="$CACHE_DIR"

echo "========================================"
echo "Stage 2 MINIMAL: Fill Skeletons (2000 samples)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration - MINIMAL TEST VERSION
# =============================================================================
INPUT_TRACES="${INPUT_TRACES:-$CB_SCRATCH/data/traces/fujitsu_b4_skeletons.jsonl}"
DS_DATA="${DS_DATA:-$CB_SCRATCH/data/traces/fujitsu_b4_ds_minimal.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$CB_SCRATCH/data/traces}"
MODE="${MODE:-both}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"
USE_VLLM="${USE_VLLM:-true}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE_DS="${TEMPERATURE_DS:-0.7}"
TEMPERATURE_DR="${TEMPERATURE_DR:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PRINT_EXAMPLES="${PRINT_EXAMPLES:-false}"
NO_TRUNCATE="${NO_TRUNCATE:-false}"
NO_WRITE_IDS="${NO_WRITE_IDS:-false}"
EXAMPLES_OUT="${EXAMPLES_OUT:-}"

N_SUCCESSFUL="${N_SUCCESSFUL:-10}"
N_CORRECT="${N_CORRECT:-5}"
N_WRONG_TOOL="${N_WRONG_TOOL:-5}"
N_NO_TOOL="${N_NO_TOOL:-5}"
N_OTHER_TOOL="${N_OTHER_TOOL:-5}"
N_FORMAT_ERRORS="${N_FORMAT_ERRORS:-5}"

# MINIMAL: Default to 2000 samples (override by setting LIMIT)
LIMIT="${LIMIT:-300}"

mkdir -p "$OUTPUT_DIR"

echo "Configuration (MINIMAL TEST):"
echo "  INPUT_TRACES: $INPUT_TRACES"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODE: $MODE"
echo "  MODEL: $MODEL"
echo "  LIMIT: $LIMIT (MINIMAL TEST)"
echo "  PRINT_EXAMPLES: $PRINT_EXAMPLES"
echo "  NO_TRUNCATE: $NO_TRUNCATE"
echo "  NO_WRITE_IDS: $NO_WRITE_IDS"
echo "  EXAMPLES_OUT: ${EXAMPLES_OUT:-<default>}"
echo "  N_SUCCESSFUL: $N_SUCCESSFUL"
echo "  N_CORRECT: $N_CORRECT"
echo "  N_WRONG_TOOL: $N_WRONG_TOOL"
echo "  N_NO_TOOL: $N_NO_TOOL"
echo "  N_OTHER_TOOL: $N_OTHER_TOOL"
echo "  N_FORMAT_ERRORS: $N_FORMAT_ERRORS"
echo ""

# Validate inputs
if [[ ! -f "$INPUT_TRACES" ]]; then
    echo "ERROR: Input traces not found: $INPUT_TRACES"
    exit 1
fi

if [[ ! -f "$TOOL_SCHEMA" ]]; then
    echo "ERROR: Tool schema not found: $TOOL_SCHEMA"
    exit 1
fi

echo "Input traces: $INPUT_TRACES"
echo "  Lines: $(wc -l < "$INPUT_TRACES")"
echo ""

# =============================================================================
# Build command arguments
# =============================================================================
COMMON_ARGS=(
    --traces "$INPUT_TRACES"
    --model "$MODEL"
    --tool-schema "$TOOL_SCHEMA"
    --temperature-ds "$TEMPERATURE_DS"
    --temperature-dr "$TEMPERATURE_DR"
    --max-tokens "$MAX_TOKENS"
    --limit "$LIMIT"
)

# Example/debug args
if [[ "$PRINT_EXAMPLES" == "true" ]]; then
    COMMON_ARGS+=(--print-examples)
fi
if [[ "$NO_TRUNCATE" == "true" ]]; then
    COMMON_ARGS+=(--no-truncate)
fi
if [[ "$NO_WRITE_IDS" == "true" ]]; then
    COMMON_ARGS+=(--no-write-ids)
fi

COMMON_ARGS+=(
    --n-successful "$N_SUCCESSFUL"
    --n-correct "$N_CORRECT"
    --n-wrong-tool "$N_WRONG_TOOL"
    --n-no-tool "$N_NO_TOOL"
    --n-other-tool "$N_OTHER_TOOL"
    --n-format-errors "$N_FORMAT_ERRORS"
)

if [[ "$MODE" == "both" && -n "$EXAMPLES_OUT" ]]; then
    echo "WARNING: EXAMPLES_OUT is ignored for MODE=both (writes per-output *.examples.json)."
    EXAMPLES_OUT=""
fi

if [[ "$USE_VLLM" == "true" ]]; then
    COMMON_ARGS+=(
        --use-vllm
        --tensor-parallel-size "$TENSOR_PARALLEL"
        --max-model-len "$MAX_MODEL_LEN"
        --batch-size "$BATCH_SIZE"
    )
fi

# =============================================================================
# Derive output filenames - MINIMAL VERSION
# =============================================================================
INPUT_BASENAME=$(basename "$INPUT_TRACES" .jsonl)
OUTPUT_BASE="${INPUT_BASENAME/_skeletons/}"

DS_OUTPUT="$OUTPUT_DIR/${OUTPUT_BASE}_ds_minimal.jsonl"
DR_OUTPUT="$OUTPUT_DIR/${OUTPUT_BASE}_dr_minimal.jsonl"

echo "Output files (MINIMAL):"
echo "  DS: $DS_OUTPUT"
echo "  DR: $DR_OUTPUT"
echo ""

# =============================================================================
# Run Generation
# =============================================================================
case "$MODE" in
    ds)
        echo "Generating DS (follows_injection) - MINIMAL"
        python src/data_generation/generate_completions.py \
            "${COMMON_ARGS[@]}" \
            --mode ds \
            --output "$DS_OUTPUT" \
            ${EXAMPLES_OUT:+--examples-out "$EXAMPLES_OUT"}
        echo "DS output: $DS_OUTPUT ($(wc -l < "$DS_OUTPUT") lines)"
        ;;

    dr)
        echo "Generating DR (ignores_injection) - MINIMAL"
        if [[ ! -f "$DS_DATA" ]]; then
            echo "ERROR: DS data not found: $DS_DATA"
            exit 1
        fi
        python src/data_generation/generate_completions.py \
            --ds-data "$DS_DATA" \
            --model "$MODEL" \
            --tool-schema "$TOOL_SCHEMA" \
            --temperature-dr "$TEMPERATURE_DR" \
            --max-tokens "$MAX_TOKENS" \
            --limit "$LIMIT" \
            --mode dr \
            --output "$DR_OUTPUT" \
            ${USE_VLLM:+--use-vllm} \
            ${TENSOR_PARALLEL:+--tensor-parallel-size "$TENSOR_PARALLEL"} \
            ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
            ${PRINT_EXAMPLES:+--print-examples} \
            ${NO_TRUNCATE:+--no-truncate} \
            ${NO_WRITE_IDS:+--no-write-ids} \
            ${EXAMPLES_OUT:+--examples-out "$EXAMPLES_OUT"}
        echo "DR output: $DR_OUTPUT ($(wc -l < "$DR_OUTPUT") lines)"
        ;;

    both)
        echo "Generating DS + DR - MINIMAL"
        python src/data_generation/generate_completions.py \
            "${COMMON_ARGS[@]}" \
            --mode both \
            --output-ds "$DS_OUTPUT" \
            --output-dr "$DR_OUTPUT"
        echo "DS output: $DS_OUTPUT"
        echo "DR output: $DR_OUTPUT"
        ;;

    *)
        echo "ERROR: Invalid MODE '$MODE'. Must be 'ds', 'dr', or 'both'."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Stage 2 MINIMAL Complete"
echo "========================================"
