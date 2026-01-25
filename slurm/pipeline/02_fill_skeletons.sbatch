#!/bin/bash
#SBATCH --job-name=fill_skeletons
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 2: Fill Skeletons (generate_completions.py)
# Generates assistant completions for B1 skeleton traces -> B2 complete traces
# GPU required for inference
# =============================================================================
#
# This script:
# 1. Takes B1 skeleton traces (no assistant messages)
# 2. Generates DS (follows_injection) completions - harmful examples (observed == simulated)
# 3. Generates DR (ignores_injection) from DS - benign examples (observed == expected)
# 4. Outputs B2 complete traces ready for ETL_B
#
# Note: DR mode is downstream from DS. It takes successful DS flips, removes injection,
# and regenerates. Use MODE=both for the full pipeline.
#
# Configuration via environment variables:
#   INPUT_TRACES    - Input skeleton traces JSONL for DS/both modes (default: fujitsu_b4_skeletons.jsonl)
#   DS_DATA         - Input DS traces JSONL for DR mode (default: fujitsu_b4_ds.jsonl)
#   OUTPUT_DIR      - Directory for output (default: $CB_SCRATCH/data/traces)
#   MODE            - Generation mode: ds, dr, or both (default: both)
#   MODEL           - Model to use (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOOL_SCHEMA     - Path to tool schema JSON
#   USE_VLLM        - Use vLLM backend (default: true)
#   TENSOR_PARALLEL - Tensor parallel size (default: 1)
#   BATCH_SIZE      - Batch size for inference (default: 32)
#   TEMPERATURE_DS  - Temperature for DS generation (default: 0.7)
#   TEMPERATURE_DR  - Temperature for DR generation (default: 0.3)
#   MAX_TOKENS      - Max tokens to generate (default: 256)
#   LIMIT           - Limit number of traces to process (optional)
#
# Optional output/debug flags:
#   PRINT_EXAMPLES  - Print collected examples (true/false, default: false)
#   NO_TRUNCATE     - Do not truncate example printing (true/false, default: false)
#   NO_WRITE_IDS    - Do not write *.ids.txt (true/false, default: false)
#   EXAMPLES_OUT    - Path for examples JSON (only for MODE=ds or MODE=dr)
#
# Optional example counts (mirrors generate_ds/generate_dr defaults):
#   N_SUCCESSFUL    - DS successful flips to collect (default: 10)
#   N_CORRECT       - Correct behavior examples (default: 5)
#   N_WRONG_TOOL    - DR wrong tool examples (default: 5)
#   N_NO_TOOL       - No tool call examples (default: 5)
#   N_OTHER_TOOL    - DS other-tool examples (default: 5)
#   N_FORMAT_ERRORS - Format error examples (default: 5)
#
# Submit:
#   sbatch slurm/pipeline/02_fill_skeletons.sbatch
#
# Or with custom settings:
#   MODE=ds TEMPERATURE_DS=0.5 sbatch slurm/pipeline/02_fill_skeletons.sbatch
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
echo "Stage 2: Fill Skeletons"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
INPUT_TRACES="${INPUT_TRACES:-$CB_SCRATCH/data/traces/fujitsu_b4_skeletons.jsonl}"
DS_DATA="${DS_DATA:-$CB_SCRATCH/data/traces/fujitsu_b4_ds.jsonl}"
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
LIMIT="${LIMIT:-}"

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

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  INPUT_TRACES: $INPUT_TRACES"
echo "  DS_DATA: $DS_DATA (for DR mode)"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODE: $MODE"
echo "  MODEL: $MODEL"
echo "  TOOL_SCHEMA: $TOOL_SCHEMA"
echo "  USE_VLLM: $USE_VLLM"
echo "  TENSOR_PARALLEL: $TENSOR_PARALLEL"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  TEMPERATURE_DS: $TEMPERATURE_DS"
echo "  TEMPERATURE_DR: $TEMPERATURE_DR"
echo "  MAX_TOKENS: $MAX_TOKENS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "  LIMIT: ${LIMIT:-unlimited}"
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
    echo "Please run 01_load_data.sbatch first or set INPUT_TRACES."
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
)

if [[ "$USE_VLLM" == "true" ]]; then
    COMMON_ARGS+=(
        --use-vllm
        --tensor-parallel-size "$TENSOR_PARALLEL"
        --max-model-len "$MAX_MODEL_LEN"
        --batch-size "$BATCH_SIZE"
    )
fi

if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi

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

# =============================================================================
# Derive output filenames from input
# =============================================================================
INPUT_BASENAME=$(basename "$INPUT_TRACES" .jsonl)
# Remove _skeletons suffix if present for cleaner naming
OUTPUT_BASE="${INPUT_BASENAME/_skeletons/}"

DS_OUTPUT="$OUTPUT_DIR/${OUTPUT_BASE}_ds.jsonl"
DR_OUTPUT="$OUTPUT_DIR/${OUTPUT_BASE}_dr.jsonl"

echo "Output files:"
echo "  DS: $DS_OUTPUT"
echo "  DR: $DR_OUTPUT"
echo ""

# =============================================================================
# Run Generation
# =============================================================================
case "$MODE" in
    ds)
        echo "========================================"
        echo "Generating DS (follows_injection) completions"
        echo "========================================"
        echo "Model follows injection and calls wrong tool (harmful examples)"
        echo ""

        python src/data_generation/generate_completions.py \
            "${COMMON_ARGS[@]}" \
            --mode ds \
            --output "$DS_OUTPUT" \
            ${EXAMPLES_OUT:+--examples-out "$EXAMPLES_OUT"}

        echo ""
        echo "DS generation complete!"
        echo "  Output: $DS_OUTPUT"
        if [[ -f "$DS_OUTPUT" ]]; then
            echo "  Lines: $(wc -l < "$DS_OUTPUT")"
        fi
        ;;

    dr)
        echo "========================================"
        echo "Generating DR (ignores_injection) completions"
        echo "========================================"
        echo "Model ignores injection and calls correct tool (benign examples)"
        echo ""

        if [[ ! -f "$DS_DATA" ]]; then
            echo "ERROR: DS data not found: $DS_DATA"
            echo "DR mode requires DS traces. Run MODE=ds first or use MODE=both."
            exit 1
        fi

        python src/data_generation/generate_completions.py \
            --ds-data "$DS_DATA" \
            --model "$MODEL" \
            --tool-schema "$TOOL_SCHEMA" \
            --temperature-dr "$TEMPERATURE_DR" \
            --max-tokens "$MAX_TOKENS" \
            --mode dr \
            --output "$DR_OUTPUT" \
            ${USE_VLLM:+--use-vllm} \
            ${TENSOR_PARALLEL:+--tensor-parallel-size "$TENSOR_PARALLEL"} \
            ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
            ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
            ${LIMIT:+--limit "$LIMIT"} \
            ${PRINT_EXAMPLES:+--print-examples} \
            ${NO_TRUNCATE:+--no-truncate} \
            ${NO_WRITE_IDS:+--no-write-ids} \
            --n-successful "$N_SUCCESSFUL" \
            --n-correct "$N_CORRECT" \
            --n-wrong-tool "$N_WRONG_TOOL" \
            --n-no-tool "$N_NO_TOOL" \
            --n-other-tool "$N_OTHER_TOOL" \
            --n-format-errors "$N_FORMAT_ERRORS" \
            ${EXAMPLES_OUT:+--examples-out "$EXAMPLES_OUT"}

        echo ""
        echo "DR generation complete!"
        echo "  Output: $DR_OUTPUT"
        if [[ -f "$DR_OUTPUT" ]]; then
            echo "  Lines: $(wc -l < "$DR_OUTPUT")"
        fi
        ;;

    both)
        echo "========================================"
        echo "Generating DS + DR completions"
        echo "========================================"
        echo "DS: Model follows injection (harmful)"
        echo "DR: Model ignores injection (benign)"
        echo ""

        python src/data_generation/generate_completions.py \
            "${COMMON_ARGS[@]}" \
            --mode both \
            --output-ds "$DS_OUTPUT" \
            --output-dr "$DR_OUTPUT"

        echo ""
        echo "DS + DR generation complete!"
        echo "  DS Output: $DS_OUTPUT"
        if [[ -f "$DS_OUTPUT" ]]; then
            echo "    Lines: $(wc -l < "$DS_OUTPUT")"
        fi
        echo "  DR Output: $DR_OUTPUT"
        if [[ -f "$DR_OUTPUT" ]]; then
            echo "    Lines: $(wc -l < "$DR_OUTPUT")"
        fi
        ;;

    *)
        echo "ERROR: Invalid MODE '$MODE'. Must be 'ds', 'dr', or 'both'."
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Stage 2 Complete: Fill Skeletons"
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*_ds.jsonl "$OUTPUT_DIR"/*_dr.jsonl 2>/dev/null || echo "  (check above for outputs)"
echo ""
echo "Next step: Run 03_lossmask.sbatch to render and apply loss masking"
