#!/bin/bash
#SBATCH --job-name=etl_b_lossmask
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 3: Render and Loss Mask (ETL_B)
# Converts trace_v1 -> render_v1 + lossmask_v1
# CPU-only (tokenization only, no GPU needed)
# =============================================================================
#
# This script:
# 1. Renders B2 complete traces using tokenizer's chat template
# 2. Applies LMP (Loss Mask Policy) to create per-token masks
# 3. Optionally applies MWCS (Mixture Weighted Curriculum Scheduling)
# 4. Outputs render_v1 and lossmask_v1 JSONL files
#
# Configuration via environment variables:
#   INPUT_TRACES      - Input trace JSONL file or comma-separated list
#   OUTPUT_DIR        - Directory for renders/lossmasks (default: $CB_SCRATCH/data)
#   TOKENIZER         - Tokenizer to use (default: meta-llama/Llama-3.1-8B-Instruct)
#   MAX_LENGTH        - Max sequence length (default: 4096)
#   POLICY            - LMP policy override (optional, uses trace's policy if not set)
#   MWCS_SCHEDULE     - Path to MWCS schedule YAML (optional)
#   MWCS_STEP         - Training step for curriculum (optional)
#   ALLOW_SKELETON    - Process skeleton traces (default: false)
#   SKELETON_POLICY   - LMP policy for skeletons (default: full_sequence)
#   INCLUDE_TEXT      - Include rendered_text in output (default: false)
#   FORCE_LLAMA       - Force Llama 3.1 tool call format (default: false)
#
# LMP Policy Options:
#   assistant_only    - Loss only on assistant messages (default for most)
#   completion_only   - Loss only on final assistant message
#   full_sequence     - Loss on all tokens
#   cb_full_sequence  - Loss on all non-system tokens (CB default)
#   tool_calls_only   - Loss only on tool call spans
#   action_prefix_only - Loss up to tool name in tool call
#   action_commitment  - Loss on action commitment tokens
#
# Submit:
#   sbatch slurm/pipeline/03_lossmask.sbatch
#
# Or process specific files:
#   INPUT_TRACES=/path/to/ds.jsonl,/path/to/dr.jsonl sbatch slurm/pipeline/03_lossmask.sbatch
#
# Or with specific policy:
#   POLICY=cb_full_sequence sbatch slurm/pipeline/03_lossmask.sbatch
#
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

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "Stage 3: Render and Loss Mask (ETL_B)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
# Auto-discover trace files if INPUT_TRACES not explicitly set
if [[ -z "${INPUT_TRACES:-}" ]]; then
    DISCOVERED_TRACES=""
    # Fujitsu B4 DS/DR traces
    if [[ -f "$CB_SCRATCH/data/traces/fujitsu_b4_ds.jsonl" ]]; then
        DISCOVERED_TRACES="$CB_SCRATCH/data/traces/fujitsu_b4_ds.jsonl"
    fi
    if [[ -f "$CB_SCRATCH/data/traces/fujitsu_b4_dr.jsonl" ]]; then
        DISCOVERED_TRACES="${DISCOVERED_TRACES:+$DISCOVERED_TRACES,}$CB_SCRATCH/data/traces/fujitsu_b4_dr.jsonl"
    fi
    # AgentDojo complete traces
    if [[ -f "$CB_SCRATCH/data/traces/agentdojo_complete.jsonl" ]]; then
        DISCOVERED_TRACES="${DISCOVERED_TRACES:+$DISCOVERED_TRACES,}$CB_SCRATCH/data/traces/agentdojo_complete.jsonl"
    fi
    INPUT_TRACES="$DISCOVERED_TRACES"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$CB_SCRATCH/data}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
POLICY="${POLICY:-}"
MWCS_SCHEDULE="${MWCS_SCHEDULE:-}"
MWCS_STEP="${MWCS_STEP:-}"
ALLOW_SKELETON="${ALLOW_SKELETON:-false}"
SKELETON_POLICY="${SKELETON_POLICY:-full_sequence}"
INCLUDE_TEXT="${INCLUDE_TEXT:-false}"
FORCE_LLAMA="${FORCE_LLAMA:-false}"
LMP_REGISTRY="${LMP_REGISTRY:-}"

# Resolve tokenizer to absolute path if it's a repo ID to avoid offline mode issues
# This bypasses transformers' "is_base_mistral" check which triggers network calls
if [[ "$TOKENIZER" != /* ]]; then
    echo "Resolving tokenizer '$TOKENIZER' to local cache path..."

    # Direct path to cached model (hardcoded for reliability)
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
fi

RENDER_DIR="$OUTPUT_DIR/renders"
LOSSMASK_DIR="$OUTPUT_DIR/lossmasks"

mkdir -p "$RENDER_DIR" "$LOSSMASK_DIR"

echo "Configuration:"
echo "  INPUT_TRACES: $INPUT_TRACES"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  RENDER_DIR: $RENDER_DIR"
echo "  LOSSMASK_DIR: $LOSSMASK_DIR"
echo "  TOKENIZER: $TOKENIZER"
echo "  MAX_LENGTH: $MAX_LENGTH"
echo "  POLICY: ${POLICY:-<from trace>}"
echo "  MWCS_SCHEDULE: ${MWCS_SCHEDULE:-<none>}"
echo "  MWCS_STEP: ${MWCS_STEP:-<none>}"
echo "  ALLOW_SKELETON: $ALLOW_SKELETON"
echo "  SKELETON_POLICY: $SKELETON_POLICY"
echo "  INCLUDE_TEXT: $INCLUDE_TEXT"
echo "  FORCE_LLAMA: $FORCE_LLAMA"
echo ""

# =============================================================================
# Build base command arguments
# =============================================================================
build_etl_b_args() {
    local trace_file="$1"
    local render_out="$2"
    local lossmask_out="$3"

    local args=(
        --traces "$trace_file"
        --render-out "$render_out"
        --lossmask-out "$lossmask_out"
        --tokenizer "$TOKENIZER"
        --max-length "$MAX_LENGTH"
    )

    if [[ -n "$POLICY" ]]; then
        args+=(--policy-override "$POLICY")
    fi

    if [[ -n "$MWCS_SCHEDULE" && -f "$MWCS_SCHEDULE" ]]; then
        args+=(--lmp-schedule "$MWCS_SCHEDULE")
    fi

    if [[ -n "$MWCS_STEP" ]]; then
        args+=(--step "$MWCS_STEP")
    fi

    if [[ "$ALLOW_SKELETON" == "true" ]]; then
        args+=(--allow-skeleton --skeleton-policy "$SKELETON_POLICY")
    fi

    if [[ "$INCLUDE_TEXT" == "true" ]]; then
        args+=(--include-rendered-text)
    fi

    if [[ "$FORCE_LLAMA" == "true" ]]; then
        args+=(--force-llama-format)
    fi

    if [[ -n "$LMP_REGISTRY" && -f "$LMP_REGISTRY" ]]; then
        args+=(--lmp-registry "$LMP_REGISTRY")
    fi

    echo "${args[@]}"
}

# =============================================================================
# Process each input file
# =============================================================================
IFS=',' read -ra TRACE_FILES <<< "$INPUT_TRACES"

total_processed=0
total_skipped=0

for trace_file in "${TRACE_FILES[@]}"; do
    # Trim whitespace
    trace_file=$(echo "$trace_file" | xargs)

    if [[ ! -f "$trace_file" ]]; then
        echo "WARNING: Trace file not found, skipping: $trace_file"
        continue
    fi

    # Derive output names from input filename
    basename=$(basename "$trace_file" .jsonl)
    render_out="$RENDER_DIR/${basename}.jsonl"
    lossmask_out="$LOSSMASK_DIR/${basename}.jsonl"

    echo "========================================"
    echo "Processing: $basename"
    echo "========================================"
    echo "  Input: $trace_file"
    echo "  Lines: $(wc -l < "$trace_file")"
    echo "  Render output: $render_out"
    echo "  Lossmask output: $lossmask_out"
    echo ""

    # Build arguments
    args=$(build_etl_b_args "$trace_file" "$render_out" "$lossmask_out")

    # Run ETL_B
    python src/schemas/tools/ETL_B.py $args

    echo ""
    if [[ -f "$render_out" ]]; then
        lines=$(wc -l < "$render_out")
        echo "  Renders written: $lines"
        total_processed=$((total_processed + lines))
    fi
    if [[ -f "$lossmask_out" ]]; then
        echo "  Lossmasks written: $(wc -l < "$lossmask_out")"
    fi
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
echo "Stage 3 Complete: Render and Loss Mask"
echo "========================================"
echo ""
echo "Total traces processed: $total_processed"
echo ""
echo "Output directories:"
echo "  Renders: $RENDER_DIR"
echo "  Lossmasks: $LOSSMASK_DIR"
echo ""
echo "Render files:"
ls -la "$RENDER_DIR"/*.jsonl 2>/dev/null || echo "  (no files found)"
echo ""
echo "Lossmask files:"
ls -la "$LOSSMASK_DIR"/*.jsonl 2>/dev/null || echo "  (no files found)"
echo ""
echo "Pipeline complete! Data is ready for training."
