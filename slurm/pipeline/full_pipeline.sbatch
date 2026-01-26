#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Full Data Pipeline
# Runs all stages sequentially within a single allocation
# =============================================================================
#
# Pipeline Flow:
#   Raw Data (Tier A)
#         |
#         v
#   01_load_data.sbatch (ETL_A)
#         |
#         +--> B1 Skeletons (Fujitsu B4)
#         |         |
#         |         v
#         |    02_fill_skeletons.sbatch
#         |         |
#         |         +--> DS traces (harmful)
#         |         |
#         |         +--> DR traces (benign)
#         |
#         +--> B2 Complete (AgentDojo) ----+
#                                          |
#                                          v
#                                    03_lossmask.sbatch (ETL_B)
#                                          |
#                                          +--> render_v1 (tokenized)
#                                          |
#                                          +--> lossmask_v1 (training-ready)
#                                                      |
#                                                      v
#                                              04_train.sbatch
#                                                      |
#                                                      v
#                                              05_eval.sbatch
#                                                      |
#                                                      v
#                                              eval_results.json
#
# Configuration via environment variables:
#   FUJITSU_B4_PATH - Path to Fujitsu B4 orchestrator attacks JSONL
#   AGENTDOJO_PATH  - Path to AgentDojo JSONL file
#   OUTPUT_BASE     - Base output directory (default: $CB_SCRATCH/data)
#   MODEL           - Model for completion generation
#   TOKENIZER       - Tokenizer for rendering
#   MODE            - Generation mode: ds, dr, or both (default: both)
#   POLICY          - LMP policy override
#   SKIP_LOAD       - Skip stage 1 if traces already exist (default: false)
#   SKIP_FILL       - Skip stage 2 if completions already exist (default: false)
#   SKIP_LOSSMASK   - Skip stage 3 if lossmasks already exist (default: false)
#   SKIP_TRAIN      - Skip stage 4 training (default: false)
#   SKIP_EVAL       - Skip stage 5 evaluation (default: false)
#
# Training configuration:
#   TOTAL_STEPS     - Training steps (default: 300)
#   BATCH_SIZE      - Per-GPU batch size (default: 8)
#   LEARNING_RATE   - Learning rate (default: 5e-5)
#   ALPHA_MAX       - Initial alpha for RR loss (default: 10.0)
#   CB_LAYERS       - CB target layers, comma-separated (default: 10,20)
#   WANDB_PROJECT   - W&B project name (default: circuit-breakers)
#   NO_WANDB        - Disable W&B logging (default: false)
#
# Submit:
#   sbatch slurm/pipeline/full_pipeline.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
SCRIPT_DIR="$REPO_DIR/slurm/pipeline"

echo "========================================"
echo "FULL DATA PIPELINE"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_BASE="${OUTPUT_BASE:-$CB_SCRATCH/data}"
TRACES_DIR="$OUTPUT_BASE/traces"
RENDERS_DIR="$OUTPUT_BASE/renders"
LOSSMASKS_DIR="$OUTPUT_BASE/lossmasks"

mkdir -p "$TRACES_DIR" "$RENDERS_DIR" "$LOSSMASKS_DIR" "$CB_SCRATCH/logs"

# Pipeline control
SKIP_LOAD="${SKIP_LOAD:-false}"
SKIP_FILL="${SKIP_FILL:-false}"
SKIP_LOSSMASK="${SKIP_LOSSMASK:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

# Default data paths
FUJITSU_B4_PATH="${FUJITSU_B4_PATH:-$REPO_DIR/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl}"
AGENTDOJO_DIR="${AGENTDOJO_DIR:-$REPO_DIR/data/agent_dojo}"

# Model/tokenizer settings
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"

# Generation settings
MODE="${MODE:-both}"
TEMPERATURE_DS="${TEMPERATURE_DS:-0.7}"
TEMPERATURE_DR="${TEMPERATURE_DR:-0.3}"
USE_VLLM="${USE_VLLM:-true}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Lossmask settings
POLICY="${POLICY:-}"
MWCS_SCHEDULE="${MWCS_SCHEDULE:-}"

# Training settings
TOTAL_STEPS="${TOTAL_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
ALPHA_MAX="${ALPHA_MAX:-10.0}"
CB_LAYERS="${CB_LAYERS:-10,20}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
LORA_R="${LORA_R:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-circuit-breakers}"
NO_WANDB="${NO_WANDB:-false}"

# Eval settings
BASELINE_MODEL="${BASELINE_MODEL:-$MODEL}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

# Output directories
MODELS_DIR="$CB_SCRATCH/models"
EVAL_DIR="$CB_SCRATCH/eval"

echo "Output Configuration:"
echo "  OUTPUT_BASE: $OUTPUT_BASE"
echo "  TRACES_DIR: $TRACES_DIR"
echo "  RENDERS_DIR: $RENDERS_DIR"
echo "  LOSSMASKS_DIR: $LOSSMASKS_DIR"
echo ""
echo "Model Configuration:"
echo "  MODEL: $MODEL"
echo "  TOKENIZER: $TOKENIZER"
echo "  MODE: $MODE"
echo ""

# =============================================================================
# Stage 1: Load Raw Data (ETL_A)
# =============================================================================
SKELETON_FILE="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"

if [[ "$SKIP_LOAD" == "true" && -f "$SKELETON_FILE" ]]; then
    echo "========================================"
    echo "SKIPPING Stage 1: Traces already exist"
    echo "========================================"
    echo "  Found: $SKELETON_FILE"
    echo ""
else
    echo "========================================"
    echo "Stage 1: Load Raw Data (ETL_A)"
    echo "========================================"

    export FUJITSU_B4_PATH
    export AGENTDOJO_DIR
    export OUTPUT_DIR="$TRACES_DIR"
    export SPLIT="train"

    bash "$SCRIPT_DIR/01_load_data.sbatch"

    echo ""
fi

# =============================================================================
# Stage 2: Fill Skeletons (generate_completions.py)
# =============================================================================
DS_FILE="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_FILE="$TRACES_DIR/fujitsu_b4_dr.jsonl"

# Determine which files to expect based on mode
FILL_NEEDED=false
if [[ "$MODE" == "ds" || "$MODE" == "both" ]] && [[ ! -f "$DS_FILE" ]]; then
    FILL_NEEDED=true
fi
if [[ "$MODE" == "dr" || "$MODE" == "both" ]] && [[ ! -f "$DR_FILE" ]]; then
    FILL_NEEDED=true
fi

if [[ "$SKIP_FILL" == "true" && "$FILL_NEEDED" == "false" ]]; then
    echo "========================================"
    echo "SKIPPING Stage 2: Completions already exist"
    echo "========================================"
    [[ -f "$DS_FILE" ]] && echo "  Found: $DS_FILE"
    [[ -f "$DR_FILE" ]] && echo "  Found: $DR_FILE"
    echo ""
else
    echo "========================================"
    echo "Stage 2: Fill Skeletons"
    echo "========================================"

    export INPUT_TRACES="$SKELETON_FILE"
    export OUTPUT_DIR="$TRACES_DIR"
    export MODE
    export MODEL
    export TOOL_SCHEMA
    export USE_VLLM
    export TENSOR_PARALLEL
    export TEMPERATURE_DS
    export TEMPERATURE_DR

    bash "$SCRIPT_DIR/02_fill_skeletons.sbatch"

    echo ""
fi

# =============================================================================
# Stage 3: Render and Loss Mask (ETL_B)
# =============================================================================
echo "========================================"
echo "Stage 3: Render and Loss Mask (ETL_B)"
echo "========================================"

# Build list of trace files to process
TRACE_FILES=""

# Add generated DS/DR files
if [[ -f "$DS_FILE" ]]; then
    TRACE_FILES="$DS_FILE"
fi
if [[ -f "$DR_FILE" ]]; then
    if [[ -n "$TRACE_FILES" ]]; then
        TRACE_FILES="$TRACE_FILES,$DR_FILE"
    else
        TRACE_FILES="$DR_FILE"
    fi
fi

# Add AgentDojo complete traces if they exist
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"
if [[ -f "$AGENTDOJO_TRACES" ]]; then
    if [[ -n "$TRACE_FILES" ]]; then
        TRACE_FILES="$TRACE_FILES,$AGENTDOJO_TRACES"
    else
        TRACE_FILES="$AGENTDOJO_TRACES"
    fi
fi

if [[ -z "$TRACE_FILES" ]]; then
    echo "ERROR: No trace files found to process!"
    echo "Please check that stages 1 and 2 completed successfully."
    exit 1
fi

echo "Processing trace files: $TRACE_FILES"

export INPUT_TRACES="$TRACE_FILES"
export OUTPUT_DIR="$OUTPUT_BASE"
export TOKENIZER
export POLICY
export MWCS_SCHEDULE

if [[ "$SKIP_LOSSMASK" == "true" ]]; then
    echo "SKIPPING Stage 3: Lossmasks"
else
    bash "$SCRIPT_DIR/03_lossmask.sbatch"
fi

# =============================================================================
# Stage 4: Training (train_schema.py)
# =============================================================================
MODEL_OUTPUT_DIR="$MODELS_DIR/cb_train_$(date +%Y%m%d_%H%M%S)"

if [[ "$SKIP_TRAIN" == "true" ]]; then
    echo "========================================"
    echo "SKIPPING Stage 4: Training"
    echo "========================================"
    # Try to find latest model
    MODEL_OUTPUT_DIR=$(ls -dt "$MODELS_DIR"/cb_train_*/final 2>/dev/null | head -1 || echo "")
    if [[ -n "$MODEL_OUTPUT_DIR" ]]; then
        MODEL_OUTPUT_DIR=$(dirname "$MODEL_OUTPUT_DIR")
        echo "  Using existing model: $MODEL_OUTPUT_DIR"
    fi
    echo ""
else
    echo "========================================"
    echo "Stage 4: Circuit Breaker Training"
    echo "========================================"

    mkdir -p "$MODEL_OUTPUT_DIR"

    # Build paths to DS/DR renders and lossmasks
    DS_RENDERS="$RENDERS_DIR/fujitsu_b4_ds.jsonl"
    DS_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_ds.jsonl"
    DR_RENDERS="$RENDERS_DIR/fujitsu_b4_dr.jsonl"
    DR_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_dr.jsonl"

    export DATA_MODE="ds_dr"
    export DS_RENDERS
    export DS_LOSSMASKS
    export DR_RENDERS
    export DR_LOSSMASKS
    export OUTPUT_DIR="$MODEL_OUTPUT_DIR"
    export MODEL_PRESET="llama-3.1-8b-instruct"
    export MODEL
    export TOTAL_STEPS
    export BATCH_SIZE
    export LEARNING_RATE
    export WARMUP_STEPS
    export ALPHA_MAX
    export CB_LAYERS
    export LOSS_WEIGHTING
    export LORA_R
    export WANDB_PROJECT
    export NO_WANDB

    bash "$SCRIPT_DIR/04_train.sbatch"

    echo ""
fi

# =============================================================================
# Stage 5: Evaluation (eval.py)
# =============================================================================
EVAL_OUTPUT_DIR="$EVAL_DIR"
EVAL_OUTPUT_NAME="eval_$(date +%Y%m%d_%H%M%S).json"

if [[ "$SKIP_EVAL" == "true" ]]; then
    echo "========================================"
    echo "SKIPPING Stage 5: Evaluation"
    echo "========================================"
    echo ""
else
    echo "========================================"
    echo "Stage 5: Evaluation"
    echo "========================================"

    mkdir -p "$EVAL_OUTPUT_DIR"

    # Use DS traces for evaluation (contains attack samples)
    EVAL_DATA="$TRACES_DIR/fujitsu_b4_ds.jsonl"

    # Find model directory
    if [[ -d "$MODEL_OUTPUT_DIR/final" ]]; then
        CB_MODEL_DIR="$MODEL_OUTPUT_DIR/final"
    else
        CB_MODEL_DIR="$MODEL_OUTPUT_DIR"
    fi

    export CB_MODEL_DIR
    export BASELINE_MODEL
    export EVAL_DATA
    export TOOL_SCHEMA
    export OUTPUT_DIR="$EVAL_OUTPUT_DIR"
    export OUTPUT_NAME="$EVAL_OUTPUT_NAME"
    export NUM_WORKERS
    export GPU_IDS
    export LIMIT="${EVAL_LIMIT:-}"

    bash "$SCRIPT_DIR/05_eval.sbatch"

    echo ""
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "========================================"
echo "FULL PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Trace files (B2 complete):"
ls -la "$TRACES_DIR"/*_ds.jsonl "$TRACES_DIR"/*_dr.jsonl "$TRACES_DIR"/*_complete.jsonl 2>/dev/null || echo "  (see $TRACES_DIR)"
echo ""
echo "Render files (Tier C):"
ls -la "$RENDERS_DIR"/*.jsonl 2>/dev/null || echo "  (see $RENDERS_DIR)"
echo ""
echo "Lossmask files (training-ready):"
ls -la "$LOSSMASKS_DIR"/*.jsonl 2>/dev/null || echo "  (see $LOSSMASKS_DIR)"
echo ""
echo "Trained model:"
if [[ -d "$MODEL_OUTPUT_DIR/final" ]]; then
    ls -la "$MODEL_OUTPUT_DIR/final" 2>/dev/null || echo "  (see $MODEL_OUTPUT_DIR)"
else
    echo "  (skipped or not found)"
fi
echo ""
echo "Evaluation results:"
if [[ -f "$EVAL_OUTPUT_DIR/$EVAL_OUTPUT_NAME" ]]; then
    ls -la "$EVAL_OUTPUT_DIR/$EVAL_OUTPUT_NAME"
    # Print quick summary
    python3 -c "
import json
with open('$EVAL_OUTPUT_DIR/$EVAL_OUTPUT_NAME', 'r') as f:
    results = json.load(f)
if 'stage1_passed' in results:
    passed = results['stage1_passed']
    print('  Stage 1 Gates: {} PASSED'.format('✅' if passed else '❌ NOT'))
" 2>/dev/null || true
else
    echo "  (skipped or not found)"
fi
echo ""
echo "Finished at: $(date)"
