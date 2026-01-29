#!/bin/bash
#SBATCH --job-name=cb_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 4: Circuit Breaker Training (train_schema.py)
# Trains the model using schema v1 data format (render_v1 + lossmask_v1)
# Multi-GPU training supported via PyTorch DDP
# =============================================================================
#
# This script:
# 1. Loads renders and lossmasks from ETL_B output
# 2. Trains using Circuit Breaker / Representation Rerouting method
# 3. Outputs LoRA adapter checkpoints and final model
#
# Data Loading Modes:
#   ds_dr   - DS traces as harmful, DR traces as benign (default)
#   labeled - Split by labels.is_harmful from traces
#   mixed   - Explicit harmful/benign render/lossmask paths
#
# Configuration via environment variables:
#   DATA_MODE       - Data loading mode: ds_dr, labeled, mixed (default: ds_dr)
#   DS_RENDERS      - Path to DS renders JSONL (ds_dr mode)
#   DS_LOSSMASKS    - Path to DS lossmasks JSONL (ds_dr mode)
#   DR_RENDERS      - Path to DR renders JSONL (ds_dr mode)
#   DR_LOSSMASKS    - Path to DR lossmasks JSONL (ds_dr mode)
#   OUTPUT_DIR      - Output directory for checkpoints (default: $CB_SCRATCH/models/cb_train)
#   MODEL_PRESET    - Config preset: llama-3.1-8b-instruct, llama-3-8b, etc.
#   MODEL           - Override base model name
#   TOTAL_STEPS     - Total training steps (default: 300)
#   BATCH_SIZE      - Per-GPU batch size (default: 8)
#   LEARNING_RATE   - Learning rate (default: 5e-5)
#   WARMUP_STEPS    - Warmup steps (default: 20)
#   ALPHA_MAX       - Initial alpha for RR loss (default: 10.0)
#   MAX_SEQ_LENGTH  - Max sequence length (default: 2048)
#   CB_LAYERS       - CB target layers, comma-separated (default: 10,20)
#   LOSS_WEIGHTING  - Loss weighting: single_alpha or dual (default: dual)
#   LORA_R          - LoRA rank (default: 16)
#   LORA_ALPHA      - LoRA alpha (default: 32)
#   WANDB_PROJECT   - W&B project name (default: circuit-breakers)
#   WANDB_RUN_NAME  - W&B run name (default: auto-generated)
#   NO_WANDB        - Set to "true" to disable W&B (default: false)
#   RESUME_FROM     - Path to checkpoint to resume from (optional)
#
# Submit:
#   sbatch slurm/pipeline/04_train.sbatch
#
# Or with custom settings:
#   TOTAL_STEPS=500 ALPHA_MAX=8.0 sbatch slurm/pipeline/04_train.sbatch
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
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "Stage 4: Circuit Breaker Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

# =============================================================================
# Configuration
# =============================================================================
DATA_MODE="${DATA_MODE:-ds_dr}"
DS_RENDERS="${DS_RENDERS:-$CB_SCRATCH/data/renders/fujitsu_b4_ds.jsonl}"
DS_LOSSMASKS="${DS_LOSSMASKS:-$CB_SCRATCH/data/lossmasks/fujitsu_b4_ds.jsonl}"
DR_RENDERS="${DR_RENDERS:-$CB_SCRATCH/data/renders/fujitsu_b4_dr.jsonl}"
DR_LOSSMASKS="${DR_LOSSMASKS:-$CB_SCRATCH/data/lossmasks/fujitsu_b4_dr.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$CB_SCRATCH/models/cb_train_$(date +%Y%m%d_%H%M%S)}"

# Model configuration
MODEL_PRESET="${MODEL_PRESET:-llama-3.1-8b-instruct}"
MODEL="${MODEL:-}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
ALPHA_MAX="${ALPHA_MAX:-10.0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"

# CB-specific
CB_LAYERS="${CB_LAYERS:-10,20}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"

# LoRA
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-circuit-breakers}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
NO_WANDB="${NO_WANDB:-false}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-50}"

# Resume
RESUME_FROM="${RESUME_FROM:-}"

# Mixed mode (optional additional data)
HARMFUL_RENDERS="${HARMFUL_RENDERS:-}"
HARMFUL_LOSSMASKS="${HARMFUL_LOSSMASKS:-}"
BENIGN_RENDERS="${BENIGN_RENDERS:-}"
BENIGN_LOSSMASKS="${BENIGN_LOSSMASKS:-}"

# Labeled mode
RENDERS="${RENDERS:-}"
LOSSMASKS="${LOSSMASKS:-}"
TRACES="${TRACES:-}"

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  DATA_MODE: $DATA_MODE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODEL_PRESET: $MODEL_PRESET"
echo "  MODEL: ${MODEL:-<from preset>}"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  WARMUP_STEPS: $WARMUP_STEPS"
echo "  ALPHA_MAX: $ALPHA_MAX"
echo "  MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH"
echo "  CB_LAYERS: $CB_LAYERS"
echo "  LOSS_WEIGHTING: $LOSS_WEIGHTING"
echo "  LORA_R: $LORA_R"
echo "  LORA_ALPHA: $LORA_ALPHA"
echo "  NO_WANDB: $NO_WANDB"
echo ""

# =============================================================================
# Build command arguments
# =============================================================================
TRAIN_ARGS=(
    --preset "$MODEL_PRESET"
    --output-dir "$OUTPUT_DIR"
    --total-steps "$TOTAL_STEPS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --alpha-max "$ALPHA_MAX"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION"
    --loss-weighting "$LOSS_WEIGHTING"
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --wandb-project "$WANDB_PROJECT"
)

# Parse CB_LAYERS (comma-separated to space-separated)
IFS=',' read -ra CB_LAYER_ARRAY <<< "$CB_LAYERS"
TRAIN_ARGS+=(--cb-target-layers "${CB_LAYER_ARRAY[@]}")

if [[ -n "$MODEL" ]]; then
    TRAIN_ARGS+=(--model "$MODEL")
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
    TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

if [[ "$NO_WANDB" == "true" ]]; then
    TRAIN_ARGS+=(--no-wandb)
fi

if [[ -n "$RESUME_FROM" && -d "$RESUME_FROM" ]]; then
    TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi

# Add data arguments based on mode
case "$DATA_MODE" in
    ds_dr)
        echo "Data mode: DS/DR (DS=harmful, DR=benign)"
        echo "  DS Renders: $DS_RENDERS"
        echo "  DS Lossmasks: $DS_LOSSMASKS"
        echo "  DR Renders: $DR_RENDERS"
        echo "  DR Lossmasks: $DR_LOSSMASKS"

        # Validate files exist
        for f in "$DS_RENDERS" "$DS_LOSSMASKS" "$DR_RENDERS" "$DR_LOSSMASKS"; do
            if [[ ! -f "$f" ]]; then
                echo "ERROR: Required file not found: $f"
                echo "Please run stages 1-3 first or set correct paths."
                exit 1
            fi
        done

        TRAIN_ARGS+=(
            --mode ds_dr
            --ds-renders "$DS_RENDERS"
            --ds-lossmasks "$DS_LOSSMASKS"
            --dr-renders "$DR_RENDERS"
            --dr-lossmasks "$DR_LOSSMASKS"
        )
        ;;

    labeled)
        echo "Data mode: Labeled (split by labels.is_harmful)"
        echo "  Renders: $RENDERS"
        echo "  Lossmasks: $LOSSMASKS"
        echo "  Traces: $TRACES"

        if [[ ! -f "$RENDERS" || ! -f "$LOSSMASKS" ]]; then
            echo "ERROR: Labeled mode requires RENDERS and LOSSMASKS"
            exit 1
        fi

        TRAIN_ARGS+=(
            --mode labeled
            --renders "$RENDERS"
            --lossmasks "$LOSSMASKS"
        )

        if [[ -n "$TRACES" && -f "$TRACES" ]]; then
            TRAIN_ARGS+=(--traces "$TRACES")
        fi
        ;;

    mixed)
        echo "Data mode: Mixed (explicit harmful/benign paths)"

        if [[ -z "$HARMFUL_RENDERS" || -z "$BENIGN_RENDERS" ]]; then
            echo "ERROR: Mixed mode requires HARMFUL_RENDERS and BENIGN_RENDERS"
            exit 1
        fi

        TRAIN_ARGS+=(--mode mixed)

        # Parse comma-separated paths
        IFS=',' read -ra HR <<< "$HARMFUL_RENDERS"
        IFS=',' read -ra HM <<< "$HARMFUL_LOSSMASKS"
        IFS=',' read -ra BR <<< "$BENIGN_RENDERS"
        IFS=',' read -ra BM <<< "$BENIGN_LOSSMASKS"

        TRAIN_ARGS+=(--harmful-renders "${HR[@]}")
        TRAIN_ARGS+=(--harmful-lossmasks "${HM[@]}")
        TRAIN_ARGS+=(--benign-renders "${BR[@]}")
        TRAIN_ARGS+=(--benign-lossmasks "${BM[@]}")
        ;;

    *)
        echo "ERROR: Invalid DATA_MODE '$DATA_MODE'. Must be 'ds_dr', 'labeled', or 'mixed'."
        exit 1
        ;;
esac

echo ""

# =============================================================================
# Run Training
# =============================================================================
echo "========================================"
echo "Starting Training"
echo "========================================"

# Determine number of GPUs for distributed training
NGPUS="${SLURM_GPUS_ON_NODE:-1}"

if [[ "$NGPUS" -gt 1 ]]; then
    echo "Running distributed training on $NGPUS GPUs"
    torchrun --nproc_per_node="$NGPUS" \
        src/training/train_schema.py "${TRAIN_ARGS[@]}"
else
    echo "Running single-GPU training"
    python src/training/train_schema.py "${TRAIN_ARGS[@]}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Stage 4 Complete: Circuit Breaker Training"
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Contents:"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "  (check above for output)"
echo ""

if [[ -d "$OUTPUT_DIR/final" ]]; then
    echo "Final model saved to: $OUTPUT_DIR/final"
fi

echo ""
echo "Next step: Run 05_eval.sbatch to evaluate the trained model"
