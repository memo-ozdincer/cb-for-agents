#!/bin/bash
#SBATCH --job-name=mvp_train_mid3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Circuit Breaker Training
# Trillium 1×H100
# =============================================================================
#
# DEPENDENCIES: Run 01-04 first!
#
# Submit:
#   sbatch slurm/05_train.sbatch
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

# Cache Setup
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,xdg_cache,xdg_config}
export HOME="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$CB_SCRATCH/data"
TRAINING_FILE="$DATA_DIR/cb_training_batches.jsonl"
OUTPUT_DIR="$CB_SCRATCH/models/cb_${SLURM_JOB_ID}"

if [[ ! -f "$TRAINING_FILE" ]]; then
    echo "ERROR: Training file not found at $TRAINING_FILE"
    echo "Please run 04_rebuild_data.sbatch first!"
    exit 1
fi

echo "Training file: $TRAINING_FILE ($(wc -l < "$TRAINING_FILE") batches)"

# Resolve Model Path
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")
BASE_MODEL="$MODEL_PATH"

# =============================================================================
# STAGE 6: TRAINING - Conservative Hyperparameters
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 6: Training (450 steps, mid3 config)"
echo "========================================"

# Run training with conservative hyperparameters
accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$BASE_MODEL" \
    --data-path "$TRAINING_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 7.5 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 450 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 4.5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.5 \
    --kl-temperature 1.0 \
    --wandb-project "circuit-breakers" \
    --wandb-run-name "cb_mid3_${SLURM_JOB_ID}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Training failed - no output directory"
    exit 1
fi

echo "✓ Training completed"

# =============================================================================
# STAGE 7: Sanity Check
# =============================================================================
echo ""
echo "========================================"
echo "STAGE 7: Sanity Check"
echo "========================================"

python src/evaluation/sanity_check.py \
    --base-model "$MODEL_PATH" \
    --adapter-path "$OUTPUT_DIR/final" \
    --epsilon 1e-4 \
    --fail-on-error

echo "✓ Sanity check passed"
