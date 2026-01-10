#!/bin/bash
#SBATCH --job-name=mvp_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Circuit Breaker Training
# Trillium 1×H100
# =============================================================================
#
# Trains Circuit Breaker adapter using:
# - Ds (harmful set): Successful tool-flip attacks
# - Dr (retain set): Paired benign twins
# - Dual coefficient schedule (cs, cr)
# - Llama 3.1 8B Instruct base model
#
# DEPENDENCIES: Run these first:
#   1. trillium_mvp_generate_ds.sbatch
#   2. trillium_mvp_generate_dr.sbatch
#   3. trillium_mvp_validate.sbatch
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_train.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p "$SCRATCH_DIR/logs"

# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch,wandb}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export WANDB_DIR="$CACHE_ROOT/wandb"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Circuit Breaker Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
echo "========================================"

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$SCRATCH_DIR/cb_mvp_data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"
DR_FILE="$DATA_DIR/dr_stage1.jsonl"
RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_mvp_adapter"

mkdir -p "$RUN_DIR"/{logs,outputs}

# =============================================================================
# Check Dependencies
# =============================================================================
if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found at $DS_FILE"
    echo "Please run trillium_mvp_generate_ds.sbatch first!"
    exit 1
fi

if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr file not found at $DR_FILE"
    echo "Please run trillium_mvp_generate_dr.sbatch first!"
    exit 1
fi

echo "Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"
echo "Dr file: $DR_FILE ($(wc -l < "$DR_FILE") samples)"

# =============================================================================
# Pre-Training Sanity Check
# =============================================================================
echo ""
echo "Running pre-training sanity check..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Loading base model to verify GPU access...')
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL')
model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto')
print('✅ Base model loaded successfully')
del model
torch.cuda.empty_cache()
"

# =============================================================================
# Run Training
# =============================================================================
echo ""
echo "Starting Circuit Breaker training..."

python scripts/train_circuit_breaker.py \
    --model-name "$BASE_MODEL" \
    --ds-path "$DS_FILE" \
    --dr-path "$DR_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --loss-weighting dual \
    --alpha-max 1.0 \
    --alpha-decay-multiplier 1.0 \
    --total-steps 300 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --warmup-steps 50 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 16 17 18 19 20 21 22 23 \
    --save-steps 100 \
    --eval-steps 50 \
    --logging-steps 10 \
    --bf16

# =============================================================================
# Post-Training Sanity Check
# =============================================================================
echo ""
echo "Running post-training sanity check..."

python scripts/circuit_breakers/sanity_check.py \
    --base-model "$BASE_MODEL" \
    --adapter-model "$OUTPUT_DIR" \
    --epsilon 1e-4

# =============================================================================
# Create symlink to latest run
# =============================================================================
ln -sfn "$RUN_DIR" "$SCRATCH_DIR/cb_runs/latest"

echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Adapter saved to: $OUTPUT_DIR"
echo "Symlink: $SCRATCH_DIR/cb_runs/latest"
echo ""
echo "Next steps:"
echo "  1. sbatch slurm/Trillium/trillium_mvp_sanity_check.sbatch"
echo "  2. sbatch slurm/Trillium/trillium_mvp_eval.sbatch"
echo "========================================"
