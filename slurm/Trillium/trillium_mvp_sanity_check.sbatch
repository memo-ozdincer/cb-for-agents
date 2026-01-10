#!/bin/bash
#SBATCH --job-name=mvp_sanity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Adapter Sanity Check (KL Divergence)
# Trillium 1×H100
# =============================================================================
#
# Verifies that the trained adapter actually changes model behavior:
# - Computes KL divergence between base and adapter outputs
# - Fails if mean KL < epsilon (adapter has no effect)
# - CRITICAL: Run this AFTER training, BEFORE evaluation
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_sanity_check.sbatch
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
mkdir -p "$CACHE_ROOT"/{hf,torch}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Adapter Sanity Check"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
echo "========================================"

# =============================================================================
# Configuration
# =============================================================================
# UPDATE THESE PATHS after training completes:
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH="$SCRATCH_DIR/cb_runs/latest/outputs/cb_mvp_adapter"  # Update with actual path

# Check if adapter exists
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "WARNING: Adapter path not found: $ADAPTER_PATH"
    echo "Running pre-training sanity check (base model only)..."
    ADAPTER_PATH=""
fi

# =============================================================================
# Run Sanity Check
# =============================================================================
if [[ -n "$ADAPTER_PATH" ]]; then
    python scripts/circuit_breakers/sanity_check.py \
        --base-model "$BASE_MODEL" \
        --adapter-model "$ADAPTER_PATH" \
        --epsilon 1e-4 \
        --fail-on-error
else
    # Pre-training: just verify base model loads
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Loading base model...')
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL')
model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto')
print('✅ Base model loaded successfully')
print(f'Model: {model.config.architectures}')
print(f'Parameters: {model.num_parameters():,}')
"
fi

echo "========================================"
echo "Sanity check complete!"
echo "========================================"
