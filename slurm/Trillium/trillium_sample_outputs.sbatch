#!/bin/bash
#SBATCH --job-name=sample_outputs
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# ============================================================================
# Sample Outputs - See what baseline vs CB model actually produces
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
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
# Cache Setup (MUST match prefetch_models.sh)
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/datasets,torch}

# CRITICAL: Set HOME to SCRATCH for vLLM/transformers bug workaround
export HOME="$SCRATCH_DIR"

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# Enable offline mode - fail fast if model not cached
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Redirect XDG directories to writable scratch space (fixes vLLM and FlashInfer permission errors)
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$FLASHINFER_WORKSPACE_DIR"

# Disable vLLM usage statistics (avoids writing to ~/.config/vllm)
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Force vLLM to use flash_attn backend instead of flashinfer
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

PROJECT_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
SCRATCH_DIR="/scratch/memoozd"
EVAL_DATA="$SCRATCH_DIR/cb_mvp_data/eval_stage1.jsonl"
CB_ADAPTER="$SCRATCH_DIR/cb_runs/195854/outputs/cb_mvp_adapter/final"
TOOL_SCHEMA="$PROJECT_DIR/configs/tool_schemas/b4_standard_v1.json"

# Activate environment
cd "$PROJECT_DIR"

echo ""
echo "Eval data: $EVAL_DATA"
echo "CB adapter: $CB_ADAPTER"
echo "Tool schema: $TOOL_SCHEMA"
echo ""

# Run sample outputs
python scripts/circuit_breakers/sample_outputs.py \
    --eval-data "$EVAL_DATA" \
    --cb-adapter "$CB_ADAPTER" \
    --tool-schema "$TOOL_SCHEMA" \
    --n 15 \
    2>&1 | tee "$SCRATCH_DIR/sample_outputs_${SLURM_JOB_ID}.txt"

echo ""
echo "=============================================="
echo "Output saved to: $SCRATCH_DIR/sample_outputs_${SLURM_JOB_ID}.txt"
echo "=============================================="
