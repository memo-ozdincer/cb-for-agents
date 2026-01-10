#!/bin/bash
#SBATCH --job-name=mvp_ds_1h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Ds (Harmful Set) with Behavioral Filtering
# Trillium 1×H100 (single GPU)
# =============================================================================
#
# Generates Circuit Breaker Ds set using behavioral filtering:
# - Only includes samples where the attack SUCCEEDS (model calls wrong tool)
# - Uses vLLM backend with batched inference (single GPU)
# - Outputs in Llama 3.1 tool-calling format
#
# Hardware: 1x NVIDIA H100 SXM (80GB)
# Estimated time: ~1-2 hours for full dataset (13k samples with batching)
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_generate_ds_1h100.sbatch
#
# =============================================================================

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
# (flashinfer has hardcoded ~/.cache paths that ignore env vars in subprocesses)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# DO NOT SET TRANSFORMERS_CACHE - deprecated and causes cache fragmentation

# =============================================================================
# Environment Info
# =============================================================================
echo "Python: $(python -V)"
echo "Which:  $(which python)"
echo ""
echo "Cache Configuration:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo ""

# Quick cache check
echo "Hub cache contents:"
ls -1d "$HF_HUB_CACHE"/models--* 2>/dev/null | head -5 || echo "  (no models cached yet - run prefetch_models.sh first!)"
echo ""

# =============================================================================
# Preflight GPU Check
# =============================================================================
python - << 'PY'
import sys
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: No GPU available!")
    sys.exit(1)
PY

if [[ $? -ne 0 ]]; then
    echo "ERROR: GPU check failed"
    exit 1
fi
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Generate Ds (Behavioral Filtering)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
echo "Backend: vLLM (batched inference)"
echo "Model: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
echo "========================================"

# =============================================================================
# Output Directories
# =============================================================================
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_data"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Run Ds Generation
# =============================================================================
# IMPORTANT: Use abliterated model so attacks can succeed
# Standard Llama-3.1-Instruct will refuse harmful completions!
#
# Using vLLM backend with batched inference for 10-20x speedup
# For faster processing, use --limit to reduce dataset size

python scripts/cb_data_generation/generate_ds_mvp.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --backend vllm \
    --tensor-parallel 1 \
    --batch-size 32 \
    --dtype bfloat16 \
    --temperature 0.7 \
    --min-yield 0.10 \
    --fail-on-low-yield \
    --output "$OUTPUT_DIR/ds_stage1.jsonl" \
    --output-ids "$OUTPUT_DIR/ds_stage1_ids.txt"

echo "========================================"
echo "Ds generation complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR/ds_stage1.jsonl"
echo "Sample count: $(wc -l < "$OUTPUT_DIR/ds_stage1.jsonl" 2>/dev/null || echo 0)"
echo ""

# IDs file is created automatically by --output-ids

echo "========================================"
