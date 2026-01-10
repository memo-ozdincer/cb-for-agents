#!/bin/bash
#SBATCH --job-name=mvp_dr_gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Dr (Retain Set) as Paired Benign Twins
# Trillium 4×H100 (tensor parallel)
# =============================================================================
#
# Generates Circuit Breaker Dr set:
# - Creates benign twins of Ds samples (removes injection)
# - Only includes samples where model calls CORRECT tool
# - Uses vLLM with tensor_parallel=4 for high-throughput inference
#
# Hardware: 4x NVIDIA H100 SXM (80GB each = 320GB total)
# Estimated time: ~30-60 minutes
#
# DEPENDENCY: Run trillium_mvp_generate_ds.sbatch first!
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_generate_dr.sbatch
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

# =============================================================================
# GPU Check
# =============================================================================
python - << 'PY'
import sys
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ERROR: No GPU available!")
    sys.exit(1)
PY
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Generate Dr (Paired Benign Twins)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 4 x H100 SXM 80GB (tensor_parallel=4)"
echo "Backend: vLLM"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "========================================"

# =============================================================================
# Check Dependencies
# =============================================================================
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_data"
DS_FILE="$OUTPUT_DIR/ds_stage1.jsonl"

if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found at $DS_FILE"
    echo "Please run trillium_mvp_generate_ds.sbatch first!"
    exit 1
fi

echo "Found Ds file: $DS_FILE"
echo "Ds sample count: $(wc -l < "$DS_FILE")"

# =============================================================================
# Run Dr Generation
# =============================================================================
# NOTE: Dr uses the standard (non-abliterated) model for benign twins
# This ensures the retain set represents normal, safe behavior
#
# vLLM with tensor_parallel=4 provides ~10-20x throughput improvement

python scripts/cb_data_generation/generate_dr_mvp.py \
    --ds-data "$DS_FILE" \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --tensor-parallel 4 \
    --batch-size 64 \
    --dtype bfloat16 \
    --temperature 0.3 \
    --output "$OUTPUT_DIR/dr_stage1.jsonl"

echo "========================================"
echo "Dr generation complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR/dr_stage1.jsonl"
echo "Dr sample count: $(wc -l < "$OUTPUT_DIR/dr_stage1.jsonl")"
echo "========================================"
