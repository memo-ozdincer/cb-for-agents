#!/bin/bash
#SBATCH --job-name=mvp_dr_gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Dr (Retain Set) as Paired Benign Twins
# Trillium 1×H100
# =============================================================================
#
# Generates Circuit Breaker Dr set:
# - Creates benign twins of Ds samples (removes injection)
# - Only includes samples where model calls CORRECT tool
# - Uses vLLM for fast inference
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
mkdir -p "$SCRATCH_DIR/logs"

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

echo "Python: $(python -V)"
echo "Which:  $(which python)"

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
echo "Stage 1 MVP: Generate Dr (Paired Benign Twins)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
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

python scripts/cb_data_generation/generate_dr_mvp.py \
    --ds-data "$DS_FILE" \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --tensor-parallel 1 \
    --temperature 0.3 \
    --max-tokens 512 \
    --output "$OUTPUT_DIR/dr_stage1.jsonl"

echo "========================================"
echo "Dr generation complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR/dr_stage1.jsonl"
echo "Dr sample count: $(wc -l < "$OUTPUT_DIR/dr_stage1.jsonl")"
echo "========================================"
