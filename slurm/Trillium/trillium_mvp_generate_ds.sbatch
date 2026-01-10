#!/bin/bash
#SBATCH --job-name=mvp_ds_gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.out
#SBATCH --error=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Ds (Harmful Set) with Behavioral Filtering
# Trillium 1×H100
# =============================================================================
#
# Generates Circuit Breaker Ds set using behavioral filtering:
# - Only includes samples where the attack SUCCEEDS (model calls wrong tool)
# - Uses vLLM for fast inference
# - Outputs in Llama 3.1 tool-calling format
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch
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
# Preflight
# =============================================================================
python - << 'PY'
import sys
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
PY

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_OFFLINE=1  # Fail fast if model not cached (no internet on compute nodes)

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Generate Ds (Behavioral Filtering)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
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

python scripts/cb_data_generation/generate_ds_mvp.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --device auto \
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
echo ""

# IDs file is created automatically by --output-ids

echo "========================================"
