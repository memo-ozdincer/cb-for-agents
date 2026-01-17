#!/bin/bash
#SBATCH --job-name=stage2_adv_safe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 2: Generate Adversarial-Safe Dr Samples
# Trillium 4×H100 (tensor parallel)
# =============================================================================
#
# Generates Dr samples where the model RESISTS injection attacks:
# - Input: Fujitsu B4 adversarial prompts
# - Output: Retain samples with correct tool calls
# - Uses low temperature for consistent resistance
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
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

export SCRATCH_DIR="$SCRATCH_DIR"
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
echo "Stage 2: Generate Adversarial-Safe Dr"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 4 x H100 SXM 80GB (tensor_parallel=4)"
echo "Backend: vLLM"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "========================================"

# =============================================================================
# Inputs / Outputs
# =============================================================================
B4_FILE="$REPO_DIR/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"
OUTPUT_FILE="$REPO_DIR/data/circuit_breakers/retain/adversarial_safe.jsonl"

if [[ ! -f "$B4_FILE" ]]; then
    echo "ERROR: B4 data not found at $B4_FILE"
    exit 1
fi

# =============================================================================
# Run Generation
# =============================================================================
python scripts/cb_data_generation/generate_adversarial_safe.py \
    --b4-data "$B4_FILE" \
    --tool-schema "$TOOL_SCHEMA" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output "$OUTPUT_FILE" \
    --target-n 10 \
    --temperature 0.1 \
    --batch-size 8 \
    --tensor-parallel-size 1 \
    --limit 50 \
    --dry-run \

echo "========================================"
echo "Adversarial-safe generation complete!"
echo "========================================"
echo "Output: $OUTPUT_FILE"
echo "Sample count: $(wc -l < "$OUTPUT_FILE")"
echo "========================================"
