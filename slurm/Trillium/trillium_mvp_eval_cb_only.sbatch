#!/bin/bash
#SBATCH --job-name=mvp_eval_vllm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# CB Model Evaluation using vLLM (FAST batched inference with 4xH100)
# 
# Uses tensor parallelism across 4 GPUs for fast batched inference.
# Much faster than sequential HuggingFace inference.
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL: Set HOME to SCRATCH *BEFORE* anything else
# This is required to work around a vLLM bug. DO NOT MOVE THIS.
# =============================================================================
export HOME="/scratch/memoozd"

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache setup (HOME already set above)
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "========================================"
echo "CB Eval with vLLM (4xH100 Tensor Parallel)"
echo "========================================"
echo "HOME=$HOME"
echo "HF_HOME=$HF_HOME"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/ 2>/dev/null | head -1)
CB_ADAPTER="${LATEST_RUN}outputs/cb_mvp_adapter/final"

if [[ ! -d "$CB_ADAPTER" ]]; then
    CHECKPOINT=$(ls -td ${LATEST_RUN}outputs/cb_mvp_adapter/checkpoint-* 2>/dev/null | head -1)
    CB_ADAPTER="$CHECKPOINT"
fi

DATA_DIR="$SCRATCH_DIR/cb_mvp_data"
EVAL_DATA="$DATA_DIR/eval_stage1.jsonl"
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_eval"
mkdir -p "$OUTPUT_DIR"

echo "Base model: $BASE_MODEL"
echo "CB adapter: $CB_ADAPTER"
echo "Eval data: $(wc -l < "$EVAL_DATA") samples"
echo "Tensor parallel: 4 GPUs"

# Run vLLM evaluation with tensor parallelism across 4 H100s
python scripts/circuit_breakers/eval_mvp_vllm.py \
    --baseline "$BASE_MODEL" \
    --cb-adapter "$CB_ADAPTER" \
    --eval-data "$EVAL_DATA" \
    --tensor-parallel 4 \
    --batch-size 32 \
    --dtype bfloat16 \
    --output "$OUTPUT_DIR/eval_vllm_${SLURM_JOB_ID}.json"

echo ""
echo "========================================"
echo "Done! Results: $OUTPUT_DIR/eval_vllm_${SLURM_JOB_ID}.json"
echo "========================================"