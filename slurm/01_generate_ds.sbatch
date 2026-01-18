#!/bin/bash
#SBATCH --job-name=mvp_ds_gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Ds (Harmful Set) with Behavioral Filtering
# Trillium 4×H100 (tensor parallel)
# =============================================================================
#
# Generates Circuit Breaker Ds set using behavioral filtering:
# - Only includes samples where the attack SUCCEEDS (model calls wrong tool)
# - Uses vLLM with tensor_parallel=4 for high-throughput batched inference
# - Outputs in Llama 3.1 tool-calling format
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/01_generate_ds.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
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
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,xdg_cache,xdg_config,flashinfer}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_DIR/flashinfer"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Model: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"

# =============================================================================
# Run Ds Generation
# =============================================================================
DATA_DIR="$CB_SCRATCH/data"
mkdir -p "$DATA_DIR"

python src/data_generation/generate_ds_mvp.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct \
    --backend vllm \
    --tensor-parallel 4 \
    --batch-size 64 \
    --dtype bfloat16 \
    --temperature 0.7 \
    --min-yield 0.10 \
    --fail-on-low-yield \
    --output "$DATA_DIR/ds_stage1.jsonl" \
    --output-ids "$DATA_DIR/ds_stage1_ids.txt"

echo "Ds generation complete! Output: $DATA_DIR/ds_stage1.jsonl"
