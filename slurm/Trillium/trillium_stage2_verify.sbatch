#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_verify
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: VERIFICATION ONLY (NO TRAINING)
# =============================================================================
#
# Runs:
# 1) Adapter sanity check
# 2) Sample outputs on Stage 2 eval set
# 3) Quick batch peek from train_batches.jsonl
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_verify.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv activation ==="
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

echo "=== Environment setup ==="
export OMP_NUM_THREADS=8

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/datasets,torch,xdg_cache,xdg_config,flashinfer}
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

RUN_DIR="$SCRATCH_DIR/stage2_verify/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# Resolve adapter path (prefer latest symlink)
ADAPTER_PATH="$SCRATCH_DIR/cb_runs/latest_stage2/outputs/cb_stage2_adapter/final"
if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "WARNING: latest_stage2 not found. Set ADAPTER_PATH manually."
  ADAPTER_PATH="/scratch/memoozd/cb_runs/208172/outputs/cb_stage2_adapter/final"
fi

# Resolve eval data
EVAL_FILE="$SCRATCH_DIR/cb_stage2_data/stage2/eval.jsonl"
if [[ ! -f "$EVAL_FILE" ]]; then
  EVAL_FILE="$REPO_DIR/data/circuit_breakers/stage2/eval.jsonl"
fi

# Resolve train batches (for inspection)
TRAIN_BATCHES="$SCRATCH_DIR/cb_stage2_data/stage2/train_batches.jsonl"
if [[ ! -f "$TRAIN_BATCHES" ]]; then
  TRAIN_BATCHES="$SCRATCH_DIR/cb_stage2_data/stage2/train_batches.jsonl"
fi

echo ""
echo "========================================"
echo "Stage 2 Verification"
echo "========================================"
echo "Adapter: $ADAPTER_PATH"
echo "Eval:    $EVAL_FILE"
echo "Batches: $TRAIN_BATCHES"
echo "========================================"
echo ""

echo "=== Adapter sanity check ==="
python scripts/circuit_breakers/sanity_check.py \
    --base-model "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter-path "$ADAPTER_PATH" \
    --epsilon 1e-4 \
    --fail-on-error \
    2>&1 | tee "$RUN_DIR/sanity_check.txt"

echo ""
echo "=== Sample outputs (Stage 2 eval) ==="
python scripts/circuit_breakers/sample_outputs.py \
    --eval-data "$EVAL_FILE" \
    --cb-adapter "$ADAPTER_PATH" \
    --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
    --n 10 \
    2>&1 | tee "$RUN_DIR/sample_outputs.txt"

echo ""
echo "=== Batch peek ==="
python - << 'PY' | tee "$RUN_DIR/batch_peek.txt"
import json
path = "/scratch/memoozd/cb_stage2_data/stage2/train_batches.jsonl"
with open(path) as f:
    for i, line in zip(range(2), f):
        b = json.loads(line)
        harmful = b["harmful"][0]
        print(f"BATCH {i}")
        print("HARMFUL USER:", harmful["messages"][-2]["content"][:200].replace("\\n"," ") + "...")
        print("HARMFUL ASSISTANT:", harmful["messages"][-1]["content"][:200].replace("\\n"," ") + "...")
        print("BENIGN COUNT:", len(b["benign"]))
        print("-"*60)
PY

echo ""
echo "========================================"
echo "Verification complete!"
echo "Results in: $RUN_DIR"
echo "========================================"
