#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --job-name=cb_llama31_8b_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Training v2 - With Critical Fixes
# =============================================================================
#
# FIXES from v1:
# 1. alpha_decay_multiplier=1.0 - Schedule completes within training steps
# 2. Data regeneration - Replaces synthetic completions with realistic tool calls
# 3. Increased steps - 300 steps for better convergence
# 4. Optional: --no-gradient-checkpointing to test gradient flow
#
# Hardware: 4x NVIDIA L40S (48GB each = 192GB total VRAM)
#
# Submit:
#   cd /project/6105522/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/Killarney/killarney_cb_llama31_8b_4xl40s_v2.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
SCRATCH_DIR="$HOME/scratch"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

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

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# Preflight checks
python - << 'PY'
import sys
def check(mod):
    try: __import__(mod)
    except Exception as e: print(f"ERROR: {mod}: {e}"); sys.exit(1)
for m in ("torch", "transformers", "peft", "accelerate"): check(m)
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY

# Environment
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export WANDB_MODE=online
export WANDB_PROJECT=circuit-breakers
export WANDB_RUN_GROUP=killarney-cb-4xl40s-v2
export WANDB_TAGS=killarney,cb,4xl40s,llama31,v2,fixed-data

export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "========================================"
echo "Circuit Breaker Training v2 - FIXED"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "KEY FIXES:"
echo "  - alpha_decay_multiplier=1.0 (schedule completes)"
echo "  - Data regenerated with realistic tool calls"
echo "  - 300 steps (vs 150)"
echo "  - Gradient checkpointing disabled for testing"
echo "========================================"

# =============================================================================
# STEP 1: Regenerate data with realistic completions
# =============================================================================
echo ""
echo "Step 1: Regenerating harmful completions..."

COMPLETIONS_FILE="data/circuit_breakers/harmful/harmful_pairs.completions.jsonl"
if [[ -f "$COMPLETIONS_FILE" ]]; then
  # Check how many are synthetic
  SYNTHETIC_COUNT=$(python -c "
import json
markers = ['[TOOL_CALL]', '(Expected:', '[TOOL_CALLS:', '[HARMFUL:']
with open('$COMPLETIONS_FILE') as f:
    samples = [json.loads(l) for l in f]
synthetic = sum(1 for s in samples if any(m in s.get('harmful_completion','') for m in markers))
print(synthetic)
")
  echo "  Found $SYNTHETIC_COUNT synthetic completions"

  if [[ "$SYNTHETIC_COUNT" -gt 0 ]]; then
    echo "  Regenerating with realistic tool calls..."
    python scripts/augmentation/generate_real_cb_completions.py \
      --input "$COMPLETIONS_FILE" \
      --mode realistic-tools
    echo "  Done regenerating completions"
  else
    echo "  Data already has realistic completions"
  fi
else
  echo "  WARNING: No completions file found at $COMPLETIONS_FILE"
fi

# =============================================================================
# STEP 2: Recreate batches with regenerated data
# =============================================================================
echo ""
echo "Step 2: Recreating training batches..."

python scripts/format_for_cb/create_cb_batches.py --batch-size 16
echo "  Done creating batches"

# =============================================================================
# STEP 3: Validate data quality
# =============================================================================
echo ""
echo "Step 3: Validating data quality..."

python - << 'PY'
import json
import sys

batches_file = "data/circuit_breakers/cb_training_batches.jsonl"
with open(batches_file) as f:
    batches = [json.loads(l) for l in f]

total_harmful = sum(len(b.get('harmful', [])) for b in batches)
with_completion = sum(
    1 for b in batches
    for s in b.get('harmful', [])
    if s.get('text') or s.get('harmful_completion')
)

# Check for synthetic completions
synthetic_markers = ['[TOOL_CALL]', '(Expected:', '[TOOL_CALLS:', '[HARMFUL:']
synthetic_count = sum(
    1 for b in batches
    for s in b.get('harmful', [])
    if any(m in s.get('harmful_completion', '') for m in synthetic_markers)
)

pct = 100 * with_completion / total_harmful if total_harmful > 0 else 0
synthetic_pct = 100 * synthetic_count / total_harmful if total_harmful > 0 else 0

print(f"  Batches: {len(batches)}")
print(f"  Harmful samples: {total_harmful}")
print(f"  With completions: {with_completion} ({pct:.1f}%)")
print(f"  Synthetic (FAKE): {synthetic_count} ({synthetic_pct:.1f}%)")
print(f"  Real completions: {total_harmful - synthetic_count} ({100-synthetic_pct:.1f}%)")

if synthetic_pct > 50:
    print("")
    print("WARNING: Still >50% synthetic completions!")
    print("CB training may be less effective.")
    print("Consider generating actual model completions.")
else:
    print("  OK: Majority have realistic completions")
PY

# =============================================================================
# STEP 4: Training with Fixed Parameters
# =============================================================================
echo ""
echo "Step 4: Starting training with fixed parameters..."
echo "========================================"

# KEY FIXES:
# --alpha-decay-multiplier 1.0  : Schedule completes at end of training
# --total-steps 300             : More steps for convergence
# --gradient-checkpointing      : Reduce memory by recomputing activations
#
# With 4x L40S (48GB each), we have ~192GB total VRAM
# DDP splits model across processes, each gets ~45GB headroom

accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-instruct \
  --loss-weighting dual \
  --total-steps 300 \
  --alpha-decay-multiplier 1.0 \
  --batch-size 1 \
  --max-seq-length 512 \
  --gradient-accumulation-steps 16 \
  --output-dir "$RUN_DIR/outputs/cb_llama31_8b_instruct_v2"

echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"

# =============================================================================
# STEP 5: Quick Evaluation (if time permits)
# =============================================================================
echo ""
echo "Step 5: Quick evaluation..."

# Check if final checkpoint exists
FINAL_CKPT="$RUN_DIR/outputs/cb_llama31_8b_instruct_v2/final"
if [[ -d "$FINAL_CKPT" ]]; then
  echo "  Final checkpoint saved at: $FINAL_CKPT"
  echo "  Run evaluation with:"
  echo "    python scripts/circuit_breakers/eval.py --model-path $FINAL_CKPT"
else
  echo "  WARNING: Final checkpoint not found"
fi

echo ""
echo "Done!"
