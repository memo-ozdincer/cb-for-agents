#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cb_mvp_datagen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# CB MVP Data Generation Pipeline - Trillium 1×H100
# =============================================================================
#
# Generates Circuit Breaker MVP dataset:
# 1. Generate Ds (harmful set) with behavioral filtering
# 2. Generate Dr (paired benign twins)
# 3. Validate format
# 4. Create evaluation set
#
# Hardware: 1x NVIDIA H100 SXM (80GB VRAM)
#           24 CPU cores, 188 GiB RAM
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   mkdir -p /scratch/memoozd/logs
#   sbatch slurm/Trillium/trillium_cb_mvp_datagen.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p /scratch/memoozd/logs

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

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
for m in ("torch", "transformers"): check(m)
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Environment
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# IMPORTANT: All outputs must go to SCRATCH (Trillium requirement)
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_data/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo "========================================"
echo "CB MVP Data Generation - Trillium"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "GPU: 1x H100 SXM (80GB)"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# =============================================================================
# STEP 1: Generate Ds with behavioral filtering
# =============================================================================
echo ""
echo "Step 1: Generating Ds (harmful set) with behavioral filtering..."
echo "========================================"

python scripts/cb_data_generation/generate_ds_mvp.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --min-yield 0.10 \
    --output-dir "$OUTPUT_DIR"

echo "Ds generation complete!"

# Check if output exists
DS_FILE="$OUTPUT_DIR/ds_stage1.jsonl"
if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found at $DS_FILE"
    exit 1
fi

echo "  Ds samples: $(wc -l < $DS_FILE)"

# =============================================================================
# STEP 2: Generate Dr (paired benign twins)
# =============================================================================
echo ""
echo "Step 2: Generating Dr (paired benign twins)..."
echo "========================================"

python scripts/cb_data_generation/generate_dr_mvp.py \
    --ds-data "$DS_FILE" \
    --output-dir "$OUTPUT_DIR"

echo "Dr generation complete!"

# Check if output exists
DR_FILE="$OUTPUT_DIR/dr_stage1.jsonl"
if [[ ! -f "$DR_FILE" ]]; then
    echo "ERROR: Dr file not found at $DR_FILE"
    exit 1
fi

echo "  Dr samples: $(wc -l < $DR_FILE)"

# =============================================================================
# STEP 3: Validate format (Ds)
# =============================================================================
echo ""
echo "Step 3: Validating Ds format..."
echo "========================================"

python scripts/cb_data_generation/validate_format.py \
    --data "$DS_FILE" \
    --set-type ds \
    --strict

echo "Ds validation complete!"

# =============================================================================
# STEP 4: Validate format (Dr)
# =============================================================================
echo ""
echo "Step 4: Validating Dr format..."
echo "========================================"

python scripts/cb_data_generation/validate_format.py \
    --data "$DR_FILE" \
    --set-type dr \
    --strict

echo "Dr validation complete!"

# =============================================================================
# STEP 5: Create evaluation set
# =============================================================================
echo ""
echo "Step 5: Creating evaluation set..."
echo "========================================"

TRAIN_IDS="$OUTPUT_DIR/ds_stage1.ids.txt"
if [[ ! -f "$TRAIN_IDS" ]]; then
    echo "WARNING: Train IDs file not found at $TRAIN_IDS"
    echo "Attempting to extract IDs from Ds file..."
    python -c "
import json
with open('$DS_FILE') as f:
    ids = [json.loads(line)['id'] for line in f]
with open('$TRAIN_IDS', 'w') as f:
    f.write('\n'.join(ids))
print(f'Extracted {len(ids)} IDs')
"
fi

python scripts/cb_data_generation/create_eval_set.py \
    --train-ids "$TRAIN_IDS" \
    --output-dir "$OUTPUT_DIR"

echo "Evaluation set created!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Data Generation Complete!"
echo "========================================"
echo "Date: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
if [[ -f "$DS_FILE" ]]; then
    echo "  ✓ Ds: $DS_FILE ($(wc -l < $DS_FILE) samples)"
else
    echo "  ✗ Ds: MISSING"
fi
if [[ -f "$DR_FILE" ]]; then
    echo "  ✓ Dr: $DR_FILE ($(wc -l < $DR_FILE) samples)"
else
    echo "  ✗ Dr: MISSING"
fi

EVAL_FILE="$OUTPUT_DIR/eval_set.jsonl"
if [[ -f "$EVAL_FILE" ]]; then
    echo "  ✓ Eval: $EVAL_FILE ($(wc -l < $EVAL_FILE) samples)"
else
    echo "  ✗ Eval: MISSING (may be expected)"
fi

echo ""
echo "Next: Run training with:"
echo "  export CB_MVP_DATA=$OUTPUT_DIR"
echo "  sbatch slurm/Trillium/trillium_cb_mvp_train_eval.sbatch"
echo "========================================"
