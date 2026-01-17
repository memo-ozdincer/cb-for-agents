#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=inspect_batches
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=00:16:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# ============================================================================
# DIAGNOSTIC: Inspect training batches to see actual data
# ============================================================================
# This prints actual samples to diagnose why:
# - Only 60% of harmful samples have <|python_tag|>
# - Mask coverage is only 1.4%
# - Gradients are zero
# ============================================================================

set -e

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load python/3.11.5
module list

echo "=== Venv activation ==="
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

# Offline mode for tokenizer
export HF_HOME="$SCRATCH_DIR/cb_cache/hf"
export HF_HUB_CACHE="$SCRATCH_DIR/cb_cache/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo ""
echo "========================================"
echo "INSPECTING TRAINING BATCHES"
echo "========================================"

# Find the batches file
STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"
BATCHES_FILE="$STAGE2_DATA_DIR/stage2/train_batches.jsonl"
QUICK_BATCHES_FILE="$STAGE2_DATA_DIR/stage2/train_batches_quick.jsonl"

if [[ -f "$QUICK_BATCHES_FILE" ]]; then
    echo "Inspecting QUICK batches: $QUICK_BATCHES_FILE"
    python scripts/circuit_breakers/inspect_training_batches.py \
        --batches "$QUICK_BATCHES_FILE" \
        --n 5 \
        --tokenize
elif [[ -f "$BATCHES_FILE" ]]; then
    echo "Inspecting FULL batches: $BATCHES_FILE"
    python scripts/circuit_breakers/inspect_training_batches.py \
        --batches "$BATCHES_FILE" \
        --n 5 \
        --tokenize
else
    echo "ERROR: No batches file found!"
    echo "  Tried: $QUICK_BATCHES_FILE"
    echo "  Tried: $BATCHES_FILE"
    exit 1
fi

echo ""
echo "========================================"
echo "Also inspecting the MERGED data (before batching)"
echo "========================================"

MERGED_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"
if [[ -f "$MERGED_FILE" ]]; then
    echo "First 3 harmful samples from merged data:"
    echo ""
    
    # Use Python to properly inspect the JSONL
    python3 << 'EOF'
import json
from pathlib import Path
import os

merged_file = Path(os.environ.get("SCRATCH_DIR", "/scratch/memoozd")) / "cb_stage2_data/stage2/train.jsonl"

harmful_count = 0
harmful_with_tag = 0
samples_shown = 0

with open(merged_file) as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        labels = sample.get("labels", {})
        
        if labels.get("split") == "harmful":
            harmful_count += 1
            assistant_raw = sample.get("assistant_raw", "")
            has_tag = "<|python_tag|>" in assistant_raw
            if has_tag:
                harmful_with_tag += 1
            
            if samples_shown < 3:
                samples_shown += 1
                print(f"--- Sample {samples_shown} ---")
                print(f"  ID: {sample.get('id', '?')}")
                print(f"  Source: {sample.get('metadata', {}).get('source', '?')}")
                print(f"  Has <|python_tag|>: {'YES' if has_tag else 'NO'}")
                print(f"  assistant_raw ({len(assistant_raw)} chars):")
                print(f"    >>> {assistant_raw[:400]}{'...' if len(assistant_raw) > 400 else ''}")
                print()

print(f"\nSUMMARY: {harmful_with_tag}/{harmful_count} harmful samples have <|python_tag|> ({100*harmful_with_tag/max(1,harmful_count):.1f}%)")
EOF
else
    echo "Merged file not found: $MERGED_FILE"
fi

echo ""
echo "========================================"
echo "INSPECTION COMPLETE"
echo "========================================"
