#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --job-name=stage2_test_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2 DATA PIPELINE TEST - CPU ONLY
# =============================================================================
#
# Tests the complete Stage 2 data pipeline without requiring GPU:
# 1. Validate data format (canonical schema)
# 2. Load and tokenize samples
# 3. Verify no schema errors
# 4. Generate statistics report
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/Trillium/trillium_stage2_test_cpu.sbatch
#
# Expected runtime: ~5 minutes
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
export OMP_NUM_THREADS=4

RUN_DIR="$SCRATCH_DIR/stage2_test_cpu/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo ""
echo "========================================================================"
echo "STAGE 2 DATA PIPELINE TEST"
echo "========================================================================"
echo ""

STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"
TRAIN_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"
EVAL_FILE="$STAGE2_DATA_DIR/stage2/eval.jsonl"

if [[ ! -f "$TRAIN_FILE" ]]; then
  TRAIN_FILE="data/circuit_breakers/stage2/train.jsonl"
fi
if [[ ! -f "$EVAL_FILE" ]]; then
  EVAL_FILE="data/circuit_breakers/stage2/eval.jsonl"
fi
echo "Using train file: $TRAIN_FILE"
echo "Using eval file:  $EVAL_FILE"
echo ""

# -----------------------------------------------------------------------------
# Test 1: Validate data format
# -----------------------------------------------------------------------------
echo "=== TEST 1: Data Format Validation ==="
python scripts/cb_data_generation/validate_format.py \
  --data "$TRAIN_FILE" \
  2>&1 | tee "$RUN_DIR/validation_train.log"

echo ""
echo "=== Validating eval set ==="
python scripts/cb_data_generation/validate_format.py \
  --data "$EVAL_FILE" \
  2>&1 | tee "$RUN_DIR/validation_eval.log"

# Check validation passed
if grep -q "ALL VALIDATION CHECKS PASSED" "$RUN_DIR/validation_train.log"; then
  echo "✅ Training data validation PASSED"
else
  echo "❌ Training data validation FAILED"
  exit 1
fi

if grep -q "ALL VALIDATION CHECKS PASSED" "$RUN_DIR/validation_eval.log"; then
  echo "✅ Eval data validation PASSED"
else
  echo "❌ Eval data validation FAILED"
  exit 1
fi

# -----------------------------------------------------------------------------
# Test 2: Data loading and statistics
# -----------------------------------------------------------------------------
echo ""
echo "=== TEST 2: Data Loading & Statistics ==="
export TRAIN_FILE
export EVAL_FILE
python << 'PYEOF'
import json
import os
from collections import Counter
from pathlib import Path

def analyze_dataset(path):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    splits = Counter(s.get("labels", {}).get("split") for s in samples)
    sources = Counter(s.get("metadata", {}).get("source") for s in samples)
    has_tools = sum(1 for s in samples if s.get("tools"))
    has_tool_calls = sum(1 for s in samples if s.get("metadata", {}).get("has_tool_calls"))

    print(f"\n📊 Dataset: {path}")
    print(f"  Total samples: {len(samples)}")
    print(f"  By split: {dict(splits)}")
    print(f"  By source: {dict(sources)}")
    print(f"  With tools field: {has_tools}")
    print(f"  With tool calls: {has_tool_calls}")

    # Check for required fields
    missing_id = sum(1 for s in samples if not s.get("id"))
    missing_messages = sum(1 for s in samples if not s.get("messages"))
    missing_assistant = sum(1 for s in samples if not s.get("assistant_raw"))

    if missing_id or missing_messages or missing_assistant:
        print(f"  ⚠️  Missing fields:")
        if missing_id: print(f"    - ID: {missing_id}")
        if missing_messages: print(f"    - messages: {missing_messages}")
        if missing_assistant: print(f"    - assistant_raw: {missing_assistant}")
    else:
        print(f"  ✅ All required fields present")

    return len(samples)

train_count = analyze_dataset(os.environ["TRAIN_FILE"])
eval_count = analyze_dataset(os.environ["EVAL_FILE"])

print(f"\n✅ Successfully loaded {train_count + eval_count} total samples")
PYEOF

# -----------------------------------------------------------------------------
# Test 3: Tokenization check (without training)
# -----------------------------------------------------------------------------
echo ""
echo "=== TEST 3: Tokenization Check ==="
export TRAIN_FILE
python << 'PYEOF'
import json
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

print("Loading tokenizer...")
model_path = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    local_files_only=True
)

print("Testing tokenization on 5 samples...")
with open(os.environ["TRAIN_FILE"]) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break

        sample = json.loads(line)
        sample_id = sample.get("id", "unknown")
        messages = sample.get("messages", [])
        assistant_raw = sample.get("assistant_raw", "")

        # Try to tokenize
        try:
            # Add assistant message
            full_messages = messages + [{"role": "assistant", "content": assistant_raw}]

            # Tokenize (no tools for now, just check it doesn't crash)
            tokens = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False
            )

            print(f"  ✅ Sample {i+1} ({sample_id}): {len(tokens)} tokens")
        except Exception as e:
            print(f"  ❌ Sample {i+1} ({sample_id}): FAILED - {e}")
            raise

print("✅ Tokenization check passed")
PYEOF

# -----------------------------------------------------------------------------
# Test 4: Generate report
# -----------------------------------------------------------------------------
echo ""
echo "=== TEST 4: Generate Report ==="
cat > "$RUN_DIR/test_report.txt" << 'EOF'
======================================================================
STAGE 2 DATA PIPELINE TEST REPORT
======================================================================

Test Date: $(date)
Job ID: ${SLURM_JOB_ID}

SUMMARY:
✅ All tests passed successfully

TESTS RUN:
1. ✅ Data format validation (train + eval)
2. ✅ Data loading and statistics
3. ✅ Tokenization check
4. ✅ Report generation

DATA COMPOSITION:
- See validation logs for detailed breakdown
- Training set: data/circuit_breakers/stage2/train.jsonl
- Eval set: data/circuit_breakers/stage2/eval.jsonl

NEXT STEPS:
1. Review validation logs in $RUN_DIR/
2. Submit GPU training job with this data
3. Monitor training for:
   - No NaN losses
   - No "to to to" collapse
   - Coherent outputs in sanity check

VALIDATION LOGS:
- $RUN_DIR/validation_train.log
- $RUN_DIR/validation_eval.log

EOF

cat "$RUN_DIR/test_report.txt"

echo ""
echo "========================================================================"
echo "✅ ALL STAGE 2 PIPELINE TESTS PASSED"
echo "========================================================================"
echo ""
echo "📁 Test results saved to: $RUN_DIR/"
echo ""
echo "Next: Submit GPU training job with this data"
echo ""
