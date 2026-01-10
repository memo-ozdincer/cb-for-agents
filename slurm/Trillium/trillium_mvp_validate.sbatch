#!/bin/bash
#SBATCH --job-name=mvp_validate
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Validate Data Format
# Trillium CPU-only (no GPU needed)
# =============================================================================
#
# Validates generated Ds and Dr against Llama 3.1 format requirements:
# - <|python_tag|> prefix, <|eom_id|>/<|eot_id|> suffix
# - No markdown wrappers or Action: prefixes
# - Consistent tool_calls_structured with assistant_raw
# - Correct is_flip_success based on set type
#
# DEPENDENCY: Run trillium_mvp_generate_ds.sbatch and trillium_mvp_generate_dr.sbatch first!
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_validate.sbatch
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
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Note: Validation is CPU-only, but we set cache vars for consistency
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
export HF_HOME="$CACHE_ROOT/hf"

echo "Python: $(python -V)"
echo ""

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Validate Data Format"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# =============================================================================
# Validate Files
# =============================================================================
OUTPUT_DIR="$SCRATCH_DIR/cb_mvp_data"
DS_FILE="$OUTPUT_DIR/ds_stage1.jsonl"
DR_FILE="$OUTPUT_DIR/dr_stage1.jsonl"
EVAL_FILE="$OUTPUT_DIR/eval_stage1.jsonl"

VALIDATION_PASSED=true

# Validate Ds
if [[ -f "$DS_FILE" ]]; then
    echo ""
    echo "--- Validating Ds: $DS_FILE ---"
    if python scripts/cb_data_generation/validate_format.py \
        --data "$DS_FILE" \
        --strict; then
        echo "✅ Ds validation PASSED"
    else
        echo "❌ Ds validation FAILED"
        VALIDATION_PASSED=false
    fi
else
    echo "⚠️  Ds file not found: $DS_FILE"
fi

# Validate Dr
if [[ -f "$DR_FILE" ]]; then
    echo ""
    echo "--- Validating Dr: $DR_FILE ---"
    if python scripts/cb_data_generation/validate_format.py \
        --data "$DR_FILE" \
        --strict; then
        echo "✅ Dr validation PASSED"
    else
        echo "❌ Dr validation FAILED"
        VALIDATION_PASSED=false
    fi
else
    echo "⚠️  Dr file not found: $DR_FILE"
fi

# Validate Eval set
if [[ -f "$EVAL_FILE" ]]; then
    echo ""
    echo "--- Validating Eval: $EVAL_FILE ---"
    if python scripts/cb_data_generation/validate_format.py \
        --data "$EVAL_FILE"; then
        echo "✅ Eval set validation PASSED"
    else
        echo "❌ Eval set validation FAILED"
        VALIDATION_PASSED=false
    fi
else
    echo "⚠️  Eval file not found: $EVAL_FILE"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"
if $VALIDATION_PASSED; then
    echo "✅ All validations PASSED"
    echo ""
    echo "Data ready for training:"
    echo "  Ds: $DS_FILE"
    echo "  Dr: $DR_FILE"
    echo "  Eval: $EVAL_FILE"
    exit 0
else
    echo "❌ Some validations FAILED"
    echo "Please review errors above and regenerate data if needed."
    exit 1
fi
