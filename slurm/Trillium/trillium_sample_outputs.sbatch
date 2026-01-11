#!/bin/bash
#SBATCH --job-name=sample_outputs
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# ============================================================================
# Sample Outputs - See what baseline vs CB model actually produces
# ============================================================================

set -e

echo "=============================================="
echo "Sample Outputs - Stage 1 CB Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=============================================="

# Paths
PROJECT_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
SCRATCH_DIR="/scratch/memoozd"
EVAL_DATA="$SCRATCH_DIR/cb_mvp_data/eval_stage1.jsonl"
CB_ADAPTER="$SCRATCH_DIR/cb_runs/195854/outputs/cb_mvp_adapter/final"
TOOL_SCHEMA="$PROJECT_DIR/configs/tool_schemas/b4_standard_v1.json"

# Activate environment
source "$PROJECT_DIR/agent_adversarial/bin/activate"

cd "$PROJECT_DIR"

echo ""
echo "Eval data: $EVAL_DATA"
echo "CB adapter: $CB_ADAPTER"
echo "Tool schema: $TOOL_SCHEMA"
echo ""

# Run sample outputs
python scripts/circuit_breakers/sample_outputs.py \
    --eval-data "$EVAL_DATA" \
    --cb-adapter "$CB_ADAPTER" \
    --tool-schema "$TOOL_SCHEMA" \
    --n 15 \
    2>&1 | tee "$SCRATCH_DIR/sample_outputs_${SLURM_JOB_ID}.txt"

echo ""
echo "=============================================="
echo "Output saved to: $SCRATCH_DIR/sample_outputs_${SLURM_JOB_ID}.txt"
echo "=============================================="
