#!/bin/bash
#SBATCH --job-name=test_kl
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# KL Divergence Test - Minimal Training with Verbose Logging
# Trillium 1×H100
# =============================================================================
#
# Tests KL divergence implementation with gradient monitoring and ablations.
#
# Submit:
#   sbatch slurm/tests/train_kl.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache Setup
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,xdg_cache,xdg_config}
export HOME="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export XDG_CACHE_HOME="$CACHE_DIR/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_DIR/xdg_config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "========================================"
echo "KL Divergence Test - Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$CB_SCRATCH/data"
TRAINING_FILE="$DATA_DIR/cb_training_batches.jsonl"
TEST_OUTPUT="$CB_SCRATCH/tests/kl_test_${SLURM_JOB_ID}"

if [[ ! -f "$TRAINING_FILE" ]]; then
    echo "ERROR: Training file not found at $TRAINING_FILE"
    exit 1
fi

echo "Training file: $TRAINING_FILE ($(wc -l < "$TRAINING_FILE") batches)"

# Resolve Model Path
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$BASE_MODEL', local_files_only=True))")

# =============================================================================
# TEST 1: Baseline (No KL) - 50 steps
# =============================================================================
echo ""
echo "========================================"
echo "TEST 1: Baseline (beta_kl=0.0) - 50 steps"
echo "========================================"

accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$TRAINING_FILE" \
    --output-dir "${TEST_OUTPUT}/baseline_nokl" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 50 \
    --logging-steps 5 \
    --save-steps 50 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.0 \
    --wandb-project "circuit-breakers-test" \
    --wandb-run-name "test_kl_baseline_${SLURM_JOB_ID}"

echo "✓ Baseline test complete"

# =============================================================================
# TEST 2: Low KL (beta=0.1) - 50 steps
# =============================================================================
echo ""
echo "========================================"
echo "TEST 2: Low KL (beta_kl=0.1) - 50 steps"
echo "========================================"

accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$TRAINING_FILE" \
    --output-dir "${TEST_OUTPUT}/low_kl_0.1" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 50 \
    --logging-steps 5 \
    --save-steps 50 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.1 \
    --kl-temperature 1.0 \
    --wandb-project "circuit-breakers-test" \
    --wandb-run-name "test_kl_0.1_${SLURM_JOB_ID}"

echo "✓ Low KL test complete"

# =============================================================================
# TEST 3: Medium KL (beta=0.3) - 50 steps [RECOMMENDED DEFAULT]
# =============================================================================
echo ""
echo "========================================"
echo "TEST 3: Medium KL (beta_kl=0.3) - 50 steps"
echo "========================================"

accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$TRAINING_FILE" \
    --output-dir "${TEST_OUTPUT}/medium_kl_0.3" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 50 \
    --logging-steps 5 \
    --save-steps 50 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.3 \
    --kl-temperature 1.0 \
    --wandb-project "circuit-breakers-test" \
    --wandb-run-name "test_kl_0.3_${SLURM_JOB_ID}"

echo "✓ Medium KL test complete"

# =============================================================================
# TEST 4: High KL (beta=0.5) - 50 steps [AGGRESSIVE]
# =============================================================================
echo ""
echo "========================================"
echo "TEST 4: High KL (beta_kl=0.5) - 50 steps"
echo "========================================"

accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$TRAINING_FILE" \
    --output-dir "${TEST_OUTPUT}/high_kl_0.5" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 50 \
    --logging-steps 5 \
    --save-steps 50 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.5 \
    --kl-temperature 1.0 \
    --wandb-project "circuit-breakers-test" \
    --wandb-run-name "test_kl_0.5_${SLURM_JOB_ID}"

echo "✓ High KL test complete"

# =============================================================================
# TEST 5: Soft KL (beta=0.3, T=2.0) - 50 steps
# =============================================================================
echo ""
echo "========================================"
echo "TEST 5: Soft KL (beta_kl=0.3, T=2.0) - 50 steps"
echo "========================================"

accelerate launch --num_processes 1 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --base-model "$MODEL_PATH" \
    --data-path "$TRAINING_FILE" \
    --output-dir "${TEST_OUTPUT}/soft_kl_T2.0" \
    --loss-weighting dual \
    --alpha-max 10.0 \
    --alpha-decay-multiplier 2.0 \
    --total-steps 50 \
    --logging-steps 5 \
    --save-steps 50 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --cb-target-layers 10 12 14 16 18 20 \
    --beta-kl 0.3 \
    --kl-temperature 2.0 \
    --wandb-project "circuit-breakers-test" \
    --wandb-run-name "test_kl_T2.0_${SLURM_JOB_ID}"

echo "✓ Soft KL test complete"

# =============================================================================
# ANALYSIS: Compare Loss Curves
# =============================================================================
echo ""
echo "========================================"
echo "ANALYSIS: Extracting Loss Metrics"
echo "========================================"

python - <<'EOF'
import json
import os
from pathlib import Path

test_dir = Path(os.environ['TEST_OUTPUT'])
tests = [
    ("baseline_nokl", "Baseline (No KL)"),
    ("low_kl_0.1", "Low KL (β=0.1)"),
    ("medium_kl_0.3", "Medium KL (β=0.3)"),
    ("high_kl_0.5", "High KL (β=0.5)"),
    ("soft_kl_T2.0", "Soft KL (β=0.3, T=2.0)"),
]

print("\n" + "=" * 80)
print("KL DIVERGENCE TEST RESULTS")
print("=" * 80)

for test_name, test_label in tests:
    print(f"\n{test_label}:")
    print("-" * 40)

    # Check if test ran
    test_path = test_dir / test_name
    if not test_path.exists():
        print("  ⚠️  Test did not run")
        continue

    # Look for wandb logs (offline mode saves to wandb dir)
    wandb_dirs = list(test_path.glob("**/wandb/offline-*"))
    if wandb_dirs:
        print(f"  ✓ Found wandb logs: {len(wandb_dirs)} run(s)")

    # Check for checkpoints
    checkpoints = list(test_path.glob("checkpoint-*"))
    final = test_path / "final"
    if final.exists():
        print(f"  ✓ Final checkpoint saved")
    elif checkpoints:
        print(f"  ✓ Checkpoints: {len(checkpoints)}")
    else:
        print(f"  ⚠️  No checkpoints found")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
1. Check W&B logs for loss curves (sync offline logs with: wandb sync)
2. Compare final loss values at step 50:
   - loss_reroute should be similar across all tests (CB effect)
   - loss_retain should be lower with KL (better benign preservation)
   - loss_kl should be ~0.1-0.5 range (if too high, reduce beta_kl)
3. Monitor gradient flow:
   - grad_norm_total should be similar across tests
   - If KL suppresses gradients, reduce beta_kl
4. Recommended default: beta_kl=0.3 (medium) unless loss_kl dominates
""")

print("\nTo sync W&B logs:")
print(f"  cd {test_dir}")
print("  wandb sync */wandb/offline-*")
EOF

echo ""
echo "========================================"
echo "✓✓✓ KL DIVERGENCE TEST COMPLETE ✓✓✓"
echo "========================================"
echo ""
echo "Test outputs saved to: $TEST_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Review loss curves in W&B"
echo "  2. Compare reroute effectiveness (should be similar across tests)"
echo "  3. Verify KL doesn't suppress CB effect"
echo "  4. Choose beta_kl based on loss_kl magnitude"
echo ""
