#!/bin/bash
#SBATCH --job-name=sweep_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Sweep Training: Test multiple policies and configs
# Assumes data is already processed through stage 3 (renders/lossmasks exist)
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

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "SWEEP TRAINING"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Data Paths (already processed)
# =============================================================================
DATA_DIR="$CB_SCRATCH/data"
RENDERS_DIR="$DATA_DIR/renders"
LOSSMASKS_DIR="$DATA_DIR/lossmasks"
TRACES_DIR="$DATA_DIR/traces"

# Fujitsu DS/DR
DS_RENDERS="$RENDERS_DIR/fujitsu_b4_ds.jsonl"
DS_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_ds.jsonl"
DR_RENDERS="$RENDERS_DIR/fujitsu_b4_dr.jsonl"
DR_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_dr.jsonl"

# AgentDojo
AGENTDOJO_RENDERS="$RENDERS_DIR/agentdojo_complete.jsonl"
AGENTDOJO_LOSSMASKS="$LOSSMASKS_DIR/agentdojo_complete.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"

echo "Data files:"
echo "  DS Renders: $(wc -l < "$DS_RENDERS") lines"
echo "  DR Renders: $(wc -l < "$DR_RENDERS") lines"
echo "  AgentDojo Renders: $(wc -l < "$AGENTDOJO_RENDERS") lines"
echo ""

# =============================================================================
# Sweep Configurations
# =============================================================================

# Define sweep parameters
ALPHAS=(5.0 10.0 15.0)
CB_LAYER_CONFIGS=("10,20" "8,16,24" "12,18")
LOSS_WEIGHTINGS=("dual" "single_alpha")
TOTAL_STEPS=300
BATCH_SIZE=8

# Create output directory for this sweep
SWEEP_DIR="$CB_SCRATCH/models/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

echo "Sweep output: $SWEEP_DIR"
echo ""
echo "Configurations to test:"
echo "  ALPHAS: ${ALPHAS[*]}"
echo "  CB_LAYERS: ${CB_LAYER_CONFIGS[*]}"
echo "  LOSS_WEIGHTING: ${LOSS_WEIGHTINGS[*]}"
echo ""

# =============================================================================
# Run Sweep
# =============================================================================
run_idx=0

for alpha in "${ALPHAS[@]}"; do
    for cb_layers in "${CB_LAYER_CONFIGS[@]}"; do
        for loss_weighting in "${LOSS_WEIGHTINGS[@]}"; do
            run_idx=$((run_idx + 1))

            # Create run name
            layers_short="${cb_layers//,/_}"
            RUN_NAME="a${alpha}_l${layers_short}_${loss_weighting}"
            RUN_DIR="$SWEEP_DIR/$RUN_NAME"

            echo "========================================"
            echo "Run $run_idx: $RUN_NAME"
            echo "========================================"
            echo "  ALPHA_MAX: $alpha"
            echo "  CB_LAYERS: $cb_layers"
            echo "  LOSS_WEIGHTING: $loss_weighting"
            echo "  Output: $RUN_DIR"
            echo ""

            mkdir -p "$RUN_DIR"

            # Parse CB_LAYERS to array
            IFS=',' read -ra CB_LAYER_ARRAY <<< "$cb_layers"

            # Build training arguments
            TRAIN_ARGS=(
                --preset "llama-3.1-8b-instruct"
                --output-dir "$RUN_DIR"
                --total-steps "$TOTAL_STEPS"
                --batch-size "$BATCH_SIZE"
                --learning-rate "5e-5"
                --warmup-steps 20
                --alpha-max "$alpha"
                --max-seq-length 2048
                --gradient-accumulation-steps 1
                --loss-weighting "$loss_weighting"
                --lora-r 16
                --lora-alpha 32
                --lora-dropout 0.05
                --logging-steps 10
                --save-steps 50
                --wandb-project "circuit-breakers-sweep"
                --wandb-run-name "$RUN_NAME"
                --cb-target-layers "${CB_LAYER_ARRAY[@]}"
                --mode ds_dr
                --ds-renders "$DS_RENDERS"
                --ds-lossmasks "$DS_LOSSMASKS"
                --dr-renders "$DR_RENDERS"
                --dr-lossmasks "$DR_LOSSMASKS"
                --additional-renders "$AGENTDOJO_RENDERS"
                --additional-lossmasks "$AGENTDOJO_LOSSMASKS"
                --additional-traces "$AGENTDOJO_TRACES"
            )

            # Run training
            python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"

            echo ""
            echo "Run $run_idx complete: $RUN_NAME"
            echo ""
        done
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Total runs: $run_idx"
echo "Output directory: $SWEEP_DIR"
echo ""
echo "Run directories:"
ls -la "$SWEEP_DIR"/
echo ""
echo "Finished at: $(date)"
