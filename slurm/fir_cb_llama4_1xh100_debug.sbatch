#!/bin/bash
#SBATCH --job-name=cb_llama4_1xh100_smoke_fir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing  # TODO: set your RAC / def-* account

# =============================================================================
# Circuit Breakers Debug/Smoke Test - Fir 1×H100
# =============================================================================
#
# Fir storage (per Alliance docs):
#   HOME    = $HOME
#   SCRATCH = $HOME/scratch
#   PROJECT = $HOME/project/${def-project-id}
#
# Recommended: submit from your SCRATCH clone:
#   cd $HOME/scratch/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/fir_cb_llama4_1xh100_debug.sbatch
#
# Notes:
# - Fir compute nodes have internet access.
# - Site policy: jobs should be >= 1 hour (>= 5 min for test jobs).
# - If you want a MIG instance instead of a full H100, replace the GPU line with:
#     #SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#   (or 2g.20gb / 3g.40gb)
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Job started: $(date)"
echo "Host:        $(hostname)"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-N/A}"
echo "Work dir:    $(pwd)"
echo "=========================================="

# --- Paths (edit if your layout differs) ---
REPO_DIR="$HOME/scratch/harmful-agents-meta-dataset"
VENV_DIR="$HOME/scratch/.venvs/cb_env"

# If you submitted from inside the repo, prefer that.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR" ]]; then
  if [[ -f "$SLURM_SUBMIT_DIR/requirements.txt" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  fi
fi

cd "$REPO_DIR"

# Must exist before Slurm opens stdout/stderr files (see #SBATCH --output/--error).
mkdir -p logs

# --- Modules / environment ---
# These are typical on Alliance clusters; adjust if Fir has different module names.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Create it once (example):"
  echo "  module load StdEnv/2023 python/3.11.5"
  echo "  python -m venv $VENV_DIR"
  echo "  source $VENV_DIR/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "=========================================="
echo "GPU / driver sanity checks"
echo "=========================================="

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python - <<'PY'
import os
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print('device[0]:', torch.cuda.get_device_name(0))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
PY

echo ""
echo "=========================================="
echo "Transformers / PEFT / Accelerate sanity"
echo "=========================================="

python - <<'PY'
import transformers, peft, accelerate
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('accelerate:', accelerate.__version__)
PY

echo ""
echo "✅ Fir smoke test passed"
echo "Job finished: $(date)"
