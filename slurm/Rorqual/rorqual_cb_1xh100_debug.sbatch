#!/bin/bash
#SBATCH --job-name=cb_1xh100_smoke_rorqual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing_gpu

# =============================================================================
# Circuit Breakers Debug/Smoke Test - Rorqual 1×H100
# =============================================================================
#
# Rorqual storage (per Alliance docs):
#   HOME    = $HOME
#   SCRATCH = $HOME/links/scratch
#   PROJECT = $HOME/links/projects/<project-name>
#   Your project realpath: /lustre09/project/6098391/memoozd
#
# Submit:
#   cd /lustre09/project/6098391/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/rorqual_cb_1xh100_debug.sbatch
#
# Notes:
# - Rorqual compute nodes have NO INTERNET ACCESS.
# - Site policy: jobs >= 1 hour (>= 5 min for test jobs), max 7 days.
# - GPU nodes: 64 cores, 498G memory, 4×H100 per node.
# - MIG instances available: h100_1g.10gb, h100_2g.20gb, h100_3g.40gb
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/lustre09/project/6098391/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p logs

module purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run setup on LOGIN NODE first (no internet on compute):"
  echo "  bash slurm/setup_rorqual.sbatch"
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
echo "✅ Rorqual smoke test passed"
echo "Job finished: $(date)"
