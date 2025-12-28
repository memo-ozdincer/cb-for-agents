#!/bin/bash
# =============================================================================
# Circuit Breakers Fir Setup Script (Alliance)
# =============================================================================
#
# Fir conventions (per Alliance docs):
# - SCRATCH is typically available at: $HOME/scratch
#   (often a symlink to /scratch/$USER/...) and is NOT backed up.
#
# This script:
# - Creates a Python venv in $HOME/scratch/.venvs/cb_env (by default)
# - Moves HF / datasets / wandb / torch caches to $HOME/scratch
# - Creates a repo-local logs/ directory used by sbatch scripts
#
# Usage (on Fir login node):
#   ssh fir.alliancecan.ca
#   cd $HOME/scratch
#   git clone <your-repo-url> harmful-agents-meta-dataset
#   cd harmful-agents-meta-dataset
#   bash scripts/hpc_setup_fir.sh
#
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  CB Fir Setup (Alliance)"
echo "=============================================="

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fir scratch is documented as $HOME/scratch; allow override if you use something else.
: "${SCRATCH_DIR:=$HOME/scratch}"

# Venv location (override via ENV_ROOT/ENV_NAME)
: "${ENV_ROOT:=$SCRATCH_DIR/.venvs}"
: "${ENV_NAME:=cb_env}"
VENV_DIR="$ENV_ROOT/$ENV_NAME"

mkdir -p "$ENV_ROOT"
mkdir -p "$REPO_DIR/logs"

echo "Repo:      $REPO_DIR"
echo "Scratch:   $SCRATCH_DIR"
echo "Env root:  $ENV_ROOT"
echo "Venv dir:  $VENV_DIR"

# Alliance modules (Fir should have these; adjust versions if needed)
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

echo "Modules loaded:"
module list

# uv install (user-space)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"

# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python 3.11
fi

source "$VENV_DIR/bin/activate"
python -V
which python

# Put caches on scratch (avoid $HOME quotas + improve IO)
: "${CACHE_ROOT:=$SCRATCH_DIR/cb_cache}"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}

export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo "Cache root: $CACHE_ROOT"

uv pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA wheels (kept consistent with the Trillium script)
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core deps (same set as scripts/hpc_setup_trillium.sh)
uv pip install \
  "transformers>=4.45.0" \
  "peft>=0.13.0" \
  "accelerate>=1.0.0" \
  "bitsandbytes>=0.44.0" \
  "deepspeed>=0.15.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "wandb>=0.18.0" \
  "tensorboard>=2.17.0" \
  "pandas>=2.0.0" \
  "numpy>=1.24.0" \
  "pyarrow>=15.0.0" \
  "datasets>=2.19.0" \
  "huggingface-hub>=0.25.0" \
  "tqdm>=4.66.0"

echo "Installing flash-attn (optional, may take time)..."
uv pip install flash-attn --no-build-isolation || true

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("num gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}:", torch.cuda.get_device_name(i))
PY

echo "=============================================="
echo "✅ Fir setup complete"
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Logs directory:"
echo "  $REPO_DIR/logs"
echo "Next: submit the Fir smoke test:"
echo "  cd $REPO_DIR && sbatch slurm/fir_cb_llama4_1xh100_debug.sbatch"
echo "=============================================="
