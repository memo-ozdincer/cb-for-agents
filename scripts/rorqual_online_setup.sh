#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Rorqual ONLINE setup (run on LOGIN node)
# =============================================================================
# Purpose:
# - Do all steps that require outbound internet (pip installs + Hugging Face downloads)
# - Populate a shared cache in PROJECT so compute-node jobs can run fully offline
#
# Usage (login node):
#   cd /lustre09/project/6098391/memoozd/harmful-agents-meta-dataset
#   export HF_TOKEN=...   # required for gated Llama models
#   bash scripts/rorqual_online_setup.sh
#
# Optional overrides:
#   PROJECT_DIR=/lustre09/project/XXXX/you \
#   VENV_DIR=$PROJECT_DIR/.venvs/cb_env \
#   HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
#   bash scripts/rorqual_online_setup.sh
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-/lustre09/project/6098391/memoozd}"
REPO_DIR="${REPO_DIR:-$PROJECT_DIR/harmful-agents-meta-dataset}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venvs/cb_env}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_DIR/cb_cache}"
HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

# pip/network tuning (login nodes can be flaky / rate-limited)
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-60}"

# PyTorch wheel source (override if your site prefers a different CUDA build)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# Set to 1 to force reinstall/upgrade torch+torchvision even if a CUDA build is detected.
FORCE_TORCH_INSTALL="${FORCE_TORCH_INSTALL:-0}"

uv_install() {
  uv pip install "$@"
}

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR not found: $REPO_DIR"
  echo "Set REPO_DIR or PROJECT_DIR, then rerun."
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs "$PROJECT_DIR/.venvs" "$CACHE_ROOT"/{hf,wandb,torch,xdg}

# Hugging Face / Transformers caches (shared across jobs)
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# --- Modules (Alliance) ---
if command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load StdEnv/2023
  module load cuda/12.6
  module load python/3.11.5
fi

# --- Create + activate venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"

echo "Upgrading packaging tools..."
uv_install --upgrade pip setuptools wheel

echo "Installing Python deps (requirements.txt)..."
uv_install -r requirements.txt

echo "Checking torch installation..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("✅ PyTorch is installed (assumed CUDA-enabled via UV)")
except ImportError as e:
    print("ERROR: torch not found. Install with UV or pip first:")
    print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    import sys
    sys.exit(1)
PY

# --- Download model into HF cache ---
# IMPORTANT: Llama models are gated. You must:
# 1) have access on your HF account
# 2) accept the model terms on Hugging Face
# 3) provide a valid HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "ERROR: HF_TOKEN is not set."
  echo "This model is gated; export HF_TOKEN and rerun:"
  echo "  export HF_TOKEN=..."
  exit 1
fi

echo ""
echo "Downloading model snapshot into HF cache (this can take a while)..."
echo "Model: $HF_MODEL_ID"

python - <<PY
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("HF_MODEL_ID", "$HF_MODEL_ID")

# By setting HF_HOME above, huggingface_hub will cache under: $HF_HOME/hub
path = snapshot_download(
    repo_id=model_id,
    token=os.environ.get("HF_TOKEN"),
    resume_download=True,
)
print("✅ Downloaded/cached at:", path)
PY

echo ""
echo "=============================================="
echo "✅ ONLINE setup complete"
echo "Repo:   $REPO_DIR"
echo "Venv:   $VENV_DIR"
echo "Cache:  $CACHE_ROOT"
echo "Model:  $HF_MODEL_ID"
echo "Next: submit the compute job (offline)"
echo "=============================================="
