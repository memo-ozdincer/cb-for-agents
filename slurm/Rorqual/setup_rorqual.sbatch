#!/bin/bash
#SBATCH --job-name=cb_setup_rorqual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=def-zhijing_gpu

# =============================================================================
# Circuit Breakers Environment Setup - Rorqual
# =============================================================================
#
# ⚠️  IMPORTANT: Rorqual compute nodes have NO INTERNET ACCESS.
#     This script will FAIL if it tries to download packages.
#
# RECOMMENDED: Run setup interactively on the LOGIN NODE instead:
#
#   ssh rorqual.alliancecan.ca
#   cd /lustre09/project/6098391/memoozd/harmful-agents-meta-dataset
#   bash slurm/setup_rorqual.sbatch   # <-- run as bash script, not sbatch
#
# Or pre-download everything on a node with internet, then rsync.
#
# Rorqual storage (per Alliance docs):
#   HOME    = $HOME
#   SCRATCH = $HOME/links/scratch
#   PROJECT = $HOME/links/projects/<project-name>
#   Your project realpath: /lustre09/project/6098391/memoozd
#
# =============================================================================

set -euo pipefail

# Use PROJECT space (more quota, backed up daily)
PROJECT_DIR="/lustre09/project/6098391/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CACHE_ROOT="$PROJECT_DIR/cb_cache"

mkdir -p "$PROJECT_DIR/.venvs" "$REPO_DIR/logs" "$CACHE_ROOT"/{hf,wandb,torch,xdg}
cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"
echo "Which:  $(which python)"

export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo ""
echo "Repo:  $REPO_DIR"
echo "Venv:  $VENV_DIR"
echo "Cache: $CACHE_ROOT"
echo ""

uv pip install --upgrade pip setuptools wheel

echo "Installing Python deps (requirements.txt)..."
uv pip install -r requirements.txt

echo "Ensuring CUDA PyTorch wheels (cu121)..."
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("num gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}:", torch.cuda.get_device_name(i))
PY

python - << 'PY'
"""Sanity check HF cache for offline compute nodes.

Rorqual compute nodes typically have no outbound internet, so the model/tokenizer
must already exist in the local HF cache under $HF_HOME.
"""
import os

model_id = "meta-llama/Llama-3.1-8B-Instruct"

try:
  from huggingface_hub import snapshot_download

  snapshot_download(
    repo_id=model_id,
    local_files_only=True,
    resume_download=True,
    allow_patterns=[
      "*.json",
      "tokenizer.*",
      "special_tokens_map.json",
      "tokenizer_config.json",
      "config.json",
      "generation_config.json",
      "model.safetensors*",
      "pytorch_model*.bin*",
    ],
  )
  print(f"✅ HF cache looks populated for: {model_id}")
except Exception as e:
  print(f"⚠️  HF cache is missing files for: {model_id}")
  print("    On the LOGIN NODE (with internet), pre-download into the cache, e.g.:")
  print(f"      huggingface-cli download {model_id} --local-dir $HF_HOME/hub --local-dir-use-symlinks False")
  print("    Then rerun your training job; compute nodes run offline.")
  print(f"    Details: {e}")
PY

echo ""
echo "=============================================="
echo "✅ Rorqual setup complete"
echo "Activate: source $VENV_DIR/bin/activate"
echo "Logs:     $REPO_DIR/logs"
echo "=============================================="
