#!/bin/bash
# =============================================================================
# Cache HuggingFace models for offline SLURM usage
# Run this on the login node (with internet) before submitting jobs
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CACHE_DIR="$CB_SCRATCH/cache"

# Models to cache
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

echo "========================================"
echo "Caching HuggingFace Models"
echo "========================================"
echo "Cache directory: $CACHE_DIR/hf"
echo ""

# Setup cache directories
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets}

# Activate venv
source "$VENV_DIR/bin/activate"

# Set cache paths
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"

# Ensure online mode for downloading
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Caching: $model"
    echo "----------------------------------------"

    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

model_name = '$model'
print(f'Downloading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Tokenizer cached: {tokenizer.name_or_path}')

print(f'Downloading config for {model_name}...')
config = AutoConfig.from_pretrained(model_name)
print(f'Config cached')

print(f'Downloading model weights for {model_name}...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print(f'Model cached: {model.config._name_or_path}')
print('Done!')
"

    echo ""
done

echo "========================================"
echo "Cache Complete"
echo "========================================"
echo ""
echo "Cached models:"
ls -la "$CACHE_DIR/hf/hub/" | grep "models--"
echo ""
echo "You can now run SLURM jobs with HF_HUB_OFFLINE=1"
