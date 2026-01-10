#!/bin/bash
# =============================================================================
# Pre-fetch Models and Data for Offline Compute Nodes
# =============================================================================
#
# Run this script on a login node (with internet) BEFORE submitting jobs.
# It downloads all HuggingFace models and tokenizers to the cache directory.
#
# Usage:
#   cd /project/def-zhijing/memoozd/harmful-agents-meta-dataset
#   bash slurm/Trillium/prefetch_models.sh
#
# =============================================================================

set -euo pipefail

echo "========================================"
echo "Pre-fetching Models for Offline Use"
echo "========================================"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

# Cache directories (same as in SLURM scripts)
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch}

export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"

echo "Cache directories:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# =============================================================================
# Load Environment
# =============================================================================
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please create it first with: python -m venv $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"
echo "Which: $(which python)"
echo ""

# =============================================================================
# Models to Download
# =============================================================================
# These are all the models used in Stage 1 MVP

MODELS=(
    # Base model for training and evaluation
    "meta-llama/Llama-3.1-8B-Instruct"
    
    # Abliterated model for Ds generation (attack data)
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
)

echo "========================================"
echo "Downloading Models"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Downloading: $MODEL ---"
    
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_name = '$MODEL'
print(f'Downloading tokenizer for {model_name}...')

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
print(f'  Tokenizer cached to: {tokenizer.name_or_path}')

print(f'Downloading model weights for {model_name}...')
print('  (This may take a while for large models)')

# Download model - use low memory mode to avoid OOM on login node
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print(f'  Model cached successfully')

# Clean up to free memory
del model
del tokenizer
import gc
gc.collect()
"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ $MODEL downloaded successfully"
    else
        echo "❌ Failed to download $MODEL"
    fi
done

# =============================================================================
# Download PEFT/LoRA dependencies
# =============================================================================
echo ""
echo "========================================"
echo "Verifying PEFT/LoRA Support"
echo "========================================"

python -c "
from peft import LoraConfig, get_peft_model
print('✅ PEFT/LoRA imports working')
"

# =============================================================================
# Verify Chat Template Support
# =============================================================================
echo ""
echo "========================================"
echo "Verifying Chat Template Support"
echo "========================================"

python -c "
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

# Test chat template
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello!'}
]

# Test with tools
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'test_tool',
            'description': 'A test tool',
            'parameters': {'type': 'object', 'properties': {}}
        }
    }
]

try:
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    print('✅ Chat template with tools works')
    print(f'  Template preview: {text[:200]}...')
except Exception as e:
    print(f'⚠️  Chat template with tools may not work: {e}')
    print('   Will fall back to basic formatting')
"

# =============================================================================
# Check Disk Space
# =============================================================================
echo ""
echo "========================================"
echo "Disk Space Usage"
echo "========================================"
echo ""
echo "Cache directory sizes:"
du -sh "$CACHE_ROOT"/* 2>/dev/null || echo "  (no cache files yet)"
echo ""
echo "Total cache size:"
du -sh "$CACHE_ROOT" 2>/dev/null || echo "  0"
echo ""
echo "Scratch quota:"
diskusage_report 2>/dev/null || df -h "$SCRATCH_DIR" 2>/dev/null || echo "  (quota check unavailable)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Pre-fetch Complete!"
echo "========================================"
echo ""
echo "Models are now cached at: $HF_HOME"
echo ""
echo "You can now submit jobs without internet:"
echo "  sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch"
echo ""
echo "Note: Make sure your SLURM scripts set these env vars:"
echo "  export HF_HOME=$HF_HOME"
echo "  export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  export HF_HUB_OFFLINE=1  # Optional: fail fast if model not cached"
echo ""
