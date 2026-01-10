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
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/datasets,torch}

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# DO NOT SET TRANSFORMERS_CACHE - it's deprecated and causes cache fragmentation
# All files will go to HF_HOME/hub where offline mode expects them

echo "Cache directories:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo ""

# =============================================================================
# HuggingFace Token
# =============================================================================
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Some gated models may fail to download."
    echo "         Export it before running: export HF_TOKEN=hf_xxx"
    echo ""
else
    echo "HF_TOKEN: Set ✓"
    echo ""
fi

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
echo "Checking and Downloading Models"
echo "========================================"
echo "Using snapshot_download ensures proper Hub cache structure for offline mode"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Checking: $MODEL ---"
    
    python -c "
import os
from huggingface_hub import snapshot_download, try_to_load_from_cache, repo_info
from pathlib import Path

model_name = '$MODEL'
hf_token = os.environ.get('HF_TOKEN')
cache_dir = os.environ['HF_HUB_CACHE']

# Check if model is already fully cached
# Look for the cache directory structure
model_cache_name = model_name.replace('/', '--')
model_cache_path = Path(cache_dir) / f'models--{model_cache_name}'

if model_cache_path.exists():
    # Check if it has snapshot directories (means model files are there)
    snapshots_path = model_cache_path / 'snapshots'
    if snapshots_path.exists() and any(snapshots_path.iterdir()):
        print(f'✅ Model already cached at: {model_cache_path}')
        print('   Skipping download (use --force to re-download)')
        exit(0)

print(f'📥 Downloading {model_name} to {cache_dir}...')
print('   (This may take a while for large models)')

# Download using snapshot_download - this creates the exact cache structure
# that offline mode expects (models--org--name/snapshots/hash/...)
try:
    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        token=hf_token,
        resume_download=True,
        local_files_only=False,  # Allow internet access
    )
    print(f'✅ Model cached successfully to: {local_path}')
except Exception as e:
    print(f'❌ Failed to download: {e}')
    exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ $MODEL ready"
    else
        echo "❌ Failed to process $MODEL"
        exit 1
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
import os

hf_token = os.environ.get('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', token=hf_token)

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
# Verify All Models Are Cached
# =============================================================================
echo ""
echo "========================================"
echo "Verifying Cached Models"
echo "========================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    # Convert model name to cache directory name: meta-llama/Llama-3.1-8B-Instruct -> meta-llama--Llama-3.1-8B-Instruct
    MODEL_CACHE_NAME="${MODEL//\/\/--}"
    MODEL_PATH="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
    
    if [[ -d "$MODEL_PATH/snapshots" ]] && [[ -n "$(ls -A "$MODEL_PATH/snapshots" 2>/dev/null)" ]]; then
        echo "✅ $MODEL"
    elif [[ -d "$MODEL_PATH" ]]; then
        echo "⚠️  $MODEL (cached but no snapshots - may need re-download)"
    else
        echo "❌ $MODEL (MISSING - not downloaded)"
    fi
done

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

# =====
echo "ℹ️  You only need to run this script once (or when adding new models)."
echo "   It's safe to run multiple times - already cached models will be skipped."
echo ""========================================================================
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
echo "  export HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  export HF_HUB_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
echo ""
echo "DO NOT SET TRANSFORMERS_CACHE (deprecated - causes cache fragmentation)"
echo ""
