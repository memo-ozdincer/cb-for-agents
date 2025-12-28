#!/bin/bash
#SBATCH --job-name=cb_llama4_1xh100_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Circuit Breakers Debug/Smoke Test - Trillium 1×H100
# =============================================================================
#
# Quick test to verify environment works before full 4-GPU training.
#
# Submit from $SCRATCH:
#   cd $SCRATCH/harmful-agents-meta-dataset
#   sbatch slurm/trillium_cb_llama4_1xh100_debug.sbatch
#
# =============================================================================

set -euo pipefail
cd "$SCRATCH"

source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$SCRATCH/.venvs/cb_env/bin/activate"

# Quick smoke test
echo "=========================================="
echo "Testing GPU access and libraries"
echo "=========================================="
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "Testing transformers/peft"
python -c "import transformers, peft, accelerate; print('transformers:', transformers.__version__); print('peft:', peft.__version__); print('accelerate:', accelerate.__version__)"

echo ""
echo "✅ Environment test passed"
