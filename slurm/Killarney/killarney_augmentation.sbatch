#!/bin/bash
#SBATCH --job-name=cb_augment
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-ssanner
#SBATCH --mail-type=ALL
#SBATCH --mail-user=memo.ozdincer@mail.utoronto.ca

##############################################################################
# Circuit Breaker Data Augmentation Pipeline
#
# This script generates additional agentic harm traces from:
# 1. Fujitsu B4 tool attack definitions → full multi-turn traces
# 2. AgentHarm behaviors → template-based completions
# 3. Attack scenario templates → diverse attack patterns
#
# No GPU required - this is CPU-only data generation.
##############################################################################

echo "=========================================="
echo "CB Data Augmentation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Setup
cd $SLURM_SUBMIT_DIR || cd /project/6105522/memoozd/harmful-agents-meta-dataset
source /project/6105522/memoozd/.venvs/cb_env/bin/activate

# Ensure PyYAML is available
pip install pyyaml --quiet 2>/dev/null || true

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Create output directory
mkdir -p data/augmented

##############################################################################
# Step 1: Generate Fujitsu B4 Traces
##############################################################################
echo "=========================================="
echo "Step 1: Generating B4 Multi-Turn Traces"
echo "=========================================="

python scripts/augmentation/generate_b4_traces.py \
    --input data/circuit_breakers/harmful/harmful_pairs.completions.jsonl \
    --output data/augmented/b4_traces.jsonl \
    --variants 1

echo "B4 traces generated: $(wc -l < data/augmented/b4_traces.jsonl 2>/dev/null || echo 0)"

##############################################################################
# Step 2: Generate AgentHarm Completions (Template Mode)
##############################################################################
echo ""
echo "=========================================="
echo "Step 2: Generating AgentHarm Completions"
echo "=========================================="

python scripts/augmentation/generate_agentharm_completions.py \
    --mode template \
    --output data/augmented/agentharm_completions.jsonl

echo "AgentHarm traces generated: $(wc -l < data/augmented/agentharm_completions.jsonl 2>/dev/null || echo 0)"

##############################################################################
# Step 3: Generate Attack Scenarios from Templates
##############################################################################
echo ""
echo "=========================================="
echo "Step 3: Generating Attack Scenarios"
echo "=========================================="

python scripts/augmentation/generate_attack_scenarios.py \
    --templates scripts/augmentation/attack_templates \
    --output data/augmented/attack_scenarios.jsonl \
    --num-per-category 100

echo "Attack scenarios generated: $(wc -l < data/augmented/attack_scenarios.jsonl 2>/dev/null || echo 0)"

##############################################################################
# Step 4: Merge Augmented Data into CB Training Pipeline
##############################################################################
echo ""
echo "=========================================="
echo "Step 4: Merging Augmented Data"
echo "=========================================="

python - << 'PYTHON_SCRIPT'
import json
from pathlib import Path

aug_dir = Path('data/augmented')
cb_dir = Path('data/circuit_breakers')

# Load existing harmful completions
existing = []
harmful_path = cb_dir / 'harmful' / 'harmful_pairs.completions.jsonl'
if harmful_path.exists():
    with open(harmful_path) as f:
        existing = [json.loads(l) for l in f if l.strip()]

print(f'Existing harmful completions: {len(existing)}')

# Count existing agentic
existing_agentic = sum(1 for r in existing if r.get('is_agentic'))
print(f'Existing agentic: {existing_agentic}')

# Load augmented data
augmented = []
for jsonl in aug_dir.glob('*.jsonl'):
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get('is_agentic'):
                    augmented.append(row)
    print(f'Loaded from {jsonl.name}: {sum(1 for r in augmented if r.get("source", "").startswith(jsonl.stem[:10]))} traces')

print(f'\nTotal augmented traces: {len(augmented)}')

# Merge (augmented goes after existing)
merged = existing + augmented

# Write to augmented file
merged_path = cb_dir / 'harmful' / 'harmful_pairs.completions.augmented.jsonl'
with open(merged_path, 'w') as f:
    for row in merged:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f'\nWrote {len(merged)} to {merged_path}')

# Final stats
agentic_count = sum(1 for r in merged if r.get('is_agentic'))
with_messages = sum(1 for r in merged if r.get('messages'))
with_tool_calls = sum(1 for r in merged if r.get('has_tool_calls'))

print(f'\n=== FINAL STATS ===')
print(f'Total samples: {len(merged)}')
print(f'Agentic: {agentic_count} ({100*agentic_count/len(merged):.1f}%)')
print(f'With messages[]: {with_messages}')
print(f'With tool_calls: {with_tool_calls}')
PYTHON_SCRIPT

##############################################################################
# Summary
##############################################################################
echo ""
echo "=========================================="
echo "Augmentation Complete"
echo "=========================================="
echo "End time: $(date)"
echo ""

echo "Files generated:"
ls -lh data/augmented/*.jsonl 2>/dev/null || echo "No augmented files found"

echo ""
echo "Merged file:"
ls -lh data/circuit_breakers/harmful/harmful_pairs.completions.augmented.jsonl 2>/dev/null || echo "Not found"

echo ""
echo "Next steps:"
echo "  1. Review augmented data quality"
echo "  2. If satisfied, use augmented file for training:"
echo "     cp data/circuit_breakers/harmful/harmful_pairs.completions.augmented.jsonl \\"
echo "        data/circuit_breakers/harmful/harmful_pairs.completions.jsonl"
echo "  3. Re-run CB batch creation:"
echo "     python scripts/format_for_cb/create_cb_batches.py"
echo "  4. Train:"
echo "     sbatch slurm/Killarney/killarney_cb_llama31_8b_4xl40s.sbatch"
