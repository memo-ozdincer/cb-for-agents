#!/bin/bash
# =============================================================================
# LOCAL STAGE 2 DATA TEST (run before cluster submission)
# =============================================================================
#
# Quick local sanity check of Stage 2 data pipeline
# Run this first to catch errors before submitting to cluster
#
# Usage:
#   chmod +x test_stage2_local.sh
#   ./test_stage2_local.sh
#
# =============================================================================

set -e

echo "========================================================================"
echo "STAGE 2 LOCAL DATA TEST"
echo "========================================================================"
echo ""

# Check we're in the right directory
if [[ ! -f "scripts/cb_data_generation/validate_format.py" ]]; then
    echo "❌ ERROR: Must run from repo root"
    exit 1
fi

# Test 1: Validation
echo "=== Test 1: Data Validation ==="
python scripts/cb_data_generation/validate_format.py \
    --data data/circuit_breakers/stage2/train.jsonl

echo ""

# Test 2: Quick stats
echo "=== Test 2: Data Statistics ==="
python << 'PYEOF'
import json
from collections import Counter

with open("data/circuit_breakers/stage2/train.jsonl") as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"Total samples: {len(samples)}")

splits = Counter(s.get("labels", {}).get("split") for s in samples)
print(f"By split: {dict(splits)}")

sources = Counter(s.get("metadata", {}).get("source") for s in samples)
print(f"By source: {dict(sources)}")

harmful = splits.get("harmful", 0)
retain = splits.get("retain", 0)
ratio = retain / harmful if harmful else 0
print(f"Dr:Ds ratio: {ratio:.2f}:1")

# Check schema
has_id = sum(1 for s in samples if s.get("id"))
has_training = sum(1 for s in samples if s.get("training"))
has_schema_version = sum(1 for s in samples if s.get("metadata", {}).get("schema_version"))

print(f"\nSchema compliance:")
print(f"  - Has 'id': {has_id}/{len(samples)} ({100*has_id/len(samples):.1f}%)")
print(f"  - Has 'training': {has_training}/{len(samples)} ({100*has_training/len(samples):.1f}%)")
print(f"  - Has 'schema_version': {has_schema_version}/{len(samples)} ({100*has_schema_version/len(samples):.1f}%)")

if has_id == len(samples) and has_training == len(samples) and has_schema_version == len(samples):
    print("\n✅ All samples conform to canonical schema")
else:
    print("\n⚠️  Some samples missing canonical schema fields")
PYEOF

echo ""

# Test 3: Sample preview
echo "=== Test 3: Sample Preview ==="
echo "First harmful sample:"
python << 'PYEOF'
import json

with open("data/circuit_breakers/stage2/train.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        if sample.get("labels", {}).get("split") == "harmful":
            print(f"  ID: {sample.get('id')}")
            print(f"  Source: {sample.get('metadata', {}).get('source')}")
            print(f"  Messages: {len(sample.get('messages', []))}")
            print(f"  Has tools: {sample.get('tools') is not None}")
            print(f"  Priority class: {sample.get('training', {}).get('priority_class')}")
            break
PYEOF

echo ""
echo "First retain sample:"
python << 'PYEOF'
import json

with open("data/circuit_breakers/stage2/train.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        if sample.get("labels", {}).get("split") == "retain":
            print(f"  ID: {sample.get('id')}")
            print(f"  Source: {sample.get('metadata', {}).get('source')}")
            print(f"  Messages: {len(sample.get('messages', []))}")
            print(f"  Has tools: {sample.get('tools') is not None}")
            print(f"  Priority class: {sample.get('training', {}).get('priority_class')}")
            break
PYEOF

echo ""
echo "========================================================================"
echo "✅ LOCAL TESTS PASSED"
echo "========================================================================"
echo ""
echo "Ready to submit to cluster:"
echo "  cd /scratch/memoozd/harmful-agents-meta-dataset"
echo "  sbatch slurm/Trillium/trillium_stage2_test_cpu.sbatch"
echo ""
