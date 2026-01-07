#!/usr/bin/env python3
"""
Master script to run the full agentic harm data augmentation pipeline.

This script orchestrates:
1. Fujitsu B4 trace generation (tool flip attacks)
2. Attack scenario generation from templates
3. Quality filtering
4. Integration with CB training data

Usage:
    python scripts/augmentation/run_augmentation_pipeline.py
    python scripts/augmentation/run_augmentation_pipeline.py --skip-b4 --num-scenarios 100
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_step(name: str, cmd: list, check: bool = True) -> bool:
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parents[2])
    if result.returncode != 0 and check:
        print(f"\n  STEP FAILED: {name}")
        return False
    print(f"\n  STEP COMPLETE: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-b4",
        action="store_true",
        help="Skip Fujitsu B4 trace generation",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip attack scenario generation",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="Number of scenarios per attack category",
    )
    parser.add_argument(
        "--b4-variants",
        type=int,
        default=1,
        help="Number of variants per B4 attack",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - don't write files",
    )
    parser.add_argument(
        "--integrate",
        action="store_true",
        help="Integrate augmented data into CB training pipeline",
    )
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parents[2]
    SCRIPTS_DIR = BASE_DIR / "scripts"
    AUG_DIR = SCRIPTS_DIR / "augmentation"
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = DATA_DIR / "augmented"

    print("#" * 70)
    print("#")
    print("#  AGENTIC HARM DATA AUGMENTATION PIPELINE")
    print("#")
    print("#" * 70)
    print(f"\nStarted: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    # Step 1: Generate B4 traces
    if not args.skip_b4:
        cmd = [
            sys.executable,
            str(AUG_DIR / "generate_b4_traces.py"),
            "--input", str(DATA_DIR / "circuit_breakers/harmful/harmful_pairs.completions.jsonl"),
            "--output", str(OUTPUT_DIR / "b4_traces.jsonl"),
            "--variants", str(args.b4_variants),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        success = run_step("Generate Fujitsu B4 Traces", cmd) and success
    else:
        print("\n(Skipping B4 trace generation)")

    # Step 2: Generate attack scenarios
    if not args.skip_scenarios:
        cmd = [
            sys.executable,
            str(AUG_DIR / "generate_attack_scenarios.py"),
            "--templates", str(AUG_DIR / "attack_templates"),
            "--output", str(OUTPUT_DIR / "attack_scenarios.jsonl"),
            "--num-per-category", str(args.num_scenarios),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        success = run_step("Generate Attack Scenarios", cmd) and success
    else:
        print("\n(Skipping attack scenario generation)")

    # Step 3: Show summary
    print("\n" + "=" * 70)
    print("  AUGMENTATION SUMMARY")
    print("=" * 70)

    if not args.dry_run:
        total_traces = 0
        for jsonl in OUTPUT_DIR.glob("*.jsonl"):
            with open(jsonl) as f:
                count = sum(1 for _ in f)
            print(f"  {jsonl.name}: {count} traces")
            total_traces += count
        print(f"\n  Total augmented traces: {total_traces}")
    else:
        print("  (dry-run - no files written)")

    # Step 4: Integration (optional)
    if args.integrate and not args.dry_run:
        print("\n" + "=" * 70)
        print("  INTEGRATING AUGMENTED DATA")
        print("=" * 70)

        # Merge augmented data into harmful pairs
        cmd = [
            sys.executable, "-c", f"""
import json
from pathlib import Path

aug_dir = Path('{OUTPUT_DIR}')
cb_dir = Path('{DATA_DIR}/circuit_breakers')

# Load existing harmful completions
existing = []
harmful_path = cb_dir / 'harmful' / 'harmful_pairs.completions.jsonl'
if harmful_path.exists():
    with open(harmful_path) as f:
        existing = [json.loads(l) for l in f if l.strip()]

print(f'Existing harmful completions: {{len(existing)}}')

# Load augmented data
augmented = []
for jsonl in aug_dir.glob('*.jsonl'):
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # Ensure it has the required fields
                if row.get('is_agentic'):
                    augmented.append(row)

print(f'Augmented traces: {{len(augmented)}}')

# Merge (augmented goes after existing)
merged = existing + augmented

# Write to new file
merged_path = cb_dir / 'harmful' / 'harmful_pairs.completions.augmented.jsonl'
with open(merged_path, 'w') as f:
    for row in merged:
        f.write(json.dumps(row, ensure_ascii=False) + '\\n')

print(f'Wrote {{len(merged)}} to {{merged_path}}')

# Stats
agentic_count = sum(1 for r in merged if r.get('is_agentic'))
with_messages = sum(1 for r in merged if r.get('messages'))
print(f'\\nAgentic: {{agentic_count}} ({{100*agentic_count/len(merged):.1f}}%)')
print(f'With messages[]: {{with_messages}}')
"""
        ]
        subprocess.run(cmd)

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)
    print(f"\nFinished: {datetime.now().isoformat()}")

    if not args.dry_run:
        print(f"""
Next steps:
  1. Review augmented data:
     head -5 {OUTPUT_DIR}/*.jsonl | python -m json.tool

  2. Integrate into CB training (if not already done):
     python scripts/augmentation/run_augmentation_pipeline.py --integrate

  3. Re-run CB data pipeline:
     python scripts/prepare_cb_training.py
""")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
