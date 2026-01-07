#!/usr/bin/env python3
"""
CIRCUIT BREAKER TRAINING DATA PREPARATION PIPELINE

This script runs the complete data preparation pipeline for Circuit Breaker training:

1. Ingest raw data from all sources (Fujitsu, AgentDojo, AgentHarm, WebArena, etc.)
2. Extract completions from raw data (CRITICAL for effective CB training)
3. Create balanced training batches from completion-style data

IMPORTANT: Circuit Breaker training requires HARMFUL COMPLETIONS, not just prompts!
The model learns to reroute representations associated with GENERATING harmful content.
Without actual harmful completions, training will be ineffective.

Usage:
    # Full pipeline (recommended)
    python scripts/prepare_cb_training.py

    # Skip ingestion (if already done)
    python scripts/prepare_cb_training.py --skip-ingest

    # Skip completion extraction (not recommended)
    python scripts/prepare_cb_training.py --skip-completions

    # Dry run
    python scripts/prepare_cb_training.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"


def run_step(name: str, cmd: list, dry_run: bool = False) -> bool:
    """Run a pipeline step and return success status."""
    print("\n" + "=" * 70)
    print(f"  STEP: {name}")
    print("=" * 70)
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    print()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))

    if result.returncode != 0:
        print(f"\n❌ STEP FAILED: {name}")
        return False

    print(f"\n✅ STEP COMPLETE: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete CB training data preparation pipeline"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip raw data ingestion (use existing harmful_pairs.jsonl/benign_pairs.jsonl)",
    )
    parser.add_argument(
        "--skip-completions",
        action="store_true",
        help="Skip completion extraction (NOT recommended - will use prompt-only data)",
    )
    parser.add_argument(
        "--fujitsu-success-only",
        action="store_true",
        default=True,
        help="Only use Fujitsu samples where attack was judged successful (default: True)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training data (default: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#  CIRCUIT BREAKER TRAINING DATA PREPARATION PIPELINE" + " " * 16 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    if args.skip_completions:
        print("\n⚠️  WARNING: --skip-completions is set!")
        print("   This will create batches from PROMPT-ONLY data.")
        print("   CB training will be INEFFECTIVE without harmful completions!")
        print("   Press Ctrl+C to abort, or wait 5 seconds to continue...")
        if not args.dry_run:
            import time
            time.sleep(5)

    # Step 1: Ingest raw data
    if not args.skip_ingest:
        success = run_step(
            "Ingest Raw Data",
            [sys.executable, str(SCRIPTS_DIR / "ingest_cb_data.py")],
            dry_run=args.dry_run,
        )
        if not success:
            return 1
    else:
        print("\n⏭️  Skipping ingestion (--skip-ingest)")

    # Step 2: Extract completions
    if not args.skip_completions:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "format_for_cb" / "split_out_cb_completions.py"),
            "--write-rejected",
        ]
        if args.fujitsu_success_only:
            cmd.append("--fujitsu-success-only")

        success = run_step(
            "Extract Completions",
            cmd,
            dry_run=args.dry_run,
        )
        if not success:
            return 1
    else:
        print("\n⏭️  Skipping completion extraction (--skip-completions)")

    # Step 3: Create training batches
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "format_for_cb" / "create_cb_batches.py"),
        "--batch-size",
        str(args.batch_size),
    ]

    success = run_step(
        "Create Training Batches",
        cmd,
        dry_run=args.dry_run,
    )
    if not success:
        return 1

    # Final summary
    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE" + " " * 49 + "#")
    print("#" * 70)

    cb_batches = BASE_DIR / "data" / "circuit_breakers" / "cb_training_batches.jsonl"
    if cb_batches.exists() and not args.dry_run:
        # Count batches
        with open(cb_batches) as f:
            n_batches = sum(1 for _ in f)
        print(f"\n  Training batches: {cb_batches}")
        print(f"  Total batches: {n_batches}")
        print(f"  Total samples: {n_batches * args.batch_size}")

    print("\n  Next: Run training with:")
    print("    python scripts/train_circuit_breaker.py --preset llama-3.1-8b-instruct")
    print("\n  Or run diagnostics first:")
    print("    python scripts/diagnose_cb_pipeline.py --device cpu")

    return 0


if __name__ == "__main__":
    sys.exit(main())
