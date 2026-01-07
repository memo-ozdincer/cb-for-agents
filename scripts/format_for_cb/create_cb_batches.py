#!/usr/bin/env python3
"""
Create CB training batches from completion-style data.

This script creates balanced 1:1 batches for Circuit Breaker training,
prioritizing completion-style data (with actual harmful/benign completions)
over prompt-only data.

IMPORTANT: For effective CB training, harmful samples MUST include actual
harmful completions (not just prompts). The model learns to reroute the
representations associated with GENERATING harmful content.

Data Priority:
1. Completion-style data (harmful_pairs.completions.jsonl, benign_pairs.completions.jsonl)
2. Prompt-only data (harmful_pairs.jsonl, benign_pairs.jsonl) - fallback only

Usage:
    # Create batches from completion data (recommended)
    python scripts/format_for_cb/create_cb_batches.py

    # Force use of prompt-only data (not recommended)
    python scripts/format_for_cb/create_cb_batches.py --use-prompt-only

    # Custom batch size
    python scripts/format_for_cb/create_cb_batches.py --batch-size 16
"""

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple


BASE_DIR = Path(__file__).resolve().parents[2]
CB_DIR = BASE_DIR / "data" / "circuit_breakers"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
    """Write list of dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def backup_file(path: Path, backup_dir: Path) -> None:
    """Create timestamped backup of existing file."""
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.{ts}.bak"
    shutil.copy2(path, backup_path)
    print(f"  Backed up: {path.name} -> {backup_path.name}")


def has_completion(sample: Dict[str, Any], is_harmful: bool) -> bool:
    """Check if sample has actual completion data."""
    # Check for pre-rendered text
    if sample.get("text"):
        return True
    # Check for completion fields
    if is_harmful:
        return bool(sample.get("harmful_completion"))
    else:
        return bool(sample.get("benign_completion"))


def load_data_with_priority(
    completion_path: Path,
    prompt_only_path: Path,
    fallback_path: Path,
    is_harmful: bool,
    use_prompt_only: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load data with priority: completions > prompt_only > fallback.

    Returns:
        (data, source_description)
    """
    if not use_prompt_only and completion_path.exists():
        data = read_jsonl(completion_path)
        if data:
            # Verify completions are present
            with_completion = sum(1 for d in data if has_completion(d, is_harmful))
            if with_completion > 0:
                return data, f"{completion_path.name} ({with_completion}/{len(data)} with completions)"

    if prompt_only_path.exists():
        data = read_jsonl(prompt_only_path)
        if data:
            return data, f"{prompt_only_path.name} (prompt-only)"

    if fallback_path.exists():
        data = read_jsonl(fallback_path)
        if data:
            return data, f"{fallback_path.name} (fallback)"

    return [], "NO DATA FOUND"


def create_balanced_batches(
    harmful: List[Dict[str, Any]],
    benign: List[Dict[str, Any]],
    batch_size: int = 16,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create strictly balanced 1:1 batches.

    Args:
        harmful: Harmful samples
        benign: Benign samples
        batch_size: Total batch size (split 50/50)
        seed: Random seed

    Returns:
        List of batches, each with harmful and benign keys
    """
    random.seed(seed)
    random.shuffle(harmful)
    random.shuffle(benign)

    samples_per_side = batch_size // 2
    max_batches = min(len(harmful), len(benign)) // samples_per_side

    batches = []
    for i in range(max_batches):
        batch = {
            "harmful": harmful[i * samples_per_side : (i + 1) * samples_per_side],
            "benign": benign[i * samples_per_side : (i + 1) * samples_per_side],
        }
        batches.append(batch)

    return batches


def analyze_completions(data: List[Dict[str, Any]], is_harmful: bool) -> Dict[str, int]:
    """Analyze completion coverage in data."""
    stats = {
        "total": len(data),
        "with_text": 0,
        "with_completion_field": 0,
        "prompt_only": 0,
    }

    completion_key = "harmful_completion" if is_harmful else "benign_completion"

    for sample in data:
        if sample.get("text"):
            stats["with_text"] += 1
        elif sample.get(completion_key):
            stats["with_completion_field"] += 1
        else:
            stats["prompt_only"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create CB training batches from completion-style data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Total samples per batch (default: 16, split 8+8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use-prompt-only",
        action="store_true",
        help="Force use of prompt-only data (NOT recommended for CB training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CB_DIR / "cb_training_batches.jsonl",
        help="Output path for batched data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CIRCUIT BREAKER BATCH CREATION")
    print("=" * 70)

    # Define data paths
    harmful_completions = CB_DIR / "harmful" / "harmful_pairs.completions.jsonl"
    harmful_prompt_only = CB_DIR / "harmful" / "harmful_pairs.prompt_only.jsonl"
    harmful_fallback = CB_DIR / "harmful" / "harmful_pairs.jsonl"

    benign_completions = CB_DIR / "benign" / "benign_pairs.completions.jsonl"
    benign_prompt_only = CB_DIR / "benign" / "benign_pairs.prompt_only.jsonl"
    benign_fallback = CB_DIR / "benign" / "benign_pairs.jsonl"

    # Load harmful data
    print("\n--- Loading Harmful Data ---")
    harmful, harmful_source = load_data_with_priority(
        harmful_completions,
        harmful_prompt_only,
        harmful_fallback,
        is_harmful=True,
        use_prompt_only=args.use_prompt_only,
    )
    print(f"  Source: {harmful_source}")
    print(f"  Loaded: {len(harmful)} samples")

    # Analyze harmful completions
    if harmful:
        stats = analyze_completions(harmful, is_harmful=True)
        print(f"  Completion coverage:")
        print(f"    - With 'text' field: {stats['with_text']}")
        print(f"    - With 'harmful_completion': {stats['with_completion_field']}")
        print(f"    - Prompt-only: {stats['prompt_only']}")

        if stats["prompt_only"] == stats["total"]:
            print("\n  ⚠️  WARNING: All harmful samples are prompt-only!")
            print("       This will NOT train effective Circuit Breakers!")
            print("       Run: python scripts/format_for_cb/split_out_cb_completions.py --fujitsu-success-only")

    # Load benign data
    print("\n--- Loading Benign Data ---")
    benign, benign_source = load_data_with_priority(
        benign_completions,
        benign_prompt_only,
        benign_fallback,
        is_harmful=False,
        use_prompt_only=args.use_prompt_only,
    )
    print(f"  Source: {benign_source}")
    print(f"  Loaded: {len(benign)} samples")

    # Analyze benign completions
    if benign:
        stats = analyze_completions(benign, is_harmful=False)
        print(f"  Completion coverage:")
        print(f"    - With 'text' field: {stats['with_text']}")
        print(f"    - With 'benign_completion': {stats['with_completion_field']}")
        print(f"    - Prompt-only: {stats['prompt_only']}")

    # Validate data
    if not harmful:
        print("\n❌ ERROR: No harmful data found!")
        print("   Run: python scripts/ingest_cb_data.py")
        return 1

    if not benign:
        print("\n❌ ERROR: No benign data found!")
        print("   Run: python scripts/ingest_cb_data.py")
        return 1

    # Create batches
    print("\n--- Creating Balanced Batches ---")
    batches = create_balanced_batches(
        harmful, benign, batch_size=args.batch_size, seed=args.seed
    )

    samples_per_side = args.batch_size // 2
    max_possible = min(len(harmful), len(benign)) // samples_per_side

    print(f"  Batch size: {args.batch_size} ({samples_per_side} harmful + {samples_per_side} benign)")
    print(f"  Created: {len(batches)} batches (max possible: {max_possible})")
    print(f"  Total samples used: {len(batches) * args.batch_size}")

    # Write output
    if args.dry_run:
        print(f"\n  [DRY RUN] Would write to: {args.output}")
    else:
        backup_dir = CB_DIR / "_backups"
        backup_file(args.output, backup_dir)
        n = write_jsonl(args.output, batches)
        print(f"\n✅ Wrote {n} batches to {args.output}")

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Check for completion coverage
    harmful_with_completion = sum(1 for h in harmful if has_completion(h, True))
    benign_with_completion = sum(1 for b in benign if has_completion(b, False))

    print(f"  Harmful samples with completions: {harmful_with_completion}/{len(harmful)} ({100*harmful_with_completion/len(harmful):.1f}%)")
    print(f"  Benign samples with completions: {benign_with_completion}/{len(benign)} ({100*benign_with_completion/len(benign):.1f}%)")

    if harmful_with_completion < len(harmful) * 0.5:
        print("\n  ⚠️  Less than 50% of harmful samples have completions!")
        print("       CB training effectiveness may be limited.")
        print("       Consider running: python scripts/format_for_cb/split_out_cb_completions.py --fujitsu-success-only")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
