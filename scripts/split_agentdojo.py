#!/usr/bin/env python3
"""Split AgentDojo traces/renders/lossmasks into harmful and benign files.

AgentDojo data has both harmful and benign samples in the same file,
distinguished by labels.is_harmful. This script splits them so they
can be used with train_schema.py's mixed mode alongside Fujitsu DS/DR.

The traces file contains the labels.is_harmful field.
Renders and lossmasks are split by looking up trace_id in the traces labels.
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_trace_labels(traces_path: Path) -> Dict[str, bool]:
    """Load trace_id -> is_harmful mapping from traces file."""
    labels = {}
    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_id = row.get("id") or row.get("trace_id")
            # Check both is_harmful (boolean) and category (string) for compatibility
            trace_labels = row.get("labels", {})
            if "is_harmful" in trace_labels:
                is_harmful = trace_labels.get("is_harmful", False)
            else:
                # AgentDojo uses labels.category = "harmful" | "benign"
                is_harmful = trace_labels.get("category") == "harmful"
            if trace_id:
                labels[trace_id] = is_harmful
    return labels


def split_traces(input_path: Path, harmful_out: Path, benign_out: Path) -> tuple:
    """Split traces file by labels.category or labels.is_harmful."""
    harmful_count = 0
    benign_count = 0

    with open(input_path) as f_in, \
         open(harmful_out, 'w') as f_harmful, \
         open(benign_out, 'w') as f_benign:

        for line in f_in:
            if not line.strip():
                continue

            row = json.loads(line)
            trace_labels = row.get("labels", {})
            # Check both is_harmful (boolean) and category (string) for compatibility
            if "is_harmful" in trace_labels:
                is_harmful = trace_labels.get("is_harmful", False)
            else:
                # AgentDojo uses labels.category = "harmful" | "benign"
                is_harmful = trace_labels.get("category") == "harmful"

            if is_harmful:
                f_harmful.write(line)
                harmful_count += 1
            else:
                f_benign.write(line)
                benign_count += 1

    return harmful_count, benign_count


def split_by_trace_id(input_path: Path, harmful_out: Path, benign_out: Path,
                      trace_labels: Dict[str, bool]) -> tuple:
    """Split file by looking up trace_id in labels mapping."""
    harmful_count = 0
    benign_count = 0
    missing_count = 0

    with open(input_path) as f_in, \
         open(harmful_out, 'w') as f_harmful, \
         open(benign_out, 'w') as f_benign:

        for line in f_in:
            if not line.strip():
                continue

            row = json.loads(line)
            trace_id = row.get("trace_id")

            if trace_id is None:
                missing_count += 1
                # Default to benign if no trace_id
                f_benign.write(line)
                benign_count += 1
                continue

            is_harmful = trace_labels.get(trace_id, False)

            if is_harmful:
                f_harmful.write(line)
                harmful_count += 1
            else:
                f_benign.write(line)
                benign_count += 1

    if missing_count > 0:
        print(f"  WARNING: {missing_count} rows had no trace_id, defaulted to benign")

    return harmful_count, benign_count


def main():
    parser = argparse.ArgumentParser(description="Split AgentDojo data into harmful/benign")
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL (required for labels)")
    parser.add_argument("--renders", type=Path, help="Input renders JSONL")
    parser.add_argument("--lossmasks", type=Path, help="Input lossmasks JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="agentdojo", help="Output filename prefix")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # First, load trace labels (this is required)
    print(f"Loading trace labels from: {args.traces}")
    trace_labels = load_trace_labels(args.traces)
    harmful_ids = sum(1 for v in trace_labels.values() if v)
    benign_ids = sum(1 for v in trace_labels.values() if not v)
    print(f"  Found {len(trace_labels)} traces: {harmful_ids} harmful, {benign_ids} benign")

    # Split traces
    if args.traces and args.traces.exists():
        harmful_out = args.output_dir / f"{args.prefix}_traces_harmful.jsonl"
        benign_out = args.output_dir / f"{args.prefix}_traces_benign.jsonl"
        print(f"\nSplitting traces: {args.traces}")
        harmful, benign = split_traces(args.traces, harmful_out, benign_out)
        print(f"  Harmful: {harmful} -> {harmful_out}")
        print(f"  Benign: {benign} -> {benign_out}")

    # Split renders using trace_id lookup
    if args.renders and args.renders.exists():
        harmful_out = args.output_dir / f"{args.prefix}_renders_harmful.jsonl"
        benign_out = args.output_dir / f"{args.prefix}_renders_benign.jsonl"
        print(f"\nSplitting renders: {args.renders}")
        harmful, benign = split_by_trace_id(args.renders, harmful_out, benign_out, trace_labels)
        print(f"  Harmful: {harmful} -> {harmful_out}")
        print(f"  Benign: {benign} -> {benign_out}")

    # Split lossmasks using trace_id lookup
    if args.lossmasks and args.lossmasks.exists():
        harmful_out = args.output_dir / f"{args.prefix}_lossmasks_harmful.jsonl"
        benign_out = args.output_dir / f"{args.prefix}_lossmasks_benign.jsonl"
        print(f"\nSplitting lossmasks: {args.lossmasks}")
        harmful, benign = split_by_trace_id(args.lossmasks, harmful_out, benign_out, trace_labels)
        print(f"  Harmful: {harmful} -> {harmful_out}")
        print(f"  Benign: {benign} -> {benign_out}")

    print("\nDone! Use these files with --mode mixed:")
    print(f"  --harmful-renders {args.output_dir}/{args.prefix}_renders_harmful.jsonl")
    print(f"  --harmful-lossmasks {args.output_dir}/{args.prefix}_lossmasks_harmful.jsonl")
    print(f"  --benign-renders {args.output_dir}/{args.prefix}_renders_benign.jsonl")
    print(f"  --benign-lossmasks {args.output_dir}/{args.prefix}_lossmasks_benign.jsonl")


if __name__ == "__main__":
    main()
