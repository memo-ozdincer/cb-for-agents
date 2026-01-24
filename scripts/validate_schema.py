#!/usr/bin/env python3
"""
Validate trace_v1 JSONL files against configs/schemas/trace_v1.json.
Also prints a small summary of missing optional fields.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from jsonschema import Draft7Validator


def _iter_jsonl(path: Path):
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


def _count_missing_fields(record: Dict[str, object], optional_fields: List[str]) -> Counter:
    counter = Counter()
    for field in optional_fields:
        if record.get(field) is None:
            counter[field] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trace_v1 JSONL files")
    parser.add_argument("--input", type=Path, required=True, help="Path to trace_v1 JSONL")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/schemas/trace_v1.json"),
        help="Path to trace_v1.json schema",
    )
    parser.add_argument("--max-errors", type=int, default=20, help="Stop after N errors")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    validator = Draft7Validator(schema)

    total = 0
    error_count = 0
    missing_counter = Counter()

    optional_fields = [
        "created_at",
        "task",
        "labels",
        "tool_attack",
        "training",
        "links",
        "signal_hints",
    ]

    for line_number, record in _iter_jsonl(args.input):
        total += 1
        for err in validator.iter_errors(record):
            error_count += 1
            path = ".".join([str(p) for p in err.path])
            print(f"Line {line_number}: {path} -> {err.message}")
            if error_count >= args.max_errors:
                print("Max errors reached, stopping.")
                break
        if error_count >= args.max_errors:
            break

        missing_counter.update(_count_missing_fields(record, optional_fields))

    print("=" * 80)
    print(f"validated: {total} records")
    print(f"errors: {error_count}")
    if total > 0:
        print("missing optional fields (count):")
        for key in optional_fields:
            print(f"  {key}: {missing_counter.get(key, 0)}")


if __name__ == "__main__":
    main()
