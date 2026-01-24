#!/usr/bin/env python3
"""
Validate render_v1 and lossmask_v1 JSONL files against schemas.
Includes consistency checks between render and lossmask when both are provided.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from jsonschema import Draft7Validator


def _iter_jsonl(path: Path):
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


def _validate_schema(path: Path, schema_path: Path, max_errors: int) -> int:
    schema = json.loads(schema_path.read_text())
    validator = Draft7Validator(schema)
    error_count = 0

    for line_number, record in _iter_jsonl(path):
        for err in validator.iter_errors(record):
            error_count += 1
            path_str = ".".join([str(p) for p in err.path])
            print(f"{path.name} line {line_number}: {path_str} -> {err.message}")
            if error_count >= max_errors:
                print("Max errors reached, stopping.")
                return error_count

    return error_count


def _validate_render_consistency(record: Dict[str, object]) -> List[str]:
    errors = []
    input_ids = record.get("input_ids") or []
    attention_mask = record.get("attention_mask") or []
    seq_len = record.get("sequence_length")

    if seq_len is not None and seq_len != len(input_ids):
        errors.append("sequence_length does not match input_ids length")

    if attention_mask and len(attention_mask) != len(input_ids):
        errors.append("attention_mask length does not match input_ids length")

    return errors


def _validate_lossmask_consistency(record: Dict[str, object]) -> List[str]:
    errors = []
    loss_mask = record.get("loss_mask") or []
    labels = record.get("labels")

    if labels is not None and len(labels) != len(loss_mask):
        errors.append("labels length does not match loss_mask length")

    return errors


def _build_render_index(render_path: Path) -> Dict[str, Dict[str, object]]:
    index = {}
    for _, record in _iter_jsonl(render_path):
        rid = record.get("render_id")
        if isinstance(rid, str):
            index[rid] = record
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate render_v1 and lossmask_v1 JSONL files")
    parser.add_argument("--render", type=Path, default=None, help="Path to render_v1 JSONL")
    parser.add_argument("--lossmask", type=Path, default=None, help="Path to lossmask_v1 JSONL")
    parser.add_argument(
        "--render-schema",
        type=Path,
        default=Path("configs/schemas/render_v1.json"),
        help="Path to render_v1.json schema",
    )
    parser.add_argument(
        "--lossmask-schema",
        type=Path,
        default=Path("configs/schemas/lossmask_v1.json"),
        help="Path to lossmask_v1.json schema",
    )
    parser.add_argument("--max-errors", type=int, default=20, help="Stop after N errors")
    args = parser.parse_args()

    error_count = 0

    if args.render and args.render.exists():
        error_count += _validate_schema(args.render, args.render_schema, args.max_errors)

        for line_number, record in _iter_jsonl(args.render):
            for err in _validate_render_consistency(record):
                error_count += 1
                print(f"{args.render.name} line {line_number}: {err}")
                if error_count >= args.max_errors:
                    print("Max errors reached, stopping.")
                    return

    if args.lossmask and args.lossmask.exists():
        error_count += _validate_schema(args.lossmask, args.lossmask_schema, args.max_errors)

        for line_number, record in _iter_jsonl(args.lossmask):
            for err in _validate_lossmask_consistency(record):
                error_count += 1
                print(f"{args.lossmask.name} line {line_number}: {err}")
                if error_count >= args.max_errors:
                    print("Max errors reached, stopping.")
                    return

    if args.render and args.lossmask and args.render.exists() and args.lossmask.exists():
        render_index = _build_render_index(args.render)
        for line_number, record in _iter_jsonl(args.lossmask):
            rid = record.get("render_id")
            if rid not in render_index:
                error_count += 1
                print(f"{args.lossmask.name} line {line_number}: render_id not found in render file")
            else:
                render_record = render_index[rid]
                if render_record.get("trace_id") != record.get("trace_id"):
                    error_count += 1
                    print(f"{args.lossmask.name} line {line_number}: trace_id mismatch for render_id {rid}")
            if error_count >= args.max_errors:
                print("Max errors reached, stopping.")
                return

    print("=" * 80)
    print(f"errors: {error_count}")


if __name__ == "__main__":
    main()
