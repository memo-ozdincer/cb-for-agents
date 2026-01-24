#!/usr/bin/env python3
"""
Quick local pipeline smoke test for Tier A -> Tier B.
Focus on Fujitsu B4 and AgentDojo datasets.

Note: Fujitsu B1, B2, B3 have been removed from the pipeline.
Use print_schema_tier_b.py for the new completeness/tier workflow.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.schemas.tools.ETL_A import (
    convert_fujitsu_b4_record,
    convert_agentdojo_record,
    FUJITSU_FILE_MAP,
)


def _iter_jsonl(path: Path):
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


def _print_trace(trace, label: str) -> None:
    data = trace.to_dict()
    print("=" * 80)
    print(label)
    print(f"id: {data.get('id')}")
    print(f"dataset: {data.get('source', {}).get('dataset')}")
    print(f"subset: {data.get('source', {}).get('subset')}")
    print(f"split: {data.get('split')}")
    print(f"completeness: {data.get('completeness', 'complete')}")
    print(f"tier: {data.get('tier', 'B2')}")
    print(f"labels: {data.get('labels')}")
    print("messages:")
    for idx, msg in enumerate(data.get("messages", [])):
        role = msg.get("role")
        content = (msg.get("content") or "").replace("\n", "\\n")
        print(f"  [{idx}] {role}: {content[:160]}")
        if msg.get("tool_calls"):
            print(f"    tool_calls: {msg.get('tool_calls')}")
    if data.get("tool_attack"):
        print(f"tool_attack: {data.get('tool_attack')}")
    if data.get("signal_hints"):
        print(f"signal_hints: {data.get('signal_hints')}")


def _load_one(path: Path, converter, predicate=None, label: str = "") -> bool:
    for line_number, record in _iter_jsonl(path):
        if predicate and not predicate(record):
            continue
        trace = converter(record, "train", line_number)
        if trace:
            tag = label or f"Sample from {path.name} (line {line_number})"
            _print_trace(trace, tag)
            return True
    return False


def main() -> None:
    fujitsu_dir = REPO_ROOT / "data" / "fujitsu"
    agentdojo_dir = REPO_ROOT / "data" / "agent_dojo"

    # Only B4 is supported now
    print("\nFujitsu B4 sample (skeleton trace - no assistant)")
    b4_path = fujitsu_dir / FUJITSU_FILE_MAP["b4"]
    if b4_path.exists():
        _load_one(b4_path, convert_fujitsu_b4_record)
    else:
        print(f"Missing: {b4_path}")

    print("\nAgentDojo samples (by file + attack/benign)")
    agentdojo_files = sorted(agentdojo_dir.glob("agentdojo-*.jsonl"))
    if not agentdojo_files:
        print("No AgentDojo files found")
    else:
        for path in agentdojo_files[:3]:  # Limit to first 3 files
            print(f"\n-- {path.name}")
            header_skipped = _load_one(
                path,
                convert_agentdojo_record,
                predicate=lambda r: bool(r.get("messages")),
                label=f"{path.name} (first valid record)",
            )
            if not header_skipped:
                print("No valid records found")
                continue

            found_attack = _load_one(
                path,
                convert_agentdojo_record,
                predicate=lambda r: r.get("metadata", {}).get("injection_task_id") is not None,
                label=f"{path.name} (attack sample)",
            )
            if not found_attack:
                print("No attack samples found")

            found_benign = _load_one(
                path,
                convert_agentdojo_record,
                predicate=lambda r: r.get("metadata", {}).get("injection_task_id") is None and bool(r.get("messages")),
                label=f"{path.name} (benign sample)",
            )
            if not found_benign:
                print("No benign samples found")


if __name__ == "__main__":
    main()
