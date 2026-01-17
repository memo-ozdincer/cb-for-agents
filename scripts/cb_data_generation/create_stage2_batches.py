#!/usr/bin/env python3
"""
Create CB training batches from Stage 2 flat samples.

Input: Stage 2 flat JSONL (one sample per line) with labels.split
Output: cb_training_batches.jsonl with:
  - harmful: list of harmful samples
  - benign: list of benign/retain samples

Each batch contains 1 harmful + N benign samples (default N=5).
Samples are converted to agentic format with an assistant message appended
so the trainer can apply chat templates correctly.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def convert_sample_to_agentic(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure sample has messages + assistant completion and mark is_agentic.
    """
    messages = list(sample.get("messages", []))
    assistant_raw = sample.get("assistant_raw", "")

    if assistant_raw:
        # Append assistant completion as final message
        messages = messages + [{"role": "assistant", "content": assistant_raw}]

    converted = dict(sample)
    converted["messages"] = messages
    converted["is_agentic"] = True
    return converted


def build_batches(
    harmful: List[Dict[str, Any]],
    benign: List[Dict[str, Any]],
    benign_per_harmful: int,
    seed: int,
    max_harmful: int | None,
    require_tool_calls_harmful: bool,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    random.shuffle(harmful)
    random.shuffle(benign)

    if require_tool_calls_harmful:
        harmful = [
            s for s in harmful
            if "<|python_tag|>" in (s.get("assistant_raw", "") or "")
        ]

    if max_harmful is not None:
        harmful = harmful[:max_harmful]

    if not harmful:
        raise ValueError("No harmful samples provided.")
    if not benign:
        raise ValueError("No benign/retain samples provided.")

    batches = []
    benign_idx = 0

    for i, h in enumerate(harmful):
        batch_benign = []
        for _ in range(benign_per_harmful):
            batch_benign.append(benign[benign_idx % len(benign)])
            benign_idx += 1

        batch = {
            "batch_id": i,
            "batch_size": 1 + benign_per_harmful,
            "harmful": [convert_sample_to_agentic(h)],
            "benign": [convert_sample_to_agentic(b) for b in batch_benign],
        }
        batches.append(batch)

    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 2 CB training batches")
    parser.add_argument("--input", type=Path, required=True, help="Stage 2 flat JSONL input")
    parser.add_argument("--output", type=Path, required=True, help="Output cb_training_batches.jsonl")
    parser.add_argument("--benign-per-harmful", type=int, default=5, help="Dr samples per Ds sample")
    parser.add_argument("--max-harmful", type=int, default=None, help="Limit number of harmful samples")
    parser.add_argument(
        "--require-tool-calls-harmful",
        action="store_true",
        help="Only keep harmful samples with <|python_tag|> in assistant_raw",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    samples = read_jsonl(args.input)

    harmful = [s for s in samples if s.get("labels", {}).get("split") == "harmful"]
    benign = [s for s in samples if s.get("labels", {}).get("split") == "retain"]

    batches = build_batches(
        harmful,
        benign,
        args.benign_per_harmful,
        args.seed,
        args.max_harmful,
        args.require_tool_calls_harmful,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for batch in batches:
            f.write(json.dumps(batch, ensure_ascii=False) + "\n")

    print(f"Created {len(batches)} batches")
    print(f"  Harmful samples: {len(harmful)}")
    print(f"  Benign samples:  {len(benign)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
