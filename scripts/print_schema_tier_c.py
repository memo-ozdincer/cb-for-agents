#!/usr/bin/env python3
"""
Quick local pipeline smoke test for Tier B -> Tier C.
Prints sample render_v1 and lossmask_v1 records.
"""

import json
import sys
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _print_render(render: Dict[str, object], label: str) -> None:
    print("=" * 80)
    print(label)
    print(f"render_id: {render.get('render_id')}")
    print(f"trace_id: {render.get('trace_id')}")
    print(f"tokenizer_id: {render.get('tokenizer_id')}")
    print(f"sequence_length: {render.get('sequence_length')}")
    if render.get("render_options"):
        print(f"render_options: {render.get('render_options')}")
    if render.get("alignment"):
        alignment = render.get("alignment") or {}
        msg_spans = alignment.get("message_spans") or []
        asst_spans = alignment.get("assistant_spans") or []
        tc_spans = alignment.get("tool_call_spans") or []
        print(f"alignment: messages={len(msg_spans)}, assistant={len(asst_spans)}, tool_calls={len(tc_spans)}")
    if render.get("signals"):
        signals = render.get("signals") or {}
        inj = signals.get("injection_spans") or []
        commits = signals.get("action_commitments") or []
        print(f"signals: injection_spans={len(inj)}, action_commitments={len(commits)}")


def _print_lossmask(lossmask: Dict[str, object], label: str) -> None:
    print("=" * 80)
    print(label)
    print(f"lossmask_id: {lossmask.get('lossmask_id')}")
    print(f"render_id: {lossmask.get('render_id')}")
    print(f"trace_id: {lossmask.get('trace_id')}")
    print(f"policy_id: {lossmask.get('policy_id')}")
    loss_mask = lossmask.get("loss_mask") or []
    labels = lossmask.get("labels") or []
    print(f"mask_len: {len(loss_mask)}")
    print(f"labels_len: {len(labels)}")
    if lossmask.get("stats"):
        print(f"stats: {lossmask.get('stats')}")
    if lossmask.get("signal_derived"):
        print(f"signal_derived: {lossmask.get('signal_derived')}")


def _load_many(path: Path, limit: int) -> Iterable[Dict[str, object]]:
    count = 0
    for record in _iter_jsonl(path):
        yield record
        count += 1
        if count >= limit:
            break


def _load_trace_index(path: Path) -> Dict[str, Dict[str, object]]:
    index = {}
    if not path.exists():
        return index
    for record in _iter_jsonl(path):
        trace_id = record.get("id")
        if isinstance(trace_id, str):
            index[trace_id] = record
    return index


def _summarize(
    renders: Iterable[Dict[str, object]],
    masks: Iterable[Dict[str, object]],
    trace_index: Dict[str, Dict[str, object]],
) -> None:
    dataset_counts = Counter()
    policy_counts = Counter()
    mixture_counts = Counter()
    mwcs_weight_stats = defaultdict(list)
    mask_ratios = defaultdict(list)

    for mask in masks:
        trace_id = mask.get("trace_id")
        trace = trace_index.get(trace_id, {}) if isinstance(trace_id, str) else {}
        dataset = (trace.get("source") or {}).get("dataset", "unknown")
        mixture = ((trace.get("training") or {}).get("mixture") or {}).get("class_id", "none")
        policy = mask.get("policy_id", "unknown")
        stats = mask.get("stats") or {}
        mask_ratio = stats.get("mask_ratio")
        sample_weight = mask.get("sample_weight")

        dataset_counts[dataset] += 1
        policy_counts[policy] += 1
        mixture_counts[mixture] += 1
        if isinstance(mask_ratio, (int, float)):
            mask_ratios[dataset].append(mask_ratio)
        if isinstance(sample_weight, (int, float)):
            mwcs_weight_stats[mixture].append(sample_weight)

    print("=" * 80)
    print("Tier C summary")
    print("datasets:")
    for name, count in dataset_counts.most_common():
        ratios = mask_ratios.get(name, [])
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        ratio_str = f"avg_mask_ratio={avg_ratio:.3f}" if avg_ratio is not None else "avg_mask_ratio=n/a"
        print(f"  {name}: {count} ({ratio_str})")

    print("policies:")
    for name, count in policy_counts.most_common():
        print(f"  {name}: {count}")

    print("mixture classes:")
    for name, count in mixture_counts.most_common():
        weights = mwcs_weight_stats.get(name, [])
        avg_weight = sum(weights) / len(weights) if weights else None
        weight_str = f"avg_weight={avg_weight:.3f}" if avg_weight is not None else "avg_weight=n/a"
        print(f"  {name}: {count} ({weight_str})")


def _pick_examples_by_dataset(
    renders: Iterable[Dict[str, object]],
    masks: Iterable[Dict[str, object]],
    trace_index: Dict[str, Dict[str, object]],
    limit_per_dataset: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    render_by_dataset: Dict[str, Dict[str, object]] = {}
    mask_by_dataset: Dict[str, Dict[str, object]] = {}
    render_counts = Counter()
    mask_counts = Counter()

    for render in renders:
        trace_id = render.get("trace_id")
        trace = trace_index.get(trace_id, {}) if isinstance(trace_id, str) else {}
        dataset = (trace.get("source") or {}).get("dataset", "unknown")
        if render_counts[dataset] < limit_per_dataset:
            render_by_dataset[f"{dataset}:{render_counts[dataset]}"] = render
            render_counts[dataset] += 1

    for mask in masks:
        trace_id = mask.get("trace_id")
        trace = trace_index.get(trace_id, {}) if isinstance(trace_id, str) else {}
        dataset = (trace.get("source") or {}).get("dataset", "unknown")
        if mask_counts[dataset] < limit_per_dataset:
            mask_by_dataset[f"{dataset}:{mask_counts[dataset]}"] = mask
            mask_counts[dataset] += 1

    return render_by_dataset, mask_by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Print sample render_v1/lossmask_v1 records")
    parser.add_argument("--render", type=Path, default=None, help="Path to render_v1 JSONL")
    parser.add_argument("--lossmask", type=Path, default=None, help="Path to lossmask_v1 JSONL")
    parser.add_argument("--traces", type=Path, default=None, help="Path to trace_v1 JSONL")
    parser.add_argument("--count", type=int, default=3, help="Number of examples to print")
    parser.add_argument("--per-dataset", type=int, default=1, help="Examples per dataset for summaries")
    args = parser.parse_args()

    render_path = REPO_ROOT / "data" / "renders" / "render_v1.jsonl"
    lossmask_path = REPO_ROOT / "data" / "lossmasks" / "lossmask_v1.jsonl"

    if args.render:
        render_path = args.render
    if args.lossmask:
        lossmask_path = args.lossmask
    trace_path = args.traces or (REPO_ROOT / "data" / "traces" / "trace_v1.jsonl")

    renders = list(_load_many(render_path, args.count)) if render_path.exists() else []
    masks = list(_load_many(lossmask_path, args.count)) if lossmask_path.exists() else []
    trace_index = _load_trace_index(trace_path)

    if render_path.exists():
        samples = renders
        if samples:
            for idx, sample in enumerate(samples, start=1):
                _print_render(sample, f"Sample render_v1 {idx}/{len(samples)} from {render_path.name}")
        else:
            print(f"No records found in {render_path}")
    else:
        print(f"Missing: {render_path}")

    if lossmask_path.exists():
        samples = masks
        if samples:
            for idx, sample in enumerate(samples, start=1):
                _print_lossmask(sample, f"Sample lossmask_v1 {idx}/{len(samples)} from {lossmask_path.name}")
        else:
            print(f"No records found in {lossmask_path}")

    if trace_index and (render_path.exists() or lossmask_path.exists()):
        full_renders = list(_iter_jsonl(render_path)) if render_path.exists() else []
        full_masks = list(_iter_jsonl(lossmask_path)) if lossmask_path.exists() else []
        _summarize(full_renders, full_masks, trace_index)

        render_by_dataset, mask_by_dataset = _pick_examples_by_dataset(
            full_renders,
            full_masks,
            trace_index,
            args.per_dataset,
        )

        for key, render in render_by_dataset.items():
            trace_id = render.get("trace_id")
            trace = trace_index.get(trace_id, {}) if isinstance(trace_id, str) else {}
            dataset = (trace.get("source") or {}).get("dataset", "unknown")
            label = f"Dataset sample render ({dataset})"
            _print_render(render, label)

        for key, mask in mask_by_dataset.items():
            trace_id = mask.get("trace_id")
            trace = trace_index.get(trace_id, {}) if isinstance(trace_id, str) else {}
            dataset = (trace.get("source") or {}).get("dataset", "unknown")
            policy = mask.get("policy_id")
            weight = mask.get("sample_weight")
            label = f"Dataset sample lossmask ({dataset}) policy={policy} weight={weight}"
            _print_lossmask(mask, label)
    else:
        print(f"Missing: {lossmask_path}")


if __name__ == "__main__":
    main()
