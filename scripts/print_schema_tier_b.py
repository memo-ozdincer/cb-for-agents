#!/usr/bin/env python3
"""
Pipeline smoke test for Tier A -> Tier B with completeness/tier tracking.

Tests the new B1 (skeleton) vs B2 (complete) tier separation:
- ETL_A marks traces with completeness and tier fields
- B1/skeleton traces have no assistant messages (need generation)
- B2/complete traces have assistant messages (ready for rendering)

Usage:
    python scripts/print_schema_tier_b.py --fujitsu-b4
    python scripts/print_schema_tier_b.py --agentdojo
    python scripts/print_schema_tier_b.py --traces data/traces/my_traces.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.schemas.trace import Trace


# =============================================================================
# JSONL Iteration
# =============================================================================

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL file."""
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _iter_jsonl_with_line(path: Path) -> Iterable[tuple]:
    """Iterate over JSONL file with line numbers."""
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


# =============================================================================
# Pretty Printing
# =============================================================================

def _print_trace(trace: Trace, label: str, verbose: bool = False) -> None:
    """Print a trace with detailed formatting."""
    data = trace.to_dict()
    
    print("=" * 80)
    print(f"🔹 {label}")
    print("-" * 80)
    
    # Core fields
    print(f"  id: {data.get('id')}")
    print(f"  dataset: {data.get('source', {}).get('dataset')}")
    print(f"  subset: {data.get('source', {}).get('subset')}")
    print(f"  split: {data.get('split')}")
    
    # NEW: Completeness and tier
    completeness = data.get('completeness', 'complete')
    tier = data.get('tier', 'B2')
    completeness_icon = "⚠️ " if completeness == "skeleton" else "✅ "
    tier_icon = "🔸" if tier == "B1" else "🔹"
    print(f"  {completeness_icon}completeness: {completeness}")
    print(f"  {tier_icon}tier: {tier}")
    
    # Labels
    labels = data.get('labels', {})
    if labels:
        category = labels.get('category', 'unknown')
        category_icon = "🔴" if category == "harmful" else "🟢"
        print(f"  {category_icon}labels: category={category}, attack_succeeded={labels.get('attack_succeeded')}")
    
    # Messages summary
    messages = data.get("messages", [])
    roles = Counter(m.get("role") for m in messages)
    has_assistant = roles.get("assistant", 0) > 0
    assistant_status = "✅ has assistant" if has_assistant else "⚠️  NO assistant (skeleton)"
    print(f"  messages: {len(messages)} total ({dict(roles)}) {assistant_status}")
    
    if verbose:
        print("  message details:")
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = (msg.get("content") or "").replace("\n", "\\n")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"    [{idx}] {role}: {content_preview}")
            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    print(f"         → tool_call: {func.get('name')}({func.get('arguments', {})})")
    
    # Tool attack info
    if data.get("tool_attack"):
        ta = data["tool_attack"]
        print(f"  tool_attack: expected={ta.get('expected_tool')} → observed={ta.get('observed_tool')}")
    
    # Signal hints
    if data.get("signal_hints"):
        sh = data["signal_hints"]
        print(f"  signal_hints: expected={sh.get('expected_tool_name')}, observed={sh.get('observed_tool_name')}")
        if sh.get("injection_char_span"):
            span = sh["injection_char_span"]
            print(f"    injection_span: msg[{span.get('message_index')}] chars {span.get('char_start')}-{span.get('char_end')}")
    
    # Training config
    if data.get("training"):
        tr = data["training"]
        print(f"  training: policy={tr.get('loss_mask_policy')}, weight={tr.get('sample_weight')}")
        if tr.get("mixture"):
            print(f"    mixture: {tr['mixture'].get('class_id')}")


def _print_summary(traces: List[Trace], title: str) -> None:
    """Print summary statistics for a list of traces."""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    if not traces:
        print("  No traces to summarize.")
        return
    
    # Count by dataset
    dataset_counts = Counter()
    completeness_counts = Counter()
    tier_counts = Counter()
    category_counts = Counter()
    has_assistant_counts = Counter()
    
    for trace in traces:
        data = trace.to_dict()
        dataset = data.get("source", {}).get("dataset", "unknown")
        completeness = data.get("completeness", "complete")
        tier = data.get("tier", "B2")
        category = data.get("labels", {}).get("category", "unknown")
        
        dataset_counts[dataset] += 1
        completeness_counts[completeness] += 1
        tier_counts[tier] += 1
        category_counts[category] += 1
        
        has_assistant = any(m.get("role") == "assistant" for m in data.get("messages", []))
        has_assistant_counts[has_assistant] += 1
    
    print(f"\n  Total traces: {len(traces)}")
    
    print("\n  📁 By dataset:")
    for name, count in dataset_counts.most_common():
        print(f"    {name}: {count}")
    
    print("\n  🔄 By completeness:")
    for name, count in completeness_counts.most_common():
        icon = "⚠️ " if name == "skeleton" else "✅ "
        print(f"    {icon}{name}: {count}")
    
    print("\n  📊 By tier:")
    for name, count in tier_counts.most_common():
        icon = "🔸" if name == "B1" else "🔹"
        print(f"    {icon}{name}: {count}")
    
    print("\n  🏷️  By category:")
    for name, count in category_counts.most_common():
        icon = "🔴" if name == "harmful" else "🟢" if name == "benign" else "⚪"
        print(f"    {icon}{name}: {count}")
    
    print("\n  💬 Has assistant messages:")
    for has, count in has_assistant_counts.most_common():
        icon = "✅" if has else "⚠️ "
        label = "Yes" if has else "No (skeleton)"
        print(f"    {icon}{label}: {count}")
    
    # Consistency check
    print("\n  🔍 Consistency check:")
    mismatches = 0
    for trace in traces:
        data = trace.to_dict()
        has_assistant = any(m.get("role") == "assistant" for m in data.get("messages", []))
        completeness = data.get("completeness", "complete")
        tier = data.get("tier", "B2")
        
        expected_completeness = "complete" if has_assistant else "skeleton"
        expected_tier = "B2" if has_assistant else "B1"
        
        if completeness != expected_completeness or tier != expected_tier:
            mismatches += 1
    
    if mismatches == 0:
        print("    ✅ All traces have consistent completeness/tier vs message content")
    else:
        print(f"    ⚠️  {mismatches} traces have mismatched completeness/tier fields!")


# =============================================================================
# ETL_A Testing
# =============================================================================

def test_etl_a_fujitsu_b4(limit: int = 5, verbose: bool = False) -> List[Trace]:
    """Test ETL_A conversion of Fujitsu B4 data."""
    from src.schemas.tools.ETL_A import convert_fujitsu_b4_record
    
    print("\n" + "=" * 80)
    print("🧪 Testing ETL_A: Fujitsu B4 (Orchestrator Tool-Flip)")
    print("   Expected: skeleton/B1 traces (no assistant messages)")
    print("=" * 80)
    
    b4_path = REPO_ROOT / "data" / "fujitsu" / "orchestrator_attacks_combined_deduplicated.jsonl"
    if not b4_path.exists():
        print(f"  ❌ File not found: {b4_path}")
        return []
    
    traces = []
    count = 0
    for line_number, record in _iter_jsonl_with_line(b4_path):
        if count >= limit:
            break
        trace = convert_fujitsu_b4_record(record, "train", line_number)
        if trace:
            traces.append(trace)
            _print_trace(trace, f"Fujitsu B4 sample {count+1}/{limit}", verbose=verbose)
            count += 1
    
    if not traces:
        print("  ❌ No traces converted")
    
    return traces


def test_etl_a_agentdojo(limit: int = 5, verbose: bool = False) -> List[Trace]:
    """Test ETL_A conversion of AgentDojo data."""
    from src.schemas.tools.ETL_A import convert_agentdojo_record
    
    print("\n" + "=" * 80)
    print("🧪 Testing ETL_A: AgentDojo")
    print("   Expected: complete/B2 traces (has assistant messages)")
    print("=" * 80)
    
    agentdojo_dir = REPO_ROOT / "data" / "agent_dojo"
    if not agentdojo_dir.exists():
        print(f"  ❌ Directory not found: {agentdojo_dir}")
        return []
    
    agentdojo_files = sorted(agentdojo_dir.glob("agentdojo-*.jsonl"))
    if not agentdojo_files:
        print("  ❌ No AgentDojo files found")
        return []
    
    traces = []
    count = 0
    
    # Get samples from first file
    path = agentdojo_files[0]
    print(f"  Using: {path.name}")
    
    for line_number, record in _iter_jsonl_with_line(path):
        if count >= limit:
            break
        if not record.get("messages"):
            continue
        trace = convert_agentdojo_record(record, "train", line_number)
        if trace:
            traces.append(trace)
            _print_trace(trace, f"AgentDojo sample {count+1}/{limit}", verbose=verbose)
            count += 1
    
    if not traces:
        print("  ❌ No traces converted")
    
    return traces


def test_existing_traces(trace_path: Path, limit: int = 5, verbose: bool = False) -> List[Trace]:
    """Load and test existing trace_v1 JSONL file."""
    print("\n" + "=" * 80)
    print(f"🧪 Loading existing traces: {trace_path.name}")
    print("=" * 80)
    
    if not trace_path.exists():
        print(f"  ❌ File not found: {trace_path}")
        return []
    
    traces = []
    count = 0
    
    for record in _iter_jsonl(trace_path):
        if count >= limit:
            break
        try:
            trace = Trace.from_dict(record)
            traces.append(trace)
            _print_trace(trace, f"Trace {count+1}/{limit}", verbose=verbose)
            count += 1
        except Exception as e:
            print(f"  ⚠️  Failed to parse trace: {e}")
    
    return traces


def load_all_traces(trace_path: Path, limit: Optional[int] = None) -> List[Trace]:
    """Load all traces from a file for summary."""
    if not trace_path.exists():
        return []
    
    traces = []
    for record in _iter_jsonl(trace_path):
        if limit and len(traces) >= limit:
            break
        try:
            trace = Trace.from_dict(record)
            traces.append(trace)
        except Exception:
            pass
    
    return traces


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Tier A -> Tier B pipeline with completeness/tier tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fujitsu-b4", action="store_true", help="Test Fujitsu B4 conversion")
    parser.add_argument("--agentdojo", action="store_true", help="Test AgentDojo conversion")
    parser.add_argument("--traces", type=Path, default=None, help="Path to existing trace_v1 JSONL")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to print")
    parser.add_argument("--summary-limit", type=int, default=1000, help="Max traces for summary stats")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full message content")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Default: run all if nothing specified
    run_all = args.all or (not args.fujitsu_b4 and not args.agentdojo and not args.traces)
    
    all_traces = []
    
    if run_all or args.fujitsu_b4:
        traces = test_etl_a_fujitsu_b4(limit=args.limit, verbose=args.verbose)
        all_traces.extend(traces)
    
    if run_all or args.agentdojo:
        traces = test_etl_a_agentdojo(limit=args.limit, verbose=args.verbose)
        all_traces.extend(traces)
    
    if args.traces:
        traces = test_existing_traces(args.traces, limit=args.limit, verbose=args.verbose)
        # For summary, load more traces
        all_traces = load_all_traces(args.traces, limit=args.summary_limit)
    
    # Print summary
    if all_traces:
        _print_summary(all_traces, "Pipeline Summary")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline smoke test complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
