#!/usr/bin/env python3
"""
Validate trace_v1 JSONL files with completeness/tier checks.

Validates:
1. Schema compliance (jsonschema)
2. Completeness/tier consistency (skeleton should have no assistant, B1=skeleton)
3. Field coverage statistics
4. Message structure integrity

Usage:
    python scripts/validate_schema_tier_b.py --input data/traces/fujitsu_b4.jsonl
    python scripts/validate_schema_tier_b.py --input data/traces/agentdojo.jsonl --strict
    python scripts/validate_schema_tier_b.py --input data/traces/*.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# JSONL Utilities
# =============================================================================

def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Iterate over JSONL file with line numbers."""
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


# =============================================================================
# Schema Validation
# =============================================================================

def validate_schema(
    path: Path,
    schema_path: Path,
    max_errors: int = 20,
) -> Tuple[int, int, List[str]]:
    """
    Validate JSONL against JSON schema.
    
    Returns: (total_records, error_count, error_messages)
    """
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        print("❌ jsonschema not installed. Run: pip install jsonschema")
        return 0, 1, ["jsonschema not installed"]
    
    schema = json.loads(schema_path.read_text())
    validator = Draft7Validator(schema)
    
    total = 0
    error_count = 0
    errors = []
    
    for line_number, record in _iter_jsonl(path):
        total += 1
        for err in validator.iter_errors(record):
            error_count += 1
            path_str = ".".join([str(p) for p in err.path])
            msg = f"Line {line_number}: {path_str} -> {err.message}"
            errors.append(msg)
            if error_count >= max_errors:
                errors.append(f"... (stopped after {max_errors} errors)")
                return total, error_count, errors
    
    return total, error_count, errors


# =============================================================================
# Completeness/Tier Validation
# =============================================================================

def validate_completeness_consistency(
    path: Path,
    max_errors: int = 20,
) -> Tuple[int, List[str]]:
    """
    Validate that completeness and tier fields match message content.
    
    Rules:
    - If has assistant message: should be completeness=complete, tier=B2
    - If no assistant message: should be completeness=skeleton, tier=B1
    
    Returns: (error_count, error_messages)
    """
    errors = []
    
    for line_number, record in _iter_jsonl(path):
        messages = record.get("messages", [])
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        
        completeness = record.get("completeness", "complete")
        tier = record.get("tier", "B2")
        
        expected_completeness = "complete" if has_assistant else "skeleton"
        expected_tier = "B2" if has_assistant else "B1"
        
        if completeness != expected_completeness:
            msg = f"Line {line_number}: completeness={completeness} but has_assistant={has_assistant} (expected {expected_completeness})"
            errors.append(msg)
        
        if tier != expected_tier:
            msg = f"Line {line_number}: tier={tier} but has_assistant={has_assistant} (expected {expected_tier})"
            errors.append(msg)
        
        if len(errors) >= max_errors:
            errors.append(f"... (stopped after {max_errors} errors)")
            break
    
    return len(errors), errors


# =============================================================================
# Message Validation
# =============================================================================

def validate_messages(
    path: Path,
    max_errors: int = 20,
) -> Tuple[int, List[str]]:
    """
    Validate message structure integrity.
    
    Checks:
    - Required fields (role, content)
    - Tool call structure
    - Role sequence validity
    
    Returns: (error_count, error_messages)
    """
    errors = []
    
    for line_number, record in _iter_jsonl(path):
        messages = record.get("messages", [])
        trace_id = record.get("id", "unknown")
        
        if not messages:
            errors.append(f"Line {line_number}: no messages in trace {trace_id}")
            continue
        
        for msg_idx, msg in enumerate(messages):
            # Check required fields
            if "role" not in msg:
                errors.append(f"Line {line_number}, msg[{msg_idx}]: missing 'role'")
            if "content" not in msg:
                errors.append(f"Line {line_number}, msg[{msg_idx}]: missing 'content'")
            
            role = msg.get("role")
            
            # Tool calls only on assistant
            if msg.get("tool_calls") and role != "assistant":
                errors.append(f"Line {line_number}, msg[{msg_idx}]: tool_calls on non-assistant role '{role}'")
            
            # Tool call structure
            if msg.get("tool_calls"):
                for tc_idx, tc in enumerate(msg["tool_calls"]):
                    if not tc.get("function"):
                        errors.append(f"Line {line_number}, msg[{msg_idx}], tc[{tc_idx}]: missing 'function'")
                    elif not tc["function"].get("name"):
                        errors.append(f"Line {line_number}, msg[{msg_idx}], tc[{tc_idx}]: missing function 'name'")
            
            # tool_call_id only on tool role
            if msg.get("tool_call_id") and role != "tool":
                errors.append(f"Line {line_number}, msg[{msg_idx}]: tool_call_id on non-tool role '{role}'")
        
        if len(errors) >= max_errors:
            errors.append(f"... (stopped after {max_errors} errors)")
            break
    
    return len(errors), errors


# =============================================================================
# Field Coverage
# =============================================================================

def compute_field_coverage(path: Path) -> Dict[str, Any]:
    """
    Compute field coverage statistics.
    
    Returns dict with:
    - total: total records
    - fields: {field_name: count_present}
    - completeness: {value: count}
    - tier: {value: count}
    - datasets: {dataset: count}
    - categories: {category: count}
    """
    stats = {
        "total": 0,
        "fields": Counter(),
        "completeness": Counter(),
        "tier": Counter(),
        "datasets": Counter(),
        "categories": Counter(),
        "has_assistant": Counter(),
        "has_tool_calls": Counter(),
        "has_signal_hints": Counter(),
    }
    
    optional_fields = [
        "created_at",
        "completeness",
        "tier",
        "task",
        "labels",
        "tool_attack",
        "training",
        "links",
        "signal_hints",
    ]
    
    for _, record in _iter_jsonl(path):
        stats["total"] += 1
        
        # Count optional fields
        for field in optional_fields:
            if record.get(field) is not None:
                stats["fields"][field] += 1
        
        # Completeness and tier
        stats["completeness"][record.get("completeness", "complete")] += 1
        stats["tier"][record.get("tier", "B2")] += 1
        
        # Dataset
        dataset = record.get("source", {}).get("dataset", "unknown")
        stats["datasets"][dataset] += 1
        
        # Category
        category = record.get("labels", {}).get("category", "unknown") if record.get("labels") else "unknown"
        stats["categories"][category] += 1
        
        # Message content
        messages = record.get("messages", [])
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        has_tool_calls = any(m.get("tool_calls") for m in messages)
        stats["has_assistant"][has_assistant] += 1
        stats["has_tool_calls"][has_tool_calls] += 1
        
        # Signal hints
        stats["has_signal_hints"][record.get("signal_hints") is not None] += 1
    
    return stats


def print_coverage_report(stats: Dict[str, Any], title: str) -> None:
    """Print field coverage report."""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    total = stats["total"]
    print(f"\n  Total records: {total}")
    
    if total == 0:
        return
    
    # Completeness distribution
    print("\n  🔄 Completeness distribution:")
    for value, count in stats["completeness"].most_common():
        pct = count / total * 100
        icon = "⚠️ " if value == "skeleton" else "✅ "
        print(f"    {icon}{value}: {count} ({pct:.1f}%)")
    
    # Tier distribution
    print("\n  📊 Tier distribution:")
    for value, count in stats["tier"].most_common():
        pct = count / total * 100
        icon = "🔸" if value == "B1" else "🔹"
        print(f"    {icon}{value}: {count} ({pct:.1f}%)")
    
    # Dataset distribution
    print("\n  📁 Dataset distribution:")
    for dataset, count in stats["datasets"].most_common():
        pct = count / total * 100
        print(f"    {dataset}: {count} ({pct:.1f}%)")
    
    # Category distribution
    print("\n  🏷️  Category distribution:")
    for cat, count in stats["categories"].most_common():
        pct = count / total * 100
        icon = "🔴" if cat == "harmful" else "🟢" if cat == "benign" else "⚪"
        print(f"    {icon}{cat}: {count} ({pct:.1f}%)")
    
    # Message content
    print("\n  💬 Message content:")
    for has, count in stats["has_assistant"].most_common():
        pct = count / total * 100
        label = "Has assistant" if has else "No assistant (skeleton)"
        icon = "✅" if has else "⚠️ "
        print(f"    {icon}{label}: {count} ({pct:.1f}%)")
    
    for has, count in stats["has_tool_calls"].most_common():
        pct = count / total * 100
        label = "Has tool calls" if has else "No tool calls"
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    # Optional fields
    print("\n  📋 Optional field coverage:")
    optional_fields = [
        "created_at", "completeness", "tier", "task", "labels",
        "tool_attack", "training", "links", "signal_hints"
    ]
    for field in optional_fields:
        count = stats["fields"].get(field, 0)
        pct = count / total * 100
        status = "✅" if pct > 90 else "🟡" if pct > 50 else "⚠️ "
        print(f"    {status} {field}: {count} ({pct:.1f}%)")


# =============================================================================
# Main Validation Pipeline
# =============================================================================

def validate_file(
    path: Path,
    schema_path: Path,
    max_errors: int = 20,
    strict: bool = False,
) -> Tuple[int, int]:
    """
    Run full validation on a trace_v1 JSONL file.
    
    Returns: (total_records, total_errors)
    """
    print("\n" + "=" * 80)
    print(f"🔍 Validating: {path.name}")
    print("=" * 80)
    
    total_errors = 0
    
    # 1. Schema validation
    print("\n  📜 Schema validation...")
    total, error_count, errors = validate_schema(path, schema_path, max_errors)
    total_errors += error_count
    
    if error_count == 0:
        print(f"    ✅ All {total} records pass schema validation")
    else:
        print(f"    ❌ {error_count} schema errors:")
        for err in errors[:10]:
            print(f"       {err}")
        if len(errors) > 10:
            print(f"       ... and {len(errors) - 10} more")
    
    # 2. Completeness consistency
    print("\n  🔄 Completeness/tier consistency...")
    error_count, errors = validate_completeness_consistency(path, max_errors)
    
    if strict:
        total_errors += error_count
    
    if error_count == 0:
        print("    ✅ All completeness/tier fields consistent with message content")
    else:
        status = "❌" if strict else "⚠️ "
        print(f"    {status} {error_count} consistency issues" + (" (strict mode)" if strict else " (warnings)") + ":")
        for err in errors[:5]:
            print(f"       {err}")
        if len(errors) > 5:
            print(f"       ... and {len(errors) - 5} more")
    
    # 3. Message validation
    print("\n  💬 Message structure validation...")
    error_count, errors = validate_messages(path, max_errors)
    total_errors += error_count
    
    if error_count == 0:
        print("    ✅ All message structures valid")
    else:
        print(f"    ❌ {error_count} message errors:")
        for err in errors[:5]:
            print(f"       {err}")
        if len(errors) > 5:
            print(f"       ... and {len(errors) - 5} more")
    
    # 4. Field coverage
    print("\n  📊 Computing field coverage...")
    stats = compute_field_coverage(path)
    print_coverage_report(stats, f"Coverage Report: {path.name}")
    
    return stats["total"], total_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate trace_v1 JSONL files with completeness/tier checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        nargs="+",
        help="Path(s) to trace_v1 JSONL file(s)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "configs" / "schemas" / "trace_v1.json",
        help="Path to trace_v1.json schema",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Stop validation after N errors per check",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat completeness/tier mismatches as errors (not warnings)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print summary",
    )
    
    args = parser.parse_args()
    
    # Check schema exists
    if not args.schema.exists():
        print(f"❌ Schema not found: {args.schema}")
        sys.exit(1)
    
    total_records = 0
    total_errors = 0
    
    for path in args.input:
        if not path.exists():
            print(f"❌ File not found: {path}")
            total_errors += 1
            continue
        
        records, errors = validate_file(
            path,
            args.schema,
            max_errors=args.max_errors,
            strict=args.strict,
        )
        total_records += records
        total_errors += errors
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  Files validated: {len(args.input)}")
    print(f"  Total records: {total_records}")
    print(f"  Total errors: {total_errors}")
    
    if total_errors == 0:
        print("\n  ✅ All validations passed!")
        sys.exit(0)
    else:
        print(f"\n  ❌ {total_errors} errors found")
        sys.exit(1)


if __name__ == "__main__":
    main()
