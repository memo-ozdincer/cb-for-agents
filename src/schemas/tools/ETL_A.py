#!/usr/bin/env python3
"""
ETL_A: Tier A (raw formats) -> Tier B (trace_v1) conversion.

This module only handles raw source files and converts them to canonical traces.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.schemas.trace import (
    Trace,
    Message,
    TraceSource,
    TraceTask,
    TraceLabels,
    TraceTraining,
    TraceToolAttack,
    TraceLinks,
    TraceMixture,
    SignalHints,
    InjectionCharSpan,
    ToolCall,
    ToolCallFunction,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Only keeping B4 - removed B1, B2, B3
FUJITSU_FILE_MAP = {
    "b4": "orchestrator_attacks_combined_deduplicated.jsonl",
}


# =============================================================================
# Helpers
# =============================================================================

def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


def _record_locator(record: Dict[str, Any], line_number: int, keys: List[str]) -> Dict[str, str]:
    for key in keys:
        value = record.get(key)
        if value:
            if key == "index":
                kind = "index"
            else:
                kind = "uuid" if "id" in key else "path"
            return {"kind": kind, "value": str(value)}
    return {"kind": "jsonl_line", "value": str(line_number)}


def _find_injection_span(content: str, injection_text: Optional[str], message_index: int) -> Optional[InjectionCharSpan]:
    if not injection_text:
        return None
    idx = content.find(injection_text)
    if idx >= 0:
        return InjectionCharSpan(
            message_index=message_index,
            char_start=idx,
            char_end=idx + len(injection_text),
        )
    return None


def _has_assistant_message(messages: List[Message]) -> bool:
    """Check if trace has any assistant messages (used to determine completeness)."""
    return any(msg.role == "assistant" for msg in messages)


def _determine_completeness(messages: List[Message]) -> str:
    """Determine completeness based on assistant message presence."""
    return "complete" if _has_assistant_message(messages) else "skeleton"


def _determine_tier(messages: List[Message]) -> str:
    """Determine tier based on completeness: B1 = skeleton, B2 = complete."""
    return "B2" if _has_assistant_message(messages) else "B1"


def _build_user_with_context(
    user_query: str,
    context_label: str,
    context_text: Optional[str],
    message_index: int,
) -> Tuple[str, Optional[InjectionCharSpan]]:
    if not context_text:
        return user_query, None

    base = user_query.strip() if user_query else ""
    if base:
        base += "\n\n"

    label_open = f"[{context_label}]\n"
    label_close = f"\n[END_{context_label}]"
    start = len(base) + len(label_open)
    content = f"{base}{label_open}{context_text}{label_close}"

    span = InjectionCharSpan(
        message_index=message_index,
        char_start=start,
        char_end=start + len(context_text),
    )
    return content, span


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if any(token in lowered for token in ["success", "unsafe", "attack succeeded"]):
            return True
        if any(token in lowered for token in ["safe", "benign", "failed", "unsuccess"]):
            return False
    return default


# =============================================================================
# Fujitsu B4 (Orchestrator Tool-Flip)
# =============================================================================

def convert_fujitsu_b4_record(record: Dict[str, Any], split: str, line_number: int) -> Optional[Trace]:
    """
    Convert Fujitsu B4 (orchestrator tool-flip) record to trace_v1.
    
    B4 records are SKELETON traces (no assistant response) that need
    generation via generate_completions.py before they can be used for training.
    """
    try:
        user_content = record.get("combined_query") or record.get("benign_query") or ""
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant with access to internal documents and web search.",
            ),
            Message(
                role="user",
                content=user_content,
            ),
        ]

        success = _coerce_bool(record.get("success"), default=True)
        injection_span = _find_injection_span(user_content, record.get("malicious_injection"), 1)

        signal_hints = SignalHints(
            injection_char_span=injection_span,
            expected_tool_name=record.get("expected_tool"),
            observed_tool_name=record.get("simulated_tool"),
        )
        
        # Determine completeness based on messages
        completeness = _determine_completeness(messages)
        tier = _determine_tier(messages)

        trace = Trace(
            id=Trace.generate_id("fujitsu_b4", messages=messages),
            completeness=completeness,
            tier=tier,
            source=TraceSource(
                dataset="fujitsu_b4",
                tier="raw",
                subset="orchestrator",
                record_locator=_record_locator(record, line_number, ["record_id", "attack_id"]),
                ingest_version="etl_a_v1",
            ),
            messages=messages,
            split=split,
            task=TraceTask(
                family="tool_flip",
                name="orchestrator",
                variant=record.get("subtype"),
            ),
            labels=TraceLabels(
                category="harmful" if success else "benign",
                security_outcome="unsafe" if success else "safe",
                attack_type="tool_flip",
                attack_succeeded=success,
            ),
            tool_attack=TraceToolAttack(
                expected_tool=record.get("expected_tool"),
                observed_tool=record.get("simulated_tool"),
                attack_vector=record.get("subtype"),
                injection_text=record.get("malicious_injection"),
            ),
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(class_id="fujitsu_b4/tool_flip"),
            ),
            links=TraceLinks(raw_id=record.get("record_id")),
            signal_hints=signal_hints,
        )
        return trace
    except Exception as exc:
        logger.warning("Failed to convert Fujitsu B4 record: %s", exc)
        return None


# =============================================================================
# AgentDojo
# =============================================================================

def _parse_tool_calls(msg: Dict[str, Any]) -> Optional[List[ToolCall]]:
    tool_calls = []
    raw_calls = msg.get("tool_calls")
    if raw_calls is None and msg.get("tool_call"):
        raw_calls = [msg.get("tool_call")]

    if not raw_calls:
        return None

    for call in raw_calls:
        if not call:
            continue
        func = call.get("function") or {}
        if isinstance(func, str):
            func = {"name": func}
        name = func.get("name") or call.get("name") or ""
        args = func.get("arguments") or call.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        tool_calls.append(
            ToolCall(
                call_id=call.get("id"),
                function=ToolCallFunction(name=name, arguments=args),
            )
        )

    return tool_calls or None


def convert_agentdojo_record(record: Dict[str, Any], split: str, line_number: int) -> Optional[Trace]:
    try:
        metadata = record.get("metadata", {})
        raw_messages = record.get("messages", [])

        if not raw_messages:
            return None

        messages: List[Message] = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            tool_calls = _parse_tool_calls(msg)
            name = msg.get("name")
            if not name and role == "tool":
                tool_call = msg.get("tool_call") or {}
                if isinstance(tool_call, dict):
                    func = tool_call.get("function") or {}
                    if isinstance(func, str):
                        name = func
                    elif isinstance(func, dict):
                        name = func.get("name")
            if role == "tool":
                tool_calls = None
            messages.append(
                Message(
                    role=role,
                    content=msg.get("content") or "",
                    name=name,
                    tool_calls=tool_calls,
                    tool_call_id=msg.get("tool_call_id"),
                    thinking=msg.get("thinking"),
                )
            )

        is_attack = metadata.get("injection_task_id") is not None
        security_passed = bool(metadata.get("security", True))
        category = "harmful" if is_attack else "benign"
        
        # Determine completeness based on messages
        completeness = _determine_completeness(messages)
        tier = _determine_tier(messages)

        trace = Trace(
            id=Trace.generate_id("agentdojo", messages=messages),
            completeness=completeness,
            tier=tier,
            source=TraceSource(
                dataset="agentdojo",
                tier="raw",
                subset=metadata.get("suite_name"),
                record_locator=_record_locator(
                    {"index": record.get("index"), **metadata},
                    line_number,
                    ["index", "name", "user_task_id", "injection_task_id"],
                ),
                ingest_version="etl_a_v1",
            ),
            messages=messages,
            split=split,
            task=TraceTask(
                family="prompt_injection",
                name=metadata.get("suite_name"),
                variant=metadata.get("attack_type") or metadata.get("injection_task_id"),
            ),
            labels=TraceLabels(
                category=category,
                security_outcome="unsafe" if is_attack and not security_passed else "safe",
                attack_type="prompt_injection" if is_attack else None,
                attack_succeeded=is_attack and not security_passed,
                capability_category=metadata.get("suite_name") if not is_attack else None,
            ),
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(
                    class_id=f"agentdojo/{'injection' if category == 'harmful' else 'benign'}"
                ),
            ),
            links=TraceLinks(raw_id=metadata.get("name")),
        )
        return trace
    except Exception as exc:
        logger.warning("Failed to convert AgentDojo record: %s", exc)
        return None


# =============================================================================
# Writers
# =============================================================================

def _write_converted(
    input_path: Path,
    converter,
    output_handle,
    split: str,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    converted = 0
    failed = 0
    for line_number, record in _iter_jsonl(input_path):
        if limit is not None and converted >= limit:
            break
        trace = converter(record, split, line_number)
        if trace:
            output_handle.write(json.dumps(trace.to_dict()) + "\n")
            converted += 1
        else:
            failed += 1
    return converted, failed


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Tier A data to trace_v1")

    parser.add_argument("--fujitsu-dir", type=Path, help="Directory containing Fujitsu JSONL files")
    parser.add_argument("--fujitsu-b4", type=Path, help="Fujitsu B4 JSONL file (orchestrator attacks)")

    parser.add_argument("--agentdojo", type=Path, help="AgentDojo JSONL file")
    parser.add_argument("--agentdojo-limit", type=int, help="Limit AgentDojo records (for testing)")

    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file for trace_v1")
    parser.add_argument("--split", default="train", help="Split assignment")

    args = parser.parse_args()

    inputs: List[Tuple[Path, Any, Optional[int]]] = []

    if args.fujitsu_dir:
        for key, filename in FUJITSU_FILE_MAP.items():
            file_path = args.fujitsu_dir / filename
            if file_path.exists():
                converter = {
                    "b4": convert_fujitsu_b4_record,
                }[key]
                inputs.append((file_path, converter, None))
            else:
                logger.warning("Missing Fujitsu file: %s", file_path)

    if args.fujitsu_b4:
        inputs.append((args.fujitsu_b4, convert_fujitsu_b4_record, None))

    if args.agentdojo:
        inputs.append((args.agentdojo, convert_agentdojo_record, args.agentdojo_limit))

    if not inputs:
        parser.error("No input files provided. Use --fujitsu-dir, --fujitsu-b4, or --agentdojo.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_converted = 0
    total_failed = 0

    with open(args.output, "w") as out_f:
        for path, converter, limit in inputs:
            logger.info("Converting %s", path)
            converted, failed = _write_converted(path, converter, out_f, args.split, limit=limit)
            total_converted += converted
            total_failed += failed

    logger.info("Converted %s traces, %s failed", total_converted, total_failed)


if __name__ == "__main__":
    main()
