#!/usr/bin/env python3
"""
Data Conversion Utilities

Convert existing data formats to the new canonical trace schema.

Usage:
    # Convert existing ds/dr stage1 files
    python -m src.schemas.tools.convert_data \
        --ds data/circuit_breakers/ds_stage1.jsonl \
        --dr data/circuit_breakers/dr_stage1.jsonl \
        --output data/traces/cb_traces.jsonl

    # Convert raw Fujitsu B4 data
    python -m src.schemas.tools.convert_data \
        --raw-fujitsu data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
        --output data/traces/fujitsu_raw_traces.jsonl

    # Convert AgentDojo data
    python -m src.schemas.tools.convert_data \
        --agentdojo data/agent_dojo/agentdojo-claude-3-5-sonnet-20241022.jsonl \
        --output data/traces/agentdojo_traces.jsonl
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.schemas import (
    Trace, Message, TraceSource, TraceLabels, TraceTraining,
    TraceToolAttack, TraceLinks, TraceMixture,
    SignalHints, InjectionCharSpan,
    ToolCall, ToolCallFunction, ToolCallExpected,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Fujitsu B4 Conversion
# =============================================================================

def parse_tool_call_from_completion(completion: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract tool name and arguments from a completion string."""
    # Pattern 1: <|python_tag|>{"name": "...", "parameters": {...}}
    if "<|python_tag|>" in completion:
        content = completion.split("<|python_tag|>", 1)[1]
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>"]:
            content = content.split(end_token)[0]
        content = content.strip()
        try:
            data = json.loads(content)
            return data.get("name"), data.get("parameters", {})
        except json.JSONDecodeError:
            pass

    # Pattern 2: {"name": "...", ...}
    match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"', completion)
    if match:
        tool_name = match.group(1)
        # Try to extract parameters
        params_match = re.search(r'"parameters"\s*:\s*(\{[^}]+\})', completion)
        params = {}
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except:
                pass
        return tool_name, params

    return None


def find_injection_span(content: str, injection_text: Optional[str]) -> Optional[InjectionCharSpan]:
    """Find the character span of injection text in content."""
    if not injection_text or not injection_text.strip():
        return None

    # Try exact match
    idx = content.find(injection_text)
    if idx >= 0:
        return InjectionCharSpan(
            message_index=1,  # Usually in user message
            char_start=idx,
            char_end=idx + len(injection_text),
        )

    # Try common injection patterns
    patterns = [
        r'<!--.*?-->',  # HTML comments
        r'/\*.*?\*/',   # C-style comments
        r'#.*$',        # Shell comments (multiline)
        r'\[INST\].*?\[/INST\]',  # Instruction markers
        r'IGNORE.*?(?=\n|$)',     # IGNORE prefix
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return InjectionCharSpan(
                message_index=1,
                char_start=match.start(),
                char_end=match.end(),
            )

    return None


def convert_fujitsu_ds_record(
    record: Dict[str, Any],
    split: str = "train",
) -> Optional[Trace]:
    """
    Convert a ds_stage1.jsonl record to canonical Trace.

    Expected format:
    {
        "id": "fujitsu_b4_42_ds",
        "source": "fujitsu_b4",
        "messages": [...],
        "harmful_completion": "...",
        "tool_attack": {...}
    }
    """
    try:
        messages = []
        for msg_data in record.get("messages", []):
            messages.append(Message(
                role=msg_data.get("role", "user"),
                content=msg_data.get("content", ""),
            ))

        # Add harmful completion as assistant message
        harmful_completion = record.get("harmful_completion", "")
        if harmful_completion:
            # Parse tool call from completion
            tool_info = parse_tool_call_from_completion(harmful_completion)
            tool_calls = None
            if tool_info:
                tool_name, tool_args = tool_info
                tool_calls = [ToolCall(
                    function=ToolCallFunction(name=tool_name, arguments=tool_args),
                    expected=ToolCallExpected(
                        should_call=True,
                        tool_name=record.get("tool_attack", {}).get("expected_tool"),
                    ),
                )]

            messages.append(Message(
                role="assistant",
                content=harmful_completion,
                tool_calls=tool_calls,
            ))

        # Extract tool attack info
        tool_attack_data = record.get("tool_attack", {})
        tool_attack = TraceToolAttack(
            expected_tool=tool_attack_data.get("expected_tool"),
            observed_tool=tool_attack_data.get("observed_tool"),
            attack_vector=tool_attack_data.get("attack_type"),
            injection_text=tool_attack_data.get("injection_text"),
        )

        # Find injection span
        user_content = messages[1].content if len(messages) > 1 else ""
        injection_span = find_injection_span(
            user_content,
            tool_attack_data.get("injection_text") or tool_attack_data.get("malicious_injection"),
        )

        # Create signal hints
        signal_hints = SignalHints(
            injection_char_span=injection_span,
            expected_tool_name=tool_attack_data.get("expected_tool"),
            observed_tool_name=tool_attack_data.get("observed_tool"),
        )

        trace = Trace(
            id=Trace.generate_id("fujitsu_b4", messages=messages),
            source=TraceSource(
                dataset="fujitsu_b4",
                tier="curated",
                subset="orchestrator",
                record_locator={"kind": "uuid", "value": record.get("id", "")},
            ),
            messages=messages,
            split=split,
            labels=TraceLabels(
                category="harmful",
                security_outcome="unsafe",
                attack_type="tool_flip",
                attack_succeeded=tool_attack_data.get("attack_succeeded", True),
            ),
            tool_attack=tool_attack,
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(class_id="fujitsu_b4/tool_flip"),
            ),
            signal_hints=signal_hints,
        )

        return trace

    except Exception as e:
        logger.warning(f"Failed to convert ds record {record.get('id', 'unknown')}: {e}")
        return None


def convert_fujitsu_dr_record(
    record: Dict[str, Any],
    ds_trace_id: Optional[str] = None,
    split: str = "train",
) -> Optional[Trace]:
    """
    Convert a dr_stage1.jsonl record to canonical Trace.

    Expected format:
    {
        "id": "fujitsu_b4_42_dr",
        "source": "fujitsu_b4",
        "messages": [...],
        "benign_completion": "...",
        "paired_ds_id": "fujitsu_b4_42_ds"
    }
    """
    try:
        messages = []
        for msg_data in record.get("messages", []):
            messages.append(Message(
                role=msg_data.get("role", "user"),
                content=msg_data.get("content", ""),
            ))

        # Add benign completion
        benign_completion = record.get("benign_completion", "")
        if benign_completion:
            tool_info = parse_tool_call_from_completion(benign_completion)
            tool_calls = None
            if tool_info:
                tool_name, tool_args = tool_info
                tool_calls = [ToolCall(
                    function=ToolCallFunction(name=tool_name, arguments=tool_args),
                )]

            messages.append(Message(
                role="assistant",
                content=benign_completion,
                tool_calls=tool_calls,
            ))

        trace = Trace(
            id=Trace.generate_id("fujitsu_b4", messages=messages),
            source=TraceSource(
                dataset="fujitsu_b4",
                tier="curated",
                subset="orchestrator",
                record_locator={"kind": "uuid", "value": record.get("id", "")},
            ),
            messages=messages,
            split=split,
            labels=TraceLabels(
                category="benign",
                security_outcome="safe",
            ),
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(class_id="fujitsu_b4/benign"),
            ),
            links=TraceLinks(
                paired_trace_id=ds_trace_id,
                raw_id=record.get("paired_ds_id"),
            ),
        )

        return trace

    except Exception as e:
        logger.warning(f"Failed to convert dr record {record.get('id', 'unknown')}: {e}")
        return None


def convert_raw_fujitsu_record(
    record: Dict[str, Any],
    split: str = "train",
) -> Optional[Trace]:
    """
    Convert a raw Fujitsu orchestrator_attacks record.

    Expected format (from data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl):
    {
        "record_id": "...",
        "attack_id": "01.O-1",
        "category": "LLM01",
        "subtype": "Hidden-JSON force web-search",
        "benign_query": "...",
        "malicious_injection": "...",
        "combined_query": "...",
        "expected_tool": "retrieve_multimodal_docs",
        "simulated_tool": "search_web",
        "success": true
    }
    """
    try:
        # Build messages
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant with access to a knowledge base and web search.",
            ),
            Message(
                role="user",
                content=record.get("combined_query", record.get("benign_query", "")),
            ),
        ]

        # Find injection span
        injection_span = find_injection_span(
            record.get("combined_query", ""),
            record.get("malicious_injection"),
        )

        signal_hints = SignalHints(
            injection_char_span=injection_span,
            expected_tool_name=record.get("expected_tool"),
            observed_tool_name=record.get("simulated_tool"),
        )

        trace = Trace(
            id=Trace.generate_id("fujitsu_b4", messages=messages),
            source=TraceSource(
                dataset="fujitsu_b4",
                tier="raw",
                subset=record.get("category", "orchestrator"),
                record_locator={"kind": "uuid", "value": record.get("record_id", "")},
            ),
            messages=messages,
            split=split,
            labels=TraceLabels(
                category="harmful" if record.get("success") else "benign",
                security_outcome="unsafe" if record.get("success") else "safe",
                attack_type=record.get("subtype", "tool_flip"),
                attack_succeeded=record.get("success", False),
            ),
            tool_attack=TraceToolAttack(
                expected_tool=record.get("expected_tool"),
                observed_tool=record.get("simulated_tool"),
                attack_vector=record.get("subtype"),
                injection_text=record.get("malicious_injection"),
            ),
            training=TraceTraining(
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(class_id="fujitsu_b4/tool_flip"),
            ),
            signal_hints=signal_hints,
        )

        return trace

    except Exception as e:
        logger.warning(f"Failed to convert raw fujitsu record: {e}")
        return None


# =============================================================================
# AgentDojo Conversion
# =============================================================================

def convert_agentdojo_record(
    record: Dict[str, Any],
    split: str = "train",
) -> Optional[Trace]:
    """
    Convert an AgentDojo record to canonical Trace.
    """
    try:
        metadata = record.get("metadata", {})
        raw_messages = record.get("messages", [])

        messages = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool calls
            tool_calls = None
            if msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {"raw": args}
                    tool_calls.append(ToolCall(
                        call_id=tc.get("id"),
                        function=ToolCallFunction(
                            name=func.get("name", ""),
                            arguments=args,
                        ),
                    ))

            messages.append(Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=msg.get("tool_call_id"),
                thinking=msg.get("thinking"),
            ))

        # Determine category
        is_attack = metadata.get("injection_task_id") is not None
        security_passed = metadata.get("security", True)
        category = "harmful" if is_attack and not security_passed else "benign"

        trace = Trace(
            id=Trace.generate_id("agentdojo", messages=messages),
            source=TraceSource(
                dataset="agentdojo",
                tier="curated",
                subset=metadata.get("suite_name"),
                record_locator={"kind": "path", "value": metadata.get("name", "")},
            ),
            messages=messages,
            split=split,
            labels=TraceLabels(
                category=category,
                security_outcome="unsafe" if is_attack and not security_passed else "safe",
                attack_type="prompt_injection" if is_attack else None,
                attack_succeeded=is_attack and not security_passed,
            ),
            training=TraceTraining(
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(
                    class_id=f"agentdojo/{'injection' if category == 'harmful' else 'benign'}"
                ),
            ),
        )

        return trace

    except Exception as e:
        logger.warning(f"Failed to convert agentdojo record: {e}")
        return None


# =============================================================================
# Batch Conversion
# =============================================================================

def convert_fujitsu_ds_dr(
    ds_path: Path,
    dr_path: Path,
    output_path: Path,
    split: str = "train",
) -> Tuple[int, int]:
    """
    Convert paired ds/dr JSONL files to canonical traces.

    Returns (num_converted, num_failed).
    """
    converted = 0
    failed = 0

    # Build DS ID -> trace ID mapping
    ds_trace_ids = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        # Convert DS records
        logger.info(f"Converting DS from {ds_path}...")
        with open(ds_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                trace = convert_fujitsu_ds_record(record, split=split)
                if trace:
                    ds_trace_ids[record.get("id", "")] = trace.id
                    out_f.write(json.dumps(trace.to_dict()) + "\n")
                    converted += 1
                else:
                    failed += 1

        # Convert DR records
        logger.info(f"Converting DR from {dr_path}...")
        with open(dr_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ds_id = record.get("paired_ds_id", "")
                ds_trace_id = ds_trace_ids.get(ds_id)
                trace = convert_fujitsu_dr_record(record, ds_trace_id=ds_trace_id, split=split)
                if trace:
                    out_f.write(json.dumps(trace.to_dict()) + "\n")
                    converted += 1
                else:
                    failed += 1

    logger.info(f"Converted {converted} traces, {failed} failed")
    return converted, failed


def convert_agentdojo(
    input_path: Path,
    output_path: Path,
    split: str = "train",
) -> Tuple[int, int]:
    """Convert AgentDojo JSONL to canonical traces."""
    converted = 0
    failed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        with open(input_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                trace = convert_agentdojo_record(record, split=split)
                if trace:
                    out_f.write(json.dumps(trace.to_dict()) + "\n")
                    converted += 1
                else:
                    failed += 1

    logger.info(f"Converted {converted} traces, {failed} failed")
    return converted, failed


def convert_existing_batches(
    input_path: Path,
    output_path: Path,
) -> Tuple[int, int]:
    """
    Convert existing cb_training_batches.jsonl format to canonical traces.

    Input format:
    {"batch_id": "...", "harmful": [...], "benign": [...]}
    """
    converted = 0
    failed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        with open(input_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                batch = json.loads(line)

                for sample in batch.get("harmful", []):
                    trace = convert_fujitsu_ds_record(sample)
                    if trace:
                        out_f.write(json.dumps(trace.to_dict()) + "\n")
                        converted += 1
                    else:
                        failed += 1

                for sample in batch.get("benign", []):
                    trace = convert_fujitsu_dr_record(sample)
                    if trace:
                        out_f.write(json.dumps(trace.to_dict()) + "\n")
                        converted += 1
                    else:
                        failed += 1

    logger.info(f"Converted {converted} traces, {failed} failed")
    return converted, failed


def load_existing_ds_dr(
    ds_path: Path,
    dr_path: Optional[Path] = None,
) -> Iterator[Trace]:
    """
    Load existing ds/dr files and yield canonical Traces.

    Useful for streaming conversion.
    """
    with open(ds_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            trace = convert_fujitsu_ds_record(record)
            if trace:
                yield trace

    if dr_path:
        with open(dr_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                trace = convert_fujitsu_dr_record(record)
                if trace:
                    yield trace


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert data to canonical trace format")

    # Fujitsu DS/DR conversion
    parser.add_argument("--ds", type=Path, help="DS stage1 JSONL file")
    parser.add_argument("--dr", type=Path, help="DR stage1 JSONL file")

    # Raw Fujitsu conversion
    parser.add_argument("--raw-fujitsu", type=Path, help="Raw Fujitsu orchestrator JSONL")

    # AgentDojo conversion
    parser.add_argument("--agentdojo", type=Path, help="AgentDojo JSONL file")

    # Existing batches conversion
    parser.add_argument("--batches", type=Path, help="Existing cb_training_batches.jsonl")

    # Output
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--split", default="train", help="Split assignment")

    args = parser.parse_args()

    if args.ds and args.dr:
        convert_fujitsu_ds_dr(args.ds, args.dr, args.output, split=args.split)
    elif args.raw_fujitsu:
        converted, failed = 0, 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as out_f:
            with open(args.raw_fujitsu, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    trace = convert_raw_fujitsu_record(record, split=args.split)
                    if trace:
                        out_f.write(json.dumps(trace.to_dict()) + "\n")
                        converted += 1
                    else:
                        failed += 1
        logger.info(f"Converted {converted} traces, {failed} failed")
    elif args.agentdojo:
        convert_agentdojo(args.agentdojo, args.output, split=args.split)
    elif args.batches:
        convert_existing_batches(args.batches, args.output)
    else:
        parser.error("Must specify --ds/--dr, --raw-fujitsu, --agentdojo, or --batches")


if __name__ == "__main__":
    main()
