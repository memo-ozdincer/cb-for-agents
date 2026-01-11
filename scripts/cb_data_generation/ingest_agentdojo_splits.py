#!/usr/bin/env python3
"""
Stage 2: Ingest AgentDojo Traces Split by Security Outcome

This script processes AgentDojo benchmark traces and splits them by security outcome:
- security == False → Ds (harmful set) - injection attacks that SUCCEEDED
- security == True → Dr (retain set) - model RESISTED the injection

AgentDojo Structure:
- Multi-turn tool-use traces with potential prompt injections
- `metadata.security`: True if model resisted attack, False if attack succeeded
- `metadata.injection_task_id`: Non-null if trace contains injection
- Full conversation with tool_calls and <thinking> blocks

Usage:
    python scripts/cb_data_generation/ingest_agentdojo_splits.py \
        --input-dir data/agent_dojo/ \
        --output-harmful data/circuit_breakers/harmful/agentdojo_failures.jsonl \
        --output-retain data/circuit_breakers/retain/agentdojo_resisted.jsonl \
        --max-each 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# AgentDojo Parsing
# =============================================================================

def extract_last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the last assistant message content, including tool calls if present.
    
    AgentDojo format has tool_calls in assistant messages.
    We need to format these for Llama 3.1 tool-calling format.
    """
    # Find last assistant message
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            # Check for tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls and len(tool_calls) > 0:
                # Format first tool call in Llama 3.1 format
                tc = tool_calls[0]
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                
                # Arguments can be string or dict
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                
                tool_call_json = {
                    "name": tool_name,
                    "parameters": args
                }
                return f'<|python_tag|>{json.dumps(tool_call_json)}<|eom_id|>'
            
            # No tool calls, return text content
            content = msg.get("content", "")
            if content:
                return content
    
    return ""


def extract_tool_name_from_assistant(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the tool name from the last assistant message with tool calls."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]
                func = tc.get("function", {})
                return func.get("name")
    return None


def convert_messages_to_standard_format(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Convert AgentDojo messages to standard format (system, user, assistant).
    
    AgentDojo has:
    - system: System prompt
    - user: User messages
    - assistant: Assistant responses with optional tool_calls
    - tool: Tool results
    
    We extract the pre-assistant context (all messages before last assistant turn).
    """
    # Find the index of the last assistant message
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = i
    
    if last_assistant_idx < 0:
        # No assistant message, return all as context
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user") and content:
                result.append({"role": role, "content": content})
        return result
    
    # Return messages up to (but not including) last assistant
    result = []
    for msg in messages[:last_assistant_idx]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Skip tool results for input context
        if role == "tool":
            continue
        
        if role in ("system", "user", "assistant") and content:
            result.append({"role": role, "content": content})
    
    return result


def load_agentdojo_file(path: Path) -> List[Dict[str, Any]]:
    """Load records from a single AgentDojo JSONL file."""
    records = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error in {path.name} line {line_num}: {e}")
                continue
    
    return records


def process_agentdojo_records(
    records: List[Dict[str, Any]],
    source_file: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process AgentDojo records and split by security outcome.
    
    Returns:
        Tuple of (harmful_samples, retain_samples)
    """
    harmful = []
    retain = []
    
    for record in records:
        metadata = record.get("metadata", {})
        messages = record.get("messages", [])
        
        # Skip records without injection (pure benign traces)
        injection_task_id = metadata.get("injection_task_id")
        if injection_task_id is None:
            continue
        
        # Get security outcome
        security = metadata.get("security", True)
        success = metadata.get("success", False)
        
        # Extract assistant content
        assistant_raw = extract_last_assistant_content(messages)
        if not assistant_raw:
            continue
        
        # Extract tool name for labels
        observed_tool = extract_tool_name_from_assistant(messages)
        
        # Convert messages to standard format
        context_messages = convert_messages_to_standard_format(messages)
        if not context_messages:
            continue
        
        # Determine if this has tool calls
        has_tool_calls = "<|python_tag|>" in assistant_raw
        
        # Create sample
        sample = {
            "messages": context_messages,
            "assistant_raw": assistant_raw,
            "labels": {
                "security": security,
                "success": success,
                "observed_tool": observed_tool,
                "injection_task_id": injection_task_id,
            },
            "metadata": {
                "source": "agentdojo",
                "source_file": source_file,
                "suite_name": metadata.get("suite_name", ""),
                "user_task_id": metadata.get("user_task_id", ""),
                "attack_type": metadata.get("attack_type", ""),
                "model": metadata.get("model", ""),
                "has_tool_calls": has_tool_calls,
            },
        }
        
        # Split by security outcome
        if security == False:
            # Attack SUCCEEDED - goes to Ds (harmful)
            sample["metadata"]["split"] = "harmful"
            harmful.append(sample)
        else:
            # Model RESISTED attack - goes to Dr (retain)
            sample["metadata"]["split"] = "retain"
            retain.append(sample)
    
    return harmful, retain


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split AgentDojo traces by security outcome"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BASE_DIR / "data/agent_dojo",
        help="Directory containing AgentDojo JSONL files",
    )
    parser.add_argument(
        "--output-harmful",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/harmful/agentdojo_failures.jsonl",
        help="Output path for harmful samples (attack succeeded)",
    )
    parser.add_argument(
        "--output-retain",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/retain/agentdojo_resisted.jsonl",
        help="Output path for retain samples (attack resisted)",
    )
    parser.add_argument(
        "--max-each",
        type=int,
        default=500,
        help="Maximum samples for each split",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="agentdojo-*.jsonl",
        help="Glob pattern for input files",
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_files = list(args.input_dir.glob(args.file_pattern))
    
    # Exclude info files
    input_files = [f for f in input_files if not f.name.endswith("_info.md")]
    
    if not input_files:
        logger.error(f"No files matching {args.file_pattern} found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} AgentDojo files")
    
    # Process all files
    all_harmful = []
    all_retain = []
    
    for file_path in input_files:
        logger.info(f"Processing {file_path.name}...")
        records = load_agentdojo_file(file_path)
        harmful, retain = process_agentdojo_records(records, file_path.name)
        all_harmful.extend(harmful)
        all_retain.extend(retain)
        logger.info(f"  Found {len(harmful)} harmful, {len(retain)} retain samples")
    
    # Apply limits
    if len(all_harmful) > args.max_each:
        logger.info(f"Limiting harmful samples from {len(all_harmful)} to {args.max_each}")
        all_harmful = all_harmful[:args.max_each]
    
    if len(all_retain) > args.max_each:
        logger.info(f"Limiting retain samples from {len(all_retain)} to {args.max_each}")
        all_retain = all_retain[:args.max_each]
    
    # Create output directories
    args.output_harmful.parent.mkdir(parents=True, exist_ok=True)
    args.output_retain.parent.mkdir(parents=True, exist_ok=True)
    
    # Save harmful samples
    with open(args.output_harmful, "w", encoding="utf-8") as f:
        for sample in all_harmful:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(all_harmful)} harmful samples to {args.output_harmful}")
    
    # Save retain samples
    with open(args.output_retain, "w", encoding="utf-8") as f:
        for sample in all_retain:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(all_retain)} retain samples to {args.output_retain}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("AGENTDOJO SPLIT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Harmful (Ds) - attack succeeded: {len(all_harmful)}")
    logger.info(f"Retain (Dr) - attack resisted: {len(all_retain)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
