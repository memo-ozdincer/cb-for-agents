#!/usr/bin/env python3
"""
Rebuild CB Training Batches - V2 (Tool-Routing Only)

This script FILTERS and rebuilds training data to include ONLY tool-routing samples
with consistent Llama 3.1 formatting.

CRITICAL FIXES from V1:
1. FILTER: Only include samples with tool calls (is_agentic, has_tool_calls, tool_attack)
2. PRESERVE: Use original messages/system prompt from the sample, don't replace
3. FORMAT: Consistent <|python_tag|>{...}<|eom_id|> for ALL tool calls
4. VALIDATE: Require 100% tool-call coverage in output

Usage:
    python scripts/cb_data_generation/rebuild_training_data_v2.py \
        --input data/cb_mvp/cb_training_batches.jsonl \
        --output data/cb_mvp/cb_training_batches_v2.jsonl \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --filter-tool-only
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Llama 3.1 Format Constants
# =============================================================================

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_HEADER_START = "<|start_header_id|>"
LLAMA_HEADER_END = "<|end_header_id|>"
LLAMA_EOT = "<|eot_id|>"
LLAMA_EOM = "<|eom_id|>"  # End of message - used for tool calls expecting response
LLAMA_PYTHON_TAG = "<|python_tag|>"


def format_llama31_header(role: str) -> str:
    """Format a role header in Llama 3.1 format."""
    return f"{LLAMA_HEADER_START}{role}{LLAMA_HEADER_END}\n\n"


def format_tool_call_json(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Format a tool call as Llama 3.1 JSON format."""
    return f'{LLAMA_PYTHON_TAG}{{"name": "{tool_name}", "parameters": {json.dumps(arguments)}}}'


# =============================================================================
# Tool Call Detection and Parsing
# =============================================================================

def is_tool_routing_sample(sample: Dict[str, Any]) -> bool:
    """Check if a sample is a tool-routing sample (not text generation)."""
    # Explicit flags
    if sample.get("is_agentic"):
        return True
    if sample.get("has_tool_calls"):
        return True
    if sample.get("tool_attack"):
        return True
    
    # Check for tool call in completion or text
    completion = sample.get("harmful_completion", "") or sample.get("benign_completion", "")
    completion = completion or sample.get("assistant_raw", "")
    text = sample.get("text", "")
    
    # Check for various tool call formats
    if LLAMA_PYTHON_TAG in completion or LLAMA_PYTHON_TAG in text:
        return True
    if "[TOOL_CALLS]" in completion or "[TOOL_CALLS]" in text:
        return True
    if re.search(r'\{"name":\s*"(search_web|retrieve_multimodal_docs)', completion):
        return True
    
    # Check metadata
    metadata = sample.get("metadata", {})
    if metadata.get("from") == "fujitsu_b4_orchestrator":
        return True
    if "B4" in metadata.get("benchmark", ""):
        return True
    
    return False


def extract_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract tool name and arguments from text containing a tool call."""
    # Method 1: <|python_tag|>{...} format
    if LLAMA_PYTHON_TAG in text:
        content = text.split(LLAMA_PYTHON_TAG, 1)[1]
        for end_token in [LLAMA_EOM, LLAMA_EOT, "</s>", "<|end"]:
            content = content.split(end_token)[0]
        content = content.strip()
        
        try:
            data = json.loads(content)
            name = data.get("name")
            params = data.get("parameters", data.get("arguments", {}))
            if name:
                return name, params if isinstance(params, dict) else {}
        except json.JSONDecodeError:
            pass
    
    # Method 2: tool_name({...}) function call syntax
    match = re.search(r'(\w+)\s*\(\s*(\{.*?\})\s*\)', text, re.DOTALL)
    if match:
        try:
            return match.group(1), json.loads(match.group(2))
        except json.JSONDecodeError:
            pass
    
    # Method 3: JSON with "name" field
    match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)".*?"parameters"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if match:
        try:
            return match.group(1), json.loads(match.group(2))
        except json.JSONDecodeError:
            pass
    
    # Method 4: [TOOL_CALLS] format (AgentDojo style) - convert to standard
    if "[TOOL_CALLS]" in text or "[TOOL_CALL:" in text:
        # Extract tool name from [TOOL_CALL: name(...)]
        match = re.search(r'\[TOOL_CALL[S]?[:\s]*(\w+)\s*\(([^)]*)\)', text)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str) if args_str.startswith("{") else {"query": args_str.strip('"')}
                return tool_name, args
            except:
                return tool_name, {}
    
    return None


def extract_all_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract ALL tool calls from text (may have multiple in sequence)."""
    tool_calls = []
    
    # Method 1: [TOOL_CALLS] [...] format (AgentDojo)
    # Format: [TOOL_CALLS] [{"id": "...", "type": "function", "function": {"name": "...", "arguments": {...}}}]
    if "[TOOL_CALLS]" in text:
        # Find the JSON array after [TOOL_CALLS] 
        # Match: [TOOL_CALLS] [<json array content>]
        # The array may contain nested objects with their own brackets
        pattern = r'\[TOOL_CALLS\]\s*(\[[\s\S]*?\](?=\s*(?:\n|$|[A-Z#]|\[TOOL_CALLS\])))'
        
        for match in re.finditer(pattern, text):
            array_str = match.group(1)
            try:
                calls = json.loads(array_str)
                if isinstance(calls, list):
                    for call in calls:
                        if call.get("type") == "function":
                            func = call.get("function", {})
                            name = func.get("name")
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    args = {"raw": args}
                            if name:
                                tool_calls.append((name, args if isinstance(args, dict) else {}))
            except json.JSONDecodeError:
                # Try to extract just the first object from the array
                try:
                    # Find first {"id"... object
                    obj_match = re.search(r'\{"id"[^}]+\}', array_str)
                    if obj_match:
                        # Try a more targeted extraction
                        inner_match = re.search(r'"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{[^}]*\}|\{\})', array_str, re.DOTALL)
                        if inner_match:
                            name = inner_match.group(1)
                            try:
                                args = json.loads(inner_match.group(2))
                            except:
                                args = {}
                            tool_calls.append((name, args))
                except:
                    pass
    
    # Method 2: <|python_tag|>{...} format
    if LLAMA_PYTHON_TAG in text:
        parts = text.split(LLAMA_PYTHON_TAG)
        for i, part in enumerate(parts[1:], 1):  # Skip first part (before first tag)
            content = part
            for end_token in [LLAMA_EOM, LLAMA_EOT, "</s>", "<|end", LLAMA_PYTHON_TAG]:
                content = content.split(end_token)[0]
            content = content.strip()
            
            try:
                data = json.loads(content)
                name = data.get("name")
                params = data.get("parameters", data.get("arguments", {}))
                if name:
                    tool_calls.append((name, params if isinstance(params, dict) else {}))
            except json.JSONDecodeError:
                # Try function call syntax: func_name({...})
                match = re.match(r'(\w+)\s*\(\s*(\{.*?\})\s*\)', content, re.DOTALL)
                if match:
                    try:
                        tool_calls.append((match.group(1), json.loads(match.group(2))))
                    except json.JSONDecodeError:
                        pass
    
    return tool_calls


def normalize_tool_call_to_llama31(completion: str) -> str:
    """
    Normalize any tool call format to consistent Llama 3.1 format.
    
    Input formats handled:
    - <|python_tag|>tool_name({...})
    - <|python_tag|>{"name": "...", "parameters": {...}}
    - tool_name({"query": "..."})
    - [TOOL_CALLS] [{"type": "function", "function": {"name": "...", "arguments": {...}}}]
    
    Output format:
    - <|python_tag|>{"name": "...", "parameters": {...}}
    
    For multi-turn tool calls, extracts the FIRST tool call (the adversarial one).
    """
    tool_calls = extract_all_tool_calls(completion)
    
    if not tool_calls:
        # Fallback to simple extraction
        tool_call = extract_tool_call(completion)
        if not tool_call:
            return completion  # Return as-is if no tool call found
        tool_calls = [tool_call]
    
    # Use the FIRST tool call (for training, we focus on initial adversarial routing)
    tool_name, tool_args = tool_calls[0]
    
    # Extract any preamble text before the tool call
    preamble = ""
    for marker in [LLAMA_PYTHON_TAG, "[TOOL_CALL", "{\"name\""]:
        if marker in completion:
            preamble = completion.split(marker)[0].strip()
            break
    
    # Build normalized tool call in Llama 3.1 format
    normalized_call = format_tool_call_json(tool_name, tool_args)
    
    if preamble:
        return f"{preamble}\n\n{normalized_call}"
    return normalized_call


# =============================================================================
# Sample Rendering
# =============================================================================

def render_sample_llama31(
    sample: Dict[str, Any],
    is_harmful: bool,
    default_system_prompt: str,
    tools_json: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    Render a sample in proper Llama 3.1 chat format WITH SPECIAL TOKENS.
    
    CRITICAL: Converts all formats to proper Llama 3.1 with:
    - <|begin_of_text|>
    - <|start_header_id|>{role}<|end_header_id|>
    - <|eot_id|> for system/user, <|eom_id|> for tool-calling assistant
    - <|python_tag|>{"name": ..., "parameters": ...} for tool calls
    
    Returns:
        (rendered_text, has_tool_call, original_completion) tuple
    """
    parts = [LLAMA_BOS]
    
    # === System Message ===
    # Priority 1: Use original messages if they have a system message
    messages = sample.get("messages", [])
    original_system = None
    for msg in messages:
        if msg.get("role") == "system":
            original_system = msg.get("content", "")
            break
    
    # Use original or default
    system_content = original_system or default_system_prompt
    
    # Append tools if provided and not already in system
    if tools_json and "tools" not in system_content.lower():
        system_content += f"\n\nYou have access to the following tools:\n\n{tools_json}"
    
    parts.append(format_llama31_header("system"))
    parts.append(system_content)
    parts.append(LLAMA_EOT)
    
    # === User Message ===
    user_prompt = None
    
    # Priority 1: From current text (if it has User: format)
    text = sample.get("text", "")
    if text.startswith("User:"):
        # Extract user message from "User: ... Assistant: ..." format
        if "Assistant:" in text:
            user_prompt = text.split("Assistant:")[0].replace("User:", "", 1).strip()
    
    # Priority 2: From messages
    if not user_prompt:
        for msg in messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
    
    # Priority 3: From tool_attack
    if not user_prompt:
        tool_attack = sample.get("tool_attack", {})
        user_prompt = tool_attack.get("combined_query", "")
    
    # Priority 4: From user_prompt field
    if not user_prompt:
        user_prompt = sample.get("user_prompt", "")
    
    if not user_prompt:
        logger.warning(f"No user prompt in sample {sample.get('id', 'unknown')}")
        return "", False, ""
    
    parts.append(format_llama31_header("user"))
    parts.append(user_prompt)
    parts.append(LLAMA_EOT)
    
    # === Assistant Completion ===
    # Priority 1: From harmful_completion/benign_completion
    completion = sample.get("harmful_completion" if is_harmful else "benign_completion", "")
    
    # Priority 2: From text (extract assistant part)
    if not completion and "Assistant:" in text:
        completion = text.split("Assistant:", 1)[1].strip()
    
    # Priority 3: From assistant_raw
    if not completion:
        completion = sample.get("assistant_raw", "")
    
    if not completion:
        logger.warning(f"No completion in sample {sample.get('id', 'unknown')}")
        return "", False, ""
    
    # Store original completion for debugging
    original_completion = completion[:500]  # First 500 chars
    
    # Normalize tool call format (convert [TOOL_CALLS] to <|python_tag|>)
    normalized_completion = normalize_tool_call_to_llama31(completion)
    has_tool_call = LLAMA_PYTHON_TAG in normalized_completion
    
    parts.append(format_llama31_header("assistant"))
    parts.append(normalized_completion)
    
    # Use <|eom_id|> for tool calls (expecting tool response), <|eot_id|> otherwise
    if has_tool_call:
        parts.append(LLAMA_EOM)
    else:
        parts.append(LLAMA_EOT)
    
    return "".join(parts), has_tool_call, original_completion


# =============================================================================
# Batch Processing
# =============================================================================

def rebuild_batch(
    batch: Dict[str, Any],
    default_system_prompt: str,
    tools_json: Optional[str],
    filter_tool_only: bool,
    collect_examples: bool = False,
    n_filtered: int = 5,
    n_rendered: int = 3,
    n_failed: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    """
    Rebuild a batch with proper Llama 3.1 format.
    
    Returns:
        (new_batch, stats_dict)
    """
    stats = {
        "harmful_total": 0,
        "harmful_tool_routing": 0,
        "harmful_rendered": 0,
        "harmful_has_tool_call": 0,
        "benign_total": 0,
        "benign_tool_routing": 0,
        "benign_rendered": 0,
        "benign_has_tool_call": 0,
    }
    
    examples = {
        "filtered_harmful": [],
        "filtered_benign": [],
        "rendered_harmful": [],
        "rendered_benign": [],
        "failed_harmful": [],
        "failed_benign": [],
    } if collect_examples else None
    
    new_batch = {
        "id": batch.get("id", "unknown"),
        "harmful": [],
        "benign": [],
    }
    
    # Process harmful samples
    for sample in batch.get("harmful", []):
        stats["harmful_total"] += 1
        
        # Filter if requested
        if filter_tool_only and not is_tool_routing_sample(sample):
            if collect_examples and len(examples["filtered_harmful"]) < n_filtered:
                examples["filtered_harmful"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "reason": "not_tool_routing",
                    "has_is_agentic": sample.get("is_agentic", False),
                    "has_tool_calls": sample.get("has_tool_calls", False),
                    "completion_preview": sample.get("harmful_completion", "")[:200],
                })
            continue
        stats["harmful_tool_routing"] += 1
        
        text, has_tool_call = render_sample_llama31(
            sample, is_harmful=True,
            default_system_prompt=default_system_prompt,
            tools_json=tools_json,
        )
        
        if text:
            new_sample = sample.copy()
            new_sample["text"] = text
            new_sample["is_agentic"] = True
            new_sample["has_tool_call_in_text"] = has_tool_call
            new_batch["harmful"].append(new_sample)
            stats["harmful_rendered"] += 1
            if has_tool_call:
                stats["harmful_has_tool_call"] += 1
            
            if collect_examples and len(examples["rendered_harmful"]) < n_rendered:
                examples["rendered_harmful"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "has_tool_call": has_tool_call,
                    "text_preview": text[:300],
                    "original_completion": sample.get("harmful_completion", "")[:200],
                })
        else:
            if collect_examples and len(examples["failed_harmful"]) < n_failed:
                examples["failed_harmful"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "reason": "render_failed",
                    "sample_keys": list(sample.keys()),
                })
    
    # Process benign samples
    for sample in batch.get("benign", []):
        stats["benign_total"] += 1
        
        if filter_tool_only and not is_tool_routing_sample(sample):
            if collect_examples and len(examples["filtered_benign"]) < n_filtered:
                examples["filtered_benign"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "reason": "not_tool_routing",
                    "has_is_agentic": sample.get("is_agentic", False),
                    "has_tool_calls": sample.get("has_tool_calls", False),
                    "completion_preview": sample.get("benign_completion", "")[:200],
                })
            continue
        stats["benign_tool_routing"] += 1
        
        text, has_tool_call = render_sample_llama31(
            sample, is_harmful=False,
            default_system_prompt=default_system_prompt,
            tools_json=tools_json,
        )
        
        if text:
            new_sample = sample.copy()
            new_sample["text"] = text
            new_sample["is_agentic"] = True
            new_sample["has_tool_call_in_text"] = has_tool_call
            new_batch["benign"].append(new_sample)
            stats["benign_rendered"] += 1
            if has_tool_call:
                stats["benign_has_tool_call"] += 1
            
            if collect_examples and len(examples["rendered_benign"]) < n_rendered:
                examples["rendered_benign"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "has_tool_call": has_tool_call,
                    "text_preview": text[:300],
                    "original_completion": sample.get("benign_completion", "")[:200],
                })
        else:
            if collect_examples and len(examples["failed_benign"]) < n_failed:
                examples["failed_benign"].append({
                    "id": sample.get("id", "unknown"),
                    "batch_id": batch.get("id"),
                    "reason": "render_failed",
                    "sample_keys": list(sample.keys()),
                })
    
    return new_batch, stats, examples


def load_tool_schema(schema_path: Optional[Path]) -> Tuple[str, Optional[str]]:
    """Load tool schema and return system prompt and tools JSON."""
    default_prompt = "You are a helpful assistant with access to tools. Use retrieve_multimodal_docs for knowledge base queries and search_web only for current events not in the knowledge base."
    
    if not schema_path or not schema_path.exists():
        logger.warning(f"Tool schema not found: {schema_path}")
        return default_prompt, None
    
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    system_prompt = schema.get("system_prompt", default_prompt)
    tools = schema.get("tools", [])
    tools_json = json.dumps(tools, indent=2) if tools else None
    
    return system_prompt, tools_json


def validate_output(output_path: Path) -> Dict[str, Any]:
    """Validate the rebuilt output file."""
    stats = {
        "total_batches": 0,
        "total_samples": 0,
        "has_text": 0,
        "has_python_tag": 0,
        "has_eom_terminator": 0,
        "has_eot_terminator": 0,
        "has_assistant_header": 0,
        "preserved_system_prompt": 0,
    }
    
    with open(output_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            batch = json.loads(line)
            stats["total_batches"] += 1
            
            for sample in batch.get("harmful", []) + batch.get("benign", []):
                stats["total_samples"] += 1
                text = sample.get("text", "")
                
                if text:
                    stats["has_text"] += 1
                if LLAMA_PYTHON_TAG in text:
                    stats["has_python_tag"] += 1
                if LLAMA_EOM in text:
                    stats["has_eom_terminator"] += 1
                if LLAMA_EOT in text:
                    stats["has_eot_terminator"] += 1
                if f"{LLAMA_HEADER_START}assistant{LLAMA_HEADER_END}" in text:
                    stats["has_assistant_header"] += 1
                # Check for non-generic system prompt (has tool routing info)
                if "retrieve_multimodal_docs" in text or "search_web" in text:
                    stats["preserved_system_prompt"] += 1
    
    return stats


def print_examples_report(examples: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print collected examples by category."""
    if not examples:
        return
    
    print(f"\n{'='*80}")
    print("REBUILD EXAMPLES BY CATEGORY")
    print('='*80)
    
    categories = [
        ("filtered_harmful", "Filtered Harmful (not tool-routing)"),
        ("filtered_benign", "Filtered Benign (not tool-routing)"),
        ("rendered_harmful", "Successfully Rendered Harmful"),
        ("rendered_benign", "Successfully Rendered Benign"),
        ("failed_harmful", "Failed to Render Harmful"),
        ("failed_benign", "Failed to Render Benign"),
    ]
    
    for key, title in categories:
        items = examples.get(key, [])
        if not items:
            continue
        
        print(f"\n--- {title} ({len(items)} examples) ---")
        
        for i, ex in enumerate(items, 1):
            print(f"\n  Example {i}/{len(items)}:")
            print(f"    ID: {ex.get('id', 'N/A')}")
            print(f"    Batch: {ex.get('batch_id', 'N/A')}")
            
            if 'reason' in ex:
                print(f"    Reason: {ex.get('reason')}")
            if 'has_tool_call' in ex:
                print(f"    Has tool call: {ex.get('has_tool_call')}")
            if 'has_is_agentic' in ex:
                print(f"    is_agentic flag: {ex.get('has_is_agentic')}")
            if 'has_tool_calls' in ex:
                print(f"    has_tool_calls flag: {ex.get('has_tool_calls')}")
            if 'completion_preview' in ex:
                print(f"    Completion preview: {ex.get('completion_preview')}")
            if 'text_preview' in ex:
                print(f"    Rendered text preview: {ex.get('text_preview')}")
            if 'original_completion' in ex:
                print(f"    Original completion: {ex.get('original_completion')}")
            if 'sample_keys' in ex:
                print(f"    Available keys: {ex.get('sample_keys')}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild CB training batches with proper Llama 3.1 format (V2)",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--tool-schema", type=Path, default=None, help="Tool schema JSON")
    parser.add_argument("--filter-tool-only", action="store_true", 
                        help="Only include tool-routing samples (RECOMMENDED)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't rebuild")
    parser.add_argument("--limit", type=int, default=None, help="Limit batches")
    parser.add_argument("--require-100-pct-tools", action="store_true",
                        help="Fail if <100% samples have tool calls")
    
    # Example collection
    example_group = parser.add_argument_group("example collection")
    example_group.add_argument(
        "--print-examples",
        action="store_true",
        help="Print collected examples to stdout",
    )
    example_group.add_argument(
        "--examples-out",
        type=Path,
        default=None,
        help="Path to save examples JSON (default: <output>.examples.json)",
    )
    example_group.add_argument(
        "--n-filtered",
        type=int,
        default=5,
        help="Number of filtered samples to collect per type",
    )
    example_group.add_argument(
        "--n-rendered",
        type=int,
        default=3,
        help="Number of successfully rendered samples to collect per type",
    )
    example_group.add_argument(
        "--n-failed",
        type=int,
        default=5,
        help="Number of failed samples to collect per type",
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        logger.info(f"Validating {args.input}...")
        stats = validate_output(args.input)
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        for k, v in stats.items():
            pct = 100 * v / max(1, stats["total_samples"]) if "total" not in k else ""
            logger.info(f"  {k}: {v}" + (f" ({pct:.1f}%)" if pct else ""))
        return 0
    
    # Load tool schema
    system_prompt, tools_json = load_tool_schema(args.tool_schema)
    logger.info(f"System prompt: {system_prompt[:100]}...")
    logger.info(f"Tools JSON: {'loaded' if tools_json else 'none'}")
    
    # Load input
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return 1
    
    batches = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))
    
    logger.info(f"Loaded {len(batches)} batches from {args.input}")
    
    if args.limit:
        batches = batches[:args.limit]
    
    # Process batches
    output_batches = []
    total_stats = {
        "harmful_total": 0, "harmful_tool_routing": 0, "harmful_rendered": 0, "harmful_has_tool_call": 0,
        "benign_total": 0, "benign_tool_routing": 0, "benign_rendered": 0, "benign_has_tool_call": 0,
    }
    
    collect_examples = args.print_examples or args.examples_out is not None
    all_examples = {
        "filtered_harmful": [],
        "filtered_benign": [],
        "rendered_harmful": [],
        "rendered_benign": [],
        "failed_harmful": [],
        "failed_benign": [],
    } if collect_examples else None
    
    for batch in batches:
        new_batch, stats, examples = rebuild_batch(
            batch, system_prompt, tools_json, args.filter_tool_only,
            collect_examples=collect_examples,
            n_filtered=args.n_filtered,
            n_rendered=args.n_rendered,
            n_failed=args.n_failed,
        )
        
        # Only include non-empty batches
        if new_batch["harmful"] or new_batch["benign"]:
            output_batches.append(new_batch)
        
        for k, v in stats.items():
            total_stats[k] += v
        
        # Merge examples (respecting global limits)
        if collect_examples and examples:
            for key in all_examples:
                # Only add if we haven't reached the limit yet
                limit = args.n_filtered if 'filtered' in key else (args.n_rendered if 'rendered' in key else args.n_failed)
                remaining = limit - len(all_examples[key])
                if remaining > 0:
                    all_examples[key].extend(examples[key][:remaining])
    
    # Print stats
    logger.info("\n" + "=" * 60)
    logger.info("REBUILD STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Input batches: {len(batches)}")
    logger.info(f"Output batches: {len(output_batches)}")
    logger.info("")
    logger.info("HARMFUL:")
    logger.info(f"  Total samples: {total_stats['harmful_total']}")
    logger.info(f"  Tool-routing: {total_stats['harmful_tool_routing']} ({100*total_stats['harmful_tool_routing']/max(1,total_stats['harmful_total']):.1f}%)")
    logger.info(f"  Rendered: {total_stats['harmful_rendered']}")
    logger.info(f"  Has <|python_tag|>: {total_stats['harmful_has_tool_call']} ({100*total_stats['harmful_has_tool_call']/max(1,total_stats['harmful_rendered']):.1f}%)")
    logger.info("")
    logger.info("BENIGN:")
    logger.info(f"  Total samples: {total_stats['benign_total']}")
    logger.info(f"  Tool-routing: {total_stats['benign_tool_routing']} ({100*total_stats['benign_tool_routing']/max(1,total_stats['benign_total']):.1f}%)")
    logger.info(f"  Rendered: {total_stats['benign_rendered']}")
    logger.info(f"  Has <|python_tag|>: {total_stats['benign_has_tool_call']} ({100*total_stats['benign_has_tool_call']/max(1,total_stats['benign_rendered']):.1f}%)")
    
    # Check tool-call coverage
    total_rendered = total_stats['harmful_rendered'] + total_stats['benign_rendered']
    total_tool_calls = total_stats['harmful_has_tool_call'] + total_stats['benign_has_tool_call']
    tool_pct = 100 * total_tool_calls / max(1, total_rendered)
    
    logger.info("")
    logger.info(f"OVERALL TOOL-CALL COVERAGE: {tool_pct:.1f}%")
    
    if args.require_100_pct_tools and tool_pct < 99.0:
        logger.error(f"FAILED: Tool-call coverage {tool_pct:.1f}% < 100%")
        return 1
    
    # Handle examples
    if collect_examples and all_examples:
        if args.print_examples:
            print_examples_report(all_examples)
        
        examples_path = args.examples_out or args.output.with_suffix(".examples.json")
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"\nWrote examples to {examples_path}")
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for batch in output_batches:
            f.write(json.dumps(batch) + "\n")
    
    logger.info(f"\nWrote {len(output_batches)} batches to {args.output}")
    
    # Print example
    if output_batches and output_batches[0].get("harmful"):
        example = output_batches[0]["harmful"][0]
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE (first harmful sample, first 1500 chars)")
        logger.info("=" * 60)
        logger.info(example.get("text", "")[:1500])
    
    # Run validation
    logger.info("\n" + "=" * 60)
    logger.info("POST-REBUILD VALIDATION")
    logger.info("=" * 60)
    val_stats = validate_output(args.output)
    for k, v in val_stats.items():
        pct = 100 * v / max(1, val_stats["total_samples"]) if "total" not in k else ""
        logger.info(f"  {k}: {v}" + (f" ({pct:.1f}%)" if pct else ""))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
