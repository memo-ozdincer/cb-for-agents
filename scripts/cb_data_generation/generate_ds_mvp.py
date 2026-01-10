#!/usr/bin/env python3
"""
MVP Circuit Breaker Set (Ds) Generator - Stage 1

Generate the Circuit Breaker Set with BEHAVIORAL FILTERING.
Unlike the full generate_ds.py, this script ONLY includes samples where
the model actually exhibits the harmful behavior (attack succeeds).

Key Stage 1 Principles:
1. Only include samples where observed_tool == simulated_tool (attack succeeded)
2. Enforce minimum yield rate (>10%) or fail
3. Use frozen tool schema (b4_standard_v1)
4. Store full messages array, not just prompt string
5. Validate Llama 3.1 format compliance

Usage:
    python scripts/cb_data_generation/generate_ds_mvp.py \
        --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/cb_mvp/ds_stage1.jsonl \
        --min-yield 0.10

    # Dry run to check yield rate
    python scripts/cb_data_generation/generate_ds_mvp.py --dry-run --limit 100
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# HuggingFace Token Resolution
# =============================================================================

def resolve_hf_token() -> Optional[str]:
    """Resolve HuggingFace token from environment."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


# =============================================================================
# Tool Schema Loading
# =============================================================================

def load_tool_schema(schema_path: Path) -> Dict[str, Any]:
    """Load frozen tool schema from JSON file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    logger.info(f"Loaded tool schema: {schema.get('version', 'unknown')}")
    return schema


def get_tools_list(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tools list from schema."""
    return schema.get("tools", [])


def get_system_prompt(schema: Dict[str, Any]) -> str:
    """Extract system prompt from schema."""
    return schema.get("system_prompt", "You are a helpful assistant with access to tools.")


# =============================================================================
# Data Source Loaders
# =============================================================================

def load_fujitsu_b4(
    path: Path,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Load Fujitsu B4 orchestrator attack records.
    
    Note: We load ALL records (not just success_only), because we need to
    test which ones actually succeed with our target model.
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract fields
            benign_query = record.get("benign_query", "")
            malicious_injection = record.get("malicious_injection", "")
            combined_query = record.get("combined_query", "")
            expected_tool = record.get("expected_tool", "")
            simulated_tool = record.get("simulated_tool", "")
            
            # Skip if missing required fields
            if not combined_query or not expected_tool or not simulated_tool:
                continue
            
            yield {
                "id": record.get("record_id", f"fujitsu_b4_{count}"),
                "source": "fujitsu_b4",
                "benign_query": benign_query,
                "malicious_injection": malicious_injection,
                "combined_query": combined_query,
                "expected_tool": expected_tool,
                "simulated_tool": simulated_tool,
                "metadata": {
                    "attack_id": record.get("attack_id"),
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                    "judge_note": record.get("judge_note"),
                    "original_success": record.get("success", False),
                },
            }
            
            count += 1
            if limit and count >= limit:
                return


# =============================================================================
# Model Loading and Generation
# =============================================================================

def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load model and tokenizer for generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    hf_token = resolve_hf_token()
    
    # Check if we're in offline mode (compute nodes have no internet)
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    logger.info(f"Loading model: {model_path}")
    if offline_mode:
        logger.info("  (offline mode - using cached files only)")
    
    # use_fast=False avoids Mistral regex patching which makes API calls even in offline mode
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=offline_mode,
        use_fast=not offline_mode,  # Use slow tokenizer in offline mode to avoid API calls
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        token=hf_token,
        local_files_only=offline_mode,
    )
    model.eval()
    
    return model, tokenizer


def generate_with_tools(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate response with tool-calling capability.
    
    Returns the raw assistant output (including special tokens).
    """
    # Apply chat template with tools
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Fallback if tools not supported in template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        # Manual formatting
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        input_text = "\n".join(parts) + "\nassistant:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the full output including special tokens for analysis
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the new tokens (assistant response)
    input_len = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    
    return response.strip()


def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract tool call from assistant response.
    
    Handles multiple formats:
    - Llama 3.1: <|python_tag|>tool_name(args)<|eom_id|>
    - JSON: {"name": "...", "parameters": {...}}
    - Function call: tool_name({"arg": "value"})
    """
    # Try Llama 3.1 format first
    if "<|python_tag|>" in response:
        content = response.split("<|python_tag|>", 1)[1]
        # Remove end tokens
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            content = content.split(end_token)[0]
        content = content.strip()
        
        # Try function call syntax: tool_name(args)
        match = re.match(r'(\w+)\s*\((.*)\)', content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Try to parse arguments
            try:
                # Handle keyword arguments: query='value'
                if '=' in args_str and not args_str.startswith('{'):
                    # Convert keyword args to dict
                    args = {}
                    for part in re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
                        args[part[0]] = part[1]
                    return {"name": tool_name, "parameters": args}
                else:
                    # Try JSON parsing
                    args = json.loads(args_str) if args_str else {}
                    return {"name": tool_name, "parameters": args}
            except (json.JSONDecodeError, ValueError):
                return {"name": tool_name, "parameters": {"raw": args_str}}
        
        # Try JSON format
        try:
            data = json.loads(content)
            name = data.get("name") or data.get("function", {}).get("name")
            params = data.get("parameters") or data.get("arguments") or {}
            if name:
                return {"name": name, "parameters": params}
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON tool call anywhere
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', response)
    if json_match:
        try:
            # Find the full JSON object
            start = response.find('{', json_match.start())
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            json_str = response[start:end]
            data = json.loads(json_str)
            name = data.get("name") or data.get("function", {}).get("name")
            params = data.get("parameters") or data.get("arguments") or {}
            if name:
                return {"name": name, "parameters": params}
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None


def validate_llama_format(assistant_raw: str) -> Tuple[bool, str]:
    """
    Validate that assistant_raw follows Llama 3.1 tool format.
    
    Returns (is_valid, error_message).
    """
    if not assistant_raw:
        return False, "Empty assistant_raw"
    
    # Must start with <|python_tag|>
    if not "<|python_tag|>" in assistant_raw:
        return False, "Missing <|python_tag|>"
    
    # Must end with <|eom_id|> or <|eot_id|>
    has_valid_end = any(
        assistant_raw.rstrip().endswith(end)
        for end in ["<|eom_id|>", "<|eot_id|>"]
    )
    if not has_valid_end:
        # Check if it's there but with trailing content
        has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
        if not has_end_token:
            return False, "Missing <|eom_id|> or <|eot_id|>"
    
    # Must NOT have markdown wrappers
    if "```" in assistant_raw:
        return False, "Contains markdown code block"
    
    # Must NOT have common prefixes
    for prefix in ["Action:", "ToolCall:", "Function:"]:
        if assistant_raw.strip().startswith(prefix):
            return False, f"Contains forbidden prefix: {prefix}"
    
    return True, ""


# =============================================================================
# MVP Ds Builder
# =============================================================================

def build_ds_mvp(
    b4_records: List[Dict[str, Any]],
    model,
    tokenizer,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.7,
    min_yield: float = 0.10,
    save_all: bool = False,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Ds from ONLY successful flips under actual runtime.
    
    Args:
        b4_records: List of B4 attack records
        model: The language model
        tokenizer: The tokenizer
        tools: Tool definitions
        system_prompt: System prompt to use
        temperature: Generation temperature
        min_yield: Minimum acceptable yield rate (default 10%)
        save_all: If True, save all records (not just successful flips)
        verbose: Print progress
    
    Returns:
        (ds_samples, stats) tuple
    """
    ds_samples = []
    all_samples = []  # For debugging
    
    stats = {
        "total": 0,
        "successful_flips": 0,
        "correct_behavior": 0,
        "no_tool_call": 0,
        "other_tool": 0,
        "format_errors": 0,
    }
    
    iterator = tqdm(b4_records, desc="Building Ds MVP") if verbose else b4_records
    
    for record in iterator:
        stats["total"] += 1
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["combined_query"]},
        ]
        
        # Generate response
        try:
            response = generate_with_tools(
                model, tokenizer, messages, tools,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Generation error for {record['id']}: {e}")
            stats["format_errors"] += 1
            continue
        
        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None
        expected_tool = record["expected_tool"]
        simulated_tool = record["simulated_tool"]
        
        # Determine outcome
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_flip_success = False
        elif observed_tool == simulated_tool:
            stats["successful_flips"] += 1
            is_flip_success = True
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_flip_success = False
        else:
            stats["other_tool"] += 1
            is_flip_success = False
        
        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
            if verbose and stats["format_errors"] <= 5:
                logger.warning(f"Format error: {format_error} in {record['id']}")
        
        # Build sample
        sample = {
            "id": record["id"],
            "messages": messages,
            "tools": "b4_standard_v1",  # Reference to frozen schema
            "assistant_raw": response,
            "tool_calls_structured": [tool_call] if tool_call else [],
            "labels": {
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "simulated_tool": simulated_tool,
                "is_flip_success": is_flip_success,
            },
            "metadata": {
                "split": "train",
                "source": "b4",
                "benign_query": record["benign_query"],
                "malicious_injection": record["malicious_injection"],
                "format_valid": is_valid,
                "format_error": format_error if not is_valid else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **record.get("metadata", {}),
            },
        }
        
        all_samples.append(sample)
        
        # ONLY include if attack succeeded
        if is_flip_success:
            ds_samples.append(sample)
    
    # Compute yield rate
    yield_rate = stats["successful_flips"] / stats["total"] if stats["total"] > 0 else 0
    stats["yield_rate"] = yield_rate
    
    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DS MVP BUILD STATS")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total']}")
    logger.info(f"Successful flips (Ds):   {stats['successful_flips']} ({yield_rate:.1%})")
    logger.info(f"Correct behavior:        {stats['correct_behavior']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Other tool:              {stats['other_tool']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Yield rate:              {yield_rate:.1%}")
    logger.info(f"Min yield threshold:     {min_yield:.1%}")
    logger.info("=" * 60)
    
    # Check yield threshold
    if yield_rate < min_yield:
        error_msg = (
            f"Ds yield too low ({yield_rate:.1%} < {min_yield:.1%}). "
            f"Cannot train tool-flip breaker from model that rarely flips. "
            f"Consider: higher temperature, different generator model, or attack prompt tuning."
        )
        logger.error(error_msg)
        stats["error"] = error_msg
    
    return (all_samples if save_all else ds_samples), stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate MVP Circuit Breaker Set (Ds) with behavioral filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data paths
    parser.add_argument(
        "--b4-data",
        type=Path,
        default=BASE_DIR / "data" / "fujitsu" / "orchestrator_attacks_combined_deduplicated.jsonl",
        help="Path to Fujitsu B4 data",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=BASE_DIR / "configs" / "tool_schemas" / "b4_standard_v1.json",
        help="Path to frozen tool schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "cb_mvp" / "ds_stage1.jsonl",
        help="Output path for Ds",
    )
    parser.add_argument(
        "--output-ids",
        type=Path,
        default=None,
        help="Output path for training IDs (for held-out split)",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype",
    )
    
    # Generation
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (higher = more varied outputs)",
    )
    parser.add_argument(
        "--min-yield",
        type=float,
        default=0.10,
        help="Minimum acceptable yield rate (default: 10%%)",
    )
    
    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all records, not just successful flips (for debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show stats without generating",
    )
    parser.add_argument(
        "--fail-on-low-yield",
        action="store_true",
        help="Exit with code 1 if yield < min-yield",
    )
    
    args = parser.parse_args()
    
    # Load tool schema
    if not args.tool_schema.exists():
        logger.error(f"Tool schema not found: {args.tool_schema}")
        return 1
    
    schema = load_tool_schema(args.tool_schema)
    tools = get_tools_list(schema)
    system_prompt = get_system_prompt(schema)
    
    # Load B4 data
    if not args.b4_data.exists():
        logger.error(f"B4 data not found: {args.b4_data}")
        return 1
    
    logger.info(f"Loading B4 data from {args.b4_data}...")
    b4_records = list(load_fujitsu_b4(args.b4_data, limit=args.limit))
    logger.info(f"Loaded {len(b4_records)} B4 records")
    
    if args.dry_run:
        logger.info("DRY RUN - showing sample records:")
        for record in b4_records[:3]:
            logger.info(f"  {record['id']}")
            logger.info(f"    expected: {record['expected_tool']}, simulated: {record['simulated_tool']}")
            logger.info(f"    query: {record['combined_query'][:80]}...")
        logger.info(f"Would write to: {args.output}")
        return 0
    
    # Load model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        torch_dtype=dtype_map[args.dtype],
    )
    
    # Build Ds MVP
    ds_samples, stats = build_ds_mvp(
        b4_records=b4_records,
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        system_prompt=system_prompt,
        temperature=args.temperature,
        min_yield=args.min_yield,
        save_all=args.save_all,
    )
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in ds_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(ds_samples)} samples to {args.output}")
    
    # Write IDs file
    if args.output_ids or True:  # Always write IDs
        ids_path = args.output_ids or args.output.with_suffix(".ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            for sample in ds_samples:
                f.write(sample["id"] + "\n")
        logger.info(f"Wrote IDs to {ids_path}")
    
    # Write stats
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote stats to {stats_path}")
    
    # Check yield threshold
    if args.fail_on_low_yield and stats.get("yield_rate", 0) < args.min_yield:
        logger.error("Yield rate below threshold - exiting with code 1")
        return 1
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
