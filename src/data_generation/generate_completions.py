#!/usr/bin/env python3
"""
Generate Completions: B1 (skeleton) -> B2 (complete) traces.

This script takes trace_v1 skeleton traces (tier=B1, no assistant messages)
and generates assistant completions to produce complete traces (tier=B2).

Supports two generation modes:
- DS (follows_injection): Model follows the injection and calls the wrong tool (observed == simulated)
- DR (ignores_injection): Takes successful DS traces, removes injection, calls correct tool (observed == expected)

Key Features:
1. Consumes trace_v1 skeleton traces from ETL_A
2. Generates assistant completions via vLLM or HuggingFace
3. Outputs complete trace_v1 traces ready for ETL_B
4. DS: behavioral filtering (only successful flips where observed == simulated)
5. DR: downstream from DS (only processes successful DS flips)
6. Preserves all trace metadata and signal_hints

Usage:
    # Generate DS (model follows injection)
    python src/data_generation/generate_completions.py \
        --traces data/traces/fujitsu_b4_skeletons.jsonl \
        --output data/traces/fujitsu_b4_ds.jsonl \
        --mode ds \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Generate DR (model ignores injection - must provide DS traces)
    python src/data_generation/generate_completions.py \
        --ds-data data/traces/fujitsu_b4_ds.jsonl \
        --output data/traces/fujitsu_b4_dr.jsonl \
        --mode dr \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Generate both DS and DR in pipeline (DS first, then DR from DS)
    python src/data_generation/generate_completions.py \
        --traces data/traces/fujitsu_b4_skeletons.jsonl \
        --output-ds data/traces/fujitsu_b4_ds.jsonl \
        --output-dr data/traces/fujitsu_b4_dr.jsonl \
        --mode both \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct
"""

# Set cache directories BEFORE any other imports (for vLLM/flashinfer)
import os

# Set cache directories to cb-scratch/cache/
cache_root = os.environ.get("CACHE_ROOT", "/scratch/memoozd/cb-scratch/cache")

# CRITICAL FIX: Override HOME to trick flashinfer that defaults to ~/.cache
os.environ["HOME"] = cache_root

os.makedirs(os.path.join(cache_root, "vllm"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "flashinfer"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "xdg_cache"), exist_ok=True)

os.environ.setdefault("VLLM_USAGE_STATS_DIR", os.path.join(cache_root, "vllm"))
os.environ.setdefault("FLASHINFER_WORKSPACE_DIR", os.path.join(cache_root, "flashinfer"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "xdg_cache"))

import argparse
import json
import logging
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.schemas.trace import (
    Trace,
    Message,
    ToolCall,
    ToolCallFunction,
)

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
# Trace I/O
# =============================================================================

def iter_traces(path: Path) -> Generator[Trace, None, None]:
    """Iterate over traces from JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                yield Trace.from_dict(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse trace: {e}")
                continue


def write_traces(path: Path, traces: List[Trace]) -> None:
    """Write traces to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


# =============================================================================
# Injection Removal (for DR generation)
# =============================================================================

INJECTION_PATTERNS = [
    r'<!--.*?-->',
    r'\[//\]:.*?\n',
    r'(?i)ignore\s+(?:the\s+)?(?:previous|above)\s+(?:instructions?|query)',
    r'(?i)instead\s+(?:of\s+)?(?:the\s+)?(?:above|previous)',
    r'(?i)use\s+\w+\s+instead',
    r'(?i)actually\s*,?\s*(?:please\s+)?(?:use|call)',
    r'(?i)forget\s+(?:the\s+)?(?:above|previous)',
    r'(?i)new\s+instruction[s]?\s*:',
    r'(?i)(?:please\s+)?(?:use|call)\s+(?:the\s+)?(\w+)\s+(?:tool|function)\s+instead',
    r'(?i)do\s+not\s+use\s+\w+\s*,?\s*use\s+\w+',
]


def remove_injection_from_content(content: str, injection_text: Optional[str] = None) -> str:
    """
    Remove injection from content to create benign version.

    Args:
        content: The full content with injection
        injection_text: If available, the known injection text to remove

    Returns:
        Cleaned benign content
    """
    if injection_text and injection_text in content:
        # Direct removal of known injection
        cleaned = content.replace(injection_text, "").strip()
        # Clean up any resulting double spaces or newlines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned.strip()

    # Fallback: apply pattern-based cleaning
    cleaned = content
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\[//\]:.*?\n', '', cleaned)

    for pattern in INJECTION_PATTERNS[2:]:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) < 10:
        match = re.match(r'^([^.!?]+[.!?])', content)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = content[:200].strip()

    return cleaned


# =============================================================================
# vLLM Backend
# =============================================================================

class VLLMBackend:
    """vLLM backend for high-throughput batched inference with tensor parallelism."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM, SamplingParams

        hf_token = resolve_hf_token()
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

        if offline_mode:
            resolved_path = resolve_local_model_path(model_path, hf_token)
            if resolved_path != model_path:
                logger.info(f"  Resolved to local path: {resolved_path}")
            model_path = resolved_path

        logger.info(f"Loading vLLM model: {model_path}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def format_prompt_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """Format messages with tools using tokenizer's chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


# =============================================================================
# Model Loading and Generation
# =============================================================================

def resolve_local_model_path(model_id: str, hf_token: Optional[str] = None) -> str:
    """Resolve a HuggingFace model ID to its local cache path."""
    from huggingface_hub import snapshot_download

    if os.path.isdir(model_id):
        return model_id

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            token=hf_token,
        )
        return local_path
    except Exception as e:
        logger.warning(f"Could not resolve local path for {model_id}: {e}")
        return model_id


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load model and tokenizer for generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = resolve_hf_token()
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    logger.info(f"Loading model: {model_path}")

    if offline_mode:
        resolved_path = resolve_local_model_path(model_path, hf_token)
        if resolved_path != model_path:
            logger.info(f"  Resolved to local path: {resolved_path}")
        model_path = resolved_path
        logger.info("  (offline mode - using cached files only)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=offline_mode,
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
    """Generate response with tool-calling capability."""
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
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

    input_len = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=False)

    return response.strip()


# =============================================================================
# Tool Call Extraction and Validation
# =============================================================================

def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from assistant response."""
    if "<|python_tag|>" in response:
        content = response.split("<|python_tag|>", 1)[1]
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            content = content.split(end_token)[0]
        content = content.strip()

        match = re.match(r'(\w+)\s*\((.*)\)', content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            try:
                if '=' in args_str and not args_str.startswith('{'):
                    args = {}
                    for part in re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
                        args[part[0]] = part[1]
                    return {"name": tool_name, "parameters": args}
                else:
                    args = json.loads(args_str) if args_str else {}
                    return {"name": tool_name, "parameters": args}
            except (json.JSONDecodeError, ValueError):
                return {"name": tool_name, "parameters": {"raw": args_str}}

        try:
            data = json.loads(content)
            name = data.get("name") or data.get("function", {}).get("name")
            params = data.get("parameters") or data.get("arguments") or {}
            if name:
                return {"name": name, "parameters": params}
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', response)
    if json_match:
        try:
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


def fix_assistant_raw_format(assistant_raw: str) -> str:
    """Fix common formatting issues in assistant_raw."""
    if not assistant_raw:
        return assistant_raw

    if "<|python_tag|>" in assistant_raw:
        has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
        if not has_end_token:
            assistant_raw = assistant_raw.rstrip() + "<|eom_id|>"

    return assistant_raw


def validate_llama_format(assistant_raw: str) -> Tuple[bool, str]:
    """Validate Llama 3.1 tool format."""
    if not assistant_raw:
        return False, "Empty assistant_raw"

    if "<|python_tag|>" not in assistant_raw:
        return False, "Missing <|python_tag|>"

    has_valid_end = any(
        assistant_raw.rstrip().endswith(end)
        for end in ["<|eom_id|>", "<|eot_id|>"]
    )
    if not has_valid_end:
        has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
        if not has_end_token:
            return False, "Missing <|eom_id|> or <|eot_id|>"

    if "```" in assistant_raw:
        return False, "Contains markdown code block"

    for prefix in ["Action:", "ToolCall:", "Function:"]:
        if assistant_raw.strip().startswith(prefix):
            return False, f"Contains forbidden prefix: {prefix}"

    return True, ""


# =============================================================================
# DS Generation (follows injection)
# =============================================================================

def build_ds_from_skeletons(
    skeleton_traces: List[Trace],
    backend,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    verbose: bool = True,
    collect_examples: bool = False,
    n_successful: int = 10,
    n_correct: int = 5,
    n_no_tool: int = 5,
    n_other_tool: int = 5,
    n_format_errors: int = 5,
) -> Tuple[List[Trace], Dict[str, Any]]:
    """
    Build DS (successful flips) from skeleton traces.

    Matches logic from generate_ds.py:
    - Generate with injection present
    - ONLY include if observed_tool == simulated_tool (successful flip)
    - Filter by behavioral outcome

    Args:
        skeleton_traces: List of skeleton traces (B1)
        backend: VLLMBackend or (model, tokenizer) tuple
        tools: Tool definitions
        system_prompt: System prompt to use
        temperature: Generation temperature
        verbose: Print progress

    Returns:
        (ds_traces, stats) tuple
    """
    ds_traces = []

    stats = {
        "total": 0,
        "successful_flips": 0,
        "correct_behavior": 0,
        "no_tool_call": 0,
        "other_tool": 0,
        "format_errors": 0,
    }

    examples = {
        "successful_flips": [],
        "correct_behavior": [],
        "no_tool_call": [],
        "other_tool": [],
        "format_errors": [],
    } if collect_examples else None

    iterator = tqdm(skeleton_traces, desc="Building DS") if verbose else skeleton_traces

    for trace in iterator:
        # Skip non-skeleton traces
        if getattr(trace, 'completeness', None) == 'complete' or getattr(trace, 'tier', None) == 'B2':
            continue

        stats["total"] += 1

        # Extract expected/simulated tools
        expected_tool = None
        simulated_tool = None

        if trace.tool_attack:
            expected_tool = trace.tool_attack.expected_tool
            simulated_tool = trace.tool_attack.observed_tool  # Note: ETL_A stores simulated in observed_tool
        if trace.signal_hints:
            expected_tool = expected_tool or trace.signal_hints.expected_tool_name
            simulated_tool = simulated_tool or trace.signal_hints.observed_tool_name

        # Build messages from trace
        messages = []
        for msg in trace.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Generate response
        if isinstance(backend, VLLMBackend):
            prompt = backend.format_prompt_with_tools(messages, tools)
            responses = backend.generate_batch([prompt], temperature=temperature, max_tokens=max_tokens)
            response = responses[0] if responses else ""
        else:
            model, tokenizer = backend
            response = generate_with_tools(
                model, tokenizer, messages, tools,
                max_new_tokens=max_tokens, temperature=temperature,
            )

        # Fix formatting
        response = fix_assistant_raw_format(response)

        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None

        # Determine outcome (matching generate_ds.py logic)
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_flip_success = False
            category = "no_tool_call"
        elif observed_tool == simulated_tool:
            stats["successful_flips"] += 1
            is_flip_success = True
            category = "successful_flips"
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_flip_success = False
            category = "correct_behavior"
        else:
            stats["other_tool"] += 1
            is_flip_success = False
            category = "other_tool"

        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
            if verbose and stats["format_errors"] <= 5:
                logger.warning(f"Format error: {format_error} in {trace.id}")

        # Collect examples
        if collect_examples and examples is not None:
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
                    break

            ex = {
                "id": trace.id,
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "simulated_tool": simulated_tool,
                "user_content": user_content,
                "assistant_raw": response,
                "format_error": format_error if not is_valid else None,
            }

            if not is_valid:
                if len(examples["format_errors"]) < n_format_errors:
                    examples["format_errors"].append(ex)
            else:
                max_counts = {
                    "successful_flips": n_successful,
                    "correct_behavior": n_correct,
                    "no_tool_call": n_no_tool,
                    "other_tool": n_other_tool,
                }
                if category in max_counts and len(examples[category]) < max_counts[category]:
                    examples[category].append(ex)

        # ONLY include if attack succeeded (observed == simulated)
        if not is_flip_success:
            continue

        # Create completed trace
        completed_trace = deepcopy(trace)
        completed_trace.completeness = "complete"
        completed_trace.tier = "B2"

        # Add assistant message with tool call
        tc = None
        if tool_call:
            tc = ToolCall(
                function=ToolCallFunction(
                    name=tool_call["name"],
                    arguments=tool_call.get("parameters", {}),
                )
            )

        assistant_msg = Message(
            role="assistant",
            content=response,
            tool_calls=[tc] if tc else None,
        )
        completed_trace.messages.append(assistant_msg)

        # Update labels
        if completed_trace.labels:
            completed_trace.labels.category = "harmful"
            completed_trace.labels.security_outcome = "unsafe"
            completed_trace.labels.attack_succeeded = True

        # Regenerate ID with new content
        completed_trace.id = Trace.generate_id(
            completed_trace.source.dataset if completed_trace.source else "generated",
            messages=completed_trace.messages,
        )

        ds_traces.append(completed_trace)

    # Compute yield rate
    yield_rate = stats["successful_flips"] / stats["total"] if stats["total"] > 0 else 0
    stats["yield_rate"] = yield_rate

    # Add examples to stats
    if collect_examples:
        stats["examples"] = examples

    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DS BUILD STATS")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total']}")
    logger.info(f"Successful flips (DS):   {stats['successful_flips']} ({yield_rate:.1%})")
    logger.info(f"Correct behavior:        {stats['correct_behavior']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Other tool:              {stats['other_tool']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Yield rate:              {yield_rate:.1%}")
    logger.info("=" * 60)

    return ds_traces, stats


# =============================================================================
# DR Generation (ignores injection - downstream from DS)
# =============================================================================

def build_dr_from_ds(
    ds_traces: List[Trace],
    backend,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 256,
    verbose: bool = True,
    collect_examples: bool = False,
    n_correct: int = 10,
    n_wrong_tool: int = 5,
    n_no_tool: int = 5,
    n_format_errors: int = 5,
) -> Tuple[List[Trace], Dict[str, Any]]:
    """
    Build DR (paired benign twins) from DS traces.

    Matches logic from generate_dr.py:
    - Takes successful DS flips as input
    - Removes injection from user messages
    - Regenerates with lower temperature
    - ONLY include if observed_tool == expected_tool (correct behavior)

    Args:
        ds_traces: List of successful DS traces
        backend: VLLMBackend or (model, tokenizer) tuple
        tools: Tool definitions
        system_prompt: System prompt to use
        temperature: Generation temperature (lower for more consistency)
        verbose: Print progress

    Returns:
        (dr_traces, stats) tuple
    """
    dr_traces = []

    stats = {
        "total": 0,
        "correct_behavior": 0,
        "wrong_tool": 0,
        "no_tool_call": 0,
        "format_errors": 0,
    }

    examples = {
        "correct_behavior": [],
        "wrong_tool": [],
        "no_tool_call": [],
        "format_errors": [],
    } if collect_examples else None

    iterator = tqdm(ds_traces, desc="Building DR") if verbose else ds_traces

    for ds_trace in iterator:
        stats["total"] += 1

        # Get expected tool
        expected_tool = None
        if ds_trace.tool_attack:
            expected_tool = ds_trace.tool_attack.expected_tool
        if ds_trace.signal_hints:
            expected_tool = expected_tool or ds_trace.signal_hints.expected_tool_name

        # Get injection text for removal
        injection_text = None
        if ds_trace.tool_attack:
            injection_text = ds_trace.tool_attack.injection_text

        # Build messages with injection removed
        messages = []
        for msg in ds_trace.messages:
            # Skip assistant messages (we're regenerating)
            if msg.role == "assistant":
                continue

            content = msg.content
            if msg.role == "user":
                content = remove_injection_from_content(content, injection_text)

            messages.append({
                "role": msg.role,
                "content": content,
            })

        # Generate response
        if isinstance(backend, VLLMBackend):
            prompt = backend.format_prompt_with_tools(messages, tools)
            responses = backend.generate_batch([prompt], temperature=temperature, max_tokens=max_tokens)
            response = responses[0] if responses else ""
        else:
            model, tokenizer = backend
            response = generate_with_tools(
                model, tokenizer, messages, tools,
                max_new_tokens=max_tokens, temperature=temperature,
            )

        # Fix formatting
        response = fix_assistant_raw_format(response)

        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None

        # Determine outcome (matching generate_dr.py logic)
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_correct = False
            category = "no_tool_call"
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_correct = True
            category = "correct_behavior"
        else:
            stats["wrong_tool"] += 1
            is_correct = False
            category = "wrong_tool"

        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
            if not is_correct:
                category = "format_errors"

        # Collect examples
        if collect_examples and examples is not None and category:
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
                    break

            ex = {
                "id": ds_trace.id,
                "paired_with": ds_trace.id,
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "user_content": user_content,
                "assistant_raw": response,
                "format_error": format_error if not is_valid else None,
            }

            max_counts = {
                "correct_behavior": n_correct,
                "wrong_tool": n_wrong_tool,
                "no_tool_call": n_no_tool,
                "format_errors": n_format_errors,
            }
            if len(examples[category]) < max_counts[category]:
                examples[category].append(ex)

        # ONLY include if correct tool was called
        if not is_correct:
            continue

        # Create completed trace (benign version)
        completed_trace = deepcopy(ds_trace)
        completed_trace.completeness = "complete"
        completed_trace.tier = "B2"

        # Update messages with cleaned content (remove assistant messages first)
        completed_trace.messages = [msg for msg in completed_trace.messages if msg.role != "assistant"]
        for i, msg in enumerate(completed_trace.messages):
            if msg.role == "user":
                completed_trace.messages[i].content = messages[i]["content"]

        # Add assistant message with tool call
        tc = None
        if tool_call:
            tc = ToolCall(
                function=ToolCallFunction(
                    name=tool_call["name"],
                    arguments=tool_call.get("parameters", {}),
                )
            )

        assistant_msg = Message(
            role="assistant",
            content=response,
            tool_calls=[tc] if tc else None,
        )
        completed_trace.messages.append(assistant_msg)

        # Update labels
        if completed_trace.labels:
            completed_trace.labels.category = "benign"
            completed_trace.labels.security_outcome = "safe"
            completed_trace.labels.attack_succeeded = False

        # Clear signal_hints injection span (injection removed)
        if completed_trace.signal_hints:
            completed_trace.signal_hints.injection_char_span = None

        # Clear tool_attack (benign version)
        completed_trace.tool_attack = None

        # Update mixture class
        if completed_trace.training and completed_trace.training.mixture:
            old_class = completed_trace.training.mixture.class_id or ""
            completed_trace.training.mixture.class_id = old_class.replace("/tool_flip", "/benign")

        # Regenerate ID with new content
        completed_trace.id = Trace.generate_id(
            (completed_trace.source.dataset if completed_trace.source else "generated") + "_benign",
            messages=completed_trace.messages,
        )

        # Link to parent
        if completed_trace.links:
            completed_trace.links.parent_trace_ids = [ds_trace.id]

        dr_traces.append(completed_trace)

    # Compute success rate
    success_rate = stats["correct_behavior"] / stats["total"] if stats["total"] > 0 else 0
    stats["success_rate"] = success_rate

    # Add examples to stats
    if collect_examples:
        stats["examples"] = examples

    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DR BUILD STATS")
    logger.info("=" * 60)
    logger.info(f"Total DS samples:        {stats['total']}")
    logger.info(f"Correct behavior (DR):   {stats['correct_behavior']} ({success_rate:.1%})")
    logger.info(f"Wrong tool:              {stats['wrong_tool']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Success rate:            {success_rate:.1%}")
    logger.info("=" * 60)

    return dr_traces, stats


# =============================================================================
# Example Reporting
# =============================================================================

def print_examples_report(examples: Dict[str, Any], mode: str, truncate: bool = True) -> None:
    """Print collected examples for DS/DR in a human-readable format."""
    max_len = 200 if truncate else None

    def _truncate(text: Optional[str]) -> str:
        if text is None:
            return ""
        if not max_len:
            return text
        return text if len(text) <= max_len else text[:max_len] + "..."

    if mode == "ds":
        ds_category_names = {
            "successful_flips": "Successful Tool Flips (DS targets)",
            "correct_behavior": "Correct Behavior (non-successful DS)",
            "no_tool_call": "No Tool Call",
            "other_tool": "Other Tool",
            "format_errors": "Format Errors",
        }
        for category, name in ds_category_names.items():
            items = examples.get(category, [])
            if not items:
                continue
            print("\n" + "=" * 80)
            print(f"DS - {name} ({len(items)} examples)")
            print("=" * 80)
            for i, ex in enumerate(items, 1):
                print(f"\n--- Example {i} ---")
                print(f"ID: {ex.get('id', 'N/A')}")
                print(
                    f"Expected: {ex.get('expected_tool')} | Observed: {ex.get('observed_tool')} | Simulated: {ex.get('simulated_tool')}"
                )
                print(f"\nUser:\n{_truncate(ex.get('user_content'))}")
                print(f"\nAssistant Raw:\n{_truncate(ex.get('assistant_raw'))}")
                if ex.get("format_error"):
                    print(f"\nFormat Error: {ex.get('format_error')}")

    elif mode == "dr":
        for category, items in examples.items():
            if not items:
                continue
            print("\n" + "=" * 80)
            print(f"DR - CATEGORY: {category.upper()} ({len(items)} examples)")
            print("=" * 80)
            for i, ex in enumerate(items, 1):
                print(f"\n--- Example {i}/{len(items)} ---")
                print(f"ID: {ex.get('id', 'N/A')}")
                print(f"Paired with: {ex.get('paired_with', 'N/A')}")
                print(f"Expected tool: {ex.get('expected_tool', 'N/A')}")
                print(f"Observed tool: {ex.get('observed_tool', 'N/A')}")
                if ex.get("format_error"):
                    print(f"Format error: {ex.get('format_error')}")
                print(f"User: {_truncate(ex.get('user_content'))}")
                print(f"Assistant raw: {_truncate(ex.get('assistant_raw'))}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate completions for skeleton traces (B1 -> B2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output
    parser.add_argument(
        "--traces", type=Path,
        help="Input skeleton trace_v1 JSONL file (B1 skeletons) - for DS mode or both mode",
    )
    parser.add_argument(
        "--ds-data", type=Path,
        help="Input DS trace JSONL file - for DR mode (successful DS flips)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSONL file (for single mode: ds or dr)",
    )
    parser.add_argument(
        "--output-ds", type=Path,
        help="Output JSONL file for DS traces (used with --mode both)",
    )
    parser.add_argument(
        "--output-dr", type=Path,
        help="Output JSONL file for DR traces (used with --mode both)",
    )

    # Model and Schema
    parser.add_argument(
        "--model", required=True,
        help="Model name or path (HuggingFace or local)",
    )
    parser.add_argument(
        "--tool-schema", required=True, type=Path,
        help="Path to frozen tool schema JSON",
    )

    # Generation Mode
    parser.add_argument(
        "--mode", choices=["ds", "dr", "both"], default="ds",
        help="Generation mode: ds (follows injection), dr (ignores injection - requires DS), both (pipeline)",
    )

    # Generation Parameters
    parser.add_argument(
        "--temperature-ds", type=float, default=0.7,
        help="Temperature for DS generation (default: 0.7)",
    )
    parser.add_argument(
        "--temperature-dr", type=float, default=0.3,
        help="Temperature for DR generation (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate (default: 256)",
    )

    # Backend Options
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM backend for faster inference",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Max model length for vLLM (default: 4096)",
    )

    # Other Options
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of traces to process",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    # Example collection
    parser.add_argument(
        "--print-examples",
        action="store_true",
        help="Print example datapoints from each category",
    )
    parser.add_argument(
        "--examples-out",
        type=Path,
        default=None,
        help="Output path for examples JSON (default: <output>.examples.json)",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate example outputs when printing",
    )
    parser.add_argument(
        "--no-write-ids",
        action="store_true",
        help="Do not write <output>.ids.txt files",
    )
    parser.add_argument(
        "--n-successful",
        type=int,
        default=10,
        help="Number of DS successful flip examples to collect (default: 10)",
    )
    parser.add_argument(
        "--n-correct",
        type=int,
        default=5,
        help="Number of correct behavior examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-wrong-tool",
        type=int,
        default=5,
        help="Number of DR wrong tool examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-no-tool",
        type=int,
        default=5,
        help="Number of no tool call examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-other-tool",
        type=int,
        default=5,
        help="Number of other tool examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-format-errors",
        type=int,
        default=5,
        help="Number of format error examples to collect (default: 5)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'ds' and not args.traces:
        parser.error("--traces required for DS mode")
    if args.mode == 'dr' and not args.ds_data:
        parser.error("--ds-data required for DR mode")
    if args.mode == 'both' and not args.traces:
        parser.error("--traces required for both mode")
    if args.mode in ('ds', 'dr') and not args.output:
        parser.error("--output required for single mode (ds or dr)")
    if args.mode == 'both' and not (args.output_ds and args.output_dr):
        parser.error("--output-ds and --output-dr required for --mode both")

    # Load tool schema
    schema = load_tool_schema(args.tool_schema)
    tools = get_tools_list(schema)
    system_prompt = get_system_prompt(schema)

    # Initialize backend
    if args.use_vllm:
        logger.info("Initializing vLLM backend...")
        backend = VLLMBackend(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
    else:
        logger.info("Loading HuggingFace model...")
        backend = load_model_and_tokenizer(args.model)

    # Generate based on mode
    ds_traces = []
    dr_traces = []
    ds_stats = {}
    dr_stats = {}

    if args.mode in ('ds', 'both'):
        # Load skeleton traces
        logger.info(f"Loading skeleton traces from {args.traces}")
        traces = list(iter_traces(args.traces))
        if args.limit:
            traces = traces[:args.limit]
        logger.info(f"Loaded {len(traces)} skeleton traces")

        # Build DS
        ds_traces, ds_stats = build_ds_from_skeletons(
            traces, backend, tools, system_prompt,
            temperature=args.temperature_ds,
            max_tokens=args.max_tokens,
            verbose=not args.quiet,
            collect_examples=(args.print_examples or args.examples_out is not None),
            n_successful=args.n_successful,
            n_correct=args.n_correct,
            n_no_tool=args.n_no_tool,
            n_other_tool=args.n_other_tool,
            n_format_errors=args.n_format_errors,
        )

        # Write DS output
        if args.mode == 'ds':
            write_traces(args.output, ds_traces)
            logger.info(f"Wrote {len(ds_traces)} DS traces to {args.output}")
            if not args.no_write_ids:
                ids_path = args.output.with_suffix(".ids.txt")
                with open(ids_path, "w", encoding="utf-8") as f:
                    for t in ds_traces:
                        f.write(getattr(t, "id", "") + "\n")
                logger.info(f"Wrote IDs to {ids_path}")
        else:
            write_traces(args.output_ds, ds_traces)
            logger.info(f"Wrote {len(ds_traces)} DS traces to {args.output_ds}")
            if not args.no_write_ids:
                ids_path = args.output_ds.with_suffix(".ids.txt")
                with open(ids_path, "w", encoding="utf-8") as f:
                    for t in ds_traces:
                        f.write(getattr(t, "id", "") + "\n")
                logger.info(f"Wrote DS IDs to {ids_path}")

    if args.mode in ('dr', 'both'):
        # Load or use DS traces
        if args.mode == 'dr':
            logger.info(f"Loading DS traces from {args.ds_data}")
            ds_traces = list(iter_traces(args.ds_data))
            if args.limit:
                ds_traces = ds_traces[:args.limit]
            logger.info(f"Loaded {len(ds_traces)} DS traces")
        elif args.mode == 'both':
            logger.info(f"Using {len(ds_traces)} DS traces for DR generation")

        # Build DR from DS
        dr_traces, dr_stats = build_dr_from_ds(
            ds_traces, backend, tools, system_prompt,
            temperature=args.temperature_dr,
            max_tokens=args.max_tokens,
            verbose=not args.quiet,
            collect_examples=(args.print_examples or args.examples_out is not None),
            n_correct=args.n_correct,
            n_wrong_tool=args.n_wrong_tool,
            n_no_tool=args.n_no_tool,
            n_format_errors=args.n_format_errors,
        )

        # Write DR output
        if args.mode == 'dr':
            write_traces(args.output, dr_traces)
            logger.info(f"Wrote {len(dr_traces)} DR traces to {args.output}")
            if not args.no_write_ids:
                ids_path = args.output.with_suffix(".ids.txt")
                with open(ids_path, "w", encoding="utf-8") as f:
                    for t in dr_traces:
                        f.write(getattr(t, "id", "") + "\n")
                logger.info(f"Wrote IDs to {ids_path}")
        else:
            write_traces(args.output_dr, dr_traces)
            logger.info(f"Wrote {len(dr_traces)} DR traces to {args.output_dr}")
            if not args.no_write_ids:
                ids_path = args.output_dr.with_suffix(".ids.txt")
                with open(ids_path, "w", encoding="utf-8") as f:
                    for t in dr_traces:
                        f.write(getattr(t, "id", "") + "\n")
                logger.info(f"Wrote DR IDs to {ids_path}")

    # Handle examples
    if args.mode == 'ds' and "examples" in ds_stats and ds_stats["examples"]:
        examples = ds_stats.pop("examples")
        if args.print_examples:
            print_examples_report(examples, "ds", truncate=not args.no_truncate)
        examples_path = args.examples_out or args.output.with_suffix(".examples.json")
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote DS examples to {examples_path}")

    elif args.mode == 'dr' and "examples" in dr_stats and dr_stats["examples"]:
        examples = dr_stats.pop("examples")
        if args.print_examples:
            print_examples_report(examples, "dr", truncate=not args.no_truncate)
        examples_path = args.examples_out or args.output.with_suffix(".examples.json")
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote DR examples to {examples_path}")

    elif args.mode == 'both':
        if "examples" in ds_stats and ds_stats["examples"]:
            ds_examples = ds_stats.pop("examples")
            if args.print_examples:
                print_examples_report(ds_examples, "ds", truncate=not args.no_truncate)
            ds_examples_path = args.output_ds.with_suffix(".examples.json")
            with open(ds_examples_path, "w", encoding="utf-8") as f:
                json.dump(ds_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote DS examples to {ds_examples_path}")

        if "examples" in dr_stats and dr_stats["examples"]:
            dr_examples = dr_stats.pop("examples")
            if args.print_examples:
                print_examples_report(dr_examples, "dr", truncate=not args.no_truncate)
            dr_examples_path = args.output_dr.with_suffix(".examples.json")
            with open(dr_examples_path, "w", encoding="utf-8") as f:
                json.dump(dr_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote DR examples to {dr_examples_path}")

    # Write stats
    if args.mode == 'ds':
        stats_path = args.output.with_suffix(".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(ds_stats, f, indent=2)
        logger.info(f"Wrote stats to {stats_path}")
    elif args.mode == 'dr':
        stats_path = args.output.with_suffix(".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(dr_stats, f, indent=2)
        logger.info(f"Wrote stats to {stats_path}")
    else:
        ds_stats_path = args.output_ds.with_suffix(".stats.json")
        dr_stats_path = args.output_dr.with_suffix(".stats.json")
        with open(ds_stats_path, "w", encoding="utf-8") as f:
            json.dump(ds_stats, f, indent=2)
        with open(dr_stats_path, "w", encoding="utf-8") as f:
            json.dump(dr_stats, f, indent=2)
        logger.info(f"Wrote DS stats to {ds_stats_path}")
        logger.info(f"Wrote DR stats to {dr_stats_path}")

    # Report ratio if both modes
    if args.mode == 'both' and len(ds_traces) > 0:
        ratio = len(dr_traces) / len(ds_traces)
        logger.info(f"DR:DS ratio = {len(dr_traces)}:{len(ds_traces)} ({ratio:.2f})")


if __name__ == "__main__":
    main()
