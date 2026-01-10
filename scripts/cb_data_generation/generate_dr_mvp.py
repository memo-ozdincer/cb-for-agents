#!/usr/bin/env python3
"""
MVP Retain Set (Dr) Generator - Stage 1

Generate the Retain Set as PAIRED BENIGN TWINS of Ds samples.
For Stage 1, we skip UltraChat/XSTest/cross-domain mixing and only create
benign twins that preserve correct tool selection behavior.

Key Stage 1 Principles:
1. For each Ds sample, create a benign twin (remove injection)
2. Only include if observed_tool == expected_tool (correct behavior)
3. Target 1:1 ratio with Ds (simple balancing)
4. Use same frozen tool schema as Ds (b4_standard_v1)
5. Validate Llama 3.1 format compliance

Usage:
    python scripts/cb_data_generation/generate_dr_mvp.py \
        --ds-data data/cb_mvp/ds_stage1.jsonl \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/cb_mvp/dr_stage1.jsonl
"""

# Set cache directories BEFORE any other imports (for vLLM/flashinfer)
import os
if "SCRATCH_DIR" not in os.environ:
    scratch_dir = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    cache_root = os.path.join(scratch_dir, "cb_cache")
else:
    cache_root = os.path.join(os.environ["SCRATCH_DIR"], "cb_cache")

os.makedirs(os.path.join(cache_root, "vllm"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "flashinfer"), exist_ok=True)
os.environ.setdefault("VLLM_USAGE_STATS_DIR", os.path.join(cache_root, "vllm"))
os.environ.setdefault("FLASHINFER_WORKSPACE_DIR", os.path.join(cache_root, "flashinfer"))

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Tool Schema Loading (reuse from generate_ds_mvp)
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
# Data Loading
# =============================================================================

def load_ds_samples(path: Path) -> List[Dict[str, Any]]:
    """Load Ds samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


# =============================================================================
# Injection Removal
# =============================================================================

# Common injection patterns to remove
INJECTION_PATTERNS = [
    # HTML comments
    r'<!--.*?-->',
    # Markdown comments
    r'\[//\]:.*?\n',
    # Common injection phrases
    r'(?i)ignore\s+(?:the\s+)?(?:previous|above)\s+(?:instructions?|query)',
    r'(?i)instead\s+(?:of\s+)?(?:the\s+)?(?:above|previous)',
    r'(?i)use\s+\w+\s+instead',
    r'(?i)actually\s*,?\s*(?:please\s+)?(?:use|call)',
    r'(?i)forget\s+(?:the\s+)?(?:above|previous)',
    r'(?i)new\s+instruction[s]?\s*:',
    # Tool-specific redirects
    r'(?i)(?:please\s+)?(?:use|call)\s+(?:the\s+)?(\w+)\s+(?:tool|function)\s+instead',
    r'(?i)do\s+not\s+use\s+\w+\s*,?\s*use\s+\w+',
]


def remove_injection(prompt: str, benign_query: Optional[str] = None) -> str:
    """
    Remove injection from a prompt to create benign twin.
    
    Args:
        prompt: The full prompt with injection
        benign_query: If available, the known benign query
    
    Returns:
        Cleaned benign prompt
    """
    # If we have the benign query, use it directly
    if benign_query and benign_query.strip():
        return benign_query.strip()
    
    # Otherwise, try to extract/clean the prompt
    cleaned = prompt
    
    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    
    # Remove markdown comments
    cleaned = re.sub(r'\[//\]:.*?\n', '', cleaned)
    
    # Apply other patterns
    for pattern in INJECTION_PATTERNS[2:]:  # Skip first two (already handled)
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up whitespace
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()
    
    # If cleaning removed too much, fall back to original
    if len(cleaned) < 10:
        # Try to extract the first sentence/question
        match = re.match(r'^([^.!?]+[.!?])', prompt)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = prompt[:200].strip()  # Truncate as last resort
    
    return cleaned


# =============================================================================
# vLLM Backend (for 4x H100 tensor parallelism)
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
        
        # Resolve to local path if in offline mode
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
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.3,
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
# Model Loading and Generation (reuse from generate_ds_mvp)
# =============================================================================

def resolve_local_model_path(model_id: str, hf_token: Optional[str] = None) -> str:
    """
    Resolve a HuggingFace model ID to its local cache path using snapshot_download.
    
    When in offline mode, we need to pass the actual local path instead of
    a Hub model ID to avoid API calls during model_info() checks.
    """
    from huggingface_hub import snapshot_download
    
    # If it's already a local path, return as-is
    if os.path.isdir(model_id):
        return model_id
    
    # Use snapshot_download with local_files_only=True to get cached path
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
    
    # Check if we're in offline mode (compute nodes have no internet)
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    logger.info(f"Loading model: {model_path}")
    
    # In offline mode, resolve Hub ID to local cache path to avoid API calls
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
    temperature: float = 0.3,
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
# MVP Dr Builder
# =============================================================================

def build_dr_mvp(
    ds_samples: List[Dict[str, Any]],
    model,
    tokenizer,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.3,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Dr as paired benign twins of Ds samples.
    
    For each Ds sample:
    1. Extract benign query (remove injection)
    2. Generate with model at lower temperature
    3. ONLY include if observed_tool == expected_tool (correct behavior)
    
    Args:
        ds_samples: List of Ds samples
        model: The language model
        tokenizer: The tokenizer
        tools: Tool definitions
        system_prompt: System prompt to use
        temperature: Generation temperature (lower = more consistent)
        verbose: Print progress
    
    Returns:
        (dr_samples, stats) tuple
    """
    dr_samples = []
    
    stats = {
        "total": 0,
        "correct_behavior": 0,
        "wrong_tool": 0,
        "no_tool_call": 0,
        "format_errors": 0,
    }
    
    iterator = tqdm(ds_samples, desc="Building Dr MVP") if verbose else ds_samples
    
    for ds_sample in iterator:
        stats["total"] += 1
        
        # Get expected tool and benign query
        labels = ds_sample.get("labels", {})
        expected_tool = labels.get("expected_tool", "")
        
        # Try to get benign query from metadata
        metadata = ds_sample.get("metadata", {})
        benign_query = metadata.get("benign_query", "")
        
        # If no benign query, try to extract from messages
        if not benign_query:
            messages = ds_sample.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    benign_query = remove_injection(original_prompt)
                    break
        
        if not benign_query:
            logger.warning(f"Could not extract benign query for {ds_sample.get('id')}")
            stats["format_errors"] += 1
            continue
        
        # Build messages with benign query
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": benign_query},
        ]
        
        # Generate response
        try:
            response = generate_with_tools(
                model, tokenizer, messages, tools,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Generation error for {ds_sample.get('id')}: {e}")
            stats["format_errors"] += 1
            continue
        
        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None
        
        # Determine outcome
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_correct = False
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_correct = True
        else:
            stats["wrong_tool"] += 1
            is_correct = False
        
        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
        
        # ONLY include if correct tool was called
        if is_correct:
            sample = {
                "id": f"{ds_sample.get('id')}_benign",
                "messages": messages,
                "tools": "b4_standard_v1",
                "assistant_raw": response,
                "tool_calls_structured": [tool_call] if tool_call else [],
                "labels": {
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "is_flip_success": False,  # Correct behavior, not a flip
                },
                "metadata": {
                    "split": "retain",
                    "source": "b4",
                    "paired_with": ds_sample.get("id"),
                    "benign_query": benign_query,
                    "format_valid": is_valid,
                    "format_error": format_error if not is_valid else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            dr_samples.append(sample)
    
    # Compute success rate
    success_rate = stats["correct_behavior"] / stats["total"] if stats["total"] > 0 else 0
    stats["success_rate"] = success_rate
    
    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DR MVP BUILD STATS")
    logger.info("=" * 60)
    logger.info(f"Total Ds samples:        {stats['total']}")
    logger.info(f"Correct behavior (Dr):   {stats['correct_behavior']} ({success_rate:.1%})")
    logger.info(f"Wrong tool:              {stats['wrong_tool']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Success rate:            {success_rate:.1%}")
    logger.info("=" * 60)
    
    return dr_samples, stats


def build_dr_mvp_vllm(
    ds_samples: List[Dict[str, Any]],
    vllm_backend: "VLLMBackend",
    tools: List[Dict[str, Any]],
    system_prompt: str,
    batch_size: int = 32,
    temperature: float = 0.3,
    max_tokens: int = 256,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Dr using vLLM backend with batched generation.
    """
    dr_samples = []
    stats = {
        "total": 0,
        "correct_behavior": 0,
        "wrong_tool": 0,
        "no_tool_call": 0,
        "format_errors": 0,
    }
    
    # Prepare all prompts
    logger.info(f"Preparing {len(ds_samples)} prompts for batched Dr generation...")
    all_prompts = []
    sample_data = []
    
    for ds_sample in ds_samples:
        labels = ds_sample.get("labels", {})
        expected_tool = labels.get("expected_tool", "")
        metadata = ds_sample.get("metadata", {})
        benign_query = metadata.get("benign_query", "")
        
        if not benign_query:
            messages = ds_sample.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    benign_query = remove_injection(original_prompt)
                    break
        
        if not benign_query:
            continue
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": benign_query},
        ]
        prompt = vllm_backend.format_prompt_with_tools(messages, tools)
        all_prompts.append(prompt)
        sample_data.append({
            "ds_sample": ds_sample,
            "messages": messages,
            "benign_query": benign_query,
            "expected_tool": expected_tool,
        })
    
    # Generate in batches
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    all_responses = []
    
    logger.info(f"Generating responses in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating Dr batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start_idx:end_idx]
        
        try:
            batch_responses = vllm_backend.generate_batch(
                batch_prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_responses.extend(batch_responses)
        except Exception as e:
            logger.error(f"Batch {batch_idx} generation error: {e}")
            all_responses.extend([""] * len(batch_prompts))
    
    # Process results
    logger.info("Processing generated responses...")
    for data, response in tqdm(zip(sample_data, all_responses), total=len(sample_data), desc="Building Dr MVP"):
        stats["total"] += 1
        
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None
        expected_tool = data["expected_tool"]
        
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_correct = False
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_correct = True
        else:
            stats["wrong_tool"] += 1
            is_correct = False
        
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
        
        if is_correct:
            sample = {
                "id": f"{data['ds_sample'].get('id')}_benign",
                "messages": data["messages"],
                "tools": "b4_standard_v1",
                "assistant_raw": response,
                "tool_calls_structured": [tool_call] if tool_call else [],
                "labels": {
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "is_flip_success": False,
                },
                "metadata": {
                    "split": "retain",
                    "source": "b4",
                    "paired_with": data["ds_sample"].get("id"),
                    "benign_query": data["benign_query"],
                    "format_valid": is_valid,
                    "format_error": format_error if not is_valid else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            dr_samples.append(sample)
    
    success_rate = stats["correct_behavior"] / stats["total"] if stats["total"] > 0 else 0
    stats["success_rate"] = success_rate
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("DR MVP BUILD STATS (vLLM)")
    logger.info("=" * 60)
    logger.info(f"Total Ds samples:        {stats['total']}")
    logger.info(f"Correct behavior (Dr):   {stats['correct_behavior']} ({success_rate:.1%})")
    logger.info(f"Wrong tool:              {stats['wrong_tool']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Success rate:            {success_rate:.1%}")
    logger.info("=" * 60)
    
    return dr_samples, stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate MVP Retain Set (Dr) as paired benign twins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data paths
    parser.add_argument(
        "--ds-data",
        type=Path,
        default=BASE_DIR / "data" / "cb_mvp" / "ds_stage1.jsonl",
        help="Path to Ds data (from generate_ds_mvp.py)",
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
        default=BASE_DIR / "data" / "cb_mvp" / "dr_stage1.jsonl",
        help="Output path for Dr",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Backend: 'transformers' (single GPU) or 'vllm' (multi-GPU tensor parallel)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (e.g., 4 for 4x H100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vLLM generation (ignored for transformers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (transformers backend only)",
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
        default=0.3,
        help="Generation temperature (lower = more consistent)",
    )
    
    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of Ds samples to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show stats without generating",
    )
    
    args = parser.parse_args()
    
    # Load tool schema
    if not args.tool_schema.exists():
        logger.error(f"Tool schema not found: {args.tool_schema}")
        return 1
    
    schema = load_tool_schema(args.tool_schema)
    tools = get_tools_list(schema)
    system_prompt = get_system_prompt(schema)
    
    # Load Ds data
    if not args.ds_data.exists():
        logger.error(f"Ds data not found: {args.ds_data}")
        return 1
    
    logger.info(f"Loading Ds data from {args.ds_data}...")
    ds_samples = load_ds_samples(args.ds_data)
    
    if args.limit:
        ds_samples = ds_samples[:args.limit]
    
    logger.info(f"Loaded {len(ds_samples)} Ds samples")
    
    if args.dry_run:
        logger.info("DRY RUN - showing sample records:")
        for sample in ds_samples[:3]:
            logger.info(f"  {sample.get('id')}")
            labels = sample.get("labels", {})
            logger.info(f"    expected: {labels.get('expected_tool')}")
            metadata = sample.get("metadata", {})
            benign_query = metadata.get("benign_query", "N/A")[:80]
            logger.info(f"    benign: {benign_query}...")
        logger.info(f"Would write to: {args.output}")
        return 0
    
    # Load model based on backend
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    if args.backend == "vllm":
        logger.info(f"Using vLLM backend with tensor_parallel={args.tensor_parallel}")
        vllm_backend = VLLMBackend(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel,
            dtype=args.dtype,
        )
        
        # Build Dr MVP with vLLM (batched)
        dr_samples, stats = build_dr_mvp_vllm(
            ds_samples=ds_samples,
            vllm_backend=vllm_backend,
            tools=tools,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
    else:
        logger.info("Using transformers backend")
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=args.device,
            torch_dtype=dtype_map[args.dtype],
        )
        
        # Build Dr MVP with transformers
        dr_samples, stats = build_dr_mvp(
            ds_samples=ds_samples,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            system_prompt=system_prompt,
            temperature=args.temperature,
        )
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in dr_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(dr_samples)} samples to {args.output}")
    
    # Write stats
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote stats to {stats_path}")
    
    # Report ratio
    ratio = len(dr_samples) / len(ds_samples) if ds_samples else 0
    logger.info(f"Dr:Ds ratio = {len(dr_samples)}:{len(ds_samples)} ({ratio:.2f})")
    if ratio < 0.5:
        logger.warning("Dr:Ds ratio < 0.5 - consider adjusting temperature or filtering")
    
    # Clean up
    if args.backend == "vllm":
        del vllm_backend
    else:
        del model
    torch.cuda.empty_cache()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
