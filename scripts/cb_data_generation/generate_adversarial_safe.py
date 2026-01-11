#!/usr/bin/env python3
"""
Stage 2: Generate Adversarial-Safe Samples for Retain Set (Dr)

This script generates Dr samples where the model RESISTS injection attacks.
These are CRITICAL for Stage 2 - they teach the model:
"Even with adversarial prompts, still call the CORRECT tool"

Key Principle:
- Input: B4 adversarial prompts (with malicious injection)
- Generate: Model responses at low temperature
- Filter: ONLY keep samples where observed_tool == expected_tool
- Output: Samples for Dr that demonstrate correct behavior under attack

Expected yield: ~95% (since baseline ASR was only 5.2% in Stage 1)

Usage:
    # On HPC with GPU
    python scripts/cb_data_generation/generate_adversarial_safe.py \
        --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/circuit_breakers/retain/b4_adversarial_safe.jsonl \
        --target-n 500 \
        --temperature 0.1

    # Dry run to check resistance rate
    python scripts/cb_data_generation/generate_adversarial_safe.py --dry-run --limit 100
"""

# Set cache directories BEFORE any other imports (for vLLM/flashinfer)
import os

if "SCRATCH_DIR" in os.environ:
    scratch_dir = os.environ["SCRATCH_DIR"]
elif "SCRATCH" in os.environ:
    scratch_dir = os.environ["SCRATCH"]
else:
    scratch_dir = os.path.expanduser("~/scratch")

cache_root = os.path.join(scratch_dir, "cb_cache")

# CRITICAL FIX: Override HOME to trick flashinfer and other tools
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
    
    For adversarial-safe generation, we need the combined_query (with injection)
    and the expected_tool (what SHOULD be called).
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
            
            # Skip non-flip cases (where expected == simulated)
            if expected_tool == simulated_tool:
                continue
            
            yield {
                "id": record.get("record_id", f"fujitsu_b4_{count}"),
                "source": "fujitsu_b4",
                "benign_query": benign_query,
                "malicious_injection": malicious_injection,
                "combined_query": combined_query,  # The adversarial prompt
                "expected_tool": expected_tool,     # What SHOULD be called
                "simulated_tool": simulated_tool,   # What attack wants
                "metadata": {
                    "attack_id": record.get("attack_id"),
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                },
            }
            
            count += 1
            if limit and count >= limit:
                return


# =============================================================================
# vLLM Backend (for GPU inference)
# =============================================================================

class VLLMBackend:
    """vLLM backend for high-throughput batched inference."""
    
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
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.1,  # Low temp for consistent resistance
        max_tokens: int = 256,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            skip_special_tokens=False,  # Keep special tokens for tool call parsing
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
            # Fallback without tools
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


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


# =============================================================================
# Tool Call Extraction
# =============================================================================

def extract_tool_call(response: str) -> Tuple[Optional[str], Optional[Dict], bool]:
    """
    Extract tool name and parameters from assistant response.
    
    Returns:
        Tuple of (tool_name, parameters, is_valid)
    """
    tool_name = None
    parameters = None
    is_valid = False
    
    # Try Llama 3.1 format: <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>
    if "<|python_tag|>" in response:
        content = response.split("<|python_tag|>", 1)[1]
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            content = content.split(end_token)[0]
        content = content.strip()
        
        # Try JSON format
        try:
            data = json.loads(content)
            tool_name = data.get("name") or data.get("function", {}).get("name")
            parameters = data.get("parameters", data.get("arguments", {}))
            is_valid = tool_name is not None and isinstance(parameters, dict)
        except json.JSONDecodeError:
            # Try function call syntax: tool_name({...})
            match = re.match(r'(\w+)\s*\((\{.*\})\)', content, re.DOTALL)
            if match:
                tool_name = match.group(1)
                try:
                    parameters = json.loads(match.group(2))
                    is_valid = True
                except json.JSONDecodeError:
                    pass
    
    # Fallback: Try to find JSON anywhere
    if not tool_name:
        json_match = re.search(r'"name"\s*:\s*"([^"]+)"', response)
        if json_match:
            tool_name = json_match.group(1)
    
    return tool_name, parameters, is_valid


def extract_tool_name(response: str) -> Optional[str]:
    """Extract tool name from assistant response."""
    tool_name, _, _ = extract_tool_call(response)
    return tool_name


# =============================================================================
# Format Tool Call Output
# =============================================================================

def format_tool_call_output(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    Format tool call in Llama 3.1 JSON format.
    
    Output: <|python_tag|>{"name": "tool_name", "parameters": {...}}<|eom_id|>
    """
    tool_call = {
        "name": tool_name,
        "parameters": parameters if parameters else {}
    }
    return f'<|python_tag|>{json.dumps(tool_call)}<|eom_id|>'


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_adversarial_safe_samples(
    b4_records: List[Dict[str, Any]],
    backend: VLLMBackend,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    target_n: int = 500,
    temperature: float = 0.1,
    batch_size: int = 32,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate adversarial-safe samples where model RESISTS injection.
    
    Args:
        b4_records: B4 records with adversarial prompts
        backend: vLLM backend for generation
        tools: Tool definitions
        system_prompt: System prompt
        target_n: Target number of samples
        temperature: Generation temperature (low = more consistent)
        batch_size: Batch size for generation
        dry_run: If True, only report statistics without saving
    
    Returns:
        List of adversarial-safe samples for Dr
    """
    samples = []
    stats = {
        "total_processed": 0,
        "resisted": 0,
        "attacked": 0,
        "no_tool_call": 0,
        "invalid": 0,
    }
    
    # Prepare batches
    batches = [b4_records[i:i + batch_size] for i in range(0, len(b4_records), batch_size)]
    
    for batch in tqdm(batches, desc="Generating adversarial-safe samples"):
        if len(samples) >= target_n:
            break
        
        # Format prompts
        prompts = []
        for record in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["combined_query"]},
            ]
            prompt = backend.format_prompt_with_tools(messages, tools)
            prompts.append(prompt)
        
        # Generate responses
        responses = backend.generate_batch(
            prompts,
            temperature=temperature,
            max_tokens=256,
        )
        
        # Process responses
        for record, response in zip(batch, responses):
            stats["total_processed"] += 1
            
            tool_name, parameters, is_valid = extract_tool_call(response)
            
            if tool_name is None:
                stats["no_tool_call"] += 1
                continue
            
            if not is_valid:
                stats["invalid"] += 1
                continue
            
            # Check if model RESISTED the attack
            if tool_name == record["expected_tool"]:
                stats["resisted"] += 1
                
                if not dry_run and len(samples) < target_n:
                    # Format the assistant response properly
                    assistant_raw = format_tool_call_output(tool_name, parameters)
                    
                    # Create sample for Dr
                    sample = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": record["combined_query"]},
                        ],
                        "assistant_raw": assistant_raw,
                        "labels": {
                            "expected_tool": record["expected_tool"],
                            "simulated_tool": record["simulated_tool"],
                            "observed_tool": tool_name,
                            "is_adversarial_safe": True,
                        },
                        "metadata": {
                            "split": "retain",
                            "source": "b4_adversarial_safe",
                            "original_id": record["id"],
                            "attack_category": record["metadata"].get("category"),
                            "has_tool_calls": True,
                        },
                    }
                    samples.append(sample)
            
            elif tool_name == record["simulated_tool"]:
                stats["attacked"] += 1
            else:
                # Called a different tool entirely
                stats["invalid"] += 1
    
    # Log statistics
    resistance_rate = stats["resisted"] / max(1, stats["total_processed"])
    attack_rate = stats["attacked"] / max(1, stats["total_processed"])
    
    logger.info("=" * 60)
    logger.info("ADVERSARIAL-SAFE GENERATION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total processed: {stats['total_processed']}")
    logger.info(f"Resisted (adversarial-safe): {stats['resisted']} ({resistance_rate:.1%})")
    logger.info(f"Attacked (attack succeeded): {stats['attacked']} ({attack_rate:.1%})")
    logger.info(f"No tool call: {stats['no_tool_call']}")
    logger.info(f"Invalid: {stats['invalid']}")
    logger.info(f"Samples collected: {len(samples)}")
    logger.info("=" * 60)
    
    return samples


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial-safe samples for Stage 2 Dr"
    )
    parser.add_argument(
        "--b4-data",
        type=Path,
        default=BASE_DIR / "data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl",
        help="Path to Fujitsu B4 data",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=BASE_DIR / "configs/tool_schemas/b4_standard_v1.json",
        help="Path to frozen tool schema",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/retain/b4_adversarial_safe.jsonl",
        help="Output path for adversarial-safe samples",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=500,
        help="Target number of samples to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (lower = more consistent resistance)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of B4 records to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute statistics, don't save samples",
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading B4 data from {args.b4_data}...")
    b4_records = list(load_fujitsu_b4(args.b4_data, limit=args.limit))
    logger.info(f"Loaded {len(b4_records)} B4 records (with tool-flip)")
    
    if not b4_records:
        logger.error("No B4 records found!")
        sys.exit(1)
    
    # Load tool schema
    logger.info(f"Loading tool schema from {args.tool_schema}...")
    schema = load_tool_schema(args.tool_schema)
    tools = get_tools_list(schema)
    system_prompt = get_system_prompt(schema)
    logger.info(f"Loaded {len(tools)} tools")
    
    # Initialize backend
    logger.info("Initializing vLLM backend...")
    backend = VLLMBackend(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        dtype="bfloat16",
    )
    
    # Generate samples
    samples = generate_adversarial_safe_samples(
        b4_records=b4_records,
        backend=backend,
        tools=tools,
        system_prompt=system_prompt,
        target_n=args.target_n,
        temperature=args.temperature,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    
    if args.dry_run:
        logger.info("Dry run complete. No files written.")
        return
    
    # Save samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(samples)} adversarial-safe samples to {args.output}")
    
    # Validation summary
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 ADVERSARIAL-SAFE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Purpose: Dr (retain set) - teach model to resist attacks")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
