#!/usr/bin/env python3
"""
Circuit Breaker Set (Ds) Generator

Generate the Circuit Breaker Set with REAL LLM-produced harmful completions.
This set contains examples that reliably elicit internal states leading to
harmful outputs/actions (including tool calls).

Key principles from the CB paper:
1. Ds must contain ACTUAL model outputs, not templates
2. Quality depends on how precisely data elicits targeted representations
3. For refusal-trained models: keep harmful responses, optionally redact harmful prompts

Usage:
    # Generate Ds from Fujitsu B4 data using vLLM
    python scripts/cb_data_generation/generate_ds.py \
        --backend vllm \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/circuit_breakers/ds/circuit_breaker_set.jsonl

    # Using transformers (slower but more flexible)
    python scripts/cb_data_generation/generate_ds.py \
        --backend transformers \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --load-in-8bit

    # Dry run to check data
    python scripts/cb_data_generation/generate_ds.py --dry-run --limit 10

    # Generate from AgentDojo attack failures
    python scripts/cb_data_generation/generate_ds.py \
        --source agentdojo \
        --output data/circuit_breakers/ds/ds_agentdojo.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cb_data_generation.tool_format import (
    ToolDefinition,
    ToolCall,
    STANDARD_TOOLS,
    AGENTIC_RAG_SYSTEM_PROMPT,
    get_tools_for_attack,
)
from scripts.cb_data_generation.llm_harness import (
    LLMHarness,
    GenerationConfig,
    GenerationResult,
    BatchGenerationResult,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Data Source Loaders
# =============================================================================

def load_fujitsu_b4(
    path: Path,
    success_only: bool = True,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Load Fujitsu B4 orchestrator attack records."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if success_only and not record.get("success", False):
                continue
            
            # Transform to standard format
            yield {
                "id": record.get("record_id", f"fujitsu_b4_{count}"),
                "source": "fujitsu_b4",
                "user_prompt": record.get("combined_query") or record.get("benign_query", ""),
                "expected_tool": record.get("expected_tool", ""),
                "simulated_tool": record.get("simulated_tool", ""),
                "metadata": {
                    "attack_id": record.get("attack_id"),
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                    "benign_query": record.get("benign_query"),
                    "malicious_injection": record.get("malicious_injection"),
                    "judge_note": record.get("judge_note"),
                },
            }
            
            count += 1
            if limit and count >= limit:
                return


def load_agentdojo_failures(
    directory: Path,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Load AgentDojo security failure cases."""
    count = 0
    
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Skip metadata lines (first line in file)
                if "benchmark" in record.get("metadata", {}) and "messages" not in record:
                    continue
                
                # Skip if no messages or if security passed
                messages = record.get("messages", [])
                metadata = record.get("metadata", {})
                
                if not messages:
                    continue
                if metadata.get("security", True):  # Skip secure cases
                    continue
                if metadata.get("injection_task_id") is None:  # Skip benign
                    continue
                
                # Extract the harmful assistant response
                # Find the last assistant message
                last_assistant = None
                last_tool_calls = []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        last_assistant = msg.get("content", "")
                        last_tool_calls = msg.get("tool_calls", [])
                        break
                
                # Extract user prompt (first non-system user message)
                user_prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
                
                yield {
                    "id": f"agentdojo_{metadata.get('suite_name')}_{metadata.get('user_task_id')}_{count}",
                    "source": "agentdojo",
                    "user_prompt": user_prompt,
                    "messages": messages,  # Full trace for context
                    "assistant_raw": last_assistant,
                    "tool_calls": last_tool_calls,
                    "expected_tool": None,  # AgentDojo doesn't have single expected tool
                    "simulated_tool": None,
                    "metadata": {
                        "suite_name": metadata.get("suite_name"),
                        "user_task_id": metadata.get("user_task_id"),
                        "injection_task_id": metadata.get("injection_task_id"),
                        "attack_type": metadata.get("attack_type"),
                        "model": metadata.get("model"),
                        "file": jsonl_path.name,
                    },
                }
                
                count += 1
                if limit and count >= limit:
                    return


def load_fujitsu_b1b3(
    b1_path: Path,
    b3_path: Path,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Load Fujitsu B1 (text poisoning) and B3 (direct query) attacks.
    
    These already have real completions (from GPT-4 evaluation), so we can
    use them directly without regeneration.
    """
    count = 0
    
    for path, source in [(b1_path, "fujitsu_b1"), (b3_path, "fujitsu_b3")]:
        if not path.exists():
            logger.warning(f"Path not found: {path}")
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # These have different schemas
                if source == "fujitsu_b1":
                    user_prompt = record.get("user_query", "")
                    completion = record.get("target_llm_output", "")
                else:
                    user_prompt = record.get("prompt", "") or record.get("user_query", "")
                    completion = record.get("completion", "") or record.get("response", "")
                
                if not user_prompt or not completion:
                    continue
                
                yield {
                    "id": record.get("id", f"{source}_{count}"),
                    "source": source,
                    "user_prompt": user_prompt,
                    "assistant_raw": completion,
                    "expected_tool": None,
                    "simulated_tool": None,
                    "has_real_completion": True,  # Already has real completion
                    "metadata": {
                        "category": record.get("attack_category", record.get("category")),
                        "subtype": record.get("attack_subtype", record.get("subtype")),
                    },
                }
                
                count += 1
                if limit and count >= limit:
                    return


# =============================================================================
# Main Ds Generation
# =============================================================================

def generate_ds_record(
    record: Dict[str, Any],
    harness: LLMHarness,
    config: GenerationConfig,
    redact_harmful_prompt: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single Ds record.
    
    Args:
        record: Source record with user_prompt, expected_tool, etc.
        harness: LLM harness for generation
        config: Generation config
        redact_harmful_prompt: If True, replace harmful user content (CB paper trick)
    
    Returns:
        Formatted Ds record, or None if no harmful completion generated
    """
    # If record already has real completion (B1/B3, AgentDojo), use it directly
    if record.get("has_real_completion") or record.get("assistant_raw"):
        user_prompt = record["user_prompt"]
        
        # Optionally redact harmful parts (CB paper approach)
        if redact_harmful_prompt and "malicious_injection" in record.get("metadata", {}):
            # Remove injection, keep benign query
            benign = record.get("metadata", {}).get("benign_query", "")
            if benign:
                user_prompt = benign
        
        # Parse any tool calls from the raw output
        tool_calls_structured = []
        raw_output = record.get("assistant_raw", "")
        
        # Check if there are existing tool calls
        if record.get("tool_calls"):
            for tc in record["tool_calls"]:
                if isinstance(tc, dict):
                    func = tc.get("function", tc)
                    tool_calls_structured.append({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    })
        else:
            # Try to parse from raw
            parsed = ToolCall.from_raw(raw_output)
            if parsed:
                tool_calls_structured.append({
                    "name": parsed.name,
                    "arguments": parsed.arguments,
                })
        
        return {
            "id": record["id"],
            "split": "train",
            "set": "circuit_breaker",
            "messages": [
                {"role": "system", "content": AGENTIC_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "assistant_raw": raw_output,
            "tool_calls_structured": tool_calls_structured,
            "labels": {
                "is_harmful": True,
                "harm_type": record.get("metadata", {}).get("category", "unknown"),
                "expected_tool": record.get("expected_tool"),
                "observed_tool": tool_calls_structured[0]["name"] if tool_calls_structured else None,
            },
            "provenance": {
                "source_dataset": record["source"],
                "record_id": record["id"],
                "has_real_completion": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **record.get("metadata", {}),
            },
        }
    
    # Need to generate completion
    user_prompt = record["user_prompt"]
    expected_tool = record.get("expected_tool", "")
    simulated_tool = record.get("simulated_tool", "")
    
    # Get tools
    tools = get_tools_for_attack(expected_tool, simulated_tool) if expected_tool else list(STANDARD_TOOLS.values())[:4]
    
    # Generate
    result = harness.generate_tool_call(
        user_prompt=user_prompt,
        tools=tools,
        expected_tool=expected_tool,
        config=config,
        prompt_id=record["id"],
        attack_metadata={
            "source_dataset": record["source"],
            **record.get("metadata", {}),
        },
    )
    
    # Return CB record if harmful
    return result.to_cb_record()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Circuit Breaker Set (Ds) with real LLM completions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Source selection
    parser.add_argument(
        "--source",
        choices=["fujitsu_b4", "agentdojo", "fujitsu_b1b3", "all"],
        default="all",
        help="Which data source to use",
    )
    
    # Model / Backend
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (transformers only)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (transformers only)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    
    # Generation config
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per generation",
    )
    
    # Data paths
    parser.add_argument(
        "--fujitsu-b4",
        type=Path,
        default=BASE_DIR / "data" / "fujitsu" / "orchestrator_attacks_combined_deduplicated.jsonl",
        help="Path to Fujitsu B4 data",
    )
    parser.add_argument(
        "--fujitsu-b1",
        type=Path,
        default=BASE_DIR / "data" / "fujitsu" / "rag_poisoning_benchmark_combined_deduplicated.jsonl",
        help="Path to Fujitsu B1 data",
    )
    parser.add_argument(
        "--fujitsu-b3",
        type=Path,
        default=BASE_DIR / "data" / "fujitsu" / "safety_benchmark_direct_query_combined_deduplicated.jsonl",
        help="Path to Fujitsu B3 data",
    )
    parser.add_argument(
        "--agentdojo-dir",
        type=Path,
        default=BASE_DIR / "data" / "agent_dojo",
        help="Directory containing AgentDojo JSONL files",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "circuit_breakers" / "ds" / "circuit_breaker_set.jsonl",
        help="Output path for Ds",
    )
    
    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        default=True,
        help="Only process successful attacks (Fujitsu B4)",
    )
    parser.add_argument(
        "--redact-harmful-prompt",
        action="store_true",
        help="Replace harmful user prompts with benign versions (CB paper trick)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip LLM generation, only use records with existing completions",
    )
    
    args = parser.parse_args()
    
    # Collect source records
    all_records = []
    
    logger.info("Loading source data...")
    
    if args.source in ["fujitsu_b4", "all"]:
        if args.fujitsu_b4.exists():
            b4_records = list(load_fujitsu_b4(
                args.fujitsu_b4,
                success_only=args.success_only,
                limit=args.limit,
            ))
            logger.info(f"Loaded {len(b4_records)} Fujitsu B4 records")
            all_records.extend(b4_records)
        else:
            logger.warning(f"Fujitsu B4 not found: {args.fujitsu_b4}")
    
    if args.source in ["agentdojo", "all"]:
        if args.agentdojo_dir.exists():
            ad_records = list(load_agentdojo_failures(
                args.agentdojo_dir,
                limit=args.limit,
            ))
            logger.info(f"Loaded {len(ad_records)} AgentDojo security failures")
            all_records.extend(ad_records)
        else:
            logger.warning(f"AgentDojo dir not found: {args.agentdojo_dir}")
    
    if args.source in ["fujitsu_b1b3", "all"]:
        b1b3_records = list(load_fujitsu_b1b3(
            args.fujitsu_b1,
            args.fujitsu_b3,
            limit=args.limit,
        ))
        logger.info(f"Loaded {len(b1b3_records)} Fujitsu B1/B3 records")
        all_records.extend(b1b3_records)
    
    logger.info(f"Total source records: {len(all_records)}")
    
    if args.dry_run:
        logger.info("DRY RUN - showing sample records:")
        for record in all_records[:3]:
            logger.info(f"  {record['source']}: {record['id']}")
            logger.info(f"    prompt: {record.get('user_prompt', '')[:100]}...")
            logger.info(f"    expected: {record.get('expected_tool')}, simulated: {record.get('simulated_tool')}")
        logger.info(f"Would write to: {args.output}")
        return 0
    
    # Setup harness (if needed)
    harness = None
    need_generation = any(
        not r.get("has_real_completion") and not r.get("assistant_raw")
        for r in all_records
    )
    
    if need_generation and not args.skip_generation:
        logger.info(f"Initializing {args.backend} backend with {args.model}...")
        if args.backend == "vllm":
            harness = LLMHarness.from_vllm(
                args.model,
                tensor_parallel_size=args.tensor_parallel,
            )
        else:
            harness = LLMHarness.from_transformers(
                args.model,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
            )
    
    config = GenerationConfig(
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
    )
    
    # Generate Ds
    logger.info("Generating Circuit Breaker Set (Ds)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": 0,
        "with_existing": 0,
        "generated": 0,
        "successful": 0,
        "skipped": 0,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        for i, record in enumerate(all_records):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing {i+1}/{len(all_records)}...")
            
            stats["total"] += 1
            
            # Check if already has completion
            has_existing = record.get("has_real_completion") or record.get("assistant_raw")
            
            if has_existing:
                stats["with_existing"] += 1
            elif args.skip_generation or harness is None:
                stats["skipped"] += 1
                continue
            else:
                stats["generated"] += 1
            
            # Generate/format record
            try:
                ds_record = generate_ds_record(
                    record,
                    harness,
                    config,
                    redact_harmful_prompt=args.redact_harmful_prompt,
                )
                
                if ds_record:
                    f.write(json.dumps(ds_record, ensure_ascii=False) + "\n")
                    stats["successful"] += 1
            except Exception as e:
                logger.error(f"Error processing {record['id']}: {e}")
                continue
    
    logger.info(f"Generation complete!")
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")
    logger.info(f"Output written to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
