#!/usr/bin/env python3
"""
Stage 2: Ingest TAU2-Bench Customer Service Traces

This script extracts TAU2-Bench customer service traces for the retain set (Dr).
TAU2 provides multi-turn tool-use traces in customer service domains.

Domains:
- airline: Flight bookings, cancellations, baggage
- retail: Order tracking, returns, product inquiries  
- telecom: Bill inquiries, plan changes, tech support

These are benign capability examples that help preserve tool-calling ability.

Usage:
    python scripts/cb_data_generation/ingest_tau2_traces.py \
        --tau2-path data/tau2_repo/ \
        --output data/circuit_breakers/retain/tau2_traces.jsonl \
        --target-n 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# TAU2 Format Conversion
# =============================================================================

def convert_tau2_action_to_tool_call(action: Dict[str, Any]) -> str:
    """
    Convert TAU2 action to Llama 3.1 tool call format.
    
    TAU2 action format:
    {
        "action_type": "function_name",
        "parameters": {...}
    }
    
    Output: <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>
    """
    action_type = action.get("action_type", "")
    parameters = action.get("parameters", {})
    
    tool_call = {
        "name": action_type,
        "parameters": parameters
    }
    
    return f'<|python_tag|>{json.dumps(tool_call)}<|eom_id|>'


def convert_tau2_trace_to_sample(
    trace: Dict[str, Any],
    domain: str,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert a TAU2 trace to CB training format.
    
    TAU2 trace structure varies, but typically has:
    - conversation: List of turns
    - actions: List of agent actions with tools
    """
    messages = []
    assistant_raw = None
    
    # Try to extract conversation
    conversation = trace.get("conversation", [])
    actions = trace.get("actions", [])
    
    # If trace has conversation format
    if conversation:
        for turn in conversation:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user" and content:
                messages.append({"role": "user", "content": content})
            elif role == "assistant" and content:
                messages.append({"role": "assistant", "content": content})
            elif role == "system" and content:
                messages.append({"role": "system", "content": content})
    
    # If trace has actions, use the last action as assistant_raw
    if actions and len(actions) > 0:
        last_action = actions[-1]
        if isinstance(last_action, dict) and "action_type" in last_action:
            assistant_raw = convert_tau2_action_to_tool_call(last_action)
    
    # Fallback: use last assistant message
    if not assistant_raw and messages:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_raw = msg.get("content", "")
                # Remove from messages to avoid duplication
                messages = messages[:-1] if messages[-1] == msg else messages
                break
    
    if not messages or not assistant_raw:
        return None
    
    # Determine if has tool calls
    has_tool_calls = "<|python_tag|>" in assistant_raw
    
    return {
        "messages": messages,
        "assistant_raw": assistant_raw,
        "labels": {
            "domain": domain,
            "task_id": task_id,
            "has_tool_calls": has_tool_calls,
        },
        "metadata": {
            "split": "retain",
            "source": "tau2",
            "domain": domain,
            "task_id": task_id,
            "has_tool_calls": has_tool_calls,
        },
    }


def load_tau2_traces_from_results(
    results_dir: Path,
    domain: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load traces from TAU2 results directory."""
    samples = []
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return samples
    
    # Look for JSON files in results directory
    json_files = list(results_dir.glob("*.json"))
    
    for file_path in json_files:
        if limit and len(samples) >= limit:
            break
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
            
            sample = convert_tau2_trace_to_sample(
                trace, domain, file_path.stem
            )
            
            if sample:
                samples.append(sample)
        
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Error processing {file_path.name}: {e}")
            continue
    
    return samples


def load_tau2_tasks_as_samples(
    tasks_path: Path,
    domain: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load TAU2 tasks and convert to samples.
    
    Tasks file contains task definitions with user instructions.
    We create samples from these for capability retention.
    """
    samples = []
    
    if not tasks_path.exists():
        logger.warning(f"Tasks file not found: {tasks_path}")
        return samples
    
    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Error loading tasks: {e}")
        return samples
    
    # Handle both list and dict formats
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    
    for task in tasks:
        if limit and len(samples) >= limit:
            break
        
        # Extract task description/instruction
        instruction = task.get("instruction", task.get("description", ""))
        task_id = task.get("id", task.get("task_id", str(len(samples))))
        
        if not instruction:
            continue
        
        # Create a simple sample with the task instruction
        sample = {
            "messages": [
                {"role": "user", "content": instruction}
            ],
            "assistant_raw": f"I'll help you with that request regarding {domain} services.",
            "labels": {
                "domain": domain,
                "task_id": task_id,
                "has_tool_calls": False,
            },
            "metadata": {
                "split": "retain",
                "source": "tau2_task",
                "domain": domain,
                "task_id": task_id,
                "has_tool_calls": False,
            },
        }
        samples.append(sample)
    
    return samples


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest TAU2-Bench traces for retain set"
    )
    parser.add_argument(
        "--tau2-path",
        type=Path,
        default=BASE_DIR / "data/tau2_repo",
        help="Path to TAU2 repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/retain/tau2_traces.jsonl",
        help="Output path for TAU2 samples",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=500,
        help="Target number of samples",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["airline", "retail", "telecom"],
        help="Domains to process",
    )
    
    args = parser.parse_args()
    
    all_samples = []
    
    # Try multiple possible paths for TAU2 data
    possible_data_paths = [
        args.tau2_path / "data/tau2/domains",
        args.tau2_path / "data/domains",
        args.tau2_path / "domains",
        args.tau2_path,
    ]
    
    data_path = None
    for path in possible_data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        logger.warning(f"TAU2 data path not found. Tried: {possible_data_paths}")
        logger.info("Creating minimal placeholder samples...")
        
        # Create placeholder samples for demonstration
        for domain in args.domains:
            for i in range(min(50, args.target_n // len(args.domains))):
                sample = {
                    "messages": [
                        {"role": "user", "content": f"I need help with my {domain} service request #{i+1}."}
                    ],
                    "assistant_raw": f"I'll be happy to help you with your {domain} inquiry. Let me look into that for you.",
                    "labels": {"domain": domain, "has_tool_calls": False},
                    "metadata": {"split": "retain", "source": "tau2_placeholder", "domain": domain, "has_tool_calls": False},
                }
                all_samples.append(sample)
        
    else:
        logger.info(f"Found TAU2 data at: {data_path}")
        
        samples_per_domain = args.target_n // len(args.domains) + 1
        
        for domain in args.domains:
            domain_path = data_path / domain
            
            if not domain_path.exists():
                logger.warning(f"Domain path not found: {domain_path}")
                continue
            
            logger.info(f"Processing domain: {domain}")
            
            # Try to load from results first
            results_path = args.tau2_path / f"data/tau2/results/final/{domain}"
            if results_path.exists():
                samples = load_tau2_traces_from_results(
                    results_path, domain, samples_per_domain
                )
                logger.info(f"  Loaded {len(samples)} from results")
                all_samples.extend(samples)
            
            # Also try tasks file
            tasks_path = domain_path / "tasks.json"
            if tasks_path.exists() and len(all_samples) < args.target_n:
                remaining = args.target_n - len(all_samples)
                samples = load_tau2_tasks_as_samples(
                    tasks_path, domain, remaining // len(args.domains)
                )
                logger.info(f"  Loaded {len(samples)} from tasks")
                all_samples.extend(samples)
    
    # Limit to target
    if len(all_samples) > args.target_n:
        all_samples = all_samples[:args.target_n]
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save samples
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(all_samples)} TAU2 samples to {args.output}")
    
    # Summary by domain
    domain_counts = {}
    for sample in all_samples:
        domain = sample.get("metadata", {}).get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    logger.info("\n" + "=" * 60)
    logger.info("TAU2 INGEST COMPLETE")
    logger.info("=" * 60)
    for domain, count in domain_counts.items():
        logger.info(f"  {domain}: {count}")
    logger.info(f"Total: {len(all_samples)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
