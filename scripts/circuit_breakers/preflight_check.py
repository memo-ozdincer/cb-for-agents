#!/usr/bin/env python3
"""
Preflight Check for CB Evaluation

Quick validation before burning a training run:
1. Tool-flip ASR only includes true flips (expected != simulated)
2. Forced-call uses JSON prefill and parser extracts correctly
3. Model emits <|eom_id|> consistently on tool calls

Usage:
    # Check eval data structure
    python scripts/circuit_breakers/preflight_check.py \
        --eval-data data/cb_mvp/eval_stage1.jsonl \
        --mode data
    
    # Run 10-sample eval and inspect outputs
    python scripts/circuit_breakers/preflight_check.py \
        --eval-data data/cb_mvp/eval_stage1.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --limit 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_eval_data(eval_path: Path) -> dict:
    """Analyze eval data for true tool-flip samples."""
    stats = {
        "total": 0,
        "has_expected_tool": 0,
        "has_simulated_tool": 0,
        "expected_eq_simulated": 0,
        "true_flips": 0,
        "missing_labels": 0,
        "tools_seen": set(),
    }
    
    samples = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    for sample in samples:
        stats["total"] += 1
        labels = sample.get("labels", {})
        expected = labels.get("expected_tool", "")
        simulated = labels.get("simulated_tool", "")
        
        if expected:
            stats["has_expected_tool"] += 1
            stats["tools_seen"].add(expected)
        if simulated:
            stats["has_simulated_tool"] += 1
            stats["tools_seen"].add(simulated)
        
        if not expected or not simulated:
            stats["missing_labels"] += 1
        elif expected == simulated:
            stats["expected_eq_simulated"] += 1
        else:
            stats["true_flips"] += 1
    
    stats["tools_seen"] = sorted(stats["tools_seen"])
    return stats, samples


def check_training_data(train_path: Path) -> dict:
    """Analyze training data for proper format."""
    stats = {
        "total_samples": 0,
        "harmful_samples": 0,
        "benign_samples": 0,
        "has_python_tag": 0,
        "has_eom_id": 0,
        "has_eot_id": 0,
        "has_assistant_header": 0,
    }
    
    with open(train_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            batch = json.loads(line)
            
            for sample in batch.get("harmful", []):
                stats["total_samples"] += 1
                stats["harmful_samples"] += 1
                text = sample.get("text", "")
                
                if "<|python_tag|>" in text:
                    stats["has_python_tag"] += 1
                if "<|eom_id|>" in text:
                    stats["has_eom_id"] += 1
                if "<|eot_id|>" in text:
                    stats["has_eot_id"] += 1
                if "<|start_header_id|>assistant<|end_header_id|>" in text:
                    stats["has_assistant_header"] += 1
            
            for sample in batch.get("benign", []):
                stats["total_samples"] += 1
                stats["benign_samples"] += 1
    
    return stats


def print_sample_details(samples, limit=5):
    """Print detailed info for a few samples."""
    print("\n" + "=" * 70)
    print("SAMPLE DETAILS (first few true flips)")
    print("=" * 70)
    
    true_flips = [
        s for s in samples 
        if s.get("labels", {}).get("expected_tool") != s.get("labels", {}).get("simulated_tool")
        and s.get("labels", {}).get("expected_tool")
        and s.get("labels", {}).get("simulated_tool")
    ]
    
    for i, sample in enumerate(true_flips[:limit]):
        labels = sample.get("labels", {})
        print(f"\n--- Sample {i+1}: {sample.get('id', 'unknown')[:50]} ---")
        print(f"  Expected tool:  {labels.get('expected_tool')}")
        print(f"  Simulated tool: {labels.get('simulated_tool')}")
        
        messages = sample.get("messages", [])
        if messages:
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg:
                content = user_msg.get("content", "")[:150]
                print(f"  User query: {content}...")


def main():
    parser = argparse.ArgumentParser(description="Preflight check for CB eval")
    parser.add_argument("--eval-data", type=Path, help="Eval data JSONL")
    parser.add_argument("--train-data", type=Path, help="Training data JSONL")
    parser.add_argument("--mode", choices=["data", "run"], default="data")
    parser.add_argument("--model", type=str, help="Model to test")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    
    # Check eval data
    if args.eval_data and args.eval_data.exists():
        print(f"\n📊 Analyzing eval data: {args.eval_data}")
        stats, samples = check_eval_data(args.eval_data)
        
        print(f"\n  Total samples: {stats['total']}")
        print(f"  Has expected_tool: {stats['has_expected_tool']}")
        print(f"  Has simulated_tool: {stats['has_simulated_tool']}")
        print(f"  expected == simulated (filtered out): {stats['expected_eq_simulated']}")
        print(f"  TRUE FLIPS (used for ASR): {stats['true_flips']}")
        print(f"  Missing labels: {stats['missing_labels']}")
        print(f"\n  Tools seen: {stats['tools_seen']}")
        
        if stats['true_flips'] == 0:
            print("\n  ⚠️  WARNING: No true tool-flip samples! ASR eval will be empty.")
        elif stats['true_flips'] < 20:
            print(f"\n  ⚠️  WARNING: Only {stats['true_flips']} true flips. Consider expanding eval set.")
        else:
            print(f"\n  ✅ {stats['true_flips']} true flip samples ready for ASR eval")
        
        print_sample_details(samples, limit=5)
    
    # Check training data
    if args.train_data and args.train_data.exists():
        print(f"\n📊 Analyzing training data: {args.train_data}")
        stats = check_training_data(args.train_data)
        
        print(f"\n  Total samples: {stats['total_samples']}")
        print(f"  Harmful: {stats['harmful_samples']}")
        print(f"  Benign: {stats['benign_samples']}")
        print(f"\n  Format checks (on harmful only):")
        print(f"    Has <|python_tag|>: {stats['has_python_tag']} ({100*stats['has_python_tag']/max(1,stats['harmful_samples']):.1f}%)")
        print(f"    Has <|eom_id|>: {stats['has_eom_id']} ({100*stats['has_eom_id']/max(1,stats['harmful_samples']):.1f}%)")
        print(f"    Has assistant header: {stats['has_assistant_header']} ({100*stats['has_assistant_header']/max(1,stats['harmful_samples']):.1f}%)")
        
        if stats['has_python_tag'] < stats['harmful_samples'] * 0.95:
            print("\n  ⚠️  WARNING: <5% of harmful samples have <|python_tag|>!")
        if stats['has_assistant_header'] < stats['harmful_samples'] * 0.95:
            print("\n  ⚠️  WARNING: <95% have Llama 3.1 assistant header!")
        else:
            print("\n  ✅ Training data format looks good")
    
    # Run mode - actually test model
    if args.mode == "run" and args.model:
        print(f"\n🚀 Running {args.limit}-sample eval with model: {args.model}")
        print("   (This requires loading the model - may take a minute)")
        
        # Import eval functions
        try:
            from eval_mvp import (
                load_model_and_tokenizer,
                load_tool_schema,
                evaluate_tool_flip_asr,
                evaluate_forced_function_call,
            )
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent))
            from eval_mvp import (
                load_model_and_tokenizer,
                load_tool_schema,
                evaluate_tool_flip_asr,
                evaluate_forced_function_call,
            )
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model)
        
        # Load eval samples (limit)
        with open(args.eval_data, "r") as f:
            eval_samples = [json.loads(line) for line in f if line.strip()][:args.limit]
        
        # Load tool schema
        schema_path = Path("configs/tool_schemas/b4_standard_v1.json")
        if schema_path.exists():
            tools, system_prompt = load_tool_schema(schema_path)
        else:
            tools = []
            system_prompt = "You are a helpful assistant with access to tools."
        
        # Run tool-flip eval
        print("\n--- Tool-flip ASR ---")
        results = evaluate_tool_flip_asr(model, tokenizer, eval_samples, tools, system_prompt)
        print(f"  ASR: {results['attack_success_rate']:.1%}")
        print(f"  Total evaluated: {results['total_samples']}")
        print(f"  Filtered out: {results['filtered_out_samples']}")
        
        # Show a few details
        print("\n  Sample outputs:")
        for d in results["details"][:3]:
            print(f"    {d['id'][:30]}: expected={d['expected_tool']}, "
                  f"simulated={d['simulated_tool']}, observed={d['observed_tool']}, "
                  f"outcome={d['outcome']}")
            print(f"      Response: {d['response_preview'][:80]}...")
        
        # Run forced-call eval
        print("\n--- Forced Function Call ---")
        results = evaluate_forced_function_call(model, tokenizer, eval_samples, tools, system_prompt)
        print(f"  Completion rate: {results['forced_call_success_rate']:.1%}")
        print(f"  Total: {results['total_samples']}")
        
        print("\n  Sample outputs:")
        for d in results["details"][:3]:
            print(f"    {d['id'][:30]}: completed={d['completed']}, "
                  f"has_json={d['has_valid_json']}, has_end={d['has_end_token']}")
            print(f"      Generated: {d['generated_part'][:60]}...")
    
    print("\n" + "=" * 70)
    print("PREFLIGHT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
