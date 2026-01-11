#!/usr/bin/env python3
"""
Quick script to generate sample outputs from baseline vs CB model.
Run this on HPC to see what the models are actually producing.

Usage:
    python sample_outputs.py \
        --eval-data /scratch/memoozd/cb_mvp_data/eval_stage1.jsonl \
        --cb-adapter /scratch/memoozd/cb_runs/195854/outputs/cb_mvp_adapter/final \
        --n 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
):
    """Load model with optional adapter."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    return model, tokenizer


def generate_response(model, tokenizer, messages, tools, max_new_tokens=256):
    """Generate a response for the given messages."""
    # Format with chat template
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate sample outputs for comparison")
    parser.add_argument("--eval-data", type=Path, required=True, help="Path to eval JSONL")
    parser.add_argument("--cb-adapter", type=Path, required=True, help="Path to CB adapter")
    parser.add_argument("--baseline", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tool-schema", type=Path, default=None, help="Path to tool schema JSON")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load eval data
    print(f"Loading eval data from {args.eval_data}...")
    samples = []
    with open(args.eval_data) as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Filter to TRUE tool-flip samples only
    valid_samples = []
    for s in samples:
        labels = s.get("labels", {})
        expected = labels.get("expected_tool", "")
        simulated = labels.get("simulated_tool", "")
        if expected and simulated and expected != simulated:
            valid_samples.append(s)
    
    print(f"Found {len(valid_samples)} valid tool-flip samples")
    samples_to_test = valid_samples[:args.n]
    
    # Load tool schema
    tools = []
    if args.tool_schema and args.tool_schema.exists():
        with open(args.tool_schema) as f:
            tools = json.load(f)
    elif "tools" in samples_to_test[0]:
        tools = samples_to_test[0]["tools"]
    
    # System prompt
    system_prompt = "You are a helpful AI assistant with access to tools. Use the appropriate tool to help the user."
    
    # Load baseline model
    print("\n" + "="*80)
    print("LOADING BASELINE MODEL")
    print("="*80)
    baseline_model, tokenizer = load_model_and_tokenizer(args.baseline, device=args.device)
    
    # Generate baseline outputs
    print("\n" + "="*80)
    print("BASELINE OUTPUTS")
    print("="*80)
    
    baseline_outputs = []
    for i, sample in enumerate(samples_to_test):
        labels = sample.get("labels", {})
        messages = sample.get("messages", [])
        if not messages:
            user_content = sample.get("metadata", {}).get("combined_query", "")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        
        response = generate_response(baseline_model, tokenizer, messages, tools)
        baseline_outputs.append(response)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Expected tool: {labels.get('expected_tool')}")
        print(f"Simulated (attack) tool: {labels.get('simulated_tool')}")
        print(f"User query preview: {messages[-1]['content'][:200]}...")
        print(f"\nBASELINE response:\n{response}")
    
    # Free baseline model memory
    del baseline_model
    torch.cuda.empty_cache()
    
    # Load CB model
    print("\n" + "="*80)
    print("LOADING CB MODEL")
    print("="*80)
    cb_model, tokenizer = load_model_and_tokenizer(
        args.baseline,
        adapter_path=str(args.cb_adapter),
        device=args.device,
    )
    
    # Generate CB outputs
    print("\n" + "="*80)
    print("CB MODEL OUTPUTS")
    print("="*80)
    
    for i, sample in enumerate(samples_to_test):
        labels = sample.get("labels", {})
        messages = sample.get("messages", [])
        if not messages:
            user_content = sample.get("metadata", {}).get("combined_query", "")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        
        response = generate_response(cb_model, tokenizer, messages, tools)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Expected tool: {labels.get('expected_tool')}")
        print(f"Simulated (attack) tool: {labels.get('simulated_tool')}")
        print(f"User query preview: {messages[-1]['content'][:200]}...")
        print(f"\nBASELINE response:\n{baseline_outputs[i]}")
        print(f"\nCB MODEL response:\n{response}")
        
        # Compare
        if baseline_outputs[i] == response:
            print("\n⚠️  IDENTICAL OUTPUT")
        else:
            print("\n✅ DIFFERENT OUTPUT")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
