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
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def resolve_local_model_path(model_id: str) -> str:
    """
    Resolve a HuggingFace model ID to its local cache path.
    Required in offline mode to avoid API calls during tokenizer loading.
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
        )
        print(f"Resolved {model_id} -> {local_path}")
        return local_path
    except Exception as e:
        print(f"Warning: Could not resolve local path for {model_id}: {e}")
        return model_id


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
):
    """Load model with optional adapter."""
    # In offline mode, resolve Hub ID to local cache path to avoid API calls
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    if offline_mode:
        model_path = resolve_local_model_path(model_path)
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=offline_mode,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=offline_mode,
    )
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    return model, tokenizer


def generate_response(model, tokenizer, messages, tools, max_new_tokens=256):
    """Generate a response for the given messages."""
    # Format with chat template
    # If tools is empty or None, don't pass tools arg
    if tools:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
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
    tools = None
    if args.tool_schema and args.tool_schema.exists():
        with open(args.tool_schema) as f:
            schema_data = json.load(f)
            # Handle wrapped format (our b4_standard_v1.json has {"tools": [...]})
            if "tools" in schema_data and isinstance(schema_data["tools"], list):
                tools = schema_data["tools"]
            elif isinstance(schema_data, list):
                tools = schema_data
            else:
                tools = schema_data
    elif samples_to_test and "tools" in samples_to_test[0]:
        tools = samples_to_test[0]["tools"]
    
    # Debug: show tools format
    if tools:
        print(f"Loaded {len(tools)} tools")
        if tools and len(tools) > 0:
            print(f"First tool format: {json.dumps(tools[0], indent=2)[:200]}...")
    else:
        print("WARNING: No tools loaded - will generate without tool context")
    
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
