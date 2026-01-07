#!/usr/bin/env python3
"""
COMPREHENSIVE CIRCUIT BREAKERS PIPELINE DIAGNOSTIC

This script performs a full audit of the CB training pipeline:
1. Data ingestion and format validation
2. Tokenization and masking correctness
3. Model weight updates and isolation
4. Loss computation validation
5. Gradient flow verification

Run before training to ensure everything is working correctly.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import argparse


def print_header(title: str, char="="):
    """Print a formatted header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}\n")


# ============================================================================
# STEP 1: DATA INGESTION AUDIT
# ============================================================================

def audit_data_ingestion(data_path: str, n_samples: int = 3) -> Dict[str, Any]:
    """
    Audit data ingestion and format validation.

    Checks:
    - Batch structure and fields
    - Harmful/benign sample distribution
    - Text vs completion fields
    - Sample quality
    """
    print_header("STEP 1: DATA INGESTION AUDIT")

    # Load batches
    batches = []
    with open(data_path, 'r') as f:
        for line in f:
            batches.append(json.loads(line))

    print(f"✓ Loaded {len(batches)} batches from {data_path}\n")

    # Analyze structure
    first_batch = batches[0]
    print(f"Batch structure:")
    print(f"  Keys: {list(first_batch.keys())}")
    print(f"  Harmful samples per batch: {len(first_batch.get('harmful', []))}")
    print(f"  Benign samples per batch: {len(first_batch.get('benign', []))}")

    # Count completion-style samples
    total_harmful = 0
    total_benign = 0

    # Harmful field counts
    harmful_with_text = 0
    harmful_with_completion = 0  # has 'harmful_completion' field
    harmful_with_user_prompt = 0
    harmful_prompt_only = 0  # has attack_prompt but no completion

    # Benign field counts
    benign_with_text = 0
    benign_with_completion = 0  # has 'benign_completion' field
    benign_with_user_prompt = 0
    benign_prompt_only = 0

    for batch in batches:
        harmful_samples = batch.get('harmful', [])
        benign_samples = batch.get('benign', [])

        total_harmful += len(harmful_samples)
        total_benign += len(benign_samples)

        for sample in harmful_samples:
            has_completion = False
            if sample.get('text'):
                harmful_with_text += 1
                has_completion = True
            if sample.get('harmful_completion'):
                harmful_with_completion += 1
                has_completion = True
            if sample.get('user_prompt'):
                harmful_with_user_prompt += 1
            if not has_completion:
                harmful_prompt_only += 1

        for sample in benign_samples:
            has_completion = False
            if sample.get('text'):
                benign_with_text += 1
                has_completion = True
            if sample.get('benign_completion'):
                benign_with_completion += 1
                has_completion = True
            if sample.get('user_prompt'):
                benign_with_user_prompt += 1
            if not has_completion:
                benign_prompt_only += 1

    print(f"\n{'─' * 80}")
    print(f"Dataset Statistics:")
    print(f"  Total batches: {len(batches)}")
    print(f"  Total harmful samples: {total_harmful}")
    print(f"  Total benign samples: {total_benign}")

    print(f"\n  HARMFUL SAMPLE ANALYSIS:")
    print(f"    With 'text' field (pre-rendered): {harmful_with_text}/{total_harmful} ({harmful_with_text/max(1,total_harmful)*100:.1f}%)")
    print(f"    With 'harmful_completion' field: {harmful_with_completion}/{total_harmful} ({harmful_with_completion/max(1,total_harmful)*100:.1f}%)")
    print(f"    With 'user_prompt' field: {harmful_with_user_prompt}/{total_harmful} ({harmful_with_user_prompt/max(1,total_harmful)*100:.1f}%)")
    print(f"    PROMPT-ONLY (no completion): {harmful_prompt_only}/{total_harmful} ({harmful_prompt_only/max(1,total_harmful)*100:.1f}%)")

    if harmful_prompt_only == total_harmful:
        print(f"\n    ❌ CRITICAL: ALL harmful samples are PROMPT-ONLY!")
        print(f"       CB training will be INEFFECTIVE without harmful completions!")
        print(f"       Run: python scripts/prepare_cb_training.py")
    elif harmful_prompt_only > total_harmful * 0.5:
        print(f"\n    ⚠️  WARNING: {harmful_prompt_only/total_harmful*100:.0f}% of harmful samples lack completions!")
        print(f"       Consider running: python scripts/format_for_cb/split_out_cb_completions.py")
    else:
        print(f"\n    ✓ GOOD: {(total_harmful - harmful_prompt_only)/total_harmful*100:.0f}% of harmful samples have completions")

    print(f"\n  BENIGN SAMPLE ANALYSIS:")
    print(f"    With 'text' field (pre-rendered): {benign_with_text}/{total_benign} ({benign_with_text/max(1,total_benign)*100:.1f}%)")
    print(f"    With 'benign_completion' field: {benign_with_completion}/{total_benign} ({benign_with_completion/max(1,total_benign)*100:.1f}%)")
    print(f"    With 'user_prompt' field: {benign_with_user_prompt}/{total_benign} ({benign_with_user_prompt/max(1,total_benign)*100:.1f}%)")
    print(f"    PROMPT-ONLY (no completion): {benign_prompt_only}/{total_benign} ({benign_prompt_only/max(1,total_benign)*100:.1f}%)")

    # Show sample harmful examples
    print_subheader(f"Sample Harmful Examples (first {n_samples})")

    for i, sample in enumerate(first_batch['harmful'][:n_samples]):
        print(f"\n{'─' * 40} Harmful Sample #{i+1} {'─' * 40}")
        print(f"  ID: {sample.get('id', 'N/A')}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"  Category: {sample.get('category', 'N/A')}")
        print(f"  Fields: {list(sample.keys())}")

        # Check completion status
        has_text = bool(sample.get('text'))
        has_completion = bool(sample.get('harmful_completion'))
        has_user_prompt = bool(sample.get('user_prompt'))

        if has_text or has_completion:
            print(f"\n  ✓ HAS COMPLETION DATA")
        else:
            print(f"\n  ❌ NO COMPLETION DATA (prompt-only)")

        if 'text' in sample and sample['text']:
            text = sample['text']
            print(f"\n  [Pre-rendered Text Field] (length: {len(text)} chars):")
            print(f"    {text[:400]}")
            if len(text) > 400:
                print(f"    ... [truncated {len(text) - 400} chars]")

        if 'user_prompt' in sample and sample['user_prompt']:
            prompt = sample['user_prompt']
            print(f"\n  [User Prompt Field] (length: {len(prompt)} chars):")
            print(f"    {prompt[:300]}")
            if len(prompt) > 300:
                print(f"    ... [truncated]")

        if 'harmful_completion' in sample and sample['harmful_completion']:
            completion = sample['harmful_completion']
            print(f"\n  [Harmful Completion Field] (length: {len(completion)} chars):")
            print(f"    {completion[:400]}")
            if len(completion) > 400:
                print(f"    ... [truncated {len(completion) - 400} chars]")

        if 'attack_prompt' in sample and sample['attack_prompt'] and not has_user_prompt:
            attack = sample['attack_prompt']
            print(f"\n  [Attack Prompt Field - LEGACY] (length: {len(attack)} chars):")
            print(f"    {attack[:300]}")
            if len(attack) > 300:
                print(f"    ... [truncated]")

    # Show sample benign examples
    print_subheader(f"Sample Benign Examples (first {n_samples})")

    for i, sample in enumerate(first_batch['benign'][:n_samples]):
        print(f"\n{'─' * 40} Benign Sample #{i+1} {'─' * 40}")
        print(f"  ID: {sample.get('id', 'N/A')}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"  Category: {sample.get('category', 'N/A')}")
        print(f"  Fields: {list(sample.keys())}")

        # Check completion status
        has_text = bool(sample.get('text'))
        has_completion = bool(sample.get('benign_completion'))
        has_user_prompt = bool(sample.get('user_prompt'))

        if has_text or has_completion:
            print(f"\n  ✓ HAS COMPLETION DATA")
        else:
            print(f"\n  ❌ NO COMPLETION DATA (prompt-only)")

        if 'text' in sample and sample['text']:
            text = sample['text']
            print(f"\n  [Pre-rendered Text Field] (length: {len(text)} chars):")
            print(f"    {text[:400]}")
            if len(text) > 400:
                print(f"    ... [truncated]")

        if 'user_prompt' in sample and sample['user_prompt']:
            prompt = sample['user_prompt']
            print(f"\n  [User Prompt Field] (length: {len(prompt)} chars):")
            print(f"    {prompt[:300]}")
            if len(prompt) > 300:
                print(f"    ... [truncated]")

        if 'benign_completion' in sample and sample['benign_completion']:
            completion = sample['benign_completion']
            print(f"\n  [Benign Completion Field] (length: {len(completion)} chars):")
            print(f"    {completion[:400]}")
            if len(completion) > 400:
                print(f"    ... [truncated]")

        if 'prompt' in sample and sample['prompt'] and not has_user_prompt:
            prompt = sample['prompt']
            print(f"\n  [Prompt Field - LEGACY] (length: {len(prompt)} chars):")
            print(f"    {prompt[:300]}")
            if len(prompt) > 300:
                print(f"    ... [truncated]")

    # Calculate completion percentage
    harmful_with_completions = total_harmful - harmful_prompt_only
    benign_with_completions = total_benign - benign_prompt_only

    return {
        'total_batches': len(batches),
        'total_harmful': total_harmful,
        'total_benign': total_benign,
        'harmful_with_completions': harmful_with_completions,
        'benign_with_completions': benign_with_completions,
        'harmful_prompt_only': harmful_prompt_only,
        'benign_prompt_only': benign_prompt_only,
        'batches': batches[:5]  # Keep first 5 for further testing
    }


# ============================================================================
# STEP 2: TOKENIZATION AND MASKING AUDIT
# ============================================================================

def find_assistant_start_position(text: str) -> int:
    """Find where assistant response starts in chat-formatted text."""
    # Llama format markers
    markers = [
        "[/INST]",
        "<|start_header_id|>assistant<|end_header_id|>",
        "assistant\n",
    ]

    for marker in markers:
        if marker in text:
            idx = text.index(marker)
            return idx + len(marker)

    return len(text)  # If no marker found, entire text is completion


def audit_tokenization_and_masking(
    batches: List[Dict],
    model_name: str,
    max_seq_length: int = 512,
    n_samples: int = 2
) -> Dict[str, Any]:
    """
    Audit tokenization and masking correctness.

    Checks:
    - Tokenization of prompts and completions
    - Completion mask computation
    - Loss mask correctness
    """
    print_header("STEP 2: TOKENIZATION AND MASKING AUDIT")

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Tokenizer loaded")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Process harmful samples
    print_subheader("Harmful Sample Tokenization")

    for i, batch in enumerate(batches[:n_samples]):
        harmful_sample = batch['harmful'][0]

        print(f"\n{'─' * 40} Harmful Batch #{i+1} {'─' * 40}")

        # Extract text
        if 'text' in harmful_sample and harmful_sample['text']:
            text = str(harmful_sample['text'])
            has_completion = True
        elif 'attack_prompt' in harmful_sample:
            # Simulate chat template
            attack_prompt = harmful_sample['attack_prompt']
            messages = [
                {"role": "user", "content": attack_prompt},
                {"role": "assistant", "content": "I cannot help with that request."}  # Dummy completion
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            has_completion = True
        else:
            text = harmful_sample.get('attack_prompt', '')
            has_completion = False

        print(f"\n  Text (length: {len(text)} chars):")
        print(f"    {text[:500]}")
        if len(text) > 500:
            print(f"    ... [truncated {len(text) - 500} chars]")

        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]

        seq_len = attention_mask.sum().item()

        print(f"\n  Tokenization:")
        print(f"    Sequence length (non-pad): {seq_len}/{max_seq_length}")
        print(f"    First 20 tokens: {input_ids[:20].tolist()}")
        print(f"    Decoded first 100 chars: {tokenizer.decode(input_ids[:20])}")

        # Compute completion mask
        if has_completion:
            completion_start_char = find_assistant_start_position(text)
            prompt_text = text[:completion_start_char]
            completion_text = text[completion_start_char:]

            print(f"\n  Completion Detection:")
            print(f"    Prompt length: {len(prompt_text)} chars")
            print(f"    Completion length: {len(completion_text)} chars")
            print(f"    Split at char: {completion_start_char}")

            # Tokenize prompt to find split point
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            prompt_len = len(prompt_tokens)

            print(f"    Prompt token count: {prompt_len}")
            print(f"    Completion starts at token: {prompt_len}")

            # Create loss mask (0 for prompt, 1 for completion)
            loss_mask = torch.zeros_like(input_ids)
            loss_mask[prompt_len:seq_len] = 1

            print(f"\n  Loss Mask:")
            print(f"    Prompt tokens (masked=0): {(loss_mask == 0).sum().item()}")
            print(f"    Completion tokens (masked=1): {(loss_mask == 1).sum().item()}")
            print(f"    Padding tokens: {(attention_mask == 0).sum().item()}")
            print(f"    Mask values (first 30): {loss_mask[:30].tolist()}")

            # Verify only completion tokens are used for loss
            completion_token_ids = input_ids[prompt_len:seq_len]
            print(f"\n  Completion Tokens (will be used for loss):")
            print(f"    Token IDs: {completion_token_ids[:15].tolist()}...")
            print(f"    Decoded: {tokenizer.decode(completion_token_ids[:15])}")
        else:
            print(f"\n  ⚠️  WARNING: No completion detected in this sample!")

    # Process benign samples
    print_subheader("Benign Sample Tokenization")

    for i, batch in enumerate(batches[:n_samples]):
        benign_sample = batch['benign'][0]

        print(f"\n{'─' * 40} Benign Batch #{i+1} {'─' * 40}")

        prompt = benign_sample.get('prompt', '')

        print(f"\n  Prompt (length: {len(prompt)} chars):")
        print(f"    {prompt[:400]}")

        # For benign, we usually only have prompt (no completion in training data)
        # So the entire text is the prompt
        encoded = tokenizer(
            prompt,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]

        seq_len = attention_mask.sum().item()

        print(f"\n  Tokenization:")
        print(f"    Sequence length (non-pad): {seq_len}/{max_seq_length}")
        print(f"    First 20 tokens: {input_ids[:20].tolist()}")
        print(f"    Decoded: {tokenizer.decode(input_ids[:20])}")

    return {
        'tokenizer': tokenizer,
    }


# ============================================================================
# STEP 3: MODEL WEIGHT UPDATES AND ISOLATION
# ============================================================================

def audit_model_weights(
    model_name: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Audit model weight updates and LoRA parameter isolation.

    Checks:
    - Which parameters are trainable (LoRA only)
    - Which parameters are frozen (base model)
    - Parameter counts
    """
    print_header("STEP 3: MODEL WEIGHT UPDATES AND ISOLATION")

    print(f"Loading base model: {model_name}")
    print(f"Device: {device}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"✓ Base model loaded")

    # Count base params
    base_params = sum(p.numel() for p in model.parameters())
    print(f"  Total base parameters: {base_params:,}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"\nApplying LoRA configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    print(f"✓ LoRA applied")

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n{'─' * 80}")
    print(f"Parameter Breakdown:")
    print(f"  Total parameters: {base_params:,}")
    print(f"  Trainable (LoRA): {trainable_params:,} ({trainable_params/base_params*100:.2f}%)")
    print(f"  Frozen (base): {frozen_params:,} ({frozen_params/base_params*100:.2f}%)")

    # Show which params are trainable
    print(f"\n{'─' * 80}")
    print(f"Trainable Parameters (LoRA adapters):")

    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)

    print(f"  Total trainable params: {len(trainable_names)}")
    print(f"  First 10 trainable params:")
    for name in trainable_names[:10]:
        print(f"    - {name}")

    if len(trainable_names) > 10:
        print(f"    ... [{len(trainable_names) - 10} more]")

    # Show which params are frozen
    print(f"\n{'─' * 80}")
    print(f"Frozen Parameters (base model):")

    frozen_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_names.append(name)

    print(f"  Total frozen params: {len(frozen_names)}")
    print(f"  Sample frozen params (first 10):")
    for name in frozen_names[:10]:
        print(f"    - {name}")

    if len(frozen_names) > 10:
        print(f"    ... [{len(frozen_names) - 10} more]")

    return {
        'model': model,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
    }


# ============================================================================
# STEP 4: FROZEN VS TRAINABLE ISOLATION
# ============================================================================

def audit_frozen_trainable_isolation(
    model,
    tokenizer,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Verify frozen and trainable models are isolated.

    Checks:
    - Adapter toggling works
    - Frozen representations differ from trainable
    """
    print_header("STEP 4: FROZEN VS TRAINABLE ISOLATION")

    # Create a test input
    test_text = "Hello, how are you today?"

    print(f"Test input: '{test_text}'")

    encoded = tokenizer(
        test_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"  Tokenized length: {input_ids.shape[1]}")

    # Forward pass with adapters ENABLED (trainable)
    print(f"\n{'─' * 80}")
    print(f"Forward Pass: Adapters ENABLED (trainable representations)")

    model.eval()
    with torch.no_grad():
        outputs_trainable = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    trainable_hidden = outputs_trainable.hidden_states[-1]  # Last layer
    print(f"  Hidden states shape: {trainable_hidden.shape}")
    print(f"  Hidden states mean: {trainable_hidden.mean().item():.6f}")
    print(f"  Hidden states std: {trainable_hidden.std().item():.6f}")
    print(f"  Sample values: {trainable_hidden[0, 0, :5].tolist()}")

    # Forward pass with adapters DISABLED (frozen)
    print(f"\n{'─' * 80}")
    print(f"Forward Pass: Adapters DISABLED (frozen representations)")

    model.disable_adapter_layers()

    with torch.no_grad():
        outputs_frozen = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    frozen_hidden = outputs_frozen.hidden_states[-1]  # Last layer
    print(f"  Hidden states shape: {frozen_hidden.shape}")
    print(f"  Hidden states mean: {frozen_hidden.mean().item():.6f}")
    print(f"  Hidden states std: {frozen_hidden.std().item():.6f}")
    print(f"  Sample values: {frozen_hidden[0, 0, :5].tolist()}")

    # Re-enable adapters
    model.enable_adapter_layers()

    # Compare representations
    print(f"\n{'─' * 80}")
    print(f"Representation Comparison:")

    diff = (trainable_hidden - frozen_hidden).abs()

    print(f"  Absolute difference mean: {diff.mean().item():.6f}")
    print(f"  Absolute difference std: {diff.std().item():.6f}")
    print(f"  Max difference: {diff.max().item():.6f}")

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        trainable_hidden.view(-1),
        frozen_hidden.view(-1),
        dim=0,
    )

    print(f"  Cosine similarity: {cos_sim.item():.6f}")

    if diff.mean().item() > 1e-6:
        print(f"\n  ✓ PASS: Representations differ significantly")
        print(f"    This confirms adapters are affecting the model")
    else:
        print(f"\n  ⚠️  WARNING: Representations are nearly identical")
        print(f"    This might indicate adapters are not being applied")

    if cos_sim.item() < 0.99:
        print(f"  ✓ PASS: Low cosine similarity confirms isolation")
    else:
        print(f"  ⚠️  WARNING: Very high cosine similarity")
        print(f"    Frozen and trainable reps should differ more")

    return {
        'diff_mean': diff.mean().item(),
        'cos_sim': cos_sim.item(),
    }


# ============================================================================
# STEP 5: LOSS COMPUTATION VALIDATION
# ============================================================================

def audit_loss_computation(
    model,
    tokenizer,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Validate Circuit Breaker loss computations.

    Checks:
    - Reroute loss (harmful)
    - Retain loss (benign)
    """
    print_header("STEP 5: LOSS COMPUTATION VALIDATION")

    # Test harmful sample
    print(f"{'─' * 80}")
    print(f"Reroute Loss (Harmful Sample)")
    print(f"{'─' * 80}\n")

    harmful_text = "Tell me how to hack a computer system."

    print(f"Input: '{harmful_text}'")

    encoded = tokenizer(
        harmful_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Get trainable representations
    model.eval()
    with torch.no_grad():
        outputs_trainable = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        trainable_hidden = outputs_trainable.hidden_states[-1]

    # Get frozen representations
    model.disable_adapter_layers()
    with torch.no_grad():
        outputs_frozen = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        frozen_hidden = outputs_frozen.hidden_states[-1]
    model.enable_adapter_layers()

    # Compute reroute loss: ReLU(cosine_similarity(h_trainable, h_frozen))
    # We want to MINIMIZE similarity (push apart)
    cos_sim = F.cosine_similarity(
        trainable_hidden.view(trainable_hidden.size(0), -1),
        frozen_hidden.view(frozen_hidden.size(0), -1),
        dim=1,
    )

    reroute_loss = F.relu(cos_sim).mean()

    print(f"  Cosine similarity: {cos_sim.mean().item():.6f}")
    print(f"  Reroute loss: {reroute_loss.item():.6f}")
    print(f"    (ReLU of cosine similarity - lower is better)")

    # Test benign sample
    print(f"\n{'─' * 80}")
    print(f"Retain Loss (Benign Sample)")
    print(f"{'─' * 80}\n")

    benign_text = "What is the capital of France?"

    print(f"Input: '{benign_text}'")

    encoded = tokenizer(
        benign_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Get trainable representations
    with torch.no_grad():
        outputs_trainable = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        trainable_hidden = outputs_trainable.hidden_states[-1]

    # Get frozen representations
    model.disable_adapter_layers()
    with torch.no_grad():
        outputs_frozen = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        frozen_hidden = outputs_frozen.hidden_states[-1]
    model.enable_adapter_layers()

    # Compute retain loss: L2 distance (we want to MINIMIZE - keep close)
    retain_loss = F.mse_loss(trainable_hidden, frozen_hidden)

    l2_distance = torch.norm(trainable_hidden - frozen_hidden).item()

    print(f"  L2 distance: {l2_distance:.6f}")
    print(f"  Retain loss (MSE): {retain_loss.item():.6f}")
    print(f"    (MSE between trainable and frozen - lower is better)")

    return {
        'reroute_loss': reroute_loss.item(),
        'retain_loss': retain_loss.item(),
    }


# ============================================================================
# STEP 6: GRADIENT FLOW VERIFICATION
# ============================================================================

def audit_gradient_flow(
    model,
    tokenizer,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Verify gradients only flow to trainable (LoRA) parameters.

    Checks:
    - Backward pass
    - Gradients on LoRA params
    - No gradients on frozen params
    """
    print_header("STEP 6: GRADIENT FLOW VERIFICATION")

    # Create test input
    test_text = "This is a test input for gradient checking."

    print(f"Test input: '{test_text}'")

    encoded = tokenizer(
        test_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Zero gradients
    model.zero_grad()

    # Forward pass
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,  # Simple LM loss
    )

    loss = outputs.loss

    print(f"  Loss: {loss.item():.6f}")

    # Backward pass
    print(f"\nComputing gradients...")
    loss.backward()

    # Check gradients
    print(f"\n{'─' * 80}")
    print(f"Gradient Analysis:")

    trainable_with_grad = 0
    trainable_without_grad = 0
    frozen_with_grad = 0
    frozen_without_grad = 0

    trainable_grad_norms = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                trainable_with_grad += 1
                grad_norm = param.grad.norm().item()
                trainable_grad_norms.append((name, grad_norm))
            else:
                trainable_without_grad += 1
        else:
            if param.grad is not None:
                frozen_with_grad += 1
            else:
                frozen_without_grad += 1

    print(f"  Trainable params with gradients: {trainable_with_grad}")
    print(f"  Trainable params WITHOUT gradients: {trainable_without_grad}")
    print(f"  Frozen params with gradients: {frozen_with_grad}")
    print(f"  Frozen params WITHOUT gradients: {frozen_without_grad}")

    if trainable_with_grad > 0:
        print(f"\n  ✓ PASS: Gradients flow to trainable (LoRA) parameters")
    else:
        print(f"\n  ❌ FAIL: No gradients on trainable parameters!")

    if frozen_with_grad == 0:
        print(f"  ✓ PASS: No gradients on frozen (base) parameters")
    else:
        print(f"  ⚠️  WARNING: {frozen_with_grad} frozen params have gradients!")

    # Show top gradients
    trainable_grad_norms.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'─' * 80}")
    print(f"Top 10 Gradient Magnitudes (Trainable Params):")

    for name, grad_norm in trainable_grad_norms[:10]:
        print(f"  {grad_norm:.6f}  -  {name}")

    return {
        'trainable_with_grad': trainable_with_grad,
        'frozen_with_grad': frozen_with_grad,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Circuit Breaker pipeline diagnostic"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/circuit_breakers/cb_training_batches.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run tests on",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of samples to show in each test",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"  CIRCUIT BREAKERS PIPELINE DIAGNOSTIC")
    print(f"{'=' * 80}\n")
    print(f"Configuration:")
    print(f"  Data: {args.data_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Sample count: {args.n_samples}")

    # Step 1: Data ingestion
    data_stats = audit_data_ingestion(args.data_path, n_samples=args.n_samples)

    # Step 2: Tokenization and masking
    tok_stats = audit_tokenization_and_masking(
        data_stats['batches'],
        args.model_name,
        max_seq_length=args.max_seq_length,
        n_samples=args.n_samples,
    )

    # Step 3: Model weights
    model_stats = audit_model_weights(args.model_name, device=args.device)

    # Step 4: Frozen vs trainable isolation
    isolation_stats = audit_frozen_trainable_isolation(
        model_stats['model'],
        tok_stats['tokenizer'],
        device=args.device,
    )

    # Step 5: Loss computation
    loss_stats = audit_loss_computation(
        model_stats['model'],
        tok_stats['tokenizer'],
        device=args.device,
    )

    # Step 6: Gradient flow
    grad_stats = audit_gradient_flow(
        model_stats['model'],
        tok_stats['tokenizer'],
        device=args.device,
    )

    # Final summary
    print_header("DIAGNOSTIC SUMMARY", char="#")

    print(f"Data Ingestion:")
    print(f"  ✓ {data_stats['total_batches']} batches loaded")
    print(f"  ✓ {data_stats['total_harmful']} harmful samples")
    print(f"  ✓ {data_stats['total_benign']} benign samples")

    # Check completion coverage
    harmful_with_comp = data_stats.get('harmful_with_completions', 0)
    harmful_prompt_only = data_stats.get('harmful_prompt_only', data_stats['total_harmful'])

    if harmful_with_comp > 0:
        pct = harmful_with_comp / data_stats['total_harmful'] * 100
        print(f"  ✓ {harmful_with_comp}/{data_stats['total_harmful']} harmful have completions ({pct:.1f}%)")
    else:
        print(f"  ❌ {harmful_with_comp}/{data_stats['total_harmful']} harmful have completions (0%)")
        print(f"     WARNING: CB training will be INEFFECTIVE without completions!")
        print(f"     Run: python scripts/prepare_cb_training.py")

    print(f"\nModel Configuration:")
    print(f"  ✓ {model_stats['trainable_params']:,} trainable (LoRA) params")
    print(f"  ✓ {model_stats['frozen_params']:,} frozen (base) params")

    print(f"\nRepresentation Isolation:")
    print(f"  ✓ Diff mean: {isolation_stats['diff_mean']:.6f}")
    print(f"  ✓ Cosine sim: {isolation_stats['cos_sim']:.6f}")

    print(f"\nLoss Computation:")
    print(f"  ✓ Reroute loss: {loss_stats['reroute_loss']:.6f}")
    print(f"  ✓ Retain loss: {loss_stats['retain_loss']:.6f}")

    print(f"\nGradient Flow:")
    print(f"  ✓ Trainable with grads: {grad_stats['trainable_with_grad']}")
    print(f"  ✓ Frozen with grads: {grad_stats['frozen_with_grad']}")

    # Overall verdict
    print(f"\n{'─' * 80}")
    issues = []
    if harmful_with_comp == 0:
        issues.append("No harmful completions - run prepare_cb_training.py")
    if isolation_stats['diff_mean'] < 1e-6:
        issues.append("Adapter isolation not verified (expected for fresh adapters)")
    if grad_stats['trainable_with_grad'] == 0:
        issues.append("No gradients on trainable params")

    if not issues:
        print(f"  ✅ ALL CHECKS PASSED - Ready for training!")
    else:
        print(f"  ⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"     - {issue}")

    print(f"\n{'#' * 80}")
    print(f"  DIAGNOSTICS COMPLETE")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
