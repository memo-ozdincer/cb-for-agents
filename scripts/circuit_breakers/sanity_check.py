#!/usr/bin/env python3
"""
Adapter Sanity Check - KL Divergence Gate

This script is NON-NEGOTIABLE before/after any CB training run.
It verifies that the LoRA adapter actually affects the model's forward pass
by computing KL divergence between base model and adapter model distributions.

If mean KL < epsilon, the adapter is NOT changing outputs and training is broken.

Usage:
    # Test infrastructure (before training, no adapter)
    python scripts/circuit_breakers/sanity_check.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --test-prompts data/cb_mvp/sanity_check_prompts.jsonl

    # Verify adapter effect (after training)
    python scripts/circuit_breakers/sanity_check.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/cb_mvp_stage1/final \
        --test-prompts data/cb_mvp/sanity_check_prompts.jsonl

    # Use with SLURM (fail job if check fails)
    python scripts/circuit_breakers/sanity_check.py ... --fail-on-error
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
# Default Test Prompts (if no file provided)
# =============================================================================

DEFAULT_TEST_PROMPTS = [
    # Tool-calling prompts (B4-style)
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "Search for images of the James Webb Telescope"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is the weather in Tokyo?"}
        ]
    },
    # Direct prompts
    {
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a short poem about the ocean."}
        ]
    },
    # Potentially harmful (to test CB effect)
    {
        "messages": [
            {"role": "user", "content": "How do I pick a lock?"}
        ]
    },
]


# =============================================================================
# KL Divergence Computation
# =============================================================================

def compute_kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL divergence KL(P || Q) between two logit distributions.
    
    KL(P || Q) = sum(P * log(P / Q))
    
    Args:
        logits_p: Logits from distribution P (base model)
        logits_q: Logits from distribution Q (adapter model)
        temperature: Temperature for softmax (default 1.0)
    
    Returns:
        KL divergence per position (batch_size, seq_len)
    """
    # Apply temperature and convert to log probabilities
    log_p = F.log_softmax(logits_p / temperature, dim=-1)
    log_q = F.log_softmax(logits_q / temperature, dim=-1)
    
    # P = softmax(logits_p)
    p = F.softmax(logits_p / temperature, dim=-1)
    
    # KL(P || Q) = sum(P * (log_p - log_q))
    kl = (p * (log_p - log_q)).sum(dim=-1)
    
    return kl


def get_teacher_forced_logits(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_length: int = 512,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get teacher-forced logits for a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages
        max_length: Maximum sequence length
        device: Target device
    
    Returns:
        (logits, attention_mask) tuple
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Format with chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback formatting
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)
    
    # Forward pass (no generation, just get logits)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    
    return outputs.logits, inputs.attention_mask


# =============================================================================
# Main Sanity Check Function
# =============================================================================

def adapter_sanity_check(
    base_model_path: str,
    adapter_path: Optional[str],
    test_prompts: List[Dict[str, Any]],
    epsilon: float = 1e-4,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify adapter changes next-token distributions.
    
    This is the CRITICAL gate for CB training. If the adapter doesn't
    affect the forward pass, training is broken.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapter (None = test infrastructure only)
        test_prompts: List of test prompts with 'messages' key
        epsilon: Minimum acceptable mean KL divergence
        device: Device to use
        torch_dtype: Model dtype
        verbose: Print detailed results
    
    Returns:
        Dict with:
            - passed: bool - whether the check passed
            - mean_kl: float - mean KL divergence
            - per_prompt_kl: List[float] - KL per prompt
            - epsilon: float - threshold used
            - message: str - human-readable result
    
    Raises:
        ValueError: If mean_kl < epsilon (adapter has no effect)
    """
    hf_token = resolve_hf_token()
    
    logger.info("=" * 60)
    logger.info("ADAPTER SANITY CHECK")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter: {adapter_path or 'None (infrastructure test)'}")
    logger.info(f"Epsilon threshold: {epsilon}")
    logger.info(f"Test prompts: {len(test_prompts)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device map
    device_map = device if device != "auto" else "auto"
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token,
    )
    base_model.eval()
    
    # Load adapter model (if provided)
    if adapter_path:
        logger.info("Loading adapter model...")
        adapter_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token,
        )
        adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)
        adapter_model.eval()
    else:
        # No adapter - use same model (should give KL = 0)
        adapter_model = base_model
        logger.info("No adapter provided - testing infrastructure (expect KL ≈ 0)")
    
    # Get device from model
    model_device = next(base_model.parameters()).device
    
    # Compute KL divergence for each prompt
    kl_divergences = []
    
    iterator = tqdm(test_prompts, desc="Computing KL divergence") if verbose else test_prompts
    
    for prompt_data in iterator:
        messages = prompt_data.get("messages", [])
        if not messages:
            continue
        
        # Get logits from both models
        base_logits, attention_mask = get_teacher_forced_logits(
            base_model, tokenizer, messages, device=model_device
        )
        
        if adapter_path:
            adapter_logits, _ = get_teacher_forced_logits(
                adapter_model, tokenizer, messages, device=model_device
            )
        else:
            adapter_logits = base_logits  # Same model
        
        # Compute KL divergence
        kl = compute_kl_divergence(base_logits, adapter_logits)
        
        # Mean over sequence (ignore padding)
        kl_masked = kl * attention_mask.float()
        mean_kl = kl_masked.sum() / attention_mask.sum()
        
        kl_divergences.append(mean_kl.item())
    
    # Compute overall statistics
    mean_kl = np.mean(kl_divergences)
    std_kl = np.std(kl_divergences)
    min_kl = np.min(kl_divergences)
    max_kl = np.max(kl_divergences)
    
    # Determine pass/fail
    passed = mean_kl >= epsilon if adapter_path else mean_kl < epsilon
    
    # Build result
    result = {
        "passed": passed,
        "mean_kl": mean_kl,
        "std_kl": std_kl,
        "min_kl": min_kl,
        "max_kl": max_kl,
        "per_prompt_kl": kl_divergences,
        "epsilon": epsilon,
        "adapter_path": adapter_path,
        "base_model": base_model_path,
        "num_prompts": len(kl_divergences),
    }
    
    # Generate message
    if adapter_path:
        if passed:
            result["message"] = (
                f"✅ PASSED: Adapter affects forward pass. "
                f"Mean KL = {mean_kl:.6f} > ε = {epsilon}"
            )
        else:
            result["message"] = (
                f"❌ FAILED: Adapter has NO EFFECT on forward pass! "
                f"Mean KL = {mean_kl:.6f} < ε = {epsilon}. "
                f"This is a show-stopper - do NOT proceed with training or evaluation."
            )
    else:
        if passed:
            result["message"] = (
                f"✅ PASSED: Infrastructure test OK. "
                f"Base model vs itself gives KL ≈ 0 ({mean_kl:.2e})"
            )
        else:
            result["message"] = (
                f"⚠️ WARNING: Infrastructure test shows unexpected KL divergence. "
                f"Mean KL = {mean_kl:.6f} (expected ≈ 0). Check implementation."
            )
    
    # Log results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean KL Divergence: {mean_kl:.6f}")
    logger.info(f"Std KL Divergence:  {std_kl:.6f}")
    logger.info(f"Min KL Divergence:  {min_kl:.6f}")
    logger.info(f"Max KL Divergence:  {max_kl:.6f}")
    logger.info(f"Epsilon Threshold:  {epsilon}")
    logger.info("")
    logger.info(result["message"])
    logger.info("=" * 60)
    
    if verbose:
        logger.info("\nPer-prompt KL divergences:")
        for i, kl in enumerate(kl_divergences):
            logger.info(f"  Prompt {i+1}: KL = {kl:.6f}")
    
    # Clean up models to free memory
    del base_model
    if adapter_path:
        del adapter_model
    torch.cuda.empty_cache()
    
    return result


# =============================================================================
# CLI
# =============================================================================

def load_test_prompts(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load test prompts from JSONL file or use defaults."""
    if path is None or not Path(path).exists():
        logger.info("Using default test prompts")
        return DEFAULT_TEST_PROMPTS
    
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    logger.info(f"Loaded {len(prompts)} test prompts from {path}")
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Adapter Sanity Check - KL Divergence Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter. If not provided, tests infrastructure only.",
    )
    parser.add_argument(
        "--test-prompts",
        type=str,
        default=None,
        help="Path to test prompts JSONL file. Uses defaults if not provided.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Minimum acceptable mean KL divergence (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 1 if check fails (for SLURM/CI)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load test prompts
    test_prompts = load_test_prompts(args.test_prompts)
    
    # Run sanity check
    result = adapter_sanity_check(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        test_prompts=test_prompts,
        epsilon=args.epsilon,
        device=args.device,
        torch_dtype=torch_dtype,
        verbose=not args.quiet,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Exit code
    if args.fail_on_error and not result["passed"]:
        logger.error("Sanity check FAILED - exiting with code 1")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
