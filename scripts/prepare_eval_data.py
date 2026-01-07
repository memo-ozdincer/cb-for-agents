#!/usr/bin/env python3
"""
Prepare evaluation datasets from training data.

Extracts held-out samples for evaluation:
- Harmful prompts (for refusal rate)
- Benign prompts (for capability preservation)
- Agent prompts (for action-based safety)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_training_batches(data_path: str) -> List[Dict[str, Any]]:
    """Load training batches."""
    batches = []
    with open(data_path, 'r') as f:
        for line in f:
            batch = json.loads(line.strip())
            batches.append(batch)
    return batches


def extract_eval_samples(
    batches: List[Dict[str, Any]],
    n_harmful: int = 200,
    n_benign: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Extract evaluation samples from batches.

    Args:
        batches: Training batches
        n_harmful: Number of harmful eval samples
        n_benign: Number of benign eval samples
        seed: Random seed

    Returns:
        (harmful_prompts, benign_prompts) tuple
    """
    random.seed(seed)

    harmful_prompts = []
    benign_prompts = []

    for batch in batches:
        # Extract harmful samples
        if 'harmful_samples' in batch:
            for sample in batch['harmful_samples']:
                prompt = sample.get('prompt', '')
                if prompt and len(harmful_prompts) < n_harmful * 2:
                    harmful_prompts.append({
                        'prompt': prompt,
                        'type': sample.get('type', 'unknown'),
                    })

        # Extract benign samples
        if 'benign_samples' in batch:
            for sample in batch['benign_samples']:
                prompt = sample.get('prompt', '')
                if prompt and len(benign_prompts) < n_benign * 2:
                    benign_prompts.append({
                        'prompt': prompt,
                        'type': sample.get('type', 'unknown'),
                    })

    # Shuffle and take subsets
    random.shuffle(harmful_prompts)
    random.shuffle(benign_prompts)

    return harmful_prompts[:n_harmful], benign_prompts[:n_benign]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare evaluation data")
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/circuit_breakers/cb_training_batches.jsonl",
        help="Path to training batches",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/circuit_breakers/eval",
        help="Output directory for eval data",
    )
    parser.add_argument(
        "--n-harmful",
        type=int,
        default=200,
        help="Number of harmful samples",
    )
    parser.add_argument(
        "--n-benign",
        type=int,
        default=200,
        help="Number of benign samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print(f"Loading training data from {args.training_data}...")
    batches = load_training_batches(args.training_data)
    print(f"Loaded {len(batches)} batches")

    print(f"\nExtracting {args.n_harmful} harmful + {args.n_benign} benign samples...")
    harmful, benign = extract_eval_samples(
        batches,
        n_harmful=args.n_harmful,
        n_benign=args.n_benign,
        seed=args.seed,
    )

    print(f"Extracted {len(harmful)} harmful, {len(benign)} benign prompts")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save harmful prompts
    harmful_path = output_dir / "harmful_eval.jsonl"
    with open(harmful_path, 'w') as f:
        for sample in harmful:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved harmful prompts to {harmful_path}")

    # Save benign prompts
    benign_path = output_dir / "benign_eval.jsonl"
    with open(benign_path, 'w') as f:
        for sample in benign:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved benign prompts to {benign_path}")

    print("\n✅ Eval data preparation complete!")


if __name__ == "__main__":
    main()
