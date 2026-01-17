#!/usr/bin/env python3
"""
Stage 2: Ingest UltraChat for General Conversation Capability

This script downloads and formats a subset of UltraChat for the retain set (Dr).
UltraChat provides general conversation capability without tool calls.

Purpose:
- Maintain general text generation capability
- Prevent model from becoming tool-call-only
- From the original CB paper: essential for maintaining general capability

Source: HuggingFaceH4/ultrachat_200k (test_sft split)

Usage:
    python scripts/cb_data_generation/ingest_ultrachat.py \
        --output data/circuit_breakers/retain/ultrachat_subset.jsonl \
        --n-samples 2000 \
        --seed 42
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# ID Generation (per canonical schema)
# =============================================================================

def generate_sample_id(source: str, split: str, content_hash_input: str) -> str:
    """Generate deterministic, collision-resistant ID per 01_DATA_SPEC.md."""
    content_hash = hashlib.sha256(content_hash_input.encode()).hexdigest()[:8]
    return f"{source}_{split}_{content_hash}"


def format_ultrachat_sample(item: Dict[str, Any], timestamp: str) -> Optional[Dict[str, Any]]:
    """
    Format an UltraChat sample for CB training with canonical schema (01_DATA_SPEC.md).

    UltraChat format:
    - messages: List of {"role": "user"|"assistant", "content": "..."}

    We need:
    - messages: All but last message (context)
    - assistant_raw: Last assistant message (completion)
    """
    messages = item.get("messages", [])

    if len(messages) < 2:
        return None

    # Last message should be assistant
    if messages[-1].get("role") != "assistant":
        return None

    # Extract context (all but last) and completion (last)
    context_messages = []
    for msg in messages[:-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            context_messages.append({"role": role, "content": content})

    if not context_messages:
        return None

    assistant_raw = messages[-1].get("content", "")
    if not assistant_raw or len(assistant_raw) < 10:
        return None

    # Generate deterministic ID
    content_hash_input = json.dumps({
        "messages": context_messages,
        "assistant_raw": assistant_raw,
    }, sort_keys=True)
    sample_id = generate_sample_id("ultrachat", "retain", content_hash_input)

    # Canonical schema
    return {
        "id": sample_id,
        "messages": context_messages,
        "assistant_raw": assistant_raw,
        "tools": None,  # UltraChat has no tool calls

        # Labels for loss masking & filtering
        "labels": {
            "split": "retain",
        },

        # Training controls
        "training": {
            "priority_class": "general_conversation",
            "sample_weight": 0.8,  # Lower priority per spec
        },

        # Provenance metadata
        "metadata": {
            "source": "ultrachat",
            "has_tool_calls": False,
            "schema_version": "stage2_v1",
            "created_at": timestamp,
            "pipeline_version": "stage2_v1",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest UltraChat for general conversation capability"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/retain/ultrachat_subset.jsonl",
        help="Output path for UltraChat samples",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of samples to extract",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_sft",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum total length of messages (chars)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached dataset only (no download)",
    )
    
    args = parser.parse_args()

    # Create timestamp once for all samples
    timestamp = datetime.now(timezone.utc).isoformat()

    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        logger.info("Creating placeholder samples for testing...")

        # Create placeholder samples with canonical schema
        samples = []
        for i in range(min(args.n_samples, 100)):
            messages = [
                {"role": "user", "content": f"Can you explain how {['AI', 'machine learning', 'neural networks', 'data science'][i % 4]} works?"}
            ]
            assistant_raw = f"I'd be happy to explain! This is a complex topic that involves many interesting concepts..."

            content_hash_input = json.dumps({
                "messages": messages,
                "assistant_raw": assistant_raw,
                "index": i,
            }, sort_keys=True)
            sample_id = generate_sample_id("ultrachat", "retain", content_hash_input)

            sample = {
                "id": sample_id,
                "messages": messages,
                "assistant_raw": assistant_raw,
                "tools": None,
                "labels": {"split": "retain"},
                "training": {
                    "priority_class": "general_conversation",
                    "sample_weight": 0.8,
                },
                "metadata": {
                    "source": "ultrachat_placeholder",
                    "has_tool_calls": False,
                    "schema_version": "stage2_v1",
                    "created_at": timestamp,
                    "pipeline_version": "stage2_v1",
                },
            }
            samples.append(sample)

        logger.info(f"Created {len(samples)} placeholder samples")

        # Save and exit
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(samples)} placeholder UltraChat samples to {args.output}")
        return

    # Load dataset
    logger.info(f"Loading UltraChat dataset (split: {args.split})...")
    try:
        if args.offline:
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split=args.split,
                download_mode="reuse_cache_if_exists",
            )
        else:
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split=args.split,
            )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Try running with internet access first, then use --offline")
        sys.exit(1)

    logger.info(f"Loaded {len(ds)} samples from {args.split}")

    # Shuffle and select
    ds = ds.shuffle(seed=args.seed)

    # Process samples
    samples = []
    skipped = 0

    for item in ds:
        if len(samples) >= args.n_samples:
            break

        sample = format_ultrachat_sample(item, timestamp)

        if sample is None:
            skipped += 1
            continue

        # Check length
        total_length = sum(
            len(str(m.get("content", ""))) for m in sample["messages"]
        ) + len(sample["assistant_raw"])

        if total_length > args.max_length:
            skipped += 1
            continue

        samples.append(sample)

    logger.info(f"Collected {len(samples)} samples (skipped {skipped})")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save samples
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(samples)} UltraChat samples to {args.output}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ULTRACHAT INGEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Samples: {len(samples)}")
    logger.info(f"Purpose: Dr (retain) - general conversation capability")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
