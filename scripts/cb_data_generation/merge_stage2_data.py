#!/usr/bin/env python3
"""
Stage 2: Merge Data for Circuit Breaker Training

This script merges all harmful and retain data sources into a single
training file with the target Dr:Ds ratio (default 5:1).

Data Sources:
  Ds (harmful/shut-down):
    - Fujitsu B4 harmful samples
    - AgentDojo injection failures
    - Existing CB harmful data
    
  Dr (retain/safe):
    - Adversarial-safe samples (model resisted injection)
    - AgentDojo successful resistance samples
    - UltraChat general conversation
    - TAU2 customer service traces
    - XSTest borderline cases

Usage:
    python scripts/cb_data_generation/merge_stage2_data.py \
        --output data/circuit_breakers/stage2_training.jsonl \
        --dr-ratio 5.0
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL file, return list of samples."""
    samples = []
    
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return samples
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {path.name}: {e}")
                continue
    
    return samples


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate a sample has required fields."""
    if not isinstance(sample, dict):
        return False
    
    # Must have messages or context
    has_context = "messages" in sample or "context" in sample
    
    # Must have response
    has_response = "assistant_raw" in sample or "completion" in sample
    
    return has_context and has_response


def normalize_sample(sample: Dict[str, Any], split: str, source: str) -> Dict[str, Any]:
    """Normalize sample to standard format."""
    # Handle messages/context field
    if "messages" not in sample and "context" in sample:
        sample["messages"] = sample.pop("context")
    
    # Handle assistant_raw/completion field
    if "assistant_raw" not in sample and "completion" in sample:
        sample["assistant_raw"] = sample.pop("completion")
    
    # Ensure metadata exists
    if "metadata" not in sample:
        sample["metadata"] = {}
    
    # Set split and source
    sample["metadata"]["split"] = split
    sample["metadata"]["source"] = source
    
    # Ensure labels exists
    if "labels" not in sample:
        sample["labels"] = {}
    
    return sample


# =============================================================================
# Data Collection
# =============================================================================

def collect_harmful_data(base_dir: Path, max_per_source: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """Collect all harmful (Ds) data sources."""
    all_samples = []
    source_counts = {}
    
    # Potential paths for harmful data
    harmful_paths = [
        # Stage 2 generated data
        (base_dir / "data/circuit_breakers/harmful/fujitsu_b4_harmful.jsonl", "b4_harmful"),
        (base_dir / "data/circuit_breakers/harmful/agentdojo_failures.jsonl", "agentdojo_failures"),
        
        # Existing CB data
        (base_dir / "data/circuit_breakers/cb_training_harmful.jsonl", "cb_harmful"),
        (base_dir / "data/cb_mvp/cb_training_harmful.jsonl", "cb_harmful_mvp"),
        
        # MVP data
        (base_dir / "data/cb_mvp/harmful.jsonl", "mvp_harmful"),
        
        # Agent-harm data
        (base_dir / "data/agent_harm/agent_harm_harmful.jsonl", "agent_harm"),
    ]
    
    for path, source_name in harmful_paths:
        if path.exists():
            samples = load_jsonl(path, max_per_source)
            
            # Normalize and validate
            valid_samples = []
            for s in samples:
                if validate_sample(s):
                    normalized = normalize_sample(s, "harmful", source_name)
                    valid_samples.append(normalized)
            
            all_samples.extend(valid_samples)
            source_counts[source_name] = len(valid_samples)
            logger.info(f"  {source_name}: {len(valid_samples)} samples")
    
    return all_samples, source_counts


def collect_retain_data(base_dir: Path, max_per_source: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """Collect all retain (Dr) data sources with priority weighting."""
    all_samples = []
    source_counts = {}
    
    # Priority order for retain data (critical first)
    retain_sources = [
        # CRITICAL: Adversarial-safe samples (model resisted injection)
        {
            "path": base_dir / "data/circuit_breakers/retain/adversarial_safe.jsonl",
            "name": "adversarial_safe",
            "priority": "critical",
            "weight": 2.0,
        },
        # AgentDojo successful resistance
        {
            "path": base_dir / "data/circuit_breakers/retain/agentdojo_resisted.jsonl",
            "name": "agentdojo_resisted",
            "priority": "high",
            "weight": 1.5,
        },
        # TAU2 customer service
        {
            "path": base_dir / "data/circuit_breakers/retain/tau2_traces.jsonl",
            "name": "tau2",
            "priority": "high",
            "weight": 1.2,
        },
        # XSTest borderline
        {
            "path": base_dir / "data/circuit_breakers/retain/xstest_borderline.jsonl",
            "name": "xstest",
            "priority": "medium",
            "weight": 1.0,
        },
        # UltraChat general
        {
            "path": base_dir / "data/circuit_breakers/retain/ultrachat_subset.jsonl",
            "name": "ultrachat",
            "priority": "medium",
            "weight": 1.0,
        },
        # Existing retain data
        {
            "path": base_dir / "data/circuit_breakers/cb_training_retain.jsonl",
            "name": "cb_retain",
            "priority": "low",
            "weight": 0.8,
        },
        {
            "path": base_dir / "data/cb_mvp/retain.jsonl",
            "name": "mvp_retain",
            "priority": "low",
            "weight": 0.8,
        },
    ]
    
    for source in retain_sources:
        path = source["path"]
        source_name = source["name"]
        weight = source["weight"]
        
        if path.exists():
            samples = load_jsonl(path, max_per_source)
            
            # Normalize and validate
            valid_samples = []
            for s in samples:
                if validate_sample(s):
                    normalized = normalize_sample(s, "retain", source_name)
                    # Add weight to metadata
                    normalized["metadata"]["weight"] = weight
                    normalized["metadata"]["priority"] = source["priority"]
                    valid_samples.append(normalized)
            
            all_samples.extend(valid_samples)
            source_counts[source_name] = len(valid_samples)
            logger.info(f"  {source_name}: {len(valid_samples)} samples (weight={weight})")
    
    return all_samples, source_counts


# =============================================================================
# Data Merging
# =============================================================================

def merge_with_ratio(
    harmful_samples: List[Dict],
    retain_samples: List[Dict],
    dr_ratio: float = 5.0,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Merge harmful and retain samples with target ratio.
    
    Args:
        harmful_samples: List of Ds samples
        retain_samples: List of Dr samples
        dr_ratio: Target Dr:Ds ratio
        shuffle: Whether to shuffle the merged data
        seed: Random seed for shuffling
    
    Returns:
        Merged list of samples
    """
    n_harmful = len(harmful_samples)
    n_retain = len(retain_samples)
    
    target_retain = int(n_harmful * dr_ratio)
    
    logger.info(f"\nMerging with Dr:Ds ratio = {dr_ratio}")
    logger.info(f"  Harmful (Ds): {n_harmful}")
    logger.info(f"  Retain (Dr): {n_retain}")
    logger.info(f"  Target retain: {target_retain}")
    
    # Adjust retain samples
    if n_retain > target_retain:
        # Prioritize by weight
        sorted_retain = sorted(
            retain_samples,
            key=lambda x: x.get("metadata", {}).get("weight", 1.0),
            reverse=True
        )
        retain_samples = sorted_retain[:target_retain]
        logger.info(f"  Using top {target_retain} retain samples by priority")
    elif n_retain < target_retain:
        # Oversample by weight
        logger.warning(f"  Not enough retain samples. Have {n_retain}, need {target_retain}")
        
        # Calculate weighted oversampling
        if n_retain > 0:
            factor = target_retain / n_retain
            oversampled = []
            
            for s in retain_samples:
                weight = s.get("metadata", {}).get("weight", 1.0)
                copies = max(1, int(weight * factor))
                oversampled.extend([s.copy() for _ in range(copies)])
            
            # Trim to exact target
            retain_samples = oversampled[:target_retain]
            logger.info(f"  Oversampled to {len(retain_samples)} retain samples")
    
    # Merge
    merged = harmful_samples + retain_samples
    
    # Add index and compute final ratios
    actual_harmful = sum(1 for s in merged if s.get("metadata", {}).get("split") == "harmful")
    actual_retain = sum(1 for s in merged if s.get("metadata", {}).get("split") == "retain")
    actual_ratio = actual_retain / max(1, actual_harmful)
    
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Harmful: {actual_harmful}")
    logger.info(f"  Retain: {actual_retain}")
    logger.info(f"  Actual ratio: {actual_ratio:.2f}")
    
    # Shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(merged)
        logger.info("  Shuffled merged dataset")
    
    return merged


def compute_statistics(samples: List[Dict]) -> Dict[str, Any]:
    """Compute statistics for the merged dataset."""
    stats = {
        "total": len(samples),
        "by_split": {},
        "by_source": {},
        "has_tool_calls": 0,
        "avg_messages": 0,
    }
    
    total_messages = 0
    
    for s in samples:
        split = s.get("metadata", {}).get("split", "unknown")
        source = s.get("metadata", {}).get("source", "unknown")
        has_tools = s.get("metadata", {}).get("has_tool_calls", False)
        
        # Count by split
        stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
        
        # Count by source
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        # Count tool calls
        if has_tools:
            stats["has_tool_calls"] += 1
        
        # Count messages
        messages = s.get("messages", [])
        total_messages += len(messages)
    
    stats["avg_messages"] = total_messages / max(1, len(samples))
    
    return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge Stage 2 training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/stage2_training.jsonl",
        help="Output path for merged training data",
    )
    parser.add_argument(
        "--dr-ratio",
        type=float,
        default=5.0,
        help="Target Dr:Ds ratio (default: 5.0)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Maximum samples per source (for testing)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the merged data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Output path for statistics JSON",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("STAGE 2 DATA MERGE")
    logger.info("=" * 60)
    
    # Collect harmful data
    logger.info("\nCollecting harmful (Ds) data:")
    harmful_samples, harmful_counts = collect_harmful_data(
        BASE_DIR, args.max_per_source
    )
    
    # Collect retain data
    logger.info("\nCollecting retain (Dr) data:")
    retain_samples, retain_counts = collect_retain_data(
        BASE_DIR, args.max_per_source
    )
    
    if not harmful_samples:
        logger.error("No harmful samples found! Run data generation scripts first.")
        return 1
    
    if not retain_samples:
        logger.error("No retain samples found! Run data generation scripts first.")
        return 1
    
    # Merge with ratio
    merged = merge_with_ratio(
        harmful_samples,
        retain_samples,
        dr_ratio=args.dr_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    
    # Compute statistics
    stats = compute_statistics(merged)
    stats["harmful_sources"] = harmful_counts
    stats["retain_sources"] = retain_counts
    stats["dr_ratio_target"] = args.dr_ratio
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged data
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in merged:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"\nSaved {len(merged)} samples to {args.output}")
    
    # Save statistics
    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {args.stats_output}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"By split:")
    for split, count in stats["by_split"].items():
        pct = 100 * count / stats["total"]
        logger.info(f"  {split}: {count} ({pct:.1f}%)")
    logger.info(f"By source:")
    for source, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total"]
        logger.info(f"  {source}: {count} ({pct:.1f}%)")
    logger.info(f"Samples with tool calls: {stats['has_tool_calls']}")
    logger.info(f"Avg messages per sample: {stats['avg_messages']:.1f}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
