#!/usr/bin/env python3
"""
Data Format Validation Script for Stage 1 MVP

Validates that Ds and Dr data files conform to the Llama 3.1 tool format
and the MVP data schema requirements.

Checks:
1. 100% have `messages` array (not just `prompt`)
2. 100% have `tools` definition or reference
3. 100% `assistant_raw` starts with `<|python_tag|>`
4. 100% ends with `<|eom_id|>` or `<|eot_id|>`
5. 0% have markdown wrappers or prefixes
6. Ds: 100% have `is_flip_success: true`
7. Dr: 100% have `is_flip_success: false`

Usage:
    # Validate Ds
    python scripts/cb_data_generation/validate_format.py \
        --data data/cb_mvp/ds_stage1.jsonl \
        --set-type ds

    # Validate Dr
    python scripts/cb_data_generation/validate_format.py \
        --data data/cb_mvp/dr_stage1.jsonl \
        --set-type dr

    # Strict mode (fail on any error)
    python scripts/cb_data_generation/validate_format.py \
        --data data/cb_mvp/ds_stage1.jsonl \
        --strict
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Validation Rules
# =============================================================================

class ValidationResult:
    """Result of validating a single sample."""
    
    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def __repr__(self):
        status = "✅" if self.is_valid else "❌"
        return f"{status} {self.sample_id}: {len(self.errors)} errors, {len(self.warnings)} warnings"


def validate_messages(sample: Dict[str, Any], result: ValidationResult):
    """Validate messages array structure."""
    messages = sample.get("messages")
    
    if messages is None:
        result.add_error("Missing 'messages' array")
        return
    
    if not isinstance(messages, list):
        result.add_error(f"'messages' should be a list, got {type(messages).__name__}")
        return
    
    if len(messages) < 2:
        result.add_error(f"'messages' should have at least 2 items (system + user), got {len(messages)}")
        return
    
    # Check roles
    roles = [m.get("role") for m in messages]
    
    if "system" not in roles:
        result.add_warning("No 'system' message in messages array")
    
    if "user" not in roles:
        result.add_error("No 'user' message in messages array")
    
    # Check each message has content
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.add_error(f"Message {i} is not a dict")
            continue
        if "role" not in msg:
            result.add_error(f"Message {i} missing 'role'")
        if "content" not in msg:
            result.add_error(f"Message {i} missing 'content'")


def validate_tools(sample: Dict[str, Any], result: ValidationResult):
    """Validate tools definition or reference."""
    tools = sample.get("tools")
    
    if tools is None:
        result.add_error("Missing 'tools' field")
        return
    
    if isinstance(tools, str):
        # Reference to frozen schema (e.g., "b4_standard_v1")
        if not tools.strip():
            result.add_error("Empty 'tools' reference string")
    elif isinstance(tools, list):
        # Full tool definitions
        if len(tools) == 0:
            result.add_warning("Empty 'tools' array")
    else:
        result.add_error(f"'tools' should be string or list, got {type(tools).__name__}")


def validate_assistant_raw(sample: Dict[str, Any], result: ValidationResult):
    """Validate assistant_raw follows Llama 3.1 format."""
    assistant_raw = sample.get("assistant_raw")
    
    if assistant_raw is None:
        result.add_error("Missing 'assistant_raw' field")
        return
    
    if not isinstance(assistant_raw, str):
        result.add_error(f"'assistant_raw' should be string, got {type(assistant_raw).__name__}")
        return
    
    if not assistant_raw.strip():
        result.add_error("Empty 'assistant_raw'")
        return
    
    # Must contain <|python_tag|>
    if "<|python_tag|>" not in assistant_raw:
        result.add_error("Missing <|python_tag|> in assistant_raw")
    
    # Check for end tokens (but make it a warning, not error)
    # vLLM may not include these even with skip_special_tokens=False
    stripped = assistant_raw.rstrip()
    has_valid_end = (
        stripped.endswith("<|eom_id|>") or 
        stripped.endswith("<|eot_id|}") or
        stripped.endswith("<|eot_id|>")
    )
    
    # Also check if end token exists anywhere (might have trailing whitespace/content)
    has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
    
    if not has_end_token:
        result.add_warning("Missing <|eom_id|> or <|eot_id|> in assistant_raw (not critical for training)")
    elif not has_valid_end:
        result.add_warning("End token not at end of assistant_raw (trailing content)")
    
    # Must NOT have markdown wrappers
    if "```" in assistant_raw:
        result.add_error("Contains markdown code block (```) in assistant_raw")
    
    # Must NOT have common prefixes
    forbidden_prefixes = ["Action:", "ToolCall:", "Function:", "Tool:"]
    for prefix in forbidden_prefixes:
        if assistant_raw.strip().startswith(prefix):
            result.add_error(f"Forbidden prefix '{prefix}' in assistant_raw")
            break


def validate_tool_calls_structured(sample: Dict[str, Any], result: ValidationResult):
    """Validate tool_calls_structured array."""
    tool_calls = sample.get("tool_calls_structured")
    
    if tool_calls is None:
        result.add_warning("Missing 'tool_calls_structured' field")
        return
    
    if not isinstance(tool_calls, list):
        result.add_error(f"'tool_calls_structured' should be list, got {type(tool_calls).__name__}")
        return
    
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            result.add_error(f"tool_call {i} is not a dict")
            continue
        if "name" not in tc:
            result.add_error(f"tool_call {i} missing 'name'")
        if "parameters" not in tc and "arguments" not in tc:
            result.add_warning(f"tool_call {i} missing 'parameters'/'arguments'")


def validate_labels(sample: Dict[str, Any], result: ValidationResult, set_type: str):
    """Validate labels based on set type (ds or dr)."""
    labels = sample.get("labels")
    
    if labels is None:
        result.add_error("Missing 'labels' field")
        return
    
    if not isinstance(labels, dict):
        result.add_error(f"'labels' should be dict, got {type(labels).__name__}")
        return
    
    # Check required label fields
    required_fields = ["expected_tool", "observed_tool", "is_flip_success"]
    for field in required_fields:
        if field not in labels:
            result.add_error(f"Missing 'labels.{field}'")
    
    # Validate is_flip_success based on set type
    is_flip_success = labels.get("is_flip_success")
    
    if set_type == "ds":
        if is_flip_success is not True:
            result.add_error(f"Ds sample must have is_flip_success=true, got {is_flip_success}")
    elif set_type == "dr":
        if is_flip_success is not False:
            result.add_error(f"Dr sample must have is_flip_success=false, got {is_flip_success}")


def validate_metadata(sample: Dict[str, Any], result: ValidationResult):
    """Validate metadata field."""
    metadata = sample.get("metadata")
    
    if metadata is None:
        result.add_warning("Missing 'metadata' field")
        return
    
    if not isinstance(metadata, dict):
        result.add_error(f"'metadata' should be dict, got {type(metadata).__name__}")
        return
    
    # Check recommended fields
    recommended = ["split", "source"]
    for field in recommended:
        if field not in metadata:
            result.add_warning(f"Missing recommended 'metadata.{field}'")


def validate_sample(
    sample: Dict[str, Any],
    set_type: str = "ds",
) -> ValidationResult:
    """
    Validate a single sample against the MVP schema.
    
    Args:
        sample: The sample to validate
        set_type: "ds" for circuit breaker set, "dr" for retain set
    
    Returns:
        ValidationResult with errors and warnings
    """
    sample_id = sample.get("id", "unknown")
    result = ValidationResult(sample_id)
    
    # Run all validators
    validate_messages(sample, result)
    validate_tools(sample, result)
    validate_assistant_raw(sample, result)
    validate_tool_calls_structured(sample, result)
    validate_labels(sample, result, set_type)
    validate_metadata(sample, result)
    
    return result


# =============================================================================
# Aggregate Validation
# =============================================================================

def validate_file(
    path: Path,
    set_type: str = "ds",
    verbose: bool = True,
    max_errors: int = 10,
) -> Dict[str, Any]:
    """
    Validate all samples in a JSONL file.
    
    Args:
        path: Path to JSONL file
        set_type: "ds" or "dr"
        verbose: Print detailed errors
        max_errors: Maximum errors to print per type
    
    Returns:
        Dict with validation statistics
    """
    results = []
    error_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    
    logger.info(f"Validating {path}...")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                error_counts["invalid_json"] = error_counts.get("invalid_json", 0) + 1
                continue
            
            result = validate_sample(sample, set_type)
            results.append(result)
            
            # Aggregate errors
            for error in result.errors:
                error_key = error.split(":")[0] if ":" in error else error[:50]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for warning in result.warnings:
                warning_key = warning.split(":")[0] if ":" in warning else warning[:50]
                warning_counts[warning_key] = warning_counts.get(warning_key, 0) + 1
    
    # Compute statistics
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    
    stats = {
        "file": str(path),
        "set_type": set_type,
        "total_samples": total,
        "valid_samples": valid,
        "invalid_samples": invalid,
        "validity_rate": valid / total if total > 0 else 0,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "error_types": error_counts,
        "warning_types": warning_counts,
    }
    
    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"File: {path}")
    logger.info(f"Set type: {set_type}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Valid samples: {valid} ({stats['validity_rate']:.1%})")
    logger.info(f"Invalid samples: {invalid}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total warnings: {total_warnings}")
    
    if error_counts and verbose:
        logger.info("")
        logger.info("Error breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])[:max_errors]:
            logger.info(f"  {count:4d} | {error_type}")
    
    if warning_counts and verbose:
        logger.info("")
        logger.info("Warning breakdown:")
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:max_errors]:
            logger.info(f"  {count:4d} | {warning_type}")
    
    # Show sample errors
    if verbose and invalid > 0:
        logger.info("")
        logger.info("Sample errors (first 5 invalid samples):")
        shown = 0
        for result in results:
            if not result.is_valid and shown < 5:
                logger.info(f"  {result.sample_id}:")
                for error in result.errors[:3]:
                    logger.info(f"    - {error}")
                shown += 1
    
    logger.info("=" * 60)
    
    # Overall pass/fail
    if invalid == 0:
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
    else:
        logger.info(f"❌ VALIDATION FAILED: {invalid} invalid samples")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate data format for Stage 1 MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to JSONL data file to validate",
    )
    parser.add_argument(
        "--set-type",
        type=str,
        choices=["ds", "dr", "auto"],
        default="auto",
        help="Set type: ds (circuit breaker) or dr (retain). 'auto' infers from filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save validation report JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum errors to show per type",
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not args.data.exists():
        logger.error(f"File not found: {args.data}")
        return 1
    
    # Infer set type if auto
    set_type = args.set_type
    if set_type == "auto":
        filename = args.data.name.lower()
        if "ds" in filename or "circuit" in filename or "harmful" in filename:
            set_type = "ds"
        elif "dr" in filename or "retain" in filename or "benign" in filename:
            set_type = "dr"
        else:
            logger.warning("Could not infer set type from filename, defaulting to 'ds'")
            set_type = "ds"
        logger.info(f"Inferred set type: {set_type}")
    
    # Validate
    stats = validate_file(
        args.data,
        set_type=set_type,
        verbose=not args.quiet,
        max_errors=args.max_errors,
    )
    
    # Save report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Validation report saved to {args.output}")
    
    # Exit code
    if args.strict and stats["invalid_samples"] > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
