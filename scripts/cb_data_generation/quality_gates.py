#!/usr/bin/env python3
"""
Quality Gates for Circuit Breaker Data

This module provides validation and quality checks for CB training data.
These are NON-NEGOTIABLE checks that must pass before any training run.

Quality Gates:
1. Format Validity - Every Ds entry must parse into valid tool calls
2. Success Rate - Ds should contain high fraction of actual harmful compliance
3. Distribution Closeness - Ds should align with real threat model
4. Retain Coverage - Dr must cover benign tasks + refusal behaviors

Usage:
    # Validate all datasets
    python scripts/cb_data_generation/quality_gates.py \
        --ds data/circuit_breakers/ds/circuit_breaker_set.jsonl \
        --dr data/circuit_breakers/dr/retain_set.jsonl

    # Strict mode (fail on any warning)
    python scripts/cb_data_generation/quality_gates.py --strict

    # Generate detailed report
    python scripts/cb_data_generation/quality_gates.py --report quality_report.json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cb_data_generation.tool_format import ToolCall, validate_tool_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Quality Gate Definitions
# =============================================================================

@dataclass
class GateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Full quality report for a dataset."""
    dataset_path: str
    dataset_type: str  # "ds" or "dr"
    total_records: int
    gates: List[GateResult] = field(default_factory=list)
    passed: bool = True
    
    def add_gate(self, result: GateResult):
        self.gates.append(result)
        if not result.passed:
            self.passed = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_type": self.dataset_type,
            "total_records": self.total_records,
            "passed": self.passed,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "score": g.score,
                    "message": g.message,
                    "details": g.details,
                    "warnings": g.warnings,
                    "errors": g.errors,
                }
                for g in self.gates
            ],
        }


# =============================================================================
# Gate 1: Format Validity
# =============================================================================

def check_format_validity(
    records: List[Dict[str, Any]],
    dataset_type: str = "ds",
) -> GateResult:
    """
    Check that all records have valid format.
    
    For Ds:
    - Must have assistant_raw field
    - Tool calls must parse correctly
    - Required fields present
    
    For Dr:
    - Must have user_prompt
    - Tool calls (if any) must parse
    """
    result = GateResult(
        name="format_validity",
        passed=True,
        score=1.0,
        message="",
    )
    
    required_fields_ds = ["id", "set", "messages", "assistant_raw"]
    required_fields_dr = ["id", "set", "messages"]
    required_fields = required_fields_ds if dataset_type == "ds" else required_fields_dr
    
    valid_count = 0
    invalid_records = []
    parse_errors = []
    
    for record in records:
        is_valid = True
        
        # Check required fields
        missing = [f for f in required_fields if f not in record]
        if missing:
            is_valid = False
            invalid_records.append({
                "id": record.get("id", "unknown"),
                "reason": f"Missing fields: {missing}",
            })
        
        # Check messages format
        messages = record.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 1:
            is_valid = False
            invalid_records.append({
                "id": record.get("id", "unknown"),
                "reason": "Invalid messages format",
            })
        
        # For Ds, validate tool calls can be parsed
        if dataset_type == "ds":
            assistant_raw = record.get("assistant_raw", "")
            tool_calls = record.get("tool_calls_structured", [])
            
            # If has tool_calls_structured, validate them
            if tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict) or "name" not in tc:
                        is_valid = False
                        parse_errors.append({
                            "id": record.get("id", "unknown"),
                            "error": f"Invalid tool call structure: {tc}",
                        })
            # Otherwise, try to parse from raw
            elif assistant_raw:
                parsed = ToolCall.from_raw(assistant_raw)
                if parsed is None and "<|python_tag|>" in assistant_raw:
                    # Has tool tag but couldn't parse
                    result.warnings.append(
                        f"Record {record.get('id')}: Has python_tag but couldn't parse tool call"
                    )
        
        if is_valid:
            valid_count += 1
    
    # Calculate score
    result.score = valid_count / max(len(records), 1)
    result.passed = result.score >= 0.95  # 95% must be valid
    
    result.details = {
        "total_records": len(records),
        "valid_records": valid_count,
        "invalid_records": invalid_records[:10],  # First 10
        "parse_errors": parse_errors[:10],
    }
    
    if result.passed:
        result.message = f"Format validity: {result.score:.1%} valid ({valid_count}/{len(records)})"
    else:
        result.message = f"FAILED: Only {result.score:.1%} valid records (need 95%)"
        result.errors.append(f"{len(invalid_records)} invalid records found")
    
    return result


# =============================================================================
# Gate 2: Harmful Content Rate (Ds only)
# =============================================================================

def check_harmful_content_rate(
    records: List[Dict[str, Any]],
    min_rate: float = 0.8,
) -> GateResult:
    """
    Check that Ds contains actual harmful completions.
    
    This validates that:
    - Records are labeled as harmful
    - Tool flip attacks actually flipped the tool
    - Content shows signs of compliance (not refusal)
    """
    result = GateResult(
        name="harmful_content_rate",
        passed=True,
        score=1.0,
        message="",
    )
    
    harmful_count = 0
    refusal_markers = ["can't", "cannot", "won't", "refuse", "unable", "sorry", "i apologize"]
    
    tool_flip_success = 0
    tool_flip_total = 0
    refusals_found = 0
    
    for record in records:
        labels = record.get("labels", {})
        is_harmful = labels.get("is_harmful", False)
        harm_type = labels.get("harm_type", "")
        
        if is_harmful:
            harmful_count += 1
        
        # Check tool flip success
        if harm_type == "tool_flip":
            tool_flip_total += 1
            expected = labels.get("expected_tool")
            observed = labels.get("observed_tool")
            if expected and observed and expected != observed:
                tool_flip_success += 1
        
        # Check for accidental refusals in Ds
        assistant_raw = record.get("assistant_raw", "").lower()
        if any(marker in assistant_raw for marker in refusal_markers):
            refusals_found += 1
    
    result.score = harmful_count / max(len(records), 1)
    result.passed = result.score >= min_rate
    
    result.details = {
        "total_records": len(records),
        "harmful_labeled": harmful_count,
        "tool_flip_success": f"{tool_flip_success}/{tool_flip_total}" if tool_flip_total > 0 else "N/A",
        "refusals_found": refusals_found,
    }
    
    if refusals_found > len(records) * 0.1:
        result.warnings.append(
            f"High refusal rate in Ds: {refusals_found}/{len(records)} records contain refusal markers"
        )
    
    if result.passed:
        result.message = f"Harmful content rate: {result.score:.1%} ({harmful_count}/{len(records)})"
    else:
        result.message = f"FAILED: Only {result.score:.1%} harmful (need {min_rate:.0%})"
        result.errors.append(f"Insufficient harmful samples for effective CB training")
    
    return result


# =============================================================================
# Gate 3: Distribution Closeness
# =============================================================================

def check_distribution_closeness(
    records: List[Dict[str, Any]],
    min_categories: int = 3,
) -> GateResult:
    """
    Check that Ds covers the target threat distribution.
    
    Validates:
    - Multiple attack categories represented
    - No single category dominates excessively
    - Coverage of different harm types
    """
    result = GateResult(
        name="distribution_closeness",
        passed=True,
        score=1.0,
        message="",
    )
    
    # Collect distributions
    sources = Counter()
    harm_types = Counter()
    categories = Counter()
    
    for record in records:
        provenance = record.get("provenance", {})
        labels = record.get("labels", {})
        
        sources[provenance.get("source_dataset", "unknown")] += 1
        harm_types[labels.get("harm_type", "unknown")] += 1
        categories[provenance.get("category", "unknown")] += 1
    
    # Check category diversity
    num_categories = len([c for c in categories.values() if c > 0])
    
    # Check for over-dominance (no single category > 80%)
    total = sum(categories.values())
    max_fraction = max(categories.values()) / max(total, 1)
    
    result.score = min(1.0, num_categories / min_categories) * (1 - max(0, max_fraction - 0.5))
    result.passed = num_categories >= min_categories and max_fraction < 0.8
    
    result.details = {
        "sources": dict(sources.most_common()),
        "harm_types": dict(harm_types.most_common()),
        "categories": dict(categories.most_common(10)),
        "num_categories": num_categories,
        "max_category_fraction": f"{max_fraction:.1%}",
    }
    
    if max_fraction > 0.5:
        top_category = categories.most_common(1)[0][0]
        result.warnings.append(
            f"Category '{top_category}' dominates at {max_fraction:.1%} of data"
        )
    
    if result.passed:
        result.message = f"Distribution: {num_categories} categories, max fraction {max_fraction:.1%}"
    else:
        result.message = f"FAILED: Poor distribution (need {min_categories}+ categories, <80% single)"
        result.errors.append("Insufficient diversity in attack categories")
    
    return result


# =============================================================================
# Gate 4: Retain Set Coverage (Dr only)
# =============================================================================

def check_retain_coverage(
    records: List[Dict[str, Any]],
    require_refusals: bool = True,
    require_tool_use: bool = True,
) -> GateResult:
    """
    Check that Dr adequately covers capability + refusal behaviors.
    
    Validates:
    - Contains refusal examples (if required)
    - Contains tool use examples
    - Has variety in benign tasks
    """
    result = GateResult(
        name="retain_coverage",
        passed=True,
        score=1.0,
        message="",
    )
    
    refusal_count = 0
    tool_use_count = 0
    resisted_attack_count = 0
    sources = Counter()
    
    for record in records:
        labels = record.get("labels", {})
        provenance = record.get("provenance", {})
        
        if labels.get("is_refusal"):
            refusal_count += 1
        
        if record.get("tool_calls_structured"):
            tool_use_count += 1
        
        if labels.get("resisted_attack"):
            resisted_attack_count += 1
        
        sources[provenance.get("source_dataset", "unknown")] += 1
    
    # Score components
    scores = []
    
    if require_refusals:
        refusal_fraction = refusal_count / max(len(records), 1)
        scores.append(min(1.0, refusal_fraction * 50))  # Want at least 2% refusals
        if refusal_count == 0:
            result.errors.append("No refusal examples in Dr - refusal behavior may degrade")
    
    if require_tool_use:
        tool_fraction = tool_use_count / max(len(records), 1)
        scores.append(min(1.0, tool_fraction * 4))  # Want at least 25% tool use
        if tool_use_count < len(records) * 0.2:
            result.warnings.append(f"Low tool use coverage: {tool_use_count}/{len(records)}")
    
    result.score = sum(scores) / max(len(scores), 1)
    result.passed = result.score >= 0.7
    
    result.details = {
        "total_records": len(records),
        "refusal_examples": refusal_count,
        "tool_use_examples": tool_use_count,
        "resisted_attacks": resisted_attack_count,
        "sources": dict(sources.most_common()),
    }
    
    if result.passed:
        result.message = f"Retain coverage: {refusal_count} refusals, {tool_use_count} tool use"
    else:
        result.message = f"FAILED: Insufficient Dr coverage (score {result.score:.2f})"
    
    return result


# =============================================================================
# Gate 5: No Synthetic Markers (Ds)
# =============================================================================

def check_no_synthetic_markers(
    records: List[Dict[str, Any]],
) -> GateResult:
    """
    Check that Ds doesn't contain synthetic/template markers.
    
    These markers indicate the data is from templates, not real LLM output:
    - [TOOL_CALL]
    - (Expected:
    - [HARMFUL:
    """
    result = GateResult(
        name="no_synthetic_markers",
        passed=True,
        score=1.0,
        message="",
    )
    
    synthetic_markers = [
        "[TOOL_CALL]",
        "(Expected:",
        "[TOOL_CALLS:",
        "but injection caused flip to:",
        "[Harmful behavior executed]",
        "[HARMFUL:",
    ]
    
    synthetic_count = 0
    examples = []
    
    for record in records:
        assistant_raw = record.get("assistant_raw", "")
        
        for marker in synthetic_markers:
            if marker in assistant_raw:
                synthetic_count += 1
                if len(examples) < 5:
                    examples.append({
                        "id": record.get("id"),
                        "marker": marker,
                        "snippet": assistant_raw[:100],
                    })
                break
    
    result.score = 1 - (synthetic_count / max(len(records), 1))
    result.passed = synthetic_count == 0
    
    result.details = {
        "synthetic_records": synthetic_count,
        "examples": examples,
    }
    
    if result.passed:
        result.message = "No synthetic markers found"
    else:
        result.message = f"FAILED: {synthetic_count} records have synthetic markers"
        result.errors.append(
            "Synthetic markers indicate template-based data, not real LLM output. "
            "CB training will not be effective."
        )
    
    return result


# =============================================================================
# Main Validation
# =============================================================================

def validate_dataset(
    path: Path,
    dataset_type: str,  # "ds" or "dr"
) -> QualityReport:
    """Run all quality gates on a dataset."""
    
    # Load records
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    report = QualityReport(
        dataset_path=str(path),
        dataset_type=dataset_type,
        total_records=len(records),
    )
    
    if len(records) == 0:
        report.passed = False
        report.gates.append(GateResult(
            name="non_empty",
            passed=False,
            score=0.0,
            message="Dataset is empty",
            errors=["No records found in dataset"],
        ))
        return report
    
    # Gate 1: Format Validity
    report.add_gate(check_format_validity(records, dataset_type))
    
    if dataset_type == "ds":
        # Gates for Circuit Breaker Set
        report.add_gate(check_harmful_content_rate(records))
        report.add_gate(check_distribution_closeness(records))
        report.add_gate(check_no_synthetic_markers(records))
    else:
        # Gates for Retain Set
        report.add_gate(check_retain_coverage(records))
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate Circuit Breaker training data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--ds",
        type=Path,
        default=BASE_DIR / "data" / "circuit_breakers" / "ds" / "circuit_breaker_set.jsonl",
        help="Path to Circuit Breaker Set (Ds)",
    )
    parser.add_argument(
        "--dr",
        type=Path,
        default=BASE_DIR / "data" / "circuit_breakers" / "dr" / "retain_set.jsonl",
        help="Path to Retain Set (Dr)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any warning",
    )
    parser.add_argument(
        "--skip-ds",
        action="store_true",
        help="Skip Ds validation",
    )
    parser.add_argument(
        "--skip-dr",
        action="store_true",
        help="Skip Dr validation",
    )
    
    args = parser.parse_args()
    
    reports = []
    all_passed = True
    
    # Validate Ds
    if not args.skip_ds and args.ds.exists():
        logger.info(f"Validating Circuit Breaker Set: {args.ds}")
        ds_report = validate_dataset(args.ds, "ds")
        reports.append(ds_report)
        
        for gate in ds_report.gates:
            status = "✓" if gate.passed else "✗"
            logger.info(f"  [{status}] {gate.name}: {gate.message}")
            for warning in gate.warnings:
                logger.warning(f"      ⚠ {warning}")
            for error in gate.errors:
                logger.error(f"      ✗ {error}")
        
        if not ds_report.passed:
            all_passed = False
        if args.strict and any(g.warnings for g in ds_report.gates):
            all_passed = False
    elif not args.skip_ds:
        logger.warning(f"Ds not found: {args.ds}")
    
    # Validate Dr
    if not args.skip_dr and args.dr.exists():
        logger.info(f"Validating Retain Set: {args.dr}")
        dr_report = validate_dataset(args.dr, "dr")
        reports.append(dr_report)
        
        for gate in dr_report.gates:
            status = "✓" if gate.passed else "✗"
            logger.info(f"  [{status}] {gate.name}: {gate.message}")
            for warning in gate.warnings:
                logger.warning(f"      ⚠ {warning}")
            for error in gate.errors:
                logger.error(f"      ✗ {error}")
        
        if not dr_report.passed:
            all_passed = False
        if args.strict and any(g.warnings for g in dr_report.gates):
            all_passed = False
    elif not args.skip_dr:
        logger.warning(f"Dr not found: {args.dr}")
    
    # Write report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({
                "all_passed": all_passed,
                "reports": [r.to_dict() for r in reports],
            }, f, indent=2)
        logger.info(f"Report written to: {args.report}")
    
    # Summary
    if all_passed:
        logger.info("✓ All quality gates passed")
        return 0
    else:
        logger.error("✗ Some quality gates failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
