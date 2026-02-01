#!/usr/bin/env python3
"""
Evaluate models from a sweep_policies run locally (without SLURM).

Usage:
    # Evaluate all policies on both fujitsu and agentdojo
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep

    # Evaluate only on agentdojo
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep --datasets agentdojo

    # Evaluate only on fujitsu
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep --datasets fujitsu

    # Quick test with limit
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep --limit 20

    # Show example traces
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep --show-examples 10

    # Evaluate a single policy
    python scripts/eval_sweep_local.py --sweep-dir /path/to/sweep --policy assistant_only
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def find_policies(sweep_dir: Path) -> list:
    """Find all policies with trained models in the sweep directory."""
    policies = []
    for policy_dir in sweep_dir.iterdir():
        if not policy_dir.is_dir():
            continue
        model_dir = policy_dir / "model"
        final_dir = model_dir / "final"
        if final_dir.exists() or (model_dir.exists() and (model_dir / "adapter_config.json").exists()):
            policies.append(policy_dir.name)
    return sorted(policies)


def get_adapter_path(sweep_dir: Path, policy: str) -> Path:
    """Get the adapter path for a policy."""
    model_dir = sweep_dir / policy / "model"
    final_dir = model_dir / "final"
    if final_dir.exists():
        return final_dir
    if (model_dir / "adapter_config.json").exists():
        return model_dir
    raise ValueError(f"No adapter found for policy {policy}")


def get_dataset_path(dataset: str, sweep_dir: Path, policy: str) -> Path:
    """Get the path to evaluation data for a dataset."""
    if dataset == "fujitsu":
        candidates = [
            PROJECT_ROOT / "data" / "traces" / "fujitsu_b4_ds.jsonl",
            PROJECT_ROOT / "data" / "traces" / "fujitsu_b4_traces.jsonl",
            PROJECT_ROOT / "data" / "cb_mvp" / "eval_stage1.jsonl",
        ]
    elif dataset == "agentdojo":
        # Prefer the split harmful traces from the sweep
        split_path = sweep_dir / policy / "agentdojo_split" / "agentdojo_traces_harmful.jsonl"
        if split_path.exists():
            return split_path
        candidates = [
            PROJECT_ROOT / "data" / "traces" / "agentdojo_traces.jsonl",
        ]
    elif dataset == "agentdojo_all":
        candidates = [
            PROJECT_ROOT / "data" / "traces" / "agentdojo_traces.jsonl",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ValueError(f"Could not find data for dataset {dataset}")


def show_example_traces(details_file: Path, n: int = 5):
    """Show example traces from evaluation details."""
    if not details_file.exists():
        print(f"  No details file: {details_file}")
        return

    samples = []
    with open(details_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("model") == "cb_model" and data.get("metric") == "tool_flip_asr":
                samples.append(data)

    print(f"\n  Example Traces (showing {min(n, len(samples))} of {len(samples)}):")
    print("  " + "-" * 70)

    for i, sample in enumerate(samples[:n]):
        outcome = sample.get("outcome", "?")
        outcome_emoji = {
            "attack_success": "❌",
            "correct_behavior": "✅",
            "no_tool_call": "⚠️",
            "other_tool": "❓",
        }.get(outcome, "?")

        print(f"\n  [{i+1}] {outcome_emoji} {outcome}")
        print(f"      ID: {sample.get('id', '?')[:60]}")
        print(f"      Expected tool:  {sample.get('expected_tool')}")
        print(f"      Simulated tool: {sample.get('simulated_tool')}")
        print(f"      Observed tool:  {sample.get('observed_tool')}")

        response = sample.get("response_preview") or sample.get("response_full", "")
        if response:
            # Truncate for display
            response_display = response[:300].replace("\n", " ")
            if len(response) > 300:
                response_display += "..."
            print(f"      Response: {response_display}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sweep_policies models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Path to sweep directory (e.g., sweeps/policy_sweep_TIMESTAMP)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Evaluate only this policy (default: all policies)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="fujitsu,agentdojo",
        help="Comma-separated datasets to evaluate on (default: fujitsu,agentdojo)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Baseline model path or HF ID",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Evaluation data JSONL (overrides --datasets)",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tool_schemas" / "b4_standard_v1.json",
        help="Tool schema JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval samples",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Show N example traces after evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda:0, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: SWEEP_DIR/evaluations)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    # Validate sweep directory
    if not args.sweep_dir.exists():
        print(f"ERROR: Sweep directory not found: {args.sweep_dir}")
        return 1

    # Find policies
    if args.policy:
        policies = [args.policy]
    else:
        policies = find_policies(args.sweep_dir)

    if not policies:
        print(f"ERROR: No trained models found in {args.sweep_dir}")
        return 1

    print(f"Found {len(policies)} policies to evaluate: {', '.join(policies)}")

    # Parse datasets
    if args.eval_data:
        # User provided explicit eval data, use single "custom" dataset
        datasets = ["custom"]
        dataset_paths = {"custom": args.eval_data}
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        dataset_paths = {}  # Will be resolved per-policy

    print(f"Datasets to evaluate: {', '.join(datasets)}")

    # Output directory
    output_dir = args.output_dir or (args.sweep_dir / "evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import evaluation module
    from src.evaluation.eval import run_mvp_evaluation
    import torch

    # results_summary[dataset] = [list of policy results]
    results_summary = {ds: [] for ds in datasets}

    for policy in policies:
        print(f"\n{'='*60}")
        print(f"Evaluating: {policy}")
        print("=" * 60)

        try:
            adapter_path = get_adapter_path(args.sweep_dir, policy)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Adapter: {adapter_path}")

        # Evaluate on each dataset
        for dataset in datasets:
            # Get dataset path
            if dataset in dataset_paths:
                eval_data = dataset_paths[dataset]
            else:
                try:
                    eval_data = get_dataset_path(dataset, args.sweep_dir, policy)
                except ValueError as e:
                    print(f"  SKIP {dataset}: {e}")
                    continue

            if not eval_data.exists():
                print(f"  SKIP {dataset}: File not found: {eval_data}")
                continue

            output_file = output_dir / f"{policy}_{dataset}_eval.json"

            print(f"\n  Dataset: {dataset}")
            print(f"    Data: {eval_data}")
            print(f"    Output: {output_file}")

            # Load eval samples
            eval_samples = []
            with open(eval_data) as f:
                for line in f:
                    if line.strip():
                        eval_samples.append(json.loads(line))
                        if args.limit and len(eval_samples) >= args.limit:
                            break

            print(f"    Samples: {len(eval_samples)}")

            # Run evaluation
            results = run_mvp_evaluation(
                baseline_model_path=args.baseline,
                cb_model_path=None,
                cb_adapter_path=str(adapter_path),
                eval_data_path=eval_data,
                tool_schema_path=args.tool_schema,
                device=args.device,
                torch_dtype=torch.bfloat16,
                verbose=not args.quiet,
                eval_samples=eval_samples,
            )

            # Save results
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Save details separately
            details_file = output_file.with_suffix(".details.jsonl")
            with open(details_file, "w") as f:
                for key in ["baseline", "cb_model"]:
                    if key in results:
                        for metric_key in ["tool_flip_asr", "forced_function_call", "capability_retention"]:
                            if metric_key in results[key] and "details" in results[key][metric_key]:
                                for detail in results[key][metric_key]["details"]:
                                    record = {"model": key, "metric": metric_key, **detail}
                                    f.write(json.dumps(record, default=str) + "\n")

            # Collect summary
            cb_asr = results.get("cb_model", {}).get("tool_flip_asr", {}).get("attack_success_rate", -1)
            baseline_asr = results.get("baseline", {}).get("tool_flip_asr", {}).get("attack_success_rate", -1)
            capability = results.get("cb_model", {}).get("capability_retention", {}).get("capability_retention", -1)
            total_samples = results.get("cb_model", {}).get("tool_flip_asr", {}).get("total_samples", 0)
            passed = results.get("stage1_passed", False)

            results_summary[dataset].append({
                "policy": policy,
                "baseline_asr": baseline_asr,
                "cb_asr": cb_asr,
                "capability": capability,
                "samples": total_samples,
                "passed": passed,
            })

            # Show examples if requested
            if args.show_examples > 0:
                show_example_traces(details_file, args.show_examples)

    # Print summary comparison per dataset
    print(f"\n{'='*60}")
    print("SWEEP EVALUATION SUMMARY")
    print("=" * 60)

    for dataset in datasets:
        if not results_summary[dataset]:
            continue

        print(f"\nDataset: {dataset}")
        print("-" * 60)

        # Sort by CB ASR (lower is better)
        summary = sorted(results_summary[dataset], key=lambda x: x["cb_asr"] if x["cb_asr"] >= 0 else float("inf"))

        print(f"{'Policy':<20} {'Baseline':>10} {'CB ASR':>10} {'Samples':>8} {'Capability':>12} {'Gate':>6}")
        print("-" * 70)

        for r in summary:
            baseline = f"{r['baseline_asr']:.1%}" if r["baseline_asr"] >= 0 else "N/A"
            cb = f"{r['cb_asr']:.1%}" if r["cb_asr"] >= 0 else "N/A"
            cap = f"{r['capability']:.1%}" if r["capability"] >= 0 else "N/A"
            status = "✅" if r["passed"] else "❌"
            print(f"{r['policy']:<20} {baseline:>10} {cb:>10} {r['samples']:>8} {cap:>12} {status:>6}")

        if summary:
            best = summary[0]
            print(f"\nBest policy: {best['policy']} ({best['cb_asr']:.1%})")

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
