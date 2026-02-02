#!/usr/bin/env python3
"""
Visualize hyperparameter sweep results with full context.

This script reads from the sweep output directory structure and displays:
- User queries with injection text (from traces)
- System prompts and available tools (from tool schema)
- Expected vs observed tool calls
- Baseline vs Circuit Breaker responses
- Aggregate metrics across configurations

Usage:
    # From sweep directory (on cluster)
    python scripts/visualize_sweep_results.py /scratch/memoozd/cb-scratch/sweeps/hparam_sweep_YYYYMMDD_HHMMSS
    
    # Show detailed samples
    python scripts/visualize_sweep_results.py <sweep_dir> --show-samples 10
    
    # Filter to successful CB blocks
    python scripts/visualize_sweep_results.py <sweep_dir> --show-samples 10 --filter-success
    
    # Analyze specific run
    python scripts/visualize_sweep_results.py <sweep_dir> --run a5.0_l10_20_assistant_only
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    if not path.exists():
        return []
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_tool_schema(schema_path: Path) -> Dict[str, Any]:
    """Load tool schema with system prompt and tool definitions."""
    if not schema_path.exists():
        return {}
    return load_json(schema_path)


def build_trace_index(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build an index of traces by ID for quick lookup."""
    return {t.get("id", ""): t for t in traces if t.get("id")}


# =============================================================================
# Context Extraction from Traces
# =============================================================================

def get_system_prompt(trace: Dict[str, Any]) -> Optional[str]:
    """Extract system prompt from trace messages."""
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    # Check signal_hints for raw_system_prompt
    signal_hints = trace.get("signal_hints", {})
    if signal_hints and signal_hints.get("raw_system_prompt"):
        return signal_hints["raw_system_prompt"]
    return None


def get_user_query(trace: Dict[str, Any]) -> Optional[str]:
    """Extract the user query from trace messages."""
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def get_injection_text(trace: Dict[str, Any]) -> Optional[str]:
    """Extract injection text from trace tool_attack or signal_hints."""
    tool_attack = trace.get("tool_attack", {})
    if tool_attack and tool_attack.get("injection_text"):
        return tool_attack["injection_text"]
    return None


def find_injection_in_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find injections in AgentDojo-style traces.
    
    AgentDojo injections are embedded in tool results (not user messages).
    They typically appear as <INFORMATION>...</INFORMATION> blocks.
    
    Returns list of {role, content_preview, injection_text} dicts.
    """
    injections = []
    messages = trace.get("messages", [])
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        
        # Look for <INFORMATION> tags (AgentDojo injection marker)
        import re
        info_pattern = re.compile(r'<INFORMATION>(.*?)</INFORMATION>', re.DOTALL | re.IGNORECASE)
        matches = info_pattern.findall(content)
        
        for match in matches:
            injections.append({
                "message_index": i,
                "role": msg.get("role", "unknown"),
                "injection_text": match.strip(),
                "content_preview": content,
            })
    
    return injections


def get_injection_span(trace: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Get character span of injection in user query."""
    signal_hints = trace.get("signal_hints", {})
    if signal_hints:
        span = signal_hints.get("injection_char_span", {})
        if span:
            return (span.get("char_start", 0), span.get("char_end", 0))
    return None


def get_expected_observed_tools(trace: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract expected and observed tools from trace."""
    # Try tool_attack first (trace_v1 format)
    tool_attack = trace.get("tool_attack", {})
    if tool_attack:
        expected = tool_attack.get("expected_tool")
        observed = tool_attack.get("observed_tool")
        if expected or observed:
            return expected, observed
    
    # Try signal_hints
    signal_hints = trace.get("signal_hints", {})
    if signal_hints:
        expected = signal_hints.get("expected_tool_name")
        observed = signal_hints.get("observed_tool_name")
        if expected or observed:
            return expected, observed
    
    # Try labels (legacy)
    labels = trace.get("labels", {})
    if labels:
        return labels.get("expected_tool"), labels.get("simulated_tool")
    
    return None, None


def get_attack_info(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Get full attack information from trace."""
    labels = trace.get("labels", {})
    tool_attack = trace.get("tool_attack", {})
    
    return {
        "category": labels.get("category"),
        "attack_type": labels.get("attack_type") or tool_attack.get("attack_vector"),
        "attack_succeeded": labels.get("attack_succeeded"),
        "security_outcome": labels.get("security_outcome"),
    }


# =============================================================================
# Display Helpers
# =============================================================================

def truncate_text(text: str, max_length: Optional[int] = None) -> str:
    """Return text without truncation unless a max_length is explicitly set."""
    if not text:
        return ""
    if max_length is None:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def highlight_injection(text: str, injection: Optional[str], use_color: bool = True) -> str:
    """Highlight injection text within user query."""
    if not injection or not text:
        return text
    
    idx = text.find(injection)
    if idx < 0:
        return text
    
    if use_color:
        # ANSI colors
        RED = "\033[91m"
        RESET = "\033[0m"
        before = text[:idx]
        after = text[idx + len(injection):]
        return f"{before}{RED}[INJECTION: {truncate_text(injection)}]{RESET}{after}"
    else:
        before = text[:idx]
        after = text[idx + len(injection):]
        return f"{before}[INJECTION: {truncate_text(injection)}]{after}"


def format_tool_list(tools: List[Dict[str, Any]]) -> str:
    """Format tool list for display."""
    tool_names = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", tool.get("name", "?"))
        tool_names.append(name)
    return ", ".join(tool_names)


# =============================================================================
# Sample Display
# =============================================================================

def print_sample_detail(
    sample: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    show_full: bool = False,
    use_color: bool = True,
) -> None:
    """Print detailed information about a single sample with full context."""
    max_len = None
    
    # Colors
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    YELLOW = "\033[93m" if use_color else ""
    BLUE = "\033[94m" if use_color else ""
    CYAN = "\033[96m" if use_color else ""
    MAGENTA = "\033[95m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Sample ID:{RESET} {sample.get('id', 'N/A')}")
    print(f"{'='*80}")
    
    # Detect dataset type from trace
    dataset_type = "unknown"
    if trace:
        source = trace.get("source", {})
        dataset = source.get("dataset", "").lower()
        if "agentdojo" in dataset:
            dataset_type = "agentdojo"
        elif "fujitsu" in dataset:
            dataset_type = "fujitsu"
    
    # Show system prompt - ALWAYS prefer embedded from trace (especially for AgentDojo)
    sys_prompt = None
    if trace:
        sys_prompt = get_system_prompt(trace)
    
    if sys_prompt:
        print(f"\n{CYAN}📋 SYSTEM PROMPT (from trace):{RESET}")
        print(f"   {truncate_text(sys_prompt, max_len)}")
    elif tool_schema:
        schema_prompt = tool_schema.get("system_prompt", "")
        if schema_prompt:
            print(f"\n{CYAN}📋 SYSTEM PROMPT (from schema - {YELLOW}may not match dataset{RESET}):{RESET}")
            print(f"   {truncate_text(schema_prompt, max_len)}")
    
            injections.append({
                "message_index": i,
                "role": msg.get("role", "unknown"),
                "injection_text": match.strip(),
                "content_preview": content,
            })
    if trace:
        user_query = get_user_query(trace)
        if user_query:
            print(f"\n{BLUE}💬 USER QUERY:{RESET}")
            print(f"   {truncate_text(user_query, max_len)}")
    
    # Show injection - different handling for Fujitsu vs AgentDojo
    if trace:
        # Fujitsu-style: injection in user query via tool_attack field
        fujitsu_injection = get_injection_text(trace)
        if fujitsu_injection:
            print(f"\n{RED}⚠️  INJECTION TEXT (in user query):{RESET}")
            print(f"   {truncate_text(fujitsu_injection, max_len)}")
        
        # AgentDojo-style: injection in tool results
        agentdojo_injections = find_injection_in_messages(trace)
        if agentdojo_injections:
            print(f"\n{RED}⚠️  INJECTIONS IN TOOL RESULTS ({len(agentdojo_injections)}):{RESET}")
            for inj in agentdojo_injections:
                print(f"\n   {MAGENTA}[Message {inj['message_index']} - {inj['role']}]{RESET}")
                print(f"   Context: {inj['content_preview']}")
                print(f"   {RED}Injection: {truncate_text(inj['injection_text'], max_len)}{RESET}")
        
        # Attack info
        attack_info = get_attack_info(trace)
        if attack_info.get("category"):
            print(f"\n{YELLOW}📊 ATTACK INFO:{RESET}")
            print(f"   Dataset: {dataset_type}")
            print(f"   Category: {attack_info.get('category', 'N/A')}")
            print(f"   Attack Type: {attack_info.get('attack_type', 'N/A')}")
            print(f"   Attack Succeeded: {attack_info.get('attack_succeeded', 'N/A')}")
    
    # Show expected and simulated tools
    expected = sample.get("expected_tool", "N/A")
    simulated = sample.get("simulated_tool", "N/A")
    print(f"\n{BOLD}📎 Expected Tool:{RESET} {GREEN}{expected}{RESET}")
    print(f"{BOLD}🎯 Simulated (Attack) Tool:{RESET} {RED}{simulated}{RESET}")
    
    # Show tool observations
    baseline_obs = sample.get("baseline_observed_tool", "N/A")
    cb_obs = sample.get("cb_observed_tool", "N/A")
    print(f"\n{BOLD}🔍 Baseline Observed Tool:{RESET} {baseline_obs}")
    print(f"{BOLD}🛡️  CB Observed Tool:{RESET} {cb_obs}")
    
    # Show outcomes with color coding
    baseline_outcome = sample.get("baseline_outcome", "N/A")
    cb_outcome = sample.get("cb_outcome", "N/A")
    
    outcome_colors = {
        "attack_success": RED,
        "correct_behavior": GREEN,
        "no_tool_call": YELLOW,
        "other_tool": BLUE,
    }
    
    b_color = outcome_colors.get(baseline_outcome, "")
    c_color = outcome_colors.get(cb_outcome, "")
    
    print(f"\n{BOLD}📊 Baseline Outcome:{RESET} {b_color}{baseline_outcome}{RESET}")
    print(f"{BOLD}📊 CB Outcome:{RESET} {c_color}{cb_outcome}{RESET}")
    
    # Improvement/regression indicator
    if baseline_outcome == "attack_success" and cb_outcome != "attack_success":
        print(f"\n{GREEN}✅ CB SUCCESSFULLY BLOCKED ATTACK{RESET}")
    elif baseline_outcome != "attack_success" and cb_outcome == "attack_success":
        print(f"\n{RED}❌ CB REGRESSION - Attack now succeeds{RESET}")
    
    # Show responses
    baseline_resp = sample.get("baseline_response", "")
    cb_resp = sample.get("cb_response", "")
    
    if baseline_resp:
        print(f"\n{BOLD}📝 BASELINE RESPONSE:{RESET}")
        print(f"   {truncate_text(baseline_resp, max_len)}")
    
    if cb_resp:
        print(f"\n{BOLD}🛡️  CB RESPONSE:{RESET}")
        print(f"   {truncate_text(cb_resp, max_len)}")


# =============================================================================
# Run Analysis
# =============================================================================

def analyze_run(
    run_dir: Path,
    traces_dir: Optional[Path] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    show_samples: int = 0,
    show_full: bool = False,
    filter_success: bool = False,
    filter_failure: bool = False,
    use_color: bool = True,
) -> Dict[str, Any]:
    """Analyze a single run directory."""
    results = {
        "run_name": run_dir.name,
        "fujitsu": None,
        "agentdojo": None,
    }
    
    # Load evaluation results
    fujitsu_eval_json = run_dir / "eval" / "fujitsu_eval.json"
    fujitsu_paired = run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl"
    agentdojo_eval_json = run_dir / "eval" / "agentdojo_eval.json"
    agentdojo_paired = run_dir / "eval" / "agentdojo_eval.paired_outputs.jsonl"
    
    # Try to load traces for context (from multiple possible locations)
    traces_index = {}
    if traces_dir:
        # Load Fujitsu traces
        fujitsu_traces_path = traces_dir / "fujitsu_b4_ds.jsonl"
        if fujitsu_traces_path.exists():
            traces = load_jsonl(fujitsu_traces_path)
            traces_index.update(build_trace_index(traces))
        
        # Load AgentDojo traces from split directory
        agentdojo_split = run_dir / "agentdojo_split" / "agentdojo_traces_harmful.jsonl"
        if agentdojo_split.exists():
            traces = load_jsonl(agentdojo_split)
            traces_index.update(build_trace_index(traces))
    
    # ---------------------------------------------------------
    # Fujitsu Analysis
    # ---------------------------------------------------------
    if fujitsu_eval_json.exists():
        eval_data = load_json(fujitsu_eval_json)
        paired = load_jsonl(fujitsu_paired)
        
        baseline = eval_data.get("baseline", {}).get("tool_flip_asr", {})
        cb = eval_data.get("cb_model", {}).get("tool_flip_asr", {})
        delta = eval_data.get("delta", {})
        
        results["fujitsu"] = {
            "baseline_asr": baseline.get("attack_success_rate", 0) * 100,
            "cb_asr": cb.get("attack_success_rate", 0) * 100,
            "delta": delta.get("tool_flip_asr", 0) * 100,
            "total": len(paired),
            "improvements": sum(1 for p in paired if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"),
            "regressions": sum(1 for p in paired if p.get("baseline_outcome") != "attack_success" and p.get("cb_outcome") == "attack_success"),
        }
        
        # Show sample details if requested
        if show_samples > 0 and paired:
            print(f"\n{'#'*80}")
            print(f"# FUJITSU SAMPLES: {run_dir.name}")
            print(f"{'#'*80}")
            
            # Filter samples
            if filter_success:
                samples_to_show = [
                    p for p in paired 
                    if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"
                ][:show_samples]
            elif filter_failure:
                samples_to_show = [
                    p for p in paired 
                    if p.get("cb_outcome") == "attack_success"
                ][:show_samples]
            else:
                samples_to_show = paired[:show_samples]
            
            for sample in samples_to_show:
                sample_id = sample.get("id", "")
                trace = traces_index.get(sample_id)
                print_sample_detail(sample, trace, tool_schema, show_full, use_color)
    
    # ---------------------------------------------------------
    # AgentDojo Analysis
    # ---------------------------------------------------------
    if agentdojo_eval_json.exists():
        eval_data = load_json(agentdojo_eval_json)
        paired = load_jsonl(agentdojo_paired)
        
        output_comparison = eval_data.get("output_comparison", {})
        
        results["agentdojo"] = {
            "total": output_comparison.get("total_compared", 0),
            "different": output_comparison.get("different", 0),
            "diff_rate": output_comparison.get("difference_rate", 0) * 100,
        }
        
        # Show sample details if requested
        if show_samples > 0 and paired:
            print(f"\n{'#'*80}")
            print(f"# AGENTDOJO SAMPLES: {run_dir.name}")
            print(f"{'#'*80}")
            
            # Filter to samples with different responses
            different = [p for p in paired if p.get("responses_differ")]
            samples_to_show = different[:show_samples] if different else paired[:show_samples]
            
            for sample in samples_to_show:
                sample_id = sample.get("id", "")
                trace = traces_index.get(sample_id)
                print_sample_detail(sample, trace, tool_schema, show_full, use_color)
    
    return results


# =============================================================================
# Summary Display
# =============================================================================

def print_summary_table(runs: List[Dict[str, Any]], use_color: bool = True) -> None:
    """Print a summary table of all runs."""
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    
    print(f"\n{BOLD}{'='*110}{RESET}")
    print(f"{BOLD}SWEEP RESULTS SUMMARY{RESET}")
    print(f"{'='*110}")
    
    # Header
    header = f"{'Run Name':<35} {'Base ASR':<12} {'CB ASR':<10} {'Reduction':<12} {'Improved':<10} {'Regressed':<10} {'AgentDojo':<12}"
    print(f"{BOLD}{header}{RESET}")
    print(f"{'-'*110}")
    
    for run in runs:
        run_name = run.get("run_name", "?")
        
        fujitsu = run.get("fujitsu", {})
        if fujitsu:
            base_asr = fujitsu.get("baseline_asr", 0)
            cb_asr = fujitsu.get("cb_asr", 0)
            reduction = base_asr - cb_asr
            improvements = fujitsu.get("improvements", 0)
            regressions = fujitsu.get("regressions", 0)
            
            base_str = f"{base_asr:.1f}%"
            cb_str = f"{cb_asr:.1f}%"
            red_str = f"{reduction:.1f}pp"
            
            # Color code based on reduction
            if reduction > 50:
                red_str = f"{GREEN}{red_str}{RESET}"
            elif reduction < 0:
                red_str = f"{RED}{red_str}{RESET}"
        else:
            base_str = cb_str = red_str = "N/A"
            improvements = regressions = 0
        
        agentdojo = run.get("agentdojo", {})
        if agentdojo:
            diff_rate = agentdojo.get("diff_rate", 0)
            agent_str = f"{diff_rate:.1f}%"
        else:
            agent_str = "N/A"
        
        print(f"{run_name:<35} {base_str:<12} {cb_str:<10} {red_str:<12} {improvements:<10} {regressions:<10} {agent_str:<12}")


def print_best_runs(runs: List[Dict[str, Any]], top_n: int = 5, use_color: bool = True) -> None:
    """Print the best performing runs."""
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    
    # Filter runs with valid Fujitsu data
    valid_runs = [r for r in runs if r.get("fujitsu") and r["fujitsu"].get("cb_asr") is not None]
    
    if not valid_runs:
        print("\nNo valid runs found with Fujitsu metrics.")
        return
    
    # Sort by CB ASR (lower is better)
    by_cb_asr = sorted(valid_runs, key=lambda r: r["fujitsu"]["cb_asr"])
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}TOP {top_n} RUNS BY LOWEST CB ASR{RESET}")
    print(f"{'='*80}")
    
    for i, run in enumerate(by_cb_asr[:top_n], 1):
        fujitsu = run["fujitsu"]
        print(f"\n{GREEN}{i}. {run['run_name']}{RESET}")
        print(f"   Baseline ASR: {fujitsu['baseline_asr']:.1f}% → "
              f"CB ASR: {fujitsu['cb_asr']:.1f}% "
              f"(Reduction: {fujitsu['baseline_asr'] - fujitsu['cb_asr']:.1f}pp)")
        print(f"   Improvements: {fujitsu.get('improvements', 0)}, "
              f"Regressions: {fujitsu.get('regressions', 0)}")
        agentdojo = run.get("agentdojo", {})
        if agentdojo:
            print(f"   AgentDojo Diff Rate: {agentdojo.get('diff_rate', 0):.1f}%")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter sweep results with full context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze sweep directory
    python scripts/visualize_sweep_results.py /path/to/sweep_dir
    
    # Show 10 detailed samples with full context
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --show-samples 10 --show-full
    
    # Show only samples where CB successfully blocked attacks
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --show-samples 5 --filter-success
    
    # Analyze specific run
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --run a5.0_l10_20_assistant_only
        """
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to sweep output directory (e.g., /scratch/.../sweeps/hparam_sweep_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="Directory containing trace files for context (default: auto-detect from sweep dir)"
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=Path("configs/tool_schemas/b4_standard_v1.json"),
        help="Path to tool schema JSON (default: configs/tool_schemas/b4_standard_v1.json)"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="Number of sample details to show per dataset"
    )
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Show full text instead of truncated"
    )
    parser.add_argument(
        "--filter-success",
        action="store_true",
        help="Only show samples where CB successfully blocked an attack"
    )
    parser.add_argument(
        "--filter-failure",
        action="store_true",
        help="Only show samples where CB failed to block an attack"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top runs to highlight"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Only analyze a specific run (by name, e.g., a5.0_l10_20_assistant_only)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export results to CSV file"
    )
    
    args = parser.parse_args()
    
    use_color = not args.no_color
    
    # Check sweep directory exists
    if not args.sweep_dir.exists():
        print(f"Error: Sweep directory not found: {args.sweep_dir}")
        sys.exit(1)
    
    print(f"Analyzing sweep: {args.sweep_dir}")
    
    # Load tool schema
    tool_schema = None
    if args.tool_schema.exists():
        tool_schema = load_tool_schema(args.tool_schema)
        print(f"Loaded tool schema: {args.tool_schema}")
    
    # Try to find traces directory
    traces_dir = args.traces_dir
    if traces_dir is None:
        # Try common locations
        possible_traces = [
            Path("/scratch/memoozd/cb-scratch/data/traces"),
            args.sweep_dir.parent.parent / "data" / "traces",
        ]
        for p in possible_traces:
            if p.exists():
                traces_dir = p
                print(f"Found traces directory: {traces_dir}")
                break
    
    # Find run directories
    run_dirs = sorted([
        d for d in args.sweep_dir.iterdir()
        if d.is_dir() and d.name.startswith("a")  # Run dirs start with alpha param
    ])
    
    if not run_dirs:
        print("No run directories found in sweep directory.")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} runs")
    
    # Filter to specific run if requested
    if args.run:
        run_dirs = [d for d in run_dirs if d.name == args.run]
        if not run_dirs:
            print(f"Run '{args.run}' not found.")
            sys.exit(1)
    
    # Analyze all runs
    all_results = []
    for run_dir in run_dirs:
        result = analyze_run(
            run_dir,
            traces_dir=traces_dir,
            tool_schema=tool_schema,
            show_samples=args.show_samples,
            show_full=args.show_full,
            filter_success=args.filter_success,
            filter_failure=args.filter_failure,
            use_color=use_color,
        )
        all_results.append(result)
    
    # Print summary
    print_summary_table(all_results, use_color)
    print_best_runs(all_results, args.top_n, use_color)
    
    # Export to CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_name", "baseline_asr", "cb_asr", "reduction",
                "improvements", "regressions", "agentdojo_diff_rate"
            ])
            for r in all_results:
                fujitsu = r.get("fujitsu") or {}
                agentdojo = r.get("agentdojo") or {}
                writer.writerow([
                    r.get("run_name", ""),
                    fujitsu.get("baseline_asr", "") if fujitsu else "",
                    fujitsu.get("cb_asr", "") if fujitsu else "",
                    fujitsu.get("baseline_asr", 0) - fujitsu.get("cb_asr", 0) if fujitsu else "",
                    fujitsu.get("improvements", "") if fujitsu else "",
                    fujitsu.get("regressions", "") if fujitsu else "",
                    agentdojo.get("diff_rate", "") if agentdojo else "",
                ])
        print(f"\nExported results to: {args.csv}")


if __name__ == "__main__":
    main()
