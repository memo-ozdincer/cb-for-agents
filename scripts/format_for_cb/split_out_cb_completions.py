#!/usr/bin/env python3
"""Split out completion-style Circuit Breaker examples (robust, copy-first).

Goal
- Keep original prompt-only CB pairs accessible.
- Create a *new* completion-style dataset for CB training where possible.

This script does NOT modify input files in-place.
It:
1) Backs up the input JSONL(s) to data/circuit_breakers/_backups/.
2) Writes a "remaining" JSONL that excludes extracted examples.
3) Writes a new JSONL with completion-style examples.
4) Optionally writes a "rejected" JSONL containing rows that were flagged as bad (with reasons),
   allowing you to isolate/remove them later.

Completion sources supported
- Fujitsu B1 (RAG poisoning): uses `target_llm_output` as harmful_completion
- Fujitsu B3 (direct query): uses `target_llm_output` as harmful_completion
- Fujitsu B2 (image poisoning): uses `baseline_output` or `mta_output` as harmful_completion
- AgentDojo traces: derives a completion by concatenating assistant messages
- AttackQA (benign): uses the ground-truth `answer` as benign_completion (from parquet if available)

Notes
- This is intentionally conservative and straightforward.
- For Llama/Mistral chat templating, store `user_prompt` + `*_completion` + a simple `text`.
  You can later regenerate `text` via tokenizer chat templates on GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    pd = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CB_DIR = DATA_DIR / "circuit_breakers"

FUJITSU_B1 = DATA_DIR / "fujitsu" / "rag_poisoning_benchmark_combined_deduplicated.jsonl"
FUJITSU_B3 = DATA_DIR / "fujitsu" / "safety_benchmark_direct_query_combined_deduplicated.jsonl"
FUJITSU_B2 = DATA_DIR / "fujitsu" / "image_poisoning_simulation_results_20250504_202954.jsonl"
ATTACKQA_PARQUET = DATA_DIR / "attackqa" / "attackqa.parquet"
AGENTDOJO_DIR = DATA_DIR / "agent_dojo"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def backup_file(path: Path, backup_dir: Path) -> Optional[Path]:
    if not path.exists() or not path.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.{ts}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def _norm_text(x: Any) -> str:
    s = str(x or "")
    s = s.replace("\x00", "")
    return s.strip()


def _get_benchmark(md: Dict[str, Any]) -> str:
    b = md.get("benchmark")
    return str(b or "")


def _compile_regexes(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for p in patterns:
        p = str(p).strip()
        if not p:
            continue
        out.append(re.compile(p, flags=re.IGNORECASE | re.MULTILINE))
    return out


def _parse_kv(s: str) -> Optional[Tuple[str, str]]:
    s = str(s or "").strip()
    if not s or "=" not in s:
        return None
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        return None
    return k, v


def _get_nested(md: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = md
    for part in str(dotted_key).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _join_assistant_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content")
            if content:
                parts.append(str(content))
            # Some AgentDojo traces store tool calls on the assistant message
            tool_calls = m.get("tool_calls")
            if tool_calls:
                parts.append("[TOOL_CALLS] " + json.dumps(tool_calls, ensure_ascii=False))
    return "\n".join(p.strip() for p in parts if p and str(p).strip()).strip()


def _first_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return str(m.get("content") or "").strip()
    return ""


def _agentdojo_index_key(md: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(md.get("user_task_id") or ""),
        str(md.get("injection_task_id") or ""),
        str(md.get("suite_name") or ""),
        str(md.get("model") or ""),
    )


def build_agentdojo_index(file_path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in read_jsonl(file_path):
        md = row.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        key = _agentdojo_index_key(md)
        # If collisions happen, keep the first.
        index.setdefault(key, row)
    return index


def build_fujitsu_index(path: Path, id_field: str = "id") -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        rid = row.get(id_field)
        if rid:
            idx[str(rid)] = row
    return idx


def _assessment_success(x: Any) -> Optional[bool]:
    """Return True/False if an assessment looks like Success/Failure; else None."""
    if isinstance(x, dict):
        a = x.get("assessment")
    else:
        a = x
    if a is None:
        return None
    s = str(a).strip().lower()
    if s == "success":
        return True
    if s == "failure":
        return False
    return None


def _as_list(x: Optional[str]) -> List[str]:
    if not x:
        return []
    return [s.strip() for s in str(x).split(",") if s.strip()]


@dataclass
class FilterConfig:
    only_sources: List[str]
    skip_sources: List[str]
    only_benchmarks: List[str]
    skip_benchmarks: List[str]
    reject_regexes: List[re.Pattern[str]]
    reject_metadata: List[Tuple[str, str]]
    min_user_chars: int
    min_completion_chars: int
    max_completion_chars: int
    exclude_rejected: bool


def _filter_reasons(
    *,
    source: str,
    benchmark: str,
    user_prompt: str,
    completion: str,
    combined_text: str,
    metadata: Dict[str, Any],
    cfg: FilterConfig,
) -> List[str]:
    reasons: List[str] = []

    if cfg.only_sources and source not in cfg.only_sources:
        reasons.append("source_not_in_only_sources")
    if cfg.skip_sources and source in cfg.skip_sources:
        reasons.append("source_in_skip_sources")

    if cfg.only_benchmarks and benchmark and benchmark not in cfg.only_benchmarks:
        reasons.append("benchmark_not_in_only_benchmarks")
    if cfg.skip_benchmarks and benchmark and benchmark in cfg.skip_benchmarks:
        reasons.append("benchmark_in_skip_benchmarks")

    if len(user_prompt) < cfg.min_user_chars:
        reasons.append("user_prompt_too_short")
    if completion and len(completion) < cfg.min_completion_chars:
        reasons.append("completion_too_short")
    if completion and cfg.max_completion_chars > 0 and len(completion) > cfg.max_completion_chars:
        reasons.append("completion_too_long")

    for rx in cfg.reject_regexes:
        if rx.search(combined_text):
            reasons.append(f"reject_regex:{rx.pattern}")

    for k, v in cfg.reject_metadata:
        actual = _get_nested(metadata, k)
        if actual is None:
            continue
        if str(actual) == v:
            reasons.append(f"reject_metadata:{k}={v}")

    return reasons


@dataclass
class SplitResult:
    extracted: int
    remaining: int
    agentic_count: int = 0  # Number of is_agentic=True samples
    with_tool_calls: int = 0  # Number with has_tool_calls=True
    with_messages: int = 0  # Number with full messages[] array


@dataclass
class SplitArtifacts:
    extracted: List[Dict[str, Any]]
    remaining: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]


def split_harmful(
    harmful_pairs_path: Path,
    out_remaining: Path,
    out_completions: Path,
    out_rejected: Optional[Path],
    cfg: FilterConfig,
    fujitsu_success_only: bool,
    fujitsu_b2_output: str,
    dry_run: bool,
) -> SplitResult:
    rows = read_jsonl(harmful_pairs_path)

    # Build indices used to fetch completions.
    fujitsu_b1_idx = build_fujitsu_index(FUJITSU_B1, id_field="id") if FUJITSU_B1.exists() else {}
    fujitsu_b3_idx = build_fujitsu_index(FUJITSU_B3, id_field="id") if FUJITSU_B3.exists() else {}
    fujitsu_b2_idx = build_fujitsu_index(FUJITSU_B2, id_field="attack_id") if FUJITSU_B2.exists() else {}

    agentdojo_indices: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, Any]]] = {}

    extracted_rows: List[Dict[str, Any]] = []
    remaining_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    for r in rows:
        src = r.get("source")
        rid = str(r.get("id") or "")
        md = r.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}

        benchmark = _get_benchmark(md)

        # Fujitsu B1/B3: use target_llm_output as harmful completion
        if src == "fujitsu" and rid.startswith("fujitsu_b1_"):
            f_id = rid.replace("fujitsu_b1_", "", 1)
            orig = fujitsu_b1_idx.get(f_id)
            completion = (orig or {}).get("target_llm_output")
            user_prompt = (orig or {}).get("user_query")
            user_prompt_n = _norm_text(user_prompt)
            completion_n = _norm_text(completion)
            if fujitsu_success_only:
                ok = _assessment_success((orig or {}).get("judge_assessment"))
                if ok is False:
                    remaining_rows.append(r)
                    continue
            if completion_n and user_prompt_n:
                candidate = {
                    "id": rid,
                    "source": "fujitsu",
                    "category": r.get("category"),
                    "subtype": r.get("subtype"),
                    "user_prompt": user_prompt_n,
                    "harmful_completion": completion_n,
                    "text": f"User: {user_prompt_n}\nAssistant: {completion_n}",
                    "metadata": {"from": "fujitsu_b1", "orig_id": f_id, **md},
                }
                reasons = _filter_reasons(
                    source=str(src or ""),
                    benchmark=benchmark,
                    user_prompt=user_prompt_n,
                    completion=completion_n,
                    combined_text=candidate.get("text", ""),
                    metadata=md,
                    cfg=cfg,
                )
                if reasons:
                    rejected_rows.append({"side": "harmful", "reasons": reasons, "row": r})
                    if not cfg.exclude_rejected:
                        remaining_rows.append(r)
                else:
                    extracted_rows.append(candidate)
                continue

        # Fujitsu B4 (Orchestrator): AGENTIC tool manipulation attacks
        # These don't have target_llm_output - they have tool_attack{} with tool flip info
        if src == "fujitsu" and rid.startswith("fujitsu_b4_"):
            tool_attack = r.get("tool_attack") or {}
            if tool_attack:
                benign_q = _norm_text(tool_attack.get("benign_query"))
                malicious_inj = _norm_text(tool_attack.get("malicious_injection"))
                combined_q = _norm_text(tool_attack.get("combined_query"))
                expected_tool = tool_attack.get("expected_tool", "")
                simulated_tool = tool_attack.get("simulated_tool", "")

                if fujitsu_success_only:
                    # B4 success = tool flip occurred
                    if md.get("success") is not True:
                        remaining_rows.append(r)
                        continue

                # For B4, create a structured "completion" that shows the harmful tool flip
                # This is the agentic harm: the agent called the WRONG tool
                user_prompt_n = combined_q if combined_q else benign_q
                harmful_completion = f"[TOOL_CALL] {simulated_tool}\n(Expected: {expected_tool}, but injection caused flip to: {simulated_tool})"

                if user_prompt_n:
                    candidate = {
                        "id": rid,
                        "source": "fujitsu",
                        "category": r.get("category"),
                        "subtype": r.get("subtype"),
                        "is_agentic": True,
                        "has_tool_calls": True,
                        "tool_attack": tool_attack,  # Preserve full tool attack structure
                        "user_prompt": user_prompt_n,
                        "harmful_completion": harmful_completion,
                        "text": f"User: {user_prompt_n}\nAssistant: {harmful_completion}",
                        "metadata": {"from": "fujitsu_b4_orchestrator", **md},
                    }
                    reasons = _filter_reasons(
                        source=str(src or ""),
                        benchmark=benchmark,
                        user_prompt=user_prompt_n,
                        completion=harmful_completion,
                        combined_text=candidate.get("text", ""),
                        metadata=md,
                        cfg=cfg,
                    )
                    if reasons:
                        rejected_rows.append({"side": "harmful", "reasons": reasons, "row": r})
                        if not cfg.exclude_rejected:
                            remaining_rows.append(r)
                    else:
                        extracted_rows.append(candidate)
                    continue

        if src == "fujitsu" and rid.startswith("fujitsu_b3_"):
            f_id = rid.replace("fujitsu_b3_", "", 1)
            orig = fujitsu_b3_idx.get(f_id)
            completion = (orig or {}).get("target_llm_output")
            user_prompt = (orig or {}).get("user_query")
            user_prompt_n = _norm_text(user_prompt)
            completion_n = _norm_text(completion)
            if fujitsu_success_only:
                ok = _assessment_success((orig or {}).get("judge_assessment"))
                if ok is False:
                    remaining_rows.append(r)
                    continue
            if completion_n and user_prompt_n:
                candidate = {
                    "id": rid,
                    "source": "fujitsu",
                    "category": r.get("category"),
                    "subtype": r.get("subtype"),
                    "user_prompt": user_prompt_n,
                    "harmful_completion": completion_n,
                    "text": f"User: {user_prompt_n}\nAssistant: {completion_n}",
                    "metadata": {"from": "fujitsu_b3", "orig_id": f_id, **md},
                }
                reasons = _filter_reasons(
                    source=str(src or ""),
                    benchmark=benchmark,
                    user_prompt=user_prompt_n,
                    completion=completion_n,
                    combined_text=candidate.get("text", ""),
                    metadata=md,
                    cfg=cfg,
                )
                if reasons:
                    rejected_rows.append({"side": "harmful", "reasons": reasons, "row": r})
                    if not cfg.exclude_rejected:
                        remaining_rows.append(r)
                else:
                    extracted_rows.append(candidate)
                continue

        # Fujitsu B2: choose baseline/mta output as completion (when available)
        if src == "fujitsu" and rid.startswith("fujitsu_b2_"):
            a_id = rid.replace("fujitsu_b2_", "", 1)
            orig = fujitsu_b2_idx.get(a_id)
            user_prompt_n = _norm_text((orig or {}).get("user_query"))

            baseline_out = _norm_text((orig or {}).get("baseline_output"))
            mta_out = _norm_text((orig or {}).get("mta_output"))
            baseline_ok = _assessment_success((orig or {}).get("baseline_judge_assessment"))
            mta_ok = _assessment_success((orig or {}).get("mta_judge_assessment"))

            chosen: str = ""
            chosen_from: str = ""

            mode = (fujitsu_b2_output or "auto").lower()
            if mode == "baseline":
                chosen = baseline_out
                chosen_from = "fujitsu_b2_baseline"
                if fujitsu_success_only and baseline_ok is False:
                    remaining_rows.append(r)
                    continue
            elif mode == "mta":
                chosen = mta_out
                chosen_from = "fujitsu_b2_mta"
                if fujitsu_success_only and mta_ok is False:
                    remaining_rows.append(r)
                    continue
            else:
                # auto: prefer successful baseline; else successful mta
                if baseline_out and (baseline_ok is True or not fujitsu_success_only):
                    chosen = baseline_out
                    chosen_from = "fujitsu_b2_baseline"
                elif mta_out and (mta_ok is True or not fujitsu_success_only):
                    chosen = mta_out
                    chosen_from = "fujitsu_b2_mta"

                if not chosen:
                    remaining_rows.append(r)
                    continue

            if user_prompt_n and chosen:
                candidate = {
                    "id": rid,
                    "source": "fujitsu",
                    "category": r.get("category"),
                    "subtype": r.get("subtype"),
                    "user_prompt": user_prompt_n,
                    "harmful_completion": chosen,
                    "text": f"User: {user_prompt_n}\nAssistant: {chosen}",
                    "metadata": {"from": chosen_from, "orig_id": a_id, **md},
                }
                reasons = _filter_reasons(
                    source=str(src or ""),
                    benchmark=benchmark,
                    user_prompt=user_prompt_n,
                    completion=chosen,
                    combined_text=candidate.get("text", ""),
                    metadata=md,
                    cfg=cfg,
                )
                if reasons:
                    rejected_rows.append({"side": "harmful", "reasons": reasons, "row": r})
                    if not cfg.exclude_rejected:
                        remaining_rows.append(r)
                else:
                    extracted_rows.append(candidate)
                continue

        # AgentDojo: derive completion from assistant messages
        # NEW: Check if messages already exist in ingested data (agentic format)
        if src == "agentdojo" and rid.startswith("agentdojo_harm_"):
            # First try to use messages already in the ingested row (new agentic format)
            messages = r.get("messages") or []

            # Fallback: re-read from original AgentDojo files
            if not messages:
                origin_file = md.get("origin_file")
                if origin_file:
                    if origin_file not in agentdojo_indices:
                        p = AGENTDOJO_DIR / origin_file
                        if p.exists():
                            agentdojo_indices[origin_file] = build_agentdojo_index(p)
                        else:
                            agentdojo_indices[origin_file] = {}

                    key = _agentdojo_index_key(md)
                    rec = agentdojo_indices.get(origin_file, {}).get(key)
                    if rec:
                        messages = rec.get("messages") or []

            if isinstance(messages, list) and messages:
                user_prompt_n = _norm_text(_first_user_message(messages))
                completion_n = _norm_text(_join_assistant_messages(messages))
                if user_prompt_n and completion_n:
                    # Check if this is truly agentic (has tool calls)
                    is_agentic = r.get("is_agentic", False)
                    has_tool_calls = r.get("has_tool_calls", False) or any(
                        m.get("tool_calls") or m.get("role") == "tool"
                        for m in messages
                    )

                    candidate = {
                        "id": rid,
                        "source": "agentdojo",
                        "category": r.get("category"),
                        "subtype": r.get("subtype"),
                        "is_agentic": is_agentic or has_tool_calls,
                        "has_tool_calls": has_tool_calls,
                        "messages": messages,  # PRESERVE FULL MULTI-TURN TRACE
                        "user_prompt": user_prompt_n,
                        "harmful_completion": completion_n,
                        "text": f"User: {user_prompt_n}\nAssistant: {completion_n}",
                        "metadata": {"from": "agentdojo_trace", **md},
                    }
                    reasons = _filter_reasons(
                        source=str(src or ""),
                        benchmark=benchmark,
                        user_prompt=user_prompt_n,
                        completion=completion_n,
                        combined_text=candidate.get("text", ""),
                        metadata=md,
                        cfg=cfg,
                    )
                    if reasons:
                        rejected_rows.append({"side": "harmful", "reasons": reasons, "row": r})
                        if not cfg.exclude_rejected:
                            remaining_rows.append(r)
                    else:
                        extracted_rows.append(candidate)
                    continue

        remaining_rows.append(r)

    if not dry_run:
        backups_dir = CB_DIR / "_backups"
        backup_file(harmful_pairs_path, backups_dir)
        backup_file(out_remaining, backups_dir)
        backup_file(out_completions, backups_dir)
        if out_rejected is not None:
            backup_file(out_rejected, backups_dir)
        write_jsonl(out_remaining, remaining_rows)
        write_jsonl(out_completions, extracted_rows)
        if out_rejected is not None:
            write_jsonl(out_rejected, rejected_rows)

    # Compute agentic statistics
    agentic_count = sum(1 for r in extracted_rows if r.get("is_agentic"))
    with_tool_calls = sum(1 for r in extracted_rows if r.get("has_tool_calls"))
    with_messages = sum(1 for r in extracted_rows if r.get("messages"))

    return SplitResult(
        extracted=len(extracted_rows),
        remaining=len(remaining_rows),
        agentic_count=agentic_count,
        with_tool_calls=with_tool_calls,
        with_messages=with_messages,
    )


def split_benign(
    benign_pairs_path: Path,
    out_remaining: Path,
    out_completions: Path,
    out_rejected: Optional[Path],
    cfg: FilterConfig,
    attackqa_parquet: Path,
    dry_run: bool,
) -> SplitResult:
    rows = read_jsonl(benign_pairs_path)

    attackqa_df = None
    if any((r.get("source") == "attackqa") for r in rows) and attackqa_parquet.exists() and pd is not None:
        try:
            attackqa_df = pd.read_parquet(attackqa_parquet)
        except Exception:
            attackqa_df = None

    agentdojo_indices: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, Any]]] = {}

    extracted_rows: List[Dict[str, Any]] = []
    remaining_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    for r in rows:
        src = r.get("source")
        rid = str(r.get("id") or "")
        md = r.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}

        benchmark = _get_benchmark(md)

        # AttackQA benign: question+answer as completion
        if src == "attackqa" and rid.startswith("attackqa_"):
            user_prompt_n = _norm_text(r.get("prompt"))
            ans = ""
            # Prefer parquet (full answer), fallback to ingested metadata (truncated)
            if attackqa_df is not None:
                try:
                    idx = int(rid.replace("attackqa_", "", 1))
                    rec = attackqa_df.loc[idx]
                    if hasattr(rec, "to_dict"):
                        recd = rec.to_dict()
                        ans = _norm_text(recd.get("answer"))
                    else:
                        ans = _norm_text(getattr(rec, "answer", ""))
                except Exception:
                    ans = ""
            if not ans:
                ans = _norm_text(md.get("answer"))

            if user_prompt_n and ans:
                candidate = {
                    "id": rid,
                    "source": "attackqa",
                    "category": r.get("category"),
                    "user_prompt": user_prompt_n,
                    "benign_completion": ans,
                    "text": f"User: {user_prompt_n}\nAssistant: {ans}",
                    "metadata": {"from": "attackqa", **md},
                }
                reasons = _filter_reasons(
                    source=str(src or ""),
                    benchmark=benchmark,
                    user_prompt=user_prompt_n,
                    completion=ans,
                    combined_text=candidate.get("text", ""),
                    metadata=md,
                    cfg=cfg,
                )
                if reasons:
                    rejected_rows.append({"side": "benign", "reasons": reasons, "row": r})
                    if not cfg.exclude_rejected:
                        remaining_rows.append(r)
                else:
                    extracted_rows.append(candidate)
                continue

        # AgentDojo benign: derive completion from assistant messages
        # NEW: Check if messages already exist in ingested data (agentic format)
        if src == "agentdojo" and rid.startswith("agentdojo_benign_"):
            # First try to use messages already in the ingested row (new agentic format)
            messages = r.get("messages") or []

            # Fallback: re-read from original AgentDojo files
            if not messages:
                origin_file = md.get("origin_file")
                if origin_file:
                    if origin_file not in agentdojo_indices:
                        p = AGENTDOJO_DIR / origin_file
                        if p.exists():
                            agentdojo_indices[origin_file] = build_agentdojo_index(p)
                        else:
                            agentdojo_indices[origin_file] = {}

                    key = _agentdojo_index_key(md)
                    rec = agentdojo_indices.get(origin_file, {}).get(key)
                    if rec:
                        messages = rec.get("messages") or []

            if isinstance(messages, list) and messages:
                user_prompt_n = _norm_text(_first_user_message(messages))
                completion_n = _norm_text(_join_assistant_messages(messages))
                if user_prompt_n and completion_n:
                    # Check if this is truly agentic (has tool calls)
                    is_agentic = r.get("is_agentic", False)
                    has_tool_calls = r.get("has_tool_calls", False) or any(
                        m.get("tool_calls") or m.get("role") == "tool"
                        for m in messages
                    )

                    candidate = {
                        "id": rid,
                        "source": "agentdojo",
                        "category": r.get("category"),
                        "is_agentic": is_agentic or has_tool_calls,
                        "has_tool_calls": has_tool_calls,
                        "messages": messages,  # PRESERVE FULL MULTI-TURN TRACE
                        "user_prompt": user_prompt_n,
                        "benign_completion": completion_n,
                        "text": f"User: {user_prompt_n}\nAssistant: {completion_n}",
                        "metadata": {"from": "agentdojo_trace", **md},
                    }
                    reasons = _filter_reasons(
                        source=str(src or ""),
                        benchmark=benchmark,
                        user_prompt=user_prompt_n,
                        completion=completion_n,
                        combined_text=candidate.get("text", ""),
                        metadata=md,
                        cfg=cfg,
                    )
                    if reasons:
                        rejected_rows.append({"side": "benign", "reasons": reasons, "row": r})
                        if not cfg.exclude_rejected:
                            remaining_rows.append(r)
                    else:
                        extracted_rows.append(candidate)
                    continue

        remaining_rows.append(r)

    if not dry_run:
        backups_dir = CB_DIR / "_backups"
        backup_file(benign_pairs_path, backups_dir)
        backup_file(out_remaining, backups_dir)
        backup_file(out_completions, backups_dir)
        if out_rejected is not None:
            backup_file(out_rejected, backups_dir)
        write_jsonl(out_remaining, remaining_rows)
        write_jsonl(out_completions, extracted_rows)
        if out_rejected is not None:
            write_jsonl(out_rejected, rejected_rows)

    # Compute agentic statistics
    agentic_count = sum(1 for r in extracted_rows if r.get("is_agentic"))
    with_tool_calls = sum(1 for r in extracted_rows if r.get("has_tool_calls"))
    with_messages = sum(1 for r in extracted_rows if r.get("messages"))

    return SplitResult(
        extracted=len(extracted_rows),
        remaining=len(remaining_rows),
        agentic_count=agentic_count,
        with_tool_calls=with_tool_calls,
        with_messages=with_messages,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Do not write files; only print counts")
    p.add_argument(
        "--write-rejected",
        action="store_true",
        help="Write rejected rows (with reasons) to *.rejected.jsonl files",
    )
    p.add_argument(
        "--exclude-rejected",
        action="store_true",
        help="If set, rejected rows are removed from prompt-only outputs as well (still written to rejected files if enabled)",
    )
    p.add_argument(
        "--only-sources",
        type=str,
        default="",
        help="Comma-separated sources eligible for extraction (others remain prompt-only). Example: agentdojo,fujitsu,attackqa",
    )
    p.add_argument(
        "--skip-sources",
        type=str,
        default="",
        help="Comma-separated sources to never extract completions from.",
    )
    p.add_argument(
        "--only-benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmark tags (metadata.benchmark) eligible for extraction.",
    )
    p.add_argument(
        "--skip-benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmark tags (metadata.benchmark) to exclude from extraction.",
    )
    p.add_argument(
        "--reject-regex",
        action="append",
        default=[],
        help="Regex (can be repeated). If it matches the candidate text, the row is rejected (and optionally excluded).",
    )
    p.add_argument(
        "--reject-metadata",
        action="append",
        default=[],
        help="Metadata exact-match rejection rule (repeatable) in the form key=value (supports dotted keys). Example: security=False",
    )
    p.add_argument("--min-user-chars", type=int, default=1, help="Reject if user_prompt shorter than this")
    p.add_argument("--min-completion-chars", type=int, default=1, help="Reject if completion shorter than this")
    p.add_argument(
        "--max-completion-chars",
        type=int,
        default=20000,
        help="Reject if completion longer than this (0 disables max)",
    )
    p.add_argument(
        "--fujitsu-success-only",
        action="store_true",
        help="For Fujitsu harmful extractions, only extract rows where the judge assessment indicates Success.",
    )
    p.add_argument(
        "--fujitsu-b2-output",
        type=str,
        default="auto",
        choices=["auto", "baseline", "mta"],
        help="Which Fujitsu B2 output field to use as completion.",
    )
    p.add_argument(
        "--attackqa-parquet",
        type=str,
        default=str(ATTACKQA_PARQUET),
        help="Path to AttackQA parquet (for full answers).",
    )
    p.add_argument(
        "--harmful-in",
        type=str,
        default=str(CB_DIR / "harmful" / "harmful_pairs.jsonl"),
        help="Input harmful_pairs.jsonl",
    )
    p.add_argument(
        "--benign-in",
        type=str,
        default=str(CB_DIR / "benign" / "benign_pairs.jsonl"),
        help="Input benign_pairs.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(CB_DIR),
        help="Base output dir (defaults to data/circuit_breakers)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = FilterConfig(
        only_sources=_as_list(args.only_sources),
        skip_sources=_as_list(args.skip_sources),
        only_benchmarks=_as_list(args.only_benchmarks),
        skip_benchmarks=_as_list(args.skip_benchmarks),
        reject_regexes=_compile_regexes(args.reject_regex or []),
        reject_metadata=[x for x in (_parse_kv(s) for s in (args.reject_metadata or [])) if x],
        min_user_chars=int(args.min_user_chars),
        min_completion_chars=int(args.min_completion_chars),
        max_completion_chars=int(args.max_completion_chars),
        exclude_rejected=bool(args.exclude_rejected),
    )

    out_dir = Path(args.out_dir)
    harmful_in = Path(args.harmful_in)
    benign_in = Path(args.benign_in)

    harmful_remaining = out_dir / "harmful" / "harmful_pairs.prompt_only.jsonl"
    harmful_completions = out_dir / "harmful" / "harmful_pairs.completions.jsonl"
    harmful_rejected = out_dir / "harmful" / "harmful_pairs.rejected.jsonl" if args.write_rejected else None

    benign_remaining = out_dir / "benign" / "benign_pairs.prompt_only.jsonl"
    benign_completions = out_dir / "benign" / "benign_pairs.completions.jsonl"
    benign_rejected = out_dir / "benign" / "benign_pairs.rejected.jsonl" if args.write_rejected else None

    hr = split_harmful(
        harmful_in,
        harmful_remaining,
        harmful_completions,
        harmful_rejected,
        cfg,
        fujitsu_success_only=bool(args.fujitsu_success_only),
        fujitsu_b2_output=str(args.fujitsu_b2_output),
        dry_run=args.dry_run,
    )
    br = split_benign(
        benign_in,
        benign_remaining,
        benign_completions,
        benign_rejected,
        cfg,
        attackqa_parquet=Path(args.attackqa_parquet),
        dry_run=args.dry_run,
    )

    print("\n=== Split Summary ===")
    print(f"Harmful extracted to completions: {hr.extracted}")
    print(f"Harmful remaining prompt-only:     {hr.remaining}")
    print(f"Benign extracted to completions:  {br.extracted}")
    print(f"Benign remaining prompt-only:     {br.remaining}")

    # AGENTIC DATA BREAKDOWN
    print(f"\n=== Agentic Data Breakdown ===")
    print(f"HARMFUL with completions:")
    print(f"  is_agentic=True:     {hr.agentic_count:>6} ({100*hr.agentic_count/hr.extracted if hr.extracted else 0:.1f}%)")
    print(f"  has_tool_calls:      {hr.with_tool_calls:>6} ({100*hr.with_tool_calls/hr.extracted if hr.extracted else 0:.1f}%)")
    print(f"  with messages[]:     {hr.with_messages:>6} (full multi-turn traces)")
    print(f"BENIGN with completions:")
    print(f"  is_agentic=True:     {br.agentic_count:>6} ({100*br.agentic_count/br.extracted if br.extracted else 0:.1f}%)")
    print(f"  has_tool_calls:      {br.with_tool_calls:>6} ({100*br.with_tool_calls/br.extracted if br.extracted else 0:.1f}%)")
    print(f"  with messages[]:     {br.with_messages:>6} (full multi-turn traces)")

    if args.dry_run:
        print("\n(dry-run) No files written.")
    else:
        print("\nWrote:")
        print(f"- {harmful_remaining}")
        print(f"- {harmful_completions}")
        if harmful_rejected is not None:
            print(f"- {harmful_rejected}")
        print(f"- {benign_remaining}")
        print(f"- {benign_completions}")
        if benign_rejected is not None:
            print(f"- {benign_rejected}")


if __name__ == "__main__":
    main()
