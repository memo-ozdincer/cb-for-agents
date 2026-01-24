#!/usr/bin/env python3
"""
ETL_B: Tier B (trace_v1) -> Tier C (render_v1 + lossmask_v1).

Renders traces with apply_chat_template and applies LMP policies to
materialize per-token loss masks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import AutoTokenizer

from src.schemas.trace import Trace
from src.schemas.render import (
    RenderedView,
    RenderOptions,
    RenderAlignment,
    MessageSpan,
    AssistantSpan,
    ToolCallSpan,
    RenderSignals,
    InjectionSpan,
    ActionCommitment,
    DetectorMetadata,
)
from src.schemas.lossmask import LossMask
from src.schemas.registry import (
    LMPRegistry,
    LMPPolicy,
    MWCSRegistry,
    load_lmp_registry,
    load_mwcs_registry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# IO Helpers
# =============================================================================

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# Registry Loading
# =============================================================================

def _default_lmp_registry() -> LMPRegistry:
    policies = {
        "assistant_only": LMPPolicy(
            name="Assistant Only",
            strategy="assistant_only",
            description="Loss only on assistant messages.",
        ),
        "completion_only": LMPPolicy(
            name="Completion Only",
            strategy="completion_only",
            description="Loss only on the final assistant message.",
        ),
        "full_sequence": LMPPolicy(
            name="Full Sequence",
            strategy="full_sequence",
            description="Loss on all tokens.",
        ),
        "tool_calls_only": LMPPolicy(
            name="Tool Calls Only",
            strategy="tool_calls_only",
            description="Loss only on tool call spans.",
        ),
        "action_prefix_only": LMPPolicy(
            name="Action Prefix Only",
            strategy="action_prefix_only",
            description="Loss up to tool name in tool call.",
        ),
        "action_commitment": LMPPolicy(
            name="Action Commitment",
            strategy="action_commitment",
            description="Loss on commitment prefix tokens.",
        ),
    }
    return LMPRegistry(version="1.0.0", policies=policies, default_policy="assistant_only")


def _load_lmp_registry_safe(path: Optional[Path]) -> LMPRegistry:
    if path is not None and path.exists():
        return load_lmp_registry(path)
    try:
        return load_lmp_registry()
    except Exception:
        logger.warning("Falling back to default LMP registry (no registry file found).")
        return _default_lmp_registry()


# =============================================================================
# Rendering Helpers
# =============================================================================

def _trace_to_chat_messages(trace: Trace) -> List[Dict[str, Any]]:
    chat_messages = []
    for msg in trace.messages:
        m: Dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
        }
        if msg.name:
            m["name"] = msg.name
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            tc_list = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                args_json = tc.function.arguments_json
                if args_json is None and args is not None:
                    args_json = json.dumps(args, ensure_ascii=False)
                tc_list.append({
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": args_json or "{}",
                    },
                })
            m["tool_calls"] = tc_list
        chat_messages.append(m)
    return chat_messages


def _compute_prefixes(
    tokenizer,
    chat_messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
) -> Tuple[List[int], List[int], List[str]]:
    token_ends: List[int] = []
    char_ends: List[int] = []
    prefix_texts: List[str] = []

    for i in range(len(chat_messages)):
        prefix = chat_messages[: i + 1]
        rendered_text = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prefix_texts.append(rendered_text)

        input_ids = tokenizer.apply_chat_template(
            prefix,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        token_ends.append(len(input_ids))
        char_ends.append(len(rendered_text))

    return token_ends, char_ends, prefix_texts


def _char_span_to_token_span(
    offsets: Optional[List[Tuple[int, int]]],
    char_start: int,
    char_end: int,
) -> Tuple[int, int]:
    if not offsets:
        return 0, 0

    token_start = None
    token_end = None

    for idx, (start, end) in enumerate(offsets):
        if end > char_start and token_start is None:
            token_start = idx
        if start < char_end:
            token_end = idx
        if start >= char_end:
            break

    if token_start is None:
        token_start = 0
    if token_end is None:
        token_end = max(0, len(offsets) - 1)

    return token_start, token_end + 1


def _compute_alignment(
    trace: Trace,
    render: RenderedView,
    tokenizer,
    rendered_text: str,
    prefix_token_ends: List[int],
    prefix_char_ends: List[int],
) -> RenderAlignment:
    message_spans: List[MessageSpan] = []
    assistant_spans: List[AssistantSpan] = []
    tool_call_spans: List[ToolCallSpan] = []

    offsets = None
    if getattr(tokenizer, "is_fast", False):
        try:
            offsets = tokenizer(
                rendered_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            ).get("offset_mapping")
        except Exception:
            offsets = None

    for i, msg in enumerate(trace.messages):
        token_start = prefix_token_ends[i - 1] if i > 0 else 0
        token_end = prefix_token_ends[i]

        message_spans.append(MessageSpan(
            message_index=i,
            role=msg.role,
            token_start=token_start,
            token_end=token_end,
        ))

        if msg.role == "assistant":
            assistant_spans.append(AssistantSpan(
                message_index=i,
                token_start=token_start,
                token_end=token_end,
            ))

            if msg.tool_calls:
                char_start = prefix_char_ends[i - 1] if i > 0 else 0
                char_end = prefix_char_ends[i]
                span_text = rendered_text[char_start:char_end]

                for j, tc in enumerate(msg.tool_calls):
                    name_pattern = f'"name"'  # fallback
                    name_pos = span_text.find(name_pattern)
                    if name_pos >= 0:
                        name_pos = span_text.find(tc.function.name, name_pos)
                    else:
                        name_pos = span_text.find(tc.function.name)

                    name_token_end = None
                    if name_pos >= 0 and offsets:
                        name_char_end = char_start + name_pos + len(tc.function.name)
                        _, name_token_end = _char_span_to_token_span(
                            offsets, char_start, name_char_end
                        )

                    tc_start = token_start
                    tc_end = token_end
                    if offsets and name_pos >= 0:
                        tc_char_start = char_start
                        tc_char_end = char_end
                        tc_start, tc_end = _char_span_to_token_span(
                            offsets, tc_char_start, tc_char_end
                        )

                    tool_call_spans.append(ToolCallSpan(
                        message_index=i,
                        call_index=j,
                        tool_name=tc.function.name,
                        token_start=tc_start,
                        token_end=tc_end,
                        name_token_end=name_token_end,
                    ))

    return RenderAlignment(
        message_spans=message_spans or None,
        assistant_spans=assistant_spans or None,
        tool_call_spans=tool_call_spans or None,
    )


def _build_basic_signals(
    trace: Trace,
    render: RenderedView,
    offsets: Optional[List[Tuple[int, int]]],
    message_char_starts: List[int],
    rendered_text: str,
) -> Optional[RenderSignals]:
    injection_spans = []
    if trace.signal_hints and trace.signal_hints.injection_char_span and offsets:
        span = trace.signal_hints.injection_char_span
        if span.message_index < 0 or span.message_index >= len(message_char_starts):
            span = None
        if span is not None:
            char_base = message_char_starts[span.message_index]
            char_start = char_base + span.char_start
            char_end = char_base + span.char_end
        else:
            char_start = None
            char_end = None

        if char_start is not None and char_end is not None:
            token_start, token_end = _char_span_to_token_span(
                offsets,
                char_start,
                char_end,
            )
            if token_end > token_start:
                injection_spans.append(InjectionSpan(
                    token_start=token_start,
                    token_end=token_end,
                    detection_method="contiguous_threshold",
                ))

    action_commitments = []
    if render.alignment and render.alignment.tool_call_spans:
        for span in render.alignment.tool_call_spans:
            end_idx = span.name_token_end or span.token_end
            if end_idx is None:
                continue
            prefix_tokens = list(range(span.token_start, min(end_idx, span.token_end)))
            action_commitments.append(ActionCommitment(
                commitment_token_idx=max(span.token_start, end_idx - 1),
                commit_type="tool_name_selected",
                assistant_message_index=span.message_index,
                committed_tool=span.tool_name,
                prefix_token_indices=prefix_tokens,
            ))

    if not injection_spans and not action_commitments:
        return None

    return RenderSignals(
        injection_spans=injection_spans or None,
        action_commitments=action_commitments or None,
        detector_metadata=DetectorMetadata(
            shock_detector_id="hint_projection" if injection_spans else None,
            commitment_detector_id="tool_name_selected" if action_commitments else None,
        ),
    )


def _extract_offsets(tokenizer, rendered_text: str) -> Optional[List[Tuple[int, int]]]:
    if getattr(tokenizer, "is_fast", False):
        try:
            return tokenizer(
                rendered_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            ).get("offset_mapping")
        except Exception:
            return None
    return None


def _compute_special_tokens(render: RenderedView, tokenizer) -> None:
    from src.schemas.render import SpecialTokenPositions

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")

    bos_position = None
    if bos_id is not None:
        for i, tid in enumerate(render.input_ids):
            if tid == bos_id:
                bos_position = i
                break

    eos_positions = None
    if eos_id is not None:
        eos_positions = [i for i, tid in enumerate(render.input_ids) if tid == eos_id] or None

    python_tag_positions = None
    if python_tag_id is not None and python_tag_id != tokenizer.unk_token_id:
        python_tag_positions = [
            i for i, tid in enumerate(render.input_ids) if tid == python_tag_id
        ] or None

    render.special_tokens = SpecialTokenPositions(
        bos_position=bos_position,
        eos_positions=eos_positions,
        python_tag_positions=python_tag_positions,
    )


# =============================================================================
# Loss Masking
# =============================================================================

def _mask_assistant_only(render: RenderedView, mask: List[float]) -> List[float]:
    if render.alignment and render.alignment.assistant_spans:
        for span in render.alignment.assistant_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 1.0
    return mask


def _mask_completion_only(render: RenderedView, mask: List[float]) -> List[float]:
    if render.alignment and render.alignment.assistant_spans:
        span = render.alignment.assistant_spans[-1]
        for i in range(span.token_start, min(span.token_end, len(mask))):
            mask[i] = 1.0
    return mask


def _mask_full_sequence(render: RenderedView, mask: List[float]) -> List[float]:
    return [1.0] * len(mask)


def _mask_tool_calls_only(render: RenderedView, mask: List[float]) -> List[float]:
    if render.alignment and render.alignment.tool_call_spans:
        for span in render.alignment.tool_call_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 1.0
    return mask


def _mask_action_prefix_only(render: RenderedView, mask: List[float]) -> List[float]:
    if render.alignment and render.alignment.tool_call_spans:
        for span in render.alignment.tool_call_spans:
            end_idx = span.name_token_end or span.token_end
            for i in range(span.token_start, min(end_idx, len(mask))):
                mask[i] = 1.0
    return mask


def _mask_action_commitment(render: RenderedView, mask: List[float]) -> List[float]:
    if render.signals and render.signals.action_commitments:
        for commitment in render.signals.action_commitments:
            if commitment.prefix_token_indices:
                for i in commitment.prefix_token_indices:
                    if 0 <= i < len(mask):
                        mask[i] = 1.0
        return mask
    return _mask_action_prefix_only(render, mask)


def _apply_lmp_policy(render: RenderedView, policy: LMPPolicy) -> List[float]:
    mask = [0.0] * len(render.input_ids)

    if policy.strategy == "assistant_only":
        return _mask_assistant_only(render, mask)
    if policy.strategy == "completion_only":
        return _mask_completion_only(render, mask)
    if policy.strategy == "full_sequence":
        return _mask_full_sequence(render, mask)
    if policy.strategy == "tool_calls_only":
        return _mask_tool_calls_only(render, mask)
    if policy.strategy == "action_prefix_only":
        return _mask_action_prefix_only(render, mask)
    if policy.strategy == "action_commitment":
        return _mask_action_commitment(render, mask)

    return _mask_assistant_only(render, mask)


# =============================================================================
# Main Pipeline
# =============================================================================

def _resolve_policy(
    trace: Trace,
    registry: LMPRegistry,
    override: Optional[str],
) -> Tuple[str, LMPPolicy]:
    if override:
        return override, registry.get_policy(override)

    policy_id = trace.training.loss_mask_policy if trace.training else None
    if policy_id and policy_id in registry.policies:
        return policy_id, registry.get_policy(policy_id)

    default_policy = registry.default_policy
    return default_policy, registry.get_policy(default_policy)


def _apply_mwcs_weight(
    trace: Trace,
    registry: Optional[MWCSRegistry],
    mwcs_schedule: Optional[str],
    step: Optional[int],
) -> float:
    base_weight = trace.training.sample_weight if trace.training else 1.0
    if not mwcs_schedule or registry is None:
        return base_weight

    try:
        schedule = registry.get_schedule(mwcs_schedule)
    except Exception:
        logger.warning("MWCS schedule not found; using base sample weight.")
        return base_weight

    class_id = trace.training.mixture.class_id if trace.training and trace.training.mixture else None
    if not class_id:
        return base_weight

    weights = schedule.get_weights_at_step(step or 0)
    return base_weight * weights.get(class_id, 1.0)


def render_trace(
    trace: Trace,
    tokenizer,
    max_length: int,
    add_generation_prompt: bool,
    include_rendered_text: bool,
) -> RenderedView:
    chat_messages = _trace_to_chat_messages(trace)

    rendered_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    encoding = tokenizer(
        rendered_text,
        max_length=max_length,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask", [1] * len(input_ids))

    render = RenderedView(
        render_id=RenderedView.generate_id(
            trace.id,
            tokenizer.name_or_path,
            render_options=RenderOptions(
                add_generation_prompt=add_generation_prompt,
                max_length=max_length,
            ),
        ),
        trace_id=trace.id,
        tokenizer_id=tokenizer.name_or_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        rendered_text=rendered_text if include_rendered_text else None,
        render_options=RenderOptions(
            add_generation_prompt=add_generation_prompt,
            max_length=max_length,
        ),
    )

    prefix_token_ends, prefix_char_ends, prefix_texts = _compute_prefixes(
        tokenizer,
        chat_messages,
        add_generation_prompt,
    )
    render.alignment = _compute_alignment(
        trace,
        render,
        tokenizer,
        prefix_texts[-1],
        prefix_token_ends,
        prefix_char_ends,
    )

    offsets = _extract_offsets(tokenizer, prefix_texts[-1])
    message_char_starts = [0] + prefix_char_ends[:-1]
    render.signals = _build_basic_signals(
        trace,
        render,
        offsets,
        message_char_starts,
        prefix_texts[-1],
    )
    _compute_special_tokens(render, tokenizer)

    return render


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL_B: trace_v1 -> render_v1 + lossmask_v1")
    parser.add_argument("--traces", required=True, type=Path, help="Input trace_v1 JSONL file")
    parser.add_argument("--render-out", required=True, type=Path, help="Output render_v1 JSONL file")
    parser.add_argument("--lossmask-out", required=True, type=Path, help="Output lossmask_v1 JSONL file")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--add-generation-prompt", action="store_true", help="Append generation prompt")
    parser.add_argument("--include-rendered-text", action="store_true", help="Include rendered_text in output")
    parser.add_argument("--lmp-registry", type=Path, default=None, help="Path to LMP registry JSON")
    parser.add_argument("--policy-override", type=str, default=None, help="Override policy ID")
    parser.add_argument("--mwcs-registry", type=Path, default=None, help="Path to MWCS registry JSON")
    parser.add_argument("--mwcs-schedule", type=str, default=None, help="MWCS schedule ID")
    parser.add_argument("--mwcs-step", type=int, default=None, help="Training step for curriculum")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of traces processed")
    parser.add_argument(
        "--allow-skeleton", 
        action="store_true", 
        help="Allow skeleton traces (tier=B1, no assistant messages). By default, skeleton traces are skipped. "
             "When enabled, uses full_sequence LMP policy for skeleton traces."
    )
    parser.add_argument(
        "--skeleton-policy",
        type=str,
        default="full_sequence",
        help="LMP policy to use for skeleton traces when --allow-skeleton is enabled (default: full_sequence)"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lmp_registry = _load_lmp_registry_safe(args.lmp_registry)
    mwcs_registry = None
    if args.mwcs_registry and args.mwcs_registry.exists():
        try:
            mwcs_registry = load_mwcs_registry(args.mwcs_registry)
        except Exception:
            mwcs_registry = None
    else:
        try:
            mwcs_registry = load_mwcs_registry()
        except Exception:
            mwcs_registry = None

    renders: List[Dict[str, Any]] = []
    masks: List[Dict[str, Any]] = []
    
    skipped_skeleton = 0
    processed = 0

    for idx, row in enumerate(_iter_jsonl(args.traces), start=1):
        if args.limit is not None and idx > args.limit:
            break
        trace = Trace.from_dict(row)
        
        # Check for skeleton traces (tier=B1 or completeness=skeleton)
        is_skeleton = (
            getattr(trace, 'tier', None) == 'B1' or 
            getattr(trace, 'completeness', None) == 'skeleton'
        )
        
        if is_skeleton and not args.allow_skeleton:
            skipped_skeleton += 1
            logger.debug("Skipping skeleton trace %s (use --allow-skeleton to process)", trace.id)
            continue
        
        render = render_trace(
            trace,
            tokenizer,
            max_length=args.max_length,
            add_generation_prompt=args.add_generation_prompt,
            include_rendered_text=args.include_rendered_text,
        )

        # For skeleton traces with --allow-skeleton, override to skeleton_policy
        policy_override = args.policy_override
        if is_skeleton and args.allow_skeleton:
            policy_override = args.skeleton_policy
            logger.debug("Using %s policy for skeleton trace %s", args.skeleton_policy, trace.id)

        policy_id, policy = _resolve_policy(trace, lmp_registry, policy_override)
        mask_values = _apply_lmp_policy(render, policy)
        sample_weight = _apply_mwcs_weight(
            trace,
            mwcs_registry,
            args.mwcs_schedule,
            args.mwcs_step,
        )

        lossmask = LossMask.from_render(
            render,
            policy_id=policy_id,
            mask_fn=lambda _: mask_values,
            policy_version=lmp_registry.version,
            policy_params=policy.params,
            sample_weight=sample_weight,
        )

        renders.append(render.to_dict())
        masks.append(lossmask.to_dict())
        processed += 1

    _write_jsonl(args.render_out, renders)
    _write_jsonl(args.lossmask_out, masks)

    logger.info("Processed %d traces, skipped %d skeleton traces", processed, skipped_skeleton)
    logger.info("Wrote %d renders to %s", len(renders), args.render_out)
    logger.info("Wrote %d lossmasks to %s", len(masks), args.lossmask_out)
    
    if skipped_skeleton > 0:
        logger.info("Hint: Use --allow-skeleton to process skeleton traces with %s policy", args.skeleton_policy)


if __name__ == "__main__":
    main()