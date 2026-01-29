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

import yaml
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
# Llama 3.1 Format Constants
# =============================================================================

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_HEADER_START = "<|start_header_id|>"
LLAMA_HEADER_END = "<|end_header_id|>"
LLAMA_EOT = "<|eot_id|>"
LLAMA_EOM = "<|eom_id|>"  # End of message - used for tool calls expecting response
LLAMA_PYTHON_TAG = "<|python_tag|>"

# Format family constants for cross-model rendering
FORMAT_FAMILY_LLAMA = "llama_python_tag"
FORMAT_FAMILY_OPENAI = "openai_json"
FORMAT_FAMILY_ANTHROPIC = "anthropic_xml"
FORMAT_FAMILY_GENERIC = "generic_json"


def _detect_tokenizer_format_family(tokenizer) -> str:
    """Detect the format family expected by a tokenizer."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    name_lower = name.lower()

    if "llama" in name_lower:
        return FORMAT_FAMILY_LLAMA
    if "gpt" in name_lower or "openai" in name_lower:
        return FORMAT_FAMILY_OPENAI
    if "claude" in name_lower or "anthropic" in name_lower:
        return FORMAT_FAMILY_ANTHROPIC

    # Check for Llama-specific tokens
    try:
        python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
        if python_tag_id != tokenizer.unk_token_id:
            return FORMAT_FAMILY_LLAMA
    except Exception:
        pass

    return FORMAT_FAMILY_GENERIC


def _should_preserve_formatting(
    trace: Trace,
    target_format: str,
    preserve_formatting: bool,
) -> bool:
    """
    Determine if we should use raw_content/arguments_json for exact replay.

    Returns True if:
    - preserve_formatting is True AND
    - The trace's source format matches the target format
    """
    if not preserve_formatting:
        return False

    source_format = None
    if trace.source:
        source_format = trace.source.format_family
    if not source_format and trace.signal_hints:
        source_format = trace.signal_hints.raw_format

    if not source_format:
        return False

    return source_format == target_format


def _is_llama_tokenizer(tokenizer) -> bool:
    """Check if tokenizer is a Llama-style tokenizer that needs special tool call handling."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    name_lower = name.lower()
    # Check for Llama model names
    if "llama" in name_lower:
        return True
    # Check for python_tag token in vocabulary
    try:
        python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
        if python_tag_id != tokenizer.unk_token_id:
            return True
    except Exception:
        pass
    return False


def _format_llama31_header(role: str) -> str:
    """Format a role header in Llama 3.1 format."""
    return f"{LLAMA_HEADER_START}{role}{LLAMA_HEADER_END}\n\n"


def _format_tool_call_json(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Format a tool call as Llama 3.1 JSON format."""
    return f'{LLAMA_PYTHON_TAG}{{"name": "{tool_name}", "parameters": {json.dumps(arguments, ensure_ascii=False)}}}'


def _format_tool_call_with_raw(
    tc,
    target_format: str,
    preserve_raw: bool,
) -> str:
    """
    Format a tool call, optionally using raw_content for exact replay.

    Args:
        tc: ToolCall object
        target_format: Target format family (e.g., FORMAT_FAMILY_LLAMA)
        preserve_raw: If True and raw_content exists, use it

    Returns:
        Formatted tool call string
    """
    # If we have raw_content and want to preserve it, use it directly
    if preserve_raw and tc.function.raw_content:
        return tc.function.raw_content

    # Otherwise, re-render using the target format
    args = tc.function.arguments or {}

    # If we have arguments_json, try to parse it for cleaner output
    if tc.function.arguments_json:
        try:
            args = json.loads(tc.function.arguments_json)
        except json.JSONDecodeError:
            pass

    if target_format == FORMAT_FAMILY_LLAMA:
        return _format_tool_call_json(tc.function.name, args)
    elif target_format == FORMAT_FAMILY_OPENAI:
        # OpenAI-style JSON format
        return json.dumps({
            "name": tc.function.name,
            "arguments": json.dumps(args, ensure_ascii=False),
        }, ensure_ascii=False)
    else:
        # Generic JSON format
        return json.dumps({
            "name": tc.function.name,
            "parameters": args,
        }, ensure_ascii=False)


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
        "cb_full_sequence": LMPPolicy(
            name="CB Full Sequence",
            strategy="cb_full_sequence",
            description="Loss on all non-system tokens (attack-agnostic).",
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
    chat_template: Optional[str] = None,
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
            chat_template=chat_template,
        )
        prefix_texts.append(rendered_text)

        input_ids = tokenizer.apply_chat_template(
            prefix,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )
        token_ends.append(len(input_ids))
        char_ends.append(len(rendered_text))

    return token_ends, char_ends, prefix_texts


def _compute_prefixes_llama31_manual(
    tokenizer,
    trace: Trace,
    add_generation_prompt: bool,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Compute prefix boundaries using manual Llama 3.1 rendering.
    
    This ensures prefix computations match the manual rendering with proper
    tool call format.
    """
    token_ends: List[int] = []
    char_ends: List[int] = []
    prefix_texts: List[str] = []
    
    for i in range(len(trace.messages)):
        # Create a partial trace with messages up to index i
        partial_messages = trace.messages[: i + 1]
        
        # Build partial rendered text manually
        parts = [LLAMA_BOS]
        for msg in partial_messages:
            parts.append(_format_llama31_header(msg.role))
            
            if msg.content:
                parts.append(msg.content)
            
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments or {}
                    if tc.function.arguments_json:
                        try:
                            args = json.loads(tc.function.arguments_json)
                        except json.JSONDecodeError:
                            pass
                    tool_call_str = _format_tool_call_json(tc.function.name, args)
                    if msg.content:
                        parts.append("\n\n")
                    parts.append(tool_call_str)
                parts.append(LLAMA_EOM)
            elif msg.role == "tool":
                parts.append(LLAMA_EOT)
            else:
                parts.append(LLAMA_EOT)
        
        # Only add generation prompt for the final prefix if requested
        if add_generation_prompt and i == len(trace.messages) - 1:
            parts.append(_format_llama31_header("assistant"))
        
        rendered_text = "".join(parts)
        prefix_texts.append(rendered_text)
        
        # Tokenize to get token count
        input_ids = tokenizer(
            rendered_text,
            add_special_tokens=False,
        )["input_ids"]
        
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


def _mask_cb_full_sequence(render: RenderedView, mask: List[float]) -> List[float]:
    """Loss on all non-system tokens. Attack-agnostic CB policy."""
    if render.alignment and render.alignment.message_spans:
        for span in render.alignment.message_spans:
            if span.role != "system":
                for i in range(span.token_start, min(span.token_end, len(mask))):
                    mask[i] = 1.0
        return mask
    return _mask_full_sequence(render, mask)


def _mask_custom(render: RenderedView, mask: List[float], params: Optional[Dict[str, Any]]) -> List[float]:
    params = params or {}
    applied = False

    roles = params.get("roles")
    if roles and render.alignment and render.alignment.message_spans:
        role_set = set(roles)
        weight = float(params.get("weight", 1.0))
        for span in render.alignment.message_spans:
            if span.role in role_set:
                for i in range(span.token_start, min(span.token_end, len(mask))):
                    mask[i] = weight
        applied = True

    token_ranges = params.get("token_ranges")
    if token_ranges:
        for tr in token_ranges:
            if isinstance(tr, dict):
                start = int(tr.get("start", 0))
                end = int(tr.get("end", 0))
                weight = float(tr.get("weight", 1.0))
            else:
                try:
                    start, end = tr
                    weight = float(params.get("weight", 1.0))
                except Exception:
                    continue
            for i in range(max(0, start), min(end, len(mask))):
                mask[i] = weight
        applied = True

    if not applied:
        return _mask_assistant_only(render, mask)
    return mask


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
    if policy.strategy == "cb_full_sequence":
        return _mask_cb_full_sequence(render, mask)
    if policy.strategy == "custom":
        return _mask_custom(render, mask, policy.params)

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


def _get_phase_value(phase: Any, key: str) -> Optional[Any]:
    if hasattr(phase, key):
        return getattr(phase, key)
    if isinstance(phase, dict):
        return phase.get(key)
    return None


def _select_phase_for_step(phases: List[Any], step: int) -> Tuple[Optional[Any], Optional[Any]]:
    if not phases:
        return None, None

    current_phase = None
    next_phase = None

    for phase in phases:
        start_step = _get_phase_value(phase, "start_step")
        end_step = _get_phase_value(phase, "end_step")

        if start_step is None or end_step is None:
            continue

        if start_step <= step < end_step:
            current_phase = phase
        elif start_step > step and next_phase is None:
            next_phase = phase

    if current_phase is None:
        first = phases[0]
        last = phases[-1]
        first_start = _get_phase_value(first, "start_step")
        last_start = _get_phase_value(last, "start_step")
        if first_start is not None and step < first_start:
            current_phase = first
        elif last_start is not None:
            current_phase = last

    return current_phase, next_phase


def _interpolate_weights(
    current_weights: Dict[str, float],
    next_weights: Dict[str, float],
    progress: float,
) -> Dict[str, float]:
    weights = {}
    all_classes = set(current_weights.keys()) | set(next_weights.keys())
    for cls in all_classes:
        w1 = current_weights.get(cls, 0.0)
        w2 = next_weights.get(cls, w1)
        weights[cls] = w1 + (w2 - w1) * progress
    return weights


def _load_mwcs_schedule_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning("MWCS schedule YAML not found at %s; using base weights.", path)
        return None
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data or None
    except Exception:
        logger.warning("Failed to load MWCS schedule YAML at %s; using base weights.", path)
        return None


def _resolve_yaml_schedule(
    schedule_data: Dict[str, Any],
    step: int,
) -> Tuple[Dict[str, float], Optional[Dict[str, str]]]:
    class_weights = schedule_data.get("class_weights", {})
    curriculum = schedule_data.get("curriculum")
    if curriculum is None and schedule_data.get("phases"):
        curriculum = schedule_data

    if not curriculum or not curriculum.get("phases"):
        return class_weights, None

    phases = curriculum.get("phases", [])
    interpolation = curriculum.get("interpolation", "none")
    current_phase, next_phase = _select_phase_for_step(phases, step)

    if current_phase is None:
        return class_weights, None

    current_weights = current_phase.get("class_weights") or class_weights
    if interpolation == "linear" and next_phase is not None:
        next_weights = next_phase.get("class_weights") or current_weights
        start_step = current_phase.get("start_step", 0)
        end_step = current_phase.get("end_step", start_step + 1)
        denom = max(1, end_step - start_step)
        progress = max(0.0, min(1.0, (step - start_step) / denom))
        weights = _interpolate_weights(current_weights, next_weights, progress)
    else:
        weights = current_weights

    lmp_overrides = current_phase.get("lmp_overrides") or None
    return weights, lmp_overrides


def _apply_mwcs_weight(
    trace: Trace,
    registry: Optional[MWCSRegistry],
    mwcs_schedule: Optional[str],
    step: Optional[int],
) -> Tuple[float, Optional[str]]:
    base_weight = trace.training.sample_weight if trace.training else 1.0
    class_id = trace.training.mixture.class_id if trace.training and trace.training.mixture else None
    if not class_id:
        return base_weight, None

    schedule_step = step or 0

    if mwcs_schedule and registry is not None:
        try:
            schedule = registry.get_schedule(mwcs_schedule)
        except Exception:
            logger.warning("MWCS schedule not found; using base sample weight.")
            return base_weight, None

        weights = schedule.get_weights_at_step(schedule_step)
        lmp_override = None
        if schedule.curriculum and schedule.curriculum.phases:
            current_phase, _ = _select_phase_for_step(schedule.curriculum.phases, schedule_step)
            if current_phase is not None:
                lmp_override = current_phase.lmp_overrides.get(class_id) if current_phase.lmp_overrides else None

        return base_weight * weights.get(class_id, 1.0), lmp_override

    return base_weight, None


def _apply_mwcs_weight_with_yaml(
    trace: Trace,
    schedule_path: Optional[Path],
    step: Optional[int],
) -> Tuple[float, Optional[str]]:
    base_weight = trace.training.sample_weight if trace.training else 1.0
    class_id = trace.training.mixture.class_id if trace.training and trace.training.mixture else None
    if not schedule_path or not class_id:
        return base_weight, None

    schedule_data = _load_mwcs_schedule_yaml(schedule_path)
    if not schedule_data:
        return base_weight, None

    weights, lmp_overrides = _resolve_yaml_schedule(schedule_data, step or 0)
    lmp_override = lmp_overrides.get(class_id) if lmp_overrides else None
    return base_weight * weights.get(class_id, 1.0), lmp_override

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


def _render_trace_llama31_manual(
    trace: Trace,
    add_generation_prompt: bool,
    preserve_formatting: bool = False,
) -> str:
    """
    Manually render a trace in Llama 3.1 format with proper tool call handling.

    Formats tool calls as:
        <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>

    Args:
        trace: The trace to render
        add_generation_prompt: Whether to add generation prompt
        preserve_formatting: If True and raw_content exists, use it for exact replay

    This is required because HuggingFace's apply_chat_template may not properly
    format tool calls for Llama 3.1 models.
    """
    parts = [LLAMA_BOS]

    for msg in trace.messages:
        parts.append(_format_llama31_header(msg.role))

        # Add content (may be empty for tool-call-only messages)
        if msg.content:
            parts.append(msg.content)

        # Handle tool calls in assistant messages
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                # Format the tool call, optionally using raw content
                tool_call_str = _format_tool_call_with_raw(
                    tc,
                    target_format=FORMAT_FAMILY_LLAMA,
                    preserve_raw=preserve_formatting,
                )

                # Add newline separator if there's content before tool call
                if msg.content:
                    parts.append("\n\n")
                parts.append(tool_call_str)

            # Use <|eom_id|> for tool calls (expecting tool response)
            parts.append(LLAMA_EOM)
        elif msg.role == "tool":
            # Tool responses use <|eot_id|>
            parts.append(LLAMA_EOT)
        else:
            # Regular messages use <|eot_id|>
            parts.append(LLAMA_EOT)

    # Add generation prompt if requested
    if add_generation_prompt:
        parts.append(_format_llama31_header("assistant"))
    
    return "".join(parts)


def _trace_has_tool_calls(trace: Trace) -> bool:
    """Check if any message in the trace has tool calls."""
    for msg in trace.messages:
        if msg.tool_calls:
            return True
    return False


def render_trace(
    trace: Trace,
    tokenizer,
    max_length: int,
    add_generation_prompt: bool,
    include_rendered_text: bool,
    chat_template: Optional[str] = None,
    force_llama_format: bool = False,
    preserve_formatting: bool = False,
) -> RenderedView:
    """
    Render a trace for a specific tokenizer.

    Args:
        trace: The trace to render
        tokenizer: Target tokenizer
        max_length: Max sequence length
        add_generation_prompt: Whether to add generation prompt
        include_rendered_text: Whether to include rendered text in output
        chat_template: Optional custom chat template
        force_llama_format: Force Llama 3.1 format
        preserve_formatting: If True, use raw_content for exact replay when
                           the source format matches the target format

    Returns:
        RenderedView with tokenized trace
    """
    chat_messages = _trace_to_chat_messages(trace)

    # Detect target format family
    target_format = _detect_tokenizer_format_family(tokenizer)

    # Determine if we should preserve raw formatting
    should_preserve = _should_preserve_formatting(trace, target_format, preserve_formatting)

    # Use manual Llama 3.1 rendering for tool calls when using a Llama tokenizer
    # This ensures proper <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|> format
    use_manual_llama_rendering = (
        (force_llama_format or _is_llama_tokenizer(tokenizer))
        and _trace_has_tool_calls(trace)
        and chat_template is None  # Only use manual if no custom template provided
    )

    if use_manual_llama_rendering:
        rendered_text = _render_trace_llama31_manual(
            trace, add_generation_prompt, preserve_formatting=should_preserve
        )
    else:
        rendered_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
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

    # Compute prefix boundaries using matching rendering method
    if use_manual_llama_rendering:
        prefix_token_ends, prefix_char_ends, prefix_texts = _compute_prefixes_llama31_manual(
            tokenizer,
            trace,
            add_generation_prompt,
        )
    else:
        prefix_token_ends, prefix_char_ends, prefix_texts = _compute_prefixes(
            tokenizer,
            chat_messages,
            add_generation_prompt,
            chat_template=chat_template,
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
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template string or path to a template file (for tokenizers without chat_template)",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--add-generation-prompt", action="store_true", help="Append generation prompt")
    parser.add_argument("--include-rendered-text", action="store_true", help="Include rendered_text in output")
    parser.add_argument("--lmp-registry", type=Path, default=None, help="Path to LMP registry JSON")
    parser.add_argument("--policy-override", type=str, default=None, help="Override policy ID")
    parser.add_argument("--mwcs-registry", type=Path, default=None, help="Path to MWCS registry JSON")
    parser.add_argument("--mwcs-schedule", type=str, default=None, help="MWCS schedule ID")
    parser.add_argument("--mwcs-step", type=int, default=None, help="Training step for curriculum")
    parser.add_argument("--lmp-schedule", type=Path, default=None, help="Path to MWCS schedule YAML for curriculum")
    parser.add_argument("--step", type=int, default=None, help="Current training step for curriculum weights")
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
    parser.add_argument(
        "--force-llama-format",
        action="store_true",
        help="Force Llama 3.1 tool call format (<|python_tag|>{...}<|eom_id|>) even for non-Llama tokenizers"
    )
    parser.add_argument(
        "--preserve-formatting",
        action="store_true",
        help="Preserve raw tool call formatting when source format matches target. "
             "Use for same-model replay (high fidelity). Without this flag (default), "
             "tool calls are re-rendered for the target model (cross-model compatible)."
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    chat_template = args.chat_template
    if chat_template and Path(chat_template).exists():
        chat_template = Path(chat_template).read_text()
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
            chat_template=chat_template,
            force_llama_format=args.force_llama_format,
            preserve_formatting=args.preserve_formatting,
        )

        # For skeleton traces with --allow-skeleton, override to skeleton_policy
        policy_override = args.policy_override
        if is_skeleton and args.allow_skeleton:
            policy_override = args.skeleton_policy
            logger.debug("Using %s policy for skeleton trace %s", args.skeleton_policy, trace.id)

        schedule_step = args.step if args.step is not None else args.mwcs_step
        sample_weight, lmp_override = _apply_mwcs_weight_with_yaml(
            trace,
            args.lmp_schedule,
            schedule_step,
        )
        if args.lmp_schedule is None:
            sample_weight, lmp_override = _apply_mwcs_weight(
                trace,
                mwcs_registry,
                args.mwcs_schedule,
                schedule_step,
            )

        if lmp_override and policy_override is None:
            policy_override = lmp_override

        policy_id, policy = _resolve_policy(trace, lmp_registry, policy_override)
        mask_values = _apply_lmp_policy(render, policy)

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