#!/usr/bin/env python3
"""
Signal Extraction Utilities

Extract shock/surprisal signals and action commitment points from traces.

Usage:
    # Compute signals for a trace file
    python -m src.schemas.tools.extract_signals \
        --traces data/traces/cb_traces.jsonl \
        --output data/renders/cb_renders.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Extract signals with specific detector
    python -m src.schemas.tools.extract_signals \
        --traces data/traces/cb_traces.jsonl \
        --shock-detector contiguous_threshold \
        --shock-threshold 6.0 \
        --commitment-detector logprob_margin \
        --margin-threshold 2.0
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.schemas import (
    Trace,
    RenderedView, RenderOptions, RenderAlignment,
    AssistantSpan, ToolCallSpan, MessageSpan,
    RenderSignals, InjectionSpan, ActionCommitment, ShockScore, DetectorMetadata,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Surprisal Computation
# =============================================================================

def compute_token_surprisals(
    input_ids: List[int],
    model,
    tokenizer,
    batch_size: int = 1,
) -> List[float]:
    """
    Compute per-token surprisal (negative log probability) using a language model.

    Args:
        input_ids: Token IDs
        model: HuggingFace model with forward() method
        tokenizer: HuggingFace tokenizer

    Returns:
        List of surprisal values (one per token). First token has surprisal 0.
    """
    import torch

    surprisals = [0.0]  # First token has no surprisal

    # Prepare input
    input_tensor = torch.tensor([input_ids]).to(model.device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute surprisal for each token (based on previous context)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    for i in range(1, len(input_ids)):
        # Surprisal of token i given context 0:i-1
        token_id = input_ids[i]
        token_log_prob = log_probs[i - 1, token_id].item()
        surprisal = -token_log_prob  # Negative log prob
        surprisals.append(surprisal)

    return surprisals


def compute_surprisals_batch(
    input_ids_list: List[List[int]],
    model,
    tokenizer,
) -> List[List[float]]:
    """Batch computation of surprisals for multiple sequences."""
    import torch

    results = []
    for input_ids in input_ids_list:
        surprisals = compute_token_surprisals(input_ids, model, tokenizer)
        results.append(surprisals)
    return results


# =============================================================================
# Shock Detection (Rule 1: WHERE)
# =============================================================================

class ShockDetector:
    """
    Detect injection spans via token-level surprisal analysis.

    Methods:
    - threshold: Tokens above fixed surprisal threshold
    - percentile: Tokens in top-p percentile of surprisal
    - contiguous_threshold: Shortest contiguous span above threshold
    - adaptive: Adaptive threshold based on local context
    """

    def __init__(
        self,
        method: str = "contiguous_threshold",
        threshold: float = 6.0,
        percentile: float = 95.0,
        min_span_length: int = 3,
        max_gap: int = 2,
    ):
        self.method = method
        self.threshold = threshold
        self.percentile = percentile
        self.min_span_length = min_span_length
        self.max_gap = max_gap

    def detect(
        self,
        surprisals: List[float],
        hint_start: Optional[int] = None,
        hint_end: Optional[int] = None,
    ) -> List[InjectionSpan]:
        """
        Detect injection spans from surprisal values.

        Args:
            surprisals: Per-token surprisal values
            hint_start: Optional hint for expected start (from trace.signal_hints)
            hint_end: Optional hint for expected end

        Returns:
            List of detected InjectionSpan objects
        """
        if self.method == "threshold":
            return self._detect_threshold(surprisals)
        elif self.method == "percentile":
            return self._detect_percentile(surprisals)
        elif self.method == "contiguous_threshold":
            return self._detect_contiguous(surprisals, hint_start, hint_end)
        elif self.method == "adaptive":
            return self._detect_adaptive(surprisals)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_threshold(self, surprisals: List[float]) -> List[InjectionSpan]:
        """Simple threshold-based detection."""
        high_surp_indices = [i for i, s in enumerate(surprisals) if s > self.threshold]
        if not high_surp_indices:
            return []

        spans = self._merge_indices(high_surp_indices)
        return [self._create_span(surprisals, start, end, "threshold")
                for start, end in spans]

    def _detect_percentile(self, surprisals: List[float]) -> List[InjectionSpan]:
        """Percentile-based detection."""
        threshold = np.percentile(surprisals, self.percentile)
        high_surp_indices = [i for i, s in enumerate(surprisals) if s > threshold]
        if not high_surp_indices:
            return []

        spans = self._merge_indices(high_surp_indices)
        return [self._create_span(surprisals, start, end, "percentile")
                for start, end in spans]

    def _detect_contiguous(
        self,
        surprisals: List[float],
        hint_start: Optional[int] = None,
        hint_end: Optional[int] = None,
    ) -> List[InjectionSpan]:
        """
        Find minimal contiguous span with high surprisal.
        This is the core Rule 1 implementation.
        """
        # Find all tokens above threshold
        above_threshold = [i for i, s in enumerate(surprisals) if s > self.threshold]
        if not above_threshold:
            return []

        # If we have hints, focus search around them
        if hint_start is not None and hint_end is not None:
            # Filter to tokens near the hint
            margin = 20  # tokens
            above_threshold = [i for i in above_threshold
                              if hint_start - margin <= i <= hint_end + margin]

        if not above_threshold:
            return []

        # Find contiguous spans (allow small gaps)
        spans = self._merge_indices(above_threshold)

        # Filter by minimum length
        spans = [(s, e) for s, e in spans if e - s >= self.min_span_length]

        if not spans:
            return []

        # Return the span with highest mean surprisal
        best_span = max(spans, key=lambda se: np.mean(surprisals[se[0]:se[1]]))
        return [self._create_span(surprisals, best_span[0], best_span[1], "contiguous_threshold")]

    def _detect_adaptive(self, surprisals: List[float]) -> List[InjectionSpan]:
        """Adaptive detection using local context."""
        window_size = 20
        spans = []

        for i in range(len(surprisals)):
            # Compute local statistics
            start_ctx = max(0, i - window_size)
            end_ctx = min(len(surprisals), i + window_size)
            local_surps = surprisals[start_ctx:end_ctx]

            local_mean = np.mean(local_surps)
            local_std = np.std(local_surps)

            # Check if current token is an outlier
            if surprisals[i] > local_mean + 2 * local_std:
                spans.append(i)

        if not spans:
            return []

        merged = self._merge_indices(spans)
        merged = [(s, e) for s, e in merged if e - s >= self.min_span_length]

        return [self._create_span(surprisals, start, end, "adaptive")
                for start, end in merged]

    def _merge_indices(self, indices: List[int]) -> List[Tuple[int, int]]:
        """Merge nearby indices into spans."""
        if not indices:
            return []

        indices = sorted(indices)
        spans = []
        start = indices[0]
        end = indices[0] + 1

        for i in indices[1:]:
            if i <= end + self.max_gap:
                end = i + 1
            else:
                spans.append((start, end))
                start = i
                end = i + 1

        spans.append((start, end))
        return spans

    def _create_span(
        self,
        surprisals: List[float],
        start: int,
        end: int,
        method: str,
    ) -> InjectionSpan:
        """Create an InjectionSpan with shock score."""
        span_surprisals = surprisals[start:end]

        # Compute deltas
        deltas = [abs(span_surprisals[i] - span_surprisals[i-1])
                  for i in range(1, len(span_surprisals))]
        max_delta = max(deltas) if deltas else 0.0

        # Compute percentile rank
        all_max = max(surprisals) if surprisals else 1.0
        percentile_rank = max(span_surprisals) / all_max if all_max > 0 else 0.0

        return InjectionSpan(
            token_start=start,
            token_end=end,
            shock_score=ShockScore(
                max_surprisal=max(span_surprisals),
                mean_surprisal=float(np.mean(span_surprisals)),
                max_delta=max_delta,
                span_length=end - start,
                percentile_rank=percentile_rank,
            ),
            detection_method=method,
            confidence=min(1.0, max(span_surprisals) / self.threshold) if self.threshold > 0 else 0.5,
        )


def detect_injection_spans(
    surprisals: List[float],
    method: str = "contiguous_threshold",
    threshold: float = 6.0,
    hint_start: Optional[int] = None,
    hint_end: Optional[int] = None,
) -> List[InjectionSpan]:
    """Convenience function for shock detection."""
    detector = ShockDetector(method=method, threshold=threshold)
    return detector.detect(surprisals, hint_start, hint_end)


# =============================================================================
# Action Commitment Detection (Rule 2: WHAT)
# =============================================================================

class CommitmentDetector:
    """
    Detect action commitment points in assistant output.

    The commitment point is where we can GUARANTEE the next action is deterministic.

    Methods:
    - tool_name_selected: Commitment when tool name is fully specified
    - json_frame_parse_valid: When JSON structure becomes parseable
    - logprob_margin: When logP(bad_tool) - logP(expected_tool) > threshold
    """

    def __init__(
        self,
        method: str = "tool_name_selected",
        margin_threshold: float = 2.0,
    ):
        self.method = method
        self.margin_threshold = margin_threshold

    def detect(
        self,
        render: RenderedView,
        trace: Trace,
        model=None,
        tokenizer=None,
    ) -> List[ActionCommitment]:
        """
        Detect action commitment points.

        Args:
            render: Rendered view with tokenization
            trace: Source trace with signal hints
            model: Model for logprob computation (optional)
            tokenizer: Tokenizer for decoding (optional)

        Returns:
            List of ActionCommitment objects
        """
        commitments = []

        # Get assistant spans and tool call spans
        if not render.alignment:
            return commitments

        tool_call_spans = render.alignment.tool_call_spans or []

        for tc_span in tool_call_spans:
            commitment = self._detect_for_tool_call(
                render, trace, tc_span, model, tokenizer
            )
            if commitment:
                commitments.append(commitment)

        return commitments

    def _detect_for_tool_call(
        self,
        render: RenderedView,
        trace: Trace,
        tc_span: ToolCallSpan,
        model,
        tokenizer,
    ) -> Optional[ActionCommitment]:
        """Detect commitment for a single tool call."""
        # Get expected vs observed tool
        expected_tool = None
        observed_tool = tc_span.tool_name

        if trace.signal_hints:
            expected_tool = trace.signal_hints.expected_tool_name
        elif trace.tool_attack:
            expected_tool = trace.tool_attack.expected_tool

        if self.method == "tool_name_selected":
            return self._detect_tool_name_selected(
                render, tc_span, expected_tool, observed_tool, tokenizer
            )
        elif self.method == "json_frame_parse_valid":
            return self._detect_json_frame(
                render, tc_span, expected_tool, observed_tool, tokenizer
            )
        elif self.method == "logprob_margin":
            return self._detect_logprob_margin(
                render, tc_span, expected_tool, observed_tool, model, tokenizer
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_tool_name_selected(
        self,
        render: RenderedView,
        tc_span: ToolCallSpan,
        expected_tool: Optional[str],
        observed_tool: str,
        tokenizer,
    ) -> ActionCommitment:
        """
        Commitment at the point where tool name is fully specified.

        For Llama 3.1 format: <|python_tag|>{"name": "search_web"
        The commitment point is after the closing quote of the tool name.
        """
        # Use name_token_end if available
        if tc_span.name_token_end is not None:
            commitment_idx = tc_span.name_token_end
        else:
            # Estimate: commitment is early in the tool call span
            # Typically: <|python_tag|> + {"name": " + tool_name + "
            # This is roughly 5-10 tokens after span start
            commitment_idx = min(tc_span.token_start + 8, tc_span.token_end - 1)

        # Get prefix tokens
        prefix_tokens = list(range(tc_span.token_start, commitment_idx + 1))

        # Decode prefix for guarantee_prefix field
        guarantee_prefix = None
        if tokenizer and render.input_ids:
            prefix_ids = render.input_ids[tc_span.token_start:commitment_idx + 1]
            guarantee_prefix = tokenizer.decode(prefix_ids)

        return ActionCommitment(
            commitment_token_idx=commitment_idx,
            commit_type="tool_name_selected",
            assistant_message_index=tc_span.message_index,
            committed_tool=observed_tool,
            expected_tool=expected_tool,
            guarantee_prefix=guarantee_prefix,
            prefix_token_indices=prefix_tokens,
        )

    def _detect_json_frame(
        self,
        render: RenderedView,
        tc_span: ToolCallSpan,
        expected_tool: Optional[str],
        observed_tool: str,
        tokenizer,
    ) -> ActionCommitment:
        """
        Commitment when JSON structure becomes parseable.

        Parse incrementally until we get a valid JSON structure with tool name.
        """
        if not tokenizer or not render.input_ids:
            # Fall back to tool_name_selected
            return self._detect_tool_name_selected(
                render, tc_span, expected_tool, observed_tool, tokenizer
            )

        commitment_idx = tc_span.token_start

        for i in range(tc_span.token_start, tc_span.token_end):
            partial_ids = render.input_ids[tc_span.token_start:i + 1]
            partial_text = tokenizer.decode(partial_ids)

            # Try to parse as JSON (may need to add closing braces)
            test_text = partial_text
            for suffix in ['"}', '"}]', '"}]}']:
                try:
                    data = json.loads(test_text + suffix)
                    if "name" in data and data["name"]:
                        commitment_idx = i
                        break
                except json.JSONDecodeError:
                    continue
            else:
                continue
            break

        prefix_tokens = list(range(tc_span.token_start, commitment_idx + 1))
        guarantee_prefix = tokenizer.decode(render.input_ids[tc_span.token_start:commitment_idx + 1])

        return ActionCommitment(
            commitment_token_idx=commitment_idx,
            commit_type="json_frame_parse_valid",
            assistant_message_index=tc_span.message_index,
            committed_tool=observed_tool,
            expected_tool=expected_tool,
            guarantee_prefix=guarantee_prefix,
            prefix_token_indices=prefix_tokens,
        )

    def _detect_logprob_margin(
        self,
        render: RenderedView,
        tc_span: ToolCallSpan,
        expected_tool: Optional[str],
        observed_tool: str,
        model,
        tokenizer,
    ) -> Optional[ActionCommitment]:
        """
        Commitment when logP(bad_tool) - logP(expected_tool) exceeds threshold.

        This requires model forward pass to compute token probabilities.
        """
        if model is None or tokenizer is None or expected_tool is None:
            # Fall back to tool_name_selected
            return self._detect_tool_name_selected(
                render, tc_span, expected_tool, observed_tool, tokenizer
            )

        import torch

        # Get token IDs for tool names
        bad_tool_ids = tokenizer.encode(observed_tool, add_special_tokens=False)
        expected_tool_ids = tokenizer.encode(expected_tool, add_special_tokens=False)

        # Find where tool name starts in the span
        # Look for the position after "name": "
        input_ids = render.input_ids
        input_tensor = torch.tensor([input_ids]).to(model.device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        log_probs = torch.log_softmax(logits, dim=-1)

        commitment_idx = tc_span.token_start
        margin_at_commitment = 0.0

        # Scan through the tool call span
        for i in range(tc_span.token_start, min(tc_span.token_start + 20, tc_span.token_end)):
            # Compare probability of next token being from bad_tool vs expected_tool
            if i + 1 < len(input_ids):
                # Get log probs for first tokens of each tool name
                if bad_tool_ids:
                    bad_logp = log_probs[i, bad_tool_ids[0]].item()
                else:
                    bad_logp = -100

                if expected_tool_ids:
                    expected_logp = log_probs[i, expected_tool_ids[0]].item()
                else:
                    expected_logp = -100

                margin = bad_logp - expected_logp

                if margin > self.margin_threshold:
                    commitment_idx = i
                    margin_at_commitment = margin
                    break

        prefix_tokens = list(range(tc_span.token_start, commitment_idx + 1))
        guarantee_prefix = tokenizer.decode(input_ids[tc_span.token_start:commitment_idx + 1])

        return ActionCommitment(
            commitment_token_idx=commitment_idx,
            commit_type="logprob_margin",
            assistant_message_index=tc_span.message_index,
            committed_tool=observed_tool,
            expected_tool=expected_tool,
            logprob_margin=margin_at_commitment,
            guarantee_prefix=guarantee_prefix,
            prefix_token_indices=prefix_tokens,
        )


def detect_action_commitment(
    render: RenderedView,
    trace: Trace,
    method: str = "tool_name_selected",
    model=None,
    tokenizer=None,
) -> List[ActionCommitment]:
    """Convenience function for commitment detection."""
    detector = CommitmentDetector(method=method)
    return detector.detect(render, trace, model, tokenizer)


# =============================================================================
# Full Signal Extraction Pipeline
# =============================================================================

def extract_signals(
    trace: Trace,
    tokenizer,
    model=None,
    shock_method: str = "contiguous_threshold",
    shock_threshold: float = 6.0,
    commitment_method: str = "tool_name_selected",
    max_length: int = 2048,
) -> Tuple[RenderedView, Optional[List[float]]]:
    """
    Full signal extraction pipeline: render trace, compute surprisals, detect signals.

    Args:
        trace: Source trace
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (optional, needed for surprisal computation)
        shock_method: Shock detection method
        shock_threshold: Surprisal threshold
        commitment_method: Commitment detection method
        max_length: Max sequence length

    Returns:
        (RenderedView with signals, surprisals list)
    """
    # Render the trace
    chat_messages = [{"role": m.role, "content": m.content} for m in trace.messages]
    rendered_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoding = tokenizer(
        rendered_text,
        max_length=max_length,
        truncation=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask", [1] * len(input_ids))

    # Create basic render
    render = RenderedView(
        render_id=RenderedView.generate_id(trace.id, tokenizer.name_or_path),
        trace_id=trace.id,
        tokenizer_id=tokenizer.name_or_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        rendered_text=rendered_text,
        render_options=RenderOptions(add_generation_prompt=False),
    )

    # Compute alignment (simplified - for production use proper alignment computation)
    # This is a placeholder - real implementation would track char->token mapping
    alignment = _compute_alignment(trace, render, tokenizer)
    render.alignment = alignment

    # Compute surprisals if model is available
    surprisals = None
    injection_spans = []

    if model is not None:
        surprisals = compute_token_surprisals(input_ids, model, tokenizer)

        # Get hint from trace if available
        hint_start, hint_end = None, None
        if trace.signal_hints and trace.signal_hints.injection_char_span:
            # Convert char span to approximate token span
            # This is a rough approximation
            span = trace.signal_hints.injection_char_span
            hint_start = int(span.char_start / 4)  # Rough char-to-token ratio
            hint_end = int(span.char_end / 4)

        # Detect injection spans
        shock_detector = ShockDetector(method=shock_method, threshold=shock_threshold)
        injection_spans = shock_detector.detect(surprisals, hint_start, hint_end)

    # Detect action commitments
    commitment_detector = CommitmentDetector(method=commitment_method)
    action_commitments = commitment_detector.detect(render, trace, model, tokenizer)

    # Create signals
    render.signals = RenderSignals(
        injection_spans=injection_spans if injection_spans else None,
        action_commitments=action_commitments if action_commitments else None,
        token_surprisals=surprisals,
        detector_metadata=DetectorMetadata(
            shock_detector_id=shock_method,
            shock_detector_params={"threshold": shock_threshold},
            commitment_detector_id=commitment_method,
            reference_model=model.config._name_or_path if model else None,
        ),
    )

    return render, surprisals


def _compute_alignment(trace: Trace, render: RenderedView, tokenizer) -> RenderAlignment:
    """
    Compute alignment between trace messages and rendered tokens.

    This is a simplified implementation. For production, use proper
    char-to-token offset tracking during tokenization.
    """
    message_spans = []
    assistant_spans = []
    tool_call_spans = []

    text = render.rendered_text
    if not text:
        return RenderAlignment()

    # Find message boundaries by searching for role markers
    # This is approximate for Llama 3.1 format
    role_markers = {
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>",
    }

    for i, msg in enumerate(trace.messages):
        marker = role_markers.get(msg.role)
        if not marker:
            continue

        marker_pos = text.find(marker)
        if marker_pos < 0:
            continue

        # Find the end of this message (next marker or EOS)
        next_marker_pos = len(text)
        for other_marker in role_markers.values():
            pos = text.find(other_marker, marker_pos + len(marker))
            if pos > 0 and pos < next_marker_pos:
                next_marker_pos = pos

        # Convert char positions to approximate token positions
        # This is rough - real implementation would use tokenizer offset mapping
        token_start = len(tokenizer.encode(text[:marker_pos]))
        token_end = len(tokenizer.encode(text[:next_marker_pos]))

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

            # Find tool calls within assistant message
            if msg.tool_calls:
                for j, tc in enumerate(msg.tool_calls):
                    # Find <|python_tag|> position
                    ptag_pos = text.find("<|python_tag|>", marker_pos)
                    if ptag_pos > 0 and ptag_pos < next_marker_pos:
                        tc_token_start = len(tokenizer.encode(text[:ptag_pos]))

                        # Find end of tool call (before <|eom_id|> or next message)
                        eom_pos = text.find("<|eom_id|>", ptag_pos)
                        if eom_pos < 0 or eom_pos > next_marker_pos:
                            eom_pos = next_marker_pos
                        tc_token_end = len(tokenizer.encode(text[:eom_pos]))

                        # Find where tool name ends
                        # Pattern: {"name": "tool_name"
                        name_end_pattern = f'"{tc.function.name}"'
                        name_end_pos = text.find(name_end_pattern, ptag_pos)
                        name_token_end = None
                        if name_end_pos > 0:
                            name_token_end = len(tokenizer.encode(text[:name_end_pos + len(name_end_pattern)]))

                        tool_call_spans.append(ToolCallSpan(
                            message_index=i,
                            call_index=j,
                            tool_name=tc.function.name,
                            token_start=tc_token_start,
                            token_end=tc_token_end,
                            name_token_end=name_token_end,
                        ))

    return RenderAlignment(
        message_spans=message_spans if message_spans else None,
        assistant_spans=assistant_spans if assistant_spans else None,
        tool_call_spans=tool_call_spans if tool_call_spans else None,
    )


# =============================================================================
# Loss Mask Policy Application
# =============================================================================

def apply_loss_mask_policy(
    render: RenderedView,
    trace: Trace,
    policy,
    tokenizer=None,
) -> List[float]:
    """
    Apply a loss masking policy to create per-token loss mask.

    Args:
        render: Rendered view with alignment and signals
        trace: Source trace
        policy: LMP policy object (from registry)
        tokenizer: Tokenizer (optional, for some policies)

    Returns:
        List of float mask values (0.0 = masked, 1.0 = unmasked)
    """
    seq_len = len(render.input_ids)
    mask = [0.0] * seq_len

    strategy = policy.strategy
    params = policy.params or {}

    if strategy == "assistant_turns":
        # Loss only on assistant tokens
        mask = _mask_assistant_only(render, mask)

    elif strategy == "tool_calls":
        # Loss only on tool call tokens
        mask = _mask_tool_calls_only(render, mask)

    elif strategy == "action_prefix":
        # Loss only on action prefix tokens
        mask = _mask_action_prefix(render, trace, mask, params)

    elif strategy == "commitment_prefix":
        # Loss only on tokens up to commitment point
        mask = _mask_commitment_prefix(render, mask, params)

    elif strategy == "shock_aware":
        # Up-weight tokens after injection spans
        mask = _mask_shock_aware(render, mask, params)

    elif strategy == "dual_span":
        # Union of injection span + commitment prefix
        mask = _mask_dual_span(render, mask, params)

    elif strategy == "progressive":
        # Soft ramp toward commitment point
        mask = _mask_progressive(render, mask, params)

    elif strategy == "injection_retain":
        # Include injection in context but mask it for loss
        mask = _mask_injection_retain(render, mask, params)

    else:
        # Default: assistant only
        mask = _mask_assistant_only(render, mask)

    return mask


def _mask_assistant_only(render: RenderedView, mask: List[float]) -> List[float]:
    """Mask only assistant turns."""
    if render.alignment and render.alignment.assistant_spans:
        for span in render.alignment.assistant_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 1.0
    return mask


def _mask_tool_calls_only(render: RenderedView, mask: List[float]) -> List[float]:
    """Mask only tool call tokens."""
    if render.alignment and render.alignment.tool_call_spans:
        for span in render.alignment.tool_call_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 1.0
    return mask


def _mask_action_prefix(
    render: RenderedView,
    trace: Trace,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Mask only action prefix (up to tool name)."""
    include_python_tag = params.get("include_python_tag", True)

    if render.alignment and render.alignment.tool_call_spans:
        for span in render.alignment.tool_call_spans:
            end_idx = span.name_token_end if span.name_token_end else span.token_end
            start_idx = span.token_start

            for i in range(start_idx, min(end_idx, len(mask))):
                mask[i] = 1.0

    return mask


def _mask_commitment_prefix(
    render: RenderedView,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Mask only tokens up to commitment point."""
    if render.signals and render.signals.action_commitments:
        for commitment in render.signals.action_commitments:
            if commitment.prefix_token_indices:
                for i in commitment.prefix_token_indices:
                    if 0 <= i < len(mask):
                        mask[i] = 1.0
    return mask


def _mask_shock_aware(
    render: RenderedView,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Up-weight tokens after injection spans."""
    # First, mask assistant turns
    mask = _mask_assistant_only(render, mask)

    # Then up-weight tokens after shock
    upweight = params.get("post_shock_upweight", 1.5)

    if render.signals and render.signals.injection_spans:
        for span in render.signals.injection_spans:
            # Up-weight tokens after the injection span
            for i in range(span.token_end, len(mask)):
                if mask[i] > 0:
                    mask[i] = min(mask[i] * upweight, 2.0)

    return mask


def _mask_dual_span(
    render: RenderedView,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Union of injection span + commitment prefix."""
    # Mask injection spans
    if render.signals and render.signals.injection_spans:
        for span in render.signals.injection_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 1.0

    # Mask commitment prefixes
    if render.signals and render.signals.action_commitments:
        for commitment in render.signals.action_commitments:
            if commitment.prefix_token_indices:
                for i in commitment.prefix_token_indices:
                    if 0 <= i < len(mask):
                        mask[i] = 1.0

    return mask


def _mask_progressive(
    render: RenderedView,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Soft ramp toward commitment point."""
    if not render.signals or not render.signals.action_commitments:
        return _mask_assistant_only(render, mask)

    ramp_length = params.get("ramp_length", 10)

    for commitment in render.signals.action_commitments:
        commit_idx = commitment.commitment_token_idx

        # Find assistant span containing this commitment
        if render.alignment and render.alignment.assistant_spans:
            for span in render.alignment.assistant_spans:
                if span.token_start <= commit_idx < span.token_end:
                    # Apply progressive ramp
                    for i in range(span.token_start, min(span.token_end, len(mask))):
                        if i <= commit_idx:
                            # Before or at commitment: full weight
                            mask[i] = 1.0
                        elif i < commit_idx + ramp_length:
                            # Ramp down after commitment
                            distance = i - commit_idx
                            mask[i] = 1.0 - (distance / ramp_length)
                        else:
                            mask[i] = 0.0

    return mask


def _mask_injection_retain(
    render: RenderedView,
    mask: List[float],
    params: dict,
) -> List[float]:
    """Include injection in context (input) but mask for loss."""
    # Mask assistant turns
    mask = _mask_assistant_only(render, mask)

    # Zero out injection spans (don't learn from the injection itself)
    if render.signals and render.signals.injection_spans:
        for span in render.signals.injection_spans:
            for i in range(span.token_start, min(span.token_end, len(mask))):
                mask[i] = 0.0

    return mask


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract signals from traces")
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output renders JSONL")
    parser.add_argument("--model", type=str, default=None, help="Model for surprisal computation")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--shock-detector", default="contiguous_threshold")
    parser.add_argument("--shock-threshold", type=float, default=6.0)
    parser.add_argument("--commitment-detector", default="tool_name_selected")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Optionally load model
    model = None
    if args.model:
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Process traces
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(args.output, "w") as out_f:
        with open(args.traces, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                trace_data = json.loads(line)
                trace = Trace.from_dict(trace_data)

                render, _ = extract_signals(
                    trace, tokenizer, model,
                    shock_method=args.shock_detector,
                    shock_threshold=args.shock_threshold,
                    commitment_method=args.commitment_detector,
                )

                out_f.write(json.dumps(render.to_dict()) + "\n")
                count += 1

                if args.limit and count >= args.limit:
                    break

                if count % 100 == 0:
                    logger.info(f"Processed {count} traces...")

    logger.info(f"Extracted signals for {count} traces -> {args.output}")


if __name__ == "__main__":
    main()
