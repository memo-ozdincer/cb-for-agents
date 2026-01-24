"""
Tier C: Rendered View Schema (render_v1)

Tokenizer/model-specific rendering of a trace. This is where
apply_chat_template output lives.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass
class RenderOptions:
    """Options used during rendering."""
    chat_template_name: str = "default"
    add_generation_prompt: bool = False
    tools_included: bool = False
    tools_digest: Optional[str] = None
    max_length: Optional[int] = None
    truncation_side: str = "right"


@dataclass
class MessageSpan:
    """Token span for a message."""
    message_index: int
    role: str
    token_start: int
    token_end: int


@dataclass
class AssistantSpan:
    """Token span for assistant completion."""
    message_index: int
    token_start: int
    token_end: int


@dataclass
class ToolCallSpan:
    """Token span for a tool call within an assistant message."""
    message_index: int
    call_index: int
    tool_name: str
    token_start: int
    token_end: int
    name_token_end: Optional[int] = None  # For action commitment masking


@dataclass
class RenderAlignment:
    """Mapping from canonical message structure to token positions."""
    message_spans: Optional[List[MessageSpan]] = None
    assistant_spans: Optional[List[AssistantSpan]] = None
    tool_call_spans: Optional[List[ToolCallSpan]] = None


@dataclass
class SpecialTokenPositions:
    """Positions of special tokens in the sequence."""
    bos_position: Optional[int] = None
    eos_positions: Optional[List[int]] = None
    python_tag_positions: Optional[List[int]] = None


# =============================================================================
# Signals (Shock Detection, Action Commitment)
# =============================================================================

@dataclass
class ShockScore:
    """Shock/surprisal metrics for an injection span."""
    max_surprisal: Optional[float] = None
    mean_surprisal: Optional[float] = None
    max_delta: Optional[float] = None  # Max delta between adjacent token surprisals
    span_length: Optional[int] = None
    percentile_rank: Optional[float] = None


@dataclass
class InjectionSpan:
    """A detected injection span with shock/surprisal metrics."""
    token_start: int
    token_end: int
    shock_score: Optional[ShockScore] = None
    detection_method: Optional[str] = None  # 'threshold', 'percentile', 'contiguous_threshold', 'adaptive'
    confidence: Optional[float] = None


@dataclass
class ActionCommitment:
    """A detected action commitment point in assistant output."""
    commitment_token_idx: int  # Token index where action becomes guaranteed
    commit_type: str  # 'tool_name_selected', 'json_frame_parse_valid', 'logprob_margin', etc.
    assistant_message_index: Optional[int] = None
    committed_tool: Optional[str] = None
    expected_tool: Optional[str] = None
    logprob_margin: Optional[float] = None  # logP(committed) - logP(expected)
    guarantee_prefix: Optional[str] = None
    prefix_token_indices: Optional[List[int]] = None


@dataclass
class DetectorMetadata:
    """Metadata about signal detectors used."""
    shock_detector_id: Optional[str] = None
    shock_detector_params: Optional[Dict[str, Any]] = None
    commitment_detector_id: Optional[str] = None
    reference_model: Optional[str] = None


@dataclass
class RenderSignals:
    """Computed signals for advanced loss masking."""
    injection_spans: Optional[List[InjectionSpan]] = None
    action_commitments: Optional[List[ActionCommitment]] = None
    token_surprisals: Optional[List[float]] = None  # Per-token surprisal values
    detector_metadata: Optional[DetectorMetadata] = None


# =============================================================================
# Main RenderedView Class
# =============================================================================

@dataclass
class RenderedView:
    """
    Rendered view record (Tier C).

    This is where tokenizer-specific information lives. Generate new
    renders when using different models/tokenizers.
    """
    render_id: str
    trace_id: str
    tokenizer_id: str
    input_ids: List[int]

    created_at: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    render_options: Optional[RenderOptions] = None
    rendered_text: Optional[str] = None  # Can omit for space
    attention_mask: Optional[List[int]] = None
    sequence_length: Optional[int] = None
    alignment: Optional[RenderAlignment] = None
    special_tokens: Optional[SpecialTokenPositions] = None
    signals: Optional[RenderSignals] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.sequence_length is None:
            self.sequence_length = len(self.input_ids)
        if self.render_options is None:
            self.render_options = RenderOptions()

    @classmethod
    def generate_id(
        cls,
        trace_id: str,
        tokenizer_id: str,
        options_hash: Optional[str] = None,
        render_options: Optional[RenderOptions] = None,
    ) -> str:
        """Generate a deterministic render ID."""
        # Extract model short name from tokenizer_id
        model_short = tokenizer_id.split("/")[-1].lower().replace("-", "_")[:20]

        # Extract trace hash from trace_id
        trace_hash = trace_id.split("_")[-1][:16]

        if options_hash is None:
            if render_options is None:
                render_options = RenderOptions()
            options_str = json.dumps({
                "chat_template": render_options.chat_template_name,
                "add_gen_prompt": render_options.add_generation_prompt,
                "max_length": render_options.max_length,
            }, sort_keys=True)
            options_hash = hashlib.sha256(options_str.encode()).hexdigest()[:8]

        return f"render_{model_short}_{trace_hash}_{options_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [_clean(item) for item in obj]
            return obj

        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderedView":
        """Create from dictionary."""
        render_options = None
        if data.get("render_options"):
            render_options = RenderOptions(**data["render_options"])

        alignment = None
        if data.get("alignment"):
            align_data = data["alignment"]
            alignment = RenderAlignment(
                message_spans=[MessageSpan(**s) for s in align_data.get("message_spans", [])] or None,
                assistant_spans=[AssistantSpan(**s) for s in align_data.get("assistant_spans", [])] or None,
                tool_call_spans=[ToolCallSpan(**s) for s in align_data.get("tool_call_spans", [])] or None,
            )

        special_tokens = None
        if data.get("special_tokens"):
            special_tokens = SpecialTokenPositions(**data["special_tokens"])

        signals = None
        if data.get("signals"):
            sig_data = data["signals"]
            injection_spans = None
            if sig_data.get("injection_spans"):
                injection_spans = []
                for isp in sig_data["injection_spans"]:
                    shock = ShockScore(**isp["shock_score"]) if isp.get("shock_score") else None
                    injection_spans.append(InjectionSpan(
                        token_start=isp["token_start"],
                        token_end=isp["token_end"],
                        shock_score=shock,
                        detection_method=isp.get("detection_method"),
                        confidence=isp.get("confidence"),
                    ))
            action_commitments = None
            if sig_data.get("action_commitments"):
                action_commitments = [
                    ActionCommitment(**ac) for ac in sig_data["action_commitments"]
                ]
            detector_meta = None
            if sig_data.get("detector_metadata"):
                detector_meta = DetectorMetadata(**sig_data["detector_metadata"])
            signals = RenderSignals(
                injection_spans=injection_spans,
                action_commitments=action_commitments,
                token_surprisals=sig_data.get("token_surprisals"),
                detector_metadata=detector_meta,
            )

        return cls(
            render_id=data["render_id"],
            trace_id=data["trace_id"],
            tokenizer_id=data["tokenizer_id"],
            input_ids=data["input_ids"],
            created_at=data.get("created_at"),
            tokenizer_revision=data.get("tokenizer_revision"),
            render_options=render_options,
            rendered_text=data.get("rendered_text"),
            attention_mask=data.get("attention_mask"),
            sequence_length=data.get("sequence_length"),
            alignment=alignment,
            special_tokens=special_tokens,
            signals=signals,
        )

    def get_assistant_token_indices(self) -> List[int]:
        """Get all token indices that belong to assistant messages."""
        indices = []
        if self.alignment and self.alignment.assistant_spans:
            for span in self.alignment.assistant_spans:
                indices.extend(range(span.token_start, span.token_end))
        return indices

    def get_tool_call_token_indices(self) -> List[int]:
        """Get all token indices that belong to tool calls."""
        indices = []
        if self.alignment and self.alignment.tool_call_spans:
            for span in self.alignment.tool_call_spans:
                indices.extend(range(span.token_start, span.token_end))
        return indices
