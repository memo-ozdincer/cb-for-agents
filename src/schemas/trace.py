"""
Tier B: Canonical Trace Schema (trace_v1)

Model-agnostic trace representation that supports:
- Multi-turn and single-turn traces
- Tool call annotations with expected vs observed
- Training controls (sample_weight, loss_mask_policy)
- Provenance and deterministic IDs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import hashlib
import json


# =============================================================================
# Sub-components
# =============================================================================

@dataclass
class ToolCallFunction:
    """Function definition within a tool call."""
    name: str
    arguments: Optional[Dict[str, Any]] = None
    arguments_json: Optional[str] = None  # Raw JSON for exact replay


@dataclass
class ToolCallExpected:
    """What SHOULD have happened (for attack traces)."""
    should_call: bool = True
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallResult:
    """Tool execution result."""
    output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class ToolCall:
    """A tool call within an assistant message."""
    function: ToolCallFunction
    call_id: Optional[str] = None
    schema_ref: Optional[Dict[str, str]] = None  # schema_version, schema_digest
    expected: Optional[ToolCallExpected] = None
    result: Optional[ToolCallResult] = None


@dataclass
class Message:
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    mid: Optional[str] = None
    name: Optional[str] = None  # Speaker name or tool name
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool responses
    thinking: Optional[str] = None  # CoT trace if captured


@dataclass
class TraceSource:
    """Provenance tracking for the trace."""
    dataset: Literal[
        "fujitsu_b4",
        "agentdojo", "tau2", "agentharm", "webarena", "weblinx", "attackqa",
        "synthetic"
    ]
    tier: Literal["raw", "curated", "derived"] = "curated"
    subset: Optional[str] = None
    record_locator: Optional[Dict[str, str]] = None  # kind, value
    ingest_version: Optional[str] = None


@dataclass
class TraceTask:
    """Task family and name for grouping."""
    family: Optional[str] = None  # e.g., 'tool_flip', 'prompt_injection'
    name: Optional[str] = None
    variant: Optional[str] = None


@dataclass
class TraceLabels:
    """Outcome labels for the trace."""
    category: Optional[Literal["harmful", "benign"]] = None
    security_outcome: Optional[Literal["safe", "unsafe", "refusal", "tool_misuse", "unknown"]] = None
    attack_type: Optional[str] = None
    attack_succeeded: Optional[bool] = None
    capability_category: Optional[str] = None


@dataclass
class TraceToolAttack:
    """Tool-flip attack details (for B4-style attacks)."""
    expected_tool: Optional[str] = None
    observed_tool: Optional[str] = None
    attack_vector: Optional[str] = None
    injection_text: Optional[str] = None


@dataclass
class TraceMixture:
    """Mixture assignment for curriculum learning."""
    class_id: Optional[str] = None  # e.g., 'fujitsu_b4/tool_flip'
    stage_tags: Optional[List[str]] = None


@dataclass
class TraceTraining:
    """Training configuration for this trace."""
    sample_weight: float = 1.0
    loss_mask_policy: str = "assistant_only"
    loss_mask_params: Optional[Dict[str, Any]] = None
    mixture: Optional[TraceMixture] = None


@dataclass
class TraceLinks:
    """Links to related traces and raw records."""
    raw_id: Optional[str] = None
    paired_trace_id: Optional[str] = None  # Ds <-> Dr linkage
    parent_trace_ids: Optional[List[str]] = None


@dataclass
class InjectionCharSpan:
    """Character span of injection text in a message."""
    message_index: int
    char_start: int
    char_end: int


@dataclass
class SignalHints:
    """
    Character-level hints for signal detection (tokenizer-agnostic).
    Token-level signals are computed in render_v1.
    """
    injection_char_span: Optional[InjectionCharSpan] = None
    expected_tool_name: Optional[str] = None  # The tool that SHOULD be called
    observed_tool_name: Optional[str] = None  # The tool that WAS called (bad/wrong)
    guarantee_prefix_hint: Optional[str] = None  # Known prefix after which action is guaranteed


# =============================================================================
# Main Trace Class
# =============================================================================

@dataclass
class Trace:
    """
    Canonical trace record (Tier B).

    This is the "one schema" for all training data. Single-turn flips are
    traces with 2-3 messages; multi-turn traces have more.
    
    Completeness and tier fields track the processing stage:
    - completeness: "skeleton" (no assistant) or "complete" (has assistant)
    - tier: "B1" (skeleton, needs generation) or "B2" (complete, ready for rendering)
    """
    id: str
    source: TraceSource
    messages: List[Message]
    split: Literal["train", "eval", "test", "dev"]

    created_at: Optional[str] = None
    completeness: Literal["skeleton", "complete"] = "complete"
    tier: Literal["B1", "B2"] = "B2"
    task: Optional[TraceTask] = None
    labels: Optional[TraceLabels] = None
    tool_attack: Optional[TraceToolAttack] = None
    training: Optional[TraceTraining] = None
    links: Optional[TraceLinks] = None
    signal_hints: Optional[SignalHints] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.training is None:
            self.training = TraceTraining()

    @classmethod
    def generate_id(
        cls,
        source_dataset: str,
        content_hash: Optional[str] = None,
        messages: Optional[List[Message]] = None,
    ) -> str:
        """Generate a deterministic trace ID."""
        if content_hash is None:
            if messages is None:
                raise ValueError("Must provide content_hash or messages")
            # Hash the message content
            content = json.dumps(
                [{"role": m.role, "content": m.content} for m in messages],
                sort_keys=True,
            )
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        return f"trace_{source_dataset}_{content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        from dataclasses import asdict

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [_clean(item) for item in obj]
            return obj

        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        """Create trace from dictionary."""
        # Parse nested structures
        source = TraceSource(**data["source"]) if "source" in data else None

        messages = []
        for msg_data in data.get("messages", []):
            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = []
                for tc_data in msg_data["tool_calls"]:
                    func = ToolCallFunction(**tc_data["function"])
                    expected = None
                    if tc_data.get("expected"):
                        expected = ToolCallExpected(**tc_data["expected"])
                    result = None
                    if tc_data.get("result"):
                        result = ToolCallResult(**tc_data["result"])
                    tool_calls.append(ToolCall(
                        function=func,
                        call_id=tc_data.get("call_id"),
                        schema_ref=tc_data.get("schema_ref"),
                        expected=expected,
                        result=result,
                    ))
            messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                mid=msg_data.get("mid"),
                name=msg_data.get("name"),
                tool_calls=tool_calls,
                tool_call_id=msg_data.get("tool_call_id"),
                thinking=msg_data.get("thinking"),
            ))

        task = TraceTask(**data["task"]) if data.get("task") else None
        labels = TraceLabels(**data["labels"]) if data.get("labels") else None
        tool_attack = TraceToolAttack(**data["tool_attack"]) if data.get("tool_attack") else None

        training = None
        if data.get("training"):
            training_data = data["training"]
            mixture = None
            if training_data.get("mixture"):
                mixture = TraceMixture(**training_data["mixture"])
            training = TraceTraining(
                sample_weight=training_data.get("sample_weight", 1.0),
                loss_mask_policy=training_data.get("loss_mask_policy", "assistant_only"),
                loss_mask_params=training_data.get("loss_mask_params"),
                mixture=mixture,
            )

        links = TraceLinks(**data["links"]) if data.get("links") else None

        signal_hints = None
        if data.get("signal_hints"):
            sh_data = data["signal_hints"]
            inj_span = None
            if sh_data.get("injection_char_span"):
                inj_span = InjectionCharSpan(**sh_data["injection_char_span"])
            signal_hints = SignalHints(
                injection_char_span=inj_span,
                expected_tool_name=sh_data.get("expected_tool_name"),
                observed_tool_name=sh_data.get("observed_tool_name"),
                guarantee_prefix_hint=sh_data.get("guarantee_prefix_hint"),
            )

        return cls(
            id=data["id"],
            source=source,
            messages=messages,
            split=data["split"],
            created_at=data.get("created_at"),
            completeness=data.get("completeness", "complete"),
            tier=data.get("tier", "B2"),
            task=task,
            labels=labels,
            tool_attack=tool_attack,
            training=training,
            links=links,
            signal_hints=signal_hints,
        )

    def get_assistant_messages(self) -> List[Message]:
        """Get all assistant messages."""
        return [m for m in self.messages if m.role == "assistant"]

    def get_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls across all assistant messages."""
        calls = []
        for msg in self.messages:
            if msg.tool_calls:
                calls.extend(msg.tool_calls)
        return calls

    def has_tool_calls(self) -> bool:
        """Check if trace contains any tool calls."""
        return len(self.get_tool_calls()) > 0

    def is_harmful(self) -> bool:
        """Check if this is a harmful trace (for CB training)."""
        if self.labels and self.labels.category:
            return self.labels.category == "harmful"
        return False

    def is_skeleton(self) -> bool:
        """Check if this is a skeleton trace (no assistant messages)."""
        return self.completeness == "skeleton" or self.tier == "B1"
    
    def is_complete(self) -> bool:
        """Check if this is a complete trace (has assistant messages)."""
        return self.completeness == "complete" or self.tier == "B2"
    def is_benign(self) -> bool:
        """Check if this is a benign trace (for CB training)."""
        if self.labels and self.labels.category:
            return self.labels.category == "benign"
        return False
