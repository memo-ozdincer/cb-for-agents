# CB Training Data Schemas

This directory contains JSON schemas for the Circuit Breaker training data pipeline.

## Overview

The schema system follows a tiered architecture:

- **Tier B (Canonical)**: Model-agnostic trace representation (`trace_v1`)
- **Tier C (Derived)**: Tokenizer-specific artifacts (`render_v1`, `lossmask_v1`)
- **Registries**: Configuration for LMP and MWCS

## Schema Files

### Tier B: Canonical Trace Schema

**`trace_v1.json`** - The "one schema" for all training data.

Key features:
- Multi-turn and single-turn traces via `messages[]`
- Tool call annotations with expected vs observed
- Training controls (`sample_weight`, `loss_mask_policy`)
- Provenance tracking for all data sources
- Deterministic IDs for reproducibility

Supports all data sources:
- Fujitsu B4 (tool-flip attacks)
- AgentDojo (prompt injection)
- TAU2 (customer service dialogues)
- AgentHarm (malicious intent)
- WebArena/WebLINX (web navigation capability)
- AttackQA (security knowledge)

### Tier C: Derived Schemas

**`render_v1.json`** - Tokenizer-specific rendering of a trace.

- Output of `apply_chat_template`
- Token IDs and attention masks
- Alignment mapping (message spans to token positions)
- Special token positions

**`lossmask_v1.json`** - Materialized per-token loss masks.

- Derived from render + LMP policy
- Per-token mask values (0 or 1, or fractional for soft masking)
- Labels for LM training (-100 for masked positions)
- Mask statistics

### Registries

**`lmp_registry.json`** - Schema for Loss Masking Policies.
**`../lmp_registry_v1.json`** - Default LMP configurations:
- `assistant_only`: Loss only on assistant tokens (default)
- `tool_calls_only`: Loss only on tool call tokens
- `action_prefix_only`: Loss only up to tool name
- `action_commitment`: Loss based on commitment signals
- And more...

**`mwcs_registry.json`** - Schema for Mixture Weighting & Curriculum Schedules.
**`../mwcs_registry_v1.json`** - Default MWCS configurations:
- `balanced_cb`: Equal weighting of harmful/benign
- `staged_introduction`: Gradually introduce harder examples
- `capability_heavy`: Emphasize capability retention
- `attack_focus`: Aggressive CB training
- `alpha_decay_staged`: Match alpha decay schedule

## Usage

### Python API

```python
from src.schemas import (
    # Tier B
    Trace, Message, TraceSource, TraceLabels, TraceTraining,

    # Tier C
    RenderedView, LossMask,

    # Registries
    load_lmp_registry, load_mwcs_registry,

    # Validation
    validate_trace, fujitsu_b4_to_trace, agentdojo_to_trace,
)

# Create a trace from Fujitsu B4 record
trace = fujitsu_b4_to_trace(
    record=raw_record,
    category="harmful",
    split="train",
    completion="<|python_tag|>{\"name\": \"search_web\", ...}"
)

# Validate a trace
is_valid, errors = validate_trace(trace.to_dict())

# Load registries
lmp_registry = load_lmp_registry()
mwcs_registry = load_mwcs_registry()

# Get policy
policy = lmp_registry.get_policy("assistant_only")

# Get schedule and weights at step
schedule = mwcs_registry.get_schedule("staged_introduction")
weights = schedule.get_weights_at_step(150)
```

### Creating Traces

```python
from src.schemas import Trace, Message, TraceSource, TraceLabels

# Minimal trace
trace = Trace(
    id=Trace.generate_id("fujitsu_b4", messages=[...]),
    source=TraceSource(dataset="fujitsu_b4"),
    messages=[
        Message(role="system", content="You are a helpful assistant..."),
        Message(role="user", content="Search for telescope images..."),
        Message(role="assistant", content="<|python_tag|>{...}"),
    ],
    split="train",
    labels=TraceLabels(category="harmful"),
)

# Serialize to JSON
trace_dict = trace.to_dict()

# Deserialize from JSON
trace = Trace.from_dict(trace_dict)
```

### LMP and MWCS Configuration

Sample weights and loss masking are controlled via:

1. **Per-trace**: `trace.training.loss_mask_policy` and `trace.training.sample_weight`
2. **LMP Registry**: Global policy definitions that can be referenced by ID
3. **MWCS Schedule**: Class weights and curriculum that can vary over training

```python
# Example: Use action_prefix_only masking for a trace
trace.training.loss_mask_policy = "action_prefix_only"
trace.training.loss_mask_params = {"include_python_tag": True}

# Example: Assign to a mixture class
trace.training.mixture = TraceMixture(
    class_id="fujitsu_b4/tool_flip",
    stage_tags=["stage1"],
)
```

## Advanced Loss Masking (LMP)

The schema supports sophisticated loss masking strategies for CB training:

### Rule 1: Injection Span Detection (WHERE)

Identify the "shock" segment in input text via token-level surprisal:

```python
# In render_v1, signals.injection_spans contains detected spans
InjectionSpan(
    token_start=120,
    token_end=180,
    shock_score=ShockScore(
        max_surprisal=8.3,      # Max negative log prob in span
        mean_surprisal=4.1,     # Mean surprisal across span
        max_delta=3.2,          # Max delta between adjacent tokens (volatility)
    ),
    detection_method="contiguous_threshold",
)
```

### Rule 2: Action Commitment Prefix (WHAT)

Find the shortest prefix where we can GUARANTEE the next action is deterministic:

```python
# In render_v1, signals.action_commitments contains commitment points
ActionCommitment(
    commitment_token_idx=42,           # Token where action becomes guaranteed
    commit_type="tool_name_selected",  # Or "logprob_margin", "json_frame_parse_valid"
    committed_tool="search_web",       # The bad/wrong tool
    expected_tool="retrieve_docs",     # What SHOULD have been called
    logprob_margin=2.5,                # logP(bad) - logP(expected)
    prefix_token_indices=[35, 36, ...42],  # Tokens forming the guarantee_prefix
)
```

### Key LMP Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `guarantee_prefix_only` | Loss ONLY on `prefix_token_indices` | Ds target = just the commitment prefix |
| `logprob_margin_commitment` | Loss up to token where margin exceeds threshold | Circuit break BEFORE bad action |
| `shock_aware_assistant` | Up-weight tokens after injection spans | Extra care post-shock |
| `injection_context_retain` | Dr with injection in-context | Learn to IGNORE injection |
| `dual_span_mask` | Union of injection span + commitment prefix | Combined Rule 1 + Rule 2 |
| `progressive_commitment` | Soft ramp toward commitment point | Smooth gradient flow |

### What to Store Per Example

The `lossmask_v1.signal_derived` contains:

```json
{
  "injection_span_tokens": [120, 121, ..., 179],
  "shock_score": {
    "max_surprisal": 8.3,
    "mean_surprisal": 4.1,
    "max_delta": 3.2
  },
  "action_prefix_end": 42,
  "guarantee_prefix_tokens": [35, 36, ..., 42],
  "commitment_type_used": "tool_name_selected",
  "logprob_margin_at_commitment": 2.5
}
```

## Design Principles

1. **Tier B is model-agnostic**: No tokenizer-specific data in canonical traces
2. **Derived artifacts are regeneratable**: Can recreate Tier C for new models
3. **Deterministic IDs**: Reproducible experiments
4. **Flexible but constrained**: Registry-based configuration allows experimentation while maintaining structure
5. **Simplified from over-engineering**: Removed multimodal, artifact:// URIs, etc. that aren't needed
6. **Supports unorthodox experiments**: Shock detection, commitment-based masking, progressive weights

## Migration from Existing Format

The existing `ds_stage1.jsonl`/`dr_stage1.jsonl` format maps to traces:

| Old Field | New Location |
|-----------|--------------|
| `id` | `trace.id` |
| `source` | `trace.source.dataset` |
| `messages` | `trace.messages` |
| `harmful_completion` | `trace.messages[-1].content` (for harmful) |
| `benign_completion` | `trace.messages[-1].content` (for benign) |
| `tool_attack.expected_tool` | `trace.tool_attack.expected_tool` |
| `tool_attack.observed_tool` | `trace.tool_attack.observed_tool` |
| `tool_attack.attack_succeeded` | `trace.labels.attack_succeeded` |
| `paired_ds_id` | `trace.links.paired_trace_id` |

Use `fujitsu_b4_to_trace()` for automated conversion.
