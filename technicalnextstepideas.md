## E. Technical Next Steps

### E.1 Add More Data

| Data Type | Source | Effort | Priority |
|-----------|--------|:------:|:--------:|
| UltraChat general capability | HuggingFace | Low | 🔴 HIGH |
| XSTest borderline cases | GitHub CSV | Low | 🔴 HIGH |
| Refusal preserve (model-generated) | Generate | Medium | 🔴 HIGH |
| Fujitsu B1/B3 (content injection) | Already in workspace | Low | 🟡 MED |
| Full WebLINX | HuggingFace | Medium | 🟢 LOW |
| HarmBench text-based | GitHub | Medium | 🟢 LOW |

### E.2 Expand Beyond "Tool Name Flip"

**2D Taxonomy: Tool Routing × Argument Integrity**

| | Correct Tool | Wrong Tool |
|---|:---:|:---:|
| **Correct Args** | ✅ Benign | ⚠️ Tool-flip (current DS) |
| **Malicious Args** | 🔴 Param injection | 🔴 Full compromise |

**Current dataset already contains param manipulation:**
```
Expected: retrieve_multimodal_docs(query="company policy")
Observed: retrieve_multimodal_docs(query="competitor secrets")
          ↑ Same tool, manipulated intent
```

**Action:** Add `expected_params` / `observed_params` to labels, expand DS to include param injection attacks.

### E.3 Add Multi-Step Injection Traces

From AgentDojo: Injections occur in tool outputs, not just user messages.

**Data generation approach:**
```python
# Generate trajectories where injection is in tool output
trajectory = [
    {"role": "user", "content": "Search for policies"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "Result: ... <!-- INJECTION --> ..."},
    {"role": "user", "content": "Now search guidelines"},
    {"role": "assistant", "tool_calls": [...]}  # ← Label this decision
]
```

### E.4 Add "Format-Hardening" Data

**Problem:** Some benign/retain samples have format errors (missing end tokens).

**Solution:**
1. Validate all tool calls are syntactically valid
2. Add synthetic samples with proper delimiters
3. Quality gate: Reject samples with format errors

```python
def validate_tool_format(text: str) -> bool:
    if "<|python_tag|>" in text:
        return text.endswith("<|eom_id|>") or text.endswith("<|eot_id|>")
    return True  # Text responses OK
```

### E.5 Bigger Models: What Stays Same vs Changes

**What Stays the Same:**
- Training objective: RR loss structure (reroute + retain)
- Memory trick: Single model with adapter toggle for frozen reps
- Data format: Same DS/DR structure
- Loss computation: Same `reroute_loss()` and `retain_loss()` functions

**What Changes:**

| Aspect | 8B Dense | 17B+ MoE |
|--------|----------|----------|
| `cb_target_layers` | [10, 20] | [12, 24, 36] — more layers, different sweet spots |
| `learning_rate` | 5e-5 | 2e-5 — lower for stability |
| `total_steps` | 150 | 300 — more steps needed |
| `batch_size` | 16 | 8 — memory constraints |
| `gradient_checkpointing` | False | True — essential |
| LoRA targets | All projections | Skip router/gate in MoE |

---