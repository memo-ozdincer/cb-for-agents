# Agentic Harm Data Augmentation Pipeline

This directory contains scripts for generating and augmenting agentic harm data
for Circuit Breaker training on agents.

## Overview

The goal is to create diverse, realistic agentic harm traces that include:
- Multi-turn conversations with tool calls
- Indirect prompt injection attacks
- Tool manipulation/flip attacks
- Memory poisoning scenarios
- Agent-to-agent attacks
- File/RAG-based attacks

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC HARM AUGMENTATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │   STAGE 1    │   │   STAGE 2    │   │   STAGE 3    │                 │
│  │  Template    │──▶│   Trace      │──▶│  Quality     │                 │
│  │  Generation  │   │  Generation  │   │  Filtering   │                 │
│  └──────────────┘   └──────────────┘   └──────────────┘                 │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  Attack scenarios    Multi-turn traces    Validated traces              │
│  from taxonomy       with tool calls      ready for training            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Sources for Augmentation

### 1. Fujitsu B4 Tool Attacks (13,246 samples)
- Have: `tool_attack{benign_query, malicious_injection, expected_tool, simulated_tool}`
- Need: Full multi-turn traces showing the tool flip in action

### 2. AgentHarm Behaviors (208 samples)
- Have: Harmful behavior prompts
- Need: Agent completions showing harmful tool use

### 3. New Attack Categories (from OWASP/Red Team taxonomy)
- Indirect prompt injection (file-based, RAG-based)
- Memory poisoning attacks
- Privilege escalation via tools
- Cross-session data leakage
- Agent-to-agent manipulation

## Scripts

### `generate_b4_traces.py`
Converts Fujitsu B4 tool attack definitions into realistic multi-turn traces.

### `generate_agentharm_completions.py`
Generates harmful agent completions for AgentHarm prompts using uncensored models.

### `generate_attack_scenarios.py`
Creates new attack scenarios from OWASP/Red Team taxonomy templates.

### `quality_filter.py`
Filters generated traces for quality and diversity.

## Usage

```bash
# Generate traces for Fujitsu B4
python scripts/augmentation/generate_b4_traces.py \
    --input data/circuit_breakers/harmful/harmful_pairs.completions.jsonl \
    --output data/augmented/b4_traces.jsonl \
    --model mistral-nemo-instruct  # or any uncensored model

# Generate AgentHarm completions
python scripts/augmentation/generate_agentharm_completions.py \
    --input data/agentharm/ \
    --output data/augmented/agentharm_traces.jsonl

# Generate new attack scenarios
python scripts/augmentation/generate_attack_scenarios.py \
    --templates scripts/augmentation/attack_templates/ \
    --output data/augmented/new_attacks.jsonl \
    --num-per-template 100

# Quality filtering
python scripts/augmentation/quality_filter.py \
    --input data/augmented/*.jsonl \
    --output data/augmented/filtered.jsonl
```

## Attack Category Templates

See `attack_templates/` for YAML templates defining:
- LLM01: Prompt Injection (Direct/Indirect)
- LLM05: Improper Output Handling
- LLM06: Excessive Agency
- Custom: Agent-to-agent attacks
- Custom: File/RAG poisoning
- Custom: Memory poisoning
