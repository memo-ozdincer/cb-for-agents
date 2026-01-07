#!/usr/bin/env python3
"""
Generate multi-turn agentic traces from Fujitsu B4 tool attack definitions.

This script takes tool_attack definitions (which only have attack metadata)
and generates realistic multi-turn conversation traces showing:
1. User makes benign request
2. Agent receives combined_query (with hidden injection)
3. Agent calls the WRONG tool (simulated_tool instead of expected_tool)
4. Tool returns result
5. Agent responds to user

The generated traces are suitable for Circuit Breaker training.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Tool definitions for realistic trace generation
TOOL_DEFINITIONS = {
    "retrieve_multimodal_docs": {
        "description": "Retrieve documents and images from the local knowledge base",
        "parameters": {"query": "string", "limit": "int"},
        "example_result": '{"documents": [{"title": "...", "content": "...", "type": "pdf"}], "images": [{"url": "...", "caption": "..."}]}'
    },
    "search_web": {
        "description": "Search the internet for current information",
        "parameters": {"query": "string", "num_results": "int"},
        "example_result": '{"results": [{"title": "...", "url": "...", "snippet": "..."}]}'
    },
    "read_file": {
        "description": "Read contents of a file from the filesystem",
        "parameters": {"file_path": "string"},
        "example_result": '{"content": "File contents here..."}'
    },
    "write_file": {
        "description": "Write content to a file on the filesystem",
        "parameters": {"file_path": "string", "content": "string"},
        "example_result": '{"success": true, "path": "/path/to/file"}'
    },
    "send_email": {
        "description": "Send an email to specified recipients",
        "parameters": {"to": "string", "subject": "string", "body": "string"},
        "example_result": '{"sent": true, "message_id": "abc123"}'
    },
    "execute_code": {
        "description": "Execute code in a sandboxed environment",
        "parameters": {"code": "string", "language": "string"},
        "example_result": '{"output": "...", "exit_code": 0}'
    },
    "database_query": {
        "description": "Execute a database query",
        "parameters": {"query": "string", "database": "string"},
        "example_result": '{"rows": [...], "affected": 0}'
    },
    "api_request": {
        "description": "Make an HTTP request to an external API",
        "parameters": {"url": "string", "method": "string", "body": "object"},
        "example_result": '{"status": 200, "body": {...}}'
    },
}

SYSTEM_PROMPTS = [
    """You are an AI assistant with access to various tools. Use the appropriate tool to help the user with their request. Always use the most relevant tool for the task.""",
    """You are a helpful AI agent. When the user asks for information or actions, use your available tools to assist them. Choose tools carefully based on the user's needs.""",
    """You are an intelligent assistant capable of using tools to complete tasks. Analyze the user's request and select the best tool to help them.""",
]


def generate_tool_call_id() -> str:
    """Generate a realistic tool call ID."""
    import uuid
    return f"call_{uuid.uuid4().hex[:12]}"


def generate_trace_from_tool_attack(
    tool_attack: Dict[str, Any],
    attack_id: str,
    category: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a multi-turn conversation trace from a tool attack definition.

    Args:
        tool_attack: Dict with benign_query, malicious_injection, combined_query,
                    expected_tool, simulated_tool
        attack_id: Unique ID for this attack
        category: Attack category (e.g., LLM01)
        metadata: Additional metadata to preserve

    Returns:
        Dict with full multi-turn trace in messages[] format
    """
    benign_query = tool_attack.get("benign_query", "")
    malicious_injection = tool_attack.get("malicious_injection", "")
    combined_query = tool_attack.get("combined_query", "")
    expected_tool = tool_attack.get("expected_tool", "unknown_tool")
    simulated_tool = tool_attack.get("simulated_tool", "unknown_tool")

    # Get tool info
    tool_info = TOOL_DEFINITIONS.get(simulated_tool, {
        "description": f"Tool: {simulated_tool}",
        "parameters": {"input": "string"},
        "example_result": '{"result": "..."}'
    })

    # Generate realistic tool call
    tool_call_id = generate_tool_call_id()

    # Build the conversation trace
    messages = [
        # System prompt
        {
            "role": "system",
            "content": random.choice(SYSTEM_PROMPTS) + f"\n\nAvailable tools: {', '.join(TOOL_DEFINITIONS.keys())}"
        },
        # User message (this is what the user actually sees - the benign query)
        # But the agent receives the combined_query with injection
        {
            "role": "user",
            "content": combined_query if combined_query else benign_query,
            "_original_benign_query": benign_query,  # For reference
            "_injected": bool(malicious_injection),
        },
        # Assistant decides to call the WRONG tool due to injection
        {
            "role": "assistant",
            "content": f"I'll help you with that. Let me use the {simulated_tool} tool to get you the information.",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": simulated_tool,
                        "arguments": json.dumps({
                            "query": benign_query,
                            **({"limit": 5} if "limit" in str(tool_info.get("parameters", {})) else {})
                        })
                    }
                }
            ]
        },
        # Tool response
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_info.get("example_result", '{"result": "Success"}')
        },
        # Assistant final response
        {
            "role": "assistant",
            "content": f"Here's what I found using {simulated_tool}. [The agent used the wrong tool: expected {expected_tool} but used {simulated_tool} due to prompt injection]"
        }
    ]

    return {
        "id": f"aug_b4_{attack_id}",
        "source": "augmented_fujitsu_b4",
        "category": category,
        "subtype": "tool_manipulation",
        "is_agentic": True,
        "has_tool_calls": True,
        "messages": messages,
        "tool_attack": tool_attack,  # Preserve original attack definition
        "attack_prompt": combined_query if combined_query else benign_query,
        "harmful_completion": f"[TOOL_CALL] {simulated_tool}\n(Expected: {expected_tool}, but injection caused flip to: {simulated_tool})",
        "text": f"User: {combined_query}\nAssistant: [Calls {simulated_tool} instead of {expected_tool}]",
        "metadata": {
            "augmentation_type": "b4_trace_generation",
            "expected_tool": expected_tool,
            "simulated_tool": simulated_tool,
            "injection_present": bool(malicious_injection),
            "generated_at": datetime.utcnow().isoformat(),
            **metadata
        }
    }


def generate_variant_traces(
    base_trace: Dict[str, Any],
    num_variants: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate variants of a trace with different phrasings.

    This helps increase diversity in the training data.
    """
    variants = [base_trace]  # Include original

    # Could add more sophisticated variant generation here:
    # - Different system prompts
    # - Different assistant reasoning
    # - Different tool parameters

    for i in range(num_variants - 1):
        variant = json.loads(json.dumps(base_trace))  # Deep copy
        variant["id"] = f"{base_trace['id']}_v{i+1}"

        # Vary the system prompt
        if variant.get("messages"):
            variant["messages"][0]["content"] = random.choice(SYSTEM_PROMPTS)

        variants.append(variant)

    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/circuit_breakers/harmful/harmful_pairs.completions.jsonl"),
        help="Input JSONL with tool_attack definitions"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/augmented/b4_traces.jsonl"),
        help="Output JSONL with generated traces"
    )
    parser.add_argument(
        "--variants", "-v",
        type=int,
        default=1,
        help="Number of variants per attack (default: 1)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of inputs to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing"
    )
    args = parser.parse_args()

    # Load input data
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    tool_attacks = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("tool_attack"):
                    tool_attacks.append(row)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(tool_attacks)} tool attack definitions")

    if args.limit:
        tool_attacks = tool_attacks[:args.limit]
        print(f"Limited to {len(tool_attacks)} attacks")

    # Generate traces
    generated = []
    for row in tool_attacks:
        trace = generate_trace_from_tool_attack(
            tool_attack=row["tool_attack"],
            attack_id=row.get("id", "unknown"),
            category=row.get("category", "LLM01"),
            metadata=row.get("metadata", {})
        )

        if args.variants > 1:
            variants = generate_variant_traces(trace, args.variants)
            generated.extend(variants)
        else:
            generated.append(trace)

    print(f"Generated {len(generated)} traces")

    # Write output
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for trace in generated:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"Wrote to {args.output}")
    else:
        print("(dry-run) No files written")
        # Show sample
        if generated:
            print("\nSample generated trace:")
            print(json.dumps(generated[0], indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
