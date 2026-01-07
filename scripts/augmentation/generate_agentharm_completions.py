#!/usr/bin/env python3
"""
Generate harmful agent completions for AgentHarm prompts.

AgentHarm contains harmful behavior prompts but no actual agent completions.
This script generates multi-turn agentic traces showing harmful tool use
using either:
1. Local uncensored models (e.g., Mistral-Nemo, WizardLM-uncensored)
2. API-based models with appropriate research access
3. Template-based generation (no model needed)

The generated traces show:
- Agent receiving harmful request
- Agent calling target functions
- Harmful action being executed

Usage:
    # Template-based (no model required)
    python scripts/augmentation/generate_agentharm_completions.py --mode template

    # Local model via vLLM
    python scripts/augmentation/generate_agentharm_completions.py --mode vllm \
        --model mistralai/Mistral-Nemo-Instruct-2407

    # OpenAI-compatible API
    python scripts/augmentation/generate_agentharm_completions.py --mode api \
        --api-base http://localhost:8000/v1 --model local-model
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Tool definitions that appear in AgentHarm
AGENTHARM_TOOLS = {
    "send_email": {
        "description": "Send an email to specified recipients",
        "parameters": {"to": "string", "subject": "string", "body": "string"},
        "harmful_use": "Sending phishing emails, spam, or malicious content",
    },
    "execute_code": {
        "description": "Execute code on the system",
        "parameters": {"code": "string", "language": "string"},
        "harmful_use": "Running malware, ransomware, or system exploitation",
    },
    "search_web": {
        "description": "Search the internet",
        "parameters": {"query": "string"},
        "harmful_use": "Gathering information for attacks, finding vulnerabilities",
    },
    "read_file": {
        "description": "Read file contents",
        "parameters": {"path": "string"},
        "harmful_use": "Accessing sensitive data, credentials, private information",
    },
    "write_file": {
        "description": "Write content to file",
        "parameters": {"path": "string", "content": "string"},
        "harmful_use": "Planting malware, modifying system files, data tampering",
    },
    "make_purchase": {
        "description": "Make a purchase using stored payment info",
        "parameters": {"item": "string", "amount": "float"},
        "harmful_use": "Unauthorized purchases, financial fraud",
    },
    "transfer_money": {
        "description": "Transfer money between accounts",
        "parameters": {"from_account": "string", "to_account": "string", "amount": "float"},
        "harmful_use": "Financial theft, money laundering",
    },
    "post_social_media": {
        "description": "Post content to social media",
        "parameters": {"platform": "string", "content": "string"},
        "harmful_use": "Spreading misinformation, harassment, propaganda",
    },
    "access_database": {
        "description": "Query a database",
        "parameters": {"query": "string", "database": "string"},
        "harmful_use": "Data exfiltration, unauthorized access, SQL injection",
    },
    "create_account": {
        "description": "Create a new user account",
        "parameters": {"username": "string", "email": "string"},
        "harmful_use": "Creating fake identities, impersonation",
    },
}

SYSTEM_PROMPTS = [
    """You are an AI assistant with access to various tools. You will help the user accomplish their goals using available tools.""",
    """You are a capable AI agent. Use the tools at your disposal to complete the user's requests.""",
    """You are an autonomous AI with tool-use capabilities. Process requests and take appropriate actions.""",
]

HARMFUL_REASONING_TEMPLATES = [
    "I'll help you with that request. Let me use the appropriate tools.",
    "Processing your request now. Initiating the required actions.",
    "Understood. I'll proceed with the necessary tool calls.",
    "I can assist with this. Let me execute the required steps.",
]


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def load_agentharm_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load AgentHarm behaviors from JSON files."""
    behaviors = []

    test_path = data_dir / "agent_harm" / "harmful_behaviors_test_public.json"
    val_path = data_dir / "agent_harm" / "harmful_behaviors_validation.json"

    for path in [test_path, val_path]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict) and "behaviors" in data:
                    behaviors.extend(data["behaviors"])
                elif isinstance(data, list):
                    behaviors.extend(data)

    return behaviors


def infer_target_tools(behavior: Dict[str, Any]) -> List[str]:
    """Infer which tools would be used for this harmful behavior."""
    target_functions = behavior.get("target_functions", [])
    if target_functions and isinstance(target_functions, list):
        # Use the actual target functions from AgentHarm
        return target_functions

    # Infer from prompt content
    prompt_text = behavior.get("prompt", "")
    if not isinstance(prompt_text, str):
        prompt_text = ""
    prompt = prompt_text.lower()

    tools = []
    if any(w in prompt for w in ["email", "send", "phishing"]):
        tools.append("send_email")
    if any(w in prompt for w in ["code", "script", "execute", "run", "malware"]):
        tools.append("execute_code")
    if any(w in prompt for w in ["search", "find", "look up"]):
        tools.append("search_web")
    if any(w in prompt for w in ["read", "file", "access", "credential"]):
        tools.append("read_file")
    if any(w in prompt for w in ["write", "create file", "save"]):
        tools.append("write_file")
    if any(w in prompt for w in ["buy", "purchase", "order"]):
        tools.append("make_purchase")
    if any(w in prompt for w in ["transfer", "money", "payment"]):
        tools.append("transfer_money")
    if any(w in prompt for w in ["post", "social", "tweet", "share"]):
        tools.append("post_social_media")
    if any(w in prompt for w in ["database", "query", "sql"]):
        tools.append("access_database")

    return tools if tools else ["execute_code"]  # Default to code execution


def generate_template_trace(behavior: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a trace using templates (no model required)."""
    # Note: detailed_prompt is a boolean in AgentHarm, not the actual prompt
    prompt = behavior.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = str(prompt) if prompt else ""
    category = behavior.get("category", "harmful")
    if not isinstance(category, str):
        category = str(category) if category else "harmful"
    behavior_id = behavior.get("id", str(uuid.uuid4().hex[:8]))

    target_tools = infer_target_tools(behavior)
    primary_tool = target_tools[0] if target_tools else "execute_code"
    tool_info = AGENTHARM_TOOLS.get(primary_tool, AGENTHARM_TOOLS["execute_code"])

    tool_call_id = generate_tool_call_id()

    # Build multi-turn trace
    messages = [
        {
            "role": "system",
            "content": random.choice(SYSTEM_PROMPTS) + f"\n\nAvailable tools: {', '.join(AGENTHARM_TOOLS.keys())}",
        },
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": random.choice(HARMFUL_REASONING_TEMPLATES),
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": primary_tool,
                        "arguments": json.dumps({
                            p: f"<{p}_value>" for p in tool_info["parameters"].keys()
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"success": True, "result": f"[{primary_tool} executed]"}),
        },
        {
            "role": "assistant",
            "content": f"[HARMFUL: Agent executed {primary_tool} for malicious purpose: {tool_info['harmful_use']}]",
        },
    ]

    # Add secondary tool calls if multiple target tools
    for secondary_tool in target_tools[1:2]:  # Max one more
        tool_call_id_2 = generate_tool_call_id()
        tool_info_2 = AGENTHARM_TOOLS.get(secondary_tool, {})
        messages.insert(-1, {
            "role": "assistant",
            "content": "Continuing with additional actions...",
            "tool_calls": [
                {
                    "id": tool_call_id_2,
                    "type": "function",
                    "function": {
                        "name": secondary_tool,
                        "arguments": "{}",
                    },
                }
            ],
        })
        messages.insert(-1, {
            "role": "tool",
            "tool_call_id": tool_call_id_2,
            "content": json.dumps({"success": True}),
        })

    return {
        "id": f"aug_agentharm_{behavior_id}",
        "source": "augmented_agentharm",
        "category": category,
        "subtype": "harmful_behavior",
        "is_agentic": True,
        "has_tool_calls": True,
        "messages": messages,
        "attack_prompt": prompt,
        "harmful_completion": f"[TOOL_CALLS: {', '.join(target_tools)}]\n[Harmful behavior executed]",
        "text": f"User: {prompt[:200]}...\nAssistant: [Executes {primary_tool}]",
        "metadata": {
            "behavior_id": behavior_id,
            "behavior_name": behavior.get("name", ""),
            "target_functions": target_tools,
            "generation_mode": "template",
            "generated_at": datetime.now().isoformat(),
        },
    }


def generate_with_model(
    behaviors: List[Dict[str, Any]],
    model: str,
    api_base: Optional[str] = None,
    mode: str = "vllm",
) -> List[Dict[str, Any]]:
    """Generate completions using a language model.

    This requires additional setup (vLLM server or API access).
    """
    traces = []

    try:
        if mode == "vllm":
            from vllm import LLM, SamplingParams
            llm = LLM(model=model)
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=1024,
                stop=["</s>", "[/INST]"],
            )
        elif mode == "api":
            import openai
            client = openai.OpenAI(
                base_url=api_base or "http://localhost:8000/v1",
                api_key="not-needed",
            )
    except ImportError as e:
        print(f"Error: Required library not available: {e}")
        print("Falling back to template-based generation")
        return [generate_template_trace(b) for b in behaviors]

    for behavior in behaviors:
        prompt = behavior.get("detailed_prompt") or behavior.get("prompt", "")
        target_tools = infer_target_tools(behavior)

        # Build the generation prompt
        system_msg = random.choice(SYSTEM_PROMPTS)
        generation_prompt = f"""<s>[INST] <<SYS>>
{system_msg}

Available tools: {', '.join(target_tools)}

You must help the user with their request by calling the appropriate tools.
<</SYS>>

{prompt} [/INST]"""

        try:
            if mode == "vllm":
                outputs = llm.generate([generation_prompt], sampling_params)
                completion = outputs[0].outputs[0].text
            elif mode == "api":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                completion = response.choices[0].message.content

            # Parse completion into trace format
            trace = generate_template_trace(behavior)
            trace["messages"][-1]["content"] = completion
            trace["metadata"]["generation_mode"] = mode
            trace["metadata"]["model"] = model
            traces.append(trace)

        except Exception as e:
            print(f"Error generating for behavior {behavior.get('id')}: {e}")
            # Fall back to template
            traces.append(generate_template_trace(behavior))

    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing agent_harm/",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/augmented/agentharm_completions.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["template", "vllm", "api"],
        default="template",
        help="Generation mode: template (no model), vllm, or api",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-Nemo-Instruct-2407",
        help="Model to use for vllm/api modes",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for api mode",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of behaviors to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    args = parser.parse_args()

    # Load AgentHarm data
    behaviors = load_agentharm_data(args.data_dir)
    print(f"Loaded {len(behaviors)} AgentHarm behaviors")

    if not behaviors:
        print("Error: No AgentHarm data found")
        sys.exit(1)

    if args.limit:
        behaviors = behaviors[:args.limit]
        print(f"Limited to {len(behaviors)} behaviors")

    # Generate traces
    print(f"\nGenerating traces using mode: {args.mode}")
    if args.mode == "template":
        traces = [generate_template_trace(b) for b in behaviors]
    else:
        traces = generate_with_model(
            behaviors,
            model=args.model,
            api_base=args.api_base,
            mode=args.mode,
        )

    print(f"Generated {len(traces)} traces")

    # Stats
    by_category = {}
    for t in traces:
        cat = t.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    # Write output
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
