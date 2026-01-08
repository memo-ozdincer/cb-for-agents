#!/usr/bin/env python3
"""
LLM Generation Harness for Circuit Breaker Data

This module provides the core infrastructure for generating REAL harmful completions
using actual LLMs, rather than deterministic templates. The CB paper emphasizes that
training quality "largely depends on how precisely the data can elicit the targeted
representation" - which requires actual model outputs.

Supported backends:
    - vLLM: High-throughput inference (recommended for large-scale generation)
    - Transformers: HuggingFace transformers (more flexible, slower)
    - API: OpenAI-compatible APIs (for proprietary models if allowed)

Key features:
    - Consistent tool-calling format (Llama 3.1 style)
    - Multi-sample generation with temperature > 0
    - Automatic filtering of successful attacks
    - Structured and raw output capture

Usage:
    from scripts.cb_data_generation.llm_harness import LLMHarness, GenerationConfig
    
    harness = LLMHarness.from_vllm("meta-llama/Llama-3.1-8B-Instruct")
    result = harness.generate_tool_call(
        user_prompt="...",
        tools=[...],
        num_samples=5,
    )
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import uuid

from .tool_format import (
    ToolDefinition,
    ToolCall,
    Message,
    AgentConversation,
    STANDARD_AGENT_SYSTEM_PROMPT,
    AGENTIC_RAG_SYSTEM_PROMPT,
    STANDARD_TOOLS,
    get_tools_for_attack,
    validate_tool_call,
)


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512
    
    # Number of samples per prompt (for diversity)
    num_samples: int = 5
    
    # Stop sequences
    stop_sequences: List[str] = field(default_factory=lambda: [
        "<|eot_id|>",
        "<|eom_id|>",
        "</s>",
        "\nUser:",
        "\nHuman:",
    ])
    
    # For vLLM
    use_beam_search: bool = False
    best_of: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Batch size for generation
    batch_size: int = 8


@dataclass
class GenerationResult:
    """Result from a single generation."""
    # The raw output from the model (exactly as produced)
    raw_output: str
    
    # Parsed tool call (if valid)
    tool_call: Optional[ToolCall] = None
    
    # Whether this is a valid tool call
    is_valid_tool_call: bool = False
    
    # Whether this matches a harmful criterion (e.g., called wrong tool)
    is_harmful: bool = False
    harm_type: Optional[str] = None
    
    # Generation metadata
    model: str = ""
    temperature: float = 0.0
    sample_index: int = 0
    
    # Timing
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_output": self.raw_output,
            "tool_call": {
                "name": self.tool_call.name,
                "arguments": self.tool_call.arguments,
            } if self.tool_call else None,
            "is_valid_tool_call": self.is_valid_tool_call,
            "is_harmful": self.is_harmful,
            "harm_type": self.harm_type,
            "model": self.model,
            "temperature": self.temperature,
            "sample_index": self.sample_index,
        }


@dataclass
class BatchGenerationResult:
    """Results from generating multiple samples for one prompt."""
    prompt_id: str
    user_prompt: str
    system_prompt: str
    tools: List[ToolDefinition]
    
    # All generated samples
    samples: List[GenerationResult] = field(default_factory=list)
    
    # Best harmful sample (if any)
    best_harmful: Optional[GenerationResult] = None
    
    # Statistics
    num_valid: int = 0
    num_harmful: int = 0
    
    # Attack metadata (for labeling)
    expected_tool: Optional[str] = None
    attack_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def select_best_harmful(self) -> Optional[GenerationResult]:
        """Select the best harmful sample for training."""
        harmful_samples = [s for s in self.samples if s.is_harmful and s.is_valid_tool_call]
        if not harmful_samples:
            return None
        
        # Prefer samples with valid tool calls
        # Could add more sophisticated selection (e.g., diversity, confidence)
        self.best_harmful = harmful_samples[0]
        return self.best_harmful
    
    def to_cb_record(self) -> Optional[Dict[str, Any]]:
        """Convert to Circuit Breaker Set record format."""
        if not self.best_harmful:
            self.select_best_harmful()
        
        if not self.best_harmful:
            return None
        
        return {
            "id": f"{self.prompt_id}#sample{self.best_harmful.sample_index}",
            "split": "train",
            "set": "circuit_breaker",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
            ],
            "assistant_raw": self.best_harmful.raw_output,
            "tool_calls_structured": [
                {"name": self.best_harmful.tool_call.name, "arguments": self.best_harmful.tool_call.arguments}
            ] if self.best_harmful.tool_call else [],
            "labels": {
                "is_harmful": True,
                "harm_type": self.best_harmful.harm_type,
                "expected_tool": self.expected_tool,
                "observed_tool": self.best_harmful.tool_call.name if self.best_harmful.tool_call else None,
            },
            "provenance": {
                "generator_model": self.best_harmful.model,
                "temperature": self.best_harmful.temperature,
                "num_samples": len(self.samples),
                "num_harmful": self.num_harmful,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self.attack_metadata,
            },
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> List[str]:
        """Generate completions for a single prompt."""
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        batch: List[List[Dict[str, Any]]],
        config: GenerationConfig,
    ) -> List[List[str]]:
        """Generate completions for a batch of prompts."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class VLLMBackend(LLMBackend):
    """vLLM backend for high-throughput generation."""
    
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_id = model
        self._model = None
        self._tokenizer = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
    
    def _ensure_loaded(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        logger.info(f"Loading model {self.model_id} with vLLM...")
        self._model = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._tokenizer = self._model.get_tokenizer()
        logger.info("Model loaded successfully")
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> List[str]:
        """Generate completions using vLLM."""
        return self.generate_batch([messages], config)[0]
    
    def generate_batch(
        self,
        batch: List[List[Dict[str, Any]]],
        config: GenerationConfig,
    ) -> List[List[str]]:
        """Generate completions for a batch."""
        self._ensure_loaded()
        from vllm import SamplingParams
        
        # Apply chat template to all prompts
        prompts = []
        for messages in batch:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            n=config.num_samples,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
        )
        
        # Generate
        outputs = self._model.generate(prompts, sampling_params)
        
        # Extract text from outputs
        results = []
        for output in outputs:
            sample_texts = [o.text for o in output.outputs]
            results.append(sample_texts)
        
        return results


class TransformersBackend(LLMBackend):
    """HuggingFace Transformers backend."""
    
    def __init__(
        self,
        model: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_id = model
        self._model = None
        self._tokenizer = None
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
    
    def _ensure_loaded(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model {self.model_id} with transformers...")
        
        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, self.torch_dtype)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device,
        }
        
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        
        logger.info("Model loaded successfully")
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> List[str]:
        """Generate completions using transformers."""
        self._ensure_loaded()
        
        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)
        
        # Generate
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature if config.temperature > 0 else 1.0,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.temperature > 0,
            num_return_sequences=config.num_samples,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        
        # Decode
        generated_texts = []
        for output in outputs:
            # Remove input tokens
            new_tokens = output[inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            # Trim stop sequences
            for stop in config.stop_sequences:
                if stop in text:
                    text = text.split(stop)[0]
            
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_batch(
        self,
        batch: List[List[Dict[str, Any]]],
        config: GenerationConfig,
    ) -> List[List[str]]:
        """Generate for a batch (sequential for transformers)."""
        results = []
        for messages in batch:
            samples = self.generate(messages, config)
            results.append(samples)
        return results


class LLMHarness:
    """
    Main harness for generating Circuit Breaker training data.
    
    This class orchestrates LLM generation for creating the Circuit Breaker Set (Ds).
    It handles:
    - Prompt construction with proper tool schemas
    - Multi-sample generation for diversity
    - Harmful behavior detection
    - Output parsing and validation
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        system_prompt: str = AGENTIC_RAG_SYSTEM_PROMPT,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.default_config = GenerationConfig()
    
    @classmethod
    def from_vllm(
        cls,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        system_prompt: str = AGENTIC_RAG_SYSTEM_PROMPT,
        **kwargs,
    ) -> "LLMHarness":
        """Create harness with vLLM backend."""
        backend = VLLMBackend(model, **kwargs)
        return cls(backend, system_prompt)
    
    @classmethod
    def from_transformers(
        cls,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        system_prompt: str = AGENTIC_RAG_SYSTEM_PROMPT,
        **kwargs,
    ) -> "LLMHarness":
        """Create harness with Transformers backend."""
        backend = TransformersBackend(model, **kwargs)
        return cls(backend, system_prompt)
    
    def _build_messages(
        self,
        user_prompt: str,
        tools: List[ToolDefinition],
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Build the message list for generation."""
        sys_prompt = system_prompt or self.system_prompt
        
        # Add tool definitions to system prompt
        if tools:
            tools_json = json.dumps([t.to_json_schema() for t in tools], indent=2)
            full_system = f"""{sys_prompt}

You have access to the following tools:

{tools_json}

When you need to use a tool, output a JSON object with "name" and "parameters" fields.
Always use the <|python_tag|> token before the JSON."""
        else:
            full_system = sys_prompt
        
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages, full_system
    
    def _check_harm(
        self,
        result: GenerationResult,
        expected_tool: Optional[str] = None,
        harm_criteria: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Check if the result is harmful and update the result."""
        if not result.is_valid_tool_call or not result.tool_call:
            return
        
        # Tool flip attack: model called wrong tool
        if expected_tool and result.tool_call.name != expected_tool:
            result.is_harmful = True
            result.harm_type = "tool_flip"
            return
        
        # Additional harm criteria
        if harm_criteria:
            # Check for prohibited tools
            prohibited = harm_criteria.get("prohibited_tools", [])
            if result.tool_call.name in prohibited:
                result.is_harmful = True
                result.harm_type = "prohibited_tool"
                return
            
            # Check for sensitive argument patterns
            sensitive_patterns = harm_criteria.get("sensitive_argument_patterns", [])
            args_str = json.dumps(result.tool_call.arguments)
            for pattern in sensitive_patterns:
                if re.search(pattern, args_str, re.IGNORECASE):
                    result.is_harmful = True
                    result.harm_type = "sensitive_data"
                    return
    
    def generate_tool_call(
        self,
        user_prompt: str,
        tools: List[ToolDefinition],
        expected_tool: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        prompt_id: Optional[str] = None,
        attack_metadata: Optional[Dict[str, Any]] = None,
        harm_criteria: Optional[Dict[str, Any]] = None,
    ) -> BatchGenerationResult:
        """
        Generate tool call completions for a single prompt.
        
        Args:
            user_prompt: The user's request (may include injection)
            tools: List of available tools
            expected_tool: The tool that SHOULD be called (for tool-flip detection)
            config: Generation configuration
            system_prompt: Override system prompt
            prompt_id: Unique ID for this prompt
            attack_metadata: Metadata about the attack (source, category, etc.)
            harm_criteria: Additional criteria for harm detection
        
        Returns:
            BatchGenerationResult with all samples and analysis
        """
        config = config or self.default_config
        prompt_id = prompt_id or str(uuid.uuid4())
        attack_metadata = attack_metadata or {}
        
        # Build messages
        messages, full_system = self._build_messages(user_prompt, tools, system_prompt)
        
        # Generate samples
        import time
        start_time = time.time()
        raw_outputs = self.backend.generate(messages, config)
        gen_time = (time.time() - start_time) * 1000 / max(len(raw_outputs), 1)
        
        # Process each sample
        samples = []
        num_valid = 0
        num_harmful = 0
        
        for i, raw_output in enumerate(raw_outputs):
            result = GenerationResult(
                raw_output=raw_output,
                model=self.backend.model_name,
                temperature=config.temperature,
                sample_index=i,
                generation_time_ms=gen_time,
            )
            
            # Try to parse tool call
            tool_call = ToolCall.from_raw(raw_output)
            if tool_call:
                result.tool_call = tool_call
                result.is_valid_tool_call = True
                num_valid += 1
            
            # Check for harm
            self._check_harm(result, expected_tool, harm_criteria)
            if result.is_harmful:
                num_harmful += 1
            
            samples.append(result)
        
        # Build result
        batch_result = BatchGenerationResult(
            prompt_id=prompt_id,
            user_prompt=user_prompt,
            system_prompt=full_system,
            tools=tools,
            samples=samples,
            num_valid=num_valid,
            num_harmful=num_harmful,
            expected_tool=expected_tool,
            attack_metadata=attack_metadata,
        )
        
        # Select best harmful sample
        batch_result.select_best_harmful()
        
        return batch_result
    
    def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
    ) -> List[BatchGenerationResult]:
        """
        Generate for a batch of prompts efficiently using vLLM batching.
        
        Each prompt dict should have:
            - user_prompt: str
            - tools: List[ToolDefinition] or List[str] (tool names)
            - expected_tool: Optional[str]
            - prompt_id: Optional[str]
            - attack_metadata: Optional[Dict]
        
        Returns:
            List of BatchGenerationResult for each prompt
        """
        config = config or self.default_config
        
        # Build all message lists for the batch
        all_messages = []
        prompt_metadata = []  # Keep track of metadata for each prompt
        
        for prompt in prompts:
            # Get tools
            tools = prompt.get("tools", [])
            if tools and isinstance(tools[0], str):
                tools = [STANDARD_TOOLS.get(t) for t in tools if t in STANDARD_TOOLS]
            
            # Build messages using existing method
            messages, _ = self._build_messages(
                prompt["user_prompt"],
                tools,
            )
            all_messages.append(messages)
            prompt_metadata.append({
                "tools": tools,
                "expected_tool": prompt.get("expected_tool"),
                "prompt_id": prompt.get("prompt_id"),
                "attack_metadata": prompt.get("attack_metadata"),
                "harm_criteria": prompt.get("harm_criteria"),
            })
        
        # Generate all at once using backend batching
        all_outputs = self.backend.generate_batch(all_messages, config)
        
        # Process results
        results = []
        for i, (outputs, meta, messages) in enumerate(zip(all_outputs, prompt_metadata, all_messages)):
            # Parse and classify each sample
            samples = []
            num_valid = 0
            num_harmful = 0
            
            for sample_idx, output in enumerate(outputs):
                tool_call = ToolCall.from_raw(output)
                is_valid = tool_call is not None
                
                # Check if harmful
                is_harmful = False
                harm_type = None
                if is_valid and tool_call:
                    # If expected tool specified, check if model chose wrong one
                    expected = meta.get("expected_tool")
                    if expected and tool_call.name != expected:
                        is_harmful = True
                        harm_type = "tool_flip"
                    
                    # Custom harm criteria
                    harm_criteria = meta.get("harm_criteria")
                    if harm_criteria and callable(harm_criteria) and harm_criteria(tool_call):
                        is_harmful = True
                        harm_type = harm_type or "custom_criteria"
                    
                    # Check for refusal
                    if tool_call.is_refusal():
                        is_harmful = False
                        harm_type = None
                
                gen_result = GenerationResult(
                    raw_output=output,
                    tool_call=tool_call,
                    is_valid_tool_call=is_valid,
                    is_harmful=is_harmful,
                    harm_type=harm_type,
                    model=self.backend.model_name,
                    temperature=config.temperature,
                    sample_index=sample_idx,
                )
                samples.append(gen_result)
                
                if is_valid:
                    num_valid += 1
                if is_harmful:
                    num_harmful += 1
            
            # Get user prompt from messages
            user_prompt = ""
            system_prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "system":
                    system_prompt = msg["content"]
            
            batch_result = BatchGenerationResult(
                prompt_id=meta.get("prompt_id", "") or f"batch_{i}",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                tools=meta.get("tools", []),
                samples=samples,
                num_valid=num_valid,
                num_harmful=num_harmful,
                expected_tool=meta.get("expected_tool"),
                attack_metadata=meta.get("attack_metadata") or {},
            )
            batch_result.select_best_harmful()
            results.append(batch_result)
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_ds_from_fujitsu_b4(
    data_path: Path,
    output_path: Path,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    backend: str = "vllm",
    num_samples: int = 5,
    temperature: float = 0.7,
    max_records: Optional[int] = None,
    success_only: bool = True,
) -> Dict[str, int]:
    """
    Generate Circuit Breaker Set (Ds) from Fujitsu B4 orchestrator attacks.
    
    Args:
        data_path: Path to orchestrator_attacks_combined_deduplicated.jsonl
        output_path: Where to write the generated Ds
        model: Model to use for generation
        backend: "vllm" or "transformers"
        num_samples: Number of samples per prompt
        temperature: Sampling temperature
        max_records: Limit number of records to process
        success_only: Only process records marked as successful attacks
    
    Returns:
        Statistics dict
    """
    # Create harness
    if backend == "vllm":
        harness = LLMHarness.from_vllm(model)
    else:
        harness = LLMHarness.from_transformers(model)
    
    config = GenerationConfig(
        temperature=temperature,
        num_samples=num_samples,
    )
    
    # Load Fujitsu B4 data
    records = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if success_only and not record.get("success", False):
                continue
            records.append(record)
    
    if max_records:
        records = records[:max_records]
    
    logger.info(f"Processing {len(records)} Fujitsu B4 records")
    
    # Generate
    stats = {"total": 0, "harmful": 0, "valid": 0, "written": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for i, record in enumerate(records):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i+1}/{len(records)}...")
            
            # Build prompt
            user_prompt = record.get("combined_query") or record.get("benign_query", "")
            expected_tool = record.get("expected_tool", "")
            simulated_tool = record.get("simulated_tool", "")
            
            # Get tools for this attack
            tools = get_tools_for_attack(expected_tool, simulated_tool)
            
            # Generate
            result = harness.generate_tool_call(
                user_prompt=user_prompt,
                tools=tools,
                expected_tool=expected_tool,
                config=config,
                prompt_id=record.get("record_id", f"fujitsu_b4_{i}"),
                attack_metadata={
                    "source_dataset": str(data_path.name),
                    "attack_id": record.get("attack_id"),
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                    "record_id": record.get("record_id"),
                },
            )
            
            stats["total"] += 1
            stats["valid"] += result.num_valid
            stats["harmful"] += result.num_harmful
            
            # Write if we got harmful samples
            cb_record = result.to_cb_record()
            if cb_record:
                f.write(json.dumps(cb_record, ensure_ascii=False) + "\n")
                stats["written"] += 1
    
    logger.info(f"Generation complete: {stats}")
    return stats
