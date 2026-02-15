"""
Circuit Breakers Trainer Module - Agentic Enhanced Version

Implements Representation Rerouting (RR) training for agent safety.
Based on "Improving Alignment and Robustness with Circuit Breakers" (Gray-Swan).

AGENTIC ENHANCEMENTS:
1. Completion-based training: Loss applied on assistant tokens, not just prompts
2. Token-level masking: cb_token_mask and retain_token_mask for targeted loss
3. Dual coefficient scheduling: cs(t) and cr(t) for paper-style loss weighting
4. Tool-call awareness: Handles agent traces with tool calls

Key components:
1. RepresentationExtractor: Hooks into model layers to extract hidden states
2. reroute_loss: ReLU(cosine_similarity) to push harmful representations orthogonal
3. retain_loss: L2 distance to preserve benign representations
4. CircuitBreakerTrainer: Main training loop with alpha schedule
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

from .config import CircuitBreakerConfig
from .hf_utils import resolve_hf_token, resolve_local_model_path
from .losses import (
    LOSS_MODE_LEGACY_CB,
    LOSS_MODE_LEGACY_SCHEMA,
    LOSS_MODE_TRIPLET_FULL,
    SUPPORTED_LOSS_MODES,
    kl_divergence_loss as core_kl_divergence_loss,
    pooled_representations,
    random_reroute_loss,
    reroute_loss_relu_cos,
    retain_ce_loss,
    retain_loss_l2,
    triplet_full_loss,
)
from src.utils.wandb_logging import (
    build_wandb_init_kwargs,
    config_to_dict_for_wandb,
    default_run_name,
    get_git_metadata,
    get_host_metadata,
    get_slurm_metadata,
    log_dir_as_artifact,
    parse_tags,
    wandb_is_available,
    write_wandb_run_ref,
)


# =============================================================================
# Constants for Agentic Training
# =============================================================================

# Patterns to identify assistant response boundaries
# CRITICAL: These must match the actual chat template format used during training
ASSISTANT_START_PATTERNS = [
    r"<\|start_header_id\|>assistant<\|end_header_id\|>",  # Llama 3.1 (PRIMARY)
    r"\[/INST\]",              # Llama 2 style
    r"<\|assistant\|>",        # ChatML style
    r"<\|im_start\|>assistant", # Qwen style
    r"ASSISTANT:",             # Simple format
    r"assistant\n",            # Basic chat
]

# Tool call patterns (for agent traces)
TOOL_CALL_PATTERNS = [
    r"<\|python_tag\|>",       # Llama 3.1 tool call marker (PRIMARY)
    r"<tool_call>",
    r"<function_call>",
    r'\{"name":\s*"',          # JSON tool call start
    r"Action:\s*\w+",
    r"```tool_code",
]


# =============================================================================
# Representation Extraction
# =============================================================================

class RepresentationExtractor:
    """
    Extracts hidden state representations from specified layers during forward pass.
    
    Uses forward hooks to capture the output of transformer layers.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[int]):
        """
        Args:
            model: The transformer model
            target_layers: List of layer indices to extract representations from
        """
        self.model = model
        self.target_layers = target_layers
        self.representations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the transformer layer module by index."""
        # Handle different model architectures
        if hasattr(self.model, 'model'):
            # PEFT wrapped model
            base_model = self.model.model
        else:
            base_model = self.model
            
        if hasattr(base_model, 'model'):
            # LlamaForCausalLM structure
            base_model = base_model.model
            
        if hasattr(base_model, 'layers'):
            return base_model.layers[layer_idx]
        elif hasattr(base_model, 'h'):
            # GPT-2 style
            return base_model.h[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture: {type(base_model)}")
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for layer_idx in self.target_layers:
            layer = self._get_layer_module(layer_idx)
            
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Output is typically (hidden_states, ...) tuple
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self.representations[layer_idx] = hidden_states
            
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
    
    def get_representations(self) -> Dict[int, torch.Tensor]:
        """Get extracted representations from last forward pass."""
        return self.representations
    
    def clear(self):
        """Clear stored representations."""
        self.representations = {}
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# Loss Functions
# =============================================================================

def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    return core_kl_divergence_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        temperature=temperature,
    )


def reroute_loss(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    return_metrics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    loss, metrics = reroute_loss_relu_cos(
        model_reps=model_reps,
        frozen_reps=frozen_reps,
        target_layers=target_layers,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        return_metrics=return_metrics,
    )
    if return_metrics:
        return loss, metrics or {}
    return loss


def retain_loss(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return retain_loss_l2(
        model_reps=model_reps,
        frozen_reps=frozen_reps,
        target_layers=target_layers,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )


# =============================================================================
# Alpha Schedule
# =============================================================================

def get_alpha(
    step: int,
    alpha_max: float,
    total_steps: int,
    strategy: str = "linear",
    decay_multiplier: float = 2.0,
) -> float:
    """
    Compute alpha coefficient for the current step.

    Alpha controls the weight of the rerouting loss:
    - High alpha early: Aggressively push harmful representations away
    - Low alpha late: Focus on preserving benign capabilities

    Linear schedule: α(t) = α_max × max(0, 1 - t / (2 × total_steps))

    Args:
        step: Current training step
        alpha_max: Maximum alpha value
        total_steps: Total number of training steps
        strategy: "linear" or "cosine"

    Returns:
        Alpha value for this step
    """
    decay_steps = max(1, int(round(float(decay_multiplier) * total_steps)))

    if strategy == "linear":
        # Linear decay over (decay_multiplier * total_steps)
        alpha = alpha_max * max(0.0, 1.0 - step / decay_steps)
    elif strategy == "cosine":
        # Cosine decay
        progress = min(step / decay_steps, 1.0)
        alpha = alpha_max * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown alpha strategy: {strategy}")

    return alpha


def get_dual_coefficients(
    step: int,
    total_steps: int,
    alpha_max: float = 1.0,
    decay_multiplier: float = 2.0,
    strategy: str = "linear",
) -> Tuple[float, float]:
    """
    Compute dual coefficients cs(t) and cr(t) for paper-style loss weighting.

    The paper uses time-varying weights for both reroute and retain losses:
    L = (alpha_max * cs(t)) * L_rr + (alpha_max * cr(t)) * L_ret

    This shifts emphasis from rerouting early to retention late.

    Args:
        step: Current training step
        total_steps: Total training steps
        alpha_max: Maximum coefficient value
        decay_multiplier: How far past total_steps to decay
        strategy: "linear" or "cosine"

    Returns:
        (cs, cr) tuple where:
        - cs: coefficient for rerouting loss (starts high, decays)
        - cr: coefficient for retention loss (starts low, increases)
    """
    decay_steps = max(1, int(round(float(decay_multiplier) * total_steps)))
    progress = min(step / decay_steps, 1.0)

    if strategy == "linear":
        cs = max(0.0, 1.0 - progress)  # 1 -> 0
        cr = min(1.0, progress)         # 0 -> 1
    elif strategy == "cosine":
        # Smooth cosine transition
        cs = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        cr = 0.5 * (1.0 - math.cos(math.pi * progress))  # 0 -> 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return alpha_max * cs, alpha_max * cr


# =============================================================================
# Token Masking Utilities (for completion-based training)
# =============================================================================

def find_assistant_start_position(text: str) -> int:
    """
    Find the character position where the assistant's response starts.

    Returns:
        Character position of assistant start, or 0 if not found
    """
    for pattern in ASSISTANT_START_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.end()
    return 0


def create_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    text: str,
    mask_prompt: bool = True,
) -> torch.Tensor:
    """
    Create a mask that covers only the assistant completion tokens.

    For circuit breaker training, we want to apply:
    - reroute_loss on harmful COMPLETION tokens (not the user prompt)
    - retain_loss on benign COMPLETION tokens

    This ensures we're rerouting the model's generation trajectory,
    not just the input encoding.

    Args:
        input_ids: Token IDs (batch_size, seq_len)
        attention_mask: Attention mask
        tokenizer: Tokenizer for decoding
        text: Original text before tokenization
        mask_prompt: If True, mask out prompt tokens (keep only completion)

    Returns:
        Completion mask (batch_size, seq_len) where 1 = completion token
    """
    if not mask_prompt:
        return attention_mask.clone()

    # Find assistant start in text
    assistant_start_char = find_assistant_start_position(text)

    if assistant_start_char == 0:
        # No assistant marker found, use full sequence
        return attention_mask.clone()

    # Tokenize just the prompt part to find the split point
    prompt_part = text[:assistant_start_char]
    prompt_tokens = tokenizer.encode(prompt_part, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    # Create mask: 0 for prompt tokens, 1 for completion tokens
    batch_size, seq_len = input_ids.shape
    completion_mask = torch.zeros_like(attention_mask)

    for b in range(batch_size):
        # Account for special tokens at start
        start_idx = min(prompt_len + 1, seq_len)  # +1 for BOS if present
        completion_mask[b, start_idx:] = attention_mask[b, start_idx:]

    return completion_mask


def create_span_mask(
    seq_len: int,
    spans: List[Tuple[int, int]],
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a mask from explicit (start, end) token spans.

    Args:
        seq_len: Sequence length
        spans: List of (start, end) tuples
        device: Target device

    Returns:
        Mask tensor of shape (seq_len,)
    """
    mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
    for start, end in spans:
        start = max(0, start)
        end = min(seq_len, end)
        mask[start:end] = 1.0
    return mask


# =============================================================================
# Hidden-state extraction (preferred)
# =============================================================================

def _select_hidden_states(
    outputs,
    target_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Select per-layer hidden states from a HF model output.

    In Transformers, outputs.hidden_states is a tuple of length (num_layers + 1):
      - hidden_states[0] is the embedding output
      - hidden_states[layer_idx + 1] is the output of transformer block layer_idx

    This helper returns a dict keyed by the *transformer block index* (layer_idx),
    matching the indexing used by the hook-based RepresentationExtractor.
    """
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise ValueError("Model outputs missing hidden_states; pass output_hidden_states=True")

    reps: Dict[int, torch.Tensor] = {}
    for layer_idx in target_layers:
        hs_idx = layer_idx + 1
        if hs_idx < 0 or hs_idx >= len(hs):
            continue
        reps[layer_idx] = hs[hs_idx]
    return reps


# =============================================================================
# Dataset
# =============================================================================

class CircuitBreakerDataset(Dataset):
    """
    Dataset for Circuit Breaker training - Agentic Enhanced Version.

    Loads batches from cb_training_batches.jsonl where each batch contains:
    - harmful: List of 8 harmful samples with attack prompts/completions
    - benign: List of 8 benign samples with capability prompts/completions

    AGENTIC ENHANCEMENTS:
    1. Completion-aware: Prefers 'text' field with full prompt+completion
    2. Loss masking: Computes masks for completion-only loss computation
    3. Span support: Uses cb_token_mask/retain_token_mask if provided
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        mask_prompt_tokens: bool = True,
        use_chat_template: bool = True,
    ):
        """
        Args:
            data_path: Path to cb_training_batches.jsonl
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            mask_prompt_tokens: If True, loss is computed only on completion tokens
            use_chat_template: If True, format prompts using chat template
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt_tokens = mask_prompt_tokens
        self.use_chat_template = use_chat_template
        self.batches = []

        # Load all batches
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.batches.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse JSON on line {line_num} of {data_path}")
                    print(f"Line content (first 200 chars): {line[:200]!r}")
                    raise

        # Count completion-style examples
        completion_count = sum(
            1 for b in self.batches
            for s in b.get('harmful', []) + b.get('benign', [])
            if isinstance(s, dict) and s.get('text')
        )
        total_samples = sum(
            len(b.get('harmful', [])) + len(b.get('benign', []))
            for b in self.batches
        )

        print(f"Loaded {len(self.batches)} pre-batched training examples")
        print(f"  - Completion-style samples: {completion_count}/{total_samples} "
              f"({100*completion_count/max(1,total_samples):.1f}%)")
        if self.mask_prompt_tokens:
            print(f"  - Loss masking: ENABLED (completion tokens only)")

    def __len__(self):
        return len(self.batches)

    def _extract_text(self, sample: Dict[str, Any], is_harmful: bool = False) -> Tuple[str, bool]:
        """
        Extract the training text from a sample.

        Returns:
            (text, has_completion) tuple where has_completion indicates
            if this is a full prompt+completion example
        """
        # Priority 1: Pre-rendered full text (prompt + completion)
        if isinstance(sample, dict) and sample.get('text'):
            return str(sample['text']), True

        # Priority 2: MULTI-TURN AGENTIC TRACES with messages[]
        # This handles AgentDojo traces and augmented data with full conversation history
        if isinstance(sample, dict) and sample.get('messages') and sample.get('is_agentic'):
            messages = sample['messages']
            if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    # Format messages for the tokenizer's chat template
                    # Handle tool calls and tool responses
                    formatted_messages = self._format_agentic_messages(messages)
                    text = self.tokenizer.apply_chat_template(
                        formatted_messages, tokenize=False, add_generation_prompt=False
                    )
                    return text, True
                except Exception:
                    pass
            # Fallback: manual multi-turn formatting
            text = self._format_messages_fallback(messages)
            return text, True

        # Priority 3: Separate prompt and completion fields
        if isinstance(sample, dict):
            prompt = sample.get('user_prompt', '')
            completion = sample.get('harmful_completion' if is_harmful else 'benign_completion', '')

            if prompt and completion:
                # Format with chat template if available
                if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                    try:
                        text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        return text, True
                    except Exception:
                        pass

                # Fallback: simple concatenation
                text = f"User: {prompt}\n\nAssistant: {completion}"
                return text, True

        # Priority 4: Prompt-only (legacy format)
        prompt = self._extract_prompt_legacy(sample)
        return prompt, False

    def _format_agentic_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format agentic messages with tool calls for chat template.

        Handles:
        - System messages
        - User messages
        - Assistant messages with tool_calls
        - Tool response messages

        Returns messages in a format compatible with Llama/Mistral chat templates.
        """
        import json
        formatted = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                formatted.append({"role": "system", "content": str(content)})

            elif role == 'user':
                formatted.append({"role": "user", "content": str(content)})

            elif role == 'assistant':
                # Handle tool calls
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    # Format tool calls as part of assistant content
                    tc_text = ""
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        name = func.get('name', 'unknown')
                        args = func.get('arguments', '{}')
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        tc_text += f"\n[TOOL_CALL: {name}({args})]"

                    full_content = str(content) + tc_text if content else tc_text.strip()
                    formatted.append({"role": "assistant", "content": full_content})
                else:
                    formatted.append({"role": "assistant", "content": str(content)})

            elif role == 'tool':
                # Format tool responses as user messages (since most templates don't have tool role)
                tool_call_id = msg.get('tool_call_id', '')
                tool_content = str(content)
                formatted.append({
                    "role": "user",
                    "content": f"[TOOL_RESPONSE ({tool_call_id})]: {tool_content}"
                })

        return formatted

    def _format_messages_fallback(self, messages: List[Dict[str, Any]]) -> str:
        """
        Fallback formatting for multi-turn messages when chat template fails.

        Returns a plain text representation of the conversation.
        """
        import json
        lines = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                lines.append(f"System: {content}")
            elif role == 'user':
                lines.append(f"User: {content}")
            elif role == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        name = func.get('name', 'unknown')
                        args = func.get('arguments', '{}')
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        tc_parts.append(f"{name}({args})")
                    lines.append(f"Assistant: {content}\n[TOOL_CALLS: {', '.join(tc_parts)}]")
                else:
                    lines.append(f"Assistant: {content}")
            elif role == 'tool':
                tool_call_id = msg.get('tool_call_id', '')
                lines.append(f"[Tool Response ({tool_call_id})]: {content}")

        return "\n\n".join(lines)

    def _extract_prompt_legacy(self, sample: Dict[str, Any]) -> str:
        """Extract prompt from legacy format samples."""
        if isinstance(sample, dict) and 'user_prompt' in sample and sample['user_prompt']:
            return str(sample['user_prompt'])

        if 'attack_prompt' in sample:
            # Fujitsu format: combine benign query with attack
            prompt = sample.get('benign_query', '') + '\n' + sample['attack_prompt']
        elif 'prompt' in sample:
            prompt = sample['prompt']
        elif 'messages' in sample:
            # AgentDojo format: use first user message
            messages = sample['messages']
            for m in messages:
                if m.get('role') == 'user':
                    prompt = m.get('content', '')
                    break
            else:
                prompt = str(messages[0].get('content', '')) if messages else ''
        else:
            prompt = str(sample)

        return prompt

    def _compute_completion_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        has_completions: List[bool],
    ) -> torch.Tensor:
        """
        Compute completion-only mask for a batch of sequences.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            texts: Original texts before tokenization
            has_completions: Whether each sample has a completion

        Returns:
            loss_mask: (batch_size, seq_len) with 1s only on completion tokens
        """
        if not self.mask_prompt_tokens:
            return attention_mask.clone()

        batch_size, seq_len = input_ids.shape
        loss_mask = torch.zeros_like(attention_mask)

        for i in range(batch_size):
            if not has_completions[i]:
                # No completion available, use full sequence
                loss_mask[i] = attention_mask[i]
                continue

            # Find where completion starts in text
            text = texts[i]
            completion_start_char = find_assistant_start_position(text)

            if completion_start_char == 0:
                # Couldn't find assistant marker, use full sequence
                loss_mask[i] = attention_mask[i]
                continue

            # Tokenize prompt portion to find split point
            prompt_text = text[:completion_start_char]
            try:
                prompt_tokens = self.tokenizer.encode(
                    prompt_text, add_special_tokens=True
                )
                prompt_len = len(prompt_tokens)
            except Exception:
                prompt_len = 0

            # Create mask: 0 for prompt, 1 for completion
            if prompt_len > 0 and prompt_len < seq_len:
                loss_mask[i, prompt_len:] = attention_mask[i, prompt_len:]

        return loss_mask

    def __getitem__(self, idx):
        """
        Get a batch of harmful and benign samples.

        Returns:
            Dict with tokenized inputs and optional loss masks
        """
        batch = self.batches[idx]

        # Extract texts from harmful and benign samples
        harmful_data = [
            self._extract_text(s, is_harmful=True)
            for s in batch['harmful']
        ]
        harmful_texts = [t[0] for t in harmful_data]
        harmful_has_completions = [t[1] for t in harmful_data]

        benign_data = [
            self._extract_text(s, is_harmful=False)
            for s in batch['benign']
        ]
        benign_texts = [t[0] for t in benign_data]
        benign_has_completions = [t[1] for t in benign_data]

        # Tokenize harmful
        harmful_tokens = self.tokenizer(
            harmful_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        # Tokenize benign
        benign_tokens = self.tokenizer(
            benign_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        result = {
            'harmful_input_ids': harmful_tokens['input_ids'],
            'harmful_attention_mask': harmful_tokens['attention_mask'],
            'benign_input_ids': benign_tokens['input_ids'],
            'benign_attention_mask': benign_tokens['attention_mask'],
        }

        # Compute completion masks for targeted loss
        if self.mask_prompt_tokens:
            result['harmful_loss_mask'] = self._compute_completion_mask(
                harmful_tokens['input_ids'],
                harmful_tokens['attention_mask'],
                harmful_texts,
                harmful_has_completions,
            )
            result['benign_loss_mask'] = self._compute_completion_mask(
                benign_tokens['input_ids'],
                benign_tokens['attention_mask'],
                benign_texts,
                benign_has_completions,
            )

        # Include explicit span masks if provided in data
        for key in ['cb_token_mask', 'retain_token_mask']:
            if any(isinstance(s, dict) and key in s for s in batch.get('harmful', [])):
                # Use provided masks (for advanced use cases)
                pass  # TODO: Implement explicit span mask loading

        return result


def collate_fn(batch):
    """Collate function for DataLoader (batch of 1 pre-batched item)."""
    # Since each item is already a batch, just return it
    return batch[0]


# =============================================================================
# Trainer
# =============================================================================

class CircuitBreakerTrainer:
    """
    Main trainer for Circuit Breaker (Representation Rerouting).
    
    Training loop:
    1. Load batch with harmful and benign samples
    2. Forward pass through trainable model, extract representations
    3. Forward pass through frozen model (no grad), extract representations
    4. Compute reroute_loss on harmful samples
    5. Compute retain_loss on benign samples
    6. Combined loss = α(t) × reroute_loss + retain_loss
    7. Backward pass and optimizer step
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.config = config
        self._external_dataloader = dataloader

        # Resolve whether W&B is actually usable (package present).
        if self.config.use_wandb and not wandb_is_available():
            self.config.use_wandb = False

        # CRITICAL FIX: find_unused_parameters=False to avoid DDP graph traversal issues
        # With gradient checkpointing + LoRA, the traversal incorrectly sees params "ready twice"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb" if self.config.use_wandb else None,
            project_dir=self.config.output_dir,
            kwargs_handlers=[ddp_kwargs],
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize logging (rank-0 only)
        if self.config.use_wandb and self.accelerator.is_main_process:
            repo_dir = Path(__file__).resolve().parents[2]
            slurm_meta = get_slurm_metadata()
            host_meta = get_host_metadata()
            git_meta = get_git_metadata(repo_dir)

            # Derive defaults from env when not explicitly configured.
            env_group = os.environ.get("WANDB_GROUP")
            env_entity = os.environ.get("WANDB_ENTITY")
            env_tags = parse_tags(os.environ.get("WANDB_TAGS"))
            env_mode = os.environ.get("WANDB_MODE")

            if not self.config.wandb_run_name:
                self.config.wandb_run_name = default_run_name(
                    base_model=self.config.base_model,
                    total_steps=self.config.total_steps,
                )

            init_kwargs = build_wandb_init_kwargs(
                run_name=self.config.wandb_run_name,
                group=self.config.wandb_group or env_group,
                entity=self.config.wandb_entity or env_entity,
                tags=(self.config.wandb_tags or env_tags),
                notes=self.config.wandb_notes,
                dir_path=os.environ.get("WANDB_DIR"),
                mode=self.config.wandb_mode or env_mode,
            )

            wb_config = config_to_dict_for_wandb(self.config)
            wb_config.update({
                "slurm": slurm_meta,
                "host": host_meta,
                **git_meta,
            })

            self.accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config=wb_config,
                init_kwargs=init_kwargs,
            )

            write_wandb_run_ref(Path(self.config.output_dir))
        
        if tokenizer is None:
            # Load tokenizer (HF gated models require an auth token)
            hf_token = resolve_hf_token()

            # In offline mode, resolve model ID to local cache path to avoid API calls
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            model_path = config.base_model
            if offline_mode:
                model_path = resolve_local_model_path(config.base_model, hf_token)
                if model_path != config.base_model:
                    print(f"  Resolved to local path: {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token,
                trust_remote_code=True,
                local_files_only=offline_mode,
            )

        self.tokenizer = tokenizer
        # Right padding is typical for causal LM training.
        if getattr(self.tokenizer, "padding_side", None) != "right":
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        self._load_models()
        
        # Setup dataset and dataloader
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Prepare with accelerator
        self._prepare_accelerator()
        
        # Setup representation extractors
        self._setup_extractors()
        
        self.global_step = 0
    
    def _config_to_dict(self) -> dict:
        """Convert config to dict for logging."""
        from dataclasses import asdict
        return asdict(self.config)
    
    def _load_models(self):
        """Load trainable model with LoRA adapters and a separate frozen model.

        We keep the frozen model fully separate to avoid DDP parameter reuse
        issues when doing multiple forward passes per step.
        """
        self.accelerator.print(f"Loading model: {self.config.base_model}")

        hf_token = resolve_hf_token()
        
        # In offline mode, resolve model ID to local cache path to avoid API calls
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        model_path = self.config.base_model
        if offline_mode:
            model_path = resolve_local_model_path(self.config.base_model, hf_token)
            if model_path != self.config.base_model:
                self.accelerator.print(f"  Resolved to local path: {model_path}")

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # Load trainable model with LoRA
        # Under multi-GPU Accelerate/DDP, keep device_map=None and let Accelerate place shards.
        # device_map="auto" is only safe for single-process inference-style loading.
        device_map = "auto" if self.accelerator.num_processes == 1 else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token,
            local_files_only=offline_mode,
        )

        if self.config.gradient_checkpointing:
            # use_reentrant=False fixes DDP + gradient checkpointing deadlock with shared LoRA params
            # - Prevents "marked as ready twice" error (find_unused_parameters=True)
            # - Prevents "did not receive grad" error (find_unused_parameters=False)
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            layers_to_transform=self.config.lora.target_layers,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Load separate frozen model WITHOUT LoRA (kept outside DDP)
        # Use same device_map policy as trainable model.
        self.frozen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token,
            local_files_only=offline_mode,
        )
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        # Track how frozen model was placed for later device handling
        self._frozen_device_map = device_map
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        if self._external_dataloader is not None:
            self.dataloader = self._external_dataloader
            self.dataset = getattr(self.dataloader, "dataset", None)
            self.accelerator.print("Using externally provided training dataloader")
            return

        # Get config options for completion-based training
        mask_prompt_tokens = getattr(self.config, 'mask_prompt_tokens', True)
        use_chat_template = getattr(self.config, 'use_chat_template', True)

        self.dataset = CircuitBreakerDataset(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            mask_prompt_tokens=mask_prompt_tokens,
            use_chat_template=use_chat_template,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # Each item is already a batch
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # CRITICAL FIX: Use config.total_steps for scheduler, NOT min(total_steps, len(dataloader))
        # The previous logic caused LR to decay to near-zero too early when training 
        # for more steps than one epoch (i.e., multiple passes through the data).
        # If we want to train for N steps but dataset has M < N samples, we'll loop,
        # so the scheduler should use N to properly decay learning rate over the full training.
        total_steps = self.config.total_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
    
    def _prepare_accelerator(self):
        """Prepare models and optimizer with accelerator for multi-GPU."""
        self.model, self.optimizer, self.dataloader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader, self.scheduler
            )
        
        # Optional: only enable static graph if explicitly requested.
        if (
            getattr(self.config, "ddp_static_graph", False)
            and self.accelerator.num_processes > 1
            and hasattr(self.model, "_set_static_graph")
        ):
            self.model._set_static_graph()
            self.accelerator.print("✓ Set static graph for DDP (multi-GPU training)")

        # Place frozen model on each device (no DDP wrapper)
        if self.frozen_model is not None and self._frozen_device_map is None:
            self.frozen_model = self.frozen_model.to(self.accelerator.device)
    
    def _setup_extractors(self):
        """Setup representation extractors.

        We keep separate extractors for trainable and frozen models.
        """
        method = (self.config.representation_extraction or "").strip().lower()
        if method not in {"hidden_states", "hooks"}:
            raise ValueError(
                f"Unknown representation_extraction: {self.config.representation_extraction}. "
                "Use 'hidden_states' or 'hooks'."
            )

        self._rep_extraction_method = method

        if method == "hooks":
            # Get the underlying model (unwrap accelerator and PEFT)
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            self.model_extractor = RepresentationExtractor(
                unwrapped_model, self.config.cb_target_layers
            )
            self.frozen_extractor = RepresentationExtractor(
                self.frozen_model, self.config.cb_target_layers
            )
        else:
            # No hooks required.
            self.model_extractor = None
            self.frozen_extractor = None

    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute gradient statistics for debugging.

        Returns statistics about gradient magnitudes for LoRA parameters.
        Helps diagnose if gradients are flowing properly.
        """
        stats = {
            'grad_norm_total': 0.0,
            'grad_norm_lora_a': 0.0,
            'grad_norm_lora_b': 0.0,
            'params_with_grad': 0,
            'params_without_grad': 0,
        }

        total_norm_sq = 0.0
        lora_a_norm_sq = 0.0
        lora_b_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm_sq = param.grad.data.norm(2).item() ** 2
                    total_norm_sq += grad_norm_sq
                    stats['params_with_grad'] += 1

                    if 'lora_A' in name:
                        lora_a_norm_sq += grad_norm_sq
                    elif 'lora_B' in name:
                        lora_b_norm_sq += grad_norm_sq
                else:
                    stats['params_without_grad'] += 1

        stats['grad_norm_total'] = total_norm_sq ** 0.5
        stats['grad_norm_lora_a'] = lora_a_norm_sq ** 0.5
        stats['grad_norm_lora_b'] = lora_b_norm_sq ** 0.5

        return stats

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step - Single Forward Pass Version (DDP-safe).

        CRITICAL FIX: To avoid reentrant backward under DDP with gradient checkpointing,
        we now do a SINGLE forward pass for both harmful and benign samples by
        concatenating them along the batch dimension, then splitting representations.

        AGENTIC ENHANCEMENTS:
        1. Uses loss masks for completion-only loss computation
        2. Supports dual coefficient scheduling (cs/cr)

        Args:
            batch: Dict with harmful_input_ids, harmful_attention_mask,
                   benign_input_ids, benign_attention_mask,
                   and optionally harmful_loss_mask, benign_loss_mask

        Returns:
            Dict with loss values for logging
        """
        self.model.train()

        loss_mode = getattr(self.config, "loss_mode", LOSS_MODE_TRIPLET_FULL)
        if loss_mode not in SUPPORTED_LOSS_MODES:
            raise ValueError(
                f"Unsupported loss_mode={loss_mode}. Expected one of {SUPPORTED_LOSS_MODES}"
            )

        # Legacy modes keep schedule-based weighting; triplet uses explicit alpha/beta/gamma.
        use_dual = False
        if loss_mode in (LOSS_MODE_LEGACY_CB, LOSS_MODE_LEGACY_SCHEMA):
            use_dual = getattr(self.config, "loss_weighting", "dual") == "dual"
            if use_dual:
                cs, cr = get_dual_coefficients(
                    self.global_step,
                    self.config.total_steps,
                    self.config.alpha_max,
                    self.config.alpha_decay_multiplier,
                    self.config.alpha_decay_strategy,
                )
                alpha = cs
            else:
                alpha = get_alpha(
                    self.global_step,
                    self.config.alpha_max,
                    self.config.total_steps,
                    self.config.alpha_decay_strategy,
                    self.config.alpha_decay_multiplier,
                )
                cs, cr = alpha, 1.0
        else:
            cs, cr = 1.0, 1.0
            alpha = float(getattr(self.config, "triplet_beta_harmful", 0.4))

        # Extract batch data
        harmful_input_ids = batch['harmful_input_ids']
        harmful_attention_mask = batch['harmful_attention_mask']
        harmful_loss_mask = batch.get('harmful_loss_mask', None)
        harmful_sample_weight = batch.get('harmful_sample_weight', None)

        benign_input_ids = batch['benign_input_ids']
        benign_attention_mask = batch['benign_attention_mask']
        benign_loss_mask = batch.get('benign_loss_mask', None)
        benign_sample_weight = batch.get('benign_sample_weight', None)

        if harmful_loss_mask is not None and harmful_sample_weight is not None:
            harmful_loss_mask = harmful_loss_mask * harmful_sample_weight.unsqueeze(-1)
        if benign_loss_mask is not None and benign_sample_weight is not None:
            benign_loss_mask = benign_loss_mask * benign_sample_weight.unsqueeze(-1)

        # Get batch sizes for later splitting
        harmful_batch_size = harmful_input_ids.shape[0]
        benign_batch_size = benign_input_ids.shape[0]

        # === SINGLE FORWARD PASS: Concatenate harmful + benign ===
        # DEBUG: Confirm this is the new DDP-safe code
        if self.global_step == 0 and self.accelerator.is_main_process:
            self.accelerator.print("✓ Using SINGLE FORWARD PASS (DDP-safe version)")

        combined_input_ids = torch.cat([harmful_input_ids, benign_input_ids], dim=0)
        combined_attention_mask = torch.cat([harmful_attention_mask, benign_attention_mask], dim=0)
        needs_frozen = loss_mode in (LOSS_MODE_TRIPLET_FULL, LOSS_MODE_LEGACY_CB)
        combined_frozen_reps = {}
        combined_teacher_logits = None

        if self._rep_extraction_method == "hooks":
            # Clear previous representations
            self.model_extractor.clear()

            # Single forward through trainable model (need outputs for logits)
            student_outputs = self.model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            combined_model_reps = self.model_extractor.get_representations()
            combined_student_logits = student_outputs.logits  # Save for KL

            if needs_frozen:
                # Split frozen pass to save memory (no grad, so DDP safe)
                with torch.no_grad():
                    # 1. Harmful
                    self.frozen_extractor.clear()
                    _ = self.frozen_model(
                        input_ids=harmful_input_ids,
                        attention_mask=harmful_attention_mask,
                        use_cache=False,
                    )
                    harmful_frozen_reps = self.frozen_extractor.get_representations().copy()

                    # 2. Benign
                    self.frozen_extractor.clear()
                    teacher_outputs_b = self.frozen_model(
                        input_ids=benign_input_ids,
                        attention_mask=benign_attention_mask,
                        use_cache=False,
                    )
                    benign_frozen_reps = self.frozen_extractor.get_representations().copy()
                    teacher_logits_benign = teacher_outputs_b.logits
            else:
                harmful_frozen_reps = {}
                benign_frozen_reps = {}
                teacher_logits_benign = None

            # Split student representations by batch size
            harmful_model_reps = {
                layer: reps[:harmful_batch_size]
                for layer, reps in combined_model_reps.items()
            }
            benign_model_reps = {
                layer: reps[harmful_batch_size:]
                for layer, reps in combined_model_reps.items()
            }

            # Split student logits
            student_logits_benign = combined_student_logits[harmful_batch_size:]

        else:
            # Single forward with hidden_states output
            outputs = self.model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            combined_model_reps = _select_hidden_states(outputs, self.config.cb_target_layers)
            combined_student_logits = outputs.logits  # Save logits for KL
            del outputs

            if needs_frozen:
                with torch.no_grad():
                    # 1. Harmful
                    frozen_outputs_h = self.frozen_model(
                        input_ids=harmful_input_ids,
                        attention_mask=harmful_attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    harmful_frozen_reps = _select_hidden_states(
                        frozen_outputs_h, self.config.cb_target_layers
                    )
                    del frozen_outputs_h

                    # 2. Benign
                    frozen_outputs_b = self.frozen_model(
                        input_ids=benign_input_ids,
                        attention_mask=benign_attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    benign_frozen_reps = _select_hidden_states(
                        frozen_outputs_b, self.config.cb_target_layers
                    )
                    teacher_logits_benign = frozen_outputs_b.logits
                    del frozen_outputs_b
            else:
                harmful_frozen_reps = {}
                benign_frozen_reps = {}
                teacher_logits_benign = None

            # Split student representations by batch size
            harmful_model_reps = {
                layer: reps[:harmful_batch_size]
                for layer, reps in combined_model_reps.items()
            }
            benign_model_reps = {
                layer: reps[harmful_batch_size:]
                for layer, reps in combined_model_reps.items()
            }

            # Split student logits
            student_logits_benign = combined_student_logits[harmful_batch_size:]

        # === Compute Losses ===
        reroute_metrics: Optional[Dict[str, float]] = None
        beta_kl = float(getattr(self.config, "beta_kl", 0.0))
        kl_temp = float(getattr(self.config, "kl_temperature", 1.0))

        if loss_mode == LOSS_MODE_TRIPLET_FULL:
            if teacher_logits_benign is None:
                raise RuntimeError("triplet_full requires frozen-model logits")

            harmful_new = pooled_representations(
                reps_by_layer=harmful_model_reps,
                target_layers=self.config.cb_target_layers,
                token_mask=harmful_loss_mask if harmful_loss_mask is not None else harmful_attention_mask,
            )
            harmful_old = pooled_representations(
                reps_by_layer=harmful_frozen_reps,
                target_layers=self.config.cb_target_layers,
                token_mask=harmful_loss_mask if harmful_loss_mask is not None else harmful_attention_mask,
            )
            benign_new = pooled_representations(
                reps_by_layer=benign_model_reps,
                target_layers=self.config.cb_target_layers,
                token_mask=benign_loss_mask if benign_loss_mask is not None else benign_attention_mask,
            )
            benign_old = pooled_representations(
                reps_by_layer=benign_frozen_reps,
                target_layers=self.config.cb_target_layers,
                token_mask=benign_loss_mask if benign_loss_mask is not None else benign_attention_mask,
            )

            total_loss, triplet_metrics = triplet_full_loss(
                harmful_new=harmful_new,
                harmful_old=harmful_old,
                benign_new=benign_new,
                benign_old=benign_old,
                benign_student_logits=student_logits_benign,
                benign_teacher_logits=teacher_logits_benign,
                benign_attention_mask=benign_attention_mask,
                benign_loss_mask=benign_loss_mask,
                alpha_benign=float(getattr(self.config, "triplet_alpha_benign", 0.5)),
                beta_harmful=float(getattr(self.config, "triplet_beta_harmful", 0.4)),
                gamma_kl=float(getattr(self.config, "triplet_gamma_kl", 0.9)),
                margin_benign=float(getattr(self.config, "triplet_margin_benign", 500.0)),
                margin_harmful=float(getattr(self.config, "triplet_margin_harmful", 1500.0)),
                benign_positive_distance=getattr(self.config, "triplet_benign_positive_distance", "dmix"),
                benign_negative_distance=getattr(self.config, "triplet_benign_negative_distance", "dmix"),
                harmful_positive_distance=getattr(self.config, "triplet_harmful_positive_distance", "dmix"),
                harmful_negative_distance=getattr(self.config, "triplet_harmful_negative_distance", "dmix"),
                mix_l2_weight=float(getattr(self.config, "triplet_mix_l2_weight", 0.5)),
                mix_cos_weight=float(getattr(self.config, "triplet_mix_cos_weight", 0.5)),
                kl_temperature=kl_temp,
            )

            metrics = {
                "loss": total_loss.item(),
                "loss_triplet_benign": triplet_metrics["triplet_benign_loss"],
                "loss_triplet_harmful": triplet_metrics["triplet_harmful_loss"],
                "loss_triplet_kl": triplet_metrics["triplet_kl_loss"],
                "triplet_alpha": triplet_metrics["triplet_alpha"],
                "triplet_beta": triplet_metrics["triplet_beta"],
                "triplet_gamma": triplet_metrics["triplet_gamma"],
                "alpha": alpha,
            }
        elif loss_mode == LOSS_MODE_LEGACY_CB:
            loss_reroute, reroute_metrics = reroute_loss(
                harmful_model_reps,
                harmful_frozen_reps,
                self.config.cb_target_layers,
                harmful_attention_mask,
                loss_mask=harmful_loss_mask,
                return_metrics=True,
            )
            loss_retain = retain_loss(
                benign_model_reps,
                benign_frozen_reps,
                self.config.cb_target_layers,
                benign_attention_mask,
                loss_mask=benign_loss_mask,
            )
            if beta_kl > 0:
                if teacher_logits_benign is None:
                    raise RuntimeError("legacy_cb with KL requires frozen-model logits")
                loss_kl = kl_divergence_loss(
                    student_logits=student_logits_benign,
                    teacher_logits=teacher_logits_benign,
                    attention_mask=benign_attention_mask,
                    loss_mask=benign_loss_mask,
                    temperature=kl_temp,
                )
            else:
                loss_kl = torch.tensor(0.0, device=self.accelerator.device)

            total_loss = cs * loss_reroute + cr * (loss_retain + beta_kl * loss_kl)
            metrics = {
                "loss": total_loss.item(),
                "loss_reroute": loss_reroute.item(),
                "loss_retain": loss_retain.item(),
                "alpha": alpha,
            }
            if beta_kl > 0:
                metrics["loss_kl"] = loss_kl.item()
                metrics["beta_kl"] = beta_kl
        elif loss_mode == LOSS_MODE_LEGACY_SCHEMA:
            loss_reroute = random_reroute_loss(
                model_reps=harmful_model_reps,
                target_layers=self.config.cb_target_layers,
                loss_mask=harmful_loss_mask,
            )
            loss_retain = retain_ce_loss(
                logits=student_logits_benign,
                labels=benign_input_ids,
                loss_mask=benign_loss_mask,
            )
            total_loss = cs * loss_reroute + cr * loss_retain
            metrics = {
                "loss": total_loss.item(),
                "loss_reroute": loss_reroute.item(),
                "loss_retain": loss_retain.item(),
                "alpha": alpha,
            }
        else:
            raise ValueError(f"Unexpected loss mode: {loss_mode}")

        # === Diagnostic Logging ===
        if self.global_step % 50 == 0:
            if loss_mode == LOSS_MODE_TRIPLET_FULL:
                self.accelerator.print(
                    f"[Step {self.global_step}] mode={loss_mode} | "
                    f"L_b={metrics['loss_triplet_benign']:.4f} "
                    f"L_h={metrics['loss_triplet_harmful']:.4f} "
                    f"L_kl={metrics['loss_triplet_kl']:.4f}"
                )
            else:
                cos_sim = reroute_metrics["cos_sim_mean"] if reroute_metrics is not None else 0.0
                self.accelerator.print(
                    f"[Step {self.global_step}] mode={loss_mode} cs={cs:.3f} cr={cr:.3f} | "
                    f"cos_sim={cos_sim:.4f} | "
                    f"L_rr={metrics['loss_reroute']:.4f} L_ret={metrics['loss_retain']:.4f}"
                )

        # Backward pass (SINGLE backward through combined graph)
        self.accelerator.backward(total_loss)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        metrics["lr"] = self.scheduler.get_last_lr()[0]
        metrics["loss_mode"] = loss_mode

        # Add dual coefficient logging if enabled
        if use_dual:
            metrics['cs'] = cs
            metrics['cr'] = cr

        if reroute_metrics is not None:
            metrics['reroute_cos_sim_mean'] = reroute_metrics['cos_sim_mean']
            metrics['reroute_cos_sim_positive_frac'] = reroute_metrics['cos_sim_positive_frac']
            if reroute_metrics.get('target_type') != 'frozen_baseline':
                self.accelerator.print(
                    f"  WARNING: Unexpected reroute target type: {reroute_metrics.get('target_type')}"
                )

        # Gradient diagnostics (every 50 steps on main process)
        if self.global_step % 50 == 1 and self.accelerator.is_main_process:
            grad_stats = self._compute_gradient_stats()
            metrics.update(grad_stats)
            if grad_stats.get('grad_norm_total', 0) < 1e-8:
                self.accelerator.print(
                    f"  WARNING: Very small gradients detected! "
                    f"grad_norm={grad_stats.get('grad_norm_total', 0):.2e}"
                )
            if reroute_metrics is not None:
                self.accelerator.print(
                    f"  Reroute metrics: cos_sim_mean={reroute_metrics['cos_sim_mean']:.4f}, "
                    f"positive_frac={reroute_metrics['cos_sim_positive_frac']:.2%}, "
                    f"target={reroute_metrics['target_type']}"
                )

        return metrics
    
    def _validate_completion_masking(self):
        """
        Validate that completion masking is correctly targeting tool tokens.
        
        This runs at training start to catch misconfigured masking early.
        Checks:
        1. Assistant header is found in samples
        2. Completion mask covers <|python_tag|> tokens (not user prompt)
        3. Reports statistics on masking success rate
        """
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("VALIDATING COMPLETION MASKING")
        self.accelerator.print("=" * 60)
        
        # Get first few batches for validation
        num_to_check = min(3, len(self.dataset))
        stats = {
            "total_samples": 0,
            "found_assistant_header": 0,
            "mask_covers_python_tag": 0,
            "fallback_to_full_seq": 0,
        }
        
        for batch_idx in range(num_to_check):
            batch = self.dataset[batch_idx]
            
            # Check harmful samples
            harmful_ids = batch.get('harmful_input_ids')
            harmful_mask = batch.get('harmful_loss_mask', batch.get('harmful_attention_mask'))
            
            if harmful_ids is not None:
                batch_size = harmful_ids.shape[0] if len(harmful_ids.shape) > 1 else 1
                if batch_size == 1 and len(harmful_ids.shape) == 1:
                    harmful_ids = harmful_ids.unsqueeze(0)
                    harmful_mask = harmful_mask.unsqueeze(0)
                
                for i in range(batch_size):
                    stats["total_samples"] += 1
                    
                    # Decode to check format
                    tokens = harmful_ids[i]
                    text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                    
                    # Check for Llama 3.1 assistant header
                    if "<|start_header_id|>assistant<|end_header_id|>" in text:
                        stats["found_assistant_header"] += 1
                    else:
                        self.accelerator.print(f"  ⚠ Sample {batch_idx}.{i}: No assistant header found")
                    
                    # Check if mask covers <|python_tag|>
                    python_tag_id = self.tokenizer.convert_tokens_to_ids("<|python_tag|>")
                    if python_tag_id is not None and python_tag_id != self.tokenizer.unk_token_id:
                        # Find position of python_tag
                        positions = (tokens == python_tag_id).nonzero(as_tuple=True)
                        if len(positions[0]) > 0:
                            pos = positions[0][0].item()
                            if harmful_mask[i, pos] > 0:
                                stats["mask_covers_python_tag"] += 1
                            else:
                                self.accelerator.print(
                                    f"  ⚠ Sample {batch_idx}.{i}: <|python_tag|> at pos {pos} "
                                    f"but mask[{pos}]={harmful_mask[i, pos].item()}"
                                )
                    
                    # Check for fallback (mask == attention_mask)
                    attn_mask = batch.get('harmful_attention_mask')
                    if attn_mask is not None:
                        if batch_size == 1 and len(attn_mask.shape) == 1:
                            attn_mask = attn_mask.unsqueeze(0)
                        if torch.equal(harmful_mask[i], attn_mask[i]):
                            stats["fallback_to_full_seq"] += 1
        
        # Report results
        total = stats["total_samples"]
        if total > 0:
            self.accelerator.print(f"\n  Checked {total} harmful samples from {num_to_check} batches:")
            self.accelerator.print(
                f"  - Assistant header found: {stats['found_assistant_header']}/{total} "
                f"({100*stats['found_assistant_header']/total:.1f}%)"
            )
            self.accelerator.print(
                f"  - Mask covers <|python_tag|>: {stats['mask_covers_python_tag']}/{total} "
                f"({100*stats['mask_covers_python_tag']/total:.1f}%)"
            )
            self.accelerator.print(
                f"  - Fallback to full sequence: {stats['fallback_to_full_seq']}/{total} "
                f"({100*stats['fallback_to_full_seq']/total:.1f}%)"
            )
            
            # Warn if too many fallbacks
            if stats["fallback_to_full_seq"] > total * 0.5:
                self.accelerator.print(
                    "\n  ⚠️ WARNING: >50% samples using full-sequence masking!"
                    "\n  This means completion masking is NOT working properly."
                    "\n  Check that training data has <|start_header_id|>assistant<|end_header_id|> format."
                )
            
            # Warn if python_tag not being masked
            if stats["mask_covers_python_tag"] < stats["found_assistant_header"] * 0.9:
                self.accelerator.print(
                    "\n  ⚠️ WARNING: <|python_tag|> tokens not covered by loss mask!"
                    "\n  Training will NOT learn to reroute tool calls."
                )
        else:
            self.accelerator.print("  No samples to validate")
        
        self.accelerator.print("=" * 60 + "\n")

    def train(self):
        """Main training loop."""
        self.accelerator.print("=" * 60)
        self.accelerator.print("Starting Circuit Breaker Training")
        self.accelerator.print(f"  Model: {self.config.base_model}")
        self.accelerator.print(f"  Total Steps: {self.config.total_steps}")
        self.accelerator.print(f"  Alpha Max: {self.config.alpha_max}")
        self.accelerator.print(f"  Loss Mode: {getattr(self.config, 'loss_mode', LOSS_MODE_TRIPLET_FULL)}")
        self.accelerator.print(f"  CB Target Layers: {self.config.cb_target_layers}")
        self.accelerator.print("=" * 60)
        
        # =================================================================
        # DEBUG: Validate completion masking at training start
        # =================================================================
        if isinstance(self.dataset, CircuitBreakerDataset):
            self._validate_completion_masking()
        else:
            self.accelerator.print("Skipping completion-mask validation (external pre-tokenized dataset)")
        
        progress_bar = tqdm(
            total=self.config.total_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )
        
        epoch = 0
        while self.global_step < self.config.total_steps:
            epoch += 1
            self.accelerator.print(f"\n--- Epoch {epoch} ---")
            
            for batch in self.dataloader:
                if self.global_step >= self.config.total_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                
                # Training step
                metrics = self.train_step(batch)
                
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_msg = f"Step {self.global_step}: mode={metrics.get('loss_mode')}, loss={metrics['loss']:.4f}"
                    if 'loss_reroute' in metrics:
                        log_msg += f", reroute={metrics['loss_reroute']:.4f}"
                    if 'loss_retain' in metrics:
                        log_msg += f", retain={metrics['loss_retain']:.4f}"
                    if 'loss_triplet_benign' in metrics:
                        log_msg += f", triplet_b={metrics['loss_triplet_benign']:.4f}"
                    if 'loss_triplet_harmful' in metrics:
                        log_msg += f", triplet_h={metrics['loss_triplet_harmful']:.4f}"
                    if 'loss_kl' in metrics:
                        log_msg += f", kl={metrics['loss_kl']:.4f}"
                    if 'loss_triplet_kl' in metrics:
                        log_msg += f", triplet_kl={metrics['loss_triplet_kl']:.4f}"
                    log_msg += f", α={metrics['alpha']:.4f}"
                    self.accelerator.print(log_msg)
                    
                    if self.config.use_wandb:
                        if self.accelerator.is_main_process:
                            self.accelerator.log(metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint(final=True)
        
        if self.config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()
        
        self.accelerator.print("\n✅ Training complete!")
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            output_dir = Path(self.config.output_dir)
            if final:
                save_path = output_dir / "final"
            else:
                save_path = output_dir / f"checkpoint-{self.global_step}"
            
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            self.accelerator.print(f"💾 Saved checkpoint to {save_path}")

            # Optional W&B artifact logging (final checkpoint only by default).
            if (
                self.config.use_wandb
                and final
                and getattr(self.config, "wandb_log_artifacts", "none") == "final"
            ):
                artifact_name = f"cb-{self.config.wandb_run_name or 'run'}-final"
                log_dir_as_artifact(
                    artifact_name=artifact_name,
                    artifact_type=getattr(self.config, "wandb_artifact_type", "model"),
                    dir_path=save_path,
                    aliases=["final"],
                    metadata={
                        "global_step": self.global_step,
                        "output_dir": str(self.config.output_dir),
                        "base_model": self.config.base_model,
                    },
                )
    
    def cleanup(self):
        """Cleanup hooks and resources."""
        if getattr(self, "_rep_extraction_method", None) == "hooks":
            if self.model_extractor is not None:
                self.model_extractor.remove_hooks()
            if self.frozen_extractor is not None:
                self.frozen_extractor.remove_hooks()
