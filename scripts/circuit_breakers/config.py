"""
Circuit Breakers Configuration Module

Defines hyperparameters and training configuration for Representation Rerouting (RR).
Based on the Gray-Swan Circuit Breakers paper methodology.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16                          # LoRA rank
    alpha: int = 32                      # LoRA alpha (scaling factor)
    dropout: float = 0.05                # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    # Which layers to apply LoRA to (0-indexed)
    # For Llama-4-Scout-17B-16E: has 48 layers, we target early-to-mid
    target_layers: List[int] = field(default_factory=lambda: list(range(0, 25)))


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for Circuit Breaker (Representation Rerouting) training.

    Key components:
    - alpha_max: Maximum weight for rerouting loss (decays over training)
    - cb_target_layers: Layers where we extract representations for RR loss
    - total_steps: Number of training steps (alpha decays to 0 at 2x this)

    AGENTIC ENHANCEMENTS:
    - loss_weighting: "single_alpha" (original) or "dual" (paper-style cs/cr)
    - mask_prompt_tokens: Apply loss only on completion tokens
    - use_chat_template: Format data with model's chat template
    """

    # === Model ===
    base_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    torch_dtype: str = "bfloat16"  # bfloat16 for H100s

    # === LoRA ===
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # === Circuit Breaker Specific ===
    # Layers to extract representations from for RR loss
    # For Llama-4-Scout (48 layers): target mid layers where concepts form
    cb_target_layers: List[int] = field(default_factory=lambda: [12, 24, 36])

    # Alpha schedule: α(t) = α_max × max(0, 1 - t / (2 × total_steps))
    alpha_max: float = 10.0              # Starting alpha (paper uses 10 for Llama-3-8B)
    alpha_decay_strategy: str = "linear" # "linear" or "cosine"

    # Alpha decay horizon as a multiple of total_steps.
    # STAGE 1 FIX: Changed from 2.0 to 1.0 so alpha reaches 0 by end of training.
    # This matches the reference implementation's behavior.
    alpha_decay_multiplier: float = 1.0

    # How to extract representations for the RR/retain losses.
    # - "hidden_states": use Transformers' output_hidden_states=True (preferred; robust)
    # - "hooks": forward hooks on transformer blocks (kept for backwards-compatibility)
    representation_extraction: str = "hidden_states"

    # === Agentic Enhancements ===
    # Loss weighting strategy:
    # - "single_alpha": L = alpha * L_rr + L_ret (original, retain weight fixed at 1.0)
    # - "dual": L = cs(t) * L_rr + cr(t) * L_ret (paper-style, both coefficients vary)
    # STAGE 1 FIX: Changed from "single_alpha" to "dual" to match reference implementation.
    # Dual coefficients: cs(t) starts high and decays, cr(t) starts low and increases.
    loss_weighting: str = "dual"

    # Completion-based training:
    # - If True, loss is computed only on assistant completion tokens
    # - This is critical for completion-style data (prompt + harmful completion)
    mask_prompt_tokens: bool = True

    # Chat template formatting:
    # - If True, format prompt+completion pairs using the tokenizer's chat template
    use_chat_template: bool = True

    # === Training ===
    total_steps: int = 300               # Training steps (scaled up for larger model)
    batch_size: int = 16                 # Per-GPU batch size (8 harmful + 8 benign)
    gradient_accumulation_steps: int = 1 # Effective batch = batch_size * grad_accum * num_gpus
    learning_rate: float = 2e-5          # Lower LR for larger model
    warmup_steps: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # === Data ===
    data_path: str = "data/circuit_breakers/cb_training_batches.jsonl"
    max_seq_length: int = 2048           # Max sequence length for tokenization
    
    # === Multi-GPU (8 x H100) ===
    num_gpus: int = 8
    deepspeed_config: Optional[str] = None  # Path to DeepSpeed config if using
    gradient_checkpointing: bool = True     # Save memory on large model
    
    # === Logging ===
    output_dir: str = "outputs/circuit_breaker"
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    use_wandb: bool = True
    wandb_project: str = "circuit-breakers"
    wandb_run_name: Optional[str] = None

    # Optional W&B metadata (kept minimal; can also be set via env vars)
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    # "online" | "offline" | "disabled" (W&B also respects WANDB_MODE)
    wandb_mode: Optional[str] = None
    # Artifact logging policy: "none" | "final"
    wandb_log_artifacts: str = "none"
    wandb_artifact_type: str = "model"
    
    # === Evaluation ===
    eval_data_path: Optional[str] = None  # Separate eval set if available
    eval_batch_size: int = 8


@dataclass
class CircuitBreakerConfigLlama3_8B(CircuitBreakerConfig):
    """Preset configuration for Llama-3-8B-Instruct (paper baseline)."""
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    cb_target_layers: List[int] = field(default_factory=lambda: [10, 20])
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_layers=list(range(0, 21))
    ))
    alpha_max: float = 10.0
    total_steps: int = 150
    learning_rate: float = 5e-5


@dataclass
class CircuitBreakerConfigLlama3_1_8B_Instruct(CircuitBreakerConfig):
    """Preset configuration for meta-llama/Llama-3.1-8B-Instruct."""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # 8B Llama variants are typically 32 layers; mid-layer targets are stable defaults.
    cb_target_layers: List[int] = field(default_factory=lambda: [10, 20])

    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_layers=list(range(0, 21))
    ))

    alpha_max: float = 10.0
    total_steps: int = 150
    learning_rate: float = 5e-5


@dataclass
class CircuitBreakerConfigMistral_7B(CircuitBreakerConfig):
    """Preset configuration for Mistral-7B-Instruct."""
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cb_target_layers: List[int] = field(default_factory=lambda: [10, 20])
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_layers=list(range(0, 21))
    ))
    alpha_max: float = 5.0
    total_steps: int = 150
    learning_rate: float = 5e-5


@dataclass  
class CircuitBreakerConfigLlama4Scout(CircuitBreakerConfig):
    """
    Configuration for Llama-4-Scout-17B-16E-Instruct (MoE model).
    
    Notes:
    - This is a Mixture of Experts model with 16 experts
    - Has 48 transformer layers
    - Requires more memory due to expert routing
    - We target router/gate projections in addition to attention
    """
    base_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    
    # MoE-specific: include router weights
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
            # Note: router weights are typically not trained with LoRA
        ],
        target_layers=list(range(0, 30))  # First 30 of 48 layers
    ))
    
    # More layers to target for CB (48 total)
    cb_target_layers: List[int] = field(default_factory=lambda: [12, 24, 36])
    
    # Adjusted hyperparameters for larger model
    alpha_max: float = 8.0               # Slightly lower for stability
    total_steps: int = 300               # More steps for larger model
    learning_rate: float = 2e-5          # Lower LR for larger model
    warmup_steps: int = 30
    batch_size: int = 8                  # Smaller batch due to model size
    gradient_accumulation_steps: int = 2  # Effective batch = 8 * 2 * 8 = 128
    gradient_checkpointing: bool = True  # Essential for MoE


# === Config Loading Utilities ===

CONFIG_PRESETS = {
    "llama-4-scout": CircuitBreakerConfigLlama4Scout,
    "llama-3-8b": CircuitBreakerConfigLlama3_8B,
    "llama-3.1-8b-instruct": CircuitBreakerConfigLlama3_1_8B_Instruct,
    "mistral-7b": CircuitBreakerConfigMistral_7B,
    "default": CircuitBreakerConfig,
}


def get_config(preset: str = "llama-4-scout", **overrides) -> CircuitBreakerConfig:
    """
    Get a configuration preset with optional overrides.
    
    Args:
        preset: One of "llama-4-scout", "llama-3-8b", "mistral-7b", "default"
        **overrides: Any config fields to override
    
    Returns:
        CircuitBreakerConfig instance
    """
    if preset not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIG_PRESETS.keys())}")
    
    config = CONFIG_PRESETS[preset]()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config field: {key}")
    
    return config


def config_to_dict(config: CircuitBreakerConfig) -> dict:
    """Convert config to dictionary for logging/saving."""
    from dataclasses import asdict
    return asdict(config)
