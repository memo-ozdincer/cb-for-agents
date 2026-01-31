#!/usr/bin/env python3
"""
Circuit Breaker Training with Schema v1 Data Format.

Supports the new ETL pipeline output:
- render_v1 JSONL files (tokenized traces with alignment)
- lossmask_v1 JSONL files (per-token loss masks)

Data loading modes:
- ds_dr: Load DS traces as harmful, DR traces as benign
- labeled: Load traces based on labels.is_harmful field
- mixed: Combine multiple sources with explicit harmful/benign paths

MWCS (Mixture Weighted Curriculum Scheduling) support:
- Pre-computed sample_weight from lossmask_v1
- Per-step weight adjustment via curriculum schedule
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType

# Local imports
from src.training.config import (
    CircuitBreakerConfig,
    get_config,
    config_to_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file."""
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_renders_and_masks(
    render_path: Path,
    lossmask_path: Path,
) -> List[Dict[str, Any]]:
    """
    Load and join render_v1 and lossmask_v1 files.

    Returns list of dicts with:
    - input_ids: token IDs
    - attention_mask: attention mask
    - loss_mask: per-token loss weights
    - sample_weight: sample-level weight (from MWCS)
    - trace_id: original trace ID
    - render_id: render ID
    - policy_id: LMP policy used
    """
    renders = {r["render_id"]: r for r in _iter_jsonl(render_path)}

    samples = []
    for mask_row in _iter_jsonl(lossmask_path):
        render_id = mask_row.get("render_id")
        render = renders.get(render_id)

        if render is None:
            logger.warning("No render found for lossmask %s", render_id)
            continue

        samples.append({
            "input_ids": render["input_ids"],
            "attention_mask": render.get("attention_mask", [1] * len(render["input_ids"])),
            "loss_mask": mask_row["loss_mask"],
            "sample_weight": mask_row.get("sample_weight", 1.0),
            "trace_id": mask_row.get("trace_id"),
            "render_id": render_id,
            "policy_id": mask_row.get("policy_id"),
        })

    return samples


def load_ds_dr_data(
    ds_render_path: Path,
    ds_lossmask_path: Path,
    dr_render_path: Path,
    dr_lossmask_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load DS (harmful) and DR (benign) data from ETL_B outputs.

    DS traces = model follows injection = harmful examples
    DR traces = model ignores injection = benign examples
    """
    logger.info("Loading DS (harmful) data from %s", ds_render_path)
    ds_samples = load_renders_and_masks(ds_render_path, ds_lossmask_path)
    logger.info("Loaded %d DS (harmful) samples", len(ds_samples))

    logger.info("Loading DR (benign) data from %s", dr_render_path)
    dr_samples = load_renders_and_masks(dr_render_path, dr_lossmask_path)
    logger.info("Loaded %d DR (benign) samples", len(dr_samples))

    return ds_samples, dr_samples


def load_labeled_data(
    render_path: Path,
    lossmask_path: Path,
    traces_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load data and split by labels.is_harmful field.

    For AgentDojo data where harmful/benign is determined by attack success.
    """
    all_samples = load_renders_and_masks(render_path, lossmask_path)

    # Load trace labels if provided
    trace_labels = {}
    if traces_path and traces_path.exists():
        for row in _iter_jsonl(traces_path):
            trace_id = row.get("id")
            labels = row.get("labels", {})
            is_harmful = labels.get("is_harmful", False)
            trace_labels[trace_id] = is_harmful

    harmful = []
    benign = []

    for sample in all_samples:
        trace_id = sample.get("trace_id")
        is_harmful = trace_labels.get(trace_id, False)

        if is_harmful:
            harmful.append(sample)
        else:
            benign.append(sample)

    logger.info("Split %d samples: %d harmful, %d benign",
                len(all_samples), len(harmful), len(benign))

    return harmful, benign


# =============================================================================
# Dataset
# =============================================================================

class SchemaDataset(Dataset):
    """Dataset for schema v1 format data."""

    def __init__(
        self,
        harmful_samples: List[Dict[str, Any]],
        benign_samples: List[Dict[str, Any]],
        max_length: int = 2048,
        pad_token_id: int = 0,
    ):
        self.harmful_samples = harmful_samples
        self.benign_samples = benign_samples
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Balance by taking min of both
        self.num_pairs = min(len(harmful_samples), len(benign_samples))

        if self.num_pairs == 0:
            raise ValueError("No paired data available (need both harmful and benign)")

        logger.info("Created dataset with %d pairs (%d harmful, %d benign available)",
                   self.num_pairs, len(harmful_samples), len(benign_samples))

    def __len__(self) -> int:
        return self.num_pairs

    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare a single sample with padding/truncation."""
        input_ids = sample["input_ids"][:self.max_length]
        attention_mask = sample["attention_mask"][:self.max_length]
        loss_mask = sample["loss_mask"][:self.max_length]

        # Pad if needed
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            loss_mask = loss_mask + [0.0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "sample_weight": torch.tensor(sample.get("sample_weight", 1.0), dtype=torch.float),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get harmful sample (cycling if needed)
        harmful_idx = idx % len(self.harmful_samples)
        harmful = self._prepare_sample(self.harmful_samples[harmful_idx])

        # Get benign sample (cycling if needed)
        benign_idx = idx % len(self.benign_samples)
        benign = self._prepare_sample(self.benign_samples[benign_idx])

        return {
            "harmful_input_ids": harmful["input_ids"],
            "harmful_attention_mask": harmful["attention_mask"],
            "harmful_loss_mask": harmful["loss_mask"],
            "harmful_sample_weight": harmful["sample_weight"],
            "benign_input_ids": benign["input_ids"],
            "benign_attention_mask": benign["attention_mask"],
            "benign_loss_mask": benign["loss_mask"],
            "benign_sample_weight": benign["sample_weight"],
        }


# =============================================================================
# Loss Computation
# =============================================================================

def compute_rr_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    target_layers: List[int],
) -> torch.Tensor:
    """
    Compute Representation Rerouting (RR) loss for harmful examples.

    Loss = -log(1 - cos_sim(h_harmful, h_random))

    This encourages harmful representations to be orthogonal to their
    original direction (randomizing them).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states

    # Get representations at target layers
    target_reps = []
    for layer_idx in target_layers:
        if layer_idx < len(hidden_states):
            target_reps.append(hidden_states[layer_idx])

    if not target_reps:
        return torch.tensor(0.0, device=input_ids.device)

    # Average across layers
    h_harmful = torch.stack(target_reps, dim=0).mean(dim=0)  # [batch, seq, hidden]

    # Create random target (unit random vectors)
    h_random = torch.randn_like(h_harmful)
    h_random = h_random / (h_random.norm(dim=-1, keepdim=True) + 1e-8)

    # Normalize harmful representations
    h_harmful_norm = h_harmful / (h_harmful.norm(dim=-1, keepdim=True) + 1e-8)

    # Cosine similarity
    cos_sim = (h_harmful_norm * h_random).sum(dim=-1)  # [batch, seq]

    # Apply loss mask (only on completion tokens)
    mask = loss_mask.float()

    # Loss: -log(1 - cos_sim) encourages orthogonality
    loss_per_token = -torch.log(1 - cos_sim.abs() + 1e-8)

    # Masked mean
    masked_loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)

    return masked_loss


def compute_retain_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute retain loss (standard cross-entropy) for benign examples.

    This maintains the model's ability to generate benign responses.
    """
    if labels is None:
        labels = input_ids.clone()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    # Get per-token loss
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    # Masked mean
    masked_loss = (loss_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)

    return masked_loss


def compute_cb_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    config: CircuitBreakerConfig,
    step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined Circuit Breaker loss.

    L = cs(t) * L_rr + cr(t) * L_retain

    Where cs(t) and cr(t) are the rerouting and retain coefficients that
    vary over training.
    """
    device = next(model.parameters()).device

    # Move batch to device
    harmful_ids = batch["harmful_input_ids"].to(device)
    harmful_mask = batch["harmful_attention_mask"].to(device)
    harmful_loss_mask = batch["harmful_loss_mask"].to(device)
    harmful_weight = batch["harmful_sample_weight"].to(device)

    benign_ids = batch["benign_input_ids"].to(device)
    benign_mask = batch["benign_attention_mask"].to(device)
    benign_loss_mask = batch["benign_loss_mask"].to(device)
    benign_weight = batch["benign_sample_weight"].to(device)

    # Compute alpha schedule
    # α(t) = α_max × max(0, 1 - t / (decay_multiplier × total_steps))
    decay_horizon = config.alpha_decay_multiplier * config.total_steps
    alpha = config.alpha_max * max(0.0, 1.0 - step / decay_horizon)

    # Compute losses
    rr_loss = compute_rr_loss(
        model,
        harmful_ids,
        harmful_mask,
        harmful_loss_mask * harmful_weight.unsqueeze(-1),
        config.cb_target_layers,
    )

    retain_loss = compute_retain_loss(
        model,
        benign_ids,
        benign_mask,
        benign_loss_mask * benign_weight.unsqueeze(-1),
    )

    # Combine losses based on weighting strategy
    if config.loss_weighting == "dual":
        # Dual coefficient: cs decays, cr increases
        cs = alpha / config.alpha_max if config.alpha_max > 0 else 1.0
        cr = 1.0 - cs * 0.5  # cr ranges from 0.5 to 1.0
        total_loss = cs * rr_loss + cr * retain_loss
        metrics = {
            "rr_loss": rr_loss.item(),
            "retain_loss": retain_loss.item(),
            "cs": cs,
            "cr": cr,
            "alpha": alpha,
        }
    else:
        # Single alpha: L = alpha * L_rr + L_retain
        total_loss = alpha * rr_loss + retain_loss
        metrics = {
            "rr_loss": rr_loss.item(),
            "retain_loss": retain_loss.item(),
            "alpha": alpha,
        }

    metrics["total_loss"] = total_loss.item()

    return total_loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

def train(
    config: CircuitBreakerConfig,
    harmful_samples: List[Dict[str, Any]],
    benign_samples: List[Dict[str, Any]],
    output_dir: Path,
    resume_from: Optional[Path] = None,
):
    """Main training loop."""

    # Setup distributed if available
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        logger.info("Training configuration:")
        logger.info("  Model: %s", config.base_model)
        logger.info("  Total steps: %d", config.total_steps)
        logger.info("  Batch size: %d", config.batch_size)
        logger.info("  Learning rate: %e", config.learning_rate)
        logger.info("  Alpha max: %.2f", config.alpha_max)
        logger.info("  Target layers: %s", config.cb_target_layers)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if is_main:
        logger.info("Loading model: %s", config.base_model)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map={"": device} if not is_distributed else None,
        local_files_only=True,
    )

    # Add LoRA
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Create dataset
    dataset = SchemaDataset(
        harmful_samples,
        benign_samples,
        max_length=config.max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
    )

    # Setup wandb
    if config.use_wandb and is_main:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config_to_dict(config),
            )
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
            config.use_wandb = False

    # Training loop
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    epoch = 0

    model.train()
    pbar = tqdm(total=config.total_steps, disable=not is_main, desc="Training")

    while global_step < config.total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            if global_step >= config.total_steps:
                break

            # Forward pass
            loss, metrics = compute_cb_loss(
                model.module if is_distributed else model,
                batch,
                config,
                global_step,
            )

            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if is_main and global_step % config.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                log_msg = (
                    f"Step {global_step}: loss={metrics['total_loss']:.4f}, "
                    f"rr={metrics['rr_loss']:.4f}, ret={metrics['retain_loss']:.4f}, "
                    f"alpha={metrics['alpha']:.2f}, lr={lr:.2e}"
                )
                logger.info(log_msg)

                if config.use_wandb:
                    import wandb
                    wandb.log({**metrics, "lr": lr}, step=global_step)

            # Save checkpoint
            if is_main and global_step > 0 and global_step % config.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(exist_ok=True)

                save_model = model.module if is_distributed else model
                save_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

                # Save training state
                torch.save({
                    "step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_dir / "training_state.pt")

                logger.info("Saved checkpoint to %s", ckpt_dir)

            global_step += 1
            pbar.update(1)

        epoch += 1

    pbar.close()

    # Save final model
    if is_main:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)

        save_model = model.module if is_distributed else model
        save_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        logger.info("Saved final model to %s", final_dir)

        if config.use_wandb:
            import wandb
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Circuit Breaker training with schema v1 data format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data loading mode
    mode_group = parser.add_argument_group("Data Loading")
    mode_group.add_argument(
        "--mode", type=str, default="ds_dr",
        choices=["ds_dr", "labeled", "mixed"],
        help="Data loading mode: ds_dr (DS=harmful, DR=benign), "
             "labeled (split by labels.is_harmful), mixed (explicit paths)",
    )

    # DS/DR mode paths
    mode_group.add_argument("--ds-renders", type=Path, help="DS renders JSONL (harmful)")
    mode_group.add_argument("--ds-lossmasks", type=Path, help="DS lossmasks JSONL (harmful)")
    mode_group.add_argument("--dr-renders", type=Path, help="DR renders JSONL (benign)")
    mode_group.add_argument("--dr-lossmasks", type=Path, help="DR lossmasks JSONL (benign)")

    # Labeled mode paths
    mode_group.add_argument("--renders", type=Path, help="Renders JSONL (for labeled mode)")
    mode_group.add_argument("--lossmasks", type=Path, help="Lossmasks JSONL (for labeled mode)")
    mode_group.add_argument("--traces", type=Path, help="Traces JSONL (for labels)")

    # Mixed mode paths
    mode_group.add_argument(
        "--harmful-renders", type=Path, nargs="+",
        help="Harmful renders JSONL paths (mixed mode)",
    )
    mode_group.add_argument(
        "--harmful-lossmasks", type=Path, nargs="+",
        help="Harmful lossmasks JSONL paths (mixed mode)",
    )
    mode_group.add_argument(
        "--benign-renders", type=Path, nargs="+",
        help="Benign renders JSONL paths (mixed mode)",
    )
    mode_group.add_argument(
        "--benign-lossmasks", type=Path, nargs="+",
        help="Benign lossmasks JSONL paths (mixed mode)",
    )

    # Model config
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--preset", type=str, default="llama-3.1-8b-instruct",
        help="Config preset (llama-4-scout, llama-3-8b, llama-3.1-8b-instruct, mistral-7b)",
    )
    model_group.add_argument("--model", type=str, help="Override base model")
    model_group.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    model_group.add_argument("--resume-from", type=Path, help="Resume from checkpoint")

    # Training hyperparameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--total-steps", type=int, help="Total training steps")
    train_group.add_argument("--batch-size", type=int, help="Batch size per GPU")
    train_group.add_argument("--learning-rate", type=float, help="Learning rate")
    train_group.add_argument("--warmup-steps", type=int, help="Warmup steps")
    train_group.add_argument("--alpha-max", type=float, help="Initial alpha value")
    train_group.add_argument("--max-seq-length", type=int, help="Max sequence length")
    train_group.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation")

    # CB specific
    cb_group = parser.add_argument_group("Circuit Breaker")
    cb_group.add_argument(
        "--cb-target-layers", type=int, nargs="+",
        help="Target layers for representation extraction",
    )
    cb_group.add_argument(
        "--loss-weighting", type=str, choices=["single_alpha", "dual"],
        help="Loss weighting strategy",
    )

    # LoRA
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    lora_group.add_argument("--lora-dropout", type=float, help="LoRA dropout")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--wandb-project", type=str, help="W&B project name")
    log_group.add_argument("--wandb-run-name", type=str, help="W&B run name")
    log_group.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    log_group.add_argument("--logging-steps", type=int, help="Log every N steps")
    log_group.add_argument("--save-steps", type=int, help="Save checkpoint every N steps")

    args = parser.parse_args()

    # Build config
    overrides = {}
    if args.model:
        overrides["base_model"] = args.model
    if args.total_steps:
        overrides["total_steps"] = args.total_steps
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.warmup_steps:
        overrides["warmup_steps"] = args.warmup_steps
    if args.alpha_max:
        overrides["alpha_max"] = args.alpha_max
    if args.max_seq_length:
        overrides["max_seq_length"] = args.max_seq_length
    if args.gradient_accumulation_steps:
        overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.cb_target_layers:
        overrides["cb_target_layers"] = args.cb_target_layers
    if args.loss_weighting:
        overrides["loss_weighting"] = args.loss_weighting
    if args.wandb_project:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_run_name:
        overrides["wandb_run_name"] = args.wandb_run_name
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.logging_steps:
        overrides["logging_steps"] = args.logging_steps
    if args.save_steps:
        overrides["save_steps"] = args.save_steps

    config = get_config(args.preset, **overrides)

    # Update LoRA config if specified
    if args.lora_r:
        config.lora.r = args.lora_r
    if args.lora_alpha:
        config.lora.alpha = args.lora_alpha
    if args.lora_dropout:
        config.lora.dropout = args.lora_dropout

    # Load data based on mode
    if args.mode == "ds_dr":
        if not all([args.ds_renders, args.ds_lossmasks, args.dr_renders, args.dr_lossmasks]):
            parser.error("DS/DR mode requires --ds-renders, --ds-lossmasks, --dr-renders, --dr-lossmasks")

        harmful_samples, benign_samples = load_ds_dr_data(
            args.ds_renders, args.ds_lossmasks,
            args.dr_renders, args.dr_lossmasks,
        )

    elif args.mode == "labeled":
        if not all([args.renders, args.lossmasks]):
            parser.error("Labeled mode requires --renders and --lossmasks")

        harmful_samples, benign_samples = load_labeled_data(
            args.renders, args.lossmasks, args.traces,
        )

    elif args.mode == "mixed":
        if not all([args.harmful_renders, args.harmful_lossmasks,
                   args.benign_renders, args.benign_lossmasks]):
            parser.error("Mixed mode requires --harmful-renders, --harmful-lossmasks, "
                        "--benign-renders, --benign-lossmasks")

        harmful_samples = []
        for render_path, mask_path in zip(args.harmful_renders, args.harmful_lossmasks):
            harmful_samples.extend(load_renders_and_masks(render_path, mask_path))

        benign_samples = []
        for render_path, mask_path in zip(args.benign_renders, args.benign_lossmasks):
            benign_samples.extend(load_renders_and_masks(render_path, mask_path))

        logger.info("Mixed mode: %d harmful, %d benign samples",
                   len(harmful_samples), len(benign_samples))

    # Run training
    train(
        config,
        harmful_samples,
        benign_samples,
        args.output_dir,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
