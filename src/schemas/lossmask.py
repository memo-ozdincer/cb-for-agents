"""
Tier C: Loss Mask Schema (lossmask_v1)

Materialized per-token loss masks, derived from a render_v1 and an LMP policy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass
class LossMaskStats:
    """Statistics about the mask."""
    total_tokens: int = 0
    masked_tokens: int = 0  # Tokens with mask=0 (no loss)
    unmasked_tokens: int = 0  # Tokens with mask>0 (has loss)
    mask_ratio: float = 0.0  # unmasked / total


@dataclass
class LossMask:
    """
    Materialized loss mask record (Tier C).

    This is where per-token masks live, derived from training.loss_mask_policy
    and the render_v1 tokenization.
    """
    lossmask_id: str
    render_id: str
    trace_id: str
    policy_id: str
    loss_mask: List[float]

    created_at: Optional[str] = None
    policy_version: Optional[str] = None
    policy_params: Optional[Dict[str, Any]] = None
    labels: Optional[List[int]] = None  # Target labels (-100 for masked)
    sample_weight: float = 1.0
    stats: Optional[LossMaskStats] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.stats is None:
            self._compute_stats()

    def _compute_stats(self):
        """Compute mask statistics."""
        total = len(self.loss_mask)
        unmasked = sum(1 for m in self.loss_mask if m > 0)
        masked = total - unmasked
        self.stats = LossMaskStats(
            total_tokens=total,
            masked_tokens=masked,
            unmasked_tokens=unmasked,
            mask_ratio=unmasked / total if total > 0 else 0.0,
        )

    @classmethod
    def generate_id(
        cls,
        policy_id: str,
        render_id: str,
    ) -> str:
        """Generate a deterministic lossmask ID."""
        # Extract render hash from render_id
        render_parts = render_id.split("_")
        render_hash = "_".join(render_parts[-2:])  # Last two parts

        # Normalize policy_id
        policy_short = policy_id.replace(":", "_").lower()[:20]

        return f"lossmask_{policy_short}_{render_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [_clean(item) for item in obj]
            return obj

        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossMask":
        """Create from dictionary."""
        stats = None
        if data.get("stats"):
            stats = LossMaskStats(**data["stats"])

        return cls(
            lossmask_id=data["lossmask_id"],
            render_id=data["render_id"],
            trace_id=data["trace_id"],
            policy_id=data["policy_id"],
            loss_mask=data["loss_mask"],
            created_at=data.get("created_at"),
            policy_version=data.get("policy_version"),
            policy_params=data.get("policy_params"),
            labels=data.get("labels"),
            sample_weight=data.get("sample_weight", 1.0),
            stats=stats,
        )

    @classmethod
    def from_render(
        cls,
        render: "RenderedView",  # Forward reference
        policy_id: str,
        mask_fn,  # Callable that takes render and returns mask
        policy_version: Optional[str] = None,
        policy_params: Optional[Dict[str, Any]] = None,
        sample_weight: float = 1.0,
    ) -> "LossMask":
        """Create a loss mask from a rendered view using a masking function."""
        loss_mask = mask_fn(render)

        # Generate labels: shifted input_ids with -100 for masked positions
        labels = []
        input_ids = render.input_ids
        for i in range(len(input_ids)):
            if i < len(input_ids) - 1:
                # Standard LM: predict next token
                if loss_mask[i] > 0:
                    labels.append(input_ids[i + 1])
                else:
                    labels.append(-100)  # Ignore in loss
            else:
                labels.append(-100)  # No target for last token

        lossmask_id = cls.generate_id(policy_id, render.render_id)

        return cls(
            lossmask_id=lossmask_id,
            render_id=render.render_id,
            trace_id=render.trace_id,
            policy_id=policy_id,
            loss_mask=loss_mask,
            policy_version=policy_version,
            policy_params=policy_params,
            labels=labels,
            sample_weight=sample_weight,
        )
