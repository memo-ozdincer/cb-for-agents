"""
LMP and MWCS Registry Management

Provides loading and management of:
- LMP (Loss Masking Policy) configurations
- MWCS (Mixture Weighting & Curriculum Schedule) configurations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import json


# =============================================================================
# LMP Registry
# =============================================================================

@dataclass
class LMPPolicy:
    """A loss masking policy definition."""
    name: str
    strategy: Literal[
        "assistant_only",
        "completion_only",
        "full_sequence",
        "tool_calls_only",
        "action_prefix_only",
        "action_commitment",
        "custom",
    ]
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    compatible_with: Optional[List[str]] = None  # ['harmful', 'benign']


@dataclass
class LMPRegistry:
    """Registry of loss masking policies."""
    version: str
    policies: Dict[str, LMPPolicy]
    default_policy: str = "assistant_only"

    def get_policy(self, policy_id: str) -> LMPPolicy:
        """Get a policy by ID."""
        if policy_id not in self.policies:
            raise KeyError(f"Unknown LMP policy: {policy_id}. Available: {list(self.policies.keys())}")
        return self.policies[policy_id]

    def list_policies(self) -> List[str]:
        """List all available policy IDs."""
        return list(self.policies.keys())

    def get_default(self) -> LMPPolicy:
        """Get the default policy."""
        return self.get_policy(self.default_policy)


def load_lmp_registry(path: Optional[Path] = None) -> LMPRegistry:
    """Load LMP registry from JSON file."""
    if path is None:
        # Default path relative to this file
        path = Path(__file__).parent.parent.parent / "configs" / "lmp_registry_v1.json"

    with open(path, "r") as f:
        data = json.load(f)

    policies = {}
    for policy_id, policy_data in data.get("policies", {}).items():
        policies[policy_id] = LMPPolicy(
            name=policy_data["name"],
            strategy=policy_data["strategy"],
            description=policy_data.get("description"),
            params=policy_data.get("params"),
            compatible_with=policy_data.get("compatible_with"),
        )

    return LMPRegistry(
        version=data.get("version", "1.0.0"),
        policies=policies,
        default_policy=data.get("default_policy", "assistant_only"),
    )


# =============================================================================
# MWCS Registry
# =============================================================================

@dataclass
class MixtureClass:
    """A mixture class definition."""
    name: str
    category: Literal["harmful", "benign"]
    description: Optional[str] = None
    source_datasets: Optional[List[str]] = None
    default_lmp: Optional[str] = None


@dataclass
class CurriculumPhase:
    """A phase in a curriculum schedule."""
    name: str
    start_step: int
    end_step: int
    class_weights: Dict[str, float]
    lmp_overrides: Optional[Dict[str, str]] = None


@dataclass
class Curriculum:
    """Curriculum schedule definition."""
    type: Literal["none", "linear", "step", "cosine", "custom"]
    phases: Optional[List[CurriculumPhase]] = None
    interpolation: str = "none"


@dataclass
class MWCSSchedule:
    """A mixture weighting & curriculum schedule."""
    name: str
    class_weights: Dict[str, float]
    description: Optional[str] = None
    curriculum: Optional[Curriculum] = None
    sampling_strategy: Literal["proportional", "balanced", "temperature", "custom"] = "proportional"
    sampling_params: Optional[Dict[str, Any]] = None

    def get_weights_at_step(self, step: int) -> Dict[str, float]:
        """Get class weights at a given training step."""
        if self.curriculum is None or self.curriculum.type == "none":
            return self.class_weights

        if self.curriculum.phases is None:
            return self.class_weights

        # Find the applicable phase(s)
        current_phase = None
        next_phase = None
        for phase in self.curriculum.phases:
            if phase.start_step <= step < phase.end_step:
                current_phase = phase
            elif phase.start_step > step and next_phase is None:
                next_phase = phase

        if current_phase is None:
            # Before first phase or after last phase
            if step < self.curriculum.phases[0].start_step:
                return self.curriculum.phases[0].class_weights
            else:
                return self.curriculum.phases[-1].class_weights

        # Handle interpolation
        if self.curriculum.interpolation == "none":
            return current_phase.class_weights

        if next_phase is None or self.curriculum.interpolation == "none":
            return current_phase.class_weights

        # Linear or smooth interpolation
        phase_progress = (step - current_phase.start_step) / (
            current_phase.end_step - current_phase.start_step
        )

        if self.curriculum.interpolation == "smooth":
            # Cosine interpolation
            import math
            phase_progress = (1 - math.cos(phase_progress * math.pi)) / 2

        # Interpolate weights
        weights = {}
        all_classes = set(current_phase.class_weights.keys()) | set(next_phase.class_weights.keys())
        for cls in all_classes:
            w1 = current_phase.class_weights.get(cls, 0.0)
            w2 = next_phase.class_weights.get(cls, w1)
            weights[cls] = w1 + (w2 - w1) * phase_progress

        return weights


@dataclass
class MWCSRegistry:
    """Registry of mixture weighting & curriculum schedules."""
    version: str
    schedules: Dict[str, MWCSSchedule]
    mixture_classes: Dict[str, MixtureClass]
    default_schedule: str = "balanced_cb"

    def get_schedule(self, schedule_id: str) -> MWCSSchedule:
        """Get a schedule by ID."""
        if schedule_id not in self.schedules:
            raise KeyError(f"Unknown MWCS schedule: {schedule_id}. Available: {list(self.schedules.keys())}")
        return self.schedules[schedule_id]

    def get_mixture_class(self, class_id: str) -> MixtureClass:
        """Get a mixture class by ID."""
        if class_id not in self.mixture_classes:
            raise KeyError(f"Unknown mixture class: {class_id}. Available: {list(self.mixture_classes.keys())}")
        return self.mixture_classes[class_id]

    def list_schedules(self) -> List[str]:
        """List all available schedule IDs."""
        return list(self.schedules.keys())

    def list_mixture_classes(self) -> List[str]:
        """List all available mixture class IDs."""
        return list(self.mixture_classes.keys())

    def get_default(self) -> MWCSSchedule:
        """Get the default schedule."""
        return self.get_schedule(self.default_schedule)


def load_mwcs_registry(path: Optional[Path] = None) -> MWCSRegistry:
    """Load MWCS registry from JSON file."""
    if path is None:
        # Default path relative to this file
        path = Path(__file__).parent.parent.parent / "configs" / "mwcs_registry_v1.json"

    with open(path, "r") as f:
        data = json.load(f)

    # Parse mixture classes
    mixture_classes = {}
    for class_id, class_data in data.get("mixture_classes", {}).items():
        mixture_classes[class_id] = MixtureClass(
            name=class_data["name"],
            category=class_data["category"],
            description=class_data.get("description"),
            source_datasets=class_data.get("source_datasets"),
            default_lmp=class_data.get("default_lmp"),
        )

    # Parse schedules
    schedules = {}
    for schedule_id, schedule_data in data.get("schedules", {}).items():
        curriculum = None
        if schedule_data.get("curriculum"):
            curr_data = schedule_data["curriculum"]
            phases = None
            if curr_data.get("phases"):
                phases = [
                    CurriculumPhase(
                        name=p["name"],
                        start_step=p["start_step"],
                        end_step=p["end_step"],
                        class_weights=p["class_weights"],
                        lmp_overrides=p.get("lmp_overrides"),
                    )
                    for p in curr_data["phases"]
                ]
            curriculum = Curriculum(
                type=curr_data["type"],
                phases=phases,
                interpolation=curr_data.get("interpolation", "none"),
            )

        schedules[schedule_id] = MWCSSchedule(
            name=schedule_data["name"],
            class_weights=schedule_data.get("class_weights", {}),
            description=schedule_data.get("description"),
            curriculum=curriculum,
            sampling_strategy=schedule_data.get("sampling_strategy", "proportional"),
            sampling_params=schedule_data.get("sampling_params"),
        )

    return MWCSRegistry(
        version=data.get("version", "1.0.0"),
        schedules=schedules,
        mixture_classes=mixture_classes,
        default_schedule=data.get("default_schedule", "balanced_cb"),
    )
