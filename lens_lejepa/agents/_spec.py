"""Typed specifications for an agent campaign.

``DownstreamSpec`` holds exactly the fields the planner may change.
``CampaignSpec`` holds everything else — the task, the data, the label budget,
the seed — plus the pool of pretrained encoders the planner may choose from.
The separation is the budget-fairness guarantee: the planner can change how a
model is trained, never what it is trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..models.adapters import canonical_adapter_name


class EncoderSpec(BaseModel):
    """One pretrained encoder in the pool."""

    name: str = Field(description="Identifier the planner uses to select it.")
    backbone: str = Field(description="Encoder name, e.g. vit_small.")
    pretrained: str | None = Field(default=None, description="Path to last_pretrain.pt.")
    method: str = Field(default="", description="Pretraining method that produced it.")


class DownstreamSpec(BaseModel):
    """The fields the planner is allowed to change."""

    encoder: str = "lens_lejepa"
    adaptation: str = "rslora"
    rank: int = 32
    alpha: float = 64.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.0
    epochs: int = 50
    batch_size: int = 128
    label_smoothing: float = 0.1


class CampaignSpec(BaseModel):
    """Everything the planner may not change, plus the pool it may choose from."""

    name: str = "campaign"
    hypothesis: str = ""
    task: str = "classification"
    data_root: str = "../datasets"
    train_dataset: str = "Model_II"
    test_dataset: str | None = "Model_II_test"
    shots: int | None = None
    train_fraction: float = 0.8
    val_fraction: float = 0.25
    cache: bool = False
    image_size: int = 160
    patch_size: int = 16
    seed: int = 42
    val_interval: int = 10
    device: str = "cuda"
    bf16: bool = True
    workers: int = 8
    grad_accum: int = 1
    loraplus_ratio: float = 2.0
    train_jitter: int = 0
    rank_schedule: list[int] = [8, 16, 32, 64]
    output_dir: str = "runs/agent"
    max_iterations: int = 8
    patience: int = 3
    min_val_delta: float = 0.005
    min_val_above_chance: float = 0.05
    encoders: list[EncoderSpec] = [EncoderSpec(name="vit_small", backbone="vit_small")]
    initial: DownstreamSpec = DownstreamSpec()

    def encoder(self, name: str) -> EncoderSpec:
        """The pooled encoder called ``name``."""
        for item in self.encoders:
            if item.name == name:
                return item
        raise KeyError(f"No encoder named {name!r} in the pool {[e.name for e in self.encoders]}.")

    def run_name_for(self, iteration: int, spec: DownstreamSpec) -> str:
        """A unique run-directory name for one iteration."""
        return f"iter{iteration:02d}_{spec.encoder}_{spec.adaptation}"

    @property
    def output_path(self) -> Path:
        """Where the campaign artifacts are written."""
        return Path(self.output_dir) / self.name


#: Allowed range for every numeric field the planner may set.
DOMAINS: dict[str, tuple[float, float]] = {
    "rank": (1, 256),
    "alpha": (0.0, 1024.0),
    "lr": (1e-6, 1e-1),
    "weight_decay": (0.0, 1.0),
    "dropout": (0.0, 0.999),
    "epochs": (1, 200),
    "batch_size": (1, 4096),
    "label_smoothing": (0.0, 1.0),
}

#: Fields that must be whole numbers.
INTEGER_FIELDS = ("rank", "epochs", "batch_size")


def validate_delta(
    spec: DownstreamSpec,
    updates: dict[str, Any],
    campaign: CampaignSpec | None = None,
) -> DownstreamSpec:
    """Apply the planner's updates to ``spec``.

    Raises ``ValueError`` for a key we do not have, a key that is fixed for the
    whole campaign, or a value outside its domain. The loop catches the error,
    records the reason and shows it to the planner on the next turn, so a
    change is never silently dropped.
    """
    allowed = set(DownstreamSpec.model_fields)
    frozen = set(CampaignSpec.model_fields) - allowed
    for key, value in (updates or {}).items():
        if key in frozen:
            raise ValueError(f"{key!r} is fixed for the whole campaign and cannot be changed.")
        if key not in allowed:
            raise ValueError(f"Unknown parameter {key!r}; allowed: {sorted(allowed)}.")
        if key == "encoder":
            if campaign is not None:
                try:
                    campaign.encoder(str(value))
                except KeyError as error:
                    raise ValueError(f"encoder {value!r} is not in the pool.") from error
            continue
        if key == "adaptation":
            canonical_adapter_name(str(value))
            continue
        low, high = DOMAINS[key]
        number = float(value)
        if not low <= number <= high:
            raise ValueError(f"{key}={value!r} is outside the allowed range [{low}, {high}].")
        if key in INTEGER_FIELDS and int(number) != number:
            raise ValueError(f"{key}={value!r} must be a whole number.")
    data = spec.model_dump()
    data.update(updates)
    return DownstreamSpec(**data)
