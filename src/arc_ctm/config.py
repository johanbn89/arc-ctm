"""Configuration objects for the synthetic operator-inference experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


NLMMode = Literal["private", "shared", "typed"]
DecoderMode = Literal["global", "cellwise"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture settings for the operator-inference CTM."""

    grid_size: int = 5
    num_colors: int = 4
    d_model: int = 64
    memory_length: int = 8
    memory_hidden: int = 16
    evidence_dim: int = 64
    task_state_dim: int = 48
    decoder_hidden: int = 128
    decoder_mode: DecoderMode = "global"
    ticks: int = 6
    sync_pairs: int = 48
    sync_self_pairs: int = 8
    nlm_mode: NLMMode = "typed"
    neuron_types: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "grid_size": self.grid_size,
            "num_colors": self.num_colors,
            "d_model": self.d_model,
            "memory_length": self.memory_length,
            "memory_hidden": self.memory_hidden,
            "evidence_dim": self.evidence_dim,
            "task_state_dim": self.task_state_dim,
            "decoder_hidden": self.decoder_hidden,
            "ticks": self.ticks,
            "sync_pairs": self.sync_pairs,
            "neuron_types": self.neuron_types,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 <= self.sync_self_pairs <= self.sync_pairs:
            raise ValueError("sync_self_pairs must be between zero and sync_pairs")
        if self.nlm_mode not in ("private", "shared", "typed"):
            raise ValueError(f"unknown nlm_mode: {self.nlm_mode}")
        if self.decoder_mode not in ("global", "cellwise"):
            raise ValueError(f"unknown decoder_mode: {self.decoder_mode}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Outer-loop meta-training settings."""

    steps: int = 500
    batch_size: int = 32
    support_size: int = 3
    query_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    support_loss_weight: float = 0.1
    operator_loss_weight: float = 0.25
    grad_clip: float = 1.0
    eval_batches: int = 8
    log_every: int = 50
    seed: int = 7

    def __post_init__(self) -> None:
        positive = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "support_size": self.support_size,
            "query_size": self.query_size,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "eval_batches": self.eval_batches,
            "log_every": self.log_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.support_loss_weight < 0:
            raise ValueError("support_loss_weight cannot be negative")
        if self.operator_loss_weight < 0:
            raise ValueError("operator_loss_weight cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
