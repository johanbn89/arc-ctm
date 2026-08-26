"""Structured recurrent hypothesis state for spatial ARC transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arc_ctm.model import InferenceOutput
from arc_ctm.operators import OPERATORS


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    """Settings for recurrent inference over a fixed spatial operator bank."""

    num_colors: int = 10
    ticks: int = 6
    memory_length: int = 4
    evidence_scale: float = 8.0
    update_hidden: int = 16

    def __post_init__(self) -> None:
        if self.num_colors <= 1:
            raise ValueError("num_colors must be greater than one")
        if self.ticks <= 0:
            raise ValueError("ticks must be positive")
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive")
        if self.evidence_scale <= 0:
            raise ValueError("evidence_scale must be positive")
        if self.update_hidden <= 0:
            raise ValueError("update_hidden must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def spatial_evidence(
    sources: Sequence[Tensor], targets: Sequence[Tensor]
) -> Tensor:
    """Return negative mismatch counts for every spatial hypothesis."""

    if len(sources) != len(targets) or not sources:
        raise ValueError("spatial evidence needs equally many non-empty sources and targets")
    scores = torch.zeros(len(OPERATORS), dtype=torch.float32)
    for source, target in zip(sources, targets):
        for operator_id, operator in enumerate(OPERATORS):
            candidate = operator.transform(source)
            if candidate.shape != target.shape:
                scores[operator_id] -= float(source.numel() + target.numel())
            else:
                scores[operator_id] -= float((candidate != target).sum())
    return scores


class SpatialCTM(nn.Module):
    """Infer one of eight dihedral transforms with a shared recurrent update."""

    def __init__(self, config: SpatialConfig) -> None:
        super().__init__()
        self.config = config
        temporal_logits = torch.full((config.memory_length,), -4.0)
        temporal_logits[-1] = 4.0
        self.temporal_logits = nn.Parameter(temporal_logits)
        self.update_network = nn.Sequential(
            nn.Linear(2, config.update_hidden),
            nn.Tanh(),
            nn.Linear(config.update_hidden, 1, bias=False),
        )

    def _state_sequence(self, evidence_logits: Tensor) -> list[Tensor]:
        if evidence_logits.ndim != 2 or evidence_logits.shape[1] != len(OPERATORS):
            raise ValueError(
                f"evidence_logits must have shape [batch, {len(OPERATORS)}]"
            )
        state_logits = evidence_logits.new_zeros(evidence_logits.shape)
        evidence_distribution = torch.softmax(
            evidence_logits * self.config.evidence_scale, dim=-1
        )
        residual_history = state_logits.unsqueeze(-1).expand(
            -1, -1, self.config.memory_length
        ).clone()
        temporal_weights = torch.softmax(self.temporal_logits, dim=0)
        states = []
        for _ in range(self.config.ticks):
            residual = evidence_distribution - state_logits.softmax(dim=-1)
            residual_history = torch.cat(
                (residual_history[..., 1:], residual.unsqueeze(-1)), dim=-1
            )
            temporal_update = torch.einsum(
                "bkm,m->bk", residual_history, temporal_weights
            )
            update_features = torch.stack((state_logits, temporal_update), dim=-1)
            state_logits = state_logits + self.update_network(update_features).squeeze(-1)
            states.append(state_logits)
        return states

    def _decode(self, state_logits: Tensor, query_inputs: Tensor) -> Tensor:
        if query_inputs.shape[-2] != query_inputs.shape[-1]:
            raise ValueError("the dihedral decoder currently requires square query grids")
        candidates = torch.stack(
            [operator.transform(query_inputs) for operator in OPERATORS], dim=2
        )
        candidate_colors = F.one_hot(
            candidates, num_classes=self.config.num_colors
        ).to(state_logits.dtype)
        probabilities = torch.einsum(
            "bk,bnkhwc->bnhwc", state_logits.softmax(dim=-1), candidate_colors
        )
        return probabilities.clamp_min(1e-8).log().permute(0, 1, 4, 2, 3)

    def forward(self, evidence_logits: Tensor, query_inputs: Tensor) -> InferenceOutput:
        if evidence_logits.shape[0] != query_inputs.shape[0]:
            raise ValueError("evidence and queries must contain the same tasks")
        states = self._state_sequence(evidence_logits)
        query_logits = torch.stack(
            [self._decode(state, query_inputs) for state in states], dim=1
        )
        task_states = torch.stack(states, dim=1)
        batch_size, _, height, width = query_inputs.shape
        support_logits = query_logits.new_empty(
            (
                batch_size,
                self.config.ticks,
                0,
                self.config.num_colors,
                height,
                width,
            )
        )
        return InferenceOutput(
            support_logits=support_logits,
            query_logits=query_logits,
            task_states=task_states,
            activated_states=task_states,
            operator_logits=task_states,
        )

    def parameter_summary(self) -> dict[str, int]:
        count = sum(parameter.numel() for parameter in self.parameters())
        return {"total": count, "trainable": count, "neuron_level_model": 0}
