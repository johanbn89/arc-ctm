"""Learned raw-pair encoder and recurrent latent operator head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arc_ctm.operators import OPERATORS


@dataclass(frozen=True, slots=True)
class CompositionModelConfig:
    grid_size: int = 5
    num_colors: int = 5
    num_color_shifts: int = 4
    pair_dim: int = 128
    evidence_dim: int = 96
    state_dim: int = 96
    hidden_dim: int = 192
    ticks: int = 4

    def __post_init__(self) -> None:
        for name in (
            "grid_size",
            "num_colors",
            "num_color_shifts",
            "pair_dim",
            "evidence_dim",
            "state_dim",
            "hidden_dim",
            "ticks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_color_shifts > self.num_colors - 1:
            raise ValueError("num_color_shifts cannot exceed num_colors-1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompositionInferenceOutput:
    support_logits: Tensor
    query_logits: Tensor
    support_intermediate_logits: Tensor
    query_intermediate_logits: Tensor
    spatial_logits: Tensor
    color_logits: Tensor
    states: Tensor


class RawPairEncoder(nn.Module):
    """Encode raw pairs with learned bilinear relational comparisons."""

    def __init__(self, config: CompositionModelConfig) -> None:
        super().__init__()
        cells = config.grid_size * config.grid_size
        grid_features = cells * config.num_colors
        self.num_colors = config.num_colors
        self.input_projection = nn.Linear(grid_features, config.pair_dim)
        self.target_projection = nn.Linear(grid_features, config.pair_dim)
        self.network = nn.Sequential(
            nn.Linear(4 * config.pair_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.pair_dim),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs_one_hot = F.one_hot(inputs, self.num_colors).to(torch.float32)
        targets_one_hot = F.one_hot(targets, self.num_colors).to(torch.float32)
        input_features = self.input_projection(inputs_one_hot.flatten(start_dim=2))
        target_features = self.target_projection(targets_one_hot.flatten(start_dim=2))
        relations = torch.cat(
            (
                input_features,
                target_features,
                input_features * target_features,
                (input_features - target_features).abs(),
            ),
            dim=-1,
        )
        pair_features = self.network(relations)
        return pair_features.mean(dim=1)


class LearnedResidualEncoder(nn.Module):
    """Learn how support prediction errors should revise the operator state."""

    def __init__(self, config: CompositionModelConfig) -> None:
        super().__init__()
        cells = config.grid_size * config.grid_size
        input_dim = 4 * cells * config.num_colors
        self.num_colors = config.num_colors
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.evidence_dim),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor, targets: Tensor, logits: Tensor) -> Tensor:
        inputs_one_hot = F.one_hot(inputs, self.num_colors).to(logits.dtype)
        targets_one_hot = F.one_hot(targets, self.num_colors).to(logits.dtype)
        probabilities = logits.softmax(dim=2).permute(0, 1, 3, 4, 2)
        residual = targets_one_hot - probabilities
        features = torch.cat(
            (inputs_one_hot, targets_one_hot, probabilities, residual), dim=-1
        )
        encoded = self.network(features.flatten(start_dim=2))
        return encoded.mean(dim=1)


class CompositionalOperatorModel(nn.Module):
    """Infer and recurrently refine a factorized executable operator state."""

    def __init__(self, config: CompositionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pair_encoder = RawPairEncoder(config)
        self.initial_state = nn.Sequential(
            nn.Linear(config.pair_dim, config.state_dim),
            nn.Tanh(),
        )
        self.residual_encoder = LearnedResidualEncoder(config)
        self.state_update = nn.GRUCell(config.evidence_dim, config.state_dim)
        self.state_normalization = nn.LayerNorm(config.state_dim)
        self.spatial_head = nn.Linear(config.state_dim, len(OPERATORS))
        self.color_head = nn.Linear(config.state_dim, config.num_color_shifts)

    def _candidate_grids(self, grids: Tensor) -> tuple[Tensor, Tensor]:
        if grids.shape[-2:] != (self.config.grid_size, self.config.grid_size):
            raise ValueError("grid shape does not match the composition model")
        spatial = torch.stack(
            [operator.transform(grids) for operator in OPERATORS], dim=2
        )
        colored = []
        for shift in range(self.config.num_color_shifts):
            shifted = ((spatial - 1 + shift) % (self.config.num_colors - 1)) + 1
            colored.append(torch.where(spatial == 0, spatial, shifted))
        colored_candidates = torch.stack(colored, dim=3)
        spatial_one_hot = F.one_hot(
            spatial, num_classes=self.config.num_colors
        ).to(torch.float32)
        colored_one_hot = F.one_hot(
            colored_candidates, num_classes=self.config.num_colors
        ).to(torch.float32)
        return spatial_one_hot, colored_one_hot

    def _decode(
        self,
        state: Tensor,
        spatial_candidates: Tensor,
        colored_candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        spatial_probabilities = self.spatial_head(state).softmax(dim=-1)
        color_probabilities = self.color_head(state).softmax(dim=-1)
        composition_probabilities = (
            spatial_probabilities.unsqueeze(-1) * color_probabilities.unsqueeze(-2)
        )
        output_probabilities = torch.einsum(
            "bsk,bnskhwc->bnhwc", composition_probabilities, colored_candidates
        )
        intermediate_probabilities = torch.einsum(
            "bs,bnshwc->bnhwc", spatial_probabilities, spatial_candidates
        )
        output_logits = output_probabilities.clamp_min(1e-8).log()
        intermediate_logits = intermediate_probabilities.clamp_min(1e-8).log()
        return (
            output_logits.permute(0, 1, 4, 2, 3).contiguous(),
            intermediate_logits.permute(0, 1, 4, 2, 3).contiguous(),
        )

    def forward(
        self,
        support_inputs: Tensor,
        support_targets: Tensor,
        query_inputs: Tensor,
    ) -> CompositionInferenceOutput:
        if support_inputs.shape != support_targets.shape:
            raise ValueError("support inputs and targets must have identical shapes")
        if support_inputs.shape[0] != query_inputs.shape[0]:
            raise ValueError("support and query batches must contain the same tasks")

        state = self.initial_state(self.pair_encoder(support_inputs, support_targets))
        support_spatial, support_composed = self._candidate_grids(support_inputs)
        query_spatial, query_composed = self._candidate_grids(query_inputs)
        support_by_tick = []
        query_by_tick = []
        support_intermediate_by_tick = []
        query_intermediate_by_tick = []
        spatial_by_tick = []
        color_by_tick = []
        states = []

        for _ in range(self.config.ticks):
            current_support, _ = self._decode(
                state, support_spatial, support_composed
            )
            evidence = self.residual_encoder(
                support_inputs, support_targets, current_support
            )
            state = self.state_normalization(self.state_update(evidence, state))
            support_logits, support_intermediate = self._decode(
                state, support_spatial, support_composed
            )
            query_logits, query_intermediate = self._decode(
                state, query_spatial, query_composed
            )
            support_by_tick.append(support_logits)
            query_by_tick.append(query_logits)
            support_intermediate_by_tick.append(support_intermediate)
            query_intermediate_by_tick.append(query_intermediate)
            spatial_by_tick.append(self.spatial_head(state))
            color_by_tick.append(self.color_head(state))
            states.append(state)

        return CompositionInferenceOutput(
            support_logits=torch.stack(support_by_tick, dim=1),
            query_logits=torch.stack(query_by_tick, dim=1),
            support_intermediate_logits=torch.stack(
                support_intermediate_by_tick, dim=1
            ),
            query_intermediate_logits=torch.stack(query_intermediate_by_tick, dim=1),
            spatial_logits=torch.stack(spatial_by_tick, dim=1),
            color_logits=torch.stack(color_by_tick, dim=1),
            states=torch.stack(states, dim=1),
        )

    def parameter_summary(self) -> dict[str, int]:
        count = sum(parameter.numel() for parameter in self.parameters())
        return {"total": count, "trainable": count}
