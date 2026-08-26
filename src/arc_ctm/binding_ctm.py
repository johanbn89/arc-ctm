"""Structured recurrent operator state for color-binding ARC tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arc_ctm.model import InferenceOutput
from arc_ctm.operators import OPERATORS


@dataclass(frozen=True, slots=True)
class BindingConfig:
    """Settings for a recurrent color-to-color binding matrix."""

    num_colors: int = 10
    ticks: int = 6
    memory_length: int = 4
    symmetric_prior: bool = True
    update_mode: Literal["analytic", "learned"] = "analytic"
    update_hidden: int = 16

    def __post_init__(self) -> None:
        if self.num_colors <= 1:
            raise ValueError("num_colors must be greater than one")
        if self.ticks <= 0:
            raise ValueError("ticks must be positive")
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive")
        if self.update_mode not in ("analytic", "learned"):
            raise ValueError(f"unknown update_mode: {self.update_mode}")
        if self.update_hidden <= 0:
            raise ValueError("update_hidden must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BindingCTM(nn.Module):
    """Infer an explicit color operator through shared residual updates.

    The recurrent state is a ``[num_colors, num_colors]`` logit matrix. Every
    binding edge uses the same temporal kernel and scalar update rate, making
    the update equivariant to a simultaneous relabeling of input/output colors.
    """

    def __init__(self, config: BindingConfig) -> None:
        super().__init__()
        self.config = config
        temporal_logits = torch.full((config.memory_length,), -4.0)
        temporal_logits[-1] = 4.0
        self.temporal_logits = nn.Parameter(temporal_logits)
        if config.update_mode == "analytic":
            self.log_step_size = nn.Parameter(torch.zeros(()))
            self.update_network: nn.Module | None = None
        else:
            self.register_parameter("log_step_size", None)
            self.update_network = nn.Sequential(
                nn.Linear(2, config.update_hidden),
                nn.Tanh(),
                nn.Linear(config.update_hidden, 1, bias=False),
            )

    def _decode(self, binding_logits: Tensor, grids: Tensor) -> Tensor:
        batch_size = grids.shape[0]
        batch_index = torch.arange(batch_size, device=grids.device).reshape(
            batch_size, *([1] * (grids.ndim - 1))
        )
        logits = binding_logits[batch_index, grids]
        return logits.permute(0, 1, 4, 2, 3).contiguous()

    def _directional_residual(
        self,
        binding_logits: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        source_one_hot = F.one_hot(
            source, num_classes=self.config.num_colors
        ).to(binding_logits.dtype)
        target_one_hot = F.one_hot(
            target, num_classes=self.config.num_colors
        ).to(binding_logits.dtype)
        association_counts = torch.einsum(
            "bnhwi,bnhwj->bij", source_one_hot, target_one_hot
        )
        return self._residual_from_counts(binding_logits, association_counts)

    def _residual_from_counts(
        self,
        binding_logits: Tensor,
        association_counts: Tensor,
    ) -> Tensor:
        """Compare the current binding distribution with observed associations."""

        row_counts = association_counts.sum(dim=-1)
        observed = row_counts > 0
        empirical = association_counts / row_counts.clamp_min(1.0).unsqueeze(-1)
        residual = empirical - binding_logits.softmax(dim=-1)
        return residual * observed.unsqueeze(-1)

    def _binding_residual(
        self,
        binding_logits: Tensor,
        support_inputs: Tensor,
        support_targets: Tensor,
    ) -> Tensor:
        forward = self._directional_residual(
            binding_logits, support_inputs, support_targets
        )
        if not self.config.symmetric_prior:
            return forward
        reverse = self._directional_residual(
            binding_logits, support_targets, support_inputs
        )
        return 0.5 * (forward + reverse)

    def _binding_residual_from_counts(
        self,
        binding_logits: Tensor,
        association_counts: Tensor,
    ) -> Tensor:
        forward = self._residual_from_counts(binding_logits, association_counts)
        if not self.config.symmetric_prior:
            return forward
        reverse = self._residual_from_counts(
            binding_logits, association_counts.transpose(1, 2)
        )
        return 0.5 * (forward + reverse)

    def _state_sequence(self, association_counts: Tensor) -> list[Tensor]:
        if association_counts.ndim != 3:
            raise ValueError("association_counts must have shape [batch, source, target]")
        expected = (self.config.num_colors, self.config.num_colors)
        if tuple(association_counts.shape[1:]) != expected:
            raise ValueError(
                f"association_counts must end in {expected}, got "
                f"{tuple(association_counts.shape[1:])}"
            )

        binding_logits = association_counts.new_zeros(
            association_counts.shape, dtype=torch.float32
        )
        residual_history = binding_logits.unsqueeze(-1).expand(
            -1, -1, -1, self.config.memory_length
        ).clone()
        temporal_weights = torch.softmax(self.temporal_logits, dim=0)
        step_size = (
            F.softplus(self.log_step_size)
            if self.log_step_size is not None
            else None
        )

        states: list[Tensor] = []
        for _ in range(self.config.ticks):
            residual = self._binding_residual_from_counts(
                binding_logits, association_counts
            )
            residual_history = torch.cat(
                (residual_history[..., 1:], residual.unsqueeze(-1)), dim=-1
            )
            temporal_update = torch.einsum(
                "bijm,m->bij", residual_history, temporal_weights
            )
            if self.update_network is None:
                if step_size is None:
                    raise RuntimeError("analytic update step was not initialized")
                binding_logits = binding_logits + step_size * temporal_update
            else:
                update_features = torch.stack(
                    (binding_logits, temporal_update), dim=-1
                )
                binding_logits = binding_logits + self.update_network(
                    update_features
                ).squeeze(-1)
            states.append(binding_logits)
        return states

    def forward_from_counts(
        self,
        association_counts: Tensor,
        query_inputs: Tensor,
    ) -> InferenceOutput:
        """Infer from aggregated support counts, allowing variable-size examples."""

        if association_counts.shape[0] != query_inputs.shape[0]:
            raise ValueError("association counts and queries must contain the same tasks")
        states = self._state_sequence(association_counts)
        batch_size, query_count, height, width = query_inputs.shape
        query_logits = torch.stack(
            [self._decode(state, query_inputs) for state in states], dim=1
        )
        task_states = torch.stack(
            [state.flatten(start_dim=1) for state in states], dim=1
        )
        operator_logits = query_logits.new_zeros(
            (batch_size, self.config.ticks, len(OPERATORS))
        )
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
            operator_logits=operator_logits,
        )

    def forward(
        self,
        support_inputs: Tensor,
        support_targets: Tensor,
        query_inputs: Tensor,
    ) -> InferenceOutput:
        if support_inputs.shape != support_targets.shape:
            raise ValueError("BindingCTM requires equal support input/output shapes")
        if support_inputs.shape[0] != query_inputs.shape[0]:
            raise ValueError("support and query batches must contain the same tasks")

        batch_size = support_inputs.shape[0]
        source_one_hot = F.one_hot(
            support_inputs, num_classes=self.config.num_colors
        ).to(torch.float32)
        target_one_hot = F.one_hot(
            support_targets, num_classes=self.config.num_colors
        ).to(torch.float32)
        association_counts = torch.einsum(
            "bnhwi,bnhwj->bij", source_one_hot, target_one_hot
        )
        states = self._state_sequence(association_counts)

        support_by_tick: list[Tensor] = []
        query_by_tick: list[Tensor] = []
        states_by_tick: list[Tensor] = []
        operator_by_tick: list[Tensor] = []
        for binding_logits in states:
            support_by_tick.append(self._decode(binding_logits, support_inputs))
            query_by_tick.append(self._decode(binding_logits, query_inputs))
            flattened_state = binding_logits.flatten(start_dim=1)
            states_by_tick.append(flattened_state)
            operator_by_tick.append(
                binding_logits.new_zeros((batch_size, len(OPERATORS)))
            )

        task_states = torch.stack(states_by_tick, dim=1)
        return InferenceOutput(
            support_logits=torch.stack(support_by_tick, dim=1),
            query_logits=torch.stack(query_by_tick, dim=1),
            task_states=task_states,
            activated_states=task_states,
            operator_logits=torch.stack(operator_by_tick, dim=1),
        )

    def parameter_summary(self) -> dict[str, int]:
        count = sum(parameter.numel() for parameter in self.parameters())
        return {"total": count, "trainable": count, "neuron_level_model": 0}
