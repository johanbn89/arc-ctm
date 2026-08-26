"""Closed-loop typed CTM for few-shot operator inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arc_ctm.config import ModelConfig
from arc_ctm.nlm import TypedNeuronLevelModel
from arc_ctm.operators import OPERATORS


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Per-tick predictions and task states."""

    support_logits: Tensor
    query_logits: Tensor
    task_states: Tensor
    activated_states: Tensor
    operator_logits: Tensor


class ConditionalGridDecoder(nn.Module):
    """Apply one inferred task state to any grid in an episode."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.grid_size = config.grid_size
        self.num_colors = config.num_colors
        grid_features = config.grid_size * config.grid_size * config.num_colors
        self.network = nn.Sequential(
            nn.Linear(grid_features + config.task_state_dim, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, grid_features),
        )

    def forward(self, task_state: Tensor, grids: Tensor) -> Tensor:
        batch_size, examples, height, width = grids.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError("grid shape does not match the model configuration")
        one_hot = F.one_hot(grids, num_classes=self.num_colors).to(task_state.dtype)
        flat_grids = one_hot.reshape(batch_size * examples, -1)
        repeated_state = task_state[:, None, :].expand(-1, examples, -1)
        repeated_state = repeated_state.reshape(batch_size * examples, -1)
        logits = self.network(torch.cat((flat_grids, repeated_state), dim=-1))
        logits = logits.reshape(
            batch_size,
            examples,
            height,
            width,
            self.num_colors,
        )
        return logits.permute(0, 1, 4, 2, 3).contiguous()


class ConditionalCellwiseDecoder(nn.Module):
    """Apply the same task-conditioned color rule independently at every cell."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_colors = config.num_colors
        self.network = nn.Sequential(
            nn.Linear(config.num_colors + config.task_state_dim, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, config.num_colors),
        )

    def forward(self, task_state: Tensor, grids: Tensor) -> Tensor:
        batch_size, examples, height, width = grids.shape
        one_hot = F.one_hot(grids, num_classes=self.num_colors).to(task_state.dtype)
        repeated_state = task_state[:, None, None, None, :].expand(
            -1, examples, height, width, -1
        )
        logits = self.network(torch.cat((one_hot, repeated_state), dim=-1))
        return logits.permute(0, 1, 4, 2, 3).contiguous()


class ResidualEvidenceEncoder(nn.Module):
    """Encode and permutation-invariantly pool support prediction failures."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cells = config.grid_size * config.grid_size
        input_dim = cells * config.num_colors * 4
        self.num_colors = config.num_colors
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.evidence_dim * 2),
            nn.GELU(),
            nn.Linear(config.evidence_dim * 2, config.evidence_dim),
            nn.LayerNorm(config.evidence_dim),
        )

    def forward(self, inputs: Tensor, targets: Tensor, logits: Tensor) -> Tensor:
        input_one_hot = F.one_hot(inputs, self.num_colors).to(logits.dtype)
        target_one_hot = F.one_hot(targets, self.num_colors).to(logits.dtype)
        probabilities = logits.softmax(dim=2).permute(0, 1, 3, 4, 2)
        residual = target_one_hot - probabilities
        features = torch.cat(
            (input_one_hot, probabilities, target_one_hot, residual), dim=-1
        )
        encoded = self.network(features.flatten(start_dim=2))
        return encoded.mean(dim=1)


class PairwiseSynchronizer(nn.Module):
    """Maintain exponentially decayed pairwise neuron synchronization traces."""

    def __init__(self, d_model: int, pairs: int, self_pairs: int) -> None:
        super().__init__()
        left = torch.arange(pairs, dtype=torch.long) % d_model
        right = (left * 7 + 3) % d_model
        right[:self_pairs] = left[:self_pairs]
        self.register_buffer("left_indices", left)
        self.register_buffer("right_indices", right)
        self.decay_rates = nn.Parameter(torch.full((pairs,), -1.5))

    def _products(self, activated_state: Tensor) -> Tensor:
        return (
            activated_state[:, self.left_indices]
            * activated_state[:, self.right_indices]
        )

    def start(self, activated_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        products = self._products(activated_state)
        denominator = torch.ones_like(products)
        return products, products, denominator

    def update(
        self,
        activated_state: Tensor,
        numerator: Tensor,
        denominator: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        decay = torch.exp(-F.softplus(self.decay_rates)).unsqueeze(0)
        numerator = decay * numerator + self._products(activated_state)
        denominator = decay * denominator + 1.0
        synchronization = numerator / denominator.sqrt()
        return synchronization, numerator, denominator


class OperatorInferenceCTM(nn.Module):
    """Infer a reusable operator state through residual-guided recurrent ticks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.decoder_mode == "cellwise":
            self.decoder = ConditionalCellwiseDecoder(config)
        else:
            self.decoder = ConditionalGridDecoder(config)
        self.evidence_encoder = ResidualEvidenceEncoder(config)
        self.synapses = nn.Sequential(
            nn.Linear(config.d_model + config.evidence_dim, 2 * config.d_model),
            nn.GLU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.GLU(),
            nn.LayerNorm(config.d_model),
        )
        self.neuron_model = TypedNeuronLevelModel(
            d_model=config.d_model,
            memory_length=config.memory_length,
            hidden_dim=config.memory_hidden,
            mode=config.nlm_mode,
            neuron_types=config.neuron_types,
            dropout=config.dropout,
        )
        self.synchronizer = PairwiseSynchronizer(
            d_model=config.d_model,
            pairs=config.sync_pairs,
            self_pairs=config.sync_self_pairs,
        )
        self.task_state_projector = nn.Sequential(
            # Pairwise products begin at a small scale. Normalization prevents the
            # projector bias from overwhelming task-specific synchronization.
            nn.LayerNorm(config.sync_pairs),
            nn.Linear(config.sync_pairs, config.task_state_dim, bias=False),
            nn.Tanh(),
        )
        self.operator_probe = nn.Linear(config.task_state_dim, len(OPERATORS))
        start_scale = (config.d_model + config.memory_length) ** -0.5
        self.start_trace = nn.Parameter(
            torch.empty(config.d_model, config.memory_length).uniform_(
                -start_scale, start_scale
            )
        )
        self.start_activated_state = nn.Parameter(
            torch.empty(config.d_model).uniform_(-start_scale, start_scale)
        )

    def forward(
        self,
        support_inputs: Tensor,
        support_targets: Tensor,
        query_inputs: Tensor,
    ) -> InferenceOutput:
        if support_inputs.shape != support_targets.shape:
            raise ValueError("support inputs and targets must have identical shapes")
        if support_inputs.ndim != 4 or query_inputs.ndim != 4:
            raise ValueError("support and query tensors must have shape [B, N, H, W]")
        if support_inputs.shape[0] != query_inputs.shape[0]:
            raise ValueError("support and query batches must contain the same tasks")

        batch_size = support_inputs.shape[0]
        history = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(
            batch_size, -1
        )
        synchronization, numerator, denominator = self.synchronizer.start(
            activated_state
        )

        support_by_tick: list[Tensor] = []
        query_by_tick: list[Tensor] = []
        task_states: list[Tensor] = []
        activated_states: list[Tensor] = []
        operator_by_tick: list[Tensor] = []

        for _ in range(self.config.ticks):
            current_task_state = self.task_state_projector(synchronization)
            current_support_logits = self.decoder(
                current_task_state, support_inputs
            )
            evidence = self.evidence_encoder(
                support_inputs, support_targets, current_support_logits
            )
            pre_activation = self.synapses(
                torch.cat((activated_state, evidence), dim=-1)
            )
            history = torch.cat(
                (history[:, :, 1:], pre_activation.unsqueeze(-1)), dim=-1
            )
            activated_state = self.neuron_model(history)
            synchronization, numerator, denominator = self.synchronizer.update(
                activated_state, numerator, denominator
            )
            next_task_state = self.task_state_projector(synchronization)

            support_by_tick.append(self.decoder(next_task_state, support_inputs))
            query_by_tick.append(self.decoder(next_task_state, query_inputs))
            task_states.append(next_task_state)
            activated_states.append(activated_state)
            operator_by_tick.append(self.operator_probe(next_task_state))

        return InferenceOutput(
            support_logits=torch.stack(support_by_tick, dim=1),
            query_logits=torch.stack(query_by_tick, dim=1),
            task_states=torch.stack(task_states, dim=1),
            activated_states=torch.stack(activated_states, dim=1),
            operator_logits=torch.stack(operator_by_tick, dim=1),
        )

    def parameter_summary(self) -> dict[str, int]:
        """Count total and temporal NLM parameters for sharing comparisons."""

        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "neuron_level_model": sum(
                parameter.numel() for parameter in self.neuron_model.parameters()
            ),
        }
