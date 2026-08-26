"""Neuron-level temporal models with configurable weight sharing."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arc_ctm.config import NLMMode


class TypedNeuronLevelModel(nn.Module):
    """Process each neuron's pre-activation history using shared temporal types.

    ``private`` assigns a separate temporal processor to every neuron, ``shared``
    uses one processor for all neurons, and ``typed`` learns a soft assignment of
    each neuron to a small bank of shared processors.
    """

    def __init__(
        self,
        d_model: int,
        memory_length: int,
        hidden_dim: int,
        mode: NLMMode,
        neuron_types: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_bases = {
            "private": d_model,
            "shared": 1,
            "typed": neuron_types,
        }[mode]

        self.first_weight = nn.Parameter(
            torch.empty(self.num_bases, memory_length, 2 * hidden_dim)
        )
        self.first_bias = nn.Parameter(torch.zeros(self.num_bases, 2 * hidden_dim))
        self.second_weight = nn.Parameter(
            torch.empty(self.num_bases, hidden_dim, 2)
        )
        self.second_bias = nn.Parameter(torch.zeros(self.num_bases, 2))
        self.neuron_gain = nn.Parameter(torch.ones(d_model))
        self.neuron_offset = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

        if mode == "typed":
            self.type_logits = nn.Parameter(torch.empty(d_model, self.num_bases))
            nn.init.normal_(self.type_logits, mean=0.0, std=0.02)
            self.register_buffer("fixed_type_weights", None)
        elif mode == "private":
            self.register_parameter("type_logits", None)
            self.register_buffer("fixed_type_weights", torch.eye(d_model))
        else:
            self.register_parameter("type_logits", None)
            self.register_buffer("fixed_type_weights", torch.ones(d_model, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound_first = math.sqrt(6.0 / (self.memory_length + 2 * self.hidden_dim))
        bound_second = math.sqrt(6.0 / (self.hidden_dim + 2))
        nn.init.uniform_(self.first_weight, -bound_first, bound_first)
        nn.init.uniform_(self.second_weight, -bound_second, bound_second)
        nn.init.zeros_(self.first_bias)
        nn.init.zeros_(self.second_bias)

    def type_weights(self) -> Tensor:
        """Return the neuron-to-temporal-type assignment matrix ``[D, K]``."""

        if self.type_logits is not None:
            return torch.softmax(self.type_logits, dim=-1)
        if self.fixed_type_weights is None:
            raise RuntimeError("fixed type assignments were not initialized")
        return self.fixed_type_weights

    def assignment_entropy(self) -> Tensor:
        """Mean normalized assignment entropy; zero for fixed assignments."""

        if self.type_logits is None or self.num_bases == 1:
            return self.first_weight.new_zeros(())
        weights = self.type_weights().clamp_min(1e-8)
        entropy = -(weights * weights.log()).sum(dim=-1)
        return (entropy / math.log(self.num_bases)).mean()

    def forward(self, history: Tensor) -> Tensor:
        if history.shape[-2:] != (self.d_model, self.memory_length):
            raise ValueError(
                "history must end with dimensions "
                f"[{self.d_model}, {self.memory_length}], got {tuple(history.shape)}"
            )

        assignments = self.type_weights()
        first_weight = torch.einsum("dk,kmh->dmh", assignments, self.first_weight)
        first_bias = torch.einsum("dk,kh->dh", assignments, self.first_bias)
        hidden = torch.einsum(
            "bdm,dmh->bdh", self.dropout(history), first_weight
        ) + first_bias
        hidden = F.glu(hidden, dim=-1)

        second_weight = torch.einsum("dk,kho->dho", assignments, self.second_weight)
        second_bias = torch.einsum("dk,ko->do", assignments, self.second_bias)
        activated = torch.einsum("bdh,dho->bdo", hidden, second_weight) + second_bias
        activated = F.glu(activated, dim=-1).squeeze(-1)
        return self.neuron_gain * activated + self.neuron_offset

