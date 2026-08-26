"""Episode generation for synthetic few-shot operator inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from arc_ctm.operators import OPERATORS, apply_operators


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """A batch of tasks, each with support demonstrations and unseen queries."""

    support_inputs: Tensor
    support_targets: Tensor
    query_inputs: Tensor
    query_targets: Tensor
    operator_ids: Tensor

    def to(self, device: torch.device | str) -> EpisodeBatch:
        return EpisodeBatch(
            support_inputs=self.support_inputs.to(device),
            support_targets=self.support_targets.to(device),
            query_inputs=self.query_inputs.to(device),
            query_targets=self.query_targets.to(device),
            operator_ids=self.operator_ids.to(device),
        )


class EpisodeGenerator:
    """Stateful deterministic generator for meta-training and evaluation episodes."""

    def __init__(self, grid_size: int, num_colors: int, seed: int) -> None:
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if num_colors <= 1:
            raise ValueError("num_colors must be greater than one")
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def random_grids(self, batch_size: int, examples: int) -> Tensor:
        return torch.randint(
            low=0,
            high=self.num_colors,
            size=(batch_size, examples, self.grid_size, self.grid_size),
            generator=self.generator,
            dtype=torch.long,
        )

    def sample(
        self,
        batch_size: int,
        support_size: int,
        query_size: int,
        *,
        operator_ids: Tensor | None = None,
        query_inputs: Tensor | None = None,
    ) -> EpisodeBatch:
        if operator_ids is None:
            operator_ids = torch.randint(
                0,
                len(OPERATORS),
                (batch_size,),
                generator=self.generator,
                dtype=torch.long,
            )
        else:
            operator_ids = operator_ids.detach().cpu().to(torch.long)
            if operator_ids.shape != (batch_size,):
                raise ValueError("operator_ids must have shape [batch_size]")

        support_inputs = self.random_grids(batch_size, support_size)
        if query_inputs is None:
            query_inputs = self.random_grids(batch_size, query_size)
        else:
            query_inputs = query_inputs.detach().cpu().to(torch.long)
            expected = (batch_size, query_size, self.grid_size, self.grid_size)
            if query_inputs.shape != expected:
                raise ValueError(f"query_inputs must have shape {expected}")

        return EpisodeBatch(
            support_inputs=support_inputs,
            support_targets=apply_operators(support_inputs, operator_ids),
            query_inputs=query_inputs,
            query_targets=apply_operators(query_inputs, operator_ids),
            operator_ids=operator_ids,
        )

