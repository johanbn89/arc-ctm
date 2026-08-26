"""Synthetic composed ARC episodes with executable supervision traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from arc_ctm.operators import OPERATORS, apply_operators


@dataclass(frozen=True, slots=True)
class CompositionSpec:
    """A spatial transform followed by a cyclic non-background recoloring."""

    spatial_id: int
    color_shift: int

    @property
    def name(self) -> str:
        return f"{OPERATORS[self.spatial_id].name} -> color_shift_{self.color_shift}"


DEFAULT_VALIDATION_SPECS = (
    CompositionSpec(1, 1),
    CompositionSpec(2, 2),
    CompositionSpec(4, 3),
    CompositionSpec(6, 1),
)


DEFAULT_HELDOUT_SPECS = (
    CompositionSpec(1, 3),
    CompositionSpec(3, 2),
    CompositionSpec(5, 1),
    CompositionSpec(7, 3),
)


@dataclass(frozen=True, slots=True)
class CompositionBatch:
    """One batch of few-shot composed-operator tasks and their traces."""

    support_inputs: Tensor
    support_intermediates: Tensor
    support_targets: Tensor
    query_inputs: Tensor
    query_intermediates: Tensor
    query_targets: Tensor
    spatial_ids: Tensor
    color_shifts: Tensor

    def to(self, device: torch.device | str) -> CompositionBatch:
        return CompositionBatch(
            support_inputs=self.support_inputs.to(device),
            support_intermediates=self.support_intermediates.to(device),
            support_targets=self.support_targets.to(device),
            query_inputs=self.query_inputs.to(device),
            query_intermediates=self.query_intermediates.to(device),
            query_targets=self.query_targets.to(device),
            spatial_ids=self.spatial_ids.to(device),
            color_shifts=self.color_shifts.to(device),
        )


def apply_color_shifts(grids: Tensor, shifts: Tensor, num_colors: int) -> Tensor:
    """Cyclically shift colors 1..C-1 while preserving background color zero."""

    if shifts.shape != (grids.shape[0],):
        raise ValueError("shifts must contain one value per batch item")
    view_shape = (grids.shape[0],) + (1,) * (grids.ndim - 1)
    expanded_shifts = shifts.reshape(view_shape)
    shifted = ((grids - 1 + expanded_shifts) % (num_colors - 1)) + 1
    return torch.where(grids == 0, grids, shifted)


def apply_compositions(
    grids: Tensor,
    spatial_ids: Tensor,
    color_shifts: Tensor,
    num_colors: int,
) -> tuple[Tensor, Tensor]:
    """Return the intermediate spatial grid and final recolored grid."""

    intermediate = apply_operators(grids, spatial_ids)
    return intermediate, apply_color_shifts(intermediate, color_shifts, num_colors)


def all_composition_specs(num_color_shifts: int) -> tuple[CompositionSpec, ...]:
    return tuple(
        CompositionSpec(spatial_id, color_shift)
        for spatial_id in range(len(OPERATORS))
        for color_shift in range(num_color_shifts)
    )


class CompositionEpisodeGenerator:
    """Generate deterministic meta-learning episodes from selected compositions."""

    def __init__(
        self,
        grid_size: int,
        num_colors: int,
        num_color_shifts: int,
        seed: int,
    ) -> None:
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if num_colors <= 2:
            raise ValueError("num_colors must be greater than two")
        if not 1 < num_color_shifts <= num_colors - 1:
            raise ValueError("num_color_shifts must be in 2..num_colors-1")
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_color_shifts = num_color_shifts
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _random_grids(self, batch_size: int, examples: int) -> Tensor:
        return torch.randint(
            0,
            self.num_colors,
            (batch_size, examples, self.grid_size, self.grid_size),
            generator=self.generator,
            dtype=torch.long,
        )

    def sample(
        self,
        batch_size: int,
        support_size: int,
        query_size: int,
        specs: Sequence[CompositionSpec],
    ) -> CompositionBatch:
        if not specs:
            raise ValueError("at least one composition spec is required")
        spec_indices = torch.randint(
            len(specs), (batch_size,), generator=self.generator
        )
        spatial_ids = torch.tensor(
            [specs[index].spatial_id for index in spec_indices.tolist()],
            dtype=torch.long,
        )
        color_shifts = torch.tensor(
            [specs[index].color_shift for index in spec_indices.tolist()],
            dtype=torch.long,
        )
        support_inputs = self._random_grids(batch_size, support_size)
        query_inputs = self._random_grids(batch_size, query_size)
        support_intermediates, support_targets = apply_compositions(
            support_inputs, spatial_ids, color_shifts, self.num_colors
        )
        query_intermediates, query_targets = apply_compositions(
            query_inputs, spatial_ids, color_shifts, self.num_colors
        )
        return CompositionBatch(
            support_inputs=support_inputs,
            support_intermediates=support_intermediates,
            support_targets=support_targets,
            query_inputs=query_inputs,
            query_intermediates=query_intermediates,
            query_targets=query_targets,
            spatial_ids=spatial_ids,
            color_shifts=color_shifts,
        )
