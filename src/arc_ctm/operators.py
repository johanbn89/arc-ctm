"""Known synthetic operators used to isolate operator inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

import torch
from torch import Tensor


GridTransform = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """A named grid transformation with a nominal primitive complexity."""

    name: str
    complexity: int
    transform: GridTransform


def _identity(grid: Tensor) -> Tensor:
    return grid.clone()


def _rotate(grid: Tensor, turns: int) -> Tensor:
    return torch.rot90(grid, k=turns, dims=(-2, -1))


def _flip_lr(grid: Tensor) -> Tensor:
    return torch.flip(grid, dims=(-1,))


def _flip_ud(grid: Tensor) -> Tensor:
    return torch.flip(grid, dims=(-2,))


def _transpose(grid: Tensor) -> Tensor:
    return grid.transpose(-2, -1)


def _anti_transpose(grid: Tensor) -> Tensor:
    return torch.flip(grid.transpose(-2, -1), dims=(-2, -1))


OPERATORS: Final[tuple[OperatorSpec, ...]] = (
    OperatorSpec("identity", 0, _identity),
    OperatorSpec("rotate_90", 1, lambda grid: _rotate(grid, 1)),
    OperatorSpec("rotate_180", 2, lambda grid: _rotate(grid, 2)),
    OperatorSpec("rotate_270", 2, lambda grid: _rotate(grid, 3)),
    OperatorSpec("flip_left_right", 1, _flip_lr),
    OperatorSpec("flip_up_down", 1, _flip_ud),
    OperatorSpec("transpose_main", 1, _transpose),
    OperatorSpec("transpose_anti", 2, _anti_transpose),
)


def apply_operators(grids: Tensor, operator_ids: Tensor) -> Tensor:
    """Apply one operator per batch item.

    Args:
        grids: Integer tensor shaped ``[batch, examples, height, width]``.
        operator_ids: Integer tensor shaped ``[batch]``.
    """

    if grids.ndim != 4:
        raise ValueError(f"grids must have shape [B, N, H, W], got {tuple(grids.shape)}")
    if operator_ids.shape != (grids.shape[0],):
        raise ValueError("operator_ids must contain exactly one id per batch item")
    if grids.shape[-2] != grids.shape[-1]:
        raise ValueError("the current dihedral operator library requires square grids")
    if operator_ids.numel() and (
        int(operator_ids.min()) < 0 or int(operator_ids.max()) >= len(OPERATORS)
    ):
        raise ValueError("operator id is outside the synthetic operator library")

    outputs = torch.empty_like(grids)
    for operator_id, spec in enumerate(OPERATORS):
        selected = operator_ids == operator_id
        if bool(selected.any()):
            outputs[selected] = spec.transform(grids[selected])
    return outputs


def operator_names(operator_ids: Tensor) -> list[str]:
    """Return readable names for a vector of operator ids."""

    return [OPERATORS[int(operator_id)].name for operator_id in operator_ids]

