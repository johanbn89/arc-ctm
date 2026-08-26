"""Metrics that distinguish demonstration fitting from operator inference."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def per_tick_accuracy(logits: Tensor, targets: Tensor) -> dict[str, list[float]]:
    """Return cell, exact-grid, and exact-task accuracy for every tick."""

    if logits.ndim != 6:
        raise ValueError("logits must have shape [B, T, N, C, H, W]")
    predictions = logits.argmax(dim=3)
    expected = targets[:, None, :, :, :]
    correct = predictions == expected
    cell = correct.to(torch.float32).mean(dim=(0, 2, 3, 4))
    grid = correct.all(dim=(-1, -2)).to(torch.float32).mean(dim=(0, 2))
    task = correct.all(dim=(-1, -2, -3)).to(torch.float32).mean(dim=0)
    return {
        "cell_accuracy": cell.detach().cpu().tolist(),
        "grid_accuracy": grid.detach().cpu().tolist(),
        "task_accuracy": task.detach().cpu().tolist(),
    }


def first_support_convergence_tick(logits: Tensor, targets: Tensor) -> Tensor:
    """Return the 1-based first tick solving every support grid, or T+1."""

    predictions = logits.argmax(dim=3)
    solved = (predictions == targets[:, None]).all(dim=(-1, -2, -3))
    ticks = logits.shape[1]
    indices = torch.arange(1, ticks + 1, device=logits.device).unsqueeze(0)
    sentinel = torch.full_like(indices, ticks + 1)
    return torch.where(solved, indices, sentinel).min(dim=1).values


def pairwise_functional_agreement(
    predictions: Tensor, group_ids: Tensor
) -> float:
    """Measure prediction agreement across independently inferred same-rule states."""

    agreements: list[Tensor] = []
    for group_id in group_ids.unique(sorted=True):
        group = predictions[group_ids == group_id]
        for left in range(group.shape[0]):
            for right in range(left + 1, group.shape[0]):
                agreements.append((group[left] == group[right]).to(torch.float32).mean())
    if not agreements:
        return 1.0
    return float(torch.stack(agreements).mean().cpu())


def between_group_functional_agreement(
    predictions: Tensor, group_ids: Tensor
) -> float:
    """Measure agreement between states inferred for different operators."""

    agreements: list[Tensor] = []
    for left in range(predictions.shape[0]):
        for right in range(left + 1, predictions.shape[0]):
            if group_ids[left] != group_ids[right]:
                agreements.append(
                    (predictions[left] == predictions[right])
                    .to(torch.float32)
                    .mean()
                )
    if not agreements:
        return 1.0
    return float(torch.stack(agreements).mean().cpu())


def pairwise_state_cosine(task_states: Tensor, group_ids: Tensor) -> float:
    """Measure cosine consistency across independently inferred same-rule states."""

    similarities: list[Tensor] = []
    for group_id in group_ids.unique(sorted=True):
        group = task_states[group_ids == group_id]
        for left in range(group.shape[0]):
            for right in range(left + 1, group.shape[0]):
                similarities.append(
                    F.cosine_similarity(group[left], group[right], dim=0)
                )
    if not similarities:
        return 1.0
    return float(torch.stack(similarities).mean().cpu())


def between_group_state_cosine(task_states: Tensor, group_ids: Tensor) -> float:
    """Measure cosine similarity between states inferred for different rules."""

    similarities: list[Tensor] = []
    for left in range(task_states.shape[0]):
        for right in range(left + 1, task_states.shape[0]):
            if group_ids[left] != group_ids[right]:
                similarities.append(
                    F.cosine_similarity(task_states[left], task_states[right], dim=0)
                )
    if not similarities:
        return 1.0
    return float(torch.stack(similarities).mean().cpu())
