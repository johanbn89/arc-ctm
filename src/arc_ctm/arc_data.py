"""Loading and episodic augmentation for genuine ARC-AGI-1 tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from arc_ctm.data import EpisodeBatch


@dataclass(frozen=True, slots=True)
class ArcPair:
    """One ARC input/output example."""

    input: Tensor
    output: Tensor


@dataclass(frozen=True, slots=True)
class ArcTask:
    """A parsed ARC task with demonstrations and public test targets."""

    task_id: str
    train: tuple[ArcPair, ...]
    test: tuple[ArcPair, ...]
    source: Path

    def common_square_size(self) -> int:
        """Return the common side length or fail for the first fixed-grid experiment."""

        shapes = {
            tuple(grid.shape)
            for pair in self.train + self.test
            for grid in (pair.input, pair.output)
        }
        if len(shapes) != 1:
            raise ValueError(
                f"task {self.task_id} does not have one common input/output shape: {sorted(shapes)}"
            )
        height, width = next(iter(shapes))
        if height != width:
            raise ValueError(
                f"task {self.task_id} uses rectangular {height}x{width} grids; "
                "the first overfit harness requires square grids"
            )
        return height


def _parse_grid(value: Any, label: str) -> Tensor:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError(f"{label} must be a non-empty rectangular list of lists")
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        raise ValueError(f"{label} must be rectangular")
    grid = torch.tensor(value, dtype=torch.long)
    if int(grid.min()) < 0 or int(grid.max()) > 9:
        raise ValueError(f"{label} contains a symbol outside the ARC range 0..9")
    return grid


def load_arc_task(path: Path) -> ArcTask:
    """Load and validate an official ARC JSON task."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"train", "test"}:
        raise ValueError(f"{path} must contain exactly 'train' and 'test'")

    def parse_pairs(name: str) -> tuple[ArcPair, ...]:
        raw_pairs = payload[name]
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(f"{name} must contain at least one pair")
        pairs = []
        for index, raw_pair in enumerate(raw_pairs):
            if set(raw_pair) != {"input", "output"}:
                raise ValueError(f"{name}[{index}] must contain input and output")
            pairs.append(
                ArcPair(
                    input=_parse_grid(raw_pair["input"], f"{name}[{index}].input"),
                    output=_parse_grid(raw_pair["output"], f"{name}[{index}].output"),
                )
            )
        return tuple(pairs)

    return ArcTask(
        task_id=path.stem,
        train=parse_pairs("train"),
        test=parse_pairs("test"),
        source=path.resolve(),
    )


def resolve_training_task(dataset_root: Path, task_id: str) -> Path:
    """Resolve a task id under the canonical ARC-AGI-1 training split."""

    path = dataset_root / "data" / "training" / f"{task_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"ARC task {task_id!r} was not found at {path}. "
            "Download https://github.com/fchollet/ARC-AGI first."
        )
    return path


def _color_permutations(batch_size: int, generator: torch.Generator) -> Tensor:
    permutations = []
    for _ in range(batch_size):
        permutation = torch.arange(10)
        permutation[1:] = torch.randperm(9, generator=generator) + 1
        permutations.append(permutation)
    return torch.stack(permutations)


def _apply_color_permutations(grids: Tensor, permutations: Tensor) -> Tensor:
    batch_index = torch.arange(grids.shape[0]).reshape(
        grids.shape[0], *([1] * (grids.ndim - 1))
    )
    return permutations[batch_index, grids]


def leave_one_out_batch(
    task: ArcTask,
    batch_size: int,
    generator: torch.Generator,
    *,
    augment_colors: bool,
    cover_all_folds: bool = False,
) -> EpisodeBatch:
    """Build episodes whose query target is never present in their support set."""

    train_count = len(task.train)
    if train_count < 2:
        raise ValueError("leave-one-out training needs at least two demonstrations")
    if cover_all_folds:
        if batch_size != train_count:
            raise ValueError("cover_all_folds requires one batch item per demonstration")
        held_out = torch.arange(train_count)
    else:
        held_out = torch.randint(
            train_count, (batch_size,), generator=generator, dtype=torch.long
        )

    support_inputs = []
    support_targets = []
    query_inputs = []
    query_targets = []
    for held_out_index in held_out.tolist():
        support = [
            pair for index, pair in enumerate(task.train) if index != held_out_index
        ]
        query = task.train[held_out_index]
        support_inputs.append(torch.stack([pair.input for pair in support]))
        support_targets.append(torch.stack([pair.output for pair in support]))
        query_inputs.append(query.input.unsqueeze(0))
        query_targets.append(query.output.unsqueeze(0))

    batch = EpisodeBatch(
        support_inputs=torch.stack(support_inputs),
        support_targets=torch.stack(support_targets),
        query_inputs=torch.stack(query_inputs),
        query_targets=torch.stack(query_targets),
        operator_ids=torch.zeros(batch_size, dtype=torch.long),
    )
    if not augment_colors:
        return batch

    permutations = _color_permutations(batch_size, generator)
    return EpisodeBatch(
        support_inputs=_apply_color_permutations(batch.support_inputs, permutations),
        support_targets=_apply_color_permutations(batch.support_targets, permutations),
        query_inputs=_apply_color_permutations(batch.query_inputs, permutations),
        query_targets=_apply_color_permutations(batch.query_targets, permutations),
        operator_ids=batch.operator_ids,
    )


def official_test_batch(task: ArcTask) -> EpisodeBatch:
    """Use every demonstration as support and official test inputs as queries."""

    return EpisodeBatch(
        support_inputs=torch.stack([pair.input for pair in task.train]).unsqueeze(0),
        support_targets=torch.stack([pair.output for pair in task.train]).unsqueeze(0),
        query_inputs=torch.stack([pair.input for pair in task.test]).unsqueeze(0),
        query_targets=torch.stack([pair.output for pair in task.test]).unsqueeze(0),
        operator_ids=torch.zeros(1, dtype=torch.long),
    )
