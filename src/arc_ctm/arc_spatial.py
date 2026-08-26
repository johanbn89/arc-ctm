"""Task-level transfer across genuine ARC spatial-transformation tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from arc_ctm.arc_data import ArcPair, ArcTask, load_arc_task, resolve_training_task
from arc_ctm.arc_viz import write_multitask_report
from arc_ctm.experiment import tick_cross_entropy
from arc_ctm.operators import OPERATORS
from arc_ctm.spatial_ctm import SpatialCTM, SpatialConfig, spatial_evidence


DEFAULT_TASK_IDS = (
    "3c9b0459",
    "6150a2bd",
    "67a3c6ac",
    "68b16354",
    "74dd1130",
    "9dfd6313",
    "ed36ccf7",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _color_permutation(generator: torch.Generator, *, augment: bool) -> Tensor:
    permutation = torch.arange(10)
    if augment:
        permutation[1:] = torch.randperm(9, generator=generator) + 1
    return permutation


def _matching_operators(pairs: Sequence[ArcPair]) -> set[int]:
    matches = set(range(len(OPERATORS)))
    for pair in pairs:
        matches = {
            operator_id
            for operator_id in matches
            if OPERATORS[operator_id].transform(pair.input).shape == pair.output.shape
            and torch.equal(OPERATORS[operator_id].transform(pair.input), pair.output)
        }
    return matches


def _task_operator(task: ArcTask) -> int:
    matches = _matching_operators(task.train)
    if len(matches) != 1:
        raise ValueError(
            f"task {task.task_id} does not identify exactly one spatial operator: "
            f"{sorted(matches)}"
        )
    operator_id = next(iter(matches))
    if operator_id == 0:
        raise ValueError(f"task {task.task_id} is only the identity transform")
    return operator_id


def _eligible_folds(task: ArcTask) -> tuple[int, ...]:
    operator_id = _task_operator(task)
    folds = []
    for held_out_index in range(len(task.train)):
        support = tuple(
            pair for index, pair in enumerate(task.train) if index != held_out_index
        )
        if _matching_operators(support) == {operator_id}:
            folds.append(held_out_index)
    if not folds:
        raise ValueError(f"task {task.task_id} has no identifiable leave-one-out fold")
    return tuple(folds)


def _validate_task(task: ArcTask) -> dict[str, Any]:
    operator_id = _task_operator(task)
    for pair_index, pair in enumerate(task.train):
        if pair.input.shape[-2] != pair.input.shape[-1]:
            raise ValueError(
                f"task {task.task_id} train[{pair_index}] is not square"
            )
    return {
        "operator_from_train_only": OPERATORS[operator_id].name,
        "operator_id": operator_id,
        "eligible_training_folds": list(_eligible_folds(task)),
    }


def _permuted(grid: Tensor, permutation: Tensor) -> Tensor:
    return permutation[grid]


def _evidence(
    pairs: Sequence[ArcPair], permutation: Tensor, device: torch.device
) -> Tensor:
    sources = [_permuted(pair.input, permutation) for pair in pairs]
    targets = [_permuted(pair.output, permutation) for pair in pairs]
    return spatial_evidence(sources, targets).unsqueeze(0).to(device)


def _episode(
    task: ArcTask,
    held_out_index: int,
    permutation: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    support = tuple(
        pair for index, pair in enumerate(task.train) if index != held_out_index
    )
    query = task.train[held_out_index]
    return (
        _evidence(support, permutation, device),
        _permuted(query.input, permutation).unsqueeze(0).unsqueeze(0).to(device),
        _permuted(query.output, permutation).unsqueeze(0).unsqueeze(0).to(device),
    )


def _discrete_prediction(output: Any, query: Tensor) -> tuple[Tensor, int]:
    """Execute the most probable spatial program at the final recurrent tick."""

    operator_id = int(output.operator_logits[0, -1].argmax().detach().cpu())
    return OPERATORS[operator_id].transform(query[0, 0]), operator_id


@torch.no_grad()
def _evaluate_pairs(
    model: SpatialCTM,
    evidence: Tensor,
    pairs: Sequence[ArcPair],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct_cells = 0
    total_cells = 0
    exact_grids = 0
    predictions: list[list[list[int]]] = []
    operator_predictions = []
    for pair in pairs:
        query = pair.input.unsqueeze(0).unsqueeze(0).to(device)
        target = pair.output.to(device)
        output = model(evidence, query)
        prediction, operator_id = _discrete_prediction(output, query)
        correct = prediction == target
        correct_cells += int(correct.sum().cpu())
        total_cells += correct.numel()
        exact_grids += int(correct.all().cpu())
        predictions.append(prediction.cpu().tolist())
        operator_predictions.append(OPERATORS[operator_id].name)
    return {
        "cell_accuracy": correct_cells / total_cells,
        "exact_grid_accuracy": exact_grids / len(pairs),
        "all_exact": exact_grids == len(pairs),
        "predictions": predictions,
        "operator_predictions": operator_predictions,
    }


@torch.no_grad()
def _evaluate_official(
    model: SpatialCTM, task: ArcTask, device: torch.device
) -> dict[str, Any]:
    evidence = _evidence(task.train, torch.arange(10), device)
    return _evaluate_pairs(model, evidence, task.test, device)


@torch.no_grad()
def _evaluate_randomized_loo(
    model: SpatialCTM,
    task: ArcTask,
    episodes: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    folds = _eligible_folds(task)
    correct_cells = 0
    total_cells = 0
    exact_grids = 0
    correct_operators = 0
    expected_operator = _task_operator(task)
    for _ in range(episodes):
        fold_index = int(torch.randint(len(folds), (), generator=generator))
        permutation = _color_permutation(generator, augment=True)
        evidence, query, target = _episode(
            task, folds[fold_index], permutation, device
        )
        output = model(evidence, query)
        prediction, operator_id = _discrete_prediction(output, query)
        target_grid = target[0, 0]
        correct = prediction == target_grid
        correct_cells += int(correct.sum().cpu())
        total_cells += correct.numel()
        exact_grids += int(correct.all().cpu())
        correct_operators += int(operator_id == expected_operator)
    return {
        "episodes": episodes,
        "cell_accuracy": correct_cells / total_cells,
        "exact_grid_accuracy": exact_grids / episodes,
        "operator_accuracy": correct_operators / episodes,
    }


@torch.no_grad()
def _evaluate_corrupted_support(
    model: SpatialCTM, task: ArcTask, device: torch.device
) -> dict[str, Any]:
    identity = torch.arange(10)
    correct_evidence = _evidence(task.train, identity, device)
    corrupted_sources = [pair.input for pair in task.train]
    corrupted_targets = [torch.flip(pair.output, dims=(-1,)) for pair in task.train]
    corrupted_evidence = spatial_evidence(
        corrupted_sources, corrupted_targets
    ).unsqueeze(0).to(device)
    metrics = _evaluate_pairs(model, corrupted_evidence, task.test, device)
    changed_cells = 0
    total_cells = 0
    for pair in task.test:
        query = pair.input.unsqueeze(0).unsqueeze(0).to(device)
        correct_output = model(correct_evidence, query)
        corrupted_output = model(corrupted_evidence, query)
        correct_prediction, _ = _discrete_prediction(correct_output, query)
        corrupted_prediction, _ = _discrete_prediction(corrupted_output, query)
        changed_cells += int((correct_prediction != corrupted_prediction).sum().cpu())
        total_cells += correct_prediction.numel()
    metrics["official_query_prediction_sensitivity"] = changed_cells / total_cells

    probe = torch.arange(9, dtype=torch.long, device=device).reshape(1, 1, 3, 3)
    correct_probe, _ = _discrete_prediction(model(correct_evidence, probe), probe)
    corrupted_probe, _ = _discrete_prediction(model(corrupted_evidence, probe), probe)
    metrics["prediction_sensitivity"] = float(
        (correct_probe != corrupted_probe).to(torch.float32).mean().cpu()
    )
    metrics["corruption"] = "flip every support output left-right"
    return metrics


def _train_fold(
    model: SpatialCTM,
    training_tasks: Sequence[ArcTask],
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    task_folds = {task.task_id: _eligible_folds(task) for task in training_tasks}
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        losses = []
        for _ in range(args.episodes_per_step):
            task_index = int(
                torch.randint(len(training_tasks), (), generator=generator)
            )
            task = training_tasks[task_index]
            folds = task_folds[task.task_id]
            held_out_index = folds[
                int(torch.randint(len(folds), (), generator=generator))
            ]
            permutation = _color_permutation(generator, augment=True)
            evidence, query, target = _episode(
                task, held_out_index, permutation, device
            )
            output = model(evidence, query)
            losses.append(tick_cross_entropy(output.query_logits, target))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.task_ids) < 2:
        raise ValueError("leave-one-task-out evaluation needs at least two task IDs")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    tasks = {
        task_id: load_arc_task(resolve_training_task(args.dataset_root, task_id))
        for task_id in args.task_ids
    }
    task_metadata = {
        task_id: _validate_task(task) for task_id, task in tasks.items()
    }
    config = SpatialConfig(
        num_colors=10,
        ticks=args.ticks,
        memory_length=args.memory_length,
        evidence_scale=args.evidence_scale,
        update_hidden=args.update_hidden,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    for holdout_index, holdout_id in enumerate(args.task_ids):
        fold_seed = args.seed + holdout_index * 10_000
        torch.manual_seed(fold_seed)
        model = SpatialCTM(config).to(device)
        holdout_task = tasks[holdout_id]
        training_tasks = [
            tasks[task_id] for task_id in args.task_ids if task_id != holdout_id
        ]
        paired_eval_seed = fold_seed + 1_000_000
        untrained = _evaluate_official(model, holdout_task, device)
        untrained_randomized = _evaluate_randomized_loo(
            model,
            holdout_task,
            args.randomized_eval_episodes,
            torch.Generator(device="cpu").manual_seed(paired_eval_seed),
            device,
        )
        print(
            f"holdout={holdout_id} operator="
            f"{task_metadata[holdout_id]['operator_from_train_only']} training_on="
            f"{','.join(task.task_id for task in training_tasks)}",
            flush=True,
        )
        history = _train_fold(
            model,
            training_tasks,
            args,
            torch.Generator(device="cpu").manual_seed(fold_seed + 1),
            device,
        )
        trained = _evaluate_official(model, holdout_task, device)
        randomized = _evaluate_randomized_loo(
            model,
            holdout_task,
            args.randomized_eval_episodes,
            torch.Generator(device="cpu").manual_seed(paired_eval_seed),
            device,
        )
        corrupted = _evaluate_corrupted_support(model, holdout_task, device)
        fold_result = {
            "holdout_task_id": holdout_id,
            "heldout_operator": task_metadata[holdout_id]["operator_from_train_only"],
            "training_task_ids": [task.task_id for task in training_tasks],
            "holdout_task_id_seen_during_training": False,
            "holdout_test_target_used_for_training": False,
            "holdout_demonstrations_used_at_inference": True,
            "seed": fold_seed,
            "history": history,
            "untrained_official_test": untrained,
            "untrained_randomized_holdout_leave_one_out": untrained_randomized,
            "trained_official_test": trained,
            "randomized_holdout_leave_one_out": randomized,
            "corrupted_holdout_support": corrupted,
        }
        folds.append(fold_result)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": config.to_dict(),
                "training_task_ids": fold_result["training_task_ids"],
                "holdout_task_id": holdout_id,
            },
            args.output / f"holdout_{holdout_id}_model.pt",
        )
        print(
            json.dumps(
                {
                    "holdout": holdout_id,
                    "official_exact": trained["all_exact"],
                    "official_cell": trained["cell_accuracy"],
                    "operator": trained["operator_predictions"],
                    "randomized_exact": randomized["exact_grid_accuracy"],
                    "corrupted_exact": corrupted["exact_grid_accuracy"],
                    "support_sensitivity": corrupted["prediction_sensitivity"],
                }
            ),
            flush=True,
        )

    official_exact = [
        float(fold["trained_official_test"]["all_exact"]) for fold in folds
    ]
    randomized_exact = [
        fold["randomized_holdout_leave_one_out"]["exact_grid_accuracy"]
        for fold in folds
    ]
    results: dict[str, Any] = {
        "experiment": "leave-one-task-out learned spatial-hypothesis updater",
        "dataset_commit": args.dataset_commit,
        "task_ids": list(args.task_ids),
        "task_selection_uses_test_outputs": False,
        "selection_rule": (
            "training demonstrations uniquely match one non-identity dihedral transform"
        ),
        "task_metadata": task_metadata,
        "model_config": config.to_dict(),
        "parameter_count": SpatialCTM(config).parameter_summary(),
        "operator_library": [operator.name for operator in OPERATORS],
        "training": {
            "steps_per_fold": args.steps,
            "episodes_per_step": args.episodes_per_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "color_augmentation": True,
        },
        "folds": folds,
        "summary": {
            "official_tasks_exact": int(sum(official_exact)),
            "official_tasks_total": len(folds),
            "official_task_exact_rate": sum(official_exact) / len(folds),
            "mean_randomized_holdout_exact_grid_accuracy": sum(randomized_exact)
            / len(randomized_exact),
        },
    }
    result_path = args.output / "results.json"
    report_path = args.output / "report.html"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_multitask_report(tasks, results, report_path)
    print(f"results: {result_path.resolve()}")
    print(f"report: {report_path.resolve()}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recurrent spatial-hypothesis updater on genuine ARC tasks "
            "and test completely held-out task IDs."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/arc_agi_1")
    )
    parser.add_argument(
        "--dataset-commit", default="399030444e0ab0cc8b4e199870fb20b863846f34"
    )
    parser.add_argument("--task-ids", nargs="+", default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--memory-length", type=int, default=4)
    parser.add_argument("--evidence-scale", type=float, default=8.0)
    parser.add_argument("--update-hidden", type=int, default=16)
    parser.add_argument("--randomized-eval-episodes", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/arc_spatial")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
