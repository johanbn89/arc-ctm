"""Leave-one-task-out transfer for genuine ARC color-substitution tasks."""

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
from arc_ctm.binding_ctm import BindingCTM, BindingConfig
from arc_ctm.experiment import tick_cross_entropy


DEFAULT_TASK_IDS = ("0d3d703e", "b1948b0a", "c8f0f002", "d511f180")


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


def _association_counts(
    pairs: Sequence[ArcPair], permutation: Tensor, device: torch.device
) -> Tensor:
    counts = torch.zeros((10, 10), dtype=torch.float32)
    for pair in pairs:
        source = permutation[pair.input].flatten()
        target = permutation[pair.output].flatten()
        indices = source * 10 + target
        counts += torch.bincount(indices, minlength=100).reshape(10, 10)
    return counts.unsqueeze(0).to(device)


def _permuted_grid(grid: Tensor, permutation: Tensor, device: torch.device) -> Tensor:
    return permutation[grid].unsqueeze(0).unsqueeze(0).to(device)


def _training_mapping(task: ArcTask) -> dict[int, int]:
    """Infer and validate a cellwise mapping from demonstrations only."""

    mapping: dict[int, int] = {}
    for pair_index, pair in enumerate(task.train):
        if pair.input.shape != pair.output.shape:
            raise ValueError(
                f"task {task.task_id} train[{pair_index}] changes grid shape"
            )
        for source, target in zip(pair.input.flatten(), pair.output.flatten()):
            source_value = int(source)
            target_value = int(target)
            previous = mapping.setdefault(source_value, target_value)
            if previous != target_value:
                raise ValueError(
                    f"task {task.task_id} is not one deterministic cellwise color mapping"
                )
    return mapping


def _eligible_folds(task: ArcTask) -> tuple[int, ...]:
    eligible = []
    for held_out_index, query in enumerate(task.train):
        support_colors = {
            int(color)
            for index, pair in enumerate(task.train)
            if index != held_out_index
            for color in pair.input.unique()
        }
        query_colors = {int(color) for color in query.input.unique()}
        if query_colors <= support_colors:
            eligible.append(held_out_index)
    if not eligible:
        raise ValueError(f"task {task.task_id} has no identifiable leave-one-out fold")
    return tuple(eligible)


def _validate_task(task: ArcTask) -> dict[str, Any]:
    mapping = _training_mapping(task)
    test_input_colors = {
        int(color) for pair in task.test for color in pair.input.unique()
    }
    unseen = sorted(test_input_colors - mapping.keys())
    if unseen:
        raise ValueError(
            f"task {task.task_id} has test-input colors absent from demonstrations: {unseen}"
        )
    return {
        "mapping_from_train_only": {str(key): value for key, value in mapping.items()},
        "eligible_training_folds": list(_eligible_folds(task)),
        "test_input_colors_covered_by_train": True,
    }


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
        _association_counts(support, permutation, device),
        _permuted_grid(query.input, permutation, device),
        _permuted_grid(query.output, permutation, device),
    )


@torch.no_grad()
def _evaluate_pairs(
    model: BindingCTM,
    counts: Tensor,
    pairs: Sequence[ArcPair],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct_cells = 0
    total_cells = 0
    exact_grids = 0
    predictions: list[list[list[int]]] = []
    for pair in pairs:
        query = pair.input.unsqueeze(0).unsqueeze(0).to(device)
        target = pair.output.to(device)
        output = model.forward_from_counts(counts, query)
        prediction = output.query_logits[0, -1, 0].argmax(dim=0)
        correct = prediction == target
        correct_cells += int(correct.sum().cpu())
        total_cells += correct.numel()
        exact_grids += int(correct.all().cpu())
        predictions.append(prediction.cpu().tolist())
    return {
        "cell_accuracy": correct_cells / total_cells,
        "exact_grid_accuracy": exact_grids / len(pairs),
        "all_exact": exact_grids == len(pairs),
        "predictions": predictions,
    }


@torch.no_grad()
def _evaluate_official(
    model: BindingCTM, task: ArcTask, device: torch.device
) -> dict[str, Any]:
    identity = torch.arange(10)
    counts = _association_counts(task.train, identity, device)
    return _evaluate_pairs(model, counts, task.test, device)


@torch.no_grad()
def _evaluate_randomized_loo(
    model: BindingCTM,
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
    for _ in range(episodes):
        fold_index = int(torch.randint(len(folds), (), generator=generator))
        permutation = _color_permutation(generator, augment=True)
        counts, query, target = _episode(
            task, folds[fold_index], permutation, device
        )
        output = model.forward_from_counts(counts, query)
        prediction = output.query_logits[0, -1, 0].argmax(dim=0)
        target_grid = target[0, 0]
        correct = prediction == target_grid
        correct_cells += int(correct.sum().cpu())
        total_cells += correct.numel()
        exact_grids += int(correct.all().cpu())
    return {
        "episodes": episodes,
        "cell_accuracy": correct_cells / total_cells,
        "exact_grid_accuracy": exact_grids / episodes,
    }


@torch.no_grad()
def _evaluate_corrupted_support(
    model: BindingCTM, task: ArcTask, device: torch.device
) -> dict[str, Any]:
    identity = torch.arange(10)
    counts = _association_counts(task.train, identity, device)
    corrupted_counts = counts.roll(shifts=1, dims=2)
    metrics = _evaluate_pairs(model, corrupted_counts, task.test, device)

    changed_cells = 0
    total_cells = 0
    for pair in task.test:
        query = pair.input.unsqueeze(0).unsqueeze(0).to(device)
        correct_output = model.forward_from_counts(counts, query)
        corrupted_output = model.forward_from_counts(corrupted_counts, query)
        correct_prediction = correct_output.query_logits[0, -1, 0].argmax(dim=0)
        corrupted_prediction = corrupted_output.query_logits[0, -1, 0].argmax(dim=0)
        changed_cells += int((correct_prediction != corrupted_prediction).sum().cpu())
        total_cells += correct_prediction.numel()
    metrics["prediction_sensitivity"] = changed_cells / total_cells
    metrics["corruption"] = "roll support association target colors by one"
    return metrics


def _train_fold(
    model: BindingCTM,
    training_tasks: Sequence[ArcTask],
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    task_folds = {task.task_id: _eligible_folds(task) for task in training_tasks}
    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        model.train()
        losses = []
        for _ in range(args.episodes_per_step):
            task_index = int(
                torch.randint(len(training_tasks), (), generator=generator)
            )
            task = training_tasks[task_index]
            folds = task_folds[task.task_id]
            fold_index = int(torch.randint(len(folds), (), generator=generator))
            permutation = _color_permutation(generator, augment=True)
            counts, query, target = _episode(
                task, folds[fold_index], permutation, device
            )
            output = model.forward_from_counts(counts, query)
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

    config = BindingConfig(
        num_colors=10,
        ticks=args.ticks,
        memory_length=args.memory_length,
        symmetric_prior=False,
        update_mode="learned",
        update_hidden=args.binding_hidden,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    for holdout_index, holdout_id in enumerate(args.task_ids):
        fold_seed = args.seed + holdout_index * 10_000
        torch.manual_seed(fold_seed)
        model = BindingCTM(config).to(device)
        holdout_task = tasks[holdout_id]
        training_tasks = [
            tasks[task_id] for task_id in args.task_ids if task_id != holdout_id
        ]
        untrained = _evaluate_official(model, holdout_task, device)
        untrained_randomized_generator = torch.Generator(device="cpu").manual_seed(
            fold_seed + 1_000_000
        )
        untrained_randomized = _evaluate_randomized_loo(
            model,
            holdout_task,
            args.randomized_eval_episodes,
            untrained_randomized_generator,
            device,
        )
        generator = torch.Generator(device="cpu").manual_seed(fold_seed + 1)
        print(
            f"holdout={holdout_id} training_on="
            f"{','.join(task.task_id for task in training_tasks)}",
            flush=True,
        )
        history = _train_fold(model, training_tasks, args, generator, device)
        trained = _evaluate_official(model, holdout_task, device)
        randomized_generator = torch.Generator(device="cpu").manual_seed(
            fold_seed + 1_000_000
        )
        randomized = _evaluate_randomized_loo(
            model,
            holdout_task,
            args.randomized_eval_episodes,
            randomized_generator,
            device,
        )
        corrupted = _evaluate_corrupted_support(model, holdout_task, device)
        fold_result = {
            "holdout_task_id": holdout_id,
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
        checkpoint_path = args.output / f"holdout_{holdout_id}_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": config.to_dict(),
                "training_task_ids": fold_result["training_task_ids"],
                "holdout_task_id": holdout_id,
            },
            checkpoint_path,
        )
        print(
            json.dumps(
                {
                    "holdout": holdout_id,
                    "official_exact": trained["all_exact"],
                    "official_cell": trained["cell_accuracy"],
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
        "experiment": "leave-one-task-out learned binding updater",
        "dataset_commit": args.dataset_commit,
        "task_ids": list(args.task_ids),
        "task_selection_uses_test_outputs": False,
        "selection_rule": (
            "deterministic same-shape cellwise mapping in training demonstrations; "
            "public test-input colors must be covered by demonstrations"
        ),
        "task_metadata": task_metadata,
        "model_config": config.to_dict(),
        "parameter_count": BindingCTM(config).parameter_summary(),
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
            "Train one shared binding updater on genuine ARC tasks and test it on "
            "completely held-out task IDs."
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
    parser.add_argument("--binding-hidden", type=int, default=16)
    parser.add_argument("--randomized-eval-episodes", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/arc_multitask")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
