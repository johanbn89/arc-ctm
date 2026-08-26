"""Overfit a CTM to one genuine ARC task without training on its test target."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch import nn

from arc_ctm.arc_data import (
    ArcTask,
    leave_one_out_batch,
    load_arc_task,
    official_test_batch,
    resolve_training_task,
)
from arc_ctm.arc_viz import write_arc_report
from arc_ctm.binding_ctm import BindingCTM, BindingConfig
from arc_ctm.config import ModelConfig
from arc_ctm.data import EpisodeBatch
from arc_ctm.experiment import tick_cross_entropy
from arc_ctm.model import InferenceOutput, OperatorInferenceCTM


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _accuracy(logits: Tensor, targets: Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=3)
    correct = predictions == targets[:, None]
    cell_by_tick = correct.to(torch.float32).mean(dim=(0, 2, 3, 4))
    exact_by_tick = correct.all(dim=(-1, -2, -3)).to(torch.float32).mean(dim=0)
    return {
        "cell_by_tick": cell_by_tick.detach().cpu().tolist(),
        "exact_by_tick": exact_by_tick.detach().cpu().tolist(),
        "final_prediction": predictions[0, -1, 0].detach().cpu().tolist(),
        "exact": bool(correct[0, -1, 0].all().detach().cpu()),
    }


@torch.no_grad()
def _predict(
    model: nn.Module, batch: EpisodeBatch, device: torch.device
) -> tuple[InferenceOutput, dict[str, Any]]:
    model.eval()
    batch = batch.to(device)
    output = model(batch.support_inputs, batch.support_targets, batch.query_inputs)
    return output, _accuracy(output.query_logits, batch.query_targets)


def _model_config(
    task: ArcTask, args: argparse.Namespace
) -> ModelConfig | BindingConfig:
    if args.architecture == "binding":
        return BindingConfig(
            num_colors=10,
            ticks=args.ticks,
            memory_length=args.memory_length,
            symmetric_prior=not args.no_involution_prior,
            update_mode=args.binding_update,
            update_hidden=args.binding_hidden,
        )
    return ModelConfig(
        grid_size=task.common_square_size(),
        num_colors=10,
        d_model=args.d_model,
        memory_length=args.memory_length,
        memory_hidden=args.memory_hidden,
        evidence_dim=args.evidence_dim,
        task_state_dim=args.task_state_dim,
        decoder_hidden=args.decoder_hidden,
        decoder_mode=args.decoder_mode,
        ticks=args.ticks,
        sync_pairs=args.sync_pairs,
        sync_self_pairs=min(args.sync_self_pairs, args.sync_pairs),
        nlm_mode=args.mode,
        neuron_types=args.types,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    task_path = args.task or resolve_training_task(args.dataset_root, args.task_id)
    task = load_arc_task(task_path)
    config = _model_config(task, args)
    if isinstance(config, BindingConfig):
        model: nn.Module = BindingCTM(config).to(device)
    else:
        model = OperatorInferenceCTM(config).to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    official_batch = official_test_batch(task)

    _, untrained = _predict(model, official_batch, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []

    for step in range(1, args.steps + 1):
        model.train()
        batch = leave_one_out_batch(
            task,
            args.batch_size,
            generator,
            augment_colors=not args.no_color_augmentation,
        ).to(device)
        output = model(batch.support_inputs, batch.support_targets, batch.query_inputs)
        query_loss = tick_cross_entropy(output.query_logits, batch.query_targets)
        support_loss = tick_cross_entropy(output.support_logits, batch.support_targets)
        loss = query_loss + args.support_loss_weight * support_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            folds = leave_one_out_batch(
                task,
                len(task.train),
                generator,
                augment_colors=False,
                cover_all_folds=True,
            )
            _, fold_metrics = _predict(model, folds, device)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "query_loss": float(query_loss.detach().cpu()),
                "support_loss": float(support_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "leave_one_out_final_cell": fold_metrics["cell_by_tick"][-1],
                "leave_one_out_final_exact": fold_metrics["exact_by_tick"][-1],
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    trained_output, trained = _predict(model, official_batch, device)
    corrupted_batch = EpisodeBatch(
        support_inputs=official_batch.support_inputs,
        support_targets=torch.roll(official_batch.support_targets, shifts=1, dims=1),
        query_inputs=official_batch.query_inputs,
        query_targets=official_batch.query_targets,
        operator_ids=official_batch.operator_ids,
    )
    corrupted_output, corrupted = _predict(model, corrupted_batch, device)
    correct_prediction = trained_output.query_logits[:, -1].argmax(dim=2)
    corrupted_prediction = corrupted_output.query_logits[:, -1].argmax(dim=2)
    support_sensitivity = float(
        (correct_prediction != corrupted_prediction).to(torch.float32).mean().cpu()
    )

    folds = leave_one_out_batch(
        task,
        len(task.train),
        generator,
        augment_colors=False,
        cover_all_folds=True,
    )
    _, fold_metrics = _predict(model, folds, device)
    randomized_eval_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 1_000_000
    )
    randomized_eval_batch = leave_one_out_batch(
        task,
        args.randomized_eval_episodes,
        randomized_eval_generator,
        augment_colors=True,
    )
    _, randomized_eval = _predict(model, randomized_eval_batch, device)
    results: dict[str, Any] = {
        "task_id": task.task_id,
        "task_source": str(task.source),
        "dataset_commit": args.dataset_commit,
        "test_target_used_for_training": False,
        "color_augmentation": not args.no_color_augmentation,
        "architecture": args.architecture,
        "model_config": config.to_dict(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "support_loss_weight": args.support_loss_weight,
            "seed": args.seed,
            "history": history,
        },
        "untrained": untrained,
        "trained": trained,
        "corrupted_support": corrupted,
        "randomized_evaluation": randomized_eval,
        "metrics": {
            "leave_one_out_final_cell": fold_metrics["cell_by_tick"][-1],
            "leave_one_out_final_exact": fold_metrics["exact_by_tick"][-1],
            "official_test_final_cell": trained["cell_by_tick"][-1],
            "official_test_exact": trained["exact"],
            "corrupted_support_final_cell": corrupted["cell_by_tick"][-1],
            "corrupted_support_exact": corrupted["exact"],
            "support_prediction_sensitivity": support_sensitivity,
            "randomized_leave_one_out_final_cell": randomized_eval["cell_by_tick"][-1],
            "randomized_leave_one_out_final_exact": randomized_eval["exact_by_tick"][-1],
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"{task.task_id}_results.json"
    checkpoint_path = args.output / f"{task.task_id}_model.pt"
    report_path = args.output / f"{task.task_id}_report.html"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(
        {"model_state": model.state_dict(), "model_config": config.to_dict()},
        checkpoint_path,
    )
    write_arc_report(task, results, report_path)
    print(f"results: {result_path.resolve()}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"report: {report_path.resolve()}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit a CTM on leave-one-out demonstrations from one ARC-AGI-1 task."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/arc_agi_1")
    )
    parser.add_argument("--dataset-commit", default="399030444e0ab0cc8b4e199870fb20b863846f34")
    parser.add_argument("--task-id", default="0d3d703e")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--support-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--architecture", choices=("ctm", "binding"), default="ctm")
    parser.add_argument("--mode", choices=("private", "shared", "typed"), default="typed")
    parser.add_argument("--types", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--memory-length", type=int, default=4)
    parser.add_argument("--memory-hidden", type=int, default=16)
    parser.add_argument("--evidence-dim", type=int, default=48)
    parser.add_argument("--task-state-dim", type=int, default=32)
    parser.add_argument("--decoder-hidden", type=int, default=96)
    parser.add_argument("--decoder-mode", choices=("global", "cellwise"), default="cellwise")
    parser.add_argument("--sync-pairs", type=int, default=48)
    parser.add_argument("--sync-self-pairs", type=int, default=8)
    parser.add_argument("--no-color-augmentation", action="store_true")
    parser.add_argument("--no-involution-prior", action="store_true")
    parser.add_argument("--binding-update", choices=("analytic", "learned"), default="analytic")
    parser.add_argument("--binding-hidden", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("artifacts/arc_overfit"))
    parser.add_argument("--randomized-eval-episodes", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
