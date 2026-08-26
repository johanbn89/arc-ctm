"""Compare I/O-only and trace-supervised unseen-composition learning."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from arc_ctm.arc_viz import write_composition_report
from arc_ctm.composition_data import (
    DEFAULT_HELDOUT_SPECS,
    DEFAULT_VALIDATION_SPECS,
    CompositionBatch,
    CompositionEpisodeGenerator,
    CompositionSpec,
    all_composition_specs,
    apply_compositions,
)
from arc_ctm.composition_model import (
    CompositionInferenceOutput,
    CompositionModelConfig,
    CompositionalOperatorModel,
)
from arc_ctm.experiment import tick_cross_entropy
from arc_ctm.operators import OPERATORS


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _tick_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    tick_count = logits.shape[1]
    weights = torch.linspace(0.5, 1.0, tick_count, device=logits.device)
    weights = weights / weights.sum()
    losses = []
    for tick in range(tick_count):
        losses.append(F.cross_entropy(logits[:, tick], targets))
    return (torch.stack(losses) * weights).sum()


def _objective(
    output: CompositionInferenceOutput,
    batch: CompositionBatch,
    *,
    support_loss_weight: float,
    trace_weight: float,
    intermediate_trace_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    query_loss = tick_cross_entropy(output.query_logits, batch.query_targets)
    support_loss = tick_cross_entropy(output.support_logits, batch.support_targets)
    spatial_trace_loss = _tick_classification_loss(
        output.spatial_logits, batch.spatial_ids
    )
    color_trace_loss = _tick_classification_loss(
        output.color_logits, batch.color_shifts
    )
    intermediate_trace_loss = tick_cross_entropy(
        output.support_intermediate_logits, batch.support_intermediates
    )
    trace_loss = (
        spatial_trace_loss
        + color_trace_loss
        + intermediate_trace_weight * intermediate_trace_loss
    )
    loss = query_loss + support_loss_weight * support_loss + trace_weight * trace_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "query_loss": float(query_loss.detach().cpu()),
        "support_loss": float(support_loss.detach().cpu()),
        "spatial_trace_loss": float(spatial_trace_loss.detach().cpu()),
        "color_trace_loss": float(color_trace_loss.detach().cpu()),
        "intermediate_trace_loss": float(intermediate_trace_loss.detach().cpu()),
    }


def _train_model(
    name: str,
    model: CompositionalOperatorModel,
    train_specs: Sequence[CompositionSpec],
    args: argparse.Namespace,
    device: torch.device,
    *,
    trace_weight: float,
) -> list[dict[str, float | int]]:
    generator = CompositionEpisodeGenerator(
        args.grid_size,
        args.num_colors,
        args.num_color_shifts,
        args.seed + 1,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        model.train()
        batch = generator.sample(
            args.batch_size,
            args.support_size,
            args.query_size,
            train_specs,
        ).to(device)
        output = model(
            batch.support_inputs, batch.support_targets, batch.query_inputs
        )
        loss, metrics = _objective(
            output,
            batch,
            support_loss_weight=args.support_loss_weight,
            trace_weight=trace_weight,
            intermediate_trace_weight=args.intermediate_trace_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record: dict[str, float | int] = {
                "step": step,
                **metrics,
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
            history.append(record)
            print(json.dumps({"model": name, **record}), flush=True)
    return history


@torch.no_grad()
def _evaluate_specs(
    model: CompositionalOperatorModel,
    specs: Sequence[CompositionSpec],
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    generator = CompositionEpisodeGenerator(
        args.grid_size,
        args.num_colors,
        args.num_color_shifts,
        seed,
    )
    per_spec: dict[str, Any] = {}
    aggregate_correct_cells = 0
    aggregate_cells = 0
    aggregate_exact = 0
    aggregate_grids = 0
    aggregate_spatial = 0
    aggregate_color = 0
    aggregate_joint = 0
    aggregate_episodes = 0

    for spec in specs:
        correct_cells = 0
        total_cells = 0
        exact_grids = 0
        total_grids = 0
        spatial_correct = 0
        color_correct = 0
        joint_correct = 0
        episode_count = 0
        batches = math.ceil(args.eval_episodes_per_spec / args.eval_batch_size)
        for batch_index in range(batches):
            remaining = args.eval_episodes_per_spec - batch_index * args.eval_batch_size
            batch_size = min(args.eval_batch_size, remaining)
            batch = generator.sample(
                batch_size,
                args.support_size,
                args.query_size,
                (spec,),
            ).to(device)
            output = model(
                batch.support_inputs, batch.support_targets, batch.query_inputs
            )
            predicted_spatial = output.spatial_logits[:, -1].argmax(dim=-1)
            predicted_color = output.color_logits[:, -1].argmax(dim=-1)
            _, predictions = apply_compositions(
                batch.query_inputs,
                predicted_spatial,
                predicted_color,
                args.num_colors,
            )
            correct = predictions == batch.query_targets
            correct_cells += int(correct.sum().cpu())
            total_cells += correct.numel()
            exact_grids += int(correct.all(dim=(-1, -2)).sum().cpu())
            total_grids += batch_size * args.query_size
            spatial_matches = predicted_spatial == batch.spatial_ids
            color_matches = predicted_color == batch.color_shifts
            spatial_correct += int(spatial_matches.sum().cpu())
            color_correct += int(color_matches.sum().cpu())
            joint_correct += int((spatial_matches & color_matches).sum().cpu())
            episode_count += batch_size

        spec_metrics = {
            "spatial_id": spec.spatial_id,
            "color_shift": spec.color_shift,
            "cell_accuracy": correct_cells / total_cells,
            "exact_grid_accuracy": exact_grids / total_grids,
            "spatial_trace_accuracy": spatial_correct / episode_count,
            "color_trace_accuracy": color_correct / episode_count,
            "joint_trace_accuracy": joint_correct / episode_count,
            "episodes": episode_count,
        }
        per_spec[spec.name] = spec_metrics
        aggregate_correct_cells += correct_cells
        aggregate_cells += total_cells
        aggregate_exact += exact_grids
        aggregate_grids += total_grids
        aggregate_spatial += spatial_correct
        aggregate_color += color_correct
        aggregate_joint += joint_correct
        aggregate_episodes += episode_count

    return {
        "cell_accuracy": aggregate_correct_cells / aggregate_cells,
        "exact_grid_accuracy": aggregate_exact / aggregate_grids,
        "spatial_trace_accuracy": aggregate_spatial / aggregate_episodes,
        "color_trace_accuracy": aggregate_color / aggregate_episodes,
        "joint_trace_accuracy": aggregate_joint / aggregate_episodes,
        "episodes": aggregate_episodes,
        "per_spec": per_spec,
    }


@torch.no_grad()
def _examples(
    io_model: CompositionalOperatorModel,
    trace_model: CompositionalOperatorModel,
    specs: Sequence[CompositionSpec],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = CompositionEpisodeGenerator(
        args.grid_size,
        args.num_colors,
        args.num_color_shifts,
        args.seed + 9_000_000,
    )
    examples = []
    for spec in specs:
        batch = generator.sample(1, args.support_size, 1, (spec,)).to(device)
        model_predictions = {}
        for name, model in (("io_only", io_model), ("trace_supervised", trace_model)):
            model.eval()
            output = model(
                batch.support_inputs, batch.support_targets, batch.query_inputs
            )
            spatial_id = int(output.spatial_logits[0, -1].argmax().cpu())
            color_shift = int(output.color_logits[0, -1].argmax().cpu())
            _, prediction = apply_compositions(
                batch.query_inputs,
                torch.tensor([spatial_id], device=device),
                torch.tensor([color_shift], device=device),
                args.num_colors,
            )
            model_predictions[name] = {
                "predicted_trace": (
                    f"{OPERATORS[spatial_id].name} -> color_shift_{color_shift}"
                ),
                "prediction": prediction[0, 0].cpu().tolist(),
                "exact": bool((prediction == batch.query_targets).all().cpu()),
            }
        examples.append(
            {
                "heldout_trace": spec.name,
                "support_inputs": batch.support_inputs[0].cpu().tolist(),
                "support_targets": batch.support_targets[0].cpu().tolist(),
                "query_input": batch.query_inputs[0, 0].cpu().tolist(),
                "query_intermediate": batch.query_intermediates[0, 0].cpu().tolist(),
                "query_target": batch.query_targets[0, 0].cpu().tolist(),
                "models": model_predictions,
            }
        )
    return examples


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    all_specs = all_composition_specs(args.num_color_shifts)
    heldout_specs = tuple(DEFAULT_HELDOUT_SPECS)
    validation_specs = tuple(DEFAULT_VALIDATION_SPECS)
    excluded_specs = validation_specs + heldout_specs
    if any(spec not in all_specs for spec in excluded_specs):
        raise ValueError("default held-out specs require at least four color shifts")
    train_specs = tuple(spec for spec in all_specs if spec not in excluded_specs)
    primitive_specs = tuple(
        spec
        for spec in train_specs
        if spec.spatial_id == 0 or spec.color_shift == 0
    )

    config = CompositionModelConfig(
        grid_size=args.grid_size,
        num_colors=args.num_colors,
        num_color_shifts=args.num_color_shifts,
        pair_dim=args.pair_dim,
        evidence_dim=args.evidence_dim,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        ticks=args.ticks,
    )
    torch.manual_seed(args.seed)
    io_model = CompositionalOperatorModel(config).to(device)
    untrained = _evaluate_specs(
        io_model,
        heldout_specs,
        args,
        device,
        seed=args.seed + 1_000_000,
    )
    io_history = _train_model(
        "io_only",
        io_model,
        train_specs,
        args,
        device,
        trace_weight=0.0,
    )
    torch.manual_seed(args.seed)
    trace_model = CompositionalOperatorModel(config).to(device)
    trace_history = _train_model(
        "trace_supervised",
        trace_model,
        train_specs,
        args,
        device,
        trace_weight=args.trace_weight,
    )

    evaluation_seed = args.seed + 2_000_000
    io_seen = _evaluate_specs(
        io_model, train_specs, args, device, seed=evaluation_seed
    )
    io_validation = _evaluate_specs(
        io_model, validation_specs, args, device, seed=evaluation_seed + 1
    )
    io_heldout = _evaluate_specs(
        io_model, heldout_specs, args, device, seed=evaluation_seed + 2
    )
    trace_seen = _evaluate_specs(
        trace_model, train_specs, args, device, seed=evaluation_seed
    )
    trace_validation = _evaluate_specs(
        trace_model, validation_specs, args, device, seed=evaluation_seed + 1
    )
    trace_heldout = _evaluate_specs(
        trace_model, heldout_specs, args, device, seed=evaluation_seed + 2
    )
    trace_primitives = _evaluate_specs(
        trace_model, primitive_specs, args, device, seed=evaluation_seed + 3
    )

    results: dict[str, Any] = {
        "experiment": "learned raw-pair encoder with unseen operator compositions",
        "model_config": config.to_dict(),
        "parameter_count": io_model.parameter_summary(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "support_size": args.support_size,
            "query_size": args.query_size,
            "learning_rate": args.learning_rate,
            "trace_weight": args.trace_weight,
            "intermediate_trace_weight": args.intermediate_trace_weight,
            "seed": args.seed,
            "train_compositions": [spec.name for spec in train_specs],
            "validation_compositions": [spec.name for spec in validation_specs],
            "heldout_compositions": [spec.name for spec in heldout_specs],
            "io_history": io_history,
            "trace_history": trace_history,
        },
        "untrained_heldout": untrained,
        "io_only": {
            "seen": io_seen,
            "validation": io_validation,
            "heldout": io_heldout,
        },
        "trace_supervised": {
            "seen": trace_seen,
            "validation": trace_validation,
            "heldout": trace_heldout,
            "primitive_only": trace_primitives,
        },
        "examples": _examples(
            io_model, trace_model, heldout_specs, args, device
        ),
        "claims": {
            "raw_pair_encoder": True,
            "operator_labels_used_as_model_input": False,
            "heldout_compositions_used_for_training": False,
            "executor_factorization_is_hand_designed": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    report_path = args.output / "report.html"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(
        {"model_state": io_model.state_dict(), "model_config": config.to_dict()},
        args.output / "io_only_model.pt",
    )
    torch.save(
        {"model_state": trace_model.state_dict(), "model_config": config.to_dict()},
        args.output / "trace_supervised_model.pt",
    )
    write_composition_report(results, report_path)
    print(
        json.dumps(
            {
                "io_heldout_exact": io_heldout["exact_grid_accuracy"],
                "trace_heldout_exact": trace_heldout["exact_grid_accuracy"],
                "trace_heldout_joint_operator": trace_heldout[
                    "joint_trace_accuracy"
                ],
            }
        ),
        flush=True,
    )
    print(f"results: {result_path.resolve()}")
    print(f"report: {report_path.resolve()}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare I/O-only and trace-supervised raw-pair encoders on unseen "
            "spatial-plus-color compositions."
        )
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-size", type=int, default=3)
    parser.add_argument("--query-size", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-colors", type=int, default=5)
    parser.add_argument("--num-color-shifts", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--pair-dim", type=int, default=128)
    parser.add_argument("--evidence-dim", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--support-loss-weight", type=float, default=0.25)
    parser.add_argument("--trace-weight", type=float, default=0.5)
    parser.add_argument("--intermediate-trace-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-episodes-per-spec", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/arc_composition")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
