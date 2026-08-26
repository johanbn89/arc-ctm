"""Training and evaluation loop for synthetic operator inference."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from arc_ctm.config import ModelConfig, TrainConfig
from arc_ctm.data import EpisodeBatch, EpisodeGenerator
from arc_ctm.metrics import (
    between_group_functional_agreement,
    between_group_state_cosine,
    first_support_convergence_tick,
    pairwise_functional_agreement,
    pairwise_state_cosine,
    per_tick_accuracy,
)
from arc_ctm.model import InferenceOutput, OperatorInferenceCTM
from arc_ctm.operators import OPERATORS


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def tick_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    ticks = logits.shape[1]
    weights = torch.linspace(0.5, 1.0, ticks, device=logits.device)
    weights = weights / weights.sum()
    losses = []
    for tick in range(ticks):
        tick_logits = logits[:, tick].flatten(0, 1)
        tick_targets = targets.flatten(0, 1)
        losses.append(F.cross_entropy(tick_logits, tick_targets))
    return (torch.stack(losses) * weights).sum()


def objective(
    output: InferenceOutput,
    batch: EpisodeBatch,
    support_loss_weight: float,
    operator_loss_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    query_loss = tick_cross_entropy(output.query_logits, batch.query_targets)
    support_loss = tick_cross_entropy(output.support_logits, batch.support_targets)
    operator_targets = batch.operator_ids[:, None].expand(
        -1, output.operator_logits.shape[1]
    )
    operator_loss = F.cross_entropy(
        output.operator_logits.flatten(0, 1), operator_targets.flatten()
    )
    total = (
        query_loss
        + support_loss_weight * support_loss
        + operator_loss_weight * operator_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "query_loss": float(query_loss.detach().cpu()),
        "support_loss": float(support_loss.detach().cpu()),
        "operator_loss": float(operator_loss.detach().cpu()),
    }


def _merge_tick_metrics(
    accumulator: dict[str, list[Tensor]], metrics: dict[str, list[float]]
) -> None:
    for name, values in metrics.items():
        accumulator[name].append(torch.tensor(values, dtype=torch.float64))


@torch.no_grad()
def evaluate(
    model: OperatorInferenceCTM,
    generator: EpisodeGenerator,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    support_metrics: dict[str, list[Tensor]] = defaultdict(list)
    query_metrics: dict[str, list[Tensor]] = defaultdict(list)
    convergence_ticks: list[Tensor] = []
    operator_accuracy: list[Tensor] = []

    for _ in range(config.eval_batches):
        batch = generator.sample(
            config.batch_size, config.support_size, config.query_size
        ).to(device)
        output = model(
            batch.support_inputs, batch.support_targets, batch.query_inputs
        )
        _merge_tick_metrics(
            support_metrics,
            per_tick_accuracy(output.support_logits, batch.support_targets),
        )
        _merge_tick_metrics(
            query_metrics,
            per_tick_accuracy(output.query_logits, batch.query_targets),
        )
        convergence_ticks.append(
            first_support_convergence_tick(
                output.support_logits, batch.support_targets
            ).cpu()
        )
        operator_predictions = output.operator_logits.argmax(dim=-1)
        operator_accuracy.append(
            (operator_predictions == batch.operator_ids[:, None])
            .to(torch.float32)
            .mean(dim=0)
            .cpu()
        )

    convergence = torch.cat(convergence_ticks).to(torch.float32)
    never_converged = convergence > model.config.ticks
    return {
        "support": {
            name: torch.stack(values).mean(dim=0).tolist()
            for name, values in support_metrics.items()
        },
        "query": {
            name: torch.stack(values).mean(dim=0).tolist()
            for name, values in query_metrics.items()
        },
        "support_convergence": {
            "mean_tick_with_sentinel": float(convergence.mean()),
            "fraction_converged": float((~never_converged).to(torch.float32).mean()),
        },
        "operator_accuracy": torch.stack(operator_accuracy).mean(dim=0).tolist(),
    }


@torch.no_grad()
def evaluate_invariance(
    model: OperatorInferenceCTM,
    generator: EpisodeGenerator,
    support_size: int,
    query_size: int,
    device: torch.device,
    subsets_per_operator: int = 3,
) -> dict[str, float]:
    """Infer each rule from several support sets and apply it to shared queries."""

    model.eval()
    operator_ids = torch.arange(len(OPERATORS)).repeat_interleave(
        subsets_per_operator
    )
    batch_size = operator_ids.numel()
    # Every inferred state is applied to the same query inputs. This makes both
    # same-rule invariance and different-rule separation directly comparable.
    base_queries = generator.random_grids(1, query_size)
    shared_queries = base_queries.expand(len(OPERATORS), -1, -1, -1)
    shared_queries = shared_queries.repeat_interleave(subsets_per_operator, dim=0)
    batch = generator.sample(
        batch_size,
        support_size,
        query_size,
        operator_ids=operator_ids,
        query_inputs=shared_queries,
    ).to(device)
    output = model(batch.support_inputs, batch.support_targets, batch.query_inputs)
    final_predictions = output.query_logits[:, -1].argmax(dim=2)
    final_states = output.task_states[:, -1]
    final_accuracy = (
        final_predictions == batch.query_targets
    ).to(torch.float32).mean()
    same_function = pairwise_functional_agreement(
        final_predictions, batch.operator_ids
    )
    different_function = between_group_functional_agreement(
        final_predictions, batch.operator_ids
    )
    same_state = pairwise_state_cosine(final_states, batch.operator_ids)
    different_state = between_group_state_cosine(
        final_states, batch.operator_ids
    )
    return {
        "same_operator_functional_agreement": same_function,
        "different_operator_functional_agreement": different_function,
        "functional_agreement_gap": same_function - different_function,
        "same_operator_task_state_cosine": same_state,
        "different_operator_task_state_cosine": different_state,
        "task_state_cosine_gap": same_state - different_state,
        "shared_query_cell_accuracy": float(final_accuracy.cpu()),
    }


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> tuple[OperatorInferenceCTM, dict[str, Any]]:
    set_seed(train_config.seed)
    model = OperatorInferenceCTM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    train_generator = EpisodeGenerator(
        model_config.grid_size, model_config.num_colors, train_config.seed + 1
    )

    model.train()
    latest_losses: dict[str, float] = {}
    for step in range(1, train_config.steps + 1):
        batch = train_generator.sample(
            train_config.batch_size,
            train_config.support_size,
            train_config.query_size,
        ).to(device)
        output = model(
            batch.support_inputs, batch.support_targets, batch.query_inputs
        )
        loss, latest_losses = objective(
            output,
            batch,
            train_config.support_loss_weight,
            train_config.operator_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )
        optimizer.step()

        if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
            final_query = per_tick_accuracy(
                output.query_logits.detach(), batch.query_targets
            )["cell_accuracy"][-1]
            print(
                json.dumps(
                    {
                        "step": step,
                        **latest_losses,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "batch_final_query_cell_accuracy": final_query,
                    }
                ),
                flush=True,
            )

    eval_generator = EpisodeGenerator(
        model_config.grid_size, model_config.num_colors, train_config.seed + 10_000
    )
    results: dict[str, Any] = {
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "parameters": model.parameter_summary(),
        "final_train_losses": latest_losses,
        "evaluation": evaluate(model, eval_generator, train_config, device),
        "invariance": evaluate_invariance(
            model,
            eval_generator,
            train_config.support_size,
            train_config.query_size,
            device,
        ),
        "type_assignment_entropy": float(
            model.neuron_model.assignment_entropy().detach().cpu()
        ),
        "type_usage": model.neuron_model.type_weights()
        .detach()
        .mean(dim=0)
        .cpu()
        .tolist(),
    }
    return model, results


def save_run(
    model: OperatorInferenceCTM,
    results: dict[str, Any],
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "operator_inference.pt"
    metrics_path = output_directory / "metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": results["model_config"],
            "train_config": results["train_config"],
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return checkpoint_path, metrics_path
