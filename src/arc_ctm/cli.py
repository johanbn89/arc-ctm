"""Command-line entry point for the synthetic experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from arc_ctm.config import ModelConfig, TrainConfig
from arc_ctm.experiment import save_run, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a typed CTM to infer synthetic grid operators."
    )
    parser.add_argument("--mode", choices=("private", "shared", "typed"), default="typed")
    parser.add_argument("--types", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-size", type=int, default=3)
    parser.add_argument("--query-size", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-colors", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--operator-loss-weight", type=float, default=0.25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-save", action="store_true")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_config = ModelConfig(
        grid_size=args.grid_size,
        num_colors=args.num_colors,
        d_model=args.d_model,
        ticks=args.ticks,
        nlm_mode=args.mode,
        neuron_types=args.types,
        sync_pairs=min(48, args.d_model),
        sync_self_pairs=min(8, args.d_model),
    )
    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        support_size=args.support_size,
        query_size=args.query_size,
        learning_rate=args.learning_rate,
        operator_loss_weight=args.operator_loss_weight,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        seed=args.seed,
    )
    device = _device(args.device)
    print(json.dumps({"device": str(device), "mode": args.mode}), flush=True)
    model, results = train(model_config, train_config, device)
    print(json.dumps(results, indent=2), flush=True)
    if not args.no_save:
        checkpoint, metrics = save_run(model, results, args.output)
        print(f"saved checkpoint: {checkpoint.resolve()}")
        print(f"saved metrics: {metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
