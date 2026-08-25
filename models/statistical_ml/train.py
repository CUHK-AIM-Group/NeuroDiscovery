"""CLI for classical NeuroClaw statistical-learning models."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.common.data import load_tabular_csv
from models.common.training import cross_validate_estimator
from models.statistical_ml.estimators import make_estimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--group-col")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_tabular_csv(
        args.features,
        target=args.target,
        subject_col=args.subject_col,
        group_col=args.group_col,
    )
    if args.dry_run:
        print(f"samples={len(data.y)} features={data.X.shape[1]} task={args.task}")
        return 0
    config = vars(args)
    metrics = cross_validate_estimator(
        data,
        estimator_factory=lambda: make_estimator(args.model, args.task, args.seed),
        task=args.task,
        output_dir=Path(args.output_dir),
        config=config,
        n_splits=args.folds,
        seed=args.seed,
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
