"""CLI for disease-subtyping models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from models.common.artifacts import RunArtifacts
from models.common.data import load_tabular_csv
from models.subtyping.estimators import fit_subtypes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.features)
    frame["_unused_target"] = 0
    temp = Path(args.output_dir) / "_subtyping_input.csv"
    temp.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temp, index=False)
    data = load_tabular_csv(temp, "_unused_target", args.subject_col)
    temp.unlink(missing_ok=True)
    result = fit_subtypes(
        data.X,
        model=args.model,
        n_clusters=args.n_clusters,
        seed=args.seed,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
    )
    artifacts = RunArtifacts(args.output_dir, vars(args))
    artifacts.write_config()
    artifacts.write_metrics(result.metrics)
    payload = {
        "subject_id": data.subject_ids,
        "subtype": result.labels,
    }
    for index in range(result.embedding.shape[1]):
        payload[f"embedding_{index + 1}"] = result.embedding[:, index]
    artifacts.write_predictions(payload)
    joblib.dump(result.model, Path(args.output_dir) / "checkpoint.joblib")
    artifacts.write_manifest()
    print(result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
