"""CLI for matrix-based imaging-genetics analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from models.common.artifacts import RunArtifacts
from models.imaging_genetics.association import (
    association_scan,
    association_scan_lmm,
    polygenic_score,
)
from models.imaging_genetics.multivariate import fit_cca, fit_pls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ matrix bundle")
    parser.add_argument(
        "--model", choices=["association", "lmm", "prs", "pls", "cca"], required=True
    )
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    blob = np.load(args.input, allow_pickle=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(output, vars(args))
    artifacts.write_config()
    if args.model in {"association", "lmm"}:
        common = {
            "genotype": blob["genotype"],
            "phenotype": blob["phenotype"],
            "variant_ids": blob["variant_id"].astype(str).tolist()
            if "variant_id" in blob
            else None,
            "covariates": blob["covariates"] if "covariates" in blob else None,
        }
        if args.model == "lmm":
            if "kinship" not in blob:
                raise ValueError("LMM mode requires a kinship matrix in the NPZ bundle")
            result = association_scan_lmm(kinship=blob["kinship"], **common)
        else:
            result = association_scan(**common)
        result.to_csv(output / "association_results.csv", index=False)
        metrics = {
            "variants_tested": int(len(result)),
            "minimum_p_value": float(result["p_value"].min()),
        }
    elif args.model == "prs":
        score = polygenic_score(blob["genotype"], blob["weights"])
        artifacts.write_predictions(
            {
                "subject_id": blob["subject_id"].astype(str)
                if "subject_id" in blob
                else np.arange(len(score)).astype(str),
                "polygenic_score": score,
            }
        )
        metrics = {"score_mean": float(score.mean()), "score_sd": float(score.std(ddof=1))}
    else:
        fit = (
            fit_pls(blob["X"], blob["Y"], args.components)
            if args.model == "pls"
            else fit_cca(blob["X"], blob["Y"], args.components)
        )
        rows = {}
        for index in range(fit.x_scores.shape[1]):
            rows[f"x_score_{index + 1}"] = fit.x_scores[:, index]
            rows[f"y_score_{index + 1}"] = fit.y_scores[:, index]
        artifacts.write_predictions(rows)
        joblib.dump(fit.model, output / "checkpoint.joblib")
        metrics = {
            f"component_{index + 1}_correlation": float(value)
            for index, value in enumerate(fit.correlations)
        }
    artifacts.write_metrics(metrics)
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
