"""Run resumable lightweight NeuroRuntime model sweeps for executable case studies."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_pilots\pilot_20260805_032408"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_model_expansion"
)
DEFAULT_CASES = (
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "progression_prediction",
    "connectome_behavior",
    "brain_age",
    "imaging_genetics",
    "prognosis",
)
CLASSIFICATION_MODELS = ("logistic", "ridge", "elastic_net", "svm")
REGRESSION_MODELS = ("ols", "ridge", "elastic_net", "svm")
SUBTYPING_MODELS = (
    "kmeans",
    "gmm",
    "spectral",
    "nmf",
    "consensus",
    "pca",
    "autoencoder",
)
SURVIVAL_MODELS = ("cox", "deepsurv")
IMAGING_GENETICS_MODELS = ("association", "pls", "cca")


@dataclass(frozen=True)
class SweepTask:
    case_study: str
    model: str
    seed: int | None
    output_dir: Path
    command: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def prepare_derived_inputs(
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    hcp = manifest["hcpya"]
    behavior = pd.read_csv(hcp["behavior_labels"], dtype={"subject_id": str})
    brain_age = pd.read_csv(hcp["brain_age_input"], dtype={"subject_id": str})
    behavior_tabular = brain_age.drop(columns=["age"]).merge(
        behavior,
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    behavior_path = input_dir / "hcpya_connectome_behavior_tabular.csv"
    behavior_tabular.to_csv(behavior_path, index=False)

    genetics_source = Path(manifest["adni"]["imaging_genetics_input"])
    blob = np.load(genetics_source, allow_pickle=True)
    multivariate_path = input_dir / "adni_pathway_prs_hippocampus_multivariate.npz"
    np.savez_compressed(
        multivariate_path,
        X=np.asarray(blob["genotype"], dtype=np.float32),
        Y=np.asarray(blob["phenotype"], dtype=np.float32).reshape(-1, 1),
        subject_id=np.asarray(blob["subject_id"]).astype(str),
    )
    return {
        "behavior_tabular": str(behavior_path),
        "imaging_genetics_multivariate": str(multivariate_path),
    }


def statistical_task(
    *,
    case_study: str,
    features: str,
    target: str,
    model: str,
    task: str,
    seed: int,
    output_dir: Path,
    folds: int,
) -> SweepTask:
    command = (
        sys.executable,
        "-m",
        "models.statistical_ml.train",
        "--features",
        features,
        "--target",
        target,
        "--model",
        model,
        "--task",
        task,
        "--folds",
        str(folds),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    )
    return SweepTask(case_study, model, seed, output_dir, command)


def build_tasks(
    *,
    manifest: dict[str, Any],
    derived: dict[str, str],
    output_dir: Path,
    cases: list[str],
    seeds: list[int],
    folds: int,
    deep_epochs: int,
) -> list[SweepTask]:
    tcp = manifest["tcp"]
    hcp = manifest["hcpya"]
    adni = manifest["adni"]
    tasks: list[SweepTask] = []

    classification_specs = {
        "biomarker_discovery": (tcp["biomarker_input"], "is_patient"),
        "differential_diagnosis": (tcp["differential_input"], "psychosis_vs_mdd"),
        "progression_prediction": (adni["progression_input"], "progression_3y"),
    }
    for case_study, (features, target) in classification_specs.items():
        if case_study not in cases:
            continue
        for seed in seeds:
            for model in CLASSIFICATION_MODELS:
                model_dir = output_dir / case_study / model / f"seed_{seed}"
                tasks.append(
                    statistical_task(
                        case_study=case_study,
                        features=features,
                        target=target,
                        model=model,
                        task="classification",
                        seed=seed,
                        output_dir=model_dir,
                        folds=folds,
                    )
                )

    if "disease_subtyping" in cases:
        for seed in seeds:
            for model in SUBTYPING_MODELS:
                model_dir = output_dir / "disease_subtyping" / model / f"seed_{seed}"
                command = (
                    sys.executable,
                    "-m",
                    "models.subtyping.train",
                    "--features",
                    tcp["subtyping_input"],
                    "--model",
                    model,
                    "--n-clusters",
                    "3",
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(deep_epochs),
                    "--output-dir",
                    str(model_dir),
                )
                tasks.append(SweepTask("disease_subtyping", model, seed, model_dir, command))

    if "connectome_behavior" in cases:
        for seed in seeds:
            for model in REGRESSION_MODELS:
                model_dir = output_dir / "connectome_behavior" / model / f"seed_{seed}"
                tasks.append(
                    statistical_task(
                        case_study="connectome_behavior",
                        features=derived["behavior_tabular"],
                        target=hcp["behavior_target"],
                        model=model,
                        task="regression",
                        seed=seed,
                        output_dir=model_dir,
                        folds=folds,
                    )
                )
            for threshold in (0.001, 0.005, 0.01, 0.05):
                model = f"cpm_p{threshold:g}"
                model_dir = output_dir / "connectome_behavior" / model / f"seed_{seed}"
                command = (
                    sys.executable,
                    "-m",
                    "models.cpm.train",
                    "--connectomes",
                    hcp["behavior_connectomes"],
                    "--labels",
                    hcp["behavior_labels"],
                    "--target",
                    hcp["behavior_target"],
                    "--task",
                    "regression",
                    "--p-threshold",
                    str(threshold),
                    "--folds",
                    str(folds),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(model_dir),
                )
                tasks.append(SweepTask("connectome_behavior", model, seed, model_dir, command))

    if "brain_age" in cases:
        for seed in seeds:
            for model in REGRESSION_MODELS:
                model_dir = output_dir / "brain_age" / model / f"seed_{seed}"
                command = (
                    sys.executable,
                    "-m",
                    "models.brain_age.train",
                    "--features",
                    hcp["brain_age_input"],
                    "--age-col",
                    "age",
                    "--model",
                    model,
                    "--folds",
                    str(folds),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(model_dir),
                )
                tasks.append(SweepTask("brain_age", model, seed, model_dir, command))

    if "prognosis" in cases:
        for seed in seeds:
            for model in SURVIVAL_MODELS:
                model_dir = output_dir / "prognosis" / model / f"seed_{seed}"
                command = (
                    sys.executable,
                    "-m",
                    "models.survival_models.train",
                    "--features",
                    adni["survival_input"],
                    "--duration-col",
                    "duration_years",
                    "--event-col",
                    "event",
                    "--model",
                    model,
                    "--folds",
                    str(folds),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(deep_epochs),
                    "--output-dir",
                    str(model_dir),
                )
                tasks.append(SweepTask("prognosis", model, seed, model_dir, command))

    if "imaging_genetics" in cases:
        genetics_inputs = {
            "association": adni["imaging_genetics_input"],
            "pls": derived["imaging_genetics_multivariate"],
            "cca": derived["imaging_genetics_multivariate"],
        }
        for model in IMAGING_GENETICS_MODELS:
            model_dir = output_dir / "imaging_genetics" / model
            command = [
                sys.executable,
                "-m",
                "models.imaging_genetics.train",
                "--input",
                genetics_inputs[model],
                "--model",
                model,
                "--output-dir",
                str(model_dir),
            ]
            if model in {"pls", "cca"}:
                command.extend(["--components", "1"])
            tasks.append(SweepTask("imaging_genetics", model, None, model_dir, tuple(command)))
    return tasks


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def compact_metric_summary(results: pd.DataFrame) -> pd.DataFrame:
    primary = {
        "biomarker_discovery": ("auroc", "higher"),
        "differential_diagnosis": ("auroc", "higher"),
        "progression_prediction": ("auroc", "higher"),
        "disease_subtyping": ("silhouette", "higher"),
        "connectome_behavior": ("pearson_r", "higher"),
        "brain_age": ("raw_mae", "lower"),
        "prognosis": ("concordance_index", "higher"),
    }
    rows: list[dict[str, Any]] = []
    for (case_study, model), group in results.groupby(["case_study", "model"]):
        if case_study == "imaging_genetics":
            metric, direction = (
                ("minimum_p_value", "lower")
                if model == "association"
                else ("component_1_correlation", "higher")
            )
        else:
            metric, direction = primary[str(case_study)]
        source = group[metric] if metric in group else pd.Series(dtype=float)
        values = pd.to_numeric(source, errors="coerce").dropna()
        rows.append(
            {
                "case_study": case_study,
                "model": model,
                "primary_metric": metric,
                "direction": direction,
                "mean": float(values.mean()) if len(values) else np.nan,
                "variance": float(values.var(ddof=1)) if len(values) > 1 else np.nan,
                "n_runs": int(len(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_study", "model"]).reset_index(drop=True)


def run_task(task: SweepTask, log_dir: Path, *, resume: bool) -> dict[str, Any]:
    metrics_path = task.output_dir / "metrics.json"
    if resume and metrics_path.is_file():
        return {
            "case_study": task.case_study,
            "model": task.model,
            "seed": task.seed,
            "status": "completed",
            "resumed": True,
            "elapsed_sec": 0.0,
            "output_dir": str(task.output_dir),
            **flatten_metrics(read_json(metrics_path)),
        }
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "fixed" if task.seed is None else str(task.seed)
    log_path = log_dir / f"{task.case_study}__{task.model}__{seed_label}.log"
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(
        list(task.command),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "$ " + subprocess.list2cmdline(list(task.command)) + "\n\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr,
        encoding="utf-8",
    )
    record = {
        "case_study": task.case_study,
        "model": task.model,
        "seed": task.seed,
        "status": "completed" if process.returncode == 0 and metrics_path.is_file() else "failed",
        "resumed": False,
        "returncode": process.returncode,
        "elapsed_sec": round(elapsed, 3),
        "output_dir": str(task.output_dir),
        "log": str(log_path),
    }
    if metrics_path.is_file():
        record.update(flatten_metrics(read_json(metrics_path)))
    elif process.stderr:
        record["error"] = process.stderr.strip().splitlines()[-1]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--seeds", default="20260805,20260806,20260807")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--deep-epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    cases = parse_csv(args.cases)
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        raise ValueError(f"Unknown case studies: {unknown}")
    seeds = [int(value) for value in parse_csv(args.seeds)]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.source_run / "data_manifest.json")
    derived = prepare_derived_inputs(manifest, output_dir)
    tasks = build_tasks(
        manifest=manifest,
        derived=derived,
        output_dir=output_dir,
        cases=cases,
        seeds=seeds,
        folds=args.folds,
        deep_epochs=args.deep_epochs,
    )
    if args.smoke:
        selected: list[SweepTask] = []
        seen: set[str] = set()
        for task in tasks:
            if task.case_study not in seen:
                selected.append(task)
                seen.add(task.case_study)
        tasks = selected

    plan = {
        "created_at": utc_now(),
        "source_run": str(args.source_run),
        "output_dir": str(output_dir),
        "cases": cases,
        "seeds": seeds,
        "folds": args.folds,
        "deep_epochs": args.deep_epochs,
        "n_tasks": len(tasks),
        "tasks": [{**asdict(task), "output_dir": str(task.output_dir)} for task in tasks],
        "not_executed": {
            "xgboost": "optional xgboost dependency is not installed",
            "random_survival_forest": "optional scikit-survival dependency is not installed",
            "lmm": "the prepared pathway-score bundle has no independent subject kinship matrix",
            "prs": "the prepared bundle has no externally estimated PRS weights",
            "temporal_models": (
                "the registered baseline progression/prognosis tasks do not yet define a "
                "leakage-safe longitudinal observation window and prediction horizon"
            ),
        },
    }
    write_json(output_dir / "sweep_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0

    records: list[dict[str, Any]] = []
    status_path = output_dir / "task_status.jsonl"
    for index, task in enumerate(tasks, 1):
        print(
            f"[{index}/{len(tasks)}] {task.case_study} model={task.model} seed={task.seed}",
            flush=True,
        )
        record = run_task(task, output_dir / "logs", resume=args.resume)
        records.append(record)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if record["status"] != "completed" and args.fail_fast:
            raise RuntimeError(f"Task failed: {record}")

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "model_sweep_results.csv", index=False)
    compact_metric_summary(summary).to_csv(
        output_dir / "model_sweep_primary_metrics.csv", index=False
    )
    metric_columns = [
        column
        for column in summary.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "returncode", "elapsed_sec"}
    ]
    if metric_columns:
        aggregate = summary.groupby(["case_study", "model"], as_index=False)[
            metric_columns
        ].agg(["mean", "var", "count"])
        aggregate.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in aggregate.columns
        ]
        aggregate.to_csv(output_dir / "model_sweep_summary.csv", index=False)
    audit = {
        "created_at": utc_now(),
        "n_tasks": len(records),
        "n_completed": sum(row["status"] == "completed" for row in records),
        "n_failed": sum(row["status"] != "completed" for row in records),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "model_sweep_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
