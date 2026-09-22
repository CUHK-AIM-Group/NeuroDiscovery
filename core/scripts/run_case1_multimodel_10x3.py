"""Run and aggregate the formal CS1 10-seed x 3-fold model experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "core" / "scripts" / "case1_multimodel_pilot.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scripts.case1_multimodel_pilot import (  # noqa: E402
    DEFAULT_MODELS,
    summarize_performance,
    write_csv_with_retry,
)


DEFAULT_DATA_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\transdiag_preprocessed")
DEFAULT_OUTPUT_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\case1_multimodel_10x3")
FORMAL_ATLASES = (
    "aal_116",
    "aal3_166",
    "basc_122",
    "cc200",
    "cc400",
    "destrieux_148",
    "dk_112",
    "dosenbach_160",
    "eickhoff_zilles",
    "glasser_360",
    "harvard_oxford_cort",
    "harvard_oxford_merged",
    "harvard_oxford_sub",
    "msdl_39",
    "power_264",
    "schaefer_100_7net",
    "schaefer_200_7net",
    "schaefer_400_7net",
    "talairach_tournoux",
)
FORMAL_DISEASES = (
    "ADHD",
    "MDD_depression",
    "OCD_OC_related",
    "PTSD_trauma",
    "anxiety",
    "bipolar",
    "eating_disorder",
    "psychosis_SZ_SZA",
    "substance_use",
)
FORMAL_SEEDS = tuple(range(20260816, 20260826))
FORMAL_FOLDS = 3
EXPECTED_MODEL_FOLDS_PER_SEED = (
    len(FORMAL_ATLASES) * len(FORMAL_DISEASES) * len(DEFAULT_MODELS) * FORMAL_FOLDS
)
EXPECTED_MODEL_FOLDS = EXPECTED_MODEL_FOLDS_PER_SEED * len(FORMAL_SEEDS)

ATTRIBUTION_KEYS = (
    "model",
    "atlas",
    "disease",
    "evidence_resolution",
    "roi_index",
    "roi_id",
    "roi_name",
    "hemisphere",
    "network",
    "feature",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def protocol_payload(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, object]:
    return {
        "schema_version": "case1-multimodel-robustness-protocol.v1",
        "created_at": utc_now(),
        "case_study_id": "case1_transdiagnostic",
        "candidate_unit": "disease_x_roi_x_imaging_feature",
        "atlases": list(FORMAL_ATLASES),
        "diseases": list(FORMAL_DISEASES),
        "models": list(DEFAULT_MODELS),
        "seeds": list(seeds),
        "folds": FORMAL_FOLDS,
        "seed_unit": "independent model-training replicate",
        "fold_unit": "held-out subject partition nested within seed",
        "aggregation": (
            "average the three folds within each seed; use seeds as independent "
            "replicates for means, sample variances, and method comparisons"
        ),
        "split_stratification": "diagnosis_x_site_if_feasible",
        "covariates": "age, sex, site, and mean_fd when available; fit on training fold only",
        "epochs": args.epochs,
        "patience": args.patience,
        "graph_density": args.graph_density,
        "expected_model_folds_per_seed": EXPECTED_MODEL_FOLDS_PER_SEED,
        "expected_model_folds": (
            len(FORMAL_ATLASES)
            * len(FORMAL_DISEASES)
            * len(DEFAULT_MODELS)
            * len(seeds)
            * FORMAL_FOLDS
        ),
        "formula": (
            f"{len(FORMAL_ATLASES)} atlases x {len(FORMAL_DISEASES)} diseases x "
            f"{len(DEFAULT_MODELS)} models x {len(seeds)} seeds x {FORMAL_FOLDS} folds"
        ),
        "transdiag_root": str(args.transdiag_root),
        "output_root": str(args.output_root),
    }


def seed_run_complete(run_dir: Path, seed: int) -> bool:
    manifest_path = run_dir / "manifest.json"
    performance_path = run_dir / "performance_folds.csv"
    if not manifest_path.is_file() or not performance_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_valid = bool(
            manifest.get("atlases") == list(FORMAL_ATLASES)
            and manifest.get("diseases") == list(FORMAL_DISEASES)
            and manifest.get("models") == list(DEFAULT_MODELS)
            and manifest.get("seeds") == [seed]
            and int(manifest.get("folds", 0)) == FORMAL_FOLDS
            and int(manifest.get("n_model_folds", 0))
            == EXPECTED_MODEL_FOLDS_PER_SEED
            and int(manifest.get("n_attribution_model_folds", 0))
            == EXPECTED_MODEL_FOLDS_PER_SEED
            and int(manifest.get("n_failed_model_folds", 0)) == 0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if not manifest_valid:
        return False

    try:
        performance = pd.read_csv(performance_path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    keys = ["atlas", "disease", "model", "seed", "fold"]
    if set(keys) - set(performance.columns):
        return False
    if len(performance) != EXPECTED_MODEL_FOLDS_PER_SEED:
        return False
    if performance.duplicated(keys).any():
        return False
    try:
        observed_seeds = set(pd.to_numeric(performance["seed"], errors="raise").astype(int))
        observed_folds = set(pd.to_numeric(performance["fold"], errors="raise").astype(int))
    except (ValueError, TypeError):
        return False
    if observed_seeds != {seed} or observed_folds != set(range(FORMAL_FOLDS)):
        return False
    if set(performance["atlas"].astype(str)) != set(FORMAL_ATLASES):
        return False
    if set(performance["disease"].astype(str)) != set(FORMAL_DISEASES):
        return False
    if set(performance["model"].astype(str)) != set(DEFAULT_MODELS):
        return False

    failure_path = run_dir / "failed_model_folds.csv"
    if failure_path.is_file() and failure_path.stat().st_size:
        try:
            if not pd.read_csv(failure_path).empty:
                return False
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return False
    return bool(
        (run_dir / "heldout_attribution_by_seed.csv").is_file()
        or (run_dir / "heldout_attribution_summary.csv").is_file()
    )


def run_seed(
    args: argparse.Namespace,
    seed: int,
    protocol_path: Path,
) -> None:
    run_name = f"seed_{seed}"
    run_dir = args.output_root / run_name
    if seed_run_complete(run_dir, seed):
        print(f"[seed {seed}] complete; reusing {run_dir}", flush=True)
        return

    command = [
        sys.executable,
        str(PILOT),
        "--case-study-id",
        "case1_transdiagnostic",
        "--transdiag-root",
        str(args.transdiag_root),
        "--out-root",
        str(args.output_root),
        "--run-name",
        run_name,
        "--atlases",
        ",".join(FORMAL_ATLASES),
        "--diseases",
        ",".join(FORMAL_DISEASES),
        "--models",
        ",".join(DEFAULT_MODELS),
        "--folds",
        str(FORMAL_FOLDS),
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--graph-density",
        str(args.graph_density),
        "--device",
        args.device,
        "--split-stratification",
        "diagnosis_x_site_if_feasible",
        "--protocol-manifest",
        str(protocol_path),
    ]
    if run_dir.exists():
        command.append("--resume")
    if args.fail_fast:
        command.append("--fail-fast")

    log_path = args.output_root / f"seed_{seed}.log"
    print(f"[seed {seed}] running; log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"seed {seed} failed with exit code {result.returncode}; see {log_path}")
    if not seed_run_complete(run_dir, seed):
        raise RuntimeError(f"seed {seed} finished without a complete manifest: {run_dir}")


def _attribution_source(run_dir: Path) -> tuple[Path, str, str]:
    by_seed = run_dir / "heldout_attribution_by_seed.csv"
    if by_seed.is_file():
        return by_seed, "importance_abs", "signed_attribution"
    legacy = run_dir / "heldout_attribution_summary.csv"
    if legacy.is_file():
        return legacy, "importance_abs_mean", "signed_attribution_mean"
    raise FileNotFoundError(f"missing attribution summary in {run_dir}")


def aggregate_attributions(output_root: Path, seeds: tuple[int, ...]) -> int:
    base_keys: pd.DataFrame | None = None
    importance_sum: np.ndarray | None = None
    importance_sum_sq: np.ndarray | None = None
    signed_sum: np.ndarray | None = None
    signed_sum_sq: np.ndarray | None = None

    for seed in seeds:
        source, importance_column, signed_column = _attribution_source(
            output_root / f"seed_{seed}"
        )
        print(f"[aggregate] attribution seed={seed} source={source.name}", flush=True)
        frame = pd.read_csv(source, low_memory=False)
        missing = set(ATTRIBUTION_KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"{source} is missing attribution keys: {sorted(missing)}")
        if "n_folds" in frame.columns and not (frame["n_folds"] == FORMAL_FOLDS).all():
            raise ValueError(f"{source} does not contain exactly {FORMAL_FOLDS} folds per row")
        keys = frame[list(ATTRIBUTION_KEYS)].copy()
        importance = pd.to_numeric(frame[importance_column], errors="raise").to_numpy(np.float64)
        signed = pd.to_numeric(frame[signed_column], errors="raise").to_numpy(np.float64)
        if not np.isfinite(importance).all() or not np.isfinite(signed).all():
            raise ValueError(f"non-finite attribution values in {source}")

        if base_keys is None:
            base_keys = keys
            importance_sum = importance.copy()
            importance_sum_sq = np.square(importance)
            signed_sum = signed.copy()
            signed_sum_sq = np.square(signed)
        else:
            if not keys.equals(base_keys):
                raise ValueError(f"attribution key order/content differs for seed {seed}")
            importance_sum += importance
            importance_sum_sq += np.square(importance)
            signed_sum += signed
            signed_sum_sq += np.square(signed)
        del frame, keys, importance, signed

    if base_keys is None:
        return 0
    assert importance_sum is not None and importance_sum_sq is not None
    assert signed_sum is not None and signed_sum_sq is not None
    n = len(seeds)
    if n > 1:
        importance_variance = np.maximum(
            (importance_sum_sq - np.square(importance_sum) / n) / (n - 1),
            0.0,
        )
        signed_variance = np.maximum(
            (signed_sum_sq - np.square(signed_sum) / n) / (n - 1),
            0.0,
        )
    else:
        importance_variance = np.full(len(base_keys), np.nan)
        signed_variance = np.full(len(base_keys), np.nan)

    summary = base_keys.assign(
        importance_abs_mean=importance_sum / n,
        importance_abs_variance=importance_variance,
        signed_attribution_mean=signed_sum / n,
        signed_attribution_variance=signed_variance,
        n_seeds=n,
    )
    write_csv_with_retry(summary, output_root / "heldout_attribution_summary.csv")
    return len(summary)


def aggregate_runs(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    protocol: dict[str, object],
) -> dict[str, object]:
    fold_frames = []
    for seed in seeds:
        run_dir = args.output_root / f"seed_{seed}"
        if not seed_run_complete(run_dir, seed):
            raise RuntimeError(f"cannot aggregate incomplete seed run: {run_dir}")
        frame = pd.read_csv(run_dir / "performance_folds.csv")
        if len(frame) != EXPECTED_MODEL_FOLDS_PER_SEED:
            raise ValueError(f"seed {seed} has {len(frame)} model folds")
        fold_frames.append(frame)

    performance = pd.concat(fold_frames, ignore_index=True)
    performance = performance.drop_duplicates(
        ["atlas", "disease", "model", "seed", "fold"], keep="last"
    )
    expected = EXPECTED_MODEL_FOLDS_PER_SEED * len(seeds)
    if len(performance) != expected:
        raise ValueError(f"combined performance has {len(performance)} rows; expected {expected}")
    write_csv_with_retry(performance, args.output_root / "performance_folds.csv")
    seed_summary, summary, best_by_cell, best_by_disease = summarize_performance(performance)
    write_csv_with_retry(seed_summary, args.output_root / "performance_by_seed.csv")
    write_csv_with_retry(summary, args.output_root / "performance_summary.csv")
    write_csv_with_retry(best_by_cell, args.output_root / "best_model_by_atlas_disease.csv")
    write_csv_with_retry(best_by_disease, args.output_root / "best_atlas_model_by_disease.csv")
    n_attribution_rows = aggregate_attributions(args.output_root, seeds)

    manifest = {
        **protocol,
        "schema_version": "case1-multimodel-robustness-aggregate.v1",
        "status": "complete",
        "completed_at": utc_now(),
        "n_model_folds": len(performance),
        "n_attribution_model_folds": len(performance),
        "n_seed_level_model_tasks": len(seed_summary),
        "n_attribution_rows": n_attribution_rows,
        "n_failed_model_folds": 0,
        "seed_run_directories": [str(args.output_root / f"seed_{seed}") for seed in seeds],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transdiag-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--graph-density", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args, seeds)
    protocol_path = args.output_root / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)

    if not args.aggregate_only:
        for seed in seeds:
            run_seed(args, seed, protocol_path)
    manifest = aggregate_runs(args, seeds, protocol)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
