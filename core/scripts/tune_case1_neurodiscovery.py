"""Tune Case Study 1 NeuroDiscovery without reading external validation data.

The tuner uses stable disease-by-source-by-ROI folds, keeping every feature of
one ROI together. Four inner folds select a configuration by cross-validation,
while a fifth fold remains outcome-masked until a single winning configuration
has been frozen. The four external cohorts are never opened by this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from core.scripts.canonical_kg_release import (
        CURRENT_CANONICAL_SHA256,
        validate_canonical_kg_release,
        write_release_manifest,
    )
    from core.scripts.case1_method_comparison import (
        add_generator_scores,
        closed_loop_neurodiscovery_order,
        kg_query_terms_for_candidates,
        load_kg_index,
        load_results,
        neurodiscovery_score_arrays,
        order_from_scores,
        sha256_file,
    )
    from core.scripts.case1_neurodiscovery_config import Case1NeuroDiscoveryConfig
    from core.scripts.case_study_score_components import (
        embedded_score_component_audit,
        load_score_component_bundle,
    )
except ModuleNotFoundError:
    from canonical_kg_release import (
        CURRENT_CANONICAL_SHA256,
        validate_canonical_kg_release,
        write_release_manifest,
    )
    from case1_method_comparison import (
        add_generator_scores,
        closed_loop_neurodiscovery_order,
        kg_query_terms_for_candidates,
        load_kg_index,
        load_results,
        neurodiscovery_score_arrays,
        order_from_scores,
        sha256_file,
    )
    from case1_neurodiscovery_config import Case1NeuroDiscoveryConfig
    from case_study_score_components import (
        embedded_score_component_audit,
        load_score_component_bundle,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_KG = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_CLAIMS = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
DEFAULT_STATE = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "CURRENT_STATE.json"
DEFAULT_OUT_DIR = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_tuning"
    r"\20260810_kgdd0d4037_roi_cv_v2"
)
TUNING_BUDGETS = (5_000, 10_000, 50_000)
OBJECTIVE_WEIGHTS = {5_000: 0.30, 10_000: 0.40, 50_000: 0.30}


def stable_fold(disease: str, source: str, roi_key: str, n_folds: int = 5) -> int:
    key = f"{disease}|{source}|{roi_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % n_folds


def add_stable_folds(scored: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    groups = scored[["disease", "source", "roi_key"]].astype(str).drop_duplicates()
    lookup = {
        (row.disease, row.source, row.roi_key): stable_fold(
            row.disease, row.source, row.roi_key, n_folds
        )
        for row in groups.itertuples(index=False)
    }
    return np.fromiter(
        (
            lookup[(str(disease), str(source), str(roi_key))]
            for disease, source, roi_key in zip(
                scored["disease"], scored["source"], scored["roi_key"], strict=False
            )
        ),
        dtype=np.int8,
        count=len(scored),
    )


def config_id(config: Case1NeuroDiscoveryConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def tuning_code_provenance() -> dict[str, object]:
    paths = {
        "tuner": Path(__file__).resolve(),
        "closed_loop": REPO_ROOT / "core/scripts/case1_method_comparison.py",
        "config": REPO_ROOT / "core/scripts/case1_neurodiscovery_config.py",
        "overlay_engine": REPO_ROOT / "core/scripts/case_study_closed_loop_engine.py",
    }
    return {name: artifact_record(path) for name, path in paths.items()}


def feedback_masking_contract() -> dict[str, object]:
    return {
        "hidden_effect_size": 0.0,
        "hidden_p_value": 1.0,
        "hidden_feedback_available": False,
        "hidden_rows_update_factor_or_pair_feedback": False,
        "purpose": (
            "Evaluation-fold and outer-holdout outcomes remain unavailable; "
            "they are not converted into inconclusive or failed feedback."
        ),
    }


def masked_feedback_frame(scored: pd.DataFrame, hidden: np.ndarray) -> pd.DataFrame:
    masked = scored.copy(deep=False)
    effects = pd.to_numeric(scored["adjusted_residual_d"], errors="coerce").to_numpy(copy=True)
    p_values = pd.to_numeric(scored["p_value"], errors="coerce").to_numpy(copy=True)
    effects[hidden] = 0.0
    p_values[hidden] = 1.0
    masked["adjusted_residual_d"] = effects
    masked["p_value"] = p_values
    feedback_available = np.ones(len(scored), dtype=bool)
    feedback_available[hidden] = False
    masked["feedback_available"] = feedback_available
    return masked


def fold_metrics(
    order: np.ndarray,
    gt: np.ndarray,
    evaluation_mask: np.ndarray,
    budgets: tuple[int, ...] = TUNING_BUDGETS,
) -> dict[str, float | int]:
    valid_order = order[(order >= 0) & (order < len(gt))]
    ordered_hits = gt[valid_order] & evaluation_mask[valid_order]
    cumulative = np.cumsum(ordered_hits)
    denominator = int(np.sum(gt & evaluation_mask))
    row: dict[str, float | int] = {"gt_total": denominator}
    objective = 0.0
    for budget in budgets:
        effective_budget = min(int(budget), len(valid_order))
        hits = int(cumulative[effective_budget - 1]) if effective_budget else 0
        recall = hits / max(denominator, 1)
        row[f"hits_at_{budget}"] = hits
        row[f"recall_at_{budget}"] = recall
        objective += OBJECTIVE_WEIGHTS[budget] * recall
    row["objective"] = objective
    return row


def static_configs(*, include_frozen_components: bool = False) -> list[Case1NeuroDiscoveryConfig]:
    del include_frozen_components
    configs = [Case1NeuroDiscoveryConfig()]
    for feature_weight in (0.10, 0.20, 0.30, 0.40, 0.50):
        for scoped_weight in (0.10, 0.20, 0.30):
            configs.append(
                Case1NeuroDiscoveryConfig(
                    global_support_weight=1.0 - scoped_weight,
                    scoped_support_weight=scoped_weight,
                    feature_support_weight=feature_weight,
                )
            )
    return list({config_id(config): config for config in configs}.values())


def schedule_variants(
    static_config: Case1NeuroDiscoveryConfig,
    max_search_budget: int,
) -> list[Case1NeuroDiscoveryConfig]:
    schedules = (
        {},
        {
            "warmup_budget": 7_500,
            "warmup_exploit_fraction": 0.60,
            "feature_support_decay_budget": 50_000,
            "feedback_weight": 0.18,
            "pair_feedback_weight": 0.16,
            "inconclusive_search_failure_weight": 0.40,
            "exploration_weight": 0.010,
            "pair_feedback_start_fraction": 0.10,
            "pair_feedback_force_fraction": 0.20,
            "min_hits_for_pair_feedback": 20,
            "pre_pair_exploit_fraction": 0.90,
            "post_pair_exploit_fraction": 0.94,
        },
        {
            "warmup_budget": 7_500,
            "warmup_exploit_fraction": 0.60,
            "feature_support_decay_budget": 50_000,
            "feedback_weight": 0.18,
            "pair_feedback_weight": 0.16,
            "inconclusive_search_failure_weight": 0.40,
            "exploration_weight": 0.010,
            "pair_feedback_start_fraction": 0.10,
            "pair_feedback_force_fraction": 0.20,
            "min_hits_for_pair_feedback": 20,
            "pre_pair_exploit_fraction": 0.90,
            "post_pair_exploit_fraction": 0.94,
            "kge_weight": 1.0,
            "novelty_weight": 1.0,
            "critic_weight": 1.0,
        },
        {
            "warmup_budget": 5_000,
            "warmup_exploit_fraction": 0.70,
            "feature_support_decay_budget": 30_000,
            "feedback_weight": 0.16,
            "pair_feedback_weight": 0.14,
            "inconclusive_search_failure_weight": 0.50,
            "exploration_weight": 0.015,
            "pair_feedback_start_fraction": 0.08,
            "pair_feedback_force_fraction": 0.20,
            "min_hits_for_pair_feedback": 15,
            "pre_pair_exploit_fraction": 0.90,
            "post_pair_exploit_fraction": 0.93,
        },
        {
            "warmup_budget": 5_000,
            "warmup_exploit_fraction": 0.70,
            "feature_support_decay_budget": 30_000,
            "feedback_weight": 0.16,
            "pair_feedback_weight": 0.14,
            "inconclusive_search_failure_weight": 0.50,
            "exploration_weight": 0.015,
            "pair_feedback_start_fraction": 0.08,
            "pair_feedback_force_fraction": 0.20,
            "min_hits_for_pair_feedback": 15,
            "pre_pair_exploit_fraction": 0.90,
            "post_pair_exploit_fraction": 0.93,
            "kge_weight": 1.0,
            "novelty_weight": 1.0,
            "critic_weight": 1.0,
        },
    )
    configs = []
    for schedule in schedules:
        configs.append(
            replace(
                static_config,
                max_closed_loop_budget=max_search_budget,
                **schedule,
            )
        )
    return list({config_id(config): config for config in configs}.values())


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in frame.columns
        if column.startswith("hits_at_")
        or column.startswith("recall_at_")
        or column == "objective"
    ]
    rows = []
    for config_key, group in frame.groupby("config_id", sort=False):
        row: dict[str, object] = {
            "config_id": config_key,
            "n_runs": int(len(group)),
            "config_json": group.iloc[0]["config_json"],
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_variance"] = (
                float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_min"] = float(np.min(values))
        row["robust_objective"] = (
            0.75 * float(row["objective_mean"])
            + 0.25 * float(row["objective_min"])
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ("robust_objective", "objective_mean", "recall_at_10000_mean")
        if column in summary
    ]
    return summary.sort_values(sort_columns, ascending=False)


def run_dynamic_configs(
    scored: pd.DataFrame,
    folds: np.ndarray,
    configs: list[Case1NeuroDiscoveryConfig],
    *,
    hidden_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    seeds: list[int],
    stage: str,
    evaluation_fold: int,
    checkpoint_path: Path | None = None,
    completed_keys: set[tuple[str, int, str, int]] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    masked = masked_feedback_frame(scored, hidden_mask)
    gt = scored["is_gt_top"].to_numpy(bool)
    rows: list[dict[str, object]] = []
    total = len(configs) * len(seeds)
    completed_keys = completed_keys if completed_keys is not None else set()
    jobs: list[tuple[Case1NeuroDiscoveryConfig, str, np.ndarray, int]] = []
    for config in configs:
        current_config_id = config_id(config)
        _global, _scoped, combined = neurodiscovery_score_arrays(scored, config)
        for seed in seeds:
            checkpoint_key = (stage, evaluation_fold, current_config_id, seed)
            if checkpoint_key in completed_keys:
                continue
            jobs.append((config, current_config_id, combined, seed))

    def execute(
        job: tuple[Case1NeuroDiscoveryConfig, str, np.ndarray, int]
    ) -> dict[str, object]:
        config, current_config_id, combined, seed = job
        run_frame = masked.copy(deep=False)
        run_frame["score_neurodiscovery"] = combined
        started = time.perf_counter()
        order = closed_loop_neurodiscovery_order(
            run_frame,
            np.random.default_rng(seed),
            seed=seed,
            trial=0,
            config=config,
        )
        metrics = fold_metrics(order, gt, evaluation_mask)
        return {
            "stage": stage,
            "evaluation_fold": evaluation_fold,
            "config_id": current_config_id,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
            "config_json": json.dumps(config.to_dict(), sort_keys=True),
            **metrics,
        }

    completed = total - len(jobs)
    max_workers = max(1, min(int(workers), len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            rows.append(row)
            checkpoint_key = (
                str(row["stage"]),
                int(row["evaluation_fold"]),
                str(row["config_id"]),
                int(row["seed"]),
            )
            completed_keys.add(checkpoint_key)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[{stage} {completed}/{total}] {row['config_id']} "
                f"seed={row['seed']} "
                f"objective={float(row['objective']):.6f} "
                f"hits@10k={int(row['hits_at_10000'])}",
                flush=True,
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=260810)
    parser.add_argument("--search-seeds", type=int, default=2)
    parser.add_argument("--holdout-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-static", type=int, default=2)
    parser.add_argument("--max-search-budget", type=int, default=50_000)
    parser.add_argument("--formal-closed-loop-budget", type=int, default=120_000)
    parser.add_argument("--score-components", type=Path)
    parser.add_argument("--score-components-manifest", type=Path)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Write development-fold static screens and stop before closed-loop search.",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Stop after inner-CV selection so the outer fold remains sealed for refinement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Validating canonical KG release...", flush=True)
    release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id="case1_transdiagnostic",
        expected_sha256=CURRENT_CANONICAL_SHA256,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    release_path = args.out_dir / "canonical_kg_release.json"
    if release_path.exists():
        previous_release = json.loads(release_path.read_text(encoding="utf-8"))
        if previous_release != release:
            raise ValueError("output directory belongs to a different canonical KG release")
    write_release_manifest(release_path, release)
    print("Loading exhaustive results...", flush=True)
    scored = load_results(args.all_tests, args.gt_top_frac)
    print("Loading streamed KG index...", flush=True)
    kg = load_kg_index(args.kg, kg_query_terms_for_candidates(scored))
    print("Computing outcome-blind static components...", flush=True)
    scored = add_generator_scores(scored, kg, args.seed)
    if bool(args.score_components) != bool(args.score_components_manifest):
        raise ValueError(
            "--score-components and --score-components-manifest must be supplied together"
        )
    if args.score_components:
        scored, score_component_audit = load_score_component_bundle(
            scored,
            table_path=args.score_components,
            manifest_path=args.score_components_manifest,
        )
    else:
        score_component_audit = embedded_score_component_audit(scored)
    folds = add_stable_folds(scored)
    gt = scored["is_gt_top"].to_numpy(bool)
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    development_folds = (0, 1, 2, 3)
    holdout_mask = folds == 4

    split_summary = (
        scored.assign(tuning_fold=folds)
        .groupby(["tuning_fold", "disease", "source"], dropna=False)
        .agg(candidates=("candidate_id", "size"), gt_hits=("is_gt_top", "sum"))
        .reset_index()
    )
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)

    static_rows = []
    for config in static_configs(
        include_frozen_components=bool(score_component_audit.get("active"))
    ):
        _global, _scoped, combined = neurodiscovery_score_arrays(scored, config)
        order = order_from_scores(combined, candidate_ids)
        for fold in development_folds:
            static_rows.append(
                {
                    "config_id": config_id(config),
                    "config_json": json.dumps(config.to_dict(), sort_keys=True),
                    "evaluation_fold": fold,
                    **fold_metrics(order, gt, folds == fold),
                }
            )
    static_runs = pd.DataFrame(static_rows)
    static_runs.to_csv(args.out_dir / "static_screen_by_fold.csv", index=False)
    static_results = aggregate_runs(static_runs)
    static_results.to_csv(args.out_dir / "static_screen.csv", index=False)
    if args.static_only:
        (args.out_dir / "static_screen_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "case1-neurodiscovery-static-screen.v1",
                    "canonical_kg_release": release,
                    "all_tests": str(args.all_tests),
                    "all_tests_sha256": sha256_file(args.all_tests),
                    "candidate_count": int(len(scored)),
                    "gt_count": int(gt.sum()),
                    "development_folds": list(development_folds),
                    "outer_holdout_fold": 4,
                    "outer_holdout_opened": False,
                    "score_components": score_component_audit,
                    "kg_index_stats": kg.stats,
                    "feedback_masking_contract": feedback_masking_contract(),
                    "code_provenance": tuning_code_provenance(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(static_results.head(10).to_string(index=False), flush=True)
        return
    top_static_configs = [
        Case1NeuroDiscoveryConfig.from_dict(json.loads(raw))
        for raw in static_results.head(args.top_static)["config_json"]
    ]
    dynamic_configs = []
    for static_config in top_static_configs:
        dynamic_configs.extend(schedule_variants(static_config, args.max_search_budget))
    dynamic_configs = list({config_id(config): config for config in dynamic_configs}.values())

    search_seeds = [args.seed + 1009 * index for index in range(args.search_seeds)]
    checkpoint_path = args.out_dir / "development_runs.checkpoint.jsonl"
    checkpoint_rows = []
    if checkpoint_path.exists():
        checkpoint_rows = [
            json.loads(line)
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    valid_config_ids = {config_id(config) for config in dynamic_configs}
    valid_stages = {f"development_fold_{fold}" for fold in development_folds}
    checkpoint_rows = [
        row
        for row in checkpoint_rows
        if str(row.get("stage")) in valid_stages
        and int(row.get("evaluation_fold", -1)) in development_folds
        and str(row.get("config_id")) in valid_config_ids
        and int(row.get("seed", -1)) in search_seeds
    ]
    completed_keys = {
        (
            str(row["stage"]),
            int(row["evaluation_fold"]),
            str(row["config_id"]),
            int(row["seed"]),
        )
        for row in checkpoint_rows
    }
    tuning_parts = [pd.DataFrame(checkpoint_rows)] if checkpoint_rows else []
    for fold in development_folds:
        part = run_dynamic_configs(
            scored,
            folds,
            dynamic_configs,
            hidden_mask=(folds == fold) | holdout_mask,
            evaluation_mask=folds == fold,
            seeds=search_seeds,
            stage=f"development_fold_{fold}",
            evaluation_fold=fold,
            checkpoint_path=checkpoint_path,
            completed_keys=completed_keys,
            workers=args.workers,
        )
        if not part.empty:
            tuning_parts.append(part)
    tuning_runs = pd.concat(tuning_parts, ignore_index=True)
    tuning_runs.to_csv(args.out_dir / "development_runs.csv", index=False)
    tuning_summary = aggregate_runs(tuning_runs)
    tuning_summary.to_csv(args.out_dir / "development_summary.csv", index=False)

    selected_search_config = Case1NeuroDiscoveryConfig.from_dict(
        json.loads(tuning_summary.iloc[0]["config_json"])
    )
    selected_formal_config = replace(
        selected_search_config,
        max_closed_loop_budget=args.formal_closed_loop_budget,
    )
    selected_formal_config.write_json(args.out_dir / "best_config.json")

    if args.skip_holdout:
        manifest = {
            "schema_version": "case1-neurodiscovery-tuning.v1",
            "all_tests": str(args.all_tests),
            "all_tests_sha256": sha256_file(args.all_tests),
            "kg": str(args.kg),
            "kg_sha256": sha256_file(args.kg),
            "canonical_kg_release": release,
            "candidate_count": int(len(scored)),
            "gt_count": int(gt.sum()),
            "split_unit": (
                "disease|source|roi_key (all features for an ROI stay together)"
            ),
            "development_folds": list(development_folds),
            "outer_holdout_fold": 4,
            "outer_holdout_status": "sealed_not_evaluated",
            "objective_budgets": list(TUNING_BUDGETS),
            "objective_weights": OBJECTIVE_WEIGHTS,
            "external_validation_used_for_tuning": False,
            "selected_search_config_id": config_id(selected_search_config),
            "selected_formal_config": selected_formal_config.to_dict(),
            "kg_index_stats": kg.stats,
            "feedback_masking_contract": feedback_masking_contract(),
            "code_provenance": tuning_code_provenance(),
            "artifacts": {
                path.name: artifact_record(path)
                for path in (
                    args.out_dir / "static_screen_by_fold.csv",
                    args.out_dir / "static_screen.csv",
                    args.out_dir / "development_runs.checkpoint.jsonl",
                    args.out_dir / "development_runs.csv",
                    args.out_dir / "development_summary.csv",
                    args.out_dir / "best_config.json",
                )
                if path.is_file()
            },
        }
        (args.out_dir / "tuning_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Selected config: {args.out_dir / 'best_config.json'}", flush=True)
        print("Outer holdout remains sealed (--skip-holdout).", flush=True)
        return

    legacy_search_config = replace(
        Case1NeuroDiscoveryConfig(),
        max_closed_loop_budget=args.max_search_budget,
    )
    holdout_configs = [selected_search_config, legacy_search_config]
    holdout_seeds = [args.seed + 50_000 + 1009 * index for index in range(args.holdout_seeds)]
    holdout_checkpoint = args.out_dir / "outer_holdout_runs.checkpoint.jsonl"
    holdout_checkpoint_rows = []
    if holdout_checkpoint.exists():
        valid_holdout_ids = {config_id(config) for config in holdout_configs}
        holdout_checkpoint_rows = [
            row
            for line in holdout_checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
            if str(row.get("stage")) == "outer_holdout"
            and int(row.get("evaluation_fold", -1)) == 4
            and str(row.get("config_id")) in valid_holdout_ids
            and int(row.get("seed", -1)) in holdout_seeds
        ]
    completed_holdout_keys = {
        (
            str(row["stage"]),
            int(row["evaluation_fold"]),
            str(row["config_id"]),
            int(row["seed"]),
        )
        for row in holdout_checkpoint_rows
    }
    new_holdout_runs = run_dynamic_configs(
        scored,
        folds,
        holdout_configs,
        hidden_mask=holdout_mask,
        evaluation_mask=holdout_mask,
        seeds=holdout_seeds,
        stage="outer_holdout",
        evaluation_fold=4,
        checkpoint_path=holdout_checkpoint,
        completed_keys=completed_holdout_keys,
        workers=args.workers,
    )
    holdout_parts = [pd.DataFrame(holdout_checkpoint_rows)] if holdout_checkpoint_rows else []
    if not new_holdout_runs.empty:
        holdout_parts.append(new_holdout_runs)
    holdout_runs = pd.concat(holdout_parts, ignore_index=True)
    holdout_runs["configuration"] = np.where(
        holdout_runs["config_id"] == config_id(selected_search_config),
        "selected",
        "legacy",
    )
    holdout_runs.to_csv(args.out_dir / "outer_holdout_runs.csv", index=False)
    holdout_summary = (
        holdout_runs.groupby("configuration", sort=False)
        .agg(
            n_runs=("seed", "size"),
            objective_mean=("objective", "mean"),
            objective_variance=("objective", "var"),
            hits_at_5000_mean=("hits_at_5000", "mean"),
            hits_at_10000_mean=("hits_at_10000", "mean"),
            hits_at_50000_mean=("hits_at_50000", "mean"),
            recall_at_5000_mean=("recall_at_5000", "mean"),
            recall_at_10000_mean=("recall_at_10000", "mean"),
            recall_at_50000_mean=("recall_at_50000", "mean"),
        )
        .reset_index()
    )
    holdout_summary.to_csv(args.out_dir / "outer_holdout_summary.csv", index=False)

    manifest = {
        "schema_version": "case1-neurodiscovery-tuning.v1",
        "all_tests": str(args.all_tests),
        "all_tests_sha256": sha256_file(args.all_tests),
        "kg": str(args.kg),
        "kg_sha256": sha256_file(args.kg),
        "canonical_kg_release": release,
        "candidate_count": int(len(scored)),
        "gt_count": int(gt.sum()),
        "split_unit": "disease|source|roi_key (all features for an ROI stay together)",
        "development_folds": list(development_folds),
        "outer_holdout_fold": 4,
        "inner_training_rule": (
            "For each inner fold, feedback uses the other three inner folds; "
            "the evaluated inner fold and outer fold are outcome-masked."
        ),
        "objective_budgets": list(TUNING_BUDGETS),
        "objective_weights": OBJECTIVE_WEIGHTS,
        "external_validation_used_for_tuning": False,
        "development_effects_hidden_from_feedback_per_fold": True,
        "outer_holdout_effects_hidden_until_config_freeze": True,
        "selected_search_config_id": config_id(selected_search_config),
        "selected_formal_config": selected_formal_config.to_dict(),
        "legacy_config": Case1NeuroDiscoveryConfig().to_dict(),
        "kg_index_stats": kg.stats,
        "score_components": score_component_audit,
        "feedback_masking_contract": feedback_masking_contract(),
        "code_provenance": tuning_code_provenance(),
        "artifacts": {
            path.name: artifact_record(path)
            for path in (
                args.out_dir / "static_screen_by_fold.csv",
                args.out_dir / "static_screen.csv",
                args.out_dir / "development_runs.checkpoint.jsonl",
                args.out_dir / "development_runs.csv",
                args.out_dir / "development_summary.csv",
                args.out_dir / "best_config.json",
                args.out_dir / "outer_holdout_runs.checkpoint.jsonl",
                args.out_dir / "outer_holdout_runs.csv",
                args.out_dir / "outer_holdout_summary.csv",
            )
            if path.is_file()
        },
    }
    (args.out_dir / "tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected config: {args.out_dir / 'best_config.json'}", flush=True)
    print(holdout_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
