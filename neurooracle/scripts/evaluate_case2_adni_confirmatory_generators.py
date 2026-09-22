"""Freeze and execute the formal Case Study 2 generator comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import ClosedLoopConfig
from core.scripts.case_study_closed_loop_engine import run_closed_loop_order
from core.scripts.case_study_feedback_adapters import adapter_for
from neurooracle.scripts.case2_confirmatory_generator_metrics import (
    aggregate_trials,
    evaluate_orders,
    paired_primary_comparisons,
)
from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import CASE2_ROOT
from neurooracle.scripts.run_case2_adni_confirmatory_mediation import (
    _access_record_name,
    _heldout_output_name,
    _holdout_axis,
    _ranking_lock_status,
    _result_lock_name,
    verify_pre_outcome_locks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_phase_holdout_v3"
DEFAULT_PLAN_CONFIG = (
    REPO_ROOT / "neurooracle" / "configs" / "case2_adni_generator_evaluation_v3.json"
)
NEURODISCOVERY = "neurodiscovery"
BASELINE_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _default_output_root(locks: Mapping[str, Any]) -> Path:
    return Path(locks["run_root"]) / "generator_evaluation"


def _primary_metric_names(plan: Mapping[str, Any]) -> list[str]:
    if "primary_metrics" in plan:
        return [str(item["name"]) for item in plan["primary_metrics"]]
    return [str(plan["primary_metric"]["name"])]


def freeze_evaluation_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Commit metrics and all generator ranking inputs before outcome access."""

    locks = verify_pre_outcome_locks(
        args.protocol_root,
        baseline_root=args.baseline_root,
        neurodiscovery_root=args.neurodiscovery_root,
    )
    output_root = args.output_root or _default_output_root(locks)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = locks["protocol"]
    access_path = (
        Path(locks["run_root"])
        / _heldout_output_name(protocol)
        / _access_record_name(protocol)
    )
    if access_path.exists():
        raise RuntimeError("generator evaluation plan cannot be frozen after outcome access")
    config = _load_json(args.plan_config)
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("generator evaluation plan targets another protocol")
    if config["budgets"] != protocol["generator_evaluation"]["budgets"]:
        raise ValueError("evaluation-plan budgets differ from the frozen protocol")
    if int(config["independent_runs_per_method"]) != int(
        protocol["generator_evaluation"]["independent_runs_per_method"]
    ):
        raise ValueError("evaluation-plan run count differs from the frozen protocol")
    expected_methods = {*BASELINE_METHODS, NEURODISCOVERY}
    if set(config["methods"]) != expected_methods:
        raise ValueError("evaluation plan does not contain exactly six baselines and ND")

    snapshot_path = output_root / "LOCKED_EVALUATION_PLAN.json"
    snapshot_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plan_lock = {
        "schema_version": "neurooracle.case2_generator_evaluation_lock.v1",
        "status": _ranking_lock_status(protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_freeze_id": locks["protocol_manifest"]["freeze_id"],
        "association_results_accessed": False,
        "evaluation_plan_path": str(snapshot_path),
        "evaluation_plan_sha256": _sha256(snapshot_path),
        "source_plan_sha256": _sha256(args.plan_config),
        "baseline_ranking_lock_sha256": locks["baseline_lock_sha256"],
        "neurodiscovery_initial_ranking_lock_sha256": locks[
            "neurodiscovery_lock_sha256"
        ],
        "primary_metrics": _primary_metric_names(config),
        "clear_sota_rule": config["clear_sota_rule"],
        "freeze_semantics": {
            "temporal_kg_freeze": False,
            "temporal_freeze_year": None,
            "kg_snapshot_pinning": True,
            "static_baseline_ranking_freeze": True,
            "neurodiscovery_initial_ranking_freeze": True,
            "neurodiscovery_sequential_batch_commit": True,
        },
    }
    lock_path = output_root / "EVALUATION_PLAN.lock.json"
    if lock_path.exists() and not args.force:
        existing = _load_json(lock_path)
        comparable = {
            key: value
            for key, value in plan_lock.items()
            if key != "created_at_utc"
        }
        existing_comparable = {
            key: value
            for key, value in existing.items()
            if key != "created_at_utc"
        }
        if existing_comparable != comparable:
            raise FileExistsError("existing evaluation-plan lock differs from this plan")
        return existing
    lock_path.write_text(
        json.dumps(plan_lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan_lock


def verify_evaluation_plan(
    args: argparse.Namespace,
    locks: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    output_root = args.output_root or _default_output_root(locks)
    lock_path = output_root / "EVALUATION_PLAN.lock.json"
    plan_lock = _load_json(lock_path)
    if plan_lock.get("status") != _ranking_lock_status(locks["protocol"]):
        raise ValueError("generator evaluation plan is not pre-outcome locked")
    if plan_lock.get("protocol_freeze_id") != locks["protocol_manifest"]["freeze_id"]:
        raise ValueError("generator evaluation plan belongs to another protocol")
    if plan_lock.get("baseline_ranking_lock_sha256") != locks["baseline_lock_sha256"]:
        raise ValueError("baseline ranking lock changed after evaluation-plan freeze")
    if plan_lock.get("neurodiscovery_initial_ranking_lock_sha256") != locks[
        "neurodiscovery_lock_sha256"
    ]:
        raise ValueError("NeuroDiscovery ranking lock changed after evaluation-plan freeze")
    snapshot_path = Path(plan_lock["evaluation_plan_path"])
    if _sha256(snapshot_path) != plan_lock.get("evaluation_plan_sha256"):
        raise ValueError("locked generator evaluation plan hash mismatch")
    return _load_json(snapshot_path), lock_path


class BatchCommitWriter:
    """Durably hash-commit each selected batch before its outcome lookup."""

    def __init__(
        self,
        path: Path,
        *,
        protocol_freeze_id: str,
        initial_ranking_commit: str,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8")
        self.protocol_freeze_id = protocol_freeze_id
        self.previous_hash = initial_ranking_commit
        self.count = 0

    def __call__(self, selection: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": "neurooracle.case2_batch_ranking_commit.v1",
            "protocol_freeze_id": self.protocol_freeze_id,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "previous_commit_sha256": self.previous_hash,
            "selection": dict(selection),
            "outcome_values_in_commit": False,
        }
        commit_sha = _canonical_sha(payload)
        payload["commit_sha256"] = commit_sha
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.previous_hash = commit_sha
        self.count += 1
        return {
            "commit_sha256": commit_sha,
            "commit_index": self.count,
            "durably_flushed_before_outcome_lookup": True,
        }

    def close(self) -> dict[str, Any]:
        self.handle.close()
        return {
            "path": str(self.path),
            "sha256": _sha256(self.path),
            "commits": self.count,
            "final_commit_sha256": self.previous_hash,
        }


def _load_confirmation_results(
    confirmation_root: Path,
    *,
    protocol_freeze_id: str,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    result_lock_path = confirmation_root / _result_lock_name(dict(protocol))
    result_lock = _load_json(result_lock_path)
    if result_lock.get("protocol_freeze_id") != protocol_freeze_id:
        raise ValueError("confirmation results belong to another protocol freeze")
    if result_lock.get("holdout_axis", _holdout_axis(dict(protocol))) != _holdout_axis(
        dict(protocol)
    ):
        raise ValueError("held-out results use another holdout axis")
    manifest_path = confirmation_root / "manifest.json"
    if _sha256(manifest_path) != result_lock.get("manifest_sha256"):
        raise ValueError("confirmation result manifest hash mismatch")
    manifest = _load_json(manifest_path)
    result_path = Path(manifest["outputs"]["results"])
    if _sha256(result_path) != result_lock.get("results_sha256"):
        raise ValueError("confirmation result table hash mismatch")
    return pd.read_parquet(result_path), result_lock, result_lock_path


def _load_static_orders(
    baseline_lock_path: Path,
    candidate_ids: list[str],
) -> dict[tuple[str, int], list[str]]:
    lock = _load_json(baseline_lock_path)
    orders: dict[tuple[str, int], list[str]] = {}
    with np.load(Path(lock["orders_path"]), allow_pickle=False) as archive:
        for record in lock["orders"]:
            index = np.asarray(archive[record["array_key"]], dtype=np.int64)
            orders[(str(record["method"]), int(record["trial"]))] = [
                candidate_ids[value] for value in index
            ]
    return orders


def _feedback_outcomes(
    results: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    frame = results.copy()
    supported_column = (
        "nominal_chain_hit"
        if _holdout_axis(dict(protocol)) == "cohort_phase"
        else "family_fdr_chain_hit"
    )
    supported = frame[supported_column].astype(bool)
    executable = frame["executable"].astype(bool)
    frame["validated"] = supported
    frame["execution_succeeded"] = executable
    frame["feedback_status"] = np.select(
        [supported, ~executable],
        ["supported", "execution_failed"],
        default="inconclusive",
    )
    frame["error"] = np.where(executable, "", frame["analysis_status"].astype(str))
    # The adaptive generator receives only categorical execution feedback. Raw
    # effects, P values, continuous evidence, and private replication labels
    # remain evaluator-only even after a candidate has been executed.
    return frame.loc[
        :,
        [
            "candidate_id",
            "validated",
            "execution_succeeded",
            "feedback_status",
            "error",
            "executable",
            "analysis_status",
        ],
    ].copy()


def _aggregate_budget_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_columns = [
        "family_fdr_chain_hits",
        "family_fdr_chain_recall",
        "global_fdr_chain_hits",
        "global_fdr_chain_recall",
        "nominal_chain_hits",
        "directional_replication_hits",
        "directional_replication_recall",
        "heldout_chain_evidence_ndcg_at_k",
    ]
    for (method, k), group in frame.groupby(["method", "k"], sort=False):
        row: dict[str, Any] = {
            "method": method,
            "k": int(k),
            "n_trials": int(group["trial"].nunique()),
        }
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_sd"] = float("nan")
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _verify_existing_closed_loop_trial(
    trial_root: Path,
    *,
    trial: int,
    candidate_ids: list[str],
    protocol_freeze_id: str,
    initial_ranking_commit: str,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Recover a completed trial without changing its pre-reveal commits."""

    commit_path = trial_root / "batch_ranking_commits.jsonl"
    trace_path = trial_root / "trace.jsonl"
    overlay_path = trial_root / "experimental_overlay.jsonl"
    for path in (commit_path, trace_path, overlay_path):
        if not path.exists():
            raise FileNotFoundError(f"incomplete closed-loop trial: {path}")

    commits = _load_jsonl(commit_path)
    traces = _load_jsonl(trace_path)
    overlays = _load_jsonl(overlay_path)
    if not commits or len(commits) != len(traces):
        raise ValueError(f"closed-loop commit/trace count mismatch: {trial_root}")

    expected_previous = initial_ranking_commit
    order: list[str] = []
    for index, (commit, trace) in enumerate(zip(commits, traces, strict=True)):
        commit_sha = str(commit.get("commit_sha256", ""))
        hash_payload = {key: value for key, value in commit.items() if key != "commit_sha256"}
        if _canonical_sha(hash_payload) != commit_sha:
            raise ValueError(f"closed-loop commit hash mismatch: {trial_root} batch {index}")
        if commit.get("previous_commit_sha256") != expected_previous:
            raise ValueError(f"closed-loop commit chain mismatch: {trial_root} batch {index}")
        if commit.get("protocol_freeze_id") != protocol_freeze_id:
            raise ValueError(f"closed-loop commit protocol mismatch: {trial_root}")
        if commit.get("outcome_values_in_commit") is not False:
            raise ValueError(f"closed-loop commit contains outcome values: {trial_root}")

        selection = commit.get("selection", {})
        selected = [str(value) for value in selection.get("candidate_ids", [])]
        expected_start = len(order) + 1
        expected_end = len(order) + len(selected)
        if (
            int(selection.get("trial", -1)) != trial
            or int(selection.get("batch", -1)) != index
            or int(selection.get("start_rank", -1)) != expected_start
            or int(selection.get("end_rank", -1)) != expected_end
            or not selected
        ):
            raise ValueError(f"invalid closed-loop batch selection: {trial_root} batch {index}")
        if trace.get("candidate_ids") != selected:
            raise ValueError(f"closed-loop trace differs from commit: {trial_root} batch {index}")
        trace_commit = trace.get("selection_commit", {})
        if (
            trace_commit.get("commit_sha256") != commit_sha
            or int(trace_commit.get("commit_index", -1)) != index + 1
            or trace_commit.get("durably_flushed_before_outcome_lookup") is not True
        ):
            raise ValueError(f"closed-loop trace commit receipt mismatch: {trial_root}")
        order.extend(selected)
        expected_previous = commit_sha

    if len(order) != len(candidate_ids) or set(order) != set(candidate_ids):
        raise ValueError(f"closed-loop commits are not a full candidate permutation: {trial_root}")
    if len(set(order)) != len(order):
        raise ValueError(f"closed-loop commits contain duplicate candidates: {trial_root}")
    if [str(row.get("candidate_id", "")) for row in overlays] != order:
        raise ValueError(f"closed-loop overlay order differs from commits: {trial_root}")

    overlay_previous = "0" * 64
    status_counts: dict[str, int] = {}
    for index, record in enumerate(overlays):
        record_hash = str(record.get("record_hash", ""))
        hash_payload = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("previous_hash") != overlay_previous:
            raise ValueError(f"closed-loop overlay chain mismatch: {trial_root} row {index}")
        if _canonical_sha(hash_payload) != record_hash:
            raise ValueError(f"closed-loop overlay hash mismatch: {trial_root} row {index}")
        provenance = record.get("provenance", {})
        if (
            provenance.get("formal_kg_mutated") is not False
            or provenance.get("outcome_observed_after_selection") is not True
        ):
            raise ValueError(f"invalid closed-loop overlay provenance: {trial_root}")
        status = str(record.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        overlay_previous = record_hash

    commit_manifest = {
        "path": str(commit_path),
        "sha256": _sha256(commit_path),
        "commits": len(commits),
        "final_commit_sha256": expected_previous,
        "trial": trial,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "recovered_from_preexisting_pre_reveal_commits": True,
    }
    overlay_manifest = {
        "trial": trial,
        "path": str(overlay_path),
        "sha256": _sha256(overlay_path),
        "records": len(overlays),
        "records_by_status": dict(sorted(status_counts.items())),
        "final_chain_hash": overlay_previous,
        "rounds": len(commits),
        "recovered_from_preexisting_pre_reveal_commits": True,
    }
    return order, commit_manifest, overlay_manifest


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    locks = verify_pre_outcome_locks(
        args.protocol_root,
        baseline_root=args.baseline_root,
        neurodiscovery_root=args.neurodiscovery_root,
    )
    plan, plan_lock_path = verify_evaluation_plan(args, locks)
    output_root = args.output_root or _default_output_root(locks)
    confirmation_root = args.confirmation_root or (
        Path(locks["run_root"]) / _heldout_output_name(locks["protocol"])
    )
    results, result_lock, result_lock_path = _load_confirmation_results(
        confirmation_root,
        protocol_freeze_id=locks["protocol_manifest"]["freeze_id"],
        protocol=locks["protocol"],
    )
    registry_path = Path(
        locks["protocol_manifest"]["artifacts"]["public_candidate_registry"]
    )
    registry = pd.read_csv(registry_path)
    candidate_ids = registry["candidate_id"].astype(str).tolist()
    if set(results["candidate_id"].astype(str)) != set(candidate_ids):
        raise ValueError("confirmation results do not match the locked registry")
    orders = _load_static_orders(locks["baseline_lock_path"], candidate_ids)

    nd_lock = _load_json(locks["neurodiscovery_lock_path"])
    nd_policy = _load_json(Path(nd_lock["policy_path"]))
    initial = pd.read_csv(Path(nd_lock["initial_ranking_path"])).sort_values(
        "initial_rank", kind="stable"
    )
    public = initial.merge(
        registry,
        on=["candidate_id", "exposure", "modality", "marker", "outcome"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    if len(public) != len(registry) or public["pathway_id"].isna().any():
        raise ValueError("NeuroDiscovery initial order could not reconstruct the registry")
    outcomes = _feedback_outcomes(results, protocol=locks["protocol"])
    config = ClosedLoopConfig(
        **nd_policy["selected_closed_loop_profile"]["config"]
    )
    config.validate()
    adapter = adapter_for("case2_pathway_mediation")
    seed_base = int(plan["neurodiscovery_seed_base"])
    trial_count = int(plan["independent_runs_per_method"])
    nd_trace_root = output_root / "neurodiscovery_closed_loop"
    nd_trace_root.mkdir(parents=True, exist_ok=True)
    commit_manifests: list[dict[str, Any]] = []
    overlay_manifests: list[dict[str, Any]] = []
    for trial in range(trial_count):
        trial_root = nd_trace_root / f"trial_{trial:02d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        commit_path = trial_root / "batch_ranking_commits.jsonl"
        if commit_path.exists() and not args.force:
            order, commit_manifest, overlay_manifest = (
                _verify_existing_closed_loop_trial(
                    trial_root,
                    trial=trial,
                    candidate_ids=candidate_ids,
                    protocol_freeze_id=locks["protocol_manifest"]["freeze_id"],
                    initial_ranking_commit=nd_lock["ranking_commit_sha256"],
                )
            )
            orders[(NEURODISCOVERY, trial)] = order
            commit_manifests.append(commit_manifest)
            overlay_manifests.append(overlay_manifest)
            continue
        if commit_path.exists():
            commit_path.unlink()
        writer = BatchCommitWriter(
            commit_path,
            protocol_freeze_id=locks["protocol_manifest"]["freeze_id"],
            initial_ranking_commit=nd_lock["ranking_commit_sha256"],
        )
        try:
            order, trace, overlay = run_closed_loop_order(
                public,
                outcomes,
                adapter=adapter,
                factor_fields=adapter.factor_fields,
                rng=np.random.default_rng(seed_base + 1009 * trial),
                config=config,
                seed=seed_base,
                trial=trial,
                overlay_path=trial_root / "experimental_overlay.jsonl",
                batch_commit_callback=writer,
            )
        finally:
            commit_manifest = writer.close()
        if commit_manifest["commits"] != overlay["rounds"]:
            raise RuntimeError("not every NeuroDiscovery batch was rank-committed")
        orders[(NEURODISCOVERY, trial)] = (
            public.iloc[order]["candidate_id"].astype(str).tolist()
        )
        trace_path = trial_root / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in trace:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        commit_manifest.update(
            {
                "trial": trial,
                "trace_path": str(trace_path),
                "trace_sha256": _sha256(trace_path),
            }
        )
        commit_manifests.append(commit_manifest)
        overlay_manifests.append({"trial": trial, **overlay})

    trial_summary, budget_metrics, rankings = evaluate_orders(
        orders,
        results,
        budgets=plan["budgets"],
        recall_targets=plan["recall_targets"],
    )
    aggregate = aggregate_trials(trial_summary)
    budget_aggregate = _aggregate_budget_metrics(budget_metrics)
    comparison_config = plan["comparison"]
    primary_metrics = _primary_metric_names(plan)
    primary_comparison_frames: list[pd.DataFrame] = []
    for metric_index, metric in enumerate(primary_metrics):
        primary_comparison_frames.append(
            paired_primary_comparisons(
                trial_summary,
                metric=metric,
                target_method=NEURODISCOVERY,
                baseline_methods=BASELINE_METHODS,
                bootstrap_resamples=int(comparison_config["bootstrap_resamples"]),
                bootstrap_seed=(
                    int(comparison_config["bootstrap_seed"])
                    + 100_000 * metric_index
                ),
            )
        )
    primary_comparisons = pd.concat(
        primary_comparison_frames, ignore_index=True
    )
    budget_comparisons: list[pd.DataFrame] = []
    budget_comparison_metric = (
        "directional_replication_hits"
        if _holdout_axis(locks["protocol"]) == "cohort_phase"
        else "family_fdr_chain_hits"
    )
    for k in plan["budgets"]:
        subset = budget_metrics.loc[budget_metrics["k"].eq(int(k))].copy()
        compared = paired_primary_comparisons(
            subset,
            metric=budget_comparison_metric,
            target_method=NEURODISCOVERY,
            baseline_methods=BASELINE_METHODS,
            bootstrap_resamples=int(comparison_config["bootstrap_resamples"]),
            bootstrap_seed=int(comparison_config["bootstrap_seed"]) + int(k),
        )
        compared.insert(1, "k", int(k))
        budget_comparisons.append(compared)
    secondary_comparisons = pd.concat(budget_comparisons, ignore_index=True)

    metric_verdicts: dict[str, dict[str, Any]] = {}
    for metric in primary_metrics:
        mean_column = f"{metric}_mean"
        target_rows = aggregate.loc[
            aggregate["method"].eq(NEURODISCOVERY), mean_column
        ]
        target_mean = (
            _finite_or_none(target_rows.iloc[0]) if len(target_rows) else None
        )
        baseline_values = pd.to_numeric(
            aggregate.loc[
                aggregate["method"].isin(BASELINE_METHODS), mean_column
            ],
            errors="coerce",
        )
        finite_baselines = baseline_values[np.isfinite(baseline_values)]
        best_baseline_mean = (
            float(finite_baselines.max())
            if len(finite_baselines) == len(BASELINE_METHODS)
            else None
        )
        comparisons = primary_comparisons.loc[
            primary_comparisons["metric"].eq(metric)
        ]
        evaluable = bool(
            target_mean is not None
            and best_baseline_mean is not None
            and len(comparisons) == len(BASELINE_METHODS)
            and comparisons["n_finite_pairs"].eq(
                int(plan["independent_runs_per_method"])
            ).all()
        )
        pairwise_passed = bool(
            evaluable and comparisons["target_superior"].all()
        )
        metric_verdicts[metric] = {
            "evaluable": evaluable,
            "neurodiscovery_mean": target_mean,
            "best_baseline_mean": best_baseline_mean,
            "neurodiscovery_has_highest_mean": bool(
                evaluable and target_mean > best_baseline_mean
            ),
            "all_six_pairwise_superiority_tests_passed": pairwise_passed,
            "passes_clear_sota_component": bool(
                evaluable and target_mean > best_baseline_mean and pairwise_passed
            ),
        }

    replication_positive_count = int(
        results.get(
            "directional_replication_hit",
            pd.Series(False, index=results.index),
        ).sum()
    )
    family_positive_count = int(results["family_fdr_chain_hit"].sum())
    sota_evaluable = bool(
        replication_positive_count > 0
        and all(item["evaluable"] for item in metric_verdicts.values())
    )
    clear_sota = bool(
        sota_evaluable
        and all(
            item["passes_clear_sota_component"]
            for item in metric_verdicts.values()
        )
    )
    verdict_status = (
        "clear_sota"
        if clear_sota
        else (
            "not_clear_sota"
            if sota_evaluable
            else (
                "not_identifiable_no_directional_replication_labels"
                if replication_positive_count == 0
                else "not_identifiable_primary_metric"
            )
        )
    )
    verdict = {
        "clear_sota": clear_sota,
        "sota_evaluable": sota_evaluable,
        "verdict_status": verdict_status,
        "rule": plan["clear_sota_rule"],
        "directional_replication_positive_count": replication_positive_count,
        "family_fdr_positive_count": family_positive_count,
        "global_fdr_positive_count": int(results["global_fdr_chain_hit"].sum()),
        "primary_metric_verdicts": metric_verdicts,
        "interpretation": (
            "No frozen ADNI3-selected chain passed independent directional Holm "
            "replication in ADNI1/GO/2, so the replication-recovery metric and a "
            "clear-SOTA claim are not identifiable. Thresholds were not relaxed."
            if replication_positive_count == 0
            else "Both prespecified primary generator metrics were evaluated."
        ),
    }

    paths = {
        "trial_summary": output_root / "case2_generator_trial_summary.csv",
        "aggregate": output_root / "case2_generator_performance_table.csv",
        "budget_metrics": output_root / "case2_generator_metrics_by_budget.csv",
        "budget_aggregate": output_root / "case2_generator_budget_aggregate.csv",
        "primary_comparisons": output_root / "case2_primary_paired_comparisons.csv",
        "secondary_comparisons": output_root / "case2_budget_paired_comparisons.csv",
        "rankings": output_root / "case2_generator_rankings.parquet",
        "verdict": output_root / "case2_sota_verdict.json",
    }
    trial_summary.to_csv(paths["trial_summary"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    budget_metrics.to_csv(paths["budget_metrics"], index=False)
    budget_aggregate.to_csv(paths["budget_aggregate"], index=False)
    primary_comparisons.to_csv(paths["primary_comparisons"], index=False)
    secondary_comparisons.to_csv(paths["secondary_comparisons"], index=False)
    rankings.to_parquet(paths["rankings"], index=False, compression="zstd")
    paths["verdict"].write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "neurooracle.case2_generator_evaluation.v1",
        "protocol_freeze_id": locks["protocol_manifest"]["freeze_id"],
        "confirmation_result_lock_sha256": _sha256(result_lock_path),
        "evaluation_plan_lock_sha256": _sha256(plan_lock_path),
        "methods": sorted(trial_summary["method"].unique()),
        "trials_per_method": trial_summary.groupby("method")["trial"].nunique().to_dict(),
        "candidate_count": len(results),
        "primary_metrics": primary_metrics,
        "verdict": verdict,
        "neurodiscovery_batch_commit_manifests": commit_manifests,
        "neurodiscovery_overlay_manifests": overlay_manifests,
        "freeze_semantics": {
            "temporal_kg_freeze": False,
            "temporal_freeze_year": None,
            "kg_snapshot_pinning": True,
            "ranking_freeze": True,
            "batchwise_reveal_with_pre_reveal_commit": True,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["manifest_sha256"] = _sha256(manifest_path)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--plan-config", type=Path, default=DEFAULT_PLAN_CONFIG)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--neurodiscovery-root", type=Path)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--freeze-plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.freeze_plan_only:
        print(json.dumps(freeze_evaluation_plan(args), indent=2, ensure_ascii=False))
    else:
        run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 13:58 HKT
