"""Independently verify a completed formal dynamic generalization benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.scripts.compare_dynamic_closed_open_loop import (
    _audit_design,
)
from neurooracle.scripts.run_formal_dynamic_generalization import (
    POLICY_METADATA_FIELDS,
    STAGES,
    _normalized_arm_command,
)
from neurooracle.scripts.summarize_hindcasting_replicates import (
    exact_sign_flip_p_value,
)
from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_source_bundle,
)
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DYNAMIC_SCHEMA = "neurodiscovery-dynamic-closed-loop-hindcasting.v4"
DEFAULT_DESIGN = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
    / "formal_dynamic_generalization_design_20260813.json"
)
METRIC_FIELDS = (
    "executed_hypotheses",
    "generation_failure_slots",
    "full_primary_hits",
    "unique_primary_discoveries",
    "recovered_future_pairs",
    "future_pair_recall",
    "full_any_hits",
    "early_primary_hits",
    "early_unique_primary_discoveries",
    "terminal_primary_hits",
    "terminal_unique_primary_discoveries",
    "terminal_any_hits",
)
EXPECTED_GENERATOR_CLOSURE_NAMES = frozenset(
    {
        "branch_engines",
        "case",
        "config",
        "graph",
        "kge_checkpoint",
        "kge_scorer",
        "profile",
        "rounds_dir",
        "seed",
        "window",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(value: Any, root: Path) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _run_key(row: Mapping[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(row["seed"]),
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _metric_key(row: Mapping[str, Any]) -> tuple[int, str, int, int, int, int]:
    return (*_run_key(row), int(row["requested_k"]))


def _expected_metric_rows(
    hidden: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    future_pair_total: int,
) -> dict[int, dict[str, float | int]]:
    expected: dict[int, dict[str, float | int]] = {}
    for requested in budgets:
        selected = hidden[: min(int(requested), len(hidden))]
        unique_primary = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if _bool(row.get("primary_hit")) and row.get("primary_discovery_key")
        }
        early_unique = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if _bool(row.get("early_primary_hit")) and row.get("primary_discovery_key")
        }
        terminal_unique = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if _bool(row.get("terminal_primary_hit")) and row.get("primary_discovery_key")
        }
        recovered = {
            pair
            for row in selected
            if _bool(row.get("primary_hit"))
            for pair in str(row.get("primary_recovered_pairs") or "").split(";")
            if pair
        }
        expected[int(requested)] = {
            "executed_hypotheses": len(selected),
            "generation_failure_slots": sum(
                _bool(row.get("generation_failure")) for row in selected
            ),
            "full_primary_hits": sum(_bool(row.get("primary_hit")) for row in selected),
            "unique_primary_discoveries": len(unique_primary),
            "recovered_future_pairs": len(recovered),
            "future_pair_recall": (
                len(recovered) / future_pair_total if future_pair_total > 0 else 0.0
            ),
            "full_any_hits": sum(_bool(row.get("any_future_hit")) for row in selected),
            "early_primary_hits": sum(
                _bool(row.get("early_primary_hit")) for row in selected
            ),
            "early_unique_primary_discoveries": len(early_unique),
            "terminal_primary_hits": sum(
                _bool(row.get("terminal_primary_hit")) for row in selected
            ),
            "terminal_unique_primary_discoveries": len(terminal_unique),
            "terminal_any_hits": sum(
                _bool(row.get("terminal_any_hit")) for row in selected
            ),
        }
    return expected


def _is_generation_failure(hypothesis: Mapping[str, Any]) -> bool:
    return (
        hypothesis.get("hypothesis_type") == "generation_failure"
        or _bool((hypothesis.get("metadata") or {}).get("generation_failure"))
    )


def _audit_fixed_budget_slots(
    hypotheses: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
) -> int:
    """Verify that every padded slot is explicit, ordered, and zero-credit."""

    failures = 0
    failure_started = False
    for expected_rank, (hypothesis, outcome) in enumerate(
        zip(hypotheses, hidden, strict=True),
        start=1,
    ):
        is_failure = _is_generation_failure(hypothesis)
        hidden_failure = _bool(outcome.get("generation_failure"))
        if hidden_failure != is_failure:
            raise ValueError("executed and hidden generation-failure flags differ")
        if not is_failure:
            if failure_started:
                raise ValueError("valid hypothesis appears after a fixed-budget failure slot")
            continue
        failure_started = True
        failures += 1
        metadata = hypothesis.get("metadata") or {}
        if (
            hypothesis.get("path")
            or str(hypothesis.get("source_id") or "")
            or str(hypothesis.get("target_id") or "")
            or not _bool(metadata.get("fixed_budget_slot"))
            or int(metadata.get("execution_rank") or -1) != expected_rank
            or int(outcome.get("execution_rank") or -1) != expected_rank
            or str(outcome.get("feedback_status") or "") != "generation_failure"
        ):
            raise ValueError("malformed fixed-budget generation-failure slot")
        if any(
            _bool(outcome.get(field))
            for field in (
                "primary_hit",
                "any_future_hit",
                "early_primary_hit",
                "terminal_primary_hit",
                "terminal_any_hit",
            )
        ):
            raise ValueError("fixed-budget generation-failure slot received hit credit")
        if any(
            str(outcome.get(field) or "")
            for field in (
                "primary_year",
                "primary_discovery_key",
                "primary_recovered_pairs",
                "first_future_year",
            )
        ):
            raise ValueError("fixed-budget generation-failure slot has outcome evidence")
    return failures


def _audit_hypothesis_temporal_fields(
    path: Path,
    *,
    freeze_year: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as raw:
        digest.update(raw.read())
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            stack: list[Any] = [row]
            while stack:
                value = stack.pop()
                if isinstance(value, Mapping):
                    for key, nested in value.items():
                        normalized = str(key).lower()
                        if normalized in {
                            "uses_future_outcomes",
                            "frontier_uses_future_outcomes",
                            "terminal_labels_available_to_generator",
                        } and _bool(nested):
                            raise ValueError(f"future outcome flag is true in {path}")
                        if normalized in {"year", "publication_year", "pub_year"}:
                            try:
                                observed_year = int(nested)
                            except (TypeError, ValueError):
                                pass
                            else:
                                if observed_year > freeze_year:
                                    raise ValueError(
                                        f"future evidence year {observed_year} in {path}"
                                    )
                        stack.append(nested)
                elif isinstance(value, list):
                    stack.extend(value)
    return count, digest.hexdigest().upper()


def _audit_execution_binding(
    execution: Mapping[str, Any],
    *,
    design_path: Path,
    bundle_manifest: Path,
) -> None:
    """Require the execution record to bind the exact frozen design and bundle."""

    design_path = design_path.resolve()
    bundle_manifest = bundle_manifest.resolve()
    if Path(str(execution.get("design_path") or "")).resolve() != design_path:
        raise ValueError("execution used a different frozen design")
    if str(execution.get("design_sha256") or "").upper() != sha256_file(design_path):
        raise ValueError("execution frozen-design hash mismatch")
    if (
        Path(str(execution.get("source_bundle_manifest") or "")).resolve()
        != bundle_manifest
    ):
        raise ValueError("execution used a different source bundle")
    if str(execution.get("source_bundle_manifest_sha256") or "").upper() != sha256_file(
        bundle_manifest
    ):
        raise ValueError("execution source-bundle hash mismatch")


def _split_runtime_contract(
    runtime_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Separate the profile manifest contract from dataclass runtime fields."""

    config = dict(runtime_config)
    profile = str(config.pop("profile", "")).strip()
    if not profile:
        raise ValueError("dynamic runtime contract lacks a profile")
    return profile, config


def _verify_execution_manifest(
    output_root: Path,
    *,
    design_path: Path,
    bundle_manifest: Path,
    rehash_references: bool,
) -> dict[str, Any]:
    execution = _read_json(output_root / "execution_manifest.json")
    if execution.get("status") != "complete" or execution.get("mode") != "formal":
        raise ValueError("formal dynamic execution is incomplete or not formal")
    records = execution.get("records") or []
    if [row.get("stage") for row in records] != list(STAGES):
        raise ValueError("formal dynamic stages are incomplete or out of order")
    _audit_execution_binding(
        execution,
        design_path=design_path,
        bundle_manifest=bundle_manifest,
    )
    for record in records:
        if int(record.get("returncode", -1)) != 0:
            raise ValueError(f"stage did not exit successfully: {record.get('stage')}")
        for phase in ("bundle_before", "bundle_after"):
            guard = record.get(phase) or {}
            if guard.get("status") != "passed":
                raise ValueError(f"stage bundle guard failed: {record.get('stage')} {phase}")
            if not guard.get("archived_source_verified"):
                raise ValueError("stage did not verify archived source")
            if not guard.get("referenced_inputs_verified"):
                raise ValueError("formal dynamic stage did not rehash references")
    closed_command = records[0].get("command") or []
    open_command = records[1].get("command") or []
    if _normalized_arm_command(closed_command) != _normalized_arm_command(open_command):
        raise ValueError("paired execution commands differ beyond feedback and output")
    if "--feedback-enabled" not in closed_command or "--no-feedback-enabled" not in open_command:
        raise ValueError("paired execution commands use wrong feedback flags")
    if rehash_references:
        verify_source_bundle(
            bundle_manifest,
            require_live_source=False,
            verify_references=True,
        )
    return execution


def _verify_arm(
    root: Path,
    *,
    arm: str,
    expected_runs: set[tuple[int, str, int, int, int]],
    budgets: Sequence[int],
    runtime_config: Mapping[str, Any],
    expected_profile: str,
    eligibility_manifest: Path,
) -> dict[str, Any]:
    top = _read_json(root / "dynamic_closed_loop_manifest.json")
    if top.get("schema_version") != EXPECTED_DYNAMIC_SCHEMA:
        raise ValueError(f"{arm} top-level dynamic schema is incompatible")
    expected_feedback = arm == "closed"
    if str((top.get("profile") or {}).get("name") or "") != expected_profile:
        raise ValueError(f"{arm} top-level profile differs from design")
    config = dict(top.get("config") or {})
    observed_feedback = config.pop("feedback_enabled", None)
    if observed_feedback is not expected_feedback or config != dict(runtime_config):
        raise ValueError(f"{arm} runtime configuration differs from design")
    if [int(value) for value in top.get("budgets") or ()] != list(budgets):
        raise ValueError(f"{arm} rank points differ from design")
    eligibility = top.get("eligibility") or {}
    if Path(str(eligibility.get("manifest_path") or "")).resolve() != eligibility_manifest:
        raise ValueError(f"{arm} used a different eligibility manifest")
    if set(_run_key(row) for row in top.get("run_summaries") or ()) != expected_runs:
        raise ValueError(f"{arm} run matrix differs from design")
    if int(top.get("runs") or 0) != len(expected_runs):
        raise ValueError(f"{arm} run count differs from design")

    top_metrics = {_metric_key(row): row for row in _read_csv(root / "metrics_by_run.csv")}
    if len(top_metrics) != len(expected_runs) * len(budgets):
        raise ValueError(f"{arm} top-level metric matrix is incomplete")
    artifact_hashes: list[str] = []
    proposed_total = 0
    for run in top.get("run_summaries") or ():
        key = _run_key(run)
        seed, case_id, freeze, start, end = key
        run_dir = root / f"seed_{seed:02d}" / case_id / f"kg{freeze}_to_{start}_{end}"
        on_disk = _read_json(run_dir / "run_manifest.json")
        if on_disk.get("schema_version") != EXPECTED_DYNAMIC_SCHEMA:
            raise ValueError(f"{arm} run schema is incompatible: {key}")
        if _run_key(on_disk) != key or on_disk.get("status") != "complete":
            raise ValueError(f"{arm} run manifest is inconsistent: {key}")
        if str((on_disk.get("profile") or {}).get("name") or "") != expected_profile:
            raise ValueError(f"{arm} run profile differs from design: {key}")
        run_config = dict(on_disk.get("config") or {})
        run_feedback = run_config.pop("feedback_enabled", None)
        if run_feedback is not expected_feedback or run_config != dict(runtime_config):
            raise ValueError(f"{arm} paired run configuration drift: {key}")
        temporal = on_disk.get("temporal_isolation") or {}
        if temporal.get("terminal_labels_available_to_generator") is not False:
            raise ValueError(f"{arm} terminal labels exposed: {key}")
        if temporal.get("generator_closure_audit_passed") is not True:
            raise ValueError(f"{arm} generator closure was not audited: {key}")
        closure_names = {
            str(value) for value in temporal.get("generator_closure_freevars") or ()
        }
        declared_forbidden = list(
            temporal.get("generator_closure_forbidden_freevars") or ()
        )
        if closure_names != EXPECTED_GENERATOR_CLOSURE_NAMES or declared_forbidden:
            raise ValueError(f"{arm} generator captured hidden future outcomes: {key}")
        if temporal.get("formal_kg_mutated") is not False:
            raise ValueError(f"{arm} formal KG was mutated: {key}")
        if int(on_disk.get("feedback_end_year") or -1) != freeze + int(
            runtime_config["feedback_years"]
        ):
            raise ValueError(f"{arm} feedback boundary differs from design: {key}")

        executed = _read_json(run_dir / "executed_hypotheses.json")
        hypotheses = executed.get("hypotheses") or []
        expected_executions = int(runtime_config["max_executions"])
        if len(hypotheses) != expected_executions or int(
            executed.get("n_hypotheses") or 0
        ) != expected_executions:
            raise ValueError(f"{arm} fixed execution budget violated: {key}")
        identifiers = [str(row.get("id") or "") for row in hypotheses]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{arm} executed hypotheses contain missing/duplicate IDs: {key}")

        hidden = _read_csv(run_dir / "hidden_outcomes.csv")
        hidden_ids = [str(row.get("candidate_id") or "") for row in hidden]
        if hidden_ids != identifiers:
            raise ValueError(f"{arm} hidden outcomes do not preserve execution order: {key}")
        failure_slots = _audit_fixed_budget_slots(hypotheses, hidden)
        if failure_slots != int(on_disk.get("generation_failure_slots") or 0):
            raise ValueError(f"{arm} generation-failure count mismatch: {key}")
        expected_rows = _expected_metric_rows(
            hidden,
            budgets=budgets,
            future_pair_total=int((on_disk.get("future_stats") or {}).get("future_unique_pairs") or 0),
        )
        per_run = {_metric_key(row): row for row in _read_csv(run_dir / "metrics_by_k.csv")}
        for requested, expected in expected_rows.items():
            metric_key = (*key, requested)
            for source_name, source in (("run", per_run), ("top", top_metrics)):
                observed = source.get(metric_key)
                if observed is None:
                    raise ValueError(f"{arm} {source_name} metrics missing {metric_key}")
                for field, expected_value in expected.items():
                    if not _close(observed.get(field), expected_value):
                        raise ValueError(
                            f"{arm} {source_name} metric mismatch {metric_key} {field}"
                        )

        feedback = _read_csv(run_dir / "feedback_overlay.csv")
        supported_available = sum(
            row.get("status") == "supported" and _bool(row.get("available_to_generator"))
            for row in feedback
        )
        withheld_supported = sum(
            row.get("status") == "supported" and not _bool(row.get("available_to_generator"))
            for row in feedback
        )
        if expected_feedback:
            if any(not _bool(row.get("available_to_generator")) for row in feedback):
                raise ValueError(f"closed arm withheld observed feedback: {key}")
            if withheld_supported:
                raise ValueError(f"closed arm withheld supported feedback: {key}")
        else:
            if any(_bool(row.get("available_to_generator")) for row in feedback):
                raise ValueError(f"open arm exposed feedback: {key}")
            if supported_available:
                raise ValueError(f"open arm exposed supported feedback: {key}")
        if supported_available != int(on_disk.get("supported_feedback_records") or 0):
            raise ValueError(f"{arm} supported feedback count mismatch: {key}")
        if withheld_supported != int(on_disk.get("withheld_supported_outcomes") or 0):
            raise ValueError(f"{arm} withheld feedback count mismatch: {key}")

        rounds = _read_csv(run_dir / "generation_rounds.csv")
        if not expected_feedback and any(_bool(row.get("feedback_active")) for row in rounds):
            raise ValueError(f"open arm activated feedback: {key}")
        proposed_count, proposed_hash = _audit_hypothesis_temporal_fields(
            run_dir / "proposed_hypotheses.jsonl.gz",
            freeze_year=freeze,
        )
        if proposed_count != int(on_disk.get("unique_proposals") or 0):
            raise ValueError(f"{arm} proposed hypothesis count mismatch: {key}")
        proposed_total += proposed_count
        artifact_hashes.extend(
            [
                proposed_hash,
                sha256_file(run_dir / "executed_hypotheses.json"),
                sha256_file(run_dir / "hidden_outcomes.csv"),
                sha256_file(run_dir / "feedback_overlay.csv"),
                sha256_file(run_dir / "generation_rounds.csv"),
                sha256_file(run_dir / "metrics_by_k.csv"),
                sha256_file(run_dir / "run_manifest.json"),
            ]
        )
    digest = hashlib.sha256("\n".join(sorted(artifact_hashes)).encode("ascii")).hexdigest().upper()
    return {
        "runs": len(expected_runs),
        "metric_rows": len(top_metrics),
        "proposed_hypotheses": proposed_total,
        "artifact_set_sha256": digest,
    }


def _aggregate_from_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    requested_k: int,
) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if int(row["requested_k"]) != requested_k:
            continue
        key = (int(row["seed"]), str(row["case_study_id"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: fmean(values) for key, values in grouped.items()}


def _verify_primary_endpoint(
    *,
    design: Mapping[str, Any],
    closed_root: Path,
    open_root: Path,
    comparison_root: Path,
) -> dict[str, Any]:
    primary = design["endpoints"]["primary"]
    metric = str(primary["metric"])
    requested_k = int(primary["k"])
    closed_rows = _read_csv(closed_root / "metrics_by_run.csv")
    open_rows = _read_csv(open_root / "metrics_by_run.csv")
    closed_case = _aggregate_from_metrics(closed_rows, metric=metric, requested_k=requested_k)
    open_case = _aggregate_from_metrics(open_rows, metric=metric, requested_k=requested_k)
    if set(closed_case) != set(open_case):
        raise ValueError("primary endpoint case/seed matrix is unpaired")
    seeds = sorted({key[0] for key in closed_case})
    case_ids = list(design["primary_matrix"]["case_study_ids"])
    closed = [fmean(closed_case[(seed, case)] for case in case_ids) for seed in seeds]
    opened = [fmean(open_case[(seed, case)] for case in case_ids) for seed in seeds]
    differences = [left - right for left, right in zip(closed, opened, strict=True)]
    expected = {
        "closed_mean": fmean(closed),
        "closed_sample_variance": variance(closed) if len(closed) > 1 else 0.0,
        "open_mean": fmean(opened),
        "open_sample_variance": variance(opened) if len(opened) > 1 else 0.0,
        "mean_paired_difference": fmean(differences),
        "paired_difference_sample_variance": (
            variance(differences) if len(differences) > 1 else 0.0
        ),
        "p_closed_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
        "closed_wins": sum(value > 0 for value in differences),
        "ties": sum(abs(value) <= 1e-15 for value in differences),
        "closed_losses": sum(value < 0 for value in differences),
    }
    rows = _read_csv(comparison_root / "mean_variance_paired_summary.csv")
    observed = next(
        (
            row
            for row in rows
            if row["scope"] == "macro_case_study_equal"
            and int(row["requested_k"]) == requested_k
            and row["metric"] == metric
        ),
        None,
    )
    if observed is None:
        raise ValueError("comparison summary lacks the registered primary endpoint")
    for field, value in expected.items():
        if not _close(observed.get(field), value):
            raise ValueError(f"primary endpoint summary mismatch: {field}")
    return {"metric": metric, "k": requested_k, "n_pairs": len(seeds), **expected}


def verify(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    output_root = args.output_root.resolve()
    design_path = args.design.resolve()
    design = _read_json(design_path)
    if design.get("status") != "frozen_before_dynamic_generalization":
        raise ValueError("dynamic generalization design is not frozen")
    bundle_manifest = _resolve(
        design["reproducibility"]["source_bundle_manifest"], workspace_root
    )
    execution = _verify_execution_manifest(
        output_root,
        design_path=design_path,
        bundle_manifest=bundle_manifest,
        rehash_references=args.rehash_references,
    )
    policy = _read_json(_resolve(design["frozen_policy"]["path"], workspace_root))
    runtime_config = dict(policy["selected_policy"])
    for field in POLICY_METADATA_FIELDS:
        runtime_config.pop(field, None)
    if runtime_config != dict(design["shared_configuration"]):
        raise ValueError("design runtime configuration differs from frozen policy")
    expected_profile, arm_runtime_config = _split_runtime_contract(runtime_config)
    eligibility_manifest = _resolve(
        design["dynamic_eligibility"]["manifest"]["path"], workspace_root
    )
    eligibility = load_locked_hindcasting_eligibility(eligibility_manifest)
    seeds = [int(value) for value in design["primary_matrix"]["seeds"]]
    expected_runs = {
        (seed, case, freeze, start, end)
        for seed in seeds
        for case, freeze, start, end in eligibility.primary_windows
    }
    budgets = [int(value) for value in design["fixed_budget_contract"]["rank_points"]]
    closed_root = output_root / "closed"
    open_root = output_root / "open"
    closed = _verify_arm(
        closed_root,
        arm="closed",
        expected_runs=expected_runs,
        budgets=budgets,
        runtime_config=arm_runtime_config,
        expected_profile=expected_profile,
        eligibility_manifest=eligibility_manifest,
    )
    opened = _verify_arm(
        open_root,
        arm="open",
        expected_runs=expected_runs,
        budgets=budgets,
        runtime_config=arm_runtime_config,
        expected_profile=expected_profile,
        eligibility_manifest=eligibility_manifest,
    )
    independent_design_audit = _audit_design(closed_root, open_root)
    recorded_design_audit = _read_json(output_root / "comparison" / "design_audit.json")
    for field in (
        "paired_runs",
        "non_feedback_configuration_identical",
        "terminal_labels_isolated",
        "formal_kg_unmodified",
        "open_feedback_records_exposed",
        "open_feedback_activated_runs",
        "open_withheld_supported_outcomes",
        "closed_supported_feedback_records",
        "closed_feedback_activated_runs",
        "warmup_execution_prefix_identical",
        "warmup_prefix_hypotheses_compared",
    ):
        if independent_design_audit.get(field) != recorded_design_audit.get(field):
            raise ValueError(f"recorded dynamic design audit mismatch: {field}")
    primary = _verify_primary_endpoint(
        design=design,
        closed_root=closed_root,
        open_root=open_root,
        comparison_root=output_root / "comparison",
    )
    result = {
        "schema_version": "formal-dynamic-generalization-verification.v1",
        "verified_at": _utc_now(),
        "status": "passed",
        "design_path": str(design_path),
        "design_sha256": sha256_file(design_path),
        "source_bundle_manifest": str(bundle_manifest),
        "source_bundle_manifest_sha256": sha256_file(bundle_manifest),
        "execution_manifest_sha256": sha256_file(
            output_root / "execution_manifest.json"
        ),
        "execution_records": len(execution.get("records") or ()),
        "primary_case_studies": len(eligibility.primary_case_study_ids),
        "primary_case_windows": len(eligibility.primary_windows),
        "seeds": len(seeds),
        "paired_runs": len(expected_runs),
        "rank_points": budgets,
        "closed": closed,
        "open": opened,
        "design_audit": independent_design_audit,
        "primary_endpoint": primary,
        "formal_kg_mutated": False,
        "future_outcomes_used_for_task_or_window_selection": False,
        "historical_exposure_acknowledged": True,
        "claim_as_untouched_confirmatory_permitted": False,
    }
    _atomic_json(output_root / "verification" / "verification_manifest.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--rehash-references", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(verify(parse_args()), indent=2, ensure_ascii=False))
