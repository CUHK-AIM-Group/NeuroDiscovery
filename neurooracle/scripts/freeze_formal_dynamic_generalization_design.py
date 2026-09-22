"""Freeze the formal cross-task dynamic closed-loop benchmark design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.scripts.run_formal_dynamic_generalization import (
    POLICY_METADATA_FIELDS,
)
from neurooracle.src.experiment_source_bundle import sha256_file
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
)
DEFAULT_STATIC_DESIGN = (
    PROTOCOL_ROOT / "formal_static_comparison_design_endpoint_v7_20260813.json"
)
DEFAULT_POLICY = PROTOCOL_ROOT / "frozen_dynamic_policy.json"
DEFAULT_ELIGIBILITY = (
    PROTOCOL_ROOT
    / "locked_dynamic_generalization_eligibility_feedback1_endpoint_v7_20260813"
    / "dynamic_generalization_manifest.json"
)
DEFAULT_OUTPUT = PROTOCOL_ROOT / "formal_dynamic_generalization_design_20260813.json"
DEFAULT_BUNDLE = (
    ROOT
    / "neurooracle/.frozen/formal_dynamic_generalization_v1/bundle_manifest.json"
)
DEFAULT_RESULT_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_dynamic_generalization_20260813"
)
DEFAULT_BUDGETS = (10, 20, 50, 100, 200, 300)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _file_record(path: Path, workspace_root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _repo_path(path, workspace_root),
        "sha256": sha256_file(path),
    }


def _verify_record(record: Mapping[str, Any], workspace_root: Path) -> Path:
    path = Path(str(record.get("path") or ""))
    path = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    expected = str(record.get("sha256") or "").upper()
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"frozen input hash mismatch: {path}")
    return path


def _policy_runtime_configuration(policy: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(policy.get("selected_policy") or {})
    if selected.get("feedback_enabled") is not True:
        raise ValueError("selected dynamic policy is not the closed-loop policy")
    missing = POLICY_METADATA_FIELDS - set(selected)
    if missing:
        raise ValueError(f"selected dynamic policy is incomplete: {sorted(missing)}")
    for field in POLICY_METADATA_FIELDS:
        selected.pop(field, None)
    return selected


def _policy_canonical_hashes(policy: Mapping[str, Any]) -> dict[str, str]:
    canonical = policy.get("canonical_release") or {}
    return {
        "knowledge_graph": str(canonical.get("knowledge_graph_sha256") or "").upper(),
        "extracted_claims": str(canonical.get("claims_sha256") or "").upper(),
        "current_state": str(canonical.get("state_sha256") or "").upper(),
    }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _verify_policy_application(
    policy: Mapping[str, Any],
    *,
    canonical_hashes: Mapping[str, str],
    workspace_root: Path,
) -> dict[str, Any] | None:
    application = policy.get("policy_application")
    if application is None:
        if _policy_canonical_hashes(policy) != dict(canonical_hashes):
            raise ValueError("dynamic policy and formal static design use different KG bytes")
        return None
    application = dict(application)
    if application.get("mode") != "unchanged_cross_release_application":
        raise ValueError("unsupported dynamic policy application mode")
    source_path = _verify_record(application["source_policy"], workspace_root)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_for_confirmatory_application":
        raise ValueError("source dynamic policy is not frozen")
    if dict(source.get("selected_policy") or {}) != dict(
        policy.get("selected_policy") or {}
    ):
        raise ValueError("cross-release application changed the selected policy")
    if application.get("source_selected_policy_sha256") != _canonical_json_sha256(
        source.get("selected_policy") or {}
    ):
        raise ValueError("source selected-policy hash mismatch")
    if dict(application.get("selection_canonical_release") or {}) != dict(
        source.get("canonical_release") or {}
    ):
        raise ValueError("selection release differs from the source policy")
    if dict(application.get("application_canonical_release") or {}) != dict(
        policy.get("canonical_release") or {}
    ):
        raise ValueError("application release differs from the policy binding")
    if _policy_canonical_hashes(policy) != dict(canonical_hashes):
        raise ValueError("policy application and formal static design use different KG bytes")
    required_false = (
        "new_release_outcomes_used_for_policy_selection",
        "post_update_retuning",
    )
    if application.get("frozen_runtime_unchanged") is not True or any(
        application.get(field) is not False for field in required_false
    ):
        raise ValueError("cross-release policy application permits outcome-guided retuning")
    original_ids = [
        str(value)
        for value in (
            (source.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    ]
    additional_ids = [
        str(value)
        for value in application.get("additional_exposed_case_study_ids") or ()
    ]
    all_ids = list(dict.fromkeys([*original_ids, *additional_ids]))
    if application.get("original_development_case_study_ids") != original_ids:
        raise ValueError("policy application changed the original development set")
    if application.get("all_excluded_case_study_ids") != all_ids:
        raise ValueError("policy application has an inconsistent exposure union")
    policy_ids = [
        str(value)
        for value in (
            (policy.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    ]
    if policy_ids != all_ids:
        raise ValueError("policy development set does not include every exposed task")
    return application


def freeze_formal_dynamic_design(
    *,
    static_design_path: Path,
    frozen_policy_path: Path,
    eligibility_manifest_path: Path,
    source_bundle_manifest: Path,
    output_path: Path,
    result_root: Path,
    workspace_root: Path,
    seeds: Sequence[int] = tuple(range(10)),
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Create or byte-verify one immutable formal dynamic design."""

    workspace_root = workspace_root.resolve()
    static_design_path = static_design_path.resolve()
    frozen_policy_path = frozen_policy_path.resolve()
    eligibility_manifest_path = eligibility_manifest_path.resolve()
    output_path = output_path.resolve()
    source_bundle_manifest = source_bundle_manifest.resolve()
    result_root = result_root.resolve()

    static = json.loads(static_design_path.read_text(encoding="utf-8"))
    if static.get("status") != "frozen_before_formal_generation":
        raise ValueError("source static design is not frozen")
    canonical = dict(static.get("canonical_release") or {})
    canonical_hashes: dict[str, str] = {}
    for label in ("knowledge_graph", "extracted_claims", "current_state"):
        _verify_record(canonical[label], workspace_root)
        canonical_hashes[label] = str(canonical[label]["sha256"]).upper()

    policy = json.loads(frozen_policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_for_confirmatory_application":
        raise ValueError("dynamic policy is not frozen")
    policy_application = _verify_policy_application(
        policy,
        canonical_hashes=canonical_hashes,
        workspace_root=workspace_root,
    )
    runtime_config = _policy_runtime_configuration(policy)
    development_ids = [
        str(value)
        for value in (
            (policy.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    ]
    if not development_ids:
        raise ValueError("dynamic policy does not declare development Case Studies")

    eligibility = load_locked_hindcasting_eligibility(eligibility_manifest_path)
    if eligibility.manifest.get("status") != "frozen_before_dynamic_generalization":
        raise ValueError("dynamic generalization eligibility is not frozen")
    selection = eligibility.manifest.get("selection_contract") or {}
    if selection.get("performance_columns_consumed") is not False:
        raise ValueError("dynamic eligibility consumed method outcomes")
    if selection.get("task_or_window_selection_after_method_scoring_permitted") is not False:
        raise ValueError("dynamic eligibility permits post-outcome selection")
    if set(selection.get("excluded_case_study_ids") or ()) != set(development_ids):
        raise ValueError("dynamic eligibility excludes a different development set")
    case_ids = list(eligibility.primary_case_study_ids)
    if set(case_ids) & set(development_ids):
        raise ValueError("development Case Studies entered the generalization matrix")
    if int(eligibility.manifest.get("feedback_years") or 0) != int(
        runtime_config["feedback_years"]
    ):
        raise ValueError("dynamic eligibility uses a different feedback interval")

    seeds = [int(value) for value in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("dynamic seeds must be unique and non-empty")
    budgets = [int(value) for value in budgets]
    if budgets != sorted(set(budgets)) or any(value <= 0 for value in budgets):
        raise ValueError("dynamic rank points must be unique, positive, and increasing")
    if budgets[-1] != int(runtime_config["max_executions"]):
        raise ValueError("largest dynamic rank point must equal max_executions")
    windows = len(eligibility.primary_windows)

    design = {
        "schema_version": "formal-dynamic-generalization-design.v1",
        "registered_at": _utc_now(),
        "status": "frozen_before_dynamic_generalization",
        "evidence_tier": (
            "preregistered_cross_task_historical_generalization; historical years "
            "were previously exposed, so this is not untouched confirmatory evidence"
        ),
        "objective": (
            "Test whether the frozen NeuroDiscovery closed-loop policy recovers more "
            "terminal future discoveries than its otherwise identical open-loop arm."
        ),
        "canonical_release": canonical,
        "source_static_design": _file_record(static_design_path, workspace_root),
        "frozen_policy": _file_record(frozen_policy_path, workspace_root),
        "policy_application": policy_application,
        "development_case_study_ids": development_ids,
        "dynamic_eligibility": {
            "manifest": _file_record(eligibility.manifest_path, workspace_root),
            "matrix": _file_record(eligibility.matrix_path, workspace_root),
        },
        "temporal_inputs": {
            "snapshot_root": static["temporal_inputs"]["snapshot_root"],
            "freeze_years": sorted({key[1] for key in eligibility.primary_windows}),
            "feedback_years": int(runtime_config["feedback_years"]),
            "terminal_interval": (
                "For each locked window, terminal evaluation starts one year after "
                "the feedback interval and ends at the static window end year."
            ),
            "semantic_endpoint_identity_version": static["temporal_inputs"].get(
                "semantic_endpoint_identity_version"
            ),
            "relation_contract_version": static["temporal_inputs"].get(
                "relation_contract_version"
            ),
            "discovery_metric_contract_version": static["temporal_inputs"].get(
                "discovery_metric_contract_version"
            ),
        },
        "primary_matrix": {
            "selection_rule": (
                "analysis_tier == primary in the method-blind one-year-feedback "
                "dynamic eligibility lock, excluding exactly the development tasks"
            ),
            "case_study_ids": case_ids,
            "case_studies": len(case_ids),
            "case_study_windows": windows,
            "seeds": seeds,
            "paired_runs_per_arm": windows * len(seeds),
            "previously_exposed_historical_years": True,
            "claim_as_untouched_confirmatory_permitted": False,
            "task_or_window_selection_after_method_scoring_permitted": False,
            "exploratory_windows_enter_primary_aggregate": False,
        },
        "arms": {
            "closed": {
                "id": "neurodiscovery_dynamic_closed_loop",
                "feedback_enabled": True,
            },
            "open": {
                "id": "neurodiscovery_dynamic_open_loop",
                "feedback_enabled": False,
            },
            "paired_difference": (
                "The arms are byte-identical in every runtime parameter except the "
                "feedback exposure flag and output directory."
            ),
        },
        "shared_configuration": runtime_config,
        "fixed_budget_contract": {
            "rank_points": budgets,
            "fixed_execution_slots_per_run": int(runtime_config["max_executions"]),
            "failed_or_invalid_slots_still_consume_budget": True,
            "post_outcome_reranking_permitted": False,
        },
        "outcome_isolation": {
            "generation_uses_only_frozen_KG": True,
            "closed_feedback_uses_only_early_interval": True,
            "open_feedback_is_withheld": True,
            "terminal_labels_available_to_generation": False,
            "formal_KG_mutation_permitted": False,
        },
        "endpoints": {
            "primary": {
                "metric": "terminal_unique_primary_discoveries",
                "k": budgets[-1],
                "aggregation": (
                    "Within each paired seed and Case Study, average eligible windows; "
                    "then average Case Studies with equal weight; finally summarize "
                    "the paired seeds."
                ),
            },
            "metrics": [
                "terminal_unique_primary_discoveries",
                "unique_primary_discoveries",
                "recovered_future_pairs",
                "future_pair_recall",
            ],
            "dispersion": "sample variance across paired seeds with ddof=1",
            "paired_test": (
                "exact one-sided paired sign-flip test; H1 closed > open"
            ),
            "multiple_testing_primary": (
                "not applicable: one preregistered primary contrast and endpoint"
            ),
            "effect_summaries": [
                "mean paired difference",
                "paired-difference sample variance",
                "wins, ties, and losses",
            ],
            "alpha": 0.05,
        },
        "output_root": _repo_path(result_root, workspace_root),
        "reproducibility": {
            "execution_runner": "neurooracle/scripts/run_formal_dynamic_generalization.py",
            "independent_verifier": (
                "neurooracle/scripts/verify_formal_dynamic_generalization.py"
            ),
            "source_bundle_manifest": _repo_path(
                source_bundle_manifest, workspace_root
            ),
            "python_executable": (
                "C:/Users/45846/anaconda3/envs/neuroclaw/python.exe"
            ),
            "python_hash_seed": 0,
            "bundle_verified_before_and_after_each_stage": True,
            "formal_runs_rehash_all_referenced_inputs": True,
            "smoke_outputs_excluded_from_formal_analysis": True,
        },
    }

    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        design["registered_at"] = existing.get("registered_at")
        if existing != design:
            raise ValueError("existing formal dynamic design differs")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(design, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return design


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-design", type=Path, default=DEFAULT_STATIC_DESIGN)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--source-bundle-manifest", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = freeze_formal_dynamic_design(
        static_design_path=args.static_design,
        frozen_policy_path=args.frozen_policy,
        eligibility_manifest_path=args.eligibility_manifest,
        source_bundle_manifest=args.source_bundle_manifest,
        output_path=args.output,
        result_root=args.result_root,
        workspace_root=args.workspace_root,
        seeds=args.seeds,
        budgets=args.budgets,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "case_studies": result["primary_matrix"]["case_studies"],
                "case_study_windows": result["primary_matrix"]["case_study_windows"],
                "paired_runs_per_arm": result["primary_matrix"]["paired_runs_per_arm"],
            },
            indent=2,
        )
    )
