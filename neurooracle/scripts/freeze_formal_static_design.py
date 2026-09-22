"""Freeze a cross-release static hindcasting benchmark design.

The generator policy may have been selected on an earlier KG release.  The
policy is applied unchanged to the current immutable canonical copy, so this
script records policy-selection provenance separately from application data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.scripts.canonical_kg_release import CURRENT_CANONICAL_SHA256
from neurooracle.src.case_studies import list_case_study_names
from neurooracle.src.experiment_source_bundle import sha256_file
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "optimization_protocol_semantic_v4_20260814"
)
DEFAULT_CANONICAL_INPUT = (
    DEFAULT_PROTOCOL_ROOT / "canonical_input_full_v2_20260814_025713"
)
DEFAULT_SNAPSHOT_ROOT = DEFAULT_PROTOCOL_ROOT / "snapshots_full_v2_endpoint_v4"
DEFAULT_ELIGIBILITY = (
    DEFAULT_PROTOCOL_ROOT
    / "eligibility_lock_v4_all17_livecanonical"
    / "eligibility_manifest.json"
)
DEFAULT_POLICY = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
    / "frozen_static_policy.json"
)
DEFAULT_OUTPUT = DEFAULT_PROTOCOL_ROOT / "formal_static_design_v4_20260814.json"
DEFAULT_RESULT_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_static_endpoint_v8_20260814"
)
DEFAULT_BUNDLE = ROOT / "neurooracle/.frozen/formal_static_endpoint_v8"
DEFAULT_BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
POLICY_FIELDS = {
    "candidate_pool_mode",
    "task_scope_fraction",
    "evidence_frontier_fraction",
    "protect_general_top_k",
    "static_score_family",
    "endpoint_canonical_quality_weight",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(path: Path, workspace_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _file_record(path: Path, workspace_root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _repo_path(path, workspace_root), "sha256": sha256_file(path)}


def _load_object(path: Path) -> dict[str, Any]:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_release(
    canonical_input: Path, workspace_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "knowledge_graph": canonical_input / "knowledge_graph.json",
        "extracted_claims": canonical_input / "extracted_claims.jsonl",
        "current_state": canonical_input / "CURRENT_STATE.json",
    }
    records = {
        label: _file_record(path, workspace_root) for label, path in paths.items()
    }
    observed = {label: record["sha256"] for label, record in records.items()}
    expected = {key: value.upper() for key, value in CURRENT_CANONICAL_SHA256.items()}
    if observed != expected:
        raise ValueError(
            "immutable canonical copy does not match CURRENT_CANONICAL_SHA256: "
            f"observed={observed} expected={expected}"
        )
    state = _load_object(paths["current_state"])
    if state.get("status") != "canonical_current":
        raise ValueError("canonical state is not canonical_current")
    general = ((state.get("formal_kg_statistics") or {}).get("general") or {})
    return (
        {
            **records,
            "release_generated_at": state.get("generated_at"),
            "papers": int(general.get("papers") or 0),
            "claims": int(general.get("claims") or 0),
        },
        state,
    )


def _selected_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if policy.get("status") != "frozen":
        raise ValueError("source static policy is not frozen")
    selected = dict(policy.get("selected_policy") or {})
    missing = POLICY_FIELDS - set(selected)
    if missing:
        raise ValueError(f"source static policy is incomplete: {sorted(missing)}")
    return selected


def freeze_formal_static_design(
    *,
    canonical_input: Path,
    snapshot_root: Path,
    eligibility_manifest_path: Path,
    frozen_policy_path: Path,
    source_bundle_dir: Path,
    output_path: Path,
    result_root: Path,
    workspace_root: Path,
    seeds: Sequence[int] = tuple(range(10)),
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Create or byte-verify one immutable current-release static design."""

    workspace_root = workspace_root.resolve()
    canonical_input = canonical_input.resolve()
    snapshot_root = snapshot_root.resolve()
    eligibility_manifest_path = eligibility_manifest_path.resolve()
    frozen_policy_path = frozen_policy_path.resolve()
    source_bundle_dir = source_bundle_dir.resolve()
    output_path = output_path.resolve()
    result_root = result_root.resolve()

    canonical, state = _canonical_release(canonical_input, workspace_root)
    eligibility = load_locked_hindcasting_eligibility(eligibility_manifest_path)
    formal_ids = tuple(str(value) for value in eligibility.manifest.get(
        "formal_case_study_ids", ()
    ))
    registry_ids = list_case_study_names()
    if formal_ids != registry_ids:
        raise ValueError("eligibility lock does not cover the ordered 17-CS registry")
    selection = eligibility.manifest.get("selection_contract") or {}
    if selection.get("performance_columns_consumed") is not False:
        raise ValueError("eligibility lock consumed method results")
    if selection.get("task_or_window_selection_after_method_scoring_permitted") is not False:
        raise ValueError("eligibility lock permits post-outcome selection")

    policy = _load_object(frozen_policy_path)
    selected = _selected_policy(policy)
    policy_selection_release = {
        key: str(value).upper()
        for key, value in (policy.get("canonical_release") or {}).items()
    }
    application_release = {
        "knowledge_graph_sha256": canonical["knowledge_graph"]["sha256"],
        "claims_sha256": canonical["extracted_claims"]["sha256"],
        "state_sha256": canonical["current_state"]["sha256"],
    }
    if policy_selection_release == application_release:
        raise ValueError(
            "cross-release design expected a previously frozen policy release"
        )

    seeds = [int(value) for value in seeds]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("formal static design requires ten unique seeds")
    budgets = [int(value) for value in budgets]
    if budgets != sorted(set(budgets)) or any(value <= 0 for value in budgets):
        raise ValueError("rank points must be unique, positive, and increasing")
    if budgets[-1] != 1000:
        raise ValueError("formal static execution budget must end at 1000")

    freeze_years = sorted({key[1] for key in eligibility.primary_windows})
    for year in freeze_years:
        snapshot = snapshot_root / f"kg_{year}"
        for name in ("knowledge_graph.json", "extracted_claims.jsonl", "manifest.json"):
            if not (snapshot / name).is_file():
                raise FileNotFoundError(snapshot / name)

    primary_ids = list(eligibility.primary_case_study_ids)
    windows = len(eligibility.primary_windows)
    policy_record = _file_record(frozen_policy_path, workspace_root)
    design = {
        "schema_version": "neurodiscovery-formal-static-hindcasting-design.v2",
        "registered_at": _utc_now(),
        "status": "frozen_before_formal_generation",
        "evidence_tier": "locked_historical_re_evaluation_not_untouched_confirmatory",
        "objective": (
            "Apply the previously frozen outcome-blind static policy unchanged to "
            "the current canonical KG release and compare it with temporally frozen "
            "baselines on every structurally executable Case Study window."
        ),
        "canonical_release": canonical,
        "canonical_state_taxonomy": state.get("taxonomy_version"),
        "policy_application": {
            "mode": "unchanged_cross_release_application",
            "source_policy": policy_record,
            "selection_release": policy_selection_release,
            "application_release": application_release,
            "new_release_future_outcomes_used_for_policy_selection": False,
            "post_update_retuning_permitted": False,
        },
        "temporal_inputs": {
            "snapshot_root": _repo_path(snapshot_root, workspace_root),
            "freeze_years": freeze_years,
            "future_horizon_years": 5,
            "eligibility_manifest": _file_record(
                eligibility.manifest_path, workspace_root
            ),
            "eligibility_matrix": _file_record(eligibility.matrix_path, workspace_root),
            "semantic_endpoint_identity_version": "semantic-endpoint-identity.v7",
            "relation_contract_version": "case_study_relation_contracts.v3",
            "discovery_metric_contract_version": "unique_future_discoveries.v1",
            "minimum_future_pairs": 10,
            "complete_path_required_for_chain_tasks": True,
        },
        "primary_matrix": {
            "selection_rule": "analysis_tier == primary in the immutable method-blind eligibility matrix",
            "formal_case_study_ids": list(formal_ids),
            "formal_case_studies": len(formal_ids),
            "case_study_ids": primary_ids,
            "case_studies": len(primary_ids),
            "case_study_windows": windows,
            "seeds": seeds,
            "paired_runs_per_method": windows * len(seeds),
            "task_or_window_selection_after_method_scoring_permitted": False,
            "exploratory_windows_enter_primary_aggregate": False,
        },
        "methods": {
            "primary": [
                {
                    "id": "neurodiscovery",
                    "class": "frozen_general_plus_scoped_evidence_map_generator",
                    "policy_path": policy_record["path"],
                    "policy_sha256": policy_record["sha256"],
                    **{field: selected[field] for field in sorted(POLICY_FIELDS)},
                    "seed_diversity_fraction": 0.35,
                    "generation_pool_size": 1200,
                    "uses_future_outcomes": False,
                },
                {
                    "id": "sciagents",
                    "class": "seeded_frozen_global_graph_path_reasoning_baseline",
                    "uses_future_outcomes": False,
                },
                {
                    "id": "openscholar_rag",
                    "class": "frozen_case_study_literature_retrieval_baseline",
                    "uses_future_outcomes": False,
                },
            ]
        },
        "fixed_budget_contract": {
            "executed_hypotheses_per_run": 1000,
            "rank_points": budgets,
            "generation_failures_consume_slots": True,
            "invalid_or_duplicate_hypotheses_consume_slots": True,
            "manual_repair_permitted": False,
            "all_methods_share_identical_case_window_seed_matrix": True,
            "random_trials_for_pool_diagnostic": 1000,
        },
        "outcome_isolation": {
            "generation_can_read": [
                "KG snapshot frozen at or before the cutoff year",
                "active canonical Case Study definition",
                "outcome-blind deterministic random seed",
            ],
            "generation_cannot_read": [
                "claims published after the freeze year",
                "future support labels",
                "future effect sizes or P values",
                "evaluation metrics",
                "another method's scores",
                "closed-loop feedback in this static comparison",
            ],
            "formal_kg_mutation_permitted": False,
            "post_outcome_parameter_tuning_permitted": False,
        },
        "endpoints": {
            "primary": {
                "metric": "unique_primary_discoveries",
                "k": 1000,
                "aggregation": (
                    "Within each paired seed and Case Study, average eligible windows; "
                    "then average primary Case Studies equally; summarize ten seeds."
                ),
            },
            "secondary": [
                "unique_primary_discoveries at every registered rank point",
                "future_pair_recall at every registered rank point",
                "primary_hits at every registered rank point",
                "per-Case-Study equal-window summaries",
            ],
            "dispersion": "sample variance across seeds with ddof=1",
            "paired_test": "exact one-sided paired sign-flip test; H1 NeuroDiscovery > comparator",
            "multiple_testing": "Holm correction within each reported endpoint family",
            "alpha": 0.05,
        },
        "output_root": _repo_path(result_root, workspace_root),
        "reproducibility": {
            "execution_runner": "neurooracle/scripts/run_formal_static_hindcasting.py",
            "independent_verifier": "neurooracle/scripts/verify_formal_static_hindcasting.py",
            "source_bundle_manifest": _repo_path(
                source_bundle_dir / "bundle_manifest.json", workspace_root
            ),
            "python_executable": "C:/Users/45846/anaconda3/envs/neuroclaw/python.exe",
            "python_hash_seed": 0,
            "bundle_verified_before_and_after_each_stage": True,
            "formal_runs_rehash_all_referenced_inputs": True,
            "smoke_outputs_excluded_from_formal_analysis": True,
        },
    }

    if output_path.is_file():
        existing = _load_object(output_path)
        design["registered_at"] = existing.get("registered_at")
        if existing != design:
            raise ValueError("existing formal static design differs")
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
    parser.add_argument("--canonical-input", type=Path, default=DEFAULT_CANONICAL_INPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source-bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    design = freeze_formal_static_design(
        canonical_input=args.canonical_input,
        snapshot_root=args.snapshot_root,
        eligibility_manifest_path=args.eligibility_manifest,
        frozen_policy_path=args.frozen_policy,
        source_bundle_dir=args.source_bundle_dir,
        output_path=args.output,
        result_root=args.result_root,
        workspace_root=args.workspace_root,
        seeds=args.seeds,
        budgets=args.budgets,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "formal_case_studies": design["primary_matrix"]["formal_case_studies"],
        "primary_case_studies": design["primary_matrix"]["case_studies"],
        "primary_windows": design["primary_matrix"]["case_study_windows"],
        "paired_runs_per_method": design["primary_matrix"]["paired_runs_per_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
