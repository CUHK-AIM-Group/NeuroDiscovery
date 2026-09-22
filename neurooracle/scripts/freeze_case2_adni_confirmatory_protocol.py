"""Freeze an outcome- or phase-held-out Case Study 2 ADNI protocol.

This command constructs the public pathway x imaging x outcome candidate
registry and hash-pins the protocol, input tables, and current KG snapshot. It
does not parse held-out-phase outcomes or compute any held-out association
statistic. A phase-held-out protocol may parse a previously sealed development
result solely to lock an independent-replication family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from core.scripts.case2_search_policy import PUBLIC_COLUMNS, build_public_registry
from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    CASE2_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PATHWAY_ROOT,
    select_curated_pathway_exposures,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "neurooracle" / "configs" / "case2_adni_confirmatory_v2.json"
DEFAULT_KG_ROOT = REPO_ROOT / "neurooracle" / "data" / "full_v2"
HOLDOUT_AXES = {"endpoint", "cohort_phase"}
FORBIDDEN_PUBLIC_TOKENS = (
    "effect",
    "coefficient",
    "beta",
    "p_value",
    "pvalue",
    "_p",
    "_q",
    "fdr",
    "result",
    "hit",
    "rank",
)
PRIVATE_REPLICATION_COLUMNS = (
    "candidate_id",
    "exposure",
    "pathway_id",
    "pathway_name",
    "pathway_source",
    "threshold_label",
    "gene_count",
    "modality",
    "marker",
    "outcome",
    "expected_a_sign",
    "expected_b_sign",
    "expected_indirect_sign",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "last_write_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "sha256": _sha256(resolved),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_unique_strings(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    items = [str(item).strip() for item in value]
    if any(not item for item in items) or len(items) != len(set(items)):
        raise ValueError(f"{label} must contain non-empty unique values")
    return items


def _resolve_case2_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CASE2_ROOT / path


def _verify_internal_content_hash(payload: Mapping[str, Any], *, label: str) -> None:
    material = dict(payload)
    observed = material.pop("content_sha256", None)
    expected = _canonical_sha256(material)
    if observed != expected:
        raise ValueError(f"{label} has an invalid internal content hash")


def load_and_validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "case_study_id",
        "cohort",
        "development_data",
        "confirmation_data",
        "pathway_exposures",
        "imaging_markers",
        "candidate_universe",
        "statistics",
        "generator_evaluation",
        "freeze_semantics",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Case 2 protocol is missing keys: {missing}")

    development = set(
        _nonempty_unique_strings(
            protocol["development_data"]["outcomes_previously_accessed"],
            label="Development outcomes",
        )
    )
    confirmation = _nonempty_unique_strings(
        protocol["confirmation_data"]["outcomes"],
        label="Confirmation outcomes",
    )
    holdout_axis = str(protocol["cohort"].get("holdout_axis", "endpoint"))
    if holdout_axis not in HOLDOUT_AXES:
        raise ValueError(f"Unsupported Case 2 holdout axis: {holdout_axis}")
    overlap = development.intersection(confirmation)
    if holdout_axis == "endpoint" and overlap:
        raise ValueError(
            "Development and confirmation outcomes overlap: " + ", ".join(sorted(overlap))
        )
    if holdout_axis == "cohort_phase":
        development_phases = _nonempty_unique_strings(
            protocol["cohort"].get("development_values"),
            label="Development cohort phases",
        )
        confirmation_phases = _nonempty_unique_strings(
            protocol["cohort"].get(
                "confirmation_values", protocol["cohort"].get("include_values")
            ),
            label="Confirmation cohort phases",
        )
        phase_overlap = set(development_phases).intersection(confirmation_phases)
        if phase_overlap:
            raise ValueError(
                "Development and confirmation cohort phases overlap: "
                + ", ".join(sorted(phase_overlap))
            )
        include_values = _nonempty_unique_strings(
            protocol["cohort"].get("include_values"),
            label="Included confirmation cohort phases",
        )
        if include_values != confirmation_phases:
            raise ValueError(
                "For cohort-phase holdout, include_values must equal confirmation_values"
            )
        for key in ("development_selection", "phase_heldout_audit"):
            if key not in protocol:
                raise ValueError(f"Cohort-phase protocol is missing {key}")
        selection = protocol["development_selection"]
        expected_replications = int(selection.get("expected_count", 0))
        if expected_replications <= 0:
            raise ValueError("Development replication family must be non-empty")
        if selection.get("private_from_generators") is not True:
            raise ValueError("Development replication identities must be private from generators")
        if int(protocol["statistics"].get("replication_family_size_expected", 0)) != (
            expected_replications
        ):
            raise ValueError("Replication family size must match development selection")
        if protocol["generator_evaluation"].get("private_labels_visible") is not False:
            raise ValueError("Generator evaluation must not expose private replication labels")
    statistics = protocol["statistics"]
    if statistics.get("family_group_columns") != ["exposure", "outcome"]:
        raise ValueError(
            "Case 2 generator FDR family must be grouped by exposure and outcome"
        )
    if int(statistics.get("family_size_expected", 0)) != len(
        protocol["imaging_markers"]
    ):
        raise ValueError("Case 2 family size must equal the imaging-marker count")

    semantics = protocol["freeze_semantics"]
    if semantics.get("temporal_kg_freeze") is not False:
        raise ValueError("Ordinary Case 2 confirmation must not claim a temporal KG freeze")
    for required_true in (
        "kg_snapshot_pinning",
        "initial_ranking_freeze",
        "sequential_batch_commit",
    ):
        if semantics.get(required_true) is not True:
            raise ValueError(f"Protocol must enable {required_true}")
    return protocol


def build_private_replication_registry(
    development_results: pd.DataFrame,
    public_registry: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Lock development-selected candidates without exporting effects or P values."""

    selection = protocol["development_selection"]
    hit_column = str(selection.get("required_boolean_column", "nominal_chain_hit"))
    alpha = float(selection.get("component_alpha", 0.05))
    required_columns = {
        *PUBLIC_COLUMNS,
        hit_column,
        "a_path_std",
        "a_path_p",
        "b_path_std",
        "b_path_p",
        "indirect_effect_std",
        "sobel_p",
    }
    missing = sorted(required_columns - set(development_results.columns))
    if missing:
        raise ValueError(f"Development result is missing replication fields: {missing}")

    selected = development_results.loc[
        development_results[hit_column].fillna(False).astype(bool)
        & pd.to_numeric(development_results["a_path_p"], errors="coerce").lt(alpha)
        & pd.to_numeric(development_results["b_path_p"], errors="coerce").lt(alpha)
        & pd.to_numeric(development_results["sobel_p"], errors="coerce").lt(alpha)
    ].copy()
    expected_count = int(selection["expected_count"])
    if len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} development-selected candidates, found {len(selected)}"
        )
    if not selected["candidate_id"].is_unique:
        raise ValueError("Development-selected candidate IDs are not unique")

    public_ids = set(public_registry["candidate_id"].astype(str))
    unknown = sorted(set(selected["candidate_id"].astype(str)) - public_ids)
    if unknown:
        raise ValueError(f"Development-selected candidates are absent from public registry: {unknown}")

    for source, target in (
        ("a_path_std", "expected_a_sign"),
        ("b_path_std", "expected_b_sign"),
        ("indirect_effect_std", "expected_indirect_sign"),
    ):
        values = pd.to_numeric(selected[source], errors="coerce")
        if values.isna().any() or values.eq(0).any():
            raise ValueError(f"Development-selected candidates have undefined {source} signs")
        selected[target] = values.gt(0).map({True: 1, False: -1}).astype(int)

    private_registry = selected.loc[:, PRIVATE_REPLICATION_COLUMNS].copy()
    private_registry = private_registry.sort_values("candidate_id", kind="stable").reset_index(
        drop=True
    )
    if any(
        token in column.lower()
        for column in private_registry.columns
        for token in ("p_value", "pvalue", "effect", "beta", "_p", "_q", "fdr")
    ):
        raise ValueError("Private replication registry unexpectedly exports result magnitudes")
    return private_registry


def load_phase_holdout_development_inputs(
    protocol: Mapping[str, Any],
    public_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Verify and parse only the already revealed development-phase result."""

    selection = protocol["development_selection"]
    result_path = _resolve_case2_path(selection["result_path"])
    seal_path = _resolve_case2_path(selection["seal_path"])
    audit_path = _resolve_case2_path(protocol["phase_heldout_audit"]["path"])
    for path in (result_path, seal_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing phase-held-out freeze input: {path}")

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _verify_internal_content_hash(seal, label="Development final-result seal")
    expected_source_protocol = str(selection["source_protocol_id"])
    if seal.get("protocol_id") != expected_source_protocol:
        raise ValueError("Development seal belongs to a different source protocol")
    result_pin = seal.get("pins", {}).get("confirmation", {}).get("results_parquet", {})
    if _sha256(result_path) != result_pin.get("sha256"):
        raise ValueError("Development result hash does not match its final seal")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verdict = audit.get("verdict", {})
    if verdict.get("pre_access_target_outcome_leak_detected") is not False:
        raise ValueError("Phase-held-out audit did not rule out pre-access leakage")
    if verdict.get("adni1_go_2_all_candidates_meet_minimum_n") is not True:
        raise ValueError("Phase-held-out sample audit did not pass for all candidates")
    sample_audit = audit.get("sample_audit", {})
    if sample_audit.get("association_or_mediation_model_fitted") is not False:
        raise ValueError("Phase-held-out feasibility audit fit an association model")
    if sample_audit.get("outcome_effect_estimates_computed") is not False:
        raise ValueError("Phase-held-out feasibility audit computed outcome effects")

    development_results = pd.read_parquet(result_path)
    private_registry = build_private_replication_registry(
        development_results, public_registry, protocol
    )
    return private_registry, {
        "development_result": result_path,
        "development_final_seal": seal_path,
        "phase_heldout_access_sample_audit": audit_path,
    }


def build_confirmation_registry(
    subjects: pd.DataFrame,
    score_manifest: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposures = select_curated_pathway_exposures(subjects, score_manifest)
    expected_exposures = int(protocol["pathway_exposures"]["expected_count"])
    if len(exposures) != expected_exposures:
        raise ValueError(
            f"Expected {expected_exposures} curated pathways, found {len(exposures)}"
        )

    candidates: list[dict[str, Any]] = []
    for exposure in exposures.to_dict(orient="records"):
        for marker_spec in protocol["imaging_markers"]:
            modality, separator, marker = str(marker_spec).partition("::")
            if not separator or not modality or not marker:
                raise ValueError(f"Invalid imaging marker spec: {marker_spec!r}")
            for outcome in protocol["confirmation_data"]["outcomes"]:
                candidates.append(
                    {
                        "exposure": exposure["score_name"],
                        "pathway_id": exposure["pathway_id"],
                        "pathway_name": exposure["pathway_name"],
                        "pathway_source": exposure["pathway_source"],
                        "threshold_label": exposure["threshold_label"],
                        "gene_count": exposure["gene_count"],
                        "modality": modality,
                        "marker": marker,
                        "outcome": outcome,
                    }
                )
    registry = build_public_registry(pd.DataFrame(candidates))
    expected_candidates = int(protocol["candidate_universe"]["expected_count"])
    if len(registry) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates, found {len(registry)}"
        )
    if tuple(registry.columns) != tuple(PUBLIC_COLUMNS):
        raise ValueError("Public registry schema changed unexpectedly")
    forbidden = [
        column
        for column in registry.columns
        if any(token in column.lower() for token in FORBIDDEN_PUBLIC_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Outcome-bearing columns leaked into registry: {forbidden}")
    return registry, exposures


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_and_validate_protocol(args.protocol)
    output_root = args.output_root or (
        CASE2_ROOT / "protocols" / str(protocol["protocol_id"])
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Frozen protocol directory is not empty: {output_root}. "
            "Use a new protocol_id instead of overwriting a completed freeze."
        )
    subjects_path = args.dataset_root / "case2_adni_subject_genetics_covariates.parquet"
    score_manifest_path = args.pathway_root / "score_manifest.csv"
    subjects = pd.read_parquet(subjects_path)
    score_manifest = pd.read_csv(score_manifest_path)
    registry, exposures = build_confirmation_registry(subjects, score_manifest, protocol)

    holdout_axis = str(protocol["cohort"].get("holdout_axis", "endpoint"))
    private_registry: pd.DataFrame | None = None
    development_files: dict[str, Path] = {}
    if holdout_axis == "cohort_phase":
        private_registry, development_files = load_phase_holdout_development_inputs(
            protocol, registry
        )

    output_root.mkdir(parents=True, exist_ok=True)

    registry_path = output_root / "public_candidate_registry.csv"
    exposures_path = output_root / "selected_pathway_exposures.csv"
    protocol_copy_path = output_root / "protocol.json"
    private_registry_path = output_root / "PRIVATE_REPLICATION_REGISTRY.csv"
    registry.to_csv(registry_path, index=False)
    exposures.to_csv(exposures_path, index=False)
    if private_registry is not None:
        private_registry.to_csv(private_registry_path, index=False)
    protocol_copy_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataset_files = {
        "subject_genetics_covariates": subjects_path,
        "clinical_visits": args.dataset_root / "case2_adni_clinical_visits.parquet",
        "imaging_markers": args.dataset_root
        / "case2_adni_primary_imaging_markers_long.parquet",
        "longitudinal_outcome_pairs": args.dataset_root
        / "case2_adni_longitudinal_outcome_pairs.parquet",
        "pathway_score_manifest": score_manifest_path,
    }
    kg_files = {
        "current_state": args.kg_root / "CURRENT_STATE.json",
        "knowledge_graph": args.kg_root / "knowledge_graph.json",
        "extracted_claims": args.kg_root / "extracted_claims.jsonl",
    }
    missing = [
        str(path)
        for path in (*dataset_files.values(), *kg_files.values(), *development_files.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing files required for protocol freeze: " + "; ".join(missing))

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock_material = {
        "protocol_sha256": _sha256(protocol_copy_path),
        "registry_sha256": _sha256(registry_path),
        "selected_exposures_sha256": _sha256(exposures_path),
        "dataset_inputs": {name: _file_pin(path) for name, path in dataset_files.items()},
        "kg_snapshot": {name: _file_pin(path) for name, path in kg_files.items()},
    }
    if private_registry is not None:
        lock_material["private_replication_registry_sha256"] = _sha256(
            private_registry_path
        )
        lock_material["development_replication_inputs"] = {
            name: _file_pin(path) for name, path in development_files.items()
        }
    freeze_id = _canonical_sha256(lock_material)
    phase_heldout = holdout_axis == "cohort_phase"
    outcome_access_audit = {
        "confirmation_association_statistics_read": False,
        "confirmation_models_fit": False,
        "confirmation_result_table_created": False,
        "parsed_inputs": [
            "subject genetics and covariates",
            "pathway score manifest",
        ],
        "nonparsed_inputs_hash_pinned_only": [
            "clinical visits",
            "imaging markers",
            "longitudinal outcome pairs",
            "knowledge graph",
            "claim store",
        ],
    }
    artifacts = {
        "protocol": str(protocol_copy_path),
        "public_candidate_registry": str(registry_path),
        "selected_pathway_exposures": str(exposures_path),
    }
    if phase_heldout:
        outcome_access_audit.update(
            {
                "development_phase_association_statistics_read": True,
                "development_phase_values": list(protocol["cohort"]["development_values"]),
                "confirmation_phase_values": list(
                    protocol["cohort"]["confirmation_values"]
                ),
                "confirmation_phase_association_statistics_read": False,
                "parsed_inputs": outcome_access_audit["parsed_inputs"]
                + [
                    "sealed development-phase result",
                    "development final-result seal",
                    "phase-held-out access and sample audit",
                ],
            }
        )
        artifacts["private_replication_registry_not_generator_input"] = str(
            private_registry_path
        )
    manifest = {
        "schema_version": (
            "neurooracle.case2_protocol_freeze.v2"
            if phase_heldout
            else "neurooracle.case2_protocol_freeze.v1"
        ),
        "status": (
            "frozen_before_phase_heldout_association_access"
            if phase_heldout
            else "frozen_before_confirmation_association_access"
        ),
        "created_at_utc": created_at,
        "freeze_id": freeze_id,
        "protocol_id": protocol["protocol_id"],
        "case_study_id": protocol["case_study_id"],
        "analysis_label": protocol["analysis_label"],
        "candidate_count": int(len(registry)),
        "pathway_count": int(registry["pathway_id"].nunique()),
        "imaging_marker_count": int(
            registry[["modality", "marker"]].drop_duplicates().shape[0]
        ),
        "outcome_count": int(registry["outcome"].nunique()),
        "confirmation_outcomes": list(protocol["confirmation_data"]["outcomes"]),
        "cohort_filter": protocol["cohort"],
        "freeze_semantics": protocol["freeze_semantics"],
        "private_replication_candidate_count": (
            int(len(private_registry)) if private_registry is not None else 0
        ),
        "generator_data_boundary": protocol["generator_evaluation"].get(
            "data_boundary"
        ),
        "outcome_access_audit": outcome_access_audit,
        "artifacts": artifacts,
        "lock_material": lock_material,
    }
    manifest_path = output_root / "protocol_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": (
            "neurooracle.case2_protocol_lock.v2"
            if phase_heldout
            else "neurooracle.case2_protocol_lock.v1"
        ),
        "freeze_id": freeze_id,
        "manifest_sha256": _sha256(manifest_path),
        "do_not_overwrite": True,
    }
    (output_root / "FREEZE.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    freeze_protocol(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 12:41 HKT
