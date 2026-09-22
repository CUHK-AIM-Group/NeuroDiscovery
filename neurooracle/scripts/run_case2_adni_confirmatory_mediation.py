"""Run the locked ADNI Case Study 2 held-out mediation analysis.

The command refuses to open held-out outcomes until the protocol,
NeuroDiscovery initial ranking, all six static baseline rankings, and the
generator evaluation plan are hash-locked. It supports the v2 endpoint-held-out
analysis and the v3 cohort-phase-held-out replication without conflating either
design with temporal hindcasting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import statsmodels

from core.scripts.case2_search_policy import PUBLIC_COLUMNS
from neurooracle.scripts.case2_confirmatory_statistics import (
    global_chain_labels,
    pathway_outcome_family_labels,
)
from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import CASE2_ROOT
from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    build_index_analysis_rows,
    build_longitudinal_covariate_matrix,
    _complete_case_mask,
)
from neurooracle.scripts.run_case2_adni_mediation_smoke import mediation_screen


DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_endpoint_holdout_v2"
EXPECTED_BASELINE_METHODS = {
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
}
MODEL_COLUMNS = (
    "a_path_std",
    "a_path_se",
    "a_path_p",
    "b_path_std",
    "b_path_se",
    "b_path_p",
    "total_effect_std",
    "direct_effect_std",
    "indirect_effect_std",
    "sobel_se",
    "sobel_z",
    "sobel_p",
    "n",
    "covariate_rank",
    "df_a",
    "df_b",
)


def _holdout_axis(protocol: dict[str, Any]) -> str:
    return str(protocol.get("cohort", {}).get("holdout_axis", "endpoint"))


def _pre_access_status(protocol: dict[str, Any]) -> str:
    if _holdout_axis(protocol) == "cohort_phase":
        return "frozen_before_phase_heldout_association_access"
    return "frozen_before_confirmation_association_access"


def _ranking_lock_status(protocol: dict[str, Any]) -> str:
    if _holdout_axis(protocol) == "cohort_phase":
        return "locked_before_phase_heldout_association_access"
    return "locked_before_confirmation_association_access"


def _run_root(protocol: dict[str, Any], freeze_id: str) -> Path:
    if _holdout_axis(protocol) == "cohort_phase":
        experiment_name = str(protocol["protocol_id"])
    else:
        experiment_name = "case2_adni_confirmatory_v1"
    return CASE2_ROOT / "experiments" / experiment_name / freeze_id[:12]


def _heldout_output_name(protocol: dict[str, Any]) -> str:
    if _holdout_axis(protocol) == "cohort_phase":
        return "hidden_phase_heldout_results"
    return "hidden_confirmation_results"


def _access_record_name(protocol: dict[str, Any]) -> str:
    if _holdout_axis(protocol) == "cohort_phase":
        return "PHASE_HELDOUT_ASSOCIATION_ACCESS_STARTED.json"
    return "CONFIRMATION_OUTCOME_ACCESS_STARTED.json"


def _result_lock_name(protocol: dict[str, Any]) -> str:
    if _holdout_axis(protocol) == "cohort_phase":
        return "PHASE_HELDOUT_RESULTS.lock.json"
    return "CONFIRMATION_RESULTS.lock.json"


def _one_sided_p(two_sided_p: Any, estimate: Any, expected_sign: Any) -> float:
    p_value = float(two_sided_p)
    effect = float(estimate)
    sign = float(expected_sign)
    if not (np.isfinite(p_value) and np.isfinite(effect) and np.isfinite(sign)):
        return float("nan")
    p_value = min(1.0, max(0.0, p_value))
    if effect * sign > 0:
        return p_value / 2.0
    if effect * sign < 0:
        return 1.0 - p_value / 2.0
    return 0.5


def _holm_adjust_fixed_family(p_values: Iterable[Any]) -> np.ndarray:
    """Holm-adjust a frozen family, counting non-estimable entries as failures."""

    cleaned: list[float] = []
    for value in p_values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1.0
        cleaned.append(number if np.isfinite(number) else 1.0)
    values = np.asarray(cleaned, dtype=float)
    values = np.clip(values, 0.0, 1.0)
    order = np.argsort(values, kind="stable")
    adjusted = np.ones(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _weakest_link_evidence(results: pd.DataFrame, *, cap: float = 50.0) -> np.ndarray:
    """Return -log10 of the weakest component P, with non-estimable rows at zero."""

    component_p = results.loc[:, ["a_path_p", "b_path_p", "sobel_p"]].apply(
        pd.to_numeric, errors="coerce"
    )
    weakest = component_p.max(axis=1, skipna=False).to_numpy(float)
    evidence = np.zeros(len(results), dtype=float)
    finite = np.isfinite(weakest)
    evidence[finite] = np.minimum(
        float(cap), -np.log10(np.clip(weakest[finite], 1e-300, 1.0))
    )
    return evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    payload = "\n".join(map(str, values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_pre_outcome_locks(
    protocol_root: Path,
    *,
    baseline_root: Path | None = None,
    neurodiscovery_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all independent ranking commitments before outcome access."""

    protocol_manifest_path = protocol_root / "protocol_freeze_manifest.json"
    protocol_lock_path = protocol_root / "FREEZE.lock.json"
    protocol = _load_json(protocol_root / "protocol.json")
    manifest = _load_json(protocol_manifest_path)
    protocol_lock = _load_json(protocol_lock_path)
    expected_protocol_status = _pre_access_status(protocol)
    if manifest.get("status") != expected_protocol_status:
        raise ValueError("Case 2 protocol is not in the pre-outcome frozen state")
    if manifest.get("freeze_id") != protocol_lock.get("freeze_id"):
        raise ValueError("protocol freeze ID differs from its lock")
    if _sha256(protocol_manifest_path) != protocol_lock.get("manifest_sha256"):
        raise ValueError("protocol freeze manifest hash mismatch")
    lock_material = manifest["lock_material"]
    frozen_artifacts = {
        "protocol": (
            Path(manifest["artifacts"]["protocol"]),
            lock_material["protocol_sha256"],
        ),
        "public_candidate_registry": (
            Path(manifest["artifacts"]["public_candidate_registry"]),
            lock_material["registry_sha256"],
        ),
        "selected_pathway_exposures": (
            Path(manifest["artifacts"]["selected_pathway_exposures"]),
            lock_material["selected_exposures_sha256"],
        ),
    }
    if _holdout_axis(protocol) == "cohort_phase":
        private_path = Path(
            manifest["artifacts"]["private_replication_registry_not_generator_input"]
        )
        frozen_artifacts["private_replication_registry"] = (
            private_path,
            lock_material["private_replication_registry_sha256"],
        )
    for name, (path, expected_sha) in frozen_artifacts.items():
        if _sha256(path) != expected_sha:
            raise ValueError(f"frozen protocol artifact changed: {name}")
    registry = pd.read_csv(frozen_artifacts["public_candidate_registry"][0])
    candidate_ids = registry["candidate_id"].astype(str).tolist()
    candidate_count = len(candidate_ids)
    freeze_id = str(manifest["freeze_id"])
    run_root = _run_root(protocol, freeze_id)
    baseline_root = baseline_root or run_root / "official_baselines_gpt55_high"
    neurodiscovery_root = neurodiscovery_root or run_root / "neurodiscovery_policy"

    baseline_lock_path = baseline_root / "BASELINE_RANKINGS.lock.json"
    nd_lock_path = neurodiscovery_root / "INITIAL_RANKING.lock.json"
    if not baseline_lock_path.is_file():
        raise FileNotFoundError(
            "All six static baseline rankings must be locked before outcome access: "
            f"{baseline_lock_path}"
        )
    if not nd_lock_path.is_file():
        raise FileNotFoundError(
            "NeuroDiscovery initial ranking must be locked before outcome access: "
            f"{nd_lock_path}"
        )
    baseline_lock = _load_json(baseline_lock_path)
    nd_lock = _load_json(nd_lock_path)
    expected_ranking_status = _ranking_lock_status(protocol)
    if baseline_lock.get("status") != expected_ranking_status:
        raise ValueError("static baseline rankings are not in the pre-outcome locked state")
    if baseline_lock.get("protocol_freeze_id") != freeze_id:
        raise ValueError("baseline rankings belong to another protocol freeze")
    if baseline_lock.get("association_results_accessed") is not False:
        raise ValueError("baseline ranking lock reports association-result access")
    if baseline_lock.get("confirmation_holdout_axis") != _holdout_axis(protocol):
        raise ValueError("baseline ranking lock uses another holdout axis")
    pairs = {
        (str(row["method"]), int(row["trial"]))
        for row in baseline_lock.get("orders", [])
    }
    expected_pairs = {
        (method, trial)
        for method in EXPECTED_BASELINE_METHODS
        for trial in range(
            int(protocol["generator_evaluation"]["independent_runs_per_method"])
        )
    }
    if pairs != expected_pairs:
        missing = sorted(expected_pairs - pairs)
        extra = sorted(pairs - expected_pairs)
        raise ValueError(
            f"baseline ranking lock is incomplete; missing={missing[:5]}, extra={extra[:5]}"
        )
    if baseline_lock.get("candidate_registry_sha256") != lock_material["registry_sha256"]:
        raise ValueError("baseline rankings use another candidate registry")
    orders_path = Path(baseline_lock["orders_path"])
    if _sha256(orders_path) != baseline_lock.get("orders_file_sha256"):
        raise ValueError("static baseline order archive hash mismatch")
    with np.load(orders_path, allow_pickle=False) as archive:
        for record in baseline_lock["orders"]:
            key = str(record["array_key"])
            if key not in archive:
                raise ValueError(f"baseline order archive is missing {key}")
            order = np.asarray(archive[key], dtype=np.int64)
            if len(order) != candidate_count or sorted(order.tolist()) != list(
                range(candidate_count)
            ):
                raise ValueError(f"baseline order is not a full permutation: {key}")
            if hashlib.sha256(order.astype(np.int32).tobytes()).hexdigest() != record.get(
                "order_index_sha256"
            ):
                raise ValueError(f"baseline order index commit mismatch: {key}")
            ranked_ids = [candidate_ids[index] for index in order]
            if _sha256_lines(ranked_ids) != record.get("ranking_commit_sha256"):
                raise ValueError(f"baseline candidate ranking commit mismatch: {key}")
    if nd_lock.get("status") != expected_ranking_status:
        raise ValueError("NeuroDiscovery initial ranking is not pre-outcome locked")
    if nd_lock.get("protocol_freeze_id") != freeze_id:
        raise ValueError("NeuroDiscovery ranking belongs to another protocol freeze")
    if nd_lock.get("association_results_accessed") is not False:
        raise ValueError("NeuroDiscovery ranking lock reports association-result access")
    if nd_lock.get("confirmation_holdout_axis") != _holdout_axis(protocol):
        raise ValueError("NeuroDiscovery ranking lock uses another holdout axis")
    if nd_lock.get("candidate_registry_sha256") != lock_material["registry_sha256"]:
        raise ValueError("NeuroDiscovery ranking uses another candidate registry")
    policy_path = Path(nd_lock["policy_path"])
    initial_path = Path(nd_lock["initial_ranking_path"])
    if _sha256(policy_path) != nd_lock.get("policy_sha256"):
        raise ValueError("NeuroDiscovery policy hash mismatch")
    if _sha256(initial_path) != nd_lock.get("initial_ranking_file_sha256"):
        raise ValueError("NeuroDiscovery initial ranking file hash mismatch")
    initial = pd.read_csv(initial_path).sort_values("initial_rank", kind="stable")
    initial_ids = initial["candidate_id"].astype(str).tolist()
    if len(initial_ids) != candidate_count or set(initial_ids) != set(candidate_ids):
        raise ValueError("NeuroDiscovery initial ranking is not a full candidate permutation")
    if _sha256_lines(initial_ids) != nd_lock.get("ranking_commit_sha256"):
        raise ValueError("NeuroDiscovery initial candidate ranking commit mismatch")
    if nd_lock.get("temporal_freeze_year") is not None:
        raise ValueError("ordinary CS2 confirmation cannot declare a temporal freeze year")
    if nd_lock.get("kg_snapshot_sha256") != baseline_lock.get("kg_snapshot_sha256"):
        raise ValueError("baseline and NeuroDiscovery locks pin different KG snapshots")
    kg_sha = lock_material["kg_snapshot"]["knowledge_graph"]["sha256"]
    if nd_lock.get("kg_snapshot_sha256") != kg_sha:
        raise ValueError("ranking locks do not match the protocol KG snapshot")
    claims_sha = lock_material["kg_snapshot"]["extracted_claims"]["sha256"]
    if baseline_lock.get("claim_store_snapshot_sha256") != claims_sha:
        raise ValueError("baseline ranking lock pins another claim-store snapshot")
    if nd_lock.get("claim_store_snapshot_sha256") != claims_sha:
        raise ValueError("NeuroDiscovery ranking lock pins another claim-store snapshot")

    for name, pin in lock_material["dataset_inputs"].items():
        path = Path(pin["path"])
        if _sha256(path) != pin["sha256"]:
            raise ValueError(f"frozen dataset input changed: {name}")
    return {
        "protocol": protocol,
        "protocol_manifest": manifest,
        "run_root": run_root,
        "baseline_root": baseline_root,
        "neurodiscovery_root": neurodiscovery_root,
        "baseline_lock_path": baseline_lock_path,
        "baseline_lock_sha256": _sha256(baseline_lock_path),
        "neurodiscovery_lock_path": nd_lock_path,
        "neurodiscovery_lock_sha256": _sha256(nd_lock_path),
    }


def _empty_result(candidate: pd.Series, *, n: int, status: str) -> dict[str, Any]:
    row = {column: candidate[column] for column in PUBLIC_COLUMNS}
    row.update({column: np.nan for column in MODEL_COLUMNS})
    row.update(
        {
            "n": int(n),
            "n_subjects": int(n),
            "analysis_status": status,
            "executable": False,
            "median_followup_years": np.nan,
            "covariates": "",
            "abs_indirect_effect_std": np.nan,
        }
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    locks = verify_pre_outcome_locks(
        args.protocol_root,
        baseline_root=args.baseline_root,
        neurodiscovery_root=args.neurodiscovery_root,
    )
    protocol = locks["protocol"]
    manifest = locks["protocol_manifest"]
    evaluation_plan_lock_path = args.evaluation_plan_lock or (
        locks["run_root"] / "generator_evaluation" / "EVALUATION_PLAN.lock.json"
    )
    evaluation_plan_lock = _load_json(evaluation_plan_lock_path)
    if evaluation_plan_lock.get("status") != _ranking_lock_status(protocol):
        raise ValueError("generator evaluation plan is not pre-outcome locked")
    if evaluation_plan_lock.get("protocol_freeze_id") != manifest["freeze_id"]:
        raise ValueError("generator evaluation plan belongs to another protocol")
    if evaluation_plan_lock.get("baseline_ranking_lock_sha256") != locks[
        "baseline_lock_sha256"
    ]:
        raise ValueError("evaluation plan does not bind the current baseline rankings")
    if evaluation_plan_lock.get(
        "neurodiscovery_initial_ranking_lock_sha256"
    ) != locks["neurodiscovery_lock_sha256"]:
        raise ValueError("evaluation plan does not bind the current ND initial ranking")
    evaluation_plan_path = Path(evaluation_plan_lock["evaluation_plan_path"])
    if _sha256(evaluation_plan_path) != evaluation_plan_lock.get(
        "evaluation_plan_sha256"
    ):
        raise ValueError("locked generator evaluation plan hash mismatch")
    evaluation_plan = _load_json(evaluation_plan_path)
    if _holdout_axis(protocol) == "cohort_phase":
        evidence_plan = evaluation_plan.get("continuous_evidence", {})
        if (
            evidence_plan.get("name") != "heldout_chain_evidence_score"
            or float(evidence_plan.get("cap", -1)) != 50.0
            or float(evidence_plan.get("non_estimable_value", -1)) != 0.0
            or evidence_plan.get("frozen_before_phase_heldout_association_access")
            is not True
        ):
            raise ValueError("v3 continuous evidence definition is not frozen")
    output_root = args.output_root or (
        locks["run_root"] / _heldout_output_name(protocol)
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"confirmation output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    access_started = {
        "schema_version": "neurooracle.case2_heldout_access.v2",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_freeze_id": manifest["freeze_id"],
        "protocol_id": protocol["protocol_id"],
        "holdout_axis": _holdout_axis(protocol),
        "protocol_manifest_sha256": _sha256(
            args.protocol_root / "protocol_freeze_manifest.json"
        ),
        "baseline_ranking_lock_sha256": locks["baseline_lock_sha256"],
        "neurodiscovery_initial_ranking_lock_sha256": locks[
            "neurodiscovery_lock_sha256"
        ],
        "generator_evaluation_plan_lock_sha256": _sha256(
            evaluation_plan_lock_path
        ),
        "temporal_freeze_year": None,
        "meaning": (
            "Held-out association inputs may be opened only after this record exists. "
            "For cohort_phase, this is independent phase validation, not endpoint "
            "holdout and not temporal hindcasting."
        ),
    }
    access_path = output_root / _access_record_name(protocol)
    access_path.write_text(
        json.dumps(access_started, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pins = manifest["lock_material"]["dataset_inputs"]
    subjects = pd.read_parquet(Path(pins["subject_genetics_covariates"]["path"]))
    clinical = pd.read_parquet(Path(pins["clinical_visits"]["path"]))
    markers = pd.read_parquet(Path(pins["imaging_markers"]["path"]))
    longitudinal = pd.read_parquet(Path(pins["longitudinal_outcome_pairs"]["path"]))
    registry = pd.read_csv(manifest["artifacts"]["public_candidate_registry"])
    exposures = pd.read_csv(manifest["artifacts"]["selected_pathway_exposures"])

    cohort = protocol["cohort"]
    include_column = str(cohort["include_column"])
    include_values = set(map(str, cohort["include_values"]))
    subjects = subjects.loc[
        subjects[include_column].astype(str).isin(include_values)
    ].copy()
    subject_ids = set(subjects["subject_id"].astype(str))
    if not subject_ids:
        raise RuntimeError("no subjects satisfy the locked confirmation cohort filter")
    clinical = clinical.loc[clinical["subject_id"].astype(str).isin(subject_ids)].copy()
    markers = markers.loc[markers["subject_id"].astype(str).isin(subject_ids)].copy()
    longitudinal = longitudinal.loc[
        longitudinal["subject_id"].astype(str).isin(subject_ids)
    ].copy()

    confirmation = protocol["confirmation_data"]
    analysis_rows = build_index_analysis_rows(
        markers,
        longitudinal,
        subjects,
        clinical,
        marker_specs=protocol["imaging_markers"],
        outcomes=confirmation["outcomes"],
        target_followup_days=int(confirmation["target_followup_days"]),
        min_followup_days=int(confirmation["min_followup_days"]),
        max_followup_days=int(confirmation["max_followup_days"]),
    )
    if not analysis_rows[include_column].astype(str).isin(include_values).all():
        raise ValueError("analysis rows contain subjects outside the locked cohort")
    family_lookup = {
        tuple(map(str, key)): frame
        for key, frame in analysis_rows.groupby(
            ["modality", "marker", "outcome"], sort=False
        )
    }
    exposure_lookup = exposures.set_index("score_name", drop=False)
    result_rows: list[dict[str, Any]] = []
    for _, candidate in registry.iterrows():
        key = (
            str(candidate["modality"]),
            str(candidate["marker"]),
            str(candidate["outcome"]),
        )
        family = family_lookup.get(key)
        if family is None:
            result_rows.append(_empty_result(candidate, n=0, status="no_analysis_rows"))
            continue
        exposure = str(candidate["exposure"])
        if exposure not in exposure_lookup.index:
            raise ValueError(f"frozen exposure is unavailable: {exposure}")
        include_icv = str(candidate["modality"]) == "smri_adnimerge"
        complete_mask = _complete_case_mask(
            family,
            exposure,
            include_icv=include_icv,
        )
        complete = family.loc[complete_mask].copy()
        if len(complete) < int(confirmation["minimum_complete_case_n"]):
            result_rows.append(
                _empty_result(candidate, n=len(complete), status="below_minimum_n")
            )
            continue
        covariates, covariate_names = build_longitudinal_covariate_matrix(
            complete,
            include_icv=include_icv,
        )
        fitted, marker_mask = mediation_screen(
            pd.to_numeric(complete[exposure], errors="coerce").to_numpy(float),
            pd.to_numeric(
                complete["annualized_decline_score"], errors="coerce"
            ).to_numpy(float),
            pd.to_numeric(complete["value"], errors="coerce")
            .to_numpy(float)
            .reshape(-1, 1),
            covariates,
        )
        if not marker_mask[0] or fitted.empty:
            result_rows.append(
                _empty_result(candidate, n=len(complete), status="non_estimable")
            )
            continue
        model = fitted.iloc[0].to_dict()
        row = {column: candidate[column] for column in PUBLIC_COLUMNS}
        row.update({column: model.get(column, np.nan) for column in MODEL_COLUMNS})
        row.update(
            {
                "n_subjects": int(len(complete)),
                "analysis_status": "estimated",
                "executable": True,
                "median_followup_years": float(complete["followup_years"].median()),
                "covariates": ";".join(covariate_names),
                "abs_indirect_effect_std": abs(float(model["indirect_effect_std"])),
            }
        )
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    if len(results) != len(registry):
        raise RuntimeError("confirmation results do not preserve the candidate universe")
    if results["candidate_id"].tolist() != registry["candidate_id"].astype(str).tolist():
        raise RuntimeError("confirmation result order differs from the frozen registry")
    family_labels, family_q = pathway_outcome_family_labels(
        results, float(protocol["statistics"]["fdr_alpha"])
    )
    global_labels, global_q = global_chain_labels(
        results, float(protocol["statistics"]["fdr_alpha"])
    )
    alpha = float(protocol["statistics"]["fdr_alpha"])
    nominal_labels = (
        pd.to_numeric(results["sobel_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )
    component_p = results.loc[:, ["a_path_p", "b_path_p", "sobel_p"]].apply(
        pd.to_numeric, errors="coerce"
    )
    results["weakest_link_p_two_sided"] = component_p.max(axis=1, skipna=False)
    evidence_cap = float(
        evaluation_plan.get("continuous_evidence", {}).get("cap", 50.0)
    )
    results["heldout_chain_evidence_score"] = _weakest_link_evidence(
        results, cap=evidence_cap
    )
    results["registered_replication_candidate"] = False
    for column in (
        "expected_a_sign",
        "expected_b_sign",
        "expected_indirect_sign",
        "a_path_p_one_sided",
        "b_path_p_one_sided",
        "sobel_p_one_sided",
        "sobel_p_one_sided_holm",
    ):
        results[column] = np.nan
    for column in (
        "a_direction_match",
        "b_direction_match",
        "indirect_direction_match",
        "directional_replication_hit",
    ):
        results[column] = False

    replication_labels = np.zeros(len(results), dtype=bool)
    if _holdout_axis(protocol) == "cohort_phase":
        private_path = Path(
            manifest["artifacts"]["private_replication_registry_not_generator_input"]
        )
        private = pd.read_csv(private_path)
        expected_replications = int(
            protocol["statistics"]["replication_family_size_expected"]
        )
        if len(private) != expected_replications or not private["candidate_id"].is_unique:
            raise ValueError("private replication registry violates the frozen family")
        result_index = pd.Series(
            np.arange(len(results), dtype=int),
            index=results["candidate_id"].astype(str),
        )
        if not set(private["candidate_id"].astype(str)).issubset(result_index.index):
            raise ValueError("private replication registry is outside the public universe")
        positions = result_index.loc[private["candidate_id"].astype(str)].to_numpy(int)
        results.loc[positions, "registered_replication_candidate"] = True
        for private_column in (
            "expected_a_sign",
            "expected_b_sign",
            "expected_indirect_sign",
        ):
            results.loc[positions, private_column] = private[private_column].to_numpy(int)

        component_specs = (
            ("a_path_std", "a_path_p", "expected_a_sign", "a"),
            ("b_path_std", "b_path_p", "expected_b_sign", "b"),
            (
                "indirect_effect_std",
                "sobel_p",
                "expected_indirect_sign",
                "indirect",
            ),
        )
        for effect_column, p_column, sign_column, prefix in component_specs:
            one_sided = [
                _one_sided_p(
                    results.at[position, p_column],
                    results.at[position, effect_column],
                    results.at[position, sign_column],
                )
                for position in positions
            ]
            output_p = "sobel_p_one_sided" if prefix == "indirect" else f"{prefix}_path_p_one_sided"
            results.loc[positions, output_p] = one_sided
            results.loc[positions, f"{prefix}_direction_match"] = [
                bool(
                    np.isfinite(float(results.at[position, effect_column]))
                    and float(results.at[position, effect_column])
                    * float(results.at[position, sign_column])
                    > 0
                )
                for position in positions
            ]

        holm_values = _holm_adjust_fixed_family(
            results.loc[positions, "sobel_p_one_sided"].to_numpy(float)
        )
        results.loc[positions, "sobel_p_one_sided_holm"] = holm_values
        replication_alpha = float(protocol["statistics"]["replication_alpha"])
        replication_labels[positions] = (
            results.loc[positions, "executable"].astype(bool).to_numpy()
            & results.loc[positions, "a_direction_match"].astype(bool).to_numpy()
            & results.loc[positions, "b_direction_match"].astype(bool).to_numpy()
            & results.loc[positions, "indirect_direction_match"].astype(bool).to_numpy()
            & pd.to_numeric(
                results.loc[positions, "a_path_p_one_sided"], errors="coerce"
            ).lt(replication_alpha).to_numpy()
            & pd.to_numeric(
                results.loc[positions, "b_path_p_one_sided"], errors="coerce"
            ).lt(replication_alpha).to_numpy()
            & pd.to_numeric(
                results.loc[positions, "sobel_p_one_sided"], errors="coerce"
            ).lt(replication_alpha).to_numpy()
            & pd.to_numeric(
                results.loc[positions, "sobel_p_one_sided_holm"], errors="coerce"
            ).lt(replication_alpha).to_numpy()
        )
        results["directional_replication_hit"] = replication_labels

    results.insert(0, "candidate_registry_rank", np.arange(1, len(results) + 1))
    results["sobel_q_family"] = family_q
    results["sobel_q_global"] = global_q
    results["family_fdr_chain_hit"] = family_labels
    results["global_fdr_chain_hit"] = global_labels
    results["nominal_chain_hit"] = nominal_labels
    results["family_registered_size"] = int(
        protocol["statistics"]["family_size_expected"]
    )
    results["family_estimable_size"] = results.groupby(
        ["exposure", "outcome"]
    )["executable"].transform("sum")

    result_prefix = (
        "case2_phase_heldout"
        if _holdout_axis(protocol) == "cohort_phase"
        else "case2_confirmation"
    )
    result_path = output_root / f"{result_prefix}_results.parquet"
    result_csv_path = output_root / f"{result_prefix}_results.csv"
    rows_path = output_root / f"{result_prefix}_analysis_rows.parquet"
    results.to_parquet(result_path, index=False, compression="zstd")
    results.to_csv(result_csv_path, index=False)
    analysis_rows.to_parquet(rows_path, index=False, compression="zstd")

    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }
    environment_path = output_root / "environment_manifest.json"
    environment_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "neurooracle.case2_heldout_results.v2",
        "protocol_freeze_id": manifest["freeze_id"],
        "protocol_id": protocol["protocol_id"],
        "analysis_label": protocol.get("analysis_label"),
        "holdout_axis": _holdout_axis(protocol),
        "analysis": (
            "association-based indirect-effect cohort-phase replication; not causal"
            if _holdout_axis(protocol) == "cohort_phase"
            else "association-based indirect-effect endpoint confirmation; not causal"
        ),
        "cohort": {include_column: sorted(include_values)},
        "subjects_in_locked_cohort": int(len(subject_ids)),
        "subjects_in_analysis_rows": int(analysis_rows["subject_id"].nunique()),
        "candidate_count": int(len(results)),
        "estimated_candidates": int(results["executable"].sum()),
        "non_estimable_candidates": int((~results["executable"]).sum()),
        "nominal_chain_hits": int(nominal_labels.sum()),
        "pathway_outcome_family_fdr_chain_hits": int(family_labels.sum()),
        "global_fdr_chain_hits": int(global_labels.sum()),
        "registered_replication_candidates": int(
            results["registered_replication_candidate"].sum()
        ),
        "directionally_replicated_registered_candidates": int(
            replication_labels.sum()
        ),
        "continuous_evidence_definition": (
            "min(50, -log10(max(a_path_p, b_path_p, sobel_p))); "
            "non-estimable candidates receive 0"
        ),
        "fdr_family_definition": protocol["statistics"]["family_definition"],
        "freeze_semantics": {
            "temporal_kg_freeze": False,
            "temporal_freeze_year": None,
            "kg_snapshot_pinning": True,
            "ranking_freeze_verified_before_outcome_access": True,
            "holdout_axis": _holdout_axis(protocol),
        },
        "lock_hashes": {
            "baseline_ranking_lock": locks["baseline_lock_sha256"],
            "neurodiscovery_initial_ranking_lock": locks[
                "neurodiscovery_lock_sha256"
            ],
            "outcome_access_record": _sha256(access_path),
            "generator_evaluation_plan_lock": _sha256(
                evaluation_plan_lock_path
            ),
        },
        "outputs": {
            "results": str(result_path),
            "analysis_rows": str(rows_path),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result_lock = {
        "schema_version": "neurooracle.case2_heldout_result_lock.v2",
        "protocol_freeze_id": manifest["freeze_id"],
        "holdout_axis": _holdout_axis(protocol),
        "results_sha256": _sha256(result_path),
        "results_csv_sha256": _sha256(result_csv_path),
        "analysis_rows_sha256": _sha256(rows_path),
        "manifest_sha256": _sha256(manifest_path),
        "do_not_retune_after_this_lock": True,
    }
    result_lock_path = output_root / _result_lock_name(protocol)
    result_lock_path.write_text(
        json.dumps(result_lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--neurodiscovery-root", type=Path)
    parser.add_argument("--evaluation-plan-lock", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 13:58 HKT
