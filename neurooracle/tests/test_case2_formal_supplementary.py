from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurooracle.scripts.case2_formal_supplementary import (
    ROW_KEY_COLUMNS,
    _fast_bootstrap_path_coefficients,
    _lstsq_path_coefficients,
    _prepare_bootstrap_arrays,
    bh_adjust_fixed,
    build_master_registry,
    build_outcome_blind_analysis_keys,
    build_readiness_audit,
    collapse_sparse_categories,
    collapse_sparse_sites,
    complete_candidate_frame,
    fit_candidate_model,
    genetic_pc_eligibility,
    load_protocol,
    orient_outcome,
)
from neurooracle.scripts.run_case2_adni_formal_supplementary import (
    run_formal_analysis,
)


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "case2_adni_formal_supplementary_v2.json"
)


def _protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH)


def test_master_registry_is_exact_fixed_168_universe() -> None:
    registry = build_master_registry(_protocol())

    assert len(registry) == 168
    assert registry["candidate_id"].is_unique
    assert registry["pathway_id"].nunique() == 7
    assert registry["marker_spec"].nunique() == 8
    assert set(registry["outcome"]) == {"ADAS13", "FAQ", "LDELTOTAL"}
    selected = dict(
        registry[["pathway_name", "threshold_label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert selected["Myelin"] == "p5em02"
    assert selected["Synaptic"] == "p5em02"
    assert all(
        selected[name] == "p1em03"
        for name in selected
        if name not in {"Myelin", "Synaptic"}
    )


def test_date_pairing_prioritizes_target_then_baseline_then_imaging() -> None:
    protocol = _protocol()
    baseline_early = pd.Timestamp("2020-01-02")
    baseline_late = pd.Timestamp("2020-07-03")
    markers = pd.DataFrame(
        {
            "visit_id": ["v_early", "v_late"],
            "subject_id": ["S1", "S1"],
            "modality": ["amyloid_pet", "amyloid_pet"],
            "marker": ["CENTILOIDS", "CENTILOIDS"],
            "imaging_date": pd.to_datetime(["2020-01-01", "2020-07-01"]),
            "qc_primary_eligible": [True, True],
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": ["S1"] * 4,
            "outcome": ["ADAS13"] * 4,
            "outcome_date": [
                baseline_early,
                baseline_late,
                baseline_early + pd.Timedelta(days=729),
                baseline_late + pd.Timedelta(days=730),
            ],
        }
    )

    result = build_outcome_blind_analysis_keys(markers, outcomes, protocol)

    assert len(result) == 1
    assert result.loc[0, "visit_id"] == "v_late"
    assert result.loc[0, "baseline_outcome_date"] == baseline_late
    assert result.loc[0, "baseline_distance_days"] == 2
    assert result.loc[0, "exact_followup_days"] == 730
    assert result.loc[0, "target_distance_days"] == 0


def test_readiness_gate_drops_whole_seven_pathway_family() -> None:
    protocol = _protocol()
    protocol["readiness_gate"]["minimum_complete_case_n"] = 2
    registry = build_master_registry(protocol)
    subjects_ids = ["S1", "S2", "S3"]
    rows = []
    marker_rows = []
    for marker in protocol["imaging_markers"]:
        marker_spec = f"{marker['modality']}::{marker['marker']}"
        for subject_id in subjects_ids:
            visit_id = f"{marker_spec}:{subject_id}"
            marker_rows.append(
                {
                    "visit_id": visit_id,
                    "modality": marker["modality"],
                    "marker": marker["marker"],
                    "qc_primary_eligible": True,
                }
            )
            if marker["modality"] == "smri_adnimerge":
                marker_rows.append(
                    {
                        "visit_id": visit_id,
                        "modality": "smri_adnimerge",
                        "marker": "ICV",
                        "qc_primary_eligible": True,
                    }
                )
            for outcome in protocol["clinical_outcomes"]:
                excluded_family = (
                    marker_spec == "amyloid_pet::CENTILOIDS"
                    and outcome["outcome"] == "ADAS13"
                )
                if excluded_family and subject_id != "S1":
                    continue
                imaging_date = pd.Timestamp("2020-01-01")
                baseline_date = pd.Timestamp("2020-01-02")
                rows.append(
                    {
                        "subject_id": subject_id,
                        "visit_id": visit_id,
                        "modality": marker["modality"],
                        "marker": marker["marker"],
                        "marker_spec": marker_spec,
                        "outcome": outcome["outcome"],
                        "readiness_family_id": f"{marker_spec}::{outcome['outcome']}",
                        "imaging_date": imaging_date,
                        "baseline_outcome_date": baseline_date,
                        "future_outcome_date": baseline_date + pd.Timedelta(days=730),
                        "baseline_distance_days": 1,
                        "exact_followup_days": 730,
                        "target_distance_days": 0,
                    }
                )
    row_keys = pd.DataFrame(rows).loc[:, list(ROW_KEY_COLUMNS)]
    markers = pd.DataFrame(marker_rows)
    subjects = pd.DataFrame(
        {
            "subject_id": subjects_ids,
            "AGE": [70.0, 71.0, 72.0],
            "sex_binary": [0, 1, 0],
            "PTEDUCAT": [16.0, 14.0, 18.0],
            "SITE": [1, 1, 2],
            "COLPROT": ["ADNI1", "ADNI1", "ADNI2"],
            "batch": ["b1", "b1", "b2"],
            **{f"PC{index}": [0.0, 0.1, -0.1] for index in range(1, 11)},
            **{
                pathway["score_name"]: [0.1, 0.2, 0.3]
                for pathway in protocol["pathway_exposures"]
            },
        }
    )
    clinical_dates = pd.DataFrame(
        {
            "subject_id": subjects_ids,
            "clinical_date": pd.to_datetime(["2019-01-01"] * 3),
        }
    )

    family, candidates, executable, executable_keys = build_readiness_audit(
        row_keys,
        markers,
        subjects,
        clinical_dates,
        registry,
        protocol,
    )

    failed = family[
        family["readiness_family_id"].eq("amyloid_pet::CENTILOIDS::ADAS13")
    ].iloc[0]
    assert failed["min_complete_case_n"] == 1
    assert not bool(failed["executable"])
    assert len(candidates) == 168
    assert len(executable) == 161
    assert (
        not executable_keys["readiness_family_id"]
        .eq("amyloid_pet::CENTILOIDS::ADAS13")
        .any()
    )


def test_corrected_model_keeps_baseline_and_followup_out_of_path_a() -> None:
    rng = np.random.default_rng(42)
    n = 180
    exposure = rng.normal(size=n)
    age = rng.normal(72, 5, size=n)
    mediator = 0.55 * exposure + 0.08 * (age - age.mean()) + rng.normal(size=n)
    baseline = rng.normal(size=n)
    followup = rng.integers(365, 1096, size=n)
    future = (
        0.60 * mediator
        + 0.20 * exposure
        + 0.50 * baseline
        + 0.001 * (followup - 730)
        + rng.normal(size=n)
    )
    frame = pd.DataFrame(
        {
            "exposure": exposure,
            "mediator": mediator,
            "future_outcome": future,
            "baseline_outcome": baseline,
            "exact_followup_days": followup,
            "age_at_imaging": age,
            "sex_binary": rng.integers(0, 2, size=n),
            "PTEDUCAT": rng.normal(16, 2, size=n),
            "SITE": rng.choice([1, 2, 3], size=n),
            "COLPROT": rng.choice(["ADNI1", "ADNI2"], size=n),
            "batch": rng.choice(["b1", "b2"], size=n),
            **{f"PC{index}": rng.normal(size=n) for index in range(1, 11)},
        }
    )

    result = fit_candidate_model(
        frame,
        include_icv=False,
        bootstrap_replicates=20,
        bootstrap_seed=20260826,
        candidate_id="synthetic",
    )

    assert "baseline_outcome_z" not in result["path_a_design_columns"]
    assert "exact_followup_days_z" not in result["path_a_design_columns"]
    assert "baseline_outcome_z" in result["path_b_design_columns"]
    assert "exact_followup_days_z" in result["path_b_design_columns"]
    assert result["a_path_std"] > 0
    assert result["b_path_std"] > 0
    assert result["bootstrap_valid_replicates"] == 20
    assert result["master_seed_count"] == 1
    assert result["master_bootstrap_seed"] == 20260826
    assert np.isfinite(result["a_path_hc3_se"])
    assert np.isfinite(result["b_path_hc3_se"])
    assert result["max_leverage_path_a"] < 1.0 - 1e-10
    assert result["max_leverage_path_b"] < 1.0 - 1e-10

    complete = complete_candidate_frame(frame, include_icv=False)
    indices = np.random.default_rng(7).integers(0, len(complete), size=len(complete))
    reference = _lstsq_path_coefficients(
        complete.iloc[indices].reset_index(drop=True), include_icv=False
    )
    optimized = _fast_bootstrap_path_coefficients(
        _prepare_bootstrap_arrays(complete, include_icv=False), indices
    )
    np.testing.assert_allclose(optimized, reference, rtol=1e-10, atol=1e-10)


def test_sparse_sites_are_pooled_before_hc3() -> None:
    frame = pd.DataFrame({"SITE": ["singleton", "large", "large", "large"]})

    collapsed = collapse_sparse_sites(frame, minimum_site_n=3)

    assert collapsed["SITE"].tolist() == [
        "__POOLED_SPARSE_SITE__",
        "large",
        "large",
        "large",
    ]
    categorical = pd.DataFrame(
        {
            "SITE": ["s1", "s1", "s1", "s2"],
            "COLPROT": ["ADNI1", "ADNI1", "ADNI1", "ADNI4"],
            "batch": ["b1", "b1", "b1", "b2"],
        }
    )
    pooled = collapse_sparse_categories(categorical, minimum_category_n=3)
    assert pooled.loc[3, "COLPROT"] == "ADNI1"
    assert pooled.loc[3, "batch"] == "b1"


def test_combined_pca_gate_excludes_only_gross_outlier() -> None:
    rng = np.random.default_rng(11)
    subjects = pd.DataFrame(
        {f"PC{index}": rng.normal(size=101) for index in range(1, 11)}
    )
    subjects.loc[0, "PC2"] = 100.0

    eligible = genetic_pc_eligibility(subjects, absolute_z_threshold=8.0)

    assert int(eligible.sum()) == 100
    assert not bool(eligible.iloc[0])


def test_orientation_and_fixed_denominator_bh() -> None:
    oriented = orient_outcome(pd.Series([10.0, 5.0]), -1.0)
    assert oriented.tolist() == [-10.0, -5.0]
    adjusted = bh_adjust_fixed(pd.Series([0.01, 0.02]), family_size=8)
    np.testing.assert_allclose(adjusted, [0.08, 0.08])


def test_formal_runner_is_fail_closed_before_lock_or_outcome_access(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        readiness_lock=tmp_path / "missing.lock.json",
        output_root=None,
        confirm_outcome_access_after_lock=False,
    )
    with pytest.raises(PermissionError, match="fail-closed"):
        run_formal_analysis(args)
