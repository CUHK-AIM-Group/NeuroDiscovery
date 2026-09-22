from __future__ import annotations

import pandas as pd
import pytest

from neurooracle.scripts.build_case2_adni_multimodal_table import (
    _write_parquet,
    build_outcome_pairs,
    build_rid_subject_map,
    normalize_rid,
)


def test_normalize_rid_handles_numeric_exports() -> None:
    assert normalize_rid(2) == "2"
    assert normalize_rid("2.0") == "2"
    assert normalize_rid(" 0042 ") == "42"
    assert normalize_rid(None) == ""


def test_build_rid_subject_map_rejects_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "RID": [2, 2],
            "PTID": ["011_S_0002", "012_S_0002"],
        }
    )
    with pytest.raises(ValueError, match="multiple PTIDs"):
        build_rid_subject_map([frame])


def test_build_outcome_pairs_uses_nearest_baseline_and_future_values() -> None:
    visits = pd.DataFrame(
        {
            "visit_id": ["amyloid_pet|011_S_0002|20200101"],
            "subject_id": ["011_S_0002"],
            "modality": ["amyloid_pet"],
            "imaging_date": [pd.Timestamp("2020-01-01")],
            "qc_primary_eligible": [True],
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": ["011_S_0002"] * 4,
            "outcome": ["MMSE"] * 4,
            "outcome_date": pd.to_datetime(
                ["2019-08-01", "2020-01-10", "2021-01-10", "2022-01-05"]
            ),
            "value": [29.0, 28.0, 26.0, 25.0],
            "worse_direction": [-1.0] * 4,
        }
    )

    same, longitudinal = build_outcome_pairs(visits, outcomes)

    assert len(same) == 1
    assert same.loc[0, "outcome_date"] == pd.Timestamp("2020-01-10")
    assert same.loc[0, "days_from_imaging"] == 9
    assert len(longitudinal) == 2
    assert longitudinal["future_value"].tolist() == [26.0, 25.0]
    assert longitudinal["decline_score"].tolist() == [2.0, 3.0]


def test_build_outcome_pairs_excludes_nonprimary_qc_visits() -> None:
    visits = pd.DataFrame(
        {
            "visit_id": ["tau_pet|011_S_0002|20200101"],
            "subject_id": ["011_S_0002"],
            "modality": ["tau_pet"],
            "imaging_date": [pd.Timestamp("2020-01-01")],
            "qc_primary_eligible": [False],
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": ["011_S_0002"],
            "outcome": ["MMSE"],
            "outcome_date": [pd.Timestamp("2020-01-01")],
            "value": [28.0],
            "worse_direction": [-1.0],
        }
    )

    same, longitudinal = build_outcome_pairs(visits, outcomes)

    assert same.empty
    assert longitudinal.empty


def test_write_parquet_normalizes_mixed_identifier_types(tmp_path) -> None:
    path = tmp_path / "mixed_identifiers.parquet"
    frame = pd.DataFrame(
        {
            "RID": [2, "3.0"],
            "VISCODE": ["bl", 12],
            "qc_detail": ["Pass", 1],
            "value": [1.0, 2.0],
        }
    )

    _write_parquet(frame, path)
    restored = pd.read_parquet(path)

    assert restored["RID"].tolist() == ["2", "3"]
    assert restored["VISCODE"].tolist() == ["bl", "12"]
    assert restored["qc_detail"].tolist() == ["Pass", "1"]
    assert restored["value"].tolist() == [1.0, 2.0]


# Last Updated At: 2026-07-31 17:52 HKT
