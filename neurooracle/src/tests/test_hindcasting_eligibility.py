from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
    sha256_file,
)


def _lock(tmp_path: Path) -> Path:
    rows = [
        {
            "case_study_id": "case_a",
            "freeze_year": 2016,
            "future_start_year": 2017,
            "future_end_year": 2021,
            "analysis_tier": "primary",
        },
        {
            "case_study_id": "case_a",
            "freeze_year": 2017,
            "future_start_year": 2018,
            "future_end_year": 2022,
            "analysis_tier": "excluded",
        },
        {
            "case_study_id": "case_b",
            "freeze_year": 2017,
            "future_start_year": 2018,
            "future_end_year": 2022,
            "analysis_tier": "primary",
        },
    ]
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "test.v1",
                "primary_case_study_ids": ["case_a", "case_b"],
                "primary_windows": 2,
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": sha256_file(matrix),
                    "rows": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_load_lock_verifies_hash_and_filters_cartesian_request(tmp_path: Path) -> None:
    eligibility = load_locked_hindcasting_eligibility(_lock(tmp_path))
    windows = [
        SimpleNamespace(freeze_year=2016, future_start_year=2017, future_end_year=2021),
        SimpleNamespace(freeze_year=2017, future_start_year=2018, future_end_year=2022),
    ]

    assert eligibility.permits("case_a", 2016, 2017, 2021)
    assert not eligibility.permits("case_a", 2017, 2018, 2022)
    assert eligibility.selected_windows(
        case_study_ids=["case_a", "case_b"], windows=windows
    ) == {
        ("case_a", 2016, 2017, 2021),
        ("case_b", 2017, 2018, 2022),
    }


def test_load_lock_rejects_tampered_matrix(tmp_path: Path) -> None:
    manifest = _lock(tmp_path)
    matrix = tmp_path / "matrix.csv"
    with matrix.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_locked_hindcasting_eligibility(manifest)


def test_selected_windows_rejects_empty_intersection(tmp_path: Path) -> None:
    eligibility = load_locked_hindcasting_eligibility(_lock(tmp_path))

    with pytest.raises(ValueError, match="do not intersect"):
        eligibility.selected_windows(
            case_study_ids=["case_a"],
            windows=[
                SimpleNamespace(
                    freeze_year=2020,
                    future_start_year=2021,
                    future_end_year=2025,
                )
            ],
        )
