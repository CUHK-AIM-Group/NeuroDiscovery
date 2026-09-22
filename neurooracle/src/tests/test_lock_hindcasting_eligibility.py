from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neurooracle.scripts.lock_hindcasting_eligibility import lock_eligibility
from neurooracle.src.case_studies import list_case_study_names
from neurooracle.src.hindcasting_eligibility import load_locked_hindcasting_eligibility


def _audit(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index, case_id in enumerate(list_case_study_names()):
        rows.append(
            {
                "case_study_id": case_id,
                "freeze_year": "2020",
                "future_start_year": "2021",
                "future_end_year": "2025",
                "structural_status": (
                    "executable" if index == 0 else "sparse" if index == 1 else "non_executable"
                ),
            }
        )
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "rows": rows,
                "status_counts": {
                    "executable": 1,
                    "sparse": 1,
                    "non_executable": len(rows) - 2,
                },
                "evaluation_contract": {
                    "semantic_endpoint_identity_version": "test.v1"
                },
            }
        ),
        encoding="utf-8",
    )
    return audit, matrix


def test_lock_is_loadable_and_method_blind(tmp_path: Path) -> None:
    audit, matrix = _audit(tmp_path)
    output = tmp_path / "locked"
    manifest = lock_eligibility(
        audit_json=audit,
        audit_matrix=matrix,
        output_dir=output,
    )

    loaded = load_locked_hindcasting_eligibility(
        output / "eligibility_manifest.json"
    )
    first, second = list_case_study_names()[:2]
    assert loaded.primary_case_study_ids == (first,)
    assert loaded.permits(first, 2020, 2021, 2025)
    assert not loaded.permits(second, 2020, 2021, 2025)
    assert manifest["selection_contract"]["performance_columns_consumed"] is False
    assert manifest["exploratory_case_study_ids"] == [second]
    assert manifest["source"]["evaluation_contract"] == {
        "semantic_endpoint_identity_version": "test.v1"
    }


def test_lock_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    audit, matrix = _audit(tmp_path)
    output = tmp_path / "locked"
    output.mkdir()
    with pytest.raises(FileExistsError):
        lock_eligibility(
            audit_json=audit,
            audit_matrix=matrix,
            output_dir=output,
        )
