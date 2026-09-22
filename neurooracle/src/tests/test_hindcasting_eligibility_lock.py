from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_hindcasting_eligibility import freeze_eligibility


HASHES = {
    "knowledge_graph": "89E40DD8096C028177FEE7CE5C8979768157D458F3CDDA9896B8A8B6A1DC9255",
    "extracted_claims": "C1444FA51A144386320E9F1A4F00ED2D203DC2482615EF9C954FDC2605EE4297",
    "current_state": "58B28B222271B5C1669A7215CCCD67BEEF47F0A303084152BA260895187BAC8B",
}


def _release() -> dict:
    return {
        "files": {
            name: {"path": name, "bytes": 1, "sha256": digest}
            for name, digest in HASHES.items()
        }
    }


def _rows() -> list[dict]:
    return [
        {
            "case_study_id": case_id,
            "freeze_year": freeze,
            "future_start_year": freeze + 1,
            "future_end_year": freeze + 2,
            "required_atoms": "imaging_marker,disease",
            "complete_path_required": False,
            "historical_primary_claims": 20,
            "future_unique_pairs": future_pairs,
            "future_complete_components": 1,
            "structural_status": status,
            "status_reason": reason,
        }
        for case_id, statuses in (
            ("case_a", ((2016, "executable", "", 12), (2017, "executable", "", 14))),
            (
                "case_b",
                (
                    (2016, "sparse", "only 5 unique future relations are evaluable", 5),
                    (2017, "non_executable", "no future relation", 0),
                ),
            ),
        )
        for freeze, status, reason, future_pairs in statuses
    ]


def _write_inputs(
    root: Path,
    rows: list[dict],
    *,
    extra_field: tuple[str, object] | None = None,
) -> tuple[Path, Path]:
    matrix = root / "executability_matrix.csv"
    materialized = [dict(row) for row in rows]
    if extra_field is not None:
        for row in materialized:
            row[extra_field[0]] = extra_field[1]
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    audit = root / "executability_audit.json"
    audit.write_text(
        json.dumps(
            {
                "snapshot_root": "snapshots",
                "canonical_release": _release(),
                "status_counts": {
                    status: sum(row["structural_status"] == status for row in rows)
                    for status in ("executable", "sparse", "non_executable")
                    if any(row["structural_status"] == status for row in rows)
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return audit, matrix


def test_freeze_eligibility_separates_primary_exploratory_and_excluded(
    tmp_path: Path,
) -> None:
    audit, matrix = _write_inputs(tmp_path, _rows())

    manifest = freeze_eligibility(
        audit_json=audit,
        matrix_csv=matrix,
        output_dir=tmp_path / "locked",
        canonical_release=_release(),
        expected_case_study_ids=("case_a", "case_b"),
        registered_at="2026-08-13T00:00:00+00:00",
    )

    assert manifest["primary_case_study_ids"] == ["case_a"]
    assert manifest["primary_windows"] == 2
    assert manifest["exploratory_case_study_ids"] == ["case_b"]
    assert manifest["exploratory_windows"] == 1
    assert manifest["tasks_without_primary_windows"] == ["case_b"]
    assert manifest["selection_contract"]["performance_columns_consumed"] is False
    assert (tmp_path / "locked" / "eligibility_manifest.json").is_file()
    assert (tmp_path / "locked" / "eligibility_matrix_locked.csv").is_file()


def test_freeze_eligibility_rejects_duplicate_case_window(tmp_path: Path) -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    audit, matrix = _write_inputs(tmp_path, rows)

    with pytest.raises(ValueError, match="duplicate Case Study/window"):
        freeze_eligibility(
            audit_json=audit,
            matrix_csv=matrix,
            output_dir=tmp_path / "locked",
            canonical_release=_release(),
            expected_case_study_ids=("case_a", "case_b"),
        )


def test_freeze_eligibility_rejects_csv_audit_drift(tmp_path: Path) -> None:
    rows = _rows()
    audit, matrix = _write_inputs(tmp_path, rows)
    changed = list(csv.DictReader(matrix.open("r", encoding="utf-8")))
    changed[0]["structural_status"] = "sparse"
    changed[0]["status_reason"] = "changed after audit"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changed[0]))
        writer.writeheader()
        writer.writerows(changed)

    with pytest.raises(ValueError, match="does not match executability audit JSON"):
        freeze_eligibility(
            audit_json=audit,
            matrix_csv=matrix,
            output_dir=tmp_path / "locked",
            canonical_release=_release(),
            expected_case_study_ids=("case_a", "case_b"),
        )


def test_freeze_eligibility_rejects_method_result_columns(tmp_path: Path) -> None:
    audit, matrix = _write_inputs(tmp_path, _rows(), extra_field=("recall", 0.5))

    with pytest.raises(ValueError, match="method-result columns"):
        freeze_eligibility(
            audit_json=audit,
            matrix_csv=matrix,
            output_dir=tmp_path / "locked",
            canonical_release=_release(),
            expected_case_study_ids=("case_a", "case_b"),
        )


def test_freeze_eligibility_is_idempotent_and_preserves_registration(
    tmp_path: Path,
) -> None:
    audit, matrix = _write_inputs(tmp_path, _rows())
    output = tmp_path / "locked"
    first = freeze_eligibility(
        audit_json=audit,
        matrix_csv=matrix,
        output_dir=output,
        canonical_release=_release(),
        expected_case_study_ids=("case_a", "case_b"),
        registered_at="2026-08-13T00:00:00+00:00",
    )
    second = freeze_eligibility(
        audit_json=audit,
        matrix_csv=matrix,
        output_dir=output,
        canonical_release=_release(),
        expected_case_study_ids=("case_a", "case_b"),
        registered_at="2099-01-01T00:00:00+00:00",
    )

    assert second == first
    assert second["registered_at"] == "2026-08-13T00:00:00+00:00"


def test_freeze_eligibility_rejects_modified_locked_matrix(tmp_path: Path) -> None:
    audit, matrix = _write_inputs(tmp_path, _rows())
    output = tmp_path / "locked"
    freeze_eligibility(
        audit_json=audit,
        matrix_csv=matrix,
        output_dir=output,
        canonical_release=_release(),
        expected_case_study_ids=("case_a", "case_b"),
    )
    with (output / "eligibility_matrix_locked.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="locked eligibility matrix differs"):
        freeze_eligibility(
            audit_json=audit,
            matrix_csv=matrix,
            output_dir=output,
            canonical_release=_release(),
            expected_case_study_ids=("case_a", "case_b"),
        )
