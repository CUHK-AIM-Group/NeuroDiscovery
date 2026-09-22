"""Freeze a method-blind Case Study/window eligibility matrix for hindcasting.

This utility consumes only the structural executability audit.  It deliberately
rejects method-performance columns and never opens a hypothesis or evaluation
artifact, so the benchmark matrix cannot be selected after seeing method scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
)
from neurooracle.src.case_studies import list_case_study_names


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_ROOT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "executability_audit_v2_kgA8C354"
)
DEFAULT_CANONICAL_INPUT = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_OUTPUT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "optimization_protocol_20260812"
    / "locked_hindcasting_eligibility"
)

STATUS_TO_TIER = {
    "executable": "primary",
    "sparse": "exploratory",
    "non_executable": "excluded",
}
KEY_FIELDS = (
    "case_study_id",
    "freeze_year",
    "future_start_year",
    "future_end_year",
)
LOCKED_FIELDS = (
    *KEY_FIELDS,
    "required_atoms",
    "complete_path_required",
    "historical_primary_claims",
    "future_unique_pairs",
    "future_complete_components",
    "structural_status",
    "status_reason",
)
FORBIDDEN_RESULT_COLUMNS = {
    "method",
    "seed",
    "score",
    "metric",
    "hits",
    "recall",
    "precision",
    "performance",
    "p_value",
    "effect_size",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return [dict(row) for row in reader], fields


def _integer(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def _boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in LOCKED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"executability row is missing fields: {missing}")
    status = str(row["structural_status"]).strip()
    if status not in STATUS_TO_TIER:
        raise ValueError(f"unknown structural_status: {status!r}")
    normalized = {
        "case_study_id": str(row["case_study_id"]).strip(),
        "freeze_year": _integer(row["freeze_year"], field="freeze_year"),
        "future_start_year": _integer(
            row["future_start_year"], field="future_start_year"
        ),
        "future_end_year": _integer(row["future_end_year"], field="future_end_year"),
        "required_atoms": str(row["required_atoms"]).strip(),
        "complete_path_required": _boolean(
            row["complete_path_required"], field="complete_path_required"
        ),
        "historical_primary_claims": _integer(
            row["historical_primary_claims"], field="historical_primary_claims"
        ),
        "future_unique_pairs": _integer(
            row["future_unique_pairs"], field="future_unique_pairs"
        ),
        "future_complete_components": _integer(
            row["future_complete_components"], field="future_complete_components"
        ),
        "structural_status": status,
        "status_reason": str(row.get("status_reason") or "").strip(),
        "analysis_tier": STATUS_TO_TIER[status],
    }
    if not (
        normalized["freeze_year"]
        < normalized["future_start_year"]
        <= normalized["future_end_year"]
    ):
        raise ValueError(f"invalid temporal window: {_row_key(normalized)}")
    return normalized


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _release_hashes(release: Mapping[str, Any]) -> dict[str, str]:
    files = release.get("files") or {}
    mapping = {
        "knowledge_graph": "knowledge_graph",
        "extracted_claims": "extracted_claims",
        "current_state": "current_state",
    }
    return {
        output_name: str((files.get(source_name) or {}).get("sha256") or "").upper()
        for output_name, source_name in mapping.items()
    }


def _audit_release_hashes(audit: Mapping[str, Any]) -> dict[str, str]:
    return _release_hashes(audit.get("canonical_release") or {})


def _window_payload(row: Mapping[str, Any]) -> dict[str, int]:
    return {
        "freeze_year": int(row["freeze_year"]),
        "future_start_year": int(row["future_start_year"]),
        "future_end_year": int(row["future_end_year"]),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (*LOCKED_FIELDS, "analysis_tier")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    return handle.getvalue().encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def freeze_eligibility(
    *,
    audit_json: Path,
    matrix_csv: Path,
    output_dir: Path,
    canonical_release: Mapping[str, Any],
    expected_case_study_ids: Sequence[str] | None = None,
    registered_at: str | None = None,
) -> dict[str, Any]:
    """Validate and freeze a structural, method-blind benchmark matrix."""

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    raw_rows, fields = _read_csv(matrix_csv)
    if not raw_rows:
        raise ValueError("executability matrix is empty")
    suspicious = sorted(FORBIDDEN_RESULT_COLUMNS.intersection(fields))
    if suspicious:
        raise ValueError(
            "executability matrix contains method-result columns: "
            f"{suspicious}"
        )

    rows = [_normalize_row(row) for row in raw_rows]
    keys = [_row_key(row) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate Case Study/window rows: {duplicates}")

    expected_ids = tuple(expected_case_study_ids or list_case_study_names())
    observed_ids = {row["case_study_id"] for row in rows}
    if observed_ids != set(expected_ids):
        raise ValueError(
            "executability matrix does not cover the formal Case Study registry: "
            f"missing={sorted(set(expected_ids) - observed_ids)}, "
            f"extra={sorted(observed_ids - set(expected_ids))}"
        )

    all_windows = {
        (
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
        for row in rows
    }
    for case_id in expected_ids:
        case_windows = {
            (
                int(row["freeze_year"]),
                int(row["future_start_year"]),
                int(row["future_end_year"]),
            )
            for row in rows
            if row["case_study_id"] == case_id
        }
        if case_windows != all_windows:
            raise ValueError(
                f"{case_id} does not cover the common temporal window matrix"
            )

    audit_rows = [_normalize_row(row) for row in (audit.get("rows") or [])]
    audit_by_key = {_row_key(row): row for row in audit_rows}
    matrix_by_key = {_row_key(row): row for row in rows}
    if audit_by_key != matrix_by_key:
        raise ValueError("executability CSV does not match executability audit JSON")
    observed_status_counts = dict(Counter(row["structural_status"] for row in rows))
    declared_status_counts = {
        str(key): int(value)
        for key, value in (audit.get("status_counts") or {}).items()
    }
    if observed_status_counts != declared_status_counts:
        raise ValueError("executability status counts do not match the audit manifest")

    release_hashes = _release_hashes(canonical_release)
    if release_hashes != {
        key: value.upper() for key, value in CURRENT_CANONICAL_SHA256.items()
    }:
        raise ValueError("canonical release does not match the frozen KG release")
    if _audit_release_hashes(audit) != release_hashes:
        raise ValueError("executability audit was built from a different KG release")

    case_order = {case_id: index for index, case_id in enumerate(expected_ids)}
    rows.sort(
        key=lambda row: (
            case_order[row["case_study_id"]],
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    locked_matrix = output_dir / "eligibility_matrix_locked.csv"
    locked_matrix_bytes = _csv_bytes(rows)
    locked_matrix_sha256 = _sha256_bytes(locked_matrix_bytes)

    tier_rows = {
        tier: [row for row in rows if row["analysis_tier"] == tier]
        for tier in ("primary", "exploratory", "excluded")
    }
    tier_case_ids = {
        tier: [
            case_id
            for case_id in expected_ids
            if any(row["case_study_id"] == case_id for row in selected)
        ]
        for tier, selected in tier_rows.items()
    }
    windows_by_case = {
        tier: {
            case_id: [
                _window_payload(row)
                for row in selected
                if row["case_study_id"] == case_id
            ]
            for case_id in tier_case_ids[tier]
        }
        for tier, selected in tier_rows.items()
    }
    manifest = {
        "schema_version": "neurodiscovery-locked-hindcasting-eligibility.v1",
        "registered_at": registered_at or datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_new_all_task_re_evaluation",
        "interpretation": (
            "Method-blind structural eligibility for locked historical "
            "re-evaluation; this is not an untouched held-out test."
        ),
        "source": {
            "executability_audit_json": str(audit_json.resolve()),
            "executability_audit_json_sha256": _sha256(audit_json),
            "executability_matrix_csv": str(matrix_csv.resolve()),
            "executability_matrix_csv_sha256": _sha256(matrix_csv),
            "snapshot_root": str(audit.get("snapshot_root") or ""),
            "canonical_release": dict(canonical_release),
        },
        "selection_contract": {
            "performance_columns_consumed": False,
            "primary_rule": "structural_status == executable",
            "exploratory_rule": "structural_status == sparse",
            "exclusion_rule": "structural_status == non_executable",
            "task_or_window_selection_after_method_scoring_permitted": False,
            "primary_aggregation_guard": (
                "Only primary-tier rows may enter the main aggregate; "
                "exploratory and excluded rows must remain separate."
            ),
        },
        "formal_case_study_ids": list(expected_ids),
        "formal_case_studies": len(expected_ids),
        "common_temporal_windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(all_windows)
        ],
        "status_counts": observed_status_counts,
        "primary_case_study_ids": tier_case_ids["primary"],
        "primary_case_studies": len(tier_case_ids["primary"]),
        "primary_windows": len(tier_rows["primary"]),
        "exploratory_case_study_ids": tier_case_ids["exploratory"],
        "exploratory_case_studies": len(tier_case_ids["exploratory"]),
        "exploratory_windows": len(tier_rows["exploratory"]),
        "tasks_without_primary_windows": [
            case_id
            for case_id in expected_ids
            if case_id not in tier_case_ids["primary"]
        ],
        "windows_by_case_study": windows_by_case,
        "locked_matrix": {
            "path": str(locked_matrix.resolve()),
            "sha256": locked_matrix_sha256,
            "rows": len(rows),
        },
        "confirmatory_test": {
            "status": "unassigned",
            "reason": (
                "Previously exposed historical years cannot be relabeled as an "
                "untouched confirmatory endpoint."
            ),
        },
    }
    manifest_path = output_dir / "eligibility_manifest.json"
    if manifest_path.is_file():
        if not locked_matrix.is_file():
            raise ValueError("locked eligibility manifest exists but its matrix is missing")
        if _sha256(locked_matrix) != locked_matrix_sha256:
            raise ValueError("locked eligibility matrix differs from the structural audit")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = dict(manifest)
        expected["registered_at"] = existing.get("registered_at")
        if existing != expected:
            raise ValueError(
                "locked eligibility manifest differs from the current structural design"
            )
        return existing

    if locked_matrix.exists():
        raise ValueError("eligibility matrix exists without a locked manifest")
    _write_bytes_atomic(locked_matrix, locked_matrix_bytes)
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "executability_audit.json",
    )
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "executability_matrix.csv",
    )
    parser.add_argument("--canonical-input", type=Path, default=DEFAULT_CANONICAL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    canonical = validate_canonical_kg_release(
        kg_path=args.canonical_input / "knowledge_graph.json",
        claims_path=args.canonical_input / "extracted_claims.jsonl",
        state_path=args.canonical_input / "CURRENT_STATE.json",
        expected_sha256=CURRENT_CANONICAL_SHA256,
    )
    manifest = freeze_eligibility(
        audit_json=args.audit_json.resolve(),
        matrix_csv=args.matrix_csv.resolve(),
        output_dir=args.output_dir.resolve(),
        canonical_release=canonical,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "primary_case_studies": manifest["primary_case_studies"],
                "primary_windows": manifest["primary_windows"],
                "exploratory_windows": manifest["exploratory_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
