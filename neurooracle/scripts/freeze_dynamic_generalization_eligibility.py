"""Freeze cross-task dynamic eligibility without reading method outcomes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
)
DEFAULT_DYNAMIC_ELIGIBILITY = (
    PROTOCOL_ROOT
    / "locked_dynamic_hindcasting_eligibility_feedback1_endpoint_v7_20260813"
    / "dynamic_eligibility_manifest.json"
)
DEFAULT_POLICY = PROTOCOL_ROOT / "frozen_dynamic_policy.json"
DEFAULT_OUTPUT = (
    PROTOCOL_ROOT
    / "locked_dynamic_generalization_eligibility_feedback1_endpoint_v7_20260813"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields))
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return handle.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def freeze_dynamic_generalization_eligibility(
    *,
    dynamic_eligibility_manifest: Path,
    frozen_policy_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Exclude development tasks using only the frozen policy declaration."""

    source = load_locked_hindcasting_eligibility(dynamic_eligibility_manifest)
    source_status = str(source.manifest.get("status") or "")
    if not source_status.startswith("locked_before_"):
        raise ValueError("source dynamic eligibility is not a frozen lock")
    selection = source.manifest.get("selection_contract") or {}
    if selection.get("performance_columns_consumed") is not False:
        raise ValueError("source dynamic eligibility consumed method outcomes")
    if selection.get("task_or_window_selection_after_method_scoring_permitted") is not False:
        raise ValueError("source dynamic eligibility permits post-outcome selection")

    policy = json.loads(frozen_policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_for_confirmatory_application":
        raise ValueError("dynamic policy is not frozen")
    development_ids = tuple(
        str(value)
        for value in (
            (policy.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    )
    if not development_ids:
        raise ValueError("frozen policy does not declare development Case Studies")
    development = set(development_ids)

    with source.matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields:
        raise ValueError("source dynamic eligibility matrix has no columns")
    declared_rows = int((source.manifest.get("locked_matrix") or {}).get("rows") or 0)
    if declared_rows and declared_rows != len(rows):
        raise ValueError("source dynamic eligibility row count is inconsistent")

    selected = [
        row
        for row in rows
        if row.get("analysis_tier") == "primary"
        and str(row.get("case_study_id") or "") not in development
    ]
    expected_keys = {
        key for key in source.primary_windows if key[0] not in development
    }
    observed_keys = {
        (
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
        for row in selected
    }
    if observed_keys != expected_keys:
        raise ValueError("generalization rows differ from the declared filtering rule")
    if not selected:
        raise ValueError("no cross-task dynamic generalization window remains")

    case_order = {
        case_id: index
        for index, case_id in enumerate(source.primary_case_study_ids)
        if case_id not in development
    }
    selected.sort(
        key=lambda row: (
            case_order[str(row["case_study_id"])],
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
    )
    primary_case_ids = [
        case_id
        for case_id in source.primary_case_study_ids
        if case_id not in development
        and any(str(row["case_study_id"]) == case_id for row in selected)
    ]

    matrix_path = output_dir / "dynamic_generalization_matrix_locked.csv"
    manifest_path = output_dir / "dynamic_generalization_manifest.json"
    matrix_bytes = _csv_bytes(selected, fields)
    matrix_hash = hashlib.sha256(matrix_bytes).hexdigest().upper()
    manifest = {
        "schema_version": "neurodiscovery-dynamic-generalization-eligibility.v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_dynamic_generalization",
        "selection_contract": {
            "performance_columns_consumed": False,
            "source_matrix": "method-blind dynamic primary eligibility only",
            "exclusion_rule": (
                "Exclude exactly the Case Studies named in the frozen policy's "
                "development_protocol.case_study_ids."
            ),
            "task_or_window_selection_after_method_scoring_permitted": False,
            "excluded_case_study_ids": list(development_ids),
        },
        "feedback_years": int(source.manifest.get("feedback_years") or 0),
        "min_early_support": int(source.manifest.get("min_early_support") or 0),
        "min_terminal_pairs": int(source.manifest.get("min_terminal_pairs") or 0),
        "canonical_release": source.manifest.get("canonical_release"),
        "snapshot_root": source.manifest.get("snapshot_root"),
        "source_dynamic_eligibility_manifest": str(source.manifest_path),
        "source_dynamic_eligibility_manifest_sha256": _sha256(source.manifest_path),
        "source_dynamic_eligibility_matrix": str(source.matrix_path),
        "source_dynamic_eligibility_matrix_sha256": source.matrix_sha256,
        "frozen_policy": str(frozen_policy_path.resolve()),
        "frozen_policy_sha256": _sha256(frozen_policy_path),
        "primary_case_study_ids": primary_case_ids,
        "primary_case_studies": len(primary_case_ids),
        "primary_windows": len(selected),
        "locked_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": matrix_hash,
            "rows": len(selected),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        if not matrix_path.is_file() or _sha256(matrix_path) != matrix_hash:
            raise ValueError("existing dynamic generalization lock differs")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["registered_at"] = existing.get("registered_at")
        if existing != manifest:
            raise ValueError("existing dynamic generalization manifest differs")
        return existing
    if matrix_path.exists():
        raise ValueError("generalization matrix exists without its manifest")
    _atomic_write(matrix_path, matrix_bytes)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamic-eligibility-manifest",
        type=Path,
        default=DEFAULT_DYNAMIC_ELIGIBILITY,
    )
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = freeze_dynamic_generalization_eligibility(
        dynamic_eligibility_manifest=args.dynamic_eligibility_manifest.resolve(),
        frozen_policy_path=args.frozen_policy.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "primary_case_studies": result["primary_case_studies"],
                "primary_windows": result["primary_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
