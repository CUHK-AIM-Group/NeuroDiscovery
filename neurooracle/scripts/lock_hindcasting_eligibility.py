"""Lock a method-blind executability audit into immutable analysis tiers."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.src.case_studies import list_case_study_names
from neurooracle.src.hindcasting_eligibility import sha256_file


SCHEMA = "neurodiscovery-locked-hindcasting-eligibility.v2"
TIER_BY_STATUS = {
    "executable": "primary",
    "sparse": "exploratory",
    "non_executable": "excluded",
}


def _window(row: Mapping[str, Any]) -> dict[str, int]:
    return {
        "freeze_year": int(row["freeze_year"]),
        "future_start_year": int(row["future_start_year"]),
        "future_end_year": int(row["future_end_year"]),
    }


def _group_windows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, int]]]]:
    grouped: dict[str, dict[str, list[dict[str, int]]]] = {
        tier: {} for tier in ("primary", "exploratory", "excluded")
    }
    for row in rows:
        tier = str(row["analysis_tier"])
        grouped[tier].setdefault(str(row["case_study_id"]), []).append(_window(row))
    return grouped


def lock_eligibility(
    *,
    audit_json: Path,
    audit_matrix: Path,
    output_dir: Path,
    status: str = "locked_before_formal_method_comparison",
) -> dict[str, Any]:
    """Copy one audit matrix, assign tiers, and seal all source hashes."""

    audit_json = audit_json.resolve()
    audit_matrix = audit_matrix.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"eligibility lock is immutable and already exists: {output_dir}"
        )
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    with audit_matrix.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(audit.get("rows") or ()):
        raise ValueError("audit JSON and matrix contain different row counts")

    formal_ids = tuple(list_case_study_names())
    observed_ids = {str(row["case_study_id"]) for row in rows}
    if observed_ids != set(formal_ids):
        raise ValueError(
            "audit does not cover the full formal registry: "
            f"missing={sorted(set(formal_ids) - observed_ids)} "
            f"extra={sorted(observed_ids - set(formal_ids))}"
        )
    for row in rows:
        structural_status = str(row.get("structural_status") or "")
        try:
            row["analysis_tier"] = TIER_BY_STATUS[structural_status]
        except KeyError as exc:
            raise ValueError(f"unknown structural status: {structural_status!r}") from exc

    output_dir.mkdir(parents=True)
    matrix_path = output_dir / "eligibility_matrix_locked.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_copy = output_dir / "executability_audit.json"
    shutil.copyfile(audit_json, audit_copy)

    tier_counts = {
        tier: sum(row["analysis_tier"] == tier for row in rows)
        for tier in ("primary", "exploratory", "excluded")
    }
    cases_by_tier = {
        tier: [
            case_id
            for case_id in formal_ids
            if any(
                row["case_study_id"] == case_id and row["analysis_tier"] == tier
                for row in rows
            )
        ]
        for tier in tier_counts
    }
    manifest = {
        "schema_version": SCHEMA,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "interpretation": (
            "Method-blind structural eligibility for locked historical "
            "re-evaluation; previously exposed years are not an untouched test."
        ),
        "source": {
            "executability_audit_json": str(audit_json),
            "executability_audit_json_sha256": sha256_file(audit_json),
            "executability_matrix_csv": str(audit_matrix),
            "executability_matrix_csv_sha256": sha256_file(audit_matrix),
            "canonical_release": audit.get("canonical_release"),
            "snapshot_root": audit.get("snapshot_root"),
            "evaluation_contract": audit.get("evaluation_contract"),
        },
        "selection_contract": {
            "performance_columns_consumed": False,
            "primary_rule": "structural_status == executable",
            "exploratory_rule": "structural_status == sparse",
            "exclusion_rule": "structural_status == non_executable",
            "task_or_window_selection_after_method_scoring_permitted": False,
            "primary_aggregation_guard": (
                "Only primary-tier rows may enter the main aggregate."
            ),
        },
        "formal_case_study_ids": list(formal_ids),
        "formal_case_studies": len(formal_ids),
        "common_temporal_windows": [
            _window(row)
            for row in rows
            if row["case_study_id"] == formal_ids[0]
        ],
        "status_counts": audit.get("status_counts"),
        "primary_case_study_ids": cases_by_tier["primary"],
        "primary_case_studies": len(cases_by_tier["primary"]),
        "primary_windows": tier_counts["primary"],
        "exploratory_case_study_ids": cases_by_tier["exploratory"],
        "exploratory_case_studies": len(cases_by_tier["exploratory"]),
        "exploratory_windows": tier_counts["exploratory"],
        "tasks_without_primary_windows": [
            case_id for case_id in formal_ids if case_id not in cases_by_tier["primary"]
        ],
        "windows_by_case_study": _group_windows(rows),
        "locked_audit": {
            "path": str(audit_copy),
            "sha256": sha256_file(audit_copy),
        },
        "locked_matrix": {
            "path": str(matrix_path),
            "sha256": sha256_file(matrix_path),
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--status", default="locked_before_formal_method_comparison"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = lock_eligibility(
        audit_json=args.audit_json,
        audit_matrix=args.audit_matrix,
        output_dir=args.output_dir,
        status=args.status,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "primary_case_studies": manifest["primary_case_studies"],
                "primary_windows": manifest["primary_windows"],
                "exploratory_windows": manifest["exploratory_windows"],
                "locked_matrix_sha256": manifest["locked_matrix"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
