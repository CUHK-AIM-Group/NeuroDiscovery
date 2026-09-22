"""Load and verify immutable method-blind hindcasting eligibility locks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


WindowKey = tuple[str, int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_manifest_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    for ancestor in (root, *root.parents):
        candidate = ancestor / path
        if candidate.is_file():
            return candidate.resolve()
    return (root / path).resolve()


def window_key(row: Mapping[str, Any]) -> WindowKey:
    return (
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


@dataclass(frozen=True)
class LockedHindcastingEligibility:
    manifest_path: Path
    matrix_path: Path
    matrix_sha256: str
    manifest: Mapping[str, Any]
    primary_windows: frozenset[WindowKey]
    primary_case_study_ids: tuple[str, ...]

    def permits(
        self,
        case_study_id: str,
        freeze_year: int,
        future_start_year: int,
        future_end_year: int,
    ) -> bool:
        return (
            case_study_id,
            int(freeze_year),
            int(future_start_year),
            int(future_end_year),
        ) in self.primary_windows

    def selected_windows(
        self,
        *,
        case_study_ids: Iterable[str],
        windows: Iterable[Any],
    ) -> frozenset[WindowKey]:
        requested_cases = set(case_study_ids)
        requested_windows = {
            (
                int(window.freeze_year),
                int(window.future_start_year),
                int(window.future_end_year),
            )
            for window in windows
        }
        selected = frozenset(
            key
            for key in self.primary_windows
            if key[0] in requested_cases and key[1:] in requested_windows
        )
        if not selected:
            raise ValueError(
                "requested Case Studies and windows do not intersect locked "
                "primary eligibility"
            )
        return selected


def load_locked_hindcasting_eligibility(
    manifest_path: Path,
) -> LockedHindcastingEligibility:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked = manifest.get("locked_matrix") or {}
    matrix_path = resolve_manifest_path(
        str(locked.get("path") or ""), manifest_path.parent
    )
    if not matrix_path.is_file():
        raise ValueError(f"locked eligibility matrix is absent: {matrix_path}")
    expected_hash = str(locked.get("sha256") or "").upper()
    observed_hash = sha256_file(matrix_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(
            "locked eligibility matrix hash mismatch: "
            f"expected={expected_hash or 'MISSING'} observed={observed_hash}"
        )

    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    primary_rows = [row for row in rows if row.get("analysis_tier") == "primary"]
    primary = frozenset(window_key(row) for row in primary_rows)
    if len(primary) != len(primary_rows):
        raise ValueError("locked eligibility matrix contains duplicate primary windows")
    expected_count = int(manifest.get("primary_windows") or 0)
    if len(primary) != expected_count:
        raise ValueError(
            "locked eligibility primary-window count mismatch: "
            f"manifest={expected_count} matrix={len(primary)}"
        )

    declared_cases = tuple(
        str(value) for value in (manifest.get("primary_case_study_ids") or ())
    )
    observed_cases = {key[0] for key in primary}
    if set(declared_cases) != observed_cases:
        raise ValueError(
            "locked eligibility primary Case Study mismatch: "
            f"missing={sorted(set(declared_cases) - observed_cases)} "
            f"extra={sorted(observed_cases - set(declared_cases))}"
        )
    return LockedHindcastingEligibility(
        manifest_path=manifest_path,
        matrix_path=matrix_path,
        matrix_sha256=observed_hash,
        manifest=manifest,
        primary_windows=primary,
        primary_case_study_ids=declared_cases,
    )
