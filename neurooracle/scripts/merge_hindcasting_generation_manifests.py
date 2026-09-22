"""Strictly merge paired frozen-hindcasting generation manifests.

The merger refuses incomplete, unpaired, cross-snapshot, mixed-budget, or
overlapping method inputs. It verifies that every hypothesis file contains the
fixed execution prefix, including explicit zero-credit generation failures.
Ranked candidates beyond that prefix may be retained for audit, but never count
as executed slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


SCHEMA = "paired-hindcasting-generation-manifest.v1"
RUN_FIELDS = (
    "seed",
    "case_study_id",
    "freeze_year",
    "future_start_year",
    "future_end_year",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolved_path(value: Any, root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    candidate = (root / path).resolve()
    if candidate.is_file():
        return candidate
    for ancestor in (root, *root.parents):
        candidate = (ancestor / path).resolve()
        if candidate.is_file():
            return candidate
    return (root / path).resolve()


def _method_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = manifest.get("methods")
    if values:
        return tuple(str(value) for value in values)
    method = str(manifest.get("method") or "")
    return (method,) if method else ()


def _run_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        str(row[field]) if field == "case_study_id" else int(row[field])
        for field in RUN_FIELDS
    )


def _eligibility_identity(manifest: Mapping[str, Any]) -> tuple[str, str]:
    eligibility = manifest.get("eligibility") or {}
    manifest_hash = str(
        manifest.get("eligibility_manifest_sha256")
        or eligibility.get("manifest_sha256")
        or ""
    ).upper()
    matrix_hash = str(
        manifest.get("eligibility_matrix_sha256")
        or eligibility.get("matrix_sha256")
        or ""
    ).upper()
    if not manifest_hash or not matrix_hash:
        raise ValueError("every source manifest must declare locked eligibility hashes")
    return manifest_hash, matrix_hash


def _snapshot_root(manifest: Mapping[str, Any]) -> Path:
    value = str(manifest.get("snapshot_root") or "")
    if not value:
        raise ValueError("every source manifest must declare snapshot_root")
    return Path(value).resolve()


def _verify_hypothesis_slots(
    row: Mapping[str, Any],
    *,
    manifest_root: Path,
    target: int,
) -> tuple[Path, int, int, int]:
    path = _resolved_path(row.get("hypotheses_path"), manifest_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses") or []
    if not isinstance(hypotheses, list):
        raise ValueError(f"hypotheses is not a list: {path}")
    if len(hypotheses) < target:
        raise ValueError(
            f"fixed-budget slot mismatch for {path}: expected_at_least={target} "
            f"observed={len(hypotheses)}"
        )
    identifiers = [str(value.get("id") or "") for value in hypotheses]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"hypothesis without id: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate hypothesis ids: {path}")
    execution_prefix = hypotheses[:target]
    failures = sum(
        value.get("hypothesis_type") == "generation_failure"
        or bool((value.get("metadata") or {}).get("generation_failure"))
        for value in execution_prefix
    )
    return path, target, failures, len(hypotheses)


def merge_generation_manifests(
    manifest_paths: Sequence[Path],
    output_dir: Path,
    eligibility_manifest: Path | None = None,
) -> dict[str, Any]:
    if len(manifest_paths) < 2:
        raise ValueError("at least two generation manifests are required")

    sources: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in manifest_paths:
        path = raw_path.resolve()
        if path.is_dir():
            path = path / "generation_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append((path, json.loads(path.read_text(encoding="utf-8"))))

    targets = {int(manifest.get("target_per_case_study") or 0) for _, manifest in sources}
    if len(targets) != 1 or next(iter(targets)) < 1:
        raise ValueError(f"source manifests use different fixed budgets: {sorted(targets)}")
    target = targets.pop()

    snapshot_roots = {_snapshot_root(manifest) for _, manifest in sources}
    if len(snapshot_roots) != 1:
        raise ValueError(f"source manifests use different snapshots: {snapshot_roots}")
    snapshot_root = snapshot_roots.pop()

    eligibility_ids = {_eligibility_identity(manifest) for _, manifest in sources}
    if len(eligibility_ids) != 1:
        raise ValueError("source manifests use different locked eligibility inputs")
    eligibility_manifest_hash, eligibility_matrix_hash = eligibility_ids.pop()

    if eligibility_manifest is None:
        declared_paths = set()
        for manifest_path, manifest in sources:
            eligibility = manifest.get("eligibility") or {}
            value = (
                manifest.get("eligibility_manifest")
                or eligibility.get("manifest_path")
            )
            if value:
                declared_paths.add(_resolved_path(value, manifest_path.parent))
        if len(declared_paths) != 1:
            raise ValueError(
                "eligibility manifest must be supplied explicitly when source "
                "manifests do not resolve to one common path"
            )
        eligibility_manifest = declared_paths.pop()
    locked_eligibility = load_locked_hindcasting_eligibility(
        eligibility_manifest.resolve()
    )
    if _sha256(locked_eligibility.manifest_path) != eligibility_manifest_hash:
        raise ValueError("eligibility manifest hash differs from source manifests")
    if locked_eligibility.matrix_sha256 != eligibility_matrix_hash:
        raise ValueError("eligibility matrix hash differs from source manifests")

    methods: list[str] = []
    rows: list[dict[str, Any]] = []
    matrix_by_method: dict[str, set[tuple[Any, ...]]] = {}
    audit_by_method: dict[str, dict[str, int]] = {}
    seen_run_keys: set[tuple[Any, ...]] = set()

    for manifest_path, manifest in sources:
        declared_methods = _method_names(manifest)
        if not declared_methods:
            raise ValueError(f"manifest declares no methods: {manifest_path}")
        overlap = set(methods) & set(declared_methods)
        if overlap:
            raise ValueError(f"method appears in multiple source manifests: {sorted(overlap)}")
        methods.extend(declared_methods)
        for method in declared_methods:
            matrix_by_method[method] = set()
            audit_by_method[method] = {
                "runs": 0,
                "executed_slots": 0,
                "failure_slots": 0,
                "stored_candidates": 0,
                "candidate_tail_beyond_budget": 0,
            }

        for raw_row in manifest.get("runs") or ():
            row = dict(raw_row)
            method = str(row.get("method") or "")
            if method not in declared_methods:
                raise ValueError(
                    f"run method {method!r} is not declared by {manifest_path}"
                )
            key = _run_key(row)
            full_key = (method, *key)
            if full_key in seen_run_keys:
                raise ValueError(f"duplicate generation run: {full_key}")
            seen_run_keys.add(full_key)
            path, slots, failures, candidate_pool_size = _verify_hypothesis_slots(
                row,
                manifest_root=manifest_path.parent,
                target=target,
            )
            row["hypotheses_path"] = str(path)
            row["n_hypotheses"] = slots
            row["candidate_pool_size"] = candidate_pool_size
            row["candidate_tail_beyond_budget"] = candidate_pool_size - slots
            row["generation_failure_slots"] = failures
            rows.append(row)
            matrix_by_method[method].add(key)
            audit_by_method[method]["runs"] += 1
            audit_by_method[method]["executed_slots"] += slots
            audit_by_method[method]["failure_slots"] += failures
            audit_by_method[method]["stored_candidates"] += candidate_pool_size
            audit_by_method[method]["candidate_tail_beyond_budget"] += (
                candidate_pool_size - slots
            )

    reference_method = methods[0]
    reference_matrix = matrix_by_method[reference_method]
    if not reference_matrix:
        raise ValueError("source manifests contain no runs")
    for method in methods[1:]:
        observed = matrix_by_method[method]
        if observed != reference_matrix:
            raise ValueError(
                f"unpaired generation matrix for {method}: "
                f"missing={sorted(reference_matrix - observed)[:3]} "
                f"extra={sorted(observed - reference_matrix)[:3]}"
            )

    seeds = sorted({int(key[0]) for key in reference_matrix})
    expected_matrix = {
        (seed, case_id, freeze, start, end)
        for seed in seeds
        for case_id, freeze, start, end in locked_eligibility.primary_windows
    }
    if reference_matrix != expected_matrix:
        raise ValueError(
            "generation matrix is incomplete relative to locked primary eligibility: "
            f"missing={sorted(expected_matrix - reference_matrix)[:3]} "
            f"extra={sorted(reference_matrix - expected_matrix)[:3]}"
        )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generation_manifest.json"
    result = {
        "schema_version": SCHEMA,
        "snapshot_root": str(snapshot_root),
        "methods": sorted(methods),
        "seeds": seeds,
        "case_studies": sorted({str(key[1]) for key in reference_matrix}),
        "windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(
                {(int(key[2]), int(key[3]), int(key[4])) for key in reference_matrix}
            )
        ],
        "target_per_case_study": target,
        "eligibility_manifest_sha256": eligibility_manifest_hash,
        "eligibility_matrix_sha256": eligibility_matrix_hash,
        "eligibility_manifest": str(locked_eligibility.manifest_path),
        "eligibility_matrix": str(locked_eligibility.matrix_path),
        "paired_case_windows_per_seed": len(reference_matrix)
        // len({int(key[0]) for key in reference_matrix}),
        "audit_by_method": audit_by_method,
        "source_manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in sources
        ],
        "runs": sorted(
            rows,
            key=lambda row: (
                str(row["method"]),
                int(row["seed"]),
                str(row["case_study_id"]),
                int(row["freeze_year"]),
            ),
        ),
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eligibility-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_generation_manifests(
        args.manifest,
        args.output_dir,
        eligibility_manifest=args.eligibility_manifest,
    )
    print(
        json.dumps(
            {
                "output": str((args.output_dir / "generation_manifest.json").resolve()),
                "methods": result["methods"],
                "runs": len(result["runs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
