#!/usr/bin/env python3
"""Verify the locked five-window NeuroDiscovery hindcasting design and assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "optimization_protocol_semantic_v4_20260814"
)
DEFAULT_DESIGN = PROTOCOL_ROOT / "formal_hindcasting_design_v2_five_windows.json"
DEFAULT_LOCK = PROTOCOL_ROOT / "formal_hindcasting_design_v2_five_windows.lock.json"

EXPECTED_METHODS = ["neurodiscovery", "sciagents", "openscholar_rag"]
EXPECTED_TASKS = [
    "case1_transdiagnostic",
    "biomarker_discovery",
    "differential_diagnosis",
    "connectome_behavior",
]
EXPECTED_WINDOWS = [
    (2016, 2017, 2021, 2017, 2018, 2019, 2021),
    (2017, 2018, 2022, 2018, 2019, 2020, 2022),
    (2018, 2019, 2023, 2019, 2020, 2021, 2023),
    (2019, 2020, 2024, 2020, 2021, 2022, 2024),
    (2020, 2021, 2025, 2021, 2022, 2023, 2025),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_hash(path: Path, expected: str, checked: list[dict[str, Any]]) -> None:
    path = path.resolve()
    for row in checked:
        if row["path"] == str(path):
            require(
                row["sha256"] == expected.upper(),
                f"conflicting registered SHA256 for {path}",
            )
            return
    require(path.is_file(), f"missing locked asset: {path}")
    actual = sha256(path)
    require(actual == expected.upper(), f"SHA256 mismatch for {path}: {actual}")
    checked.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})


def resolve_from_protocol(raw_path: str) -> Path:
    return (PROTOCOL_ROOT / raw_path).resolve()


def resolve_locked_path(asset: dict[str, Any]) -> Path:
    path_base = asset.get("path_base", "protocol_root")
    roots = {
        "protocol_root": PROTOCOL_ROOT,
        "repo_root": REPO_ROOT,
    }
    require(path_base in roots, f"unsupported locked path base: {path_base}")
    return (roots[path_base] / asset["path"]).resolve()


def verify_source_bundle(
    manifest_path: Path, checked: list[dict[str, Any]]
) -> None:
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema_version")
        == "neurodiscovery-experiment-source-bundle.v1",
        f"unexpected source-bundle schema: {manifest_path}",
    )
    bundle_root = manifest_path.parent.resolve()
    for asset in manifest.get("files", []):
        path = (bundle_root / asset["bundle_path"]).resolve()
        require(
            path.is_relative_to(bundle_root),
            f"source-bundle path escapes bundle root: {path}",
        )
        verify_hash(path, asset["sha256"], checked)
    for asset in manifest.get("references", []):
        path = Path(asset["path"]).resolve()
        require(
            path.is_relative_to(REPO_ROOT),
            f"source-bundle reference escapes repository: {path}",
        )
        verify_hash(path, asset["sha256"], checked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--lock", dest="lock_path", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design_path = args.design.resolve()
    lock_path = args.lock_path.resolve()
    design = load_json(design_path)
    lock = load_json(lock_path)
    checked: list[dict[str, Any]] = []

    require(lock["status"] == "locked", "design lock is not active")
    require(
        design["status"] == "locked_before_2016_2018_execution",
        "design status is not locked",
    )
    require(
        design_path.name == lock["design_path"],
        "lock points to a different design filename",
    )
    verify_hash(design_path, lock["design_sha256"], checked)

    require(design["methods"]["primary"] == EXPECTED_METHODS, "method set changed")
    require(
        design["task_scope"]["primary_balanced"] == EXPECTED_TASKS,
        "primary task set changed",
    )
    windows = [
        (
            row["freeze_year"],
            row["future_start_year"],
            row["future_end_year"],
            row["feedback_start_year"],
            row["feedback_end_year"],
            row["terminal_start_year"],
            row["terminal_end_year"],
        )
        for row in design["temporal_windows"]
    ]
    require(windows == EXPECTED_WINDOWS, "temporal windows changed")
    require(design["replication"]["seeds"] == list(range(10)), "seed set changed")
    require(
        design["replication"]["experiment_counts"]
        == [10, 20, 50, 100, 200, 500, 1000],
        "experiment-count grid changed",
    )
    computed_runs = (
        len(EXPECTED_TASKS)
        * len(EXPECTED_WINDOWS)
        * len(EXPECTED_METHODS)
        * len(design["replication"]["seeds"])
    )
    require(
        computed_runs == design["run_matrix"]["total_method_runs"] == 600,
        "run-matrix total changed",
    )

    verify_hash(
        resolve_from_protocol(design["supersedes"]["design_path"]),
        design["supersedes"]["design_sha256"],
        checked,
    )
    for key in ("knowledge_graph", "extracted_claims"):
        asset = design["canonical_release"][key]
        verify_hash(resolve_from_protocol(asset["path"]), asset["sha256"], checked)
    verify_hash(
        resolve_from_protocol(design["selection_lock"]["path"]),
        design["selection_lock"]["sha256"],
        checked,
    )
    verify_hash(
        resolve_from_protocol(design["selection_lock"]["protocol_path"]),
        design["selection_lock"]["protocol_sha256"],
        checked,
    )
    source_bundles = design["source_bundles"]
    require(
        set(source_bundles) == {"neurodiscovery", "baselines", "rule"},
        "source-bundle set changed",
    )
    for key in ("neurodiscovery", "baselines"):
        asset = source_bundles[key]
        manifest_path = resolve_locked_path(asset)
        verify_hash(manifest_path, asset["sha256"], checked)
        verify_source_bundle(manifest_path, checked)
    eligibility = design["dynamic_eligibility_lock"]
    verify_hash(
        resolve_from_protocol(eligibility["manifest_path"]),
        eligibility["manifest_sha256"],
        checked,
    )
    verify_hash(
        resolve_from_protocol(eligibility["matrix_path"]),
        eligibility["matrix_sha256"],
        checked,
    )
    for asset in design["historical_assets"]:
        verify_hash(
            resolve_from_protocol(asset["snapshot_manifest"]),
            asset["snapshot_manifest_sha256"],
            checked,
        )
        verify_hash(
            resolve_from_protocol(asset["kge_checkpoint"]),
            asset["kge_sha256"],
            checked,
        )
    reuse = design["reusable_v1_results"]
    for key in (
        "neurodiscovery_part_x_manifest",
        "neurodiscovery_part_y_manifest",
        "baseline_generation_manifest",
        "baseline_completion_audit",
        "baseline_evaluation_manifest",
    ):
        asset = reuse[key]
        verify_hash(resolve_from_protocol(asset["path"]), asset["sha256"], checked)

    print(
        json.dumps(
            {
                "status": "passed",
                "design_id": design["design_id"],
                "design_sha256": lock["design_sha256"],
                "methods": EXPECTED_METHODS,
                "tasks": EXPECTED_TASKS,
                "windows": len(EXPECTED_WINDOWS),
                "seeds": len(design["replication"]["seeds"]),
                "total_method_runs": computed_runs,
                "locked_assets_checked": len(checked),
                "locked_asset_bytes_checked": sum(row["bytes"] for row in checked),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
