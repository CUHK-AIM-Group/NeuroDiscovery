"""Build release-local snapshots and lock method-blind v3 task eligibility.

Legacy audit modules deliberately pin the historical v2 KG constant.  This
orchestrator keeps those modules backward compatible and injects the hashes from
one immutable v3 release manifest for this run only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.scripts.canonical_kg_release import validate_canonical_kg_release
from neurooracle.scripts.build_temporal_kg_snapshot import (
    build_snapshot,
    snapshot_matches_input,
)
from neurooracle.scripts.manage_hindcasting_v3 import (
    DEFAULT_V3_ROOT,
    sha256_file,
    verify_release,
)
from neurooracle.scripts.run_case_study_hindcasting import DEFAULT_WINDOWS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relocated_validator() -> Callable[..., dict[str, object]]:
    def validate(**kwargs: Any) -> dict[str, object]:
        kwargs["allow_relocated_artifacts"] = True
        return validate_canonical_kg_release(**kwargs)

    return validate


def _build_or_reuse_snapshots(
    *, input_dir: Path, snapshot_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in DEFAULT_WINDOWS:
        output_dir = snapshot_root / f"kg_{window.freeze_year}"
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            _require(
                snapshot_matches_input(manifest, input_dir, window.freeze_year),
                f"existing snapshot does not match immutable release: {output_dir}",
            )
            reused = True
            print(f"[reuse] snapshot KG_{window.freeze_year}", flush=True)
        else:
            _require(
                not output_dir.exists() or not any(output_dir.iterdir()),
                f"partial snapshot directory exists without a manifest: {output_dir}",
            )
            print(f"[build] snapshot KG_{window.freeze_year}", flush=True)
            manifest = build_snapshot(input_dir, output_dir, window.freeze_year)
            reused = False
        records.append(
            {
                "freeze_year": window.freeze_year,
                "directory": str(output_dir.resolve()),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "knowledge_graph_bytes": (output_dir / "knowledge_graph.json").stat().st_size,
                "extracted_claims_bytes": (output_dir / "extracted_claims.jsonl").stat().st_size,
                "reused": reused,
            }
        )
    return records


def _complete_primary_tasks(
    static_manifest: Mapping[str, Any],
    dynamic_manifest: Mapping[str, Any],
    dynamic_manifest_path: Path,
) -> list[str]:
    static_windows = static_manifest.get("windows_by_case_study") or {}
    static_primary = static_windows.get("primary") or {}
    dynamic_record = dynamic_manifest.get("locked_matrix") or {}
    dynamic_path = Path(str(dynamic_record.get("path") or ""))
    if not dynamic_path.is_file():
        dynamic_path = dynamic_manifest_path.parent / dynamic_path.name
    _require(dynamic_path.is_file(), f"missing dynamic eligibility matrix: {dynamic_path}")
    _require(
        sha256_file(dynamic_path)
        == str(dynamic_record.get("sha256") or "").upper(),
        "dynamic eligibility matrix hash mismatch",
    )
    dynamic_years: dict[str, set[int]] = {}
    with dynamic_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("analysis_tier") or "") == "primary":
                dynamic_years.setdefault(str(row["case_study_id"]), set()).add(
                    int(row["freeze_year"])
                )
    required_years = {window.freeze_year for window in DEFAULT_WINDOWS}
    complete = []
    for task_id, windows in static_primary.items():
        static_years = {int(row["freeze_year"]) for row in windows}
        if static_years == required_years and dynamic_years.get(task_id) == required_years:
            complete.append(str(task_id))
    registry_order = list(static_manifest.get("formal_case_study_ids") or ())
    return sorted(complete, key=registry_order.index)


def prepare_eligibility(
    *, release_manifest_path: Path, v3_root: Path
) -> dict[str, Any]:
    release_manifest_path = release_manifest_path.resolve()
    release_verification = verify_release(release_manifest_path, deep=True)
    release = _read_json(release_manifest_path)
    release_id = str(release["release_id"])
    expected_hashes = dict(release_verification["artifact_hashes"])
    input_dir = release_manifest_path.parent

    workspace = (v3_root.resolve() / "eligibility" / release_id).resolve()
    _require(
        workspace.parent == (v3_root.resolve() / "eligibility").resolve(),
        "eligibility workspace escapes v3 root",
    )
    pipeline_manifest_path = workspace / "eligibility_pipeline_manifest.json"
    if pipeline_manifest_path.is_file():
        existing = _read_json(pipeline_manifest_path)
        lock = _read_json(workspace / "eligibility_pipeline.lock.json")
        _require(
            sha256_file(pipeline_manifest_path)
            == str(lock.get("pipeline_manifest_sha256") or "").upper(),
            "eligibility pipeline manifest differs from its lock",
        )
        _require(existing.get("release_id") == release_id, "eligibility pipeline release mismatch")
        return existing

    snapshot_root = workspace / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshots = _build_or_reuse_snapshots(
        input_dir=input_dir, snapshot_root=snapshot_root
    )

    # Imports are intentionally delayed: the historical modules keep their v2
    # constants for old commands, while this process supplies one v3 release.
    from neurooracle.scripts import audit_case_study_hindcasting_executability as static_audit
    from neurooracle.scripts import audit_dynamic_hindcasting_eligibility as dynamic_audit
    from neurooracle.scripts import freeze_hindcasting_eligibility as static_freeze
    from neurooracle.src.case_studies import CASE_STUDIES

    static_audit.CURRENT_CANONICAL_SHA256 = expected_hashes
    static_audit.validate_canonical_kg_release = _relocated_validator()
    static_freeze.CURRENT_CANONICAL_SHA256 = expected_hashes
    dynamic_audit.CURRENT_CANONICAL_SHA256 = expected_hashes

    audit_root = workspace / "executability_audit"
    audit_json = audit_root / "executability_audit.json"
    audit_csv = audit_root / "executability_matrix.csv"
    _require(
        not audit_json.exists() and not audit_csv.exists(),
        "unlocked executability audit already exists; preserve it and audit the partial state",
    )
    print("[audit] static executability across all 17 tasks", flush=True)
    static_audit.audit(
        input_dir=input_dir,
        snapshot_root=snapshot_root,
        output_dir=audit_root,
        windows=tuple(DEFAULT_WINDOWS),
        cases=tuple(CASE_STUDIES),
    )

    canonical = validate_canonical_kg_release(
        kg_path=input_dir / "knowledge_graph.json",
        claims_path=input_dir / "extracted_claims.jsonl",
        state_path=input_dir / "CURRENT_STATE.json",
        expected_sha256=expected_hashes,
        allow_relocated_artifacts=True,
    )
    static_lock_root = workspace / "static_eligibility"
    print("[lock] static eligibility", flush=True)
    static_manifest = static_freeze.freeze_eligibility(
        audit_json=audit_json,
        matrix_csv=audit_csv,
        output_dir=static_lock_root,
        canonical_release=canonical,
    )
    static_manifest_path = static_lock_root / "eligibility_manifest.json"

    dynamic_root = workspace / "dynamic_eligibility"
    print("[audit+lock] dynamic early/terminal eligibility", flush=True)
    dynamic_manifest = dynamic_audit.audit_dynamic_eligibility(
        input_dir=input_dir,
        snapshot_root=snapshot_root,
        static_eligibility_manifest=static_manifest_path,
        output_dir=dynamic_root,
        feedback_years=2,
        min_early_support=1,
        min_terminal_pairs=10,
        allow_relocated_canonical_artifacts=True,
    )
    dynamic_manifest_path = dynamic_root / "dynamic_eligibility_manifest.json"

    complete_primary = _complete_primary_tasks(
        static_manifest, dynamic_manifest, dynamic_manifest_path
    )
    created_at = datetime.now(timezone.utc).isoformat()
    pipeline = {
        "schema_version": "neurodiscovery-hindcasting-v3-eligibility-pipeline.v1",
        "status": "locked_method_blind_eligibility",
        "created_at": created_at,
        "release_id": release_id,
        "release_manifest": str(release_manifest_path),
        "release_manifest_sha256": sha256_file(release_manifest_path),
        "release_artifact_hashes": expected_hashes,
        "method_outputs_consumed": False,
        "snapshots": snapshots,
        "executability_audit": {
            "json": str(audit_json.resolve()),
            "json_sha256": sha256_file(audit_json),
            "csv": str(audit_csv.resolve()),
            "csv_sha256": sha256_file(audit_csv),
        },
        "static_eligibility": {
            "manifest": str(static_manifest_path.resolve()),
            "manifest_sha256": sha256_file(static_manifest_path),
        },
        "dynamic_eligibility": {
            "manifest": str(dynamic_manifest_path.resolve()),
            "manifest_sha256": sha256_file(dynamic_manifest_path),
            "feedback_years": 2,
            "minimum_early_relations": 1,
            "minimum_terminal_relations": 10,
        },
        "five_window_primary_task_ids_before_semantic_gates": complete_primary,
        "semantic_gate_rule": "Release-declared semantic gates are applied after structural eligibility and before creating a locked cohort.",
    }
    _atomic_json(pipeline_manifest_path, pipeline)
    _atomic_json(
        workspace / "eligibility_pipeline.lock.json",
        {
            "schema_version": "neurodiscovery-hindcasting-v3-eligibility-pipeline-lock.v1",
            "release_id": release_id,
            "locked_at": created_at,
            "pipeline_manifest": "eligibility_pipeline_manifest.json",
            "pipeline_manifest_sha256": sha256_file(pipeline_manifest_path),
            "immutable": True,
        },
    )
    return pipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = prepare_eligibility(
        release_manifest_path=args.release_manifest,
        v3_root=args.v3_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release_id": result["release_id"],
                "five_window_primary_task_ids_before_semantic_gates": result[
                    "five_window_primary_task_ids_before_semantic_gates"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
