"""Register immutable KG releases and append-only task cohorts for hindcasting v3.

The v3 experiment deliberately separates three identities:

* the fixed experimental protocol;
* an immutable, content-addressed KG release; and
* an immutable cohort of tasks admitted on that release.

This lets the live KG continue to grow without changing the evidence available to
an already registered result.  A later KG or task expansion creates a new release
or cohort; existing manifests are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.scripts.canonical_kg_release import validate_canonical_kg_release


DEFAULT_V3_ROOT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "formal_hindcasting_v3_expandable_20260826"
)
DEFAULT_PROTOCOL = DEFAULT_V3_ROOT / "protocol" / "formal_hindcasting_design_v3.json"
CANONICAL_FILENAMES = {
    "knowledge_graph": "knowledge_graph.json",
    "extracted_claims": "extracted_claims.jsonl",
    "current_state": "CURRENT_STATE.json",
}
WINDOWS = (
    (2016, 2017, 2021),
    (2017, 2018, 2022),
    (2018, 2019, 2023),
    (2019, 2020, 2024),
    (2020, 2021, 2025),
)
METHODS = ("neurodiscovery", "sciagents", "openscholar_rag")
SEEDS = tuple(range(10))
CHUNK_BYTES = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _protocol_record(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = _read_json(protocol_path)
    lock_path = protocol_path.with_name("formal_hindcasting_design_v3.lock.json")
    _require(lock_path.is_file(), f"missing protocol lock: {lock_path}")
    lock = _read_json(lock_path)
    digest = sha256_file(protocol_path)
    _require(
        digest == str(lock.get("protocol_sha256") or "").upper(),
        "v3 protocol differs from its lock",
    )
    _require(
        protocol.get("schema_version") == "neurodiscovery-hindcasting-formal-design.v3",
        "unexpected v3 protocol schema",
    )
    return {
        "path": str(protocol_path),
        "sha256": digest,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "design_id": protocol.get("design_id"),
    }


def _source_paths(input_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    input_dir = input_dir.resolve()
    state_path = input_dir / CANONICAL_FILENAMES["current_state"]
    state = _read_json(state_path)
    records = state.get("canonical_files") or {}
    paths = {"current_state": state_path}
    for key in ("knowledge_graph", "extracted_claims"):
        record = records.get(key) or {}
        candidate = Path(str(record.get("path") or "")).resolve()
        _require(candidate.is_file(), f"CURRENT_STATE artifact is missing: {candidate}")
        _require(
            candidate.parent == input_dir,
            f"refusing a canonical artifact outside --input-dir: {candidate}",
        )
        _require(
            candidate.name == CANONICAL_FILENAMES[key],
            f"unexpected canonical filename for {key}: {candidate.name}",
        )
        paths[key] = candidate
    return state, paths


def _observations(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        stat = path.stat()
        result[key] = {
            "path": str(path.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def _copy_and_hash(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(CHUNK_BYTES), b""):
            dst.write(chunk)
            digest.update(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    shutil.copystat(source, destination)
    return digest.hexdigest().upper()


def _expected_hashes_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts") or {}
    return {
        key: str((artifacts.get(key) or {}).get("sha256") or "").upper()
        for key in CANONICAL_FILENAMES
    }


def _release_id(state: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    generated = str(state.get("generated_at") or "")
    digits = "".join(character for character in generated[:10] if character.isdigit())
    date_token = digits or datetime.now(timezone.utc).strftime("%Y%m%d")
    return (
        f"kg_{date_token}_{hashes['knowledge_graph'][:12].lower()}_"
        f"{hashes['extracted_claims'][:8].lower()}"
    )


def _safe_remove_staging(staging: Path, releases_root: Path) -> None:
    resolved = staging.resolve()
    root = releases_root.resolve()
    _require(resolved.parent == root, f"unsafe staging cleanup target: {resolved}")
    _require(resolved.name.startswith(".staging_"), f"unsafe staging name: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def verify_release(manifest_path: Path, *, deep: bool = True) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == "neurodiscovery-hindcasting-kg-release.v3",
        "unexpected v3 release schema",
    )
    release_dir = manifest_path.parent
    lock_path = release_dir / "release.lock.json"
    lock = _read_json(lock_path)
    manifest_hash = sha256_file(manifest_path)
    _require(
        manifest_hash == str(lock.get("release_manifest_sha256") or "").upper(),
        "release manifest differs from its lock",
    )
    _require(lock.get("release_id") == manifest.get("release_id"), "release ID mismatch")

    expected = _expected_hashes_from_manifest(manifest)
    paths: dict[str, Path] = {}
    artifacts = manifest.get("artifacts") or {}
    for key, filename in CANONICAL_FILENAMES.items():
        record = artifacts.get(key) or {}
        relative_path = Path(str(record.get("relative_path") or filename))
        _require(not relative_path.is_absolute(), f"absolute release artifact path: {relative_path}")
        artifact_path = (release_dir / relative_path).resolve()
        _require(artifact_path.parent == release_dir, f"release artifact escapes release dir: {artifact_path}")
        _require(artifact_path.is_file(), f"missing release artifact: {artifact_path}")
        _require(
            artifact_path.stat().st_size == int(record.get("bytes") or -1),
            f"release artifact size mismatch: {key}",
        )
        paths[key] = artifact_path

    if deep:
        validate_canonical_kg_release(
            kg_path=paths["knowledge_graph"],
            claims_path=paths["extracted_claims"],
            state_path=paths["current_state"],
            expected_sha256=expected,
            allow_relocated_artifacts=True,
        )

    protocol = manifest.get("protocol") or {}
    protocol_path = Path(str(protocol.get("path") or ""))
    _require(protocol_path.is_file(), f"missing protocol referenced by release: {protocol_path}")
    _require(
        sha256_file(protocol_path) == str(protocol.get("sha256") or "").upper(),
        "release references a different v3 protocol",
    )
    return {
        "release_id": manifest["release_id"],
        "release_manifest": str(manifest_path),
        "release_manifest_sha256": manifest_hash,
        "deep_verified": deep,
        "artifact_hashes": expected,
    }


def register_release(*, input_dir: Path, v3_root: Path, protocol_path: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    v3_root = v3_root.resolve()
    releases_root = (v3_root / "releases").resolve()
    releases_root.mkdir(parents=True, exist_ok=True)
    protocol_record = _protocol_record(protocol_path)
    state, source_paths = _source_paths(input_dir)

    before_observation = _observations(source_paths)
    canonical_before = validate_canonical_kg_release(
        kg_path=source_paths["knowledge_graph"],
        claims_path=source_paths["extracted_claims"],
        state_path=source_paths["current_state"],
        expected_sha256=None,
    )
    source_hashes = {
        key: str(value["sha256"]).upper()
        for key, value in (canonical_before.get("files") or {}).items()
    }
    release_id = _release_id(state, source_hashes)
    release_dir = releases_root / release_id
    manifest_path = release_dir / "release_manifest.json"
    if release_dir.exists():
        verified = verify_release(manifest_path, deep=True)
        _require(
            verified["artifact_hashes"] == source_hashes,
            "content-addressed release directory exists for different source bytes",
        )
        return {**verified, "created": False}

    staging = releases_root / f".staging_{release_id}_{uuid.uuid4().hex}"
    _require(not staging.exists(), f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        copied_hashes: dict[str, str] = {}
        for key in ("knowledge_graph", "extracted_claims", "current_state"):
            destination = staging / CANONICAL_FILENAMES[key]
            copied_hashes[key] = _copy_and_hash(source_paths[key], destination)
            _require(
                copied_hashes[key] == source_hashes[key],
                f"copy hash mismatch for {key}",
            )

        after_observation = _observations(source_paths)
        _require(
            after_observation == before_observation,
            "live canonical KG metadata changed while registering the release",
        )
        canonical_after = validate_canonical_kg_release(
            kg_path=source_paths["knowledge_graph"],
            claims_path=source_paths["extracted_claims"],
            state_path=source_paths["current_state"],
            expected_sha256=source_hashes,
        )
        _require(
            {
                key: str(value["sha256"]).upper()
                for key, value in (canonical_after.get("files") or {}).items()
            }
            == source_hashes,
            "live canonical KG changed while registering the release",
        )

        registered_at = datetime.now(timezone.utc).isoformat()
        artifact_records = {
            key: {
                "relative_path": CANONICAL_FILENAMES[key],
                "bytes": (staging / CANONICAL_FILENAMES[key]).stat().st_size,
                "sha256": copied_hashes[key],
            }
            for key in CANONICAL_FILENAMES
        }
        mapping = state.get("case2_component_mapping") or {}
        manifest = {
            "schema_version": "neurodiscovery-hindcasting-kg-release.v3",
            "release_id": release_id,
            "status": "immutable_registered",
            "registered_at": registered_at,
            "source_generated_at": state.get("generated_at"),
            "taxonomy_version": state.get("taxonomy_version"),
            "protocol": protocol_record,
            "artifacts": artifact_records,
            "source_stability": {
                "rule": "source sizes, mtimes and SHA-256 values must remain identical before, during and after the byte copy",
                "before": before_observation,
                "after": after_observation,
            },
            "formal_kg_statistics": state.get("formal_kg_statistics"),
            "validation_protocols": state.get("validation_protocols"),
            "semantic_admission_gates": {
                "case2_pathway_mediation": {
                    "status": (
                        "pending_semantic_reaudit"
                        if bool(mapping.get("semantic_reaudit_deferred_until_v3_complete"))
                        else "not_declared_pending"
                    ),
                    "source_mapping": mapping,
                    "rule": "cannot enter a locked primary cohort until release-specific semantic reaudit evidence says passed",
                }
            },
            "immutability_contract": {
                "overwrite_permitted": False,
                "live_kg_updates_affect_this_release": False,
                "verification": "release.lock.json pins this manifest; this manifest pins all three canonical artifacts by size and SHA-256",
            },
        }
        staged_manifest = staging / "release_manifest.json"
        _atomic_json(staged_manifest, manifest)
        manifest_hash = sha256_file(staged_manifest)
        _atomic_json(
            staging / "release.lock.json",
            {
                "schema_version": "neurodiscovery-hindcasting-release-lock.v3",
                "release_id": release_id,
                "locked_at": registered_at,
                "release_manifest": "release_manifest.json",
                "release_manifest_sha256": manifest_hash,
                "immutable": True,
            },
        )
        staging.replace(release_dir)
    except BaseException:
        if staging.exists():
            _safe_remove_staging(staging, releases_root)
        raise

    return {**verify_release(manifest_path, deep=True), "created": True}


def _manifest_and_lock(path: Path, expected_schema: str, lock_name: str, hash_key: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    manifest = _read_json(path)
    _require(manifest.get("schema_version") == expected_schema, f"unexpected manifest schema: {path}")
    lock = _read_json(path.parent / lock_name)
    digest = sha256_file(path)
    _require(digest == str(lock.get(hash_key) or "").upper(), f"manifest differs from lock: {path}")
    return manifest, digest


def _resolve_matrix(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    record = manifest.get("locked_matrix") or {}
    raw_path = Path(str(record.get("path") or ""))
    if raw_path.is_file():
        return raw_path.resolve()
    local = (manifest_path.parent / raw_path.name).resolve()
    _require(local.is_file(), f"locked eligibility matrix is missing: {raw_path}")
    return local


def _primary_task_windows(manifest_path: Path) -> dict[str, set[int]]:
    manifest = _read_json(manifest_path)
    matrix = _resolve_matrix(manifest_path, manifest)
    expected_hash = str((manifest.get("locked_matrix") or {}).get("sha256") or "").upper()
    _require(sha256_file(matrix) == expected_hash, f"eligibility matrix hash mismatch: {matrix}")
    result: dict[str, set[int]] = {}
    with matrix.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("analysis_tier") or "") != "primary":
                continue
            task_id = str(row.get("case_study_id") or "")
            result.setdefault(task_id, set()).add(int(row["freeze_year"]))
    return result


def _semantic_gate_passed(path: Path, *, task_id: str, release_id: str) -> dict[str, Any]:
    evidence = _read_json(path)
    _require(evidence.get("task_id") == task_id, f"semantic gate task mismatch: {path}")
    _require(evidence.get("release_id") == release_id, f"semantic gate release mismatch: {path}")
    _require(evidence.get("status") == "passed", f"semantic gate has not passed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "status": "passed",
    }


def _task_list(values: Iterable[str]) -> list[str]:
    tasks: list[str] = []
    for value in values:
        for task in value.split(","):
            normalized = task.strip()
            if normalized and normalized not in tasks:
                tasks.append(normalized)
    _require(bool(tasks), "at least one --task is required")
    return tasks


def _execution_assets_record(
    *,
    eligibility_pipeline_path: Path,
    kge_assets_manifest_path: Path,
    expected_release_id: str,
    expected_release_manifest_path: Path,
    expected_release_manifest_hash: str,
    deep: bool,
) -> dict[str, Any]:
    # Imported lazily so release and candidate-cohort operations do not load the
    # KGE stack.  Verification itself does not import torch.
    from neurooracle.scripts.prepare_hindcasting_v3_kge import (
        verify_kge_assets_manifest,
    )

    verified = verify_kge_assets_manifest(
        kge_assets_manifest_path,
        expected_pipeline_path=eligibility_pipeline_path,
        deep=deep,
    )
    _require(
        verified["release_id"] == expected_release_id,
        "execution assets use another KG release",
    )
    pipeline = verified["eligibility_pipeline"]
    _require(
        Path(pipeline["release_manifest"]).resolve()
        == expected_release_manifest_path.resolve(),
        "execution assets reference another release manifest",
    )
    _require(
        pipeline["release_manifest_sha256"] == expected_release_manifest_hash,
        "execution assets pin another release-manifest hash",
    )
    _require(
        pipeline["method_outputs_consumed"] is False,
        "execution assets were selected using method outputs",
    )
    year_assets = verified["year_assets"]
    _require(
        [record["freeze_year"] for record in year_assets]
        == [window[0] for window in WINDOWS],
        "execution assets do not cover the five locked windows",
    )
    return {
        "eligibility_pipeline": pipeline,
        "kge": {
            "status": verified["status"],
            "manifest": verified["manifest"],
            "lock": verified["lock"],
            "config": verified["config"],
            "year_assets": year_assets,
            "cross_release_reuse_permitted": False,
        },
    }


def verify_cohort(
    manifest_path: Path, *, deep_execution_assets: bool = True
) -> dict[str, Any]:
    manifest, digest = _manifest_and_lock(
        manifest_path,
        "neurodiscovery-hindcasting-task-cohort.v3",
        "cohort.lock.json",
        "cohort_manifest_sha256",
    )
    protocol = manifest.get("protocol") or {}
    protocol_path = Path(str(protocol.get("path") or ""))
    _require(protocol_path.is_file(), f"missing cohort protocol: {protocol_path}")
    _require(sha256_file(protocol_path) == str(protocol.get("sha256") or "").upper(), "cohort protocol hash mismatch")
    release_record = manifest.get("kg_release") or {}
    release_path = Path(str(release_record.get("manifest_path") or ""))
    _require(release_path.is_file(), f"missing cohort KG release: {release_path}")
    _require(sha256_file(release_path) == str(release_record.get("manifest_sha256") or "").upper(), "cohort KG release hash mismatch")
    status = str(manifest.get("status") or "")
    runnable = bool(manifest.get("runnable"))
    _require(status in {"candidate", "locked"}, "unknown cohort status")
    _require(
        runnable == (status == "locked"),
        "cohort runnable flag does not match its status",
    )
    pinned_assets = manifest.get("execution_assets")
    if status == "locked":
        _require(isinstance(pinned_assets, dict), "locked cohort has no execution assets")
    if pinned_assets is not None:
        pipeline_record = pinned_assets.get("eligibility_pipeline") or {}
        kge_record = pinned_assets.get("kge") or {}
        kge_manifest_record = kge_record.get("manifest") or {}
        observed_assets = _execution_assets_record(
            eligibility_pipeline_path=Path(str(pipeline_record.get("path") or "")),
            kge_assets_manifest_path=Path(str(kge_manifest_record.get("path") or "")),
            expected_release_id=str(release_record.get("release_id") or ""),
            expected_release_manifest_path=release_path,
            expected_release_manifest_hash=str(
                release_record.get("manifest_sha256") or ""
            ).upper(),
            deep=deep_execution_assets,
        )
        _require(
            pinned_assets == observed_assets,
            "cohort execution-asset pins differ from verified assets",
        )
    tasks = list(manifest.get("task_ids") or ())
    matrix = manifest.get("run_matrix") or {}
    _require(int(matrix.get("tasks") or 0) == len(tasks), "cohort task count mismatch")
    expected_runs = len(tasks) * len(WINDOWS) * len(METHODS) * len(SEEDS)
    _require(int(matrix.get("total_method_runs") or -1) == expected_runs, "cohort run count mismatch")
    return {
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest": str(manifest_path.resolve()),
        "cohort_manifest_sha256": digest,
        "status": status,
        "runnable": runnable,
        "tasks": tasks,
        "total_method_runs": expected_runs,
        "execution_assets_verified": pinned_assets is not None,
        "deep_execution_assets_verified": bool(
            pinned_assets is not None and deep_execution_assets
        ),
    }


def create_cohort(
    *,
    cohort_id: str,
    release_manifest_path: Path,
    tasks: Sequence[str],
    status: str,
    mode: str,
    v3_root: Path,
    protocol_path: Path,
    preflight_path: Path | None = None,
    static_eligibility_path: Path | None = None,
    dynamic_eligibility_path: Path | None = None,
    eligibility_pipeline_path: Path | None = None,
    kge_assets_manifest_path: Path | None = None,
    semantic_gate_paths: Sequence[Path] = (),
    parent_cohort_path: Path | None = None,
) -> dict[str, Any]:
    _require(status in {"candidate", "locked"}, f"unknown cohort status: {status}")
    _require(mode in {"full_release_matrix", "extension_only"}, f"unknown cohort mode: {mode}")
    _require(cohort_id and all(character.isalnum() or character in "-_" for character in cohort_id), "cohort ID must contain only letters, digits, '-' and '_'")
    tasks = _task_list(tasks)
    protocol_record = _protocol_record(protocol_path)
    release_verified = verify_release(release_manifest_path, deep=True)
    release_manifest = _read_json(release_manifest_path.resolve())
    release_id = str(release_manifest["release_id"])

    release_dir = release_manifest_path.resolve().parent
    state = _read_json(release_dir / CANONICAL_FILENAMES["current_state"])
    supported = set(
        (((state.get("validation_protocols") or {}).get("hindcasting") or {}).get("supported_case_studies") or ())
    )
    statistics = set(((state.get("formal_kg_statistics") or {}).get("case_studies") or {}))
    unknown = sorted(set(tasks) - supported)
    _require(not unknown, f"tasks are not in the release hindcasting registry: {unknown}")
    missing_stats = sorted(set(tasks) - statistics)
    _require(not missing_stats, f"tasks have no release statistics: {missing_stats}")

    parent_record: dict[str, Any] | None = None
    if parent_cohort_path is not None:
        parent = verify_cohort(parent_cohort_path)
        parent_manifest = _read_json(parent_cohort_path.resolve())
        parent_tasks = set(parent["tasks"])
        if mode == "full_release_matrix":
            _require(parent_tasks < set(tasks), "a successor full cohort must strictly add tasks")
        else:
            _require(parent_tasks.isdisjoint(tasks), "an extension-only cohort must not repeat parent tasks")
        parent_record = {
            "cohort_id": parent["cohort_id"],
            "manifest_path": str(parent_cohort_path.resolve()),
            "manifest_sha256": parent["cohort_manifest_sha256"],
            "release_id": ((parent_manifest.get("kg_release") or {}).get("release_id")),
        }

    preflight_record: dict[str, Any] | None = None
    preflight_assessments: dict[str, Any] = {}
    if preflight_path is not None:
        preflight_path = preflight_path.resolve()
        preflight = _read_json(preflight_path)
        _require(preflight.get("release_id") == release_id, "preflight uses another KG release")
        declared_release_hash = str(preflight.get("release_manifest_sha256") or "").upper()
        _require(
            not declared_release_hash
            or declared_release_hash == sha256_file(release_manifest_path.resolve()),
            "preflight pins a different release manifest",
        )
        preflight_assessments = {
            str(row.get("task_id")): row
            for row in (preflight.get("task_assessments") or ())
        }
        preflight_record = {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "status": preflight.get("status"),
        }

    audit_records: dict[str, Any] = {}
    static_primary: dict[str, set[int]] = {}
    dynamic_primary: dict[str, set[int]] = {}
    if static_eligibility_path is not None:
        static_eligibility_path = static_eligibility_path.resolve()
        static_primary = _primary_task_windows(static_eligibility_path)
        audit_records["static_eligibility"] = {
            "path": str(static_eligibility_path),
            "sha256": sha256_file(static_eligibility_path),
        }
    if dynamic_eligibility_path is not None:
        dynamic_eligibility_path = dynamic_eligibility_path.resolve()
        dynamic_primary = _primary_task_windows(dynamic_eligibility_path)
        audit_records["dynamic_eligibility"] = {
            "path": str(dynamic_eligibility_path),
            "sha256": sha256_file(dynamic_eligibility_path),
        }

    semantic_evidence: dict[str, Any] = {}
    for path in semantic_gate_paths:
        evidence = _read_json(path.resolve())
        task_id = str(evidence.get("task_id") or "")
        semantic_evidence[task_id] = _semantic_gate_passed(
            path.resolve(), task_id=task_id, release_id=release_id
        )

    release_gates = release_manifest.get("semantic_admission_gates") or {}
    task_records = []
    required_years = {window[0] for window in WINDOWS}
    for task_id in tasks:
        gate = release_gates.get(task_id) or {}
        gate_pending = str(gate.get("status") or "").startswith("pending")
        if status == "locked":
            _require(static_eligibility_path is not None, "locked cohort requires static eligibility")
            _require(dynamic_eligibility_path is not None, "locked cohort requires dynamic eligibility")
            _require(static_primary.get(task_id) == required_years, f"task lacks five primary static windows: {task_id}")
            _require(dynamic_primary.get(task_id) == required_years, f"task lacks five primary dynamic windows: {task_id}")
            if gate_pending:
                _require(task_id in semantic_evidence, f"task has an unresolved semantic admission gate: {task_id}")
            admission = "admitted"
        else:
            assessment = preflight_assessments.get(task_id) or {}
            admission = str(assessment.get("admission_status") or "pending_official_audit")
            if gate_pending and "semantic" not in admission:
                admission = "pending_semantic_reaudit_and_official_audit"
        task_records.append(
            {
                "task_id": task_id,
                "admission_status": admission,
                "semantic_gate": semantic_evidence.get(task_id) or gate or None,
                "release_statistics": (((state.get("formal_kg_statistics") or {}).get("case_studies") or {}).get(task_id)),
            }
        )

    created_at = datetime.now(timezone.utc).isoformat()
    cohorts_root = (v3_root.resolve() / "cohorts").resolve()
    cohorts_root.mkdir(parents=True, exist_ok=True)
    cohort_dir = (cohorts_root / cohort_id).resolve()
    _require(cohort_dir.parent == cohorts_root, "cohort path escapes v3 root")
    _require(not cohort_dir.exists(), f"cohort is immutable and already exists: {cohort_dir}")
    release_manifest_hash = sha256_file(release_manifest_path.resolve())
    _require(
        (eligibility_pipeline_path is None) == (kge_assets_manifest_path is None),
        "eligibility pipeline and KGE asset manifest must be supplied together",
    )
    if status == "locked":
        _require(
            eligibility_pipeline_path is not None,
            "locked cohort requires a pinned eligibility pipeline",
        )
        _require(
            kge_assets_manifest_path is not None,
            "locked cohort requires pinned KGE assets",
        )
    execution_assets = None
    if eligibility_pipeline_path is not None and kge_assets_manifest_path is not None:
        execution_assets = _execution_assets_record(
            eligibility_pipeline_path=eligibility_pipeline_path,
            kge_assets_manifest_path=kge_assets_manifest_path,
            expected_release_id=release_id,
            expected_release_manifest_path=release_manifest_path,
            expected_release_manifest_hash=release_manifest_hash,
            deep=True,
        )
    manifest = {
        "schema_version": "neurodiscovery-hindcasting-task-cohort.v3",
        "cohort_id": cohort_id,
        "status": status,
        "runnable": status == "locked",
        "created_at": created_at,
        "mode": mode,
        "protocol": protocol_record,
        "kg_release": {
            "release_id": release_id,
            "manifest_path": str(release_manifest_path.resolve()),
            "manifest_sha256": release_manifest_hash,
            "artifact_hashes": release_verified["artifact_hashes"],
        },
        "parent_cohort": parent_record,
        "task_ids": list(tasks),
        "task_records": task_records,
        "execution_assets": execution_assets,
        "method_blind_selection_evidence": {
            "preflight": preflight_record,
            "formal_audits": audit_records,
            "method_outputs_consumed": False,
        },
        "run_matrix": {
            "tasks": len(tasks),
            "windows": len(WINDOWS),
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "total_method_runs": len(tasks) * len(WINDOWS) * len(METHODS) * len(SEEDS),
            "runs_per_method": len(tasks) * len(WINDOWS) * len(SEEDS),
        },
        "aggregation_contract": {
            "full_release_matrix": mode == "full_release_matrix",
            "cross_release_primary_pooling_permitted": False,
            "extension_only_results_enter_parent_primary_aggregate": False,
            "rule": (
                "A primary aggregate contains only cells from this one cohort and one KG release. "
                "To publish an expanded primary aggregate on a newer KG release, rerun all cohort tasks and all methods on that release."
            ),
        },
        "immutability_contract": {
            "add_or_remove_tasks_in_place": False,
            "successor_rule": "create a new cohort manifest and retain this one",
            "outcome_based_task_admission_permitted": False,
        },
    }
    staging = cohorts_root / f".staging_{cohort_id}_{uuid.uuid4().hex}"
    _require(not staging.exists(), f"cohort staging path already exists: {staging}")
    staging.mkdir()
    try:
        staged_manifest = staging / "cohort_manifest.json"
        _atomic_json(staged_manifest, manifest)
        _atomic_json(
            staging / "cohort.lock.json",
            {
                "schema_version": "neurodiscovery-hindcasting-cohort-lock.v3",
                "cohort_id": cohort_id,
                "locked_at": created_at,
                "cohort_manifest": "cohort_manifest.json",
                "cohort_manifest_sha256": sha256_file(staged_manifest),
                "immutable": True,
            },
        )
        staging.replace(cohort_dir)
    except BaseException:
        if staging.exists():
            _safe_remove_staging(staging, cohorts_root)
        raise
    manifest_path = cohort_dir / "cohort_manifest.json"
    return verify_cohort(manifest_path, deep_execution_assets=False)


def list_registry(v3_root: Path) -> dict[str, Any]:
    v3_root = v3_root.resolve()
    releases = []
    for path in sorted((v3_root / "releases").glob("*/release_manifest.json")):
        try:
            releases.append(verify_release(path, deep=False))
        except Exception as exc:  # surface damaged entries without hiding the rest
            releases.append({"release_manifest": str(path), "error": str(exc)})
    cohorts = []
    for path in sorted((v3_root / "cohorts").glob("*/cohort_manifest.json")):
        try:
            cohorts.append(verify_cohort(path, deep_execution_assets=False))
        except Exception as exc:
            cohorts.append({"cohort_manifest": str(path), "error": str(exc)})
    return {"v3_root": str(v3_root), "releases": releases, "cohorts": cohorts}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register-release", help="Copy and register the current canonical KG as an immutable release")
    register.add_argument("--input-dir", type=Path, default=REPO / "neurooracle" / "data" / "full_v2")

    verify_release_parser = subparsers.add_parser("verify-release", help="Verify a registered release and all artifact hashes")
    verify_release_parser.add_argument("--manifest", type=Path, required=True)
    verify_release_parser.add_argument("--shallow", action="store_true")

    cohort = subparsers.add_parser("create-cohort", help="Create a new immutable task cohort")
    cohort.add_argument("--cohort-id", required=True)
    cohort.add_argument("--release-manifest", type=Path, required=True)
    cohort.add_argument("--task", action="append", required=True, help="Repeat or pass a comma-separated task list")
    cohort.add_argument("--status", choices=("candidate", "locked"), default="candidate")
    cohort.add_argument("--mode", choices=("full_release_matrix", "extension_only"), default="full_release_matrix")
    cohort.add_argument("--preflight", type=Path)
    cohort.add_argument("--static-eligibility", type=Path)
    cohort.add_argument("--dynamic-eligibility", type=Path)
    cohort.add_argument("--eligibility-pipeline", type=Path)
    cohort.add_argument("--kge-assets-manifest", type=Path)
    cohort.add_argument("--semantic-gate", type=Path, action="append", default=[])
    cohort.add_argument("--parent-cohort", type=Path)

    verify_cohort_parser = subparsers.add_parser("verify-cohort", help="Verify an immutable cohort manifest")
    verify_cohort_parser.add_argument("--manifest", type=Path, required=True)

    subparsers.add_parser("list", help="List registered releases and cohorts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "register-release":
        result = register_release(
            input_dir=args.input_dir,
            v3_root=args.v3_root,
            protocol_path=args.protocol,
        )
    elif args.command == "verify-release":
        result = verify_release(args.manifest, deep=not args.shallow)
    elif args.command == "create-cohort":
        result = create_cohort(
            cohort_id=args.cohort_id,
            release_manifest_path=args.release_manifest,
            tasks=args.task,
            status=args.status,
            mode=args.mode,
            v3_root=args.v3_root,
            protocol_path=args.protocol,
            preflight_path=args.preflight,
            static_eligibility_path=args.static_eligibility,
            dynamic_eligibility_path=args.dynamic_eligibility,
            eligibility_pipeline_path=args.eligibility_pipeline,
            kge_assets_manifest_path=args.kge_assets_manifest,
            semantic_gate_paths=args.semantic_gate,
            parent_cohort_path=args.parent_cohort,
        )
    elif args.command == "verify-cohort":
        result = verify_cohort(args.manifest)
    else:
        result = list_registry(args.v3_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
