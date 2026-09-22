"""Verify immutable experiment source bundles before and after execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_BUNDLE_SCHEMA = "neurodiscovery-experiment-source-bundle.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_source_bundle(
    manifest_path: Path,
    *,
    require_live_source: bool = True,
    verify_references: bool = True,
) -> dict[str, Any]:
    """Verify archived bytes, live source bytes, and hash-only data inputs."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
        raise ValueError("source bundle uses an incompatible schema")

    bundle_root = manifest_path.parent
    archived_paths: set[Path] = set()
    live_paths: set[Path] = set()
    for record in manifest.get("files") or ():
        expected = str(record.get("sha256") or "").upper()
        raw_bytes = record.get("bytes")
        expected_bytes = int(raw_bytes) if raw_bytes is not None else -1
        archived = (bundle_root / str(record.get("bundle_path") or "")).resolve()
        live = Path(str(record.get("source_path") or "")).resolve()
        if not expected or expected_bytes < 0:
            raise ValueError(f"incomplete source bundle record: {record}")
        if archived in archived_paths:
            raise ValueError(f"duplicate archived source path: {archived}")
        archived_paths.add(archived)
        if not archived.is_file() or archived.stat().st_size != expected_bytes:
            raise ValueError(f"archived source size mismatch: {archived}")
        if sha256_file(archived) != expected:
            raise ValueError(f"archived source hash mismatch: {archived}")
        if require_live_source:
            if live in live_paths:
                raise ValueError(f"duplicate live source path: {live}")
            live_paths.add(live)
            if not live.is_file() or live.stat().st_size != expected_bytes:
                raise ValueError(f"live source size mismatch: {live}")
            if sha256_file(live) != expected:
                raise ValueError(f"live source differs from frozen bundle: {live}")

    reference_bytes = 0
    for record in manifest.get("references") or ():
        expected = str(record.get("sha256") or "").upper()
        raw_bytes = record.get("bytes")
        expected_bytes = int(raw_bytes) if raw_bytes is not None else -1
        path = Path(str(record.get("path") or "")).resolve()
        if not expected or expected_bytes < 0 or not path.is_file():
            raise ValueError(f"incomplete source bundle reference: {record}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"referenced input size mismatch: {path}")
        if verify_references and sha256_file(path) != expected:
            raise ValueError(f"referenced input differs from frozen bundle: {path}")
        reference_bytes += expected_bytes

    return {
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "bundled_files": len(manifest.get("files") or ()),
        "referenced_files": len(manifest.get("references") or ()),
        "referenced_bytes": reference_bytes,
        "live_source_verified": bool(require_live_source),
        "archived_source_verified": True,
        "referenced_inputs_verified": bool(verify_references),
    }


def verify_bundle_member(
    manifest_path: Path,
    member_path: Path,
) -> dict[str, Any]:
    """Verify one live configuration against its exact frozen bundle record."""

    manifest_path = manifest_path.resolve()
    member_path = member_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matching = [
        record
        for record in manifest.get("files") or ()
        if Path(str(record.get("source_path") or "")).resolve() == member_path
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one source-bundle record for {member_path}, found {len(matching)}"
        )
    record = matching[0]
    expected = str(record.get("sha256") or "").upper()
    if not member_path.is_file() or sha256_file(member_path) != expected:
        raise ValueError(f"bundle member differs from frozen bytes: {member_path}")
    return {
        "path": str(member_path),
        "sha256": expected,
        "roles": list(record.get("roles") or ()),
    }


__all__ = [
    "SOURCE_BUNDLE_SCHEMA",
    "sha256_file",
    "verify_bundle_member",
    "verify_source_bundle",
]
