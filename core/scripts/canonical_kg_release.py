"""Validate and record the canonical NeuroOracle KG release used by experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


CANONICAL_TAXONOMY_VERSION = "case_study_membership.v2"
CURRENT_CANONICAL_RELEASE_ID = "full_v2_20260825_092909z"
CURRENT_CANONICAL_SHA256 = {
    "knowledge_graph": "2C02732582DA9907C68300D791C3560B8E66CC268D987D453E7C971BD6CDEFF5",
    "extracted_claims": "705B079989FDEC3D7756CF737F12B41756EB8058F8ED66A009D6F832489F5BA4",
    "current_state": "744B75718B2BEEBFDAF9055595CB2161ED841643ACD58F365F55717BC7B5360E",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_canonical_kg_release(
    *,
    kg_path: Path,
    claims_path: Path,
    state_path: Path,
    case_study_id: str | None = None,
    expected_sha256: Mapping[str, str] | None = CURRENT_CANONICAL_SHA256,
    allow_relocated_artifacts: bool = False,
) -> dict[str, object]:
    """Fail closed unless paths, state metadata, sizes, and hashes agree."""

    paths = {
        "knowledge_graph": kg_path.resolve(),
        "extracted_claims": claims_path.resolve(),
        "current_state": state_path.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical {name}: {path}")

    state = json.loads(paths["current_state"].read_text(encoding="utf-8"))
    _require(state.get("status") == "canonical_current", "KG state is not canonical_current")
    _require(
        state.get("taxonomy_version") == CANONICAL_TAXONOMY_VERSION,
        "KG state does not use case_study_membership.v2",
    )
    audit = state.get("full_graph_taxonomy_audit") or {}
    _require(bool(audit.get("is_canonical")), "full-graph taxonomy audit is not canonical")
    _require(
        audit.get("schema_version") == CANONICAL_TAXONOMY_VERSION,
        "full-graph taxonomy audit has an unexpected schema",
    )

    state_files = state.get("canonical_files") or {}
    relocated_from: dict[str, str] = {}
    for name in ("knowledge_graph", "extracted_claims"):
        record = state_files.get(name) or {}
        recorded_path = Path(str(record.get("path", ""))).resolve()
        if recorded_path != paths[name]:
            _require(
                allow_relocated_artifacts,
                f"{name} path does not match CURRENT_STATE.json",
            )
            relocated_from[name] = str(recorded_path)
        _require(
            int(record.get("bytes", -1)) == paths[name].stat().st_size,
            f"{name} size does not match CURRENT_STATE.json",
        )

    statistics = state.get("formal_kg_statistics") or {}
    general = statistics.get("general") or {}
    claim_store = state.get("extracted_claim_store") or {}
    entities = audit.get("entities_scanned") or {}
    _require(
        int(general.get("claims", -1)) == int(claim_store.get("rows", -2)),
        "formal claim count and extracted claim-store rows disagree",
    )
    _require(
        int(general.get("claims", -1)) == int(entities.get("claim_nodes", -2)),
        "formal claim count and internal claim-node count disagree",
    )
    if case_study_id is not None:
        case_studies = statistics.get("case_studies") or {}
        _require(
            case_study_id in case_studies,
            f"unknown canonical Case Study ID: {case_study_id}",
        )

    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if expected_sha256 is not None:
        for name, expected in expected_sha256.items():
            _require(name in actual_hashes, f"unknown expected hash artifact: {name}")
            _require(
                actual_hashes[name] == str(expected).upper(),
                f"{name} SHA-256 mismatch: {actual_hashes[name]} != {expected}",
            )

    release = {
        "schema_version": "neurooracle-canonical-release.v1",
        "release_id": CURRENT_CANONICAL_RELEASE_ID,
        "status": state["status"],
        "taxonomy_version": state["taxonomy_version"],
        "release_generated_at": state.get("generated_at"),
        "case_study_id": case_study_id,
        "relocated_artifacts": bool(relocated_from),
        "relocated_from": relocated_from,
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_hashes[name],
            }
            for name, path in paths.items()
        },
        "general_statistics": general,
        "case_study_statistics": (
            (statistics.get("case_studies") or {}).get(case_study_id)
            if case_study_id is not None
            else None
        ),
        "quality": statistics.get("quality"),
        "reaudit": state.get("case_study_membership_reaudit"),
    }
    return release


def write_release_manifest(path: Path, release: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(release), indent=2) + "\n", encoding="utf-8")


# Updated: 2026-08-14 04:44 HKT
