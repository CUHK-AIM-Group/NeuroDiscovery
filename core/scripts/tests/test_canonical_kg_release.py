from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.scripts.canonical_kg_release import (
    sha256_file,
    validate_canonical_kg_release,
)


def _release_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    kg = tmp_path / "knowledge_graph.json"
    claims = tmp_path / "extracted_claims.jsonl"
    state = tmp_path / "CURRENT_STATE.json"
    kg.write_text('{"concepts": {}, "edges": []}\n', encoding="utf-8")
    claims.write_text('{"claim_id": "CLM:1"}\n', encoding="utf-8")
    payload = {
        "status": "canonical_current",
        "taxonomy_version": "case_study_membership.v2",
        "generated_at": "2026-08-10T00:00:00+08:00",
        "canonical_files": {
            "knowledge_graph": {"path": str(kg), "bytes": kg.stat().st_size},
            "extracted_claims": {"path": str(claims), "bytes": claims.stat().st_size},
        },
        "formal_kg_statistics": {
            "general": {"papers": 1, "claims": 1},
            "case_studies": {"case1_transdiagnostic": {"papers": 1, "claims": 1}},
            "quality": {},
        },
        "extracted_claim_store": {"rows": 1},
        "full_graph_taxonomy_audit": {
            "schema_version": "case_study_membership.v2",
            "is_canonical": True,
            "entities_scanned": {"claim_nodes": 1},
        },
        "case_study_membership_reaudit": {"claims_reviewed": 1},
    }
    state.write_text(json.dumps(payload), encoding="utf-8")
    hashes = {
        "knowledge_graph": sha256_file(kg),
        "extracted_claims": sha256_file(claims),
        "current_state": sha256_file(state),
    }
    return kg, claims, state, hashes


def test_canonical_release_validates_and_records_all_artifacts(tmp_path: Path) -> None:
    kg, claims, state, hashes = _release_fixture(tmp_path)
    release = validate_canonical_kg_release(
        kg_path=kg,
        claims_path=claims,
        state_path=state,
        case_study_id="case1_transdiagnostic",
        expected_sha256=hashes,
    )
    assert release["taxonomy_version"] == "case_study_membership.v2"
    assert release["case_study_statistics"] == {"papers": 1, "claims": 1}
    assert release["files"]["extracted_claims"]["sha256"] == hashes["extracted_claims"]


def test_canonical_release_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    kg, claims, state, hashes = _release_fixture(tmp_path)
    hashes["knowledge_graph"] = "0" * 64
    with pytest.raises(ValueError, match="knowledge_graph SHA-256 mismatch"):
        validate_canonical_kg_release(
            kg_path=kg,
            claims_path=claims,
            state_path=state,
            case_study_id="case1_transdiagnostic",
            expected_sha256=hashes,
        )


def test_relocated_release_requires_explicit_opt_in(tmp_path: Path) -> None:
    kg, claims, state, hashes = _release_fixture(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    archived_kg = archive / kg.name
    archived_claims = archive / claims.name
    archived_kg.write_bytes(kg.read_bytes())
    archived_claims.write_bytes(claims.read_bytes())

    with pytest.raises(ValueError, match="path does not match CURRENT_STATE.json"):
        validate_canonical_kg_release(
            kg_path=archived_kg,
            claims_path=archived_claims,
            state_path=state,
            expected_sha256=hashes,
        )

    release = validate_canonical_kg_release(
        kg_path=archived_kg,
        claims_path=archived_claims,
        state_path=state,
        expected_sha256=hashes,
        allow_relocated_artifacts=True,
    )
    assert release["relocated_artifacts"] is True
    assert release["relocated_from"] == {
        "knowledge_graph": str(kg.resolve()),
        "extracted_claims": str(claims.resolve()),
    }
