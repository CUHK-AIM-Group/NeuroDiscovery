from __future__ import annotations

import json

import pytest

from neurooracle.src.claim_extractor import ExtractionResult
from neurooracle.src.claim_ingestion import ingest_claims, persist_ingestion_results
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import Claim, Evidence, PaperRef


def _candidate(
    paper: PaperRef,
    *,
    claim_id: str,
    subject: str,
    predicate: str,
    obj: str,
    raw_text: str,
    confidence: float,
    subject_type: str,
    object_type: str,
    study_type: str,
) -> Claim:
    return Claim(
        id=claim_id,
        subject_id="",
        subject_name=subject,
        predicate=predicate,
        object_id="",
        object_name=obj,
        confidence=confidence,
        evidence=Evidence(study_type=study_type),
        source_paper=paper,
        raw_text=raw_text,
        metadata={
            "subject_type": subject_type,
            "object_type": object_type,
        },
    )


def _mixed_result(
    *,
    namespace: str = "persistence",
    pmid: str = "99000123",
) -> ExtractionResult:
    paper = PaperRef(
        pmid=pmid,
        title="Persistence contract smoke paper",
        year=2026,
        journal="Test Journal",
    )
    return ExtractionResult(
        paper=paper,
        claims=[
            _candidate(
                paper,
                claim_id=f"CLM:{namespace}:accepted",
                subject="hippocampal volume",
                predicate="predicts",
                obj="verbal episodic memory deficit",
                raw_text=(
                    "Lower hippocampal volume predicted verbal episodic memory deficit."
                ),
                confidence=0.9,
                subject_type="biomarker",
                object_type="clinical_outcome",
                study_type="sMRI",
            ),
            _candidate(
                paper,
                claim_id=f"CLM:{namespace}:modality",
                subject="functional MRI scanning",
                predicate="distinguishes",
                obj="schizophrenia",
                raw_text="Functional MRI scanning distinguished schizophrenia.",
                confidence=0.9,
                subject_type="method",
                object_type="disease",
                study_type="fMRI",
            ),
            _candidate(
                paper,
                claim_id=f"CLM:{namespace}:low-confidence",
                subject="APOE genotype",
                predicate="predicts",
                obj="late dementia",
                raw_text="APOE genotype predicted late dementia.",
                confidence=0.1,
                subject_type="gene",
                object_type="disease",
                study_type="cohort",
            ),
        ],
    )


def test_ingestion_outcomes_drive_accepted_only_canonical_persistence(tmp_path):
    kg = KnowledgeGraph()
    result = _mixed_result()

    summary = ingest_claims(
        kg,
        [result],
        refine_vague_predicates=False,
        keep_noise=True,
        require_final_scope_audit=False,
    )

    assert summary["candidate_claims"] == 3
    assert summary["claims_added"] == 1
    assert summary["accepted_claim_ids"] == ["CLM:persistence:accepted"]
    assert [item["status"] for item in summary["claim_outcomes"]] == [
        "accepted",
        "rejected",
        "rejected",
    ]
    assert [item["reason"] for item in summary["rejected_claims"]] == [
        "modality_method_guard",
        "low_confidence",
    ]

    claims_path = tmp_path / "extracted_claims.jsonl"
    first = persist_ingestion_results(
        kg,
        [result],
        summary,
        claims_path,
        label="persistence_contract_test",
    )
    second = persist_ingestion_results(
        kg,
        [result],
        summary,
        claims_path,
        label="persistence_contract_test",
    )

    canonical = [
        json.loads(line)
        for line in claims_path.read_text(encoding="utf-8").splitlines()
    ]
    rejections = [
        json.loads(line)
        for line in (tmp_path / "claim_ingestion_rejections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [record["id"] for record in canonical] == ["CLM:persistence:accepted"]
    assert {record["claim_id"] for record in rejections} == {
        "CLM:persistence:modality",
        "CLM:persistence:low-confidence",
    }
    assert all(record["schema_version"] == "claim-ingestion-rejection.v1" for record in rejections)
    assert first["claims_written"] == 1
    assert first["rejections_written"] == 2
    assert second["claims_written"] == 0
    assert second["rejections_written"] == 0


def test_persistence_fails_closed_when_outcome_order_does_not_match(tmp_path):
    kg = KnowledgeGraph()
    result = _mixed_result(namespace="mismatch", pmid="99000456")
    summary = ingest_claims(
        kg,
        [result],
        refine_vague_predicates=False,
        keep_noise=True,
        require_final_scope_audit=False,
    )
    bad_summary = {
        **summary,
        "claim_outcomes": list(reversed(summary["claim_outcomes"])),
    }
    claims_path = tmp_path / "extracted_claims.jsonl"

    with pytest.raises(ValueError, match="outcome order mismatch"):
        persist_ingestion_results(
            kg,
            [result],
            bad_summary,
            claims_path,
            label="persistence_contract_test",
        )

    assert not claims_path.exists()
    assert not (tmp_path / "claim_ingestion_rejections.jsonl").exists()
