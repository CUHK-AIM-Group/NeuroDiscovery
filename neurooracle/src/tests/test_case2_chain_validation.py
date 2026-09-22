from __future__ import annotations

import json
from types import SimpleNamespace

from neurooracle.src.case2_chain_validation import (
    CASE2_CHAIN_VALIDATION_FIELD,
    build_case2_paper_chain_validation,
    validate_case2_paper_chain_record,
)
from neurooracle.src.case_study_membership_contract import (
    source_text_sha256,
    validate_final_scope_reaudit,
)
from neurooracle.src.case_study_membership_policy import gates_for_labels
from neurooracle.src.claim_extractor import ClaimExtractor
from neurooracle.src.schema import PaperRef


SOURCE = (
    "A polygenic risk score was associated with lower hippocampal volume at "
    "baseline. Hippocampal volume mediated the association between the score "
    "and cognitive decline over five years."
)
CHAIN_EVIDENCE = {
    "genetic_or_pathway": ["polygenic risk score"],
    "brain_imaging_or_physiology": ["hippocampal volume"],
    "longitudinal_clinical_or_cognitive_outcome": [
        "cognitive decline over five years"
    ],
    "mediation_or_causal_chain": ["mediated the association"],
}


def _case2_items() -> list[dict]:
    upstream_labels = ["case2_pathway_mediation", "imaging_genetics"]
    downstream_labels = ["case2_pathway_mediation", "prognosis"]
    return [
        {
            "subject": "polygenic risk score",
            "subject_type": "GENE_TARGET",
            "predicate": "correlates_with",
            "object": "hippocampal volume",
            "object_type": "IMAGING_MARKER",
            "raw_sentence": (
                "A polygenic risk score was associated with lower hippocampal "
                "volume at baseline."
            ),
            "case_study_ids": upstream_labels,
            "case_study_gates": gates_for_labels(upstream_labels),
            "scope_evidence_spans": [
                "polygenic risk score",
                "hippocampal volume",
            ],
            "scope_confidence": 0.97,
            "scope_decision_basis": (
                "This claim is the genetic-to-neural leg of the verified chain."
            ),
            "case2_paper_chain_evidence": CHAIN_EVIDENCE,
        },
        {
            "subject": "hippocampal volume",
            "subject_type": "IMAGING_MARKER",
            "predicate": "mediates",
            "object": "cognitive decline over five years",
            "object_type": "OUTCOME",
            "raw_sentence": (
                "Hippocampal volume mediated the association between the score "
                "and cognitive decline over five years."
            ),
            "case_study_ids": downstream_labels,
            "case_study_gates": gates_for_labels(downstream_labels),
            "scope_evidence_spans": [
                "mediated the association",
                "cognitive decline over five years",
            ],
            "scope_confidence": 0.98,
            "scope_decision_basis": (
                "This claim states the neural mediation and longitudinal outcome leg."
            ),
            "case2_paper_chain_evidence": CHAIN_EVIDENCE,
        },
    ]


def test_case2_paper_chain_requires_all_grounded_components():
    items = _case2_items()
    context_hash = source_text_sha256(SOURCE)
    record = build_case2_paper_chain_validation(
        items,
        source_text=SOURCE,
        source_context_sha256=context_hash,
    )

    assert record["valid"] is True
    assert record["participating_claim_indices"] == [0, 1]
    assert validate_case2_paper_chain_record(
        record,
        source_context_sha256=context_hash,
        claim_index=1,
        case2_labeled=True,
    )

    incomplete = _case2_items()
    for item in incomplete:
        item["case2_paper_chain_evidence"] = {
            **CHAIN_EVIDENCE,
            "mediation_or_causal_chain": [],
        }
    rejected = build_case2_paper_chain_validation(
        incomplete,
        source_text=SOURCE,
        source_context_sha256=context_hash,
    )
    assert rejected["valid"] is False
    assert "mediation_or_causal_chain:missing" in rejected["reasons"]


def test_case2_component_claims_are_sealed_without_required_chain_record():
    extractor = ClaimExtractor(
        model="luna",
        api_key="test-key",
        lock_model=False,
        strict_scope_audit=True,
    )
    items = _case2_items()
    for item in items:
        item.pop("case2_paper_chain_evidence")
    claims = extractor._parse_response(
        json.dumps(items),
        PaperRef(pmid="case2-chain", title="Complete chain", year=2026),
        scope_source_text=SOURCE,
        scope_reviewer_id="luna",
    )

    assert extractor.lock_model is True
    assert len(claims) == 2
    assert claims[0].claim_case_study_ids == [
        "case2_pathway_mediation",
        "imaging_genetics",
    ]
    assert claims[1].claim_case_study_ids == [
        "case2_pathway_mediation",
        "prognosis",
    ]
    assert all(CASE2_CHAIN_VALIDATION_FIELD not in claim.metadata for claim in claims)
    assert all(validate_final_scope_reaudit(claim.to_dict()) for claim in claims)


def test_case2_parse_does_not_require_complete_same_paper_chain():
    extractor = ClaimExtractor(
        model="luna",
        api_key="test-key",
        lock_model=True,
        strict_scope_audit=True,
    )
    items = _case2_items()
    for item in items:
        item["case2_paper_chain_evidence"] = {
            **CHAIN_EVIDENCE,
            "longitudinal_clinical_or_cognitive_outcome": [],
        }
    claims = extractor._parse_response(
        json.dumps(items),
        PaperRef(pmid="case2-incomplete", title="Incomplete chain", year=2026),
        scope_source_text=SOURCE,
    )
    assert len(claims) == 2
    assert all("case2_pathway_mediation" in claim.claim_case_study_ids for claim in claims)


def test_strict_extraction_sends_the_complete_abstract():
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content="[]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_completions = FakeCompletions()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )
    extractor = ClaimExtractor(
        model="luna",
        api_key="test-key",
        lock_model=True,
        strict_scope_audit=True,
    )
    extractor._clients = [fake_client]
    abstract = "BEGIN " + ("complete abstract evidence " * 150) + "FINAL_SENTINEL"

    result = extractor.extract_from_abstract(
        abstract,
        PaperRef(pmid="full-abstract", title="Full abstract", year=2026),
    )

    assert not result.error
    prompt = fake_completions.kwargs["messages"][1]["content"]
    assert abstract in prompt
    assert "FINAL_SENTINEL" in prompt
    assert fake_completions.kwargs["temperature"] == 0.0
