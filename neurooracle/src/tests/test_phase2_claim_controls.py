from __future__ import annotations

from neurooracle.src.claim_extractor import ClaimExtractor, _normalize_predicate
from neurooracle.src.claim_ingestion import _is_vague_endpoint_name, ingest_claims
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.schema import Claim, Evidence, PaperRef


def _claim(
    *,
    claim_id: str,
    subject: str,
    predicate: str,
    obj: str,
    raw_text: str,
    confidence: float = 0.6,
    study_type: str = "",
    sample_size: int | None = None,
    subject_type: str = "gene",
    object_type: str = "disease",
) -> Claim:
    return Claim(
        id=claim_id,
        subject_id="",
        subject_name=subject,
        predicate=predicate,
        object_id="",
        object_name=obj,
        confidence=confidence,
        evidence=Evidence(
            study_type=study_type,
            sample_size=sample_size,
        ),
        source_paper=PaperRef(
            pmid="12345",
            title="Smoke paper",
            year=2026,
            journal="Test Journal",
        ),
        raw_text=raw_text,
        metadata={
            "subject_type": subject_type,
            "object_type": object_type,
        },
    )


def test_claim_extractor_can_lock_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = ClaimExtractor(model="gpt-5.2", api_key="test-key", lock_model=True)
    assert extractor.lock_model is True
    assert extractor._cascade == ["gpt-5.2"]
    worker = extractor._get_worker_cascade()
    assert worker.model == "gpt-5.2"
    worker.record_failure()
    assert worker.model == "gpt-5.2"


def test_claim_extractor_rejects_non_closed_predicates():
    extractor = ClaimExtractor(model="claude-sonnet-4-6", api_key="test-key", lock_model=True)
    paper = PaperRef(pmid="12345", title="Smoke paper", year=2026)

    assert _normalize_predicate("compensates_for") == "modulates"
    assert _normalize_predicate("precedes") == "predicts"

    claim = extractor._item_to_claim(
        {
            "subject": "newly activated regions",
            "predicate": "not_a_real_predicate",
            "object": "minor brain deterioration",
            "raw_sentence": "New regions appeared in response to minor brain deterioration.",
        },
        paper,
    )
    assert claim is None


def test_claim_extractor_normalizes_directional_association_endpoint():
    extractor = ClaimExtractor(model="claude-sonnet-4-6", api_key="test-key", lock_model=True)
    paper = PaperRef(pmid="12345", title="Smoke paper", year=2026)

    claim = extractor._item_to_claim(
        {
            "subject": "regional cerebral blood flow",
            "predicate": "reduces",
            "object": "severity of dementia",
            "raw_sentence": (
                "Lower regional cerebral blood flow is related to the severity "
                "of dementia and survival."
            ),
        },
        paper,
    )

    assert claim is not None
    assert claim.predicate == "correlates_with"


def test_background_review_gene_claim_is_skipped():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:bgskip",
        subject="APOE",
        predicate="gene_associated_with_disease",
        obj="Alzheimer disease",
        raw_text="APOE is an established risk factor for Alzheimer disease.",
        study_type="review",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()
    summary = ingest_claims(kg, [result], refine_vague_predicates=False)
    assert summary["claims_skipped_background"] == 1
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:bgskip")


def test_background_suspect_claim_is_downweighted_not_rewritten():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:bgmark",
        subject="APOE",
        predicate="is_risk_factor_for",
        obj="Alzheimer disease",
        raw_text=(
            "In a separate study, APOE epsilon4 gene dose, an established "
            "Alzheimer disease risk factor, was correlated with hypometabolism."
        ),
        study_type="PET",
        sample_size=141,
    )
    result = type("R", (), {"claims": [claim], "error": ""})()
    summary = ingest_claims(kg, [result], refine_vague_predicates=False)
    assert summary["claims_marked_background"] == 1
    assert summary["claims_added"] == 1
    stored = kg.get_concept("CLM:bgmark")
    assert stored is not None
    stored_meta = stored.metadata
    assert stored_meta["predicate"] == "is_risk_factor_for"
    assert stored_meta["metadata"]["background_suspect"] is True
    assert stored_meta["confidence"] == 0.3


def test_noise_words_do_not_reject_specific_measurable_phrases():
    assert HypothesisEngine._is_noisy_entity("risk")
    assert HypothesisEngine._is_noisy_entity("change")
    assert HypothesisEngine._is_noisy_entity("volume")
    assert HypothesisEngine._is_noisy_entity("quality of recovery")

    assert not HypothesisEngine._is_noisy_entity("polygenic risk score")
    assert not HypothesisEngine._is_noisy_entity("hippocampal complex cortical thickness change")
    assert not HypothesisEngine._is_noisy_entity("entorhinal cortex thickness change")
    assert not HypothesisEngine._is_noisy_entity("hippocampus volume")


def test_vague_endpoint_keeps_specific_cognitive_clinical_phenotypes():
    assert _is_vague_endpoint_name("deficit")
    assert _is_vague_endpoint_name("cognitive deficit")
    assert _is_vague_endpoint_name("verbal deficit")
    assert _is_vague_endpoint_name("clinical features")
    assert _is_vague_endpoint_name("overall cognitive decline")

    assert not _is_vague_endpoint_name("memory deficit")
    assert not _is_vague_endpoint_name("verbal episodic memory deficit")
    assert not _is_vague_endpoint_name("visual memory deficits")
    assert not _is_vague_endpoint_name("24-month MMSE decline")
    assert not _is_vague_endpoint_name("hippocampal dysfunction")
    assert not _is_vague_endpoint_name("Alzheimer disease cognitive decline")


def test_ingestion_keeps_specific_deficit_endpoint():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:specificdeficit",
        subject="hippocampal volume",
        predicate="predicts",
        obj="verbal episodic memory deficit",
        raw_text="Lower hippocampal volume predicted verbal episodic memory deficit.",
        study_type="sMRI",
        subject_type="biomarker",
        object_type="cognitive_function",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_added"] == 1
    assert summary["entities_dropped"] == 0
    assert kg.has_concept("CLM_CONCEPT:verbal_episodic_memory_deficit")


def test_ingestion_keeps_specific_plural_deficits_endpoint():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:specificdeficits",
        subject="verbal memory impairment",
        predicate="predicts",
        obj="visual memory deficits",
        raw_text="Verbal memory impairment preceded visual memory deficits.",
        study_type="case_control",
        subject_type="cognitive_function",
        object_type="cognitive_function",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_added"] == 1
    assert summary["entities_dropped"] == 0
    assert kg.has_concept("CLM_CONCEPT:visual_memory_deficits")


def test_ingestion_guard_skips_pure_modality_subjects():
    kg = KnowledgeGraph()
    claims = [
        _claim(
            claim_id="CLM:ctmodality",
            subject="computed tomography",
            predicate="is_biomarker_of",
            obj="reversible causes of dementia",
            raw_text="Computed tomography is used to exclude reversible causes of dementia.",
            study_type="CT",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:fmrimodality",
            subject="functional magnetic resonance imaging",
            predicate="predicts",
            obj="Alzheimer disease",
            raw_text="Functional magnetic resonance imaging was used in Alzheimer disease.",
            study_type="fMRI",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:spectscan",
            subject="single-photon emission tomography scanning",
            predicate="distinguishes",
            obj="Alzheimer disease",
            raw_text="Single-photon emission tomography scanning can distinguish Alzheimer disease.",
            study_type="SPECT",
            subject_type="biomarker",
            object_type="disease",
        ),
    ]
    result = type("R", (), {"claims": claims, "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 3
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:ctmodality")
    assert not kg.has_concept("CLM:fmrimodality")
    assert not kg.has_concept("CLM:spectscan")


def test_ingestion_guard_skips_modality_object_without_measurement():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:modalityobject",
        subject="APOE",
        predicate="distinguishes",
        obj="quantitative MRI measurements",
        raw_text="APOE status differed across quantitative MRI measurements.",
        study_type="sMRI",
        subject_type="gene",
        object_type="biomarker",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 1
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:modalityobject")


def test_ingestion_guard_keeps_modality_derived_measurements():
    kg = KnowledgeGraph()
    claims = [
        _claim(
            claim_id="CLM:fdghypometabolism",
            subject="FDG hypometabolism",
            predicate="is_biomarker_of",
            obj="Alzheimer disease",
            raw_text="FDG hypometabolism was associated with Alzheimer disease.",
            study_type="PET",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:datbinding",
            subject="dopamine transporter binding",
            predicate="is_biomarker_of",
            obj="dementia with Lewy bodies",
            raw_text="Reduced dopamine transporter binding distinguished dementia with Lewy bodies.",
            study_type="SPECT",
            subject_type="biomarker",
            object_type="disease",
        ),
    ]
    result = type("R", (), {"claims": claims, "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 0
    assert summary["claims_added"] == 2
    assert kg.has_concept("CLM:fdghypometabolism")
    assert kg.has_concept("CLM:datbinding")


def test_ingestion_guard_skips_procedure_as_treatment_subject():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:injectionprocedure",
        subject="intraperitoneal injection of macrophage M2 cells",
        predicate="treats",
        obj="motor defect",
        raw_text="Intraperitoneal injection of macrophage M2 cells improved motor defects.",
        study_type="animal",
        subject_type="intervention",
        object_type="clinical_outcome",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 1
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:injectionprocedure")
