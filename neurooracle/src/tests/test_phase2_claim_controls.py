from __future__ import annotations

from neurooracle.src.claim_extractor import ClaimExtractor, EXTRACTION_PROMPT, _normalize_predicate
from neurooracle.src.claim_ingestion import (
    _is_vague_endpoint_name,
    _normalize_entity_type,
    ingest_claims,
)
from neurooracle.src.batch_extract import _collect_pubmed_abstract
from neurooracle.src.chain_extract import _select_failed_pmids, _select_second_pass_pmids
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.schema import Claim, DomainTag, Evidence, PaperRef


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
    evidence_direction: str = "",
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
            direction=evidence_direction,
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


class _FakeCache:
    def __init__(self, abstracts: dict[str, str]):
        self.abstracts = abstracts

    def get(self, pmid: str):
        abstract = self.abstracts.get(str(pmid))
        if abstract is None:
            return None
        return abstract, PaperRef(pmid=str(pmid), title="Cached paper", year=2026)


def test_retry_failed_selector_reads_error_rows(tmp_path):
    source = tmp_path / "papers_metadata.csv"
    source.write_text(
        "pmid,n_claims_extracted,extraction_error\n"
        "1,0,\n"
        "2,0,Request timed out.\n"
        "3,4,\n"
        "4,0,Connection error\n",
        encoding="utf-8",
    )

    assert _select_failed_pmids(source) == ["2", "4"]
    assert _select_failed_pmids(source, max_papers=1) == ["2"]


def test_second_pass_selector_keeps_only_likely_zero_claim_candidates(tmp_path):
    source = tmp_path / "papers_metadata.csv"
    source.write_text(
        "pmid,n_claims_extracted,extraction_error\n"
        "short,0,\n"
        "aim_only,0,\n"
        "candidate,0,\n"
        "has_claims,2,\n"
        "failed,0,Request timed out.\n",
        encoding="utf-8",
    )
    cache = _FakeCache({
        "short": "Results showed a difference.",
        "aim_only": "To compare brain imaging methods. " * 40,
        "candidate": "Background text. Results showed lower regional blood flow in dementia. " * 25,
        "has_claims": "Results showed a biomarker association. " * 25,
        "failed": "Results showed a biomarker association. " * 25,
    })

    assert _select_second_pass_pmids(
        source,
        cache,
        min_abstract_chars=200,
        require_result_cue=True,
    ) == ["candidate"]
    assert _select_second_pass_pmids(
        source,
        cache,
        min_abstract_chars=200,
        require_result_cue=False,
    ) == ["aim_only", "candidate"]


def test_claim_extractor_can_lock_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = ClaimExtractor(model="gpt-5.2", api_key="test-key", lock_model=True)
    assert extractor.lock_model is True
    assert extractor._cascade == ["gpt-5.2"]
    worker = extractor._get_worker_cascade()
    assert worker.model == "gpt-5.2"
    worker.record_failure()
    assert worker.model == "gpt-5.2"


def test_claim_extractor_prompt_has_case1_meta_analysis_guidance():
    assert "Transdiagnostic / meta-analysis neuroimaging results" in EXTRACTION_PROMPT
    assert "Large consortium, ENIGMA, meta-analysis" in EXTRACTION_PROMPT
    assert "cortical volume normative deviation" in EXTRACTION_PROMPT
    assert "regional cortical thickness" in EXTRACTION_PROMPT
    assert "multiple psychiatric disorders" in EXTRACTION_PROMPT
    assert "Do NOT split one reported bilateral or" in EXTRACTION_PROMPT
    assert "Do NOT extract explicit null findings" in EXTRACTION_PROMPT


def test_pubmed_abstract_parser_keeps_structured_result_sections():
    import xml.etree.ElementTree as ET

    article = ET.fromstring(
        """
        <PubmedArticle>
          <MedlineCitation>
            <Article>
              <Abstract>
                <AbstractText Label="BACKGROUND">Introductory rationale.</AbstractText>
                <AbstractText Label="METHODS">ENIGMA meta-analysis.</AbstractText>
                <AbstractText Label="RESULTS">Adults with MDD had thinner cortical gray matter.</AbstractText>
                <AbstractText Label="CONCLUSIONS">MDD showed cortical alterations.</AbstractText>
              </Abstract>
            </Article>
          </MedlineCitation>
        </PubmedArticle>
        """
    )

    abstract = _collect_pubmed_abstract(article)

    assert "BACKGROUND: Introductory rationale." in abstract
    assert "METHODS: ENIGMA meta-analysis." in abstract
    assert "RESULTS: Adults with MDD had thinner cortical gray matter." in abstract
    assert "CONCLUSIONS: MDD showed cortical alterations." in abstract


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


def test_claim_extractor_skips_negated_claim_items():
    extractor = ClaimExtractor(model="claude-sonnet-4-6", api_key="test-key", lock_model=True)
    paper = PaperRef(pmid="12345", title="Smoke paper", year=2026)

    claim = extractor._item_to_claim(
        {
            "subject": "cortical thickness",
            "predicate": "distinguishes",
            "object": "major depressive disorder",
            "negated": True,
            "raw_sentence": (
                "Adolescents with MDD had no differences in cortical thickness "
                "compared with matched controls."
            ),
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


def test_low_confidence_claim_is_skipped_after_background_penalty():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:lowconfbg",
        subject="Alzheimer disease",
        predicate="is_biomarker_of",
        obj="dementia",
        raw_text="Alzheimer disease has been associated with dementia.",
        confidence=0.5,
        study_type="case_control",
        subject_type="disease",
        object_type="clinical_outcome",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()
    summary = ingest_claims(kg, [result], refine_vague_predicates=False)
    assert summary["claims_marked_background"] == 1
    assert summary["claims_skipped_low_confidence"] == 1
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:lowconfbg")


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


def test_ingestion_normalizes_atom_style_entity_types():
    kg = KnowledgeGraph()
    claims = [
        _claim(
            claim_id="CLM:atomtypes1",
            subject="FDG hypometabolism smoke marker",
            predicate="predicts",
            obj="global cognitive decline smoke outcome",
            raw_text=(
                "FDG hypometabolism smoke marker predicted global cognitive "
                "decline smoke outcome."
            ),
            study_type="PET",
            subject_type="IMAGING_MARKER",
            object_type="OUTCOME",
        ),
        _claim(
            claim_id="CLM:atomtypes2",
            subject="APOE pathway smoke score",
            predicate="correlates_with",
            obj="amyloid PET smoke marker",
            raw_text="APOE pathway smoke score correlated with amyloid PET smoke marker.",
            study_type="PET",
            subject_type="GENE_TARGET",
            object_type="imaging_marker",
        ),
    ]
    result = type("R", (), {"claims": claims, "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert _normalize_entity_type("IMAGING_MARKER") == "imaging_marker"
    assert _normalize_entity_type("GENE-TARGET") == "gene_target"
    assert _normalize_entity_type("clinical event") == "clinical_event"
    assert summary["claims_added"] == 2

    imaging_marker = kg.get_concept("CLM_CONCEPT:FDG_hypometabolism_smoke_marker")
    outcome = kg.get_concept("CLM_CONCEPT:global_cognitive_decline_smoke_outcome")
    gene_target = kg.get_concept("CLM_CONCEPT:APOE_pathway_smoke_score")
    imaging_marker_2 = kg.get_concept("CLM_CONCEPT:amyloid_PET_smoke_marker")

    assert imaging_marker is not None
    assert outcome is not None
    assert gene_target is not None
    assert imaging_marker_2 is not None
    assert imaging_marker.domain_tags == [DomainTag.BIOMARKER.value]
    assert outcome.domain_tags == [DomainTag.TREATMENT_OUTCOME.value]
    assert gene_target.domain_tags == [DomainTag.GENE.value]
    assert imaging_marker_2.domain_tags == [DomainTag.BIOMARKER.value]


def test_ingestion_guard_skips_generic_method_and_imaging_entities():
    kg = KnowledgeGraph()
    claims = [
        _claim(
            claim_id="CLM:registrationmethod",
            subject="fully deformable registration methods",
            predicate="increases",
            obj="agreement between automated segmentations and expert manual segmentations",
            raw_text="Fully deformable registration methods increased agreement between automated segmentations and expert manual segmentations.",
            study_type="sMRI",
            subject_type="method",
            object_type="method_outcome",
        ),
        _claim(
            claim_id="CLM:neuroimagingmethods",
            subject="neuroimaging methods",
            predicate="distinguishes",
            obj="Alzheimer disease",
            raw_text="Neuroimaging methods may help distinguish Alzheimer disease.",
            study_type="review",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:mriscans",
            subject="multiple serial MRI scans",
            predicate="reduces",
            obj="required sample size in therapeutic trials",
            raw_text="Multiple serial MRI scans reduced required sample size in therapeutic trials.",
            study_type="sMRI",
            subject_type="method",
            object_type="method_outcome",
        ),
        _claim(
            claim_id="CLM:testbattery",
            subject="neuropsychological test battery",
            predicate="predicts",
            obj="Alzheimer disease",
            raw_text="A neuropsychological test battery was evaluated for Alzheimer disease prediction.",
            study_type="case_control",
            subject_type="clinical_marker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:methodcorrelation",
            subject="automated white matter hyperintensity quantification method",
            predicate="correlates_with",
            obj="manual white matter hyperintensity ratings",
            raw_text="The automated white matter hyperintensity quantification method correlated with manual white matter hyperintensity ratings.",
            study_type="sMRI",
            subject_type="method",
            object_type="method_outcome",
        ),
    ]
    result = type("R", (), {"claims": claims, "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 5
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:registrationmethod")
    assert not kg.has_concept("CLM:neuroimagingmethods")
    assert not kg.has_concept("CLM:mriscans")
    assert not kg.has_concept("CLM:testbattery")
    assert not kg.has_concept("CLM:methodcorrelation")


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
        _claim(
            claim_id="CLM:brainspectperfusion",
            subject="brain perfusion SPECT",
            predicate="distinguishes",
            obj="dementia",
            raw_text="Brain perfusion SPECT distinguished dementia groups.",
            study_type="SPECT",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:clockdrawingscore",
            subject="clock drawing test score",
            predicate="is_biomarker_of",
            obj="severity of dementia",
            raw_text="Clock drawing test score was associated with severity of dementia.",
            study_type="case_control",
            subject_type="clinical_marker",
            object_type="clinical_outcome",
        ),
        _claim(
            claim_id="CLM:fdgmetabolism",
            subject="FDG-PET cerebral metabolism",
            predicate="predicts",
            obj="Alzheimer disease progression",
            raw_text="FDG-PET cerebral metabolism predicted Alzheimer disease progression.",
            study_type="PET",
            subject_type="biomarker",
            object_type="disease",
        ),
        _claim(
            claim_id="CLM:mriinfarctcount",
            subject="MRI-identified cerebral infarct count",
            predicate="predicts",
            obj="dementia",
            raw_text="MRI-identified cerebral infarct count predicted dementia.",
            study_type="sMRI",
            subject_type="biomarker",
            object_type="disease",
        ),
    ]
    result = type("R", (), {"claims": claims, "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_modality_method"] == 0
    assert summary["claims_added"] == 6
    assert kg.has_concept("CLM:fdghypometabolism")
    assert kg.has_concept("CLM:datbinding")
    assert kg.has_concept("CLM:brainspectperfusion")
    assert kg.has_concept("CLM:clockdrawingscore")
    assert kg.has_concept("CLM:fdgmetabolism")
    assert kg.has_concept("CLM:mriinfarctcount")


def test_ingestion_normalizes_biomarker_abundance_in_disease():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:plaquesincrease",
        subject="amyloid-beta senile plaques",
        predicate="increases",
        obj="Alzheimer disease",
        raw_text="Amyloid-beta senile plaques show increased accumulation in Alzheimer disease.",
        subject_type="biomarker",
        object_type="disease",
        evidence_direction="increased accumulation in AD",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["predicates_refined"] == 1
    node = kg.get_concept("CLM:plaquesincrease")
    assert node is not None
    assert node.metadata["predicate"] == "is_associated_with"
    assert node.metadata["metadata"]["predicate_original"] == "increases"
    assert node.metadata["metadata"]["predicate_normalized_reason"] == "biomarker abundance in disease"


def test_ingestion_normalizes_reduced_task_measurement_in_disease_group():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:taskreduced",
        subject="word-stem completion priming",
        predicate="reduces",
        obj="Alzheimer disease",
        raw_text=(
            "Compared with normal old adults, AD patients showed reduced "
            "priming on a word-stem completion task."
        ),
        confidence=0.6,
        study_type="PET",
        subject_type="cognitive_function",
        object_type="disease",
        evidence_direction="decrease",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["predicates_refined"] == 1
    node = kg.get_concept("CLM:taskreduced")
    assert node is not None
    assert node.metadata["predicate"] == "distinguishes"
    assert node.metadata["metadata"]["predicate_original"] == "reduces"
    assert node.metadata["metadata"]["predicate_normalized_reason"] == "measurement differs in disease group"


def test_ingestion_guard_skips_disease_object_absent_from_evidence():
    kg = KnowledgeGraph()
    claim = _claim(
        claim_id="CLM:injecteddisease",
        subject="IL22",
        predicate="increases",
        obj="schizophrenia",
        raw_text=(
            "This short review examines the STAT3/AhR axis and "
            "downregulation of IL-22 and BDNF with subsequent increase in "
            "gut barrier permeability."
        ),
        confidence=0.6,
        study_type="review",
        subject_type="gene",
        object_type="disease",
    )
    result = type("R", (), {"claims": [claim], "error": ""})()

    summary = ingest_claims(kg, [result], refine_vague_predicates=False)

    assert summary["claims_skipped_unsupported_endpoint"] == 1
    assert summary["claims_added"] == 0
    assert not kg.has_concept("CLM:injecteddisease")


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
