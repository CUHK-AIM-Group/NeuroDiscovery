"""Manual Case Study 1 claim curation for transdiagnostic neuroimaging papers.

This module is intentionally LLM-free. It converts hand-reviewed PubMed
abstract findings into the same Claim objects used by Phase 2 extraction, so
the downstream ingestion, entity resolution, deduplication, and graph edge
generation stay identical to automatic extraction.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .abstract_cache import AbstractCache, default_cache_path
from .case_targeted_extract import _resolve_data_paths
from .claim_extractor import ExtractionResult
from .claim_ingestion import ingest_claims
from .schema import Claim, Evidence, PaperRef
from .storage import load_graph, save_graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualClaimSpec:
    pmid: str
    subject: str
    predicate: str
    object: str
    raw_sentence: str
    direction: str
    subject_type: str = "IMAGING_MARKER"
    object_type: str = "DISEASE"
    study_type: str = "meta_analysis"
    methodology: str = "manual Case Study 1 curation from PubMed abstract"
    effect_metric: str = ""
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    confidence: float = 0.8
    conditions: tuple[str, ...] = ()


CURATED_CASE1_CLAIMS: tuple[ManualClaimSpec, ...] = (
    ManualClaimSpec(
        pmid="32857118",
        subject="shared cortical thickness group-difference profile",
        predicate="distinguishes",
        object="multiple psychiatric disorders",
        object_type="DISEASE",
        sample_size=28321,
        direction="shared cortical thickness difference profile across ADHD, ASD, bipolar disorder, major depressive disorder, obsessive-compulsive disorder, and schizophrenia",
        raw_sentence=(
            "Principal component analysis revealed a shared profile of difference "
            "in cortical thickness across the 6 disorders (48% variance explained)."
        ),
    ),
    ManualClaimSpec(
        pmid="32857118",
        subject="cortical thickness group-difference profile",
        predicate="is_associated_with",
        object="multiple psychiatric disorders",
        object_type="DISEASE",
        sample_size=28321,
        direction="cortical thickness group-difference profiles were linked to shared neurobiologic processes across multiple psychiatric disorders",
        raw_sentence=(
            "In this study, shared neurobiologic processes were associated with "
            "differences in cortical thickness across multiple psychiatric disorders."
        ),
    ),
    ManualClaimSpec(
        pmid="30988201",
        subject="frontoparietal network connectivity",
        predicate="is_associated_with",
        object="affective and psychotic pathology",
        object_type="OUTCOME",
        study_type="fMRI",
        sample_size=1010,
        direction="graded disruptions in frontoparietal network connectivity were associated with affective and psychotic illnesses",
        raw_sentence=(
            "The presence of affective and psychotic illnesses was associated with "
            "graded disruptions in frontoparietal network connectivity "
            "(encompassing aspects of dorsolateral prefrontal, dorsomedial "
            "prefrontal, lateral parietal, and posterior temporal cortices)."
        ),
    ),
    ManualClaimSpec(
        pmid="30988201",
        subject="default network integrity",
        predicate="distinguishes",
        object="psychotic illness",
        object_type="DISEASE",
        study_type="fMRI",
        sample_size=1010,
        direction="default network integrity was preferentially disrupted in psychotic illness",
        raw_sentence=(
            "Conversely, other properties of network connectivity, including "
            "default network integrity, were preferentially disrupted in patients "
            "with psychotic illness, but not patients without psychotic symptoms."
        ),
    ),
    ManualClaimSpec(
        pmid="33879764",
        subject="cortical volume normative deviation",
        predicate="predicts",
        object="psychopathology dimensions",
        object_type="OUTCOME",
        study_type="cohort",
        methodology="manual Case Study 1 curation from PubMed abstract; normative modeling",
        direction="multivariate normative deviations improved out-of-sample prediction of psychopathology dimensions compared with raw cortical volume",
        raw_sentence=(
            "We found that multivariate patterns of deviations yielded improved "
            "out-of-sample prediction of psychopathology dimensions compared to "
            "multivariate patterns of raw cortical volume."
        ),
    ),
    ManualClaimSpec(
        pmid="33879764",
        subject="ventromedial prefrontal cortical volume normative deviation",
        predicate="correlates_with",
        object="overall psychopathology",
        object_type="OUTCOME",
        study_type="cohort",
        methodology="manual Case Study 1 curation from PubMed abstract; normative modeling",
        direction="stronger correlation with overall psychopathology than with specific dimensions",
        raw_sentence=(
            "We also found that correlations between overall psychopathology and "
            "deviations in ventromedial prefrontal, inferior temporal, and dorsal "
            "anterior cingulate cortices were stronger than those observed for "
            "specific dimensions of psychopathology (e.g., anxious-misery)."
        ),
    ),
    ManualClaimSpec(
        pmid="33879764",
        subject="inferior temporal cortical volume normative deviation",
        predicate="correlates_with",
        object="overall psychopathology",
        object_type="OUTCOME",
        study_type="cohort",
        methodology="manual Case Study 1 curation from PubMed abstract; normative modeling",
        direction="stronger correlation with overall psychopathology than with specific dimensions",
        raw_sentence=(
            "We also found that correlations between overall psychopathology and "
            "deviations in ventromedial prefrontal, inferior temporal, and dorsal "
            "anterior cingulate cortices were stronger than those observed for "
            "specific dimensions of psychopathology (e.g., anxious-misery)."
        ),
    ),
    ManualClaimSpec(
        pmid="33879764",
        subject="dorsal anterior cingulate cortical volume normative deviation",
        predicate="correlates_with",
        object="overall psychopathology",
        object_type="OUTCOME",
        study_type="cohort",
        methodology="manual Case Study 1 curation from PubMed abstract; normative modeling",
        direction="stronger correlation with overall psychopathology than with specific dimensions",
        raw_sentence=(
            "We also found that correlations between overall psychopathology and "
            "deviations in ventromedial prefrontal, inferior temporal, and dorsal "
            "anterior cingulate cortices were stronger than those observed for "
            "specific dimensions of psychopathology (e.g., anxious-misery)."
        ),
    ),
    ManualClaimSpec(
        pmid="33879764",
        subject="cortical volume normative deviation",
        predicate="distinguishes",
        object="depression and attention-deficit hyperactivity disorder",
        object_type="DISEASE",
        study_type="cohort",
        methodology="manual Case Study 1 curation from PubMed abstract; normative modeling",
        direction="spatially overlapping effects between depression and ADHD diminished after controlling for overall psychopathology",
        raw_sentence=(
            "We observed spatially overlapping effects between these groups that "
            "diminished when controlling for overall psychopathology."
        ),
    ),
    ManualClaimSpec(
        pmid="32539527",
        subject="hippocampal volume",
        predicate="distinguishes",
        object="attention-deficit hyperactivity disorder",
        sample_size=12201,
        direction="children with ADHD had smaller hippocampal volumes compared with children with OCD",
        raw_sentence=(
            "Children with ADHD compared with those with OCD had smaller "
            "hippocampal volumes, possibly influenced by IQ."
        ),
        conditions=("children", "ADHD compared with OCD"),
    ),
    ManualClaimSpec(
        pmid="32539527",
        subject="intracranial volume",
        predicate="distinguishes",
        object="attention-deficit hyperactivity disorder",
        sample_size=12201,
        direction="children and adolescents with ADHD had smaller intracranial volume than controls and OCD or ASD groups",
        raw_sentence=(
            "Children and adolescents with ADHD also had smaller intracranial "
            "volume than control subjects and those with OCD or ASD."
        ),
        conditions=("children and adolescents",),
    ),
    ManualClaimSpec(
        pmid="32539527",
        subject="frontal cortical thickness",
        predicate="distinguishes",
        object="autism spectrum disorder",
        sample_size=12201,
        direction="adults with ASD had thicker frontal cortices than controls and other clinical groups",
        raw_sentence=(
            "Adults with ASD showed thicker frontal cortices compared with adult "
            "control subjects and other clinical groups."
        ),
        conditions=("adults",),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="cortical thickness",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=9572,
        effect_metric="Cohen's d",
        effect_size=-0.53,
        direction="widespread thinner cortex in schizophrenia compared with healthy volunteers",
        raw_sentence=(
            "Compared with healthy volunteers, individuals with schizophrenia "
            "have widespread thinner cortex (left/right hemisphere: Cohen's d = "
            "-0.530/-0.516) and smaller surface area (left/right hemisphere: "
            "Cohen's d = -0.251/-0.254), with the largest effect sizes for both "
            "in frontal and temporal lobe regions."
        ),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="cortical surface area",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=9572,
        effect_metric="Cohen's d",
        effect_size=-0.251,
        direction="smaller cortical surface area in schizophrenia compared with healthy volunteers",
        raw_sentence=(
            "Compared with healthy volunteers, individuals with schizophrenia "
            "have widespread thinner cortex (left/right hemisphere: Cohen's d = "
            "-0.530/-0.516) and smaller surface area (left/right hemisphere: "
            "Cohen's d = -0.251/-0.254), with the largest effect sizes for both "
            "in frontal and temporal lobe regions."
        ),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="regional cortical thickness",
        predicate="correlates_with",
        object="normalized medication dose",
        object_type="INDIVIDUAL_DATA",
        sample_size=9572,
        direction="negative correlation",
        raw_sentence=(
            "Regional cortical thickness showed significant negative correlations "
            "with normalized medication dose, symptom severity, and duration of "
            "illness and positive correlations with age at onset."
        ),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="regional cortical thickness",
        predicate="correlates_with",
        object="symptom severity",
        object_type="OUTCOME",
        sample_size=9572,
        direction="negative correlation",
        raw_sentence=(
            "Regional cortical thickness showed significant negative correlations "
            "with normalized medication dose, symptom severity, and duration of "
            "illness and positive correlations with age at onset."
        ),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="regional cortical thickness",
        predicate="correlates_with",
        object="duration of illness",
        object_type="OUTCOME",
        sample_size=9572,
        direction="negative correlation",
        raw_sentence=(
            "Regional cortical thickness showed significant negative correlations "
            "with normalized medication dose, symptom severity, and duration of "
            "illness and positive correlations with age at onset."
        ),
    ),
    ManualClaimSpec(
        pmid="29960671",
        subject="regional cortical thickness",
        predicate="correlates_with",
        object="age at onset",
        object_type="INDIVIDUAL_DATA",
        sample_size=9572,
        direction="positive correlation",
        raw_sentence=(
            "Regional cortical thickness showed significant negative correlations "
            "with normalized medication dose, symptom severity, and duration of "
            "illness and positive correlations with age at onset."
        ),
    ),
    ManualClaimSpec(
        pmid="27137745",
        subject="cortical gray matter thickness in orbitofrontal cortex, cingulate cortex, insula, and temporal lobes",
        predicate="distinguishes",
        object="major depressive disorder",
        sample_size=10105,
        effect_metric="Cohen's d",
        direction="adults with MDD had thinner cortical gray matter than controls",
        raw_sentence=(
            "Adults with MDD had thinner cortical gray matter than controls in "
            "the orbitofrontal cortex (OFC), anterior and posterior cingulate, "
            "insula and temporal lobes (Cohen's d effect sizes: -0.10 to -0.14)."
        ),
        conditions=("adults",),
    ),
    ManualClaimSpec(
        pmid="27137745",
        subject="total cortical surface area",
        predicate="distinguishes",
        object="major depressive disorder",
        sample_size=10105,
        direction="adolescents with MDD had lower total surface area than matched controls",
        raw_sentence=(
            "Compared to matched controls, adolescents with MDD had lower total "
            "surface area (but no differences in cortical thickness) and regional "
            "reductions in frontal regions (medial OFC and superior frontal gyrus) "
            "and primary and higher-order visual, somatosensory and motor areas "
            "(d: -0.26 to -0.57)."
        ),
        conditions=("adolescents",),
    ),
    ManualClaimSpec(
        pmid="27137745",
        subject="regional cortical surface area in frontal, visual, somatosensory, and motor areas",
        predicate="distinguishes",
        object="major depressive disorder",
        sample_size=10105,
        direction="adolescents with MDD had regional surface area reductions",
        raw_sentence=(
            "Compared to matched controls, adolescents with MDD had lower total "
            "surface area (but no differences in cortical thickness) and regional "
            "reductions in frontal regions (medial OFC and superior frontal gyrus) "
            "and primary and higher-order visual, somatosensory and motor areas "
            "(d: -0.26 to -0.57)."
        ),
        conditions=("adolescents",),
    ),
    ManualClaimSpec(
        pmid="26122586",
        subject="hippocampal volume",
        predicate="distinguishes",
        object="major depressive disorder",
        sample_size=8927,
        effect_metric="Cohen's d",
        effect_size=-0.14,
        direction="MDD patients had significantly lower hippocampal volumes than controls",
        raw_sentence=(
            "Relative to controls, patients had significantly lower hippocampal "
            "volumes (Cohen's d=-0.14, % difference=-1.24)."
        ),
    ),
    ManualClaimSpec(
        pmid="26122586",
        subject="hippocampal volume",
        predicate="distinguishes",
        object="recurrent major depressive disorder",
        sample_size=8927,
        effect_metric="Cohen's d",
        effect_size=-0.17,
        direction="lower hippocampal volume effect was driven by recurrent MDD",
        raw_sentence=(
            "This effect was driven by patients with recurrent MDD (Cohen's "
            "d=-0.17, % difference=-1.44), and we detected no differences "
            "between first episode patients and controls."
        ),
    ),
    ManualClaimSpec(
        pmid="26122586",
        subject="hippocampal volume",
        predicate="correlates_with",
        object="age at onset",
        object_type="INDIVIDUAL_DATA",
        sample_size=8927,
        effect_metric="Cohen's d",
        effect_size=-0.20,
        direction="age of onset <=21 was associated with smaller hippocampus",
        raw_sentence=(
            "Age of onset <=21 was associated with a smaller hippocampus "
            "(Cohen's d=-0.20, % difference=-1.85) and a trend toward smaller "
            "amygdala (Cohen's d=-0.11, % difference=-1.23) and larger lateral "
            "ventricles (Cohen's d=0.12, % difference=5.11)."
        ),
    ),
    ManualClaimSpec(
        pmid="26122586",
        subject="caudate volume",
        predicate="correlates_with",
        object="antipsychotic medication use",
        object_type="INDIVIDUAL_DATA",
        sample_size=8927,
        direction="higher proportion of antipsychotic medication users was associated with larger caudate volumes in MDD samples",
        raw_sentence=(
            "Samples with a higher proportion of antipsychotic medication users "
            "showed larger caudate volumes in MDD patients compared with controls."
        ),
    ),
    ManualClaimSpec(
        pmid="28461699",
        subject="frontal, temporal, and parietal cortical gray matter thickness",
        predicate="distinguishes",
        object="bipolar disorder",
        sample_size=6503,
        direction="BD patients had thinner cortical gray matter in frontal, temporal, and parietal regions",
        raw_sentence=(
            "In BD, cortical gray matter was thinner in frontal, temporal and "
            "parietal regions of both brain hemispheres."
        ),
    ),
    ManualClaimSpec(
        pmid="28461699",
        subject="cortical thickness in frontal, medial parietal, and occipital regions",
        predicate="correlates_with",
        object="duration of illness",
        object_type="OUTCOME",
        sample_size=6503,
        direction="longer duration of illness was associated with reduced cortical thickness",
        raw_sentence=(
            "Longer duration of illness (after accounting for age at the time of "
            "scanning) was associated with reduced cortical thickness in frontal, "
            "medial parietal and occipital regions."
        ),
    ),
    ManualClaimSpec(
        pmid="28461699",
        subject="cortical thickness and surface area",
        predicate="correlates_with",
        object="lithium, antiepileptic, and antipsychotic treatment",
        object_type="DRUG",
        sample_size=6503,
        direction="commonly prescribed medications were associated with cortical thickness and surface area",
        raw_sentence=(
            "We found that several commonly prescribed medications, including "
            "lithium, antiepileptic and antipsychotic treatment showed significant "
            "associations with cortical thickness and surface area, even after "
            "accounting for patients who received multiple medications."
        ),
    ),
    ManualClaimSpec(
        pmid="28461699",
        subject="cortical surface area",
        predicate="correlates_with",
        object="history of psychosis",
        object_type="OUTCOME",
        sample_size=6503,
        direction="history of psychosis was associated with reduced cortical surface area",
        raw_sentence=(
            "We found evidence of reduced cortical surface area associated with a "
            "history of psychosis but no associations with mood state at the time "
            "of scanning."
        ),
    ),
    ManualClaimSpec(
        pmid="26033243",
        subject="hippocampal volume",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=4568,
        effect_metric="Cohen's d",
        effect_size=-0.46,
        direction="schizophrenia patients had smaller hippocampal volumes than controls",
        raw_sentence=(
            "Compared with healthy controls, patients with schizophrenia had "
            "smaller hippocampus (Cohen's d=-0.46), amygdala (d=-0.31), "
            "thalamus (d=-0.31), accumbens (d=-0.25) and intracranial volumes "
            "(d=-0.12), as well as larger pallidum (d=0.21) and lateral "
            "ventricle volumes (d=0.37)."
        ),
    ),
    ManualClaimSpec(
        pmid="26033243",
        subject="amygdala volume",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=4568,
        effect_metric="Cohen's d",
        effect_size=-0.31,
        direction="schizophrenia patients had smaller amygdala volumes than controls",
        raw_sentence=(
            "Compared with healthy controls, patients with schizophrenia had "
            "smaller hippocampus (Cohen's d=-0.46), amygdala (d=-0.31), "
            "thalamus (d=-0.31), accumbens (d=-0.25) and intracranial volumes "
            "(d=-0.12), as well as larger pallidum (d=0.21) and lateral "
            "ventricle volumes (d=0.37)."
        ),
    ),
    ManualClaimSpec(
        pmid="26033243",
        subject="thalamus volume",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=4568,
        effect_metric="Cohen's d",
        effect_size=-0.31,
        direction="schizophrenia patients had smaller thalamus volumes than controls",
        raw_sentence=(
            "Compared with healthy controls, patients with schizophrenia had "
            "smaller hippocampus (Cohen's d=-0.46), amygdala (d=-0.31), "
            "thalamus (d=-0.31), accumbens (d=-0.25) and intracranial volumes "
            "(d=-0.12), as well as larger pallidum (d=0.21) and lateral "
            "ventricle volumes (d=0.37)."
        ),
    ),
    ManualClaimSpec(
        pmid="26033243",
        subject="lateral ventricle volume",
        predicate="distinguishes",
        object="schizophrenia",
        sample_size=4568,
        effect_metric="Cohen's d",
        effect_size=0.37,
        direction="schizophrenia patients had larger lateral ventricle volumes than controls",
        raw_sentence=(
            "Compared with healthy controls, patients with schizophrenia had "
            "smaller hippocampus (Cohen's d=-0.46), amygdala (d=-0.31), "
            "thalamus (d=-0.31), accumbens (d=-0.25) and intracranial volumes "
            "(d=-0.12), as well as larger pallidum (d=0.21) and lateral "
            "ventricle volumes (d=0.37)."
        ),
    ),
    ManualClaimSpec(
        pmid="26033243",
        subject="putamen and pallidum volume",
        predicate="correlates_with",
        object="duration of illness",
        object_type="OUTCOME",
        sample_size=4568,
        direction="putamen and pallidum volume augmentations were positively associated with duration of illness",
        raw_sentence=(
            "Putamen and pallidum volume augmentations were positively associated "
            "with duration of illness and hippocampal deficits scaled with the "
            "proportion of unmedicated patients."
        ),
    ),
    ManualClaimSpec(
        pmid="29377733",
        subject="transverse temporal cortex surface area",
        predicate="distinguishes",
        object="obsessive-compulsive disorder",
        sample_size=3665,
        direction="adult OCD patients had lower transverse temporal cortex surface area than controls",
        raw_sentence=(
            "In adult OCD patients versus controls, we found a significantly lower "
            "surface area for the transverse temporal cortex and a thinner "
            "inferior parietal cortex."
        ),
        conditions=("adults",),
    ),
    ManualClaimSpec(
        pmid="29377733",
        subject="inferior parietal cortical thickness",
        predicate="distinguishes",
        object="obsessive-compulsive disorder",
        sample_size=3665,
        direction="adult OCD patients had thinner inferior parietal cortex than controls",
        raw_sentence=(
            "In adult OCD patients versus controls, we found a significantly lower "
            "surface area for the transverse temporal cortex and a thinner "
            "inferior parietal cortex."
        ),
        conditions=("adults",),
    ),
    ManualClaimSpec(
        pmid="29377733",
        subject="whole-brain cortical thickness",
        predicate="correlates_with",
        object="medication status",
        object_type="INDIVIDUAL_DATA",
        sample_size=3665,
        direction="medicated adult OCD patients showed thinner cortices throughout the brain",
        raw_sentence=(
            "Medicated adult OCD patients also showed thinner cortices throughout "
            "the brain."
        ),
        conditions=("medicated adults",),
    ),
    ManualClaimSpec(
        pmid="29377733",
        subject="inferior and superior parietal cortical thickness",
        predicate="distinguishes",
        object="obsessive-compulsive disorder",
        sample_size=3665,
        direction="pediatric OCD patients had thinner inferior and superior parietal cortices than controls",
        raw_sentence=(
            "In pediatric OCD patients compared with controls, we found "
            "significantly thinner inferior and superior parietal cortices, but "
            "none of the regions analyzed showed significant differences in "
            "surface area."
        ),
        conditions=("pediatric",),
    ),
    ManualClaimSpec(
        pmid="29377733",
        subject="frontal cortical surface area",
        predicate="correlates_with",
        object="medication status",
        object_type="INDIVIDUAL_DATA",
        sample_size=3665,
        direction="medicated pediatric OCD patients had lower frontal surface area",
        raw_sentence=(
            "However, medicated pediatric OCD patients had lower surface area in "
            "frontal regions."
        ),
        conditions=("medicated pediatric patients",),
    ),
)


def _spec_from_dict(row: dict) -> ManualClaimSpec:
    allowed = set(ManualClaimSpec.__dataclass_fields__)
    data = {k: v for k, v in row.items() if k in allowed}
    if "conditions" in data and isinstance(data["conditions"], list):
        data["conditions"] = tuple(str(x) for x in data["conditions"])
    return ManualClaimSpec(**data)


def load_external_manual_claim_specs(path: Path) -> tuple[ManualClaimSpec, ...]:
    """Load additional manual Case Study 1 claim specs from JSONL."""
    if not path.exists():
        return ()
    specs: list[ManualClaimSpec] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                specs.append(_spec_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid manual Case Study 1 claim spec at {path}:{lineno}: {exc}") from exc
    return tuple(specs)


def _claim_id(pmid: str, ordinal: int) -> str:
    return f"CLM:CASE1MAN:{pmid}:{ordinal:03d}"


def _claim_from_spec(spec: ManualClaimSpec, paper: PaperRef, ordinal: int) -> Claim:
    return Claim(
        id=_claim_id(spec.pmid, ordinal),
        subject_id="",
        subject_name=spec.subject,
        predicate=spec.predicate,
        object_id="",
        object_name=spec.object,
        negated=False,
        confidence=spec.confidence,
        evidence=Evidence(
            study_type=spec.study_type,
            methodology=spec.methodology,
            effect_size=spec.effect_size,
            effect_metric=spec.effect_metric,
            sample_size=spec.sample_size,
            replicability="manual_abstract_curated",
            direction=spec.direction,
        ),
        source_paper=paper,
        raw_text=spec.raw_sentence,
        paper_scope=["case1"],
        metadata={
            "subject_type": spec.subject_type,
            "object_type": spec.object_type,
            "subject_canonical_hint": "",
            "object_canonical_hint": "",
            "subject_atlas": "",
            "object_atlas": "",
            "conditions": list(spec.conditions),
            "population": None,
            "raw_stats": {},
            "manual_curation": True,
            "curation_scope": "case1_transdiagnostic",
        },
    )


def build_manual_case1_results(
    cache: AbstractCache,
    pmids: Optional[set[str]] = None,
    extra_specs: tuple[ManualClaimSpec, ...] = (),
) -> list[ExtractionResult]:
    """Build manual Case Study 1 ExtractionResult objects from the local abstract cache."""
    specs = CURATED_CASE1_CLAIMS + extra_specs
    by_pmid: dict[str, list[tuple[int, ManualClaimSpec]]] = {}
    for ordinal, spec in enumerate(specs, start=1):
        if pmids is not None and spec.pmid not in pmids:
            continue
        by_pmid.setdefault(spec.pmid, []).append((ordinal, spec))

    results: list[ExtractionResult] = []
    for pmid in sorted(by_pmid):
        cached = cache.get(pmid)
        if cached is None:
            logger.warning("manual Case Study 1 PMID missing from cache: %s", pmid)
            continue
        _abstract, paper = cached
        claims = [_claim_from_spec(spec, paper, ordinal) for ordinal, spec in by_pmid[pmid]]
        results.append(ExtractionResult(paper=paper, claims=claims, raw_response="manual_case1_curated"))
    return results


def _existing_claim_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cid = json.loads(line).get("id")
            except Exception:
                continue
            if cid:
                ids.add(str(cid))
    return ids


def _append_manual_claims(jsonl_path: Path, results: list[ExtractionResult], label: str) -> int:
    existing = _existing_claim_ids(jsonl_path)
    written = 0
    with jsonl_path.open("a", encoding="utf-8") as f:
        for result in results:
            for claim in result.claims:
                if claim.id in existing:
                    continue
                record = claim.to_dict()
                record["disease"] = label
                record["year"] = result.paper.year or 0
                record["extraction_timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing.add(claim.id)
                written += 1
    return written


def _append_manual_metadata(papers_csv: Path, results: list[ExtractionResult], label: str, cache: AbstractCache) -> int:
    existing: set[tuple[str, str]] = set()
    if papers_csv.exists():
        with papers_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing.add((row.get("pmid") or "", row.get("disease") or ""))

    need_header = not papers_csv.exists() or papers_csv.stat().st_size == 0
    fieldnames = [
        "pmid", "doi", "title", "authors", "year", "journal", "disease",
        "abstract_length", "n_claims_extracted", "extraction_timestamp", "extraction_error",
    ]
    written = 0
    with papers_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        for result in results:
            key = (result.paper.pmid or "", label)
            if key in existing:
                continue
            cached = cache.get(result.paper.pmid or "")
            abstract_len = len(cached[0]) if cached else 0
            writer.writerow({
                "pmid": result.paper.pmid,
                "doi": result.paper.doi,
                "title": result.paper.title,
                "authors": result.paper.authors,
                "year": result.paper.year,
                "journal": result.paper.journal,
                "disease": label,
                "abstract_length": abstract_len,
                "n_claims_extracted": len(result.claims),
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_error": "",
            })
            existing.add(key)
            written += 1
    return written


def _filter_existing_claim_ids(
    results: list[ExtractionResult],
    existing_ids: set[str],
) -> list[ExtractionResult]:
    """Drop claims already recorded in extracted_claims.jsonl."""
    filtered: list[ExtractionResult] = []
    for result in results:
        claims = [claim for claim in result.claims if claim.id not in existing_ids]
        if claims:
            filtered.append(ExtractionResult(
                paper=result.paper,
                claims=claims,
                raw_response=result.raw_response,
                error=result.error,
            ))
    return filtered


def run_manual_case1_claim_ingestion(
    *,
    data_dir: Optional[Path] = None,
    pmids: Optional[list[str]] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
) -> dict:
    """Ingest hand-curated Case Study 1 claims into a Phase 2 run directory."""
    paths = _resolve_data_paths(data_dir)
    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    kg = load_graph(paths["graph"])

    pmid_filter = {str(p) for p in pmids} if pmids else None
    extra_path = paths["data_dir"] / "manual_case1_claims.jsonl"
    extra_specs = load_external_manual_claim_specs(extra_path)
    all_results = build_manual_case1_results(cache, pmid_filter, extra_specs)
    existing_ids = _existing_claim_ids(paths["claims"])
    results = _filter_existing_claim_ids(all_results, existing_ids)
    raw_claims = sum(len(r.claims) for r in results)

    before = kg.stats()
    ingest_summary = ingest_claims(
        kg,
        results,
        refine_vague_predicates=False,
        keep_noise=keep_noise,
        strict_phase1=strict_phase1,
    )
    after = kg.stats()
    save_graph(kg, paths["graph"])

    label = "manual_case1_curated"
    claims_written = _append_manual_claims(paths["claims"], results, label)
    metadata_written = _append_manual_metadata(paths["papers_csv"], results, label, cache)

    summary = {
        "label": label,
        "external_specs_path": str(extra_path),
        "external_manual_claim_specs": len(extra_specs),
        "papers_curated": len(all_results),
        "manual_claim_specs": sum(len(r.claims) for r in all_results),
        "new_manual_claim_specs": raw_claims,
        "claims_written": claims_written,
        "metadata_rows_written": metadata_written,
        "ingest": ingest_summary,
        "concepts_added": after["n_concepts"] - before["n_concepts"],
        "edges_added": after["n_edges"] - before["n_edges"],
        "graph_concepts": after["n_concepts"],
        "graph_edges": after["n_edges"],
    }
    logger.info("manual Case Study 1 curation summary: %s", summary)
    return summary


