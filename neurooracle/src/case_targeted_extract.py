"""Case-study targeted Phase 2 extraction.

This module complements the broad disease-year crawl and the KG-term chain
queries with hand-curated PubMed searches for a specific case-study need.
The first presets target Case Study 1 and Case Study 2:

    Case Study 1: IMAGING_MARKER -> TRANSDIAGNOSTIC_DISEASE_OR_OUTCOME

    GENE_TARGET -> IMAGING_MARKER -> OUTCOME

The search is intentionally task-focused, but extraction and ingestion reuse
the standard Phase 2 prompt, model cascade, cache, and claim guards.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Optional

from .abstract_cache import AbstractCache, default_cache_path
from .batch_extract import (
    DATA_DIR,
    _append_to_csv,
    _fetch_pubmed_details,
    _init_csv,
    _search_pubmed,
)
from .claim_extractor import ClaimExtractor
from .claim_ingestion import ingest_claims
from .storage import load_graph, save_graph

logger = logging.getLogger(__name__)


def _tiab(term: str) -> str:
    term = term.strip()
    if not term:
        raise ValueError("empty PubMed term")
    if any(ch in term for ch in (" ", "-", "/")) and "*" not in term:
        return f'"{term}"[Title/Abstract]'
    return f"{term}[Title/Abstract]"


def _or_group(terms: list[str]) -> str:
    return "(" + " OR ".join(_tiab(t) for t in terms) + ")"


def _year_clause(year_start: int, year_end: int) -> str:
    return f"{year_start}:{year_end}[pdat]"


def _human_neuro_clause() -> str:
    return (
        "(brain[Title/Abstract] OR neural[Title/Abstract] OR "
        "neuroimag*[Title/Abstract] OR neurolog*[Title/Abstract] OR "
        "psychiatr*[Title/Abstract] OR cognit*[Title/Abstract] OR "
        "cortex[Title/Abstract] OR cortical[Title/Abstract] OR "
        "\"central nervous system\"[Title/Abstract])"
    )


GENETIC_GENERAL = [
    "polygenic risk score",
    "polygenic score",
    "PRS",
    "GWAS",
    "genome-wide association",
    "genetic risk",
    "genotype",
    "genetic variant",
    "allele",
    "SNP",
    "gene expression",
    "transcriptomic",
    "pathway",
]

GENE_TARGETS = [
    "APOE",
    "GBA",
    "MAPT",
    "TREM2",
    "PSEN1",
    "APP",
    "BDNF",
    "COMT",
    "DRD2",
    "SLC6A4",
    "CACNA1C",
    "GRIN2A",
    "DISC1",
]

IMAGING_MARKERS = [
    "cortical thickness",
    "cortical surface area",
    "gray matter volume",
    "grey matter volume",
    "hippocampal volume",
    "entorhinal thickness",
    "brain volume",
    "atrophy",
    "white matter integrity",
    "fractional anisotropy",
    "mean diffusivity",
    "functional connectivity",
    "default mode network",
    "resting-state fMRI",
    "amyloid PET",
    "tau PET",
    "FDG PET",
    "FDG hypometabolism",
    "SUVR",
    "dopamine transporter",
    "DAT SPECT",
    "striatal binding",
    "neuroimaging",
    "MRI",
    "fMRI",
    "DTI",
    "PET",
]

OUTCOMES = [
    "cognitive decline",
    "cognition",
    "memory",
    "executive function",
    "conversion",
    "clinical progression",
    "dementia",
    "mild cognitive impairment",
    "MMSE",
    "ADAS-Cog",
    "CDR-SB",
    "UPDRS",
    "motor symptoms",
    "PANSS",
    "psychosis",
    "depression severity",
    "HAMD",
    "MADRS",
    "treatment response",
]

DISEASE_OUTCOME_ANCHORS = [
    "Alzheimer disease",
    "mild cognitive impairment",
    "Parkinson disease",
    "schizophrenia",
    "major depression",
    "bipolar disorder",
]

CASE1_DISORDERS = [
    "schizophrenia",
    "schizoaffective disorder",
    "psychosis",
    "bipolar disorder",
    "major depression",
    "major depressive disorder",
    "autism spectrum disorder",
    "autism",
    "ASD",
    "attention deficit hyperactivity disorder",
    "ADHD",
    "obsessive compulsive disorder",
    "OCD",
    "anxiety disorder",
    "generalized anxiety disorder",
    "posttraumatic stress disorder",
    "PTSD",
    "substance use disorder",
    "alcohol use disorder",
    "eating disorder",
    "anorexia nervosa",
]

CASE1_TRANSDIAGNOSTIC_TERMS = [
    "transdiagnostic",
    "cross-disorder",
    "cross diagnostic",
    "psychiatric disorders",
    "mental disorders",
    "shared neural",
    "shared brain",
    "common brain",
    "RDoC",
    "Research Domain Criteria",
    "p-factor",
    "general psychopathology",
    "HiTOP",
    "internalizing",
    "externalizing",
    "case-control",
    "meta-analysis",
    "ENIGMA",
    "UK Biobank",
    "Human Connectome Project",
    "ABCD",
    "Philadelphia Neurodevelopmental Cohort",
    "UCLA Consortium for Neuropsychiatric Phenomics",
    "Transdiagnostic Connectome Project",
]

CASE1_IMAGING_MARKERS = [
    "cortical thickness",
    "surface area",
    "gray matter volume",
    "grey matter volume",
    "subcortical volume",
    "hippocampal volume",
    "amygdala volume",
    "white matter integrity",
    "fractional anisotropy",
    "mean diffusivity",
    "functional connectivity",
    "resting-state fMRI",
    "default mode network",
    "salience network",
    "frontoparietal network",
    "structural covariance",
    "connectome",
    "graph theory",
    "ALFF",
    "fALFF",
    "ReHo",
    "regional homogeneity",
    "amplitude of low frequency fluctuation",
    "brain age",
    "normative modeling",
    "normative deviation",
    "MRI",
    "fMRI",
    "DTI",
    "neuroimaging",
]

CASE1_OUTCOMES = [
    "symptom severity",
    "cognitive performance",
    "executive function",
    "working memory",
    "negative symptoms",
    "positive symptoms",
    "depression severity",
    "anxiety symptoms",
    "social cognition",
    "functional impairment",
    "internalizing symptoms",
    "externalizing symptoms",
    "anhedonia",
    "amotivation",
    "emotion regulation",
    "neurocognitive performance",
    "diagnosis",
    "transdiagnostic dimension",
    "general psychopathology",
    "p-factor",
    "psychopathology",
    "PANSS",
    "HAMD",
    "MADRS",
    "PHQ-9",
]

CASE_TARGETED_PRESETS = {
    "case1_transdiagnostic",
    "case2_pathway_mediation",
    "case2_supplemental_classic",
}


def canonical_case_targeted_preset(preset: str) -> str:
    return preset


def build_case_targeted_queries(
    preset: str,
    *,
    year_start: int,
    year_end: int,
) -> list[str]:
    """Return PubMed queries for a case-targeted preset."""
    preset = canonical_case_targeted_preset(preset)
    if preset not in CASE_TARGETED_PRESETS:
        raise ValueError(
            "unknown preset: "
            f"{preset!r}; valid: {', '.join(sorted(CASE_TARGETED_PRESETS))}"
        )

    years = _year_clause(year_start, year_end)
    if preset == "case1_transdiagnostic":
        disorders = _or_group(CASE1_DISORDERS)
        transdx = _or_group(CASE1_TRANSDIAGNOSTIC_TERMS)
        imaging = _or_group(CASE1_IMAGING_MARKERS)
        outcomes = _or_group(CASE1_OUTCOMES)
        neuro = _human_neuro_clause()
        review = (
            "(review[Publication Type] OR systematic review[Title/Abstract] "
            "OR meta-analysis[Publication Type] OR consortium[Title/Abstract])"
        )
        return [
            f"{transdx} AND {imaging} AND {disorders} AND {neuro} AND {years}",
            f"{disorders} AND {imaging} AND {outcomes} AND {neuro} AND {years}",
            (
                f"{_or_group(['ENIGMA', 'large-scale', 'mega-analysis', 'meta-analysis'])} "
                f"AND {_or_group(['cortical thickness', 'surface area', 'subcortical volume', 'MRI'])} "
                f"AND {disorders} AND {years}"
            ),
            (
                f"{_or_group(['resting-state fMRI', 'functional connectivity', 'default mode network', 'salience network'])} "
                f"AND {disorders} AND {outcomes} AND {years}"
            ),
            (
                f"{_or_group(['white matter integrity', 'fractional anisotropy', 'mean diffusivity', 'DTI'])} "
                f"AND {disorders} AND {outcomes} AND {years}"
            ),
            (
                f"{_or_group(['brain age', 'brain-age', 'normative modeling', 'normative model'])} "
                f"AND {_or_group(['psychiatric disorders', 'mental disorders', 'schizophrenia', 'bipolar disorder', 'depression'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['transdiagnostic', 'cross-disorder', 'shared neural'])} "
                f"AND {_or_group(['cognitive performance', 'executive function', 'social cognition', 'symptom severity'])} "
                f"AND {imaging} AND {years}"
            ),
            f"{disorders} AND {imaging} AND {review} AND {years}",
            (
                f"{_or_group(['RDoC', 'Research Domain Criteria', 'p-factor', 'general psychopathology', 'HiTOP', 'internalizing', 'externalizing'])} "
                f"AND {imaging} AND {neuro} AND {years}"
            ),
            (
                f"{_or_group(['ABCD', 'Philadelphia Neurodevelopmental Cohort', 'PNC', 'UCLA Consortium for Neuropsychiatric Phenomics', 'Transdiagnostic Connectome Project', 'TCP'])} "
                f"AND {_or_group(['mental health', 'psychopathology', 'psychiatric symptoms', 'cognition', 'neurocognitive'])} "
                f"AND {_or_group(['MRI', 'fMRI', 'functional connectivity', 'cortical thickness', 'brain imaging'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['ALFF', 'fALFF', 'ReHo', 'regional homogeneity', 'graph theory', 'connectome', 'connectomics'])} "
                f"AND {disorders} AND {outcomes} AND {years}"
            ),
            (
                f"{_or_group(['autism spectrum disorder', 'ASD', 'ADHD', 'OCD', 'PTSD', 'eating disorder', 'substance use disorder'])} "
                f"AND {_or_group(['functional connectivity', 'resting-state fMRI', 'cortical thickness', 'DTI', 'white matter'])} "
                f"AND {_or_group(['symptom severity', 'social cognition', 'executive function', 'internalizing', 'externalizing', 'diagnosis'])} "
                f"AND {years}"
            ),
        ]

    if preset == "case2_supplemental_classic":
        review = "(review[Publication Type] OR systematic review[Title/Abstract] OR meta-analysis[Publication Type] OR overview[Title/Abstract])"
        return [
            (
                f"{_or_group(['polygenic risk score', 'polygenic score', 'PRS', 'GWAS'])} "
                f"AND {_or_group(['Alzheimer disease', 'cortical thickness', 'hippocampal volume', 'MRI'])} "
                f"AND {review} AND {years}"
            ),
            (
                f"{_or_group(['APOE', 'apolipoprotein E'])} "
                f"AND {_or_group(['amyloid PET', 'tau PET', 'hippocampal volume', 'cognitive decline'])} "
                f"AND {review} AND {years}"
            ),
            (
                f"{_or_group(['ADNI', 'Alzheimer Disease Neuroimaging Initiative'])} "
                f"AND {_or_group(['amyloid PET', 'tau PET', 'MRI', 'cognitive decline'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['imaging genetics', 'genomic imaging', 'DISC1', 'COMT'])} "
                f"AND {_or_group(['schizophrenia', 'bipolar disorder', 'cortical thickness', 'functional connectivity'])} "
                f"AND {review} AND {years}"
            ),
            (
                f"{_or_group(['UK Biobank', 'polygenic risk score', 'genetic correlation'])} "
                f"AND {_or_group(['brain structure', 'MRI', 'cognition', 'cortical thickness'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['brain age', 'brain-age', 'genetic risk', 'APOE'])} "
                f"AND {_or_group(['Alzheimer disease', 'Parkinson disease', 'cognition', 'cognitive decline'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['white matter hyperintensities', 'white matter integrity', 'polygenic risk'])} "
                f"AND {_or_group(['cognitive decline', 'Alzheimer disease', 'dementia'])} "
                f"AND {years}"
            ),
            (
                f"{_or_group(['transcriptomic', 'gene expression', 'molecular pathway'])} "
                f"AND {_or_group(['neuroimaging', 'MRI', 'PET', 'cognition', 'Alzheimer disease'])} "
                f"AND {review} AND {years}"
            ),
        ]

    genetic_general = _or_group(GENETIC_GENERAL)
    gene_targets = _or_group(GENE_TARGETS)
    genetic_any = "(" + " OR ".join([genetic_general, gene_targets]) + ")"
    imaging = _or_group(IMAGING_MARKERS)
    outcomes = _or_group(OUTCOMES)
    disease_outcomes = _or_group(DISEASE_OUTCOME_ANCHORS)
    neuro = _human_neuro_clause()

    queries = [
        f"{genetic_general} AND {imaging} AND {outcomes} AND {neuro} AND {years}",
        f"{gene_targets} AND {imaging} AND {outcomes} AND {neuro} AND {years}",
        f"{genetic_any} AND {imaging} AND {disease_outcomes} AND {neuro} AND {years}",
        (
            f"{_or_group(['polygenic risk score', 'polygenic score', 'PRS', 'GWAS'])} "
            f"AND {_or_group(['cortical thickness', 'hippocampal volume', 'gray matter volume', 'brain volume', 'MRI', 'DTI'])} "
            f"AND {outcomes} AND {years}"
        ),
        (
            f"{_or_group(['APOE', 'TREM2', 'PSEN1', 'APP'])} "
            f"AND {_or_group(['amyloid PET', 'tau PET', 'FDG PET', 'FDG hypometabolism', 'hippocampal volume', 'entorhinal thickness'])} "
            f"AND {_or_group(['cognitive decline', 'conversion', 'MMSE', 'ADAS-Cog', 'CDR-SB', 'dementia', 'mild cognitive impairment'])} "
            f"AND {years}"
        ),
        (
            f"{_or_group(['GBA', 'MAPT', 'SNCA', 'LRRK2'])} "
            f"AND {_or_group(['MRI', 'fMRI', 'DTI', 'dopamine transporter', 'DAT SPECT', 'striatal binding', 'cortical thickness'])} "
            f"AND {_or_group(['cognition', 'cognitive decline', 'UPDRS', 'motor symptoms', 'Parkinson disease'])} "
            f"AND {years}"
        ),
        (
            f"{_or_group(['BDNF', 'COMT', 'DRD2', 'SLC6A4', 'CACNA1C', 'DISC1'])} "
            f"AND {_or_group(['functional connectivity', 'resting-state fMRI', 'gray matter volume', 'cortical thickness', 'hippocampal volume'])} "
            f"AND {_or_group(['PANSS', 'psychosis', 'depression severity', 'HAMD', 'MADRS', 'treatment response', 'executive function'])} "
            f"AND {years}"
        ),
        (
            f"{_or_group(['gene expression', 'transcriptomic', 'pathway', 'molecular pathway', 'proteomic'])} "
            f"AND {imaging} AND {outcomes} AND {neuro} AND {years}"
        ),
    ]

    # Preserve order while deduplicating in case terms converge later.
    deduped: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def build_case_targeted_search_phrases(preset: str) -> list[str]:
    """Return plain-text search phrases for APIs without PubMed query syntax."""
    preset = canonical_case_targeted_preset(preset)
    if preset not in CASE_TARGETED_PRESETS:
        raise ValueError(
            "unknown preset: "
            f"{preset!r}; valid: {', '.join(sorted(CASE_TARGETED_PRESETS))}"
        )

    if preset == "case2_supplemental_classic":
        return [
            "polygenic risk score Alzheimer cortical thickness hippocampal volume MRI review",
            "APOE amyloid PET tau PET hippocampal volume cognitive decline review",
            "ADNI amyloid PET tau PET MRI cognitive decline Alzheimer review",
            "imaging genetics schizophrenia bipolar cortical thickness DISC1 COMT review",
            "UK Biobank polygenic risk score brain structure cognition MRI",
            "brain age genetic risk APOE Parkinson Alzheimer cognition review",
            "genetic predisposition white matter hyperintensities cognitive decline Alzheimer",
            "transcriptomic gene expression neuroimaging cognition Alzheimer review",
        ]

    if preset == "case1_transdiagnostic":
        return [
            "transdiagnostic neuroimaging psychiatric disorders cortical thickness ENIGMA",
            "cross-disorder MRI schizophrenia bipolar depression cortical thickness",
            "psychiatric disorders functional connectivity symptom severity fMRI",
            "resting-state fMRI transdiagnostic psychopathology cognitive performance",
            "white matter integrity DTI psychiatric disorders executive function",
            "brain age normative modeling psychiatric disorders depression schizophrenia",
            "ENIGMA schizophrenia bipolar major depression subcortical volume meta-analysis",
            "shared brain abnormalities ADHD OCD anxiety depression schizophrenia bipolar MRI",
            "cross diagnostic neuroimaging social cognition symptom dimensions",
            "UK Biobank mental health brain imaging transdiagnostic cognition",
            "RDoC p-factor general psychopathology neuroimaging psychiatric symptoms",
            "HiTOP internalizing externalizing brain imaging psychopathology",
            "ABCD brain imaging mental health cognition psychopathology",
            "Philadelphia Neurodevelopmental Cohort neuroimaging psychopathology cognition",
            "UCLA Consortium for Neuropsychiatric Phenomics fMRI schizophrenia bipolar ADHD",
            "Transdiagnostic Connectome Project functional connectivity psychopathology",
            "ALFF ReHo graph theory connectome psychiatric disorders symptom severity",
            "autism ADHD OCD PTSD resting-state fMRI symptom severity social cognition",
        ]

    return [
        "polygenic risk score cortical thickness cognitive decline Alzheimer MRI",
        "polygenic score brain volume dementia neuroimaging",
        "GWAS white matter hyperintensity cortical atrophy dementia",
        "APOE amyloid PET tau PET hippocampal volume cognitive decline",
        "TREM2 PSEN1 APP neuroimaging cognition Alzheimer disease",
        "GBA MAPT SNCA LRRK2 MRI cognition UPDRS Parkinson disease",
        "BDNF COMT DRD2 SLC6A4 functional connectivity schizophrenia depression",
        "gene expression transcriptomic pathway neuroimaging cognition",
        "genetic variant functional connectivity cognition psychiatric disorder",
        "molecular pathway PET MRI cognitive decline neurodegeneration",
    ]


def _resolve_data_paths(data_dir: Optional[Path]) -> dict[str, Path]:
    ddir = Path(data_dir) if data_dir is not None else DATA_DIR
    ddir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": ddir,
        "papers_csv": ddir / "papers_metadata.csv",
        "collection_csv": ddir / "collection_metadata.csv",
        "graph": ddir / "knowledge_graph.json",
        "claims": ddir / "extracted_claims.jsonl",
    }


def _load_seen_pmids(papers_csv: Path) -> set[str]:
    if not papers_csv.exists():
        return set()
    seen: set[str] = set()
    with open(papers_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmid = (row.get("pmid") or "").strip()
            if pmid:
                seen.add(pmid)
    return seen


def _normalise_doi(value: str) -> str:
    doi = (value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = doi.removeprefix("doi:")
    doi = doi.removesuffix(".pdf")
    return doi.rstrip("/")


def _extract_doi(text: str) -> str:
    """Extract a DOI-like token from search result text."""
    if not text:
        return ""
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    if not match:
        return ""
    doi = match.group(0)
    doi = doi.rstrip(".,;:)\"'<>]")
    return _normalise_doi(doi)


def _extract_pmid(text: str) -> str:
    """Extract a PubMed identifier from URL or result text."""
    if not text:
        return ""
    patterns = [
        r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,10})",
        r"\bPMID[:\s]+(\d{6,10})\b",
        r"\bPubMed[:\s]+(\d{6,10})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _clean_search_title(value: str) -> str:
    title = (value or "").strip()
    if "|" in title:
        title = title.split("|", 1)[0].strip()
    title = re.sub(r"\s+-\s+(PMC|PubMed|ScienceDirect|SpringerLink)\s*$", "",
                   title, flags=re.IGNORECASE)
    return title


def _load_seen_refs(papers_csv: Path) -> tuple[set[str], set[str]]:
    if not papers_csv.exists():
        return set(), set()
    ids: set[str] = set()
    dois: set[str] = set()
    with open(papers_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmid = (row.get("pmid") or "").strip()
            if pmid:
                ids.add(pmid)
            doi = _normalise_doi(row.get("doi") or "")
            if doi:
                dois.add(doi)
    return ids, dois


def _merge_seen_refs(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    dois: set[str] = set()
    for path in paths:
        path_ids, path_dois = _load_seen_refs(path)
        ids.update(path_ids)
        dois.update(path_dois)
    return ids, dois


def _init_collection_csv(csv_path: Path) -> None:
    """Initialize collect-only metadata CSV."""
    header = [
        "pmid", "doi", "title", "authors", "year", "journal",
        "source", "preset", "abstract_length", "collected_at",
    ]
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def _append_collection_metadata(
    csv_path: Path,
    papers: list[tuple[str, object]],
    *,
    source: str,
    preset: str,
) -> None:
    if not papers:
        return
    _init_collection_csv(csv_path)
    now = datetime.now().isoformat()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for abstract, ref in papers:
            writer.writerow([
                getattr(ref, "pmid", ""),
                getattr(ref, "doi", ""),
                getattr(ref, "title", ""),
                getattr(ref, "authors", ""),
                getattr(ref, "year", ""),
                getattr(ref, "journal", ""),
                source,
                preset,
                len(abstract or ""),
                now,
            ])


def _ensure_papers_cached(papers: list[tuple[str, object]], data_dir: Path) -> int:
    cache = AbstractCache(default_cache_path(data_dir))
    to_cache = []
    for abstract, ref in papers:
        cache_id = str(getattr(ref, "pmid", "") or "").strip()
        if not cache_id or cache.get(cache_id) is not None:
            continue
        to_cache.append((cache_id, abstract, ref))
    if not to_cache:
        return 0
    return cache.put_many(to_cache)


def _append_claims_with_paper_year(jsonl_path: Path, results: list, label: str) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for result in results:
            if result is None:
                continue
            paper = result.paper
            for claim in result.claims:
                record = claim.to_dict()
                record["disease"] = label
                record["year"] = paper.year or 0
                record["extraction_timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_paper_metadata(papers_csv: Path, papers: list, results: list, label: str) -> None:
    papers_meta = []
    for (abstract, ref), result in zip(papers, results):
        papers_meta.append({
            "pmid": ref.pmid,
            "doi": ref.doi,
            "title": ref.title,
            "authors": ref.authors,
            "year": ref.year,
            "journal": ref.journal,
            "disease": label,
            "abstract_length": len(abstract),
            "n_claims": len(result.claims) if result else 0,
            "timestamp": datetime.now().isoformat(),
            "extraction_error": result.error if result else "missing extraction result",
        })
    _append_to_csv(papers_csv, papers_meta)


def _abstract_from_openalex_index(index: dict | None) -> str:
    if not index:
        return ""
    by_pos: dict[int, str] = {}
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                by_pos[int(pos)] = token
            except Exception:
                continue
    if not by_pos:
        return ""
    return " ".join(by_pos[pos] for pos in sorted(by_pos))


def _openalex_id_suffix(value: str) -> str:
    value = (value or "").rstrip("/")
    return value.rsplit("/", 1)[-1] if value else ""


def _pmid_from_openalex_ids(ids: dict) -> str:
    raw = ids.get("pmid") or ""
    if not raw:
        return ""
    return _openalex_id_suffix(str(raw))


def _normalise_openalex_work(work: dict) -> tuple[str, str, object] | None:
    from .schema import PaperRef

    abstract = _abstract_from_openalex_index(work.get("abstract_inverted_index"))
    if not abstract.strip():
        return None

    ids = work.get("ids") or {}
    pmid = _pmid_from_openalex_ids(ids)
    openalex_id = _openalex_id_suffix(work.get("id") or ids.get("openalex") or "")
    cache_id = pmid or (f"OA:{openalex_id}" if openalex_id else "")
    if not cache_id:
        return None

    authors = []
    for authorship in (work.get("authorships") or [])[:5]:
        author = authorship.get("author") or {}
        name = author.get("display_name") or ""
        if name:
            authors.append(name)

    host = ((work.get("primary_location") or {}).get("source") or {})
    doi = _normalise_doi(ids.get("doi") or work.get("doi") or "")
    paper_ref = PaperRef(
        pmid=cache_id,
        doi=doi,
        title=work.get("display_name") or work.get("title") or "",
        authors=", ".join(authors),
        year=work.get("publication_year"),
        journal=host.get("display_name") or "",
    )
    return cache_id, abstract, paper_ref


def _first_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = unescape(str(value)).strip()
        if text:
            return re.sub(r"\s+", " ", text)
    return ""


def _parse_year(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _normalise_europepmc_result(result: dict) -> tuple[str, str, object] | None:
    from .schema import PaperRef

    abstract = _first_text(result.get("abstractText"))
    if not abstract:
        return None

    pmid = _first_text(result.get("pmid"))
    pmcid = _first_text(result.get("pmcid"))
    doi = _normalise_doi(_first_text(result.get("doi")))
    source = _first_text(result.get("source")).upper()
    europepmc_id = _first_text(result.get("id"))
    if pmid:
        cache_id = pmid
    elif pmcid:
        cache_id = f"PMCID:{pmcid}"
    elif doi:
        cache_id = f"EPMC:{doi}"
    elif europepmc_id:
        cache_id = f"EPMC:{source}:{europepmc_id}" if source else f"EPMC:{europepmc_id}"
    else:
        return None

    paper_ref = PaperRef(
        pmid=cache_id,
        doi=doi,
        title=_first_text(result.get("title")),
        authors=_first_text(result.get("authorString")),
        year=_parse_year(result.get("pubYear") or result.get("firstPublicationDate")),
        journal=_first_text(result.get("journalTitle")),
    )
    return cache_id, abstract, paper_ref


def _search_europepmc(
    query: str,
    *,
    year_start: int,
    year_end: int,
    max_results: int,
) -> list[tuple[str, str, object]]:
    import requests

    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": (
            f"({query}) AND HAS_ABSTRACT:Y AND "
            f"FIRST_PDATE:[{year_start}-01-01 TO {year_end}-12-31]"
        ),
        "resultType": "core",
        "format": "json",
        "pageSize": max(1, min(max_results, 1000)),
        "sort": "CITED desc",
    }
    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=45)
            if resp.status_code in (429, 500, 502, 503):
                logger.warning(
                    "Europe PMC search %s, backing off %.0fs (attempt %d)",
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 400:
                logger.warning("Europe PMC rejected search query: %r", query)
                return []
            resp.raise_for_status()
            results = (((resp.json() or {}).get("resultList") or {}).get("result") or [])
            records = []
            for item in results:
                if isinstance(item, dict):
                    rec = _normalise_europepmc_result(item)
                    if rec is not None:
                        records.append(rec)
            return records
        except Exception as exc:
            logger.warning("Europe PMC search failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff *= 2
    return []


def _arxiv_id_from_url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if not value:
        return ""
    arxiv_id = value.rsplit("/", 1)[-1]
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id


def _normalise_arxiv_entry(entry) -> tuple[str, str, object] | None:
    from .schema import PaperRef

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    id_el = entry.find("atom:id", ns)
    summary_el = entry.find("atom:summary", ns)
    abstract = _first_text(summary_el.text if summary_el is not None else "")
    arxiv_id = _arxiv_id_from_url(id_el.text if id_el is not None else "")
    if not arxiv_id or not abstract:
        return None

    title_el = entry.find("atom:title", ns)
    published_el = entry.find("atom:published", ns)
    doi_el = entry.find("arxiv:doi", ns)
    journal_el = entry.find("arxiv:journal_ref", ns)
    authors = []
    for author_el in entry.findall("atom:author", ns)[:5]:
        name_el = author_el.find("atom:name", ns)
        name = _first_text(name_el.text if name_el is not None else "")
        if name:
            authors.append(name)
    if len(entry.findall("atom:author", ns)) > 5:
        authors.append("et al.")

    cache_id = f"ARXIV:{arxiv_id}"
    paper_ref = PaperRef(
        pmid=cache_id,
        doi=_normalise_doi(_first_text(doi_el.text if doi_el is not None else "")),
        title=_first_text(title_el.text if title_el is not None else ""),
        authors=", ".join(authors),
        year=_parse_year(published_el.text if published_el is not None else ""),
        journal=_first_text(journal_el.text if journal_el is not None else "arXiv"),
    )
    return cache_id, abstract, paper_ref


def _arxiv_search_query(phrase: str) -> str:
    terms = [
        term for term in re.findall(r"[A-Za-z0-9]+", phrase)
        if len(term) > 2 and term.lower() not in _PREPRINT_STOPWORDS
    ]
    if not terms:
        return f'all:"{phrase}"'
    return " AND ".join(f"all:{term}" for term in terms[:7])


def _search_arxiv(
    query: str,
    *,
    year_start: int,
    year_end: int,
    max_results: int,
) -> list[tuple[str, str, object]]:
    import requests
    import xml.etree.ElementTree as ET

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": _arxiv_search_query(query),
        "start": 0,
        "max_results": max(1, min(max_results, 100)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=45)
            if resp.status_code in (429, 500, 502, 503):
                logger.warning(
                    "arXiv search %s, backing off %.0fs (attempt %d)",
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            records = []
            for entry in root.findall("atom:entry", ns):
                rec = _normalise_arxiv_entry(entry)
                if rec is None:
                    continue
                year = getattr(rec[2], "year", None)
                if year is not None and not (year_start <= int(year) <= year_end):
                    continue
                records.append(rec)
            return records
        except Exception as exc:
            logger.warning("arXiv search failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff *= 2
    return []


_PREPRINT_STOPWORDS = {
    "and", "or", "the", "of", "in", "to", "with", "for", "a", "an",
    "by", "on", "from", "study", "studies", "disease", "disorder",
    "disorders", "brain",
}


def _content_matches_any_phrase(text: str, phrases: list[str]) -> bool:
    content_terms = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not content_terms:
        return False
    for phrase in phrases:
        terms = [
            t for t in re.findall(r"[a-z0-9]+", phrase.lower())
            if len(t) > 2 and t not in _PREPRINT_STOPWORDS
        ]
        if not terms:
            continue
        overlap = sum(1 for term in terms if term in content_terms)
        if overlap >= max(2, min(4, int(len(terms) * 0.35))):
            return True
    return False


def _content_matches_case1_preprint(text: str) -> bool:
    """Stricter local filter for broad preprint date scans.

    bioRxiv/medRxiv do not expose a normal keyword-search endpoint here, so we
    scan date windows and filter locally. Case Study 1 terms contain broad
    words such as cognition, brain and health; require anchors from all three
    concept groups to avoid collecting unrelated biomedical preprints.
    """
    content = (text or "").lower()

    def has_any(patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, content) for pattern in patterns)

    disorder_or_dimension = has_any((
        r"\btransdiagnostic\b",
        r"\bcross[-\s]?disorder\b",
        r"\bcross[-\s]?diagnostic\b",
        r"\bpsychopatholog",
        r"\bpsychiatr",
        r"\bmental health\b",
        r"\bschizophren",
        r"\bpsychosis\b",
        r"\bpsychotic\b",
        r"\bbipolar\b",
        r"\bdepress",
        r"\badhd\b",
        r"\battention[-\s]?deficit",
        r"\bautis",
        r"\basd\b",
        r"\bobsessive[-\s]?compulsive\b",
        r"\bocd\b",
        r"\banxiety\b",
        r"\bptsd\b",
        r"\bpost[-\s]?traumatic stress\b",
        r"\bsubstance use\b",
        r"\balcohol use\b",
        r"\beating disorder\b",
        r"\banorexia\b",
        r"\binternalizing\b",
        r"\bexternalizing\b",
        r"\bp[-\s]?factor\b",
    ))
    imaging_or_network = has_any((
        r"\bneuroimag",
        r"\bmri\b",
        r"\bfmri\b",
        r"\bdti\b",
        r"\bdiffusion mri\b",
        r"\bcortical\b",
        r"\bcortex\b",
        r"\bthickness\b",
        r"\bsurface area\b",
        r"\bgray matter\b",
        r"\bgrey matter\b",
        r"\bsubcortical\b",
        r"\bhippocamp",
        r"\bamygdala\b",
        r"\bwhite matter\b",
        r"\bfractional anisotropy\b",
        r"\bmean diffusivity\b",
        r"\bfunctional connect",
        r"\bstructural connect",
        r"\bconnectome\b",
        r"\bdefault mode\b",
        r"\bsalience network\b",
        r"\bfrontoparietal\b",
        r"\bbrain[-\s]?age\b",
        r"\bnormative model",
    ))
    outcome_or_dimension = has_any((
        r"\bcognit",
        r"\bexecutive function\b",
        r"\bworking memory\b",
        r"\bsocial cognition\b",
        r"\bsymptom",
        r"\bseverity\b",
        r"\bdiagnos",
        r"\bdimension",
        r"\bclinical\b",
        r"\bfunctioning\b",
        r"\bimpairment\b",
        r"\bnegative symptoms\b",
        r"\bpositive symptoms\b",
        r"\bpanss\b",
        r"\bhamd\b",
        r"\bmadrs\b",
        r"\bphq[-\s]?9\b",
    ))
    return disorder_or_dimension and imaging_or_network and outcome_or_dimension


def _content_matches_preprint_preset(text: str, phrases: list[str], preset: str) -> bool:
    if canonical_case_targeted_preset(preset) == "case1_transdiagnostic":
        return _content_matches_case1_preprint(text)
    return _content_matches_any_phrase(text, phrases)


def _normalise_preprint_result(item: dict, server: str) -> tuple[str, str, object] | None:
    from .schema import PaperRef

    abstract = _first_text(item.get("abstract"))
    doi = _normalise_doi(_first_text(item.get("doi")))
    if not abstract or not doi:
        return None
    server_key = server.upper()
    cache_id = f"{server_key}:{doi}"
    published = _normalise_doi(_first_text(item.get("published")))
    journal = _first_text(item.get("server"), server)
    if published:
        journal = f"{journal}; published={published}"
    paper_ref = PaperRef(
        pmid=cache_id,
        doi=doi,
        title=_first_text(item.get("title")),
        authors=_first_text(item.get("authors")),
        year=_parse_year(item.get("date")),
        journal=journal,
    )
    return cache_id, abstract, paper_ref


def _search_preprint_server(
    server: str,
    *,
    preset: str,
    phrases: list[str],
    year_start: int,
    year_end: int,
    target_papers: int,
    max_scan: int,
) -> tuple[list[tuple[str, str, object]], int]:
    import requests

    selected: list[tuple[str, str, object]] = []
    seen_ids: set[str] = set()
    scanned = 0
    page_size = 30
    backoff = 2.0
    for year in range(year_end, year_start - 1, -1):
        if scanned >= max_scan or len(selected) >= target_papers:
            break
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        cursor = 0
        while scanned < max_scan and len(selected) < target_papers:
            url = f"https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json"
            try:
                resp = requests.get(url, timeout=45)
                if resp.status_code in (429, 500, 502, 503):
                    logger.warning(
                        "%s search %s, backing off %.0fs",
                        server,
                        resp.status_code,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                body = resp.json() or {}
            except Exception as exc:
                logger.warning(
                    "%s search failed for %d at cursor %d: %s",
                    server,
                    year,
                    cursor,
                    exc,
                )
                break

            records = [r for r in (body.get("collection") or []) if isinstance(r, dict)]
            if not records:
                break
            for item in records:
                scanned += 1
                rec = _normalise_preprint_result(item, server)
                if rec is None:
                    continue
                cache_id, abstract, paper = rec
                text = " ".join([getattr(paper, "title", ""), abstract])
                if cache_id in seen_ids or not _content_matches_preprint_preset(text, phrases, preset):
                    continue
                seen_ids.add(cache_id)
                selected.append(rec)
                if len(selected) >= target_papers or scanned >= max_scan:
                    break
            if len(records) < page_size:
                break
            cursor += len(records)
            time.sleep(0.4)
    return selected, scanned


def _search_openalex(
    query: str,
    *,
    year_start: int,
    year_end: int,
    max_results: int,
    sort: str = "",
) -> list[tuple[str, str, object]]:
    import requests

    url = "https://api.openalex.org/works"
    per_page = min(max_results, 200)
    params = {
        "search": query,
        "filter": (
            f"from_publication_date:{year_start}-01-01,"
            f"to_publication_date:{year_end}-12-31"
        ),
        "per-page": per_page,
        "select": (
            "id,doi,display_name,title,publication_year,publication_date,ids,"
            "primary_location,authorships,abstract_inverted_index"
        ),
    }
    if sort:
        params["sort"] = sort
    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 400:
                logger.warning("OpenAlex rejected search query: %r", query)
                return []
            if resp.status_code in (429, 500, 502, 503):
                logger.warning(
                    "OpenAlex search %s, backing off %.0fs (attempt %d)",
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            records = []
            for work in resp.json().get("results") or []:
                rec = _normalise_openalex_work(work)
                if rec is not None:
                    records.append(rec)
            return records
        except Exception as exc:
            logger.warning("OpenAlex search failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff *= 2
    return []


def _search_openalex_by_doi(doi: str) -> tuple[str, str, object] | None:
    import requests

    doi = _normalise_doi(doi)
    if not doi:
        return None
    params = {
        "filter": f"doi:{doi}",
        "per-page": 1,
        "select": (
            "id,doi,display_name,title,publication_year,publication_date,ids,"
            "primary_location,authorships,abstract_inverted_index"
        ),
    }
    try:
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        resp.raise_for_status()
        for work in resp.json().get("results") or []:
            rec = _normalise_openalex_work(work)
            if rec is not None:
                return rec
    except Exception as exc:
        logger.warning("OpenAlex DOI lookup failed for %s: %s", doi, exc)
    return None


def _resolve_openalex_by_title(
    title: str,
    *,
    year_start: int,
    year_end: int,
) -> tuple[str, str, object] | None:
    title = _clean_search_title(title)
    title_norm = _normalise_title(title)
    if len(title_norm) < 20:
        return None
    records = _search_openalex(
        title,
        year_start=year_start,
        year_end=year_end,
        max_results=3,
    )
    for rec in records:
        paper_title = _normalise_title(getattr(rec[2], "title", ""))
        if not paper_title:
            continue
        if title_norm in paper_title or paper_title in title_norm:
            return rec
        title_terms = set(title_norm.split())
        paper_terms = set(paper_title.split())
        if title_terms and len(title_terms & paper_terms) / len(title_terms) >= 0.72:
            return rec
    return None


def _search_anysearch(query: str, *, max_results: int) -> list[dict]:
    import requests

    payload = {
        "query": query,
        "max_results": max(1, min(max_results, 50)),
        "domain": "academic",
        "language": "en",
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("ANYSEARCH_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://api.anysearch.com/v1/search",
                json=payload,
                headers=headers,
                timeout=45,
            )
            if resp.status_code in (429, 500, 502, 503):
                logger.warning(
                    "AnySearch %s, backing off %.0fs (attempt %d)",
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") not in (0, None):
                logger.warning("AnySearch returned code=%s message=%s",
                               body.get("code"), body.get("message"))
                return []
            data = body.get("data") or {}
            return [r for r in (data.get("results") or []) if isinstance(r, dict)]
        except Exception as exc:
            logger.warning("AnySearch failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff *= 2
    return []


def _resolve_anysearch_result(
    result: dict,
    *,
    year_start: int,
    year_end: int,
    cache: AbstractCache,
) -> tuple[str, str, object] | None:
    url = result.get("url") or ""
    title = result.get("title") or ""
    snippet = result.get("snippet") or ""
    content = result.get("content") or ""
    haystack = "\n".join([url, title, snippet, content])

    pmid = _extract_pmid(haystack)
    if pmid:
        cached = cache.get(pmid)
        if cached is not None:
            abstract, paper = cached
            return pmid, abstract, paper
        papers = _fetch_pubmed_details([pmid], cache=cache)
        if papers:
            abstract, paper = papers[0]
            return pmid, abstract, paper

    doi = _extract_doi(haystack)
    if doi:
        rec = _search_openalex_by_doi(doi)
        if rec is not None:
            return rec

    return _resolve_openalex_by_title(title, year_start=year_start, year_end=year_end)


def _select_pubmed_papers(
    *,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_pmids: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    queries = build_case_targeted_queries(
        preset,
        year_start=year_start,
        year_end=year_end,
    )
    logger.info(
        "case-targeted source=pubmed preset=%s, queries=%d, target_papers=%d",
        preset,
        len(queries),
        target_papers,
    )

    selected_pmids: list[str] = []
    selected_seen: set[str] = set()
    query_hits: list[dict] = []
    skipped_seen = 0
    for idx, query in enumerate(queries, 1):
        pmids = _search_pubmed(query, max_results_per_query)
        added = 0
        for pmid in pmids:
            if pmid in selected_seen:
                continue
            selected_seen.add(pmid)
            if pmid in seen_pmids:
                skipped_seen += 1
                continue
            selected_pmids.append(pmid)
            added += 1
            if len(selected_pmids) >= target_papers:
                break
        query_hits.append({
            "query_index": idx,
            "hits": len(pmids),
            "new_added": added,
        })
        logger.info(
            "  query %d/%d: hits=%d, added=%d, selected=%d/%d",
            idx,
            len(queries),
            len(pmids),
            added,
            len(selected_pmids),
            target_papers,
        )
        if len(selected_pmids) >= target_papers:
            break
        time.sleep(0.4)

    if not selected_pmids:
        logger.info("case-targeted PubMed search found no new PMIDs")
        return [], skipped_seen, query_hits

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    papers = _fetch_pubmed_details(selected_pmids, cache=cache)
    logger.info(
        "fetched %d papers with abstracts from %d selected PMIDs",
        len(papers),
        len(selected_pmids),
    )
    return papers, skipped_seen, query_hits


def _select_openalex_papers(
    *,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    phrases = build_case_targeted_search_phrases(preset)
    logger.info(
        "case-targeted source=openalex preset=%s, queries=%d, target_papers=%d",
        preset,
        len(phrases),
        target_papers,
    )

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    selected: list[tuple[str, str, object]] = []
    selected_ids: set[str] = set()
    selected_dois: set[str] = set()
    skipped_seen = 0
    query_hits: list[dict] = []
    for idx, phrase in enumerate(phrases, 1):
        records = _search_openalex(
            phrase,
            year_start=year_start,
            year_end=year_end,
            max_results=max_results_per_query,
            sort="cited_by_count:desc" if canonical_case_targeted_preset(preset) == "case2_supplemental_classic" else "",
        )
        added = 0
        for cache_id, abstract, paper in records:
            doi = _normalise_doi(getattr(paper, "doi", ""))
            if cache_id in selected_ids or (doi and doi in selected_dois):
                continue
            selected_ids.add(cache_id)
            if doi:
                selected_dois.add(doi)
            if cache_id in seen_ids or (doi and doi in seen_dois):
                skipped_seen += 1
                continue

            rec = cache.get(cache_id)
            if rec is not None:
                cached_abstract, cached_paper = rec
                selected.append((cache_id, cached_abstract, cached_paper))
            else:
                selected.append((cache_id, abstract, paper))
            added += 1
            if len(selected) >= target_papers:
                break
        query_hits.append({
            "query_index": idx,
            "hits": len(records),
            "new_added": added,
        })
        logger.info(
            "  openalex query %d/%d: hits_with_abstract=%d, added=%d, selected=%d/%d",
            idx,
            len(phrases),
            len(records),
            added,
            len(selected),
            target_papers,
        )
        if len(selected) >= target_papers:
            break
        time.sleep(0.8)

    if not selected:
        logger.info("case-targeted OpenAlex search found no new abstracts")
        return [], skipped_seen, query_hits

    to_cache = [(cache_id, abstract, paper) for cache_id, abstract, paper in selected
                if cache.get(cache_id) is None]
    if to_cache:
        n_cached = cache.put_many(to_cache)
        logger.info("  cache: wrote %d OpenAlex abstracts (%s total)",
                    n_cached, f"{len(cache):,}")
    papers = [(abstract, paper) for _cache_id, abstract, paper in selected]
    logger.info(
        "selected %d OpenAlex papers with abstracts (skipped seen=%d)",
        len(papers),
        skipped_seen,
    )
    return papers, skipped_seen, query_hits


def _select_search_adapter_papers(
    *,
    source: str,
    search_fn,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    phrases = build_case_targeted_search_phrases(preset)
    logger.info(
        "case-targeted source=%s preset=%s, queries=%d, target_papers=%d",
        source,
        preset,
        len(phrases),
        target_papers,
    )

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    selected: list[tuple[str, str, object]] = []
    selected_ids: set[str] = set()
    selected_dois: set[str] = set()
    skipped_seen = 0
    query_hits: list[dict] = []
    for idx, phrase in enumerate(phrases, 1):
        records = search_fn(
            phrase,
            year_start=year_start,
            year_end=year_end,
            max_results=max_results_per_query,
        )
        added = 0
        for cache_id, abstract, paper in records:
            doi = _normalise_doi(getattr(paper, "doi", ""))
            if cache_id in selected_ids or (doi and doi in selected_dois):
                continue
            selected_ids.add(cache_id)
            if doi:
                selected_dois.add(doi)
            if cache_id in seen_ids or (doi and doi in seen_dois):
                skipped_seen += 1
                continue

            rec = cache.get(cache_id)
            if rec is not None:
                cached_abstract, cached_paper = rec
                selected.append((cache_id, cached_abstract, cached_paper))
            else:
                selected.append((cache_id, abstract, paper))
            added += 1
            if len(selected) >= target_papers:
                break
        query_hits.append({
            "query_index": idx,
            "hits": len(records),
            "new_added": added,
        })
        logger.info(
            "  %s query %d/%d: hits_with_abstract=%d, added=%d, selected=%d/%d",
            source,
            idx,
            len(phrases),
            len(records),
            added,
            len(selected),
            target_papers,
        )
        if len(selected) >= target_papers:
            break
        time.sleep(3.1 if source == "arxiv" else 0.8)

    if not selected:
        logger.info("case-targeted %s search found no new abstracts", source)
        return [], skipped_seen, query_hits

    to_cache = [(cache_id, abstract, paper) for cache_id, abstract, paper in selected
                if cache.get(cache_id) is None]
    if to_cache:
        n_cached = cache.put_many(to_cache)
        logger.info("  cache: wrote %d %s abstracts (%s total)",
                    n_cached, source, f"{len(cache):,}")
    papers = [(abstract, paper) for _cache_id, abstract, paper in selected]
    logger.info(
        "selected %d %s papers with abstracts (skipped seen=%d)",
        len(papers),
        source,
        skipped_seen,
    )
    return papers, skipped_seen, query_hits


def _select_europepmc_papers(
    *,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    return _select_search_adapter_papers(
        source="europepmc",
        search_fn=_search_europepmc,
        preset=preset,
        year_start=year_start,
        year_end=year_end,
        target_papers=target_papers,
        max_results_per_query=max_results_per_query,
        seen_ids=seen_ids,
        seen_dois=seen_dois,
        paths=paths,
    )


def _select_arxiv_papers(
    *,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    return _select_search_adapter_papers(
        source="arxiv",
        search_fn=_search_arxiv,
        preset=preset,
        year_start=year_start,
        year_end=year_end,
        target_papers=target_papers,
        max_results_per_query=max_results_per_query,
        seen_ids=seen_ids,
        seen_dois=seen_dois,
        paths=paths,
    )


def _select_preprint_papers(
    *,
    source: str,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    phrases = build_case_targeted_search_phrases(preset)
    logger.info(
        "case-targeted source=%s preset=%s, local-filter phrases=%d, target_papers=%d",
        source,
        preset,
        len(phrases),
        target_papers,
    )

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    max_scan = max(max_results_per_query, target_papers * 20)
    records, scanned = _search_preprint_server(
        source,
        preset=preset,
        phrases=phrases,
        year_start=year_start,
        year_end=year_end,
        target_papers=target_papers,
        max_scan=max_scan,
    )
    selected: list[tuple[str, str, object]] = []
    selected_ids: set[str] = set()
    selected_dois: set[str] = set()
    skipped_seen = 0
    for cache_id, abstract, paper in records:
        doi = _normalise_doi(getattr(paper, "doi", ""))
        if cache_id in selected_ids or (doi and doi in selected_dois):
            continue
        selected_ids.add(cache_id)
        if doi:
            selected_dois.add(doi)
        if cache_id in seen_ids or (doi and doi in seen_dois):
            skipped_seen += 1
            continue
        rec = cache.get(cache_id)
        if rec is not None:
            cached_abstract, cached_paper = rec
            selected.append((cache_id, cached_abstract, cached_paper))
        else:
            selected.append((cache_id, abstract, paper))

    query_hits = [{
        "query_index": 1,
        "scanned": scanned,
        "hits": len(records),
        "new_added": len(selected),
    }]
    if not selected:
        logger.info(
            "case-targeted %s scan found no new abstracts (scanned=%d)",
            source,
            scanned,
        )
        return [], skipped_seen, query_hits

    to_cache = [(cache_id, abstract, paper) for cache_id, abstract, paper in selected
                if cache.get(cache_id) is None]
    if to_cache:
        n_cached = cache.put_many(to_cache)
        logger.info("  cache: wrote %d %s abstracts (%s total)",
                    n_cached, source, f"{len(cache):,}")
    papers = [(abstract, paper) for _cache_id, abstract, paper in selected]
    logger.info(
        "selected %d %s preprints with abstracts (skipped seen=%d, scanned=%d)",
        len(papers),
        source,
        skipped_seen,
        scanned,
    )
    return papers, skipped_seen, query_hits


def _select_anysearch_papers(
    *,
    preset: str,
    year_start: int,
    year_end: int,
    target_papers: int,
    max_results_per_query: int,
    seen_ids: set[str],
    seen_dois: set[str],
    paths: dict[str, Path],
) -> tuple[list, int, list[dict]]:
    phrases = build_case_targeted_search_phrases(preset)
    logger.info(
        "case-targeted source=anysearch preset=%s, queries=%d, target_papers=%d",
        preset,
        len(phrases),
        target_papers,
    )

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    selected: list[tuple[str, str, object]] = []
    selected_ids: set[str] = set()
    selected_dois: set[str] = set()
    selected_urls: set[str] = set()
    skipped_seen = 0
    query_hits: list[dict] = []
    for idx, phrase in enumerate(phrases, 1):
        results = _search_anysearch(phrase, max_results=max_results_per_query)
        added = 0
        resolved = 0
        for result in results:
            url = (result.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            if url:
                selected_urls.add(url)

            rec = _resolve_anysearch_result(
                result,
                year_start=year_start,
                year_end=year_end,
                cache=cache,
            )
            if rec is None:
                continue
            resolved += 1
            cache_id, abstract, paper = rec
            doi = _normalise_doi(getattr(paper, "doi", ""))
            if cache_id in selected_ids or (doi and doi in selected_dois):
                continue
            selected_ids.add(cache_id)
            if doi:
                selected_dois.add(doi)
            if cache_id in seen_ids or (doi and doi in seen_dois):
                skipped_seen += 1
                continue

            if cache.get(cache_id) is None:
                cache.put(cache_id, abstract, paper)
            selected.append((cache_id, abstract, paper))
            added += 1
            if len(selected) >= target_papers:
                break
        query_hits.append({
            "query_index": idx,
            "hits": len(results),
            "resolved": resolved,
            "new_added": added,
        })
        logger.info(
            "  anysearch query %d/%d: hits=%d, resolved=%d, added=%d, selected=%d/%d",
            idx,
            len(phrases),
            len(results),
            resolved,
            added,
            len(selected),
            target_papers,
        )
        if len(selected) >= target_papers:
            break
        time.sleep(1.0)

    if not selected:
        logger.info("case-targeted AnySearch discovery found no new abstracts")
        return [], skipped_seen, query_hits

    papers = [(abstract, paper) for _cache_id, abstract, paper in selected]
    logger.info(
        "selected %d AnySearch-discovered papers with primary-source abstracts "
        "(skipped seen=%d)",
        len(papers),
        skipped_seen,
    )
    return papers, skipped_seen, query_hits


def run_case_targeted_extraction(
    *,
    preset: str = "case2_pathway_mediation",
    year_start: int = 2010,
    year_end: int = 2026,
    target_papers: int = 200,
    max_results_per_query: int = 100,
    source: str = "pubmed",
    max_workers: int = 2,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    include_seen: bool = False,
    lock_model: bool = False,
    collect_only: bool = False,
) -> dict:
    """Run a hand-curated case-study literature search.

    With ``collect_only=True``, only collect/cache abstracts and write
    collection metadata; do not extract claims or write the graph.
    """
    paths = _resolve_data_paths(data_dir)
    if collect_only:
        _init_collection_csv(paths["collection_csv"])
        seen_paths = [paths["collection_csv"], paths["papers_csv"]]
        seen_ids, seen_dois = (set(), set()) if include_seen else _merge_seen_refs(seen_paths)
    else:
        kg = load_graph(paths["graph"])
        logger.info(
            "loaded graph: %s concepts, %s edges",
            kg.stats()["n_concepts"],
            kg.stats()["n_edges"],
        )
        _init_csv(paths["papers_csv"])
        seen_ids, seen_dois = (set(), set()) if include_seen else _load_seen_refs(paths["papers_csv"])
    logger.info("seen refs index: ids=%s dois=%s",
                f"{len(seen_ids):,}", f"{len(seen_dois):,}")

    if source == "pubmed":
        papers, skipped_seen, query_hits = _select_pubmed_papers(
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_pmids=seen_ids,
            paths=paths,
        )
    elif source == "openalex":
        papers, skipped_seen, query_hits = _select_openalex_papers(
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_ids=seen_ids,
            seen_dois=seen_dois,
            paths=paths,
        )
    elif source == "europepmc":
        papers, skipped_seen, query_hits = _select_europepmc_papers(
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_ids=seen_ids,
            seen_dois=seen_dois,
            paths=paths,
        )
    elif source == "arxiv":
        papers, skipped_seen, query_hits = _select_arxiv_papers(
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_ids=seen_ids,
            seen_dois=seen_dois,
            paths=paths,
        )
    elif source in ("biorxiv", "medrxiv"):
        papers, skipped_seen, query_hits = _select_preprint_papers(
            source=source,
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_ids=seen_ids,
            seen_dois=seen_dois,
            paths=paths,
        )
    elif source == "anysearch":
        papers, skipped_seen, query_hits = _select_anysearch_papers(
            preset=preset,
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_results_per_query=max_results_per_query,
            seen_ids=seen_ids,
            seen_dois=seen_dois,
            paths=paths,
        )
    else:
        raise ValueError(
            f"unknown source: {source!r}; valid: pubmed, openalex, europepmc, "
            "arxiv, biorxiv, medrxiv, anysearch"
        )

    if not papers:
        return {
            "preset": preset,
            "source": source,
            "mode": "collect-only" if collect_only else "extract",
            "total_papers": 0,
            "total_claims": 0,
            "skipped_seen": skipped_seen,
            "query_hits": query_hits,
        }

    if collect_only:
        newly_cached = _ensure_papers_cached(papers, paths["data_dir"])
        _append_collection_metadata(
            paths["collection_csv"],
            papers,
            source=source,
            preset=preset,
        )
        summary = {
            "preset": preset,
            "source": source,
            "mode": "collect-only",
            "total_papers": len(papers),
            "total_claims": 0,
            "skipped_seen": skipped_seen,
            "query_hits": query_hits,
            "abstract_cache": str(default_cache_path(paths["data_dir"])),
            "collection_metadata": str(paths["collection_csv"]),
            "newly_cached": newly_cached,
        }
        logger.info("")
        logger.info("  CASE-TARGETED COLLECT-ONLY SUMMARY for %s source=%s", preset, source)
        logger.info("    papers collected:  %d", summary["total_papers"])
        logger.info("    skipped seen refs: %d", summary["skipped_seen"])
        logger.info("    newly cached:      %d", summary["newly_cached"])
        logger.info("    abstract cache:    %s", summary["abstract_cache"])
        logger.info("    metadata CSV:      %s", summary["collection_metadata"])
        return summary

    extractor = ClaimExtractor(lock_model=lock_model)
    label = f"case_targeted:{preset}" if source == "pubmed" else f"case_targeted:{source}:{preset}"
    results = extractor.extract_batch(papers, max_workers=max_workers)
    raw_claims = sum(len(r.claims) for r in results if r and r.claims)

    before = kg.stats()
    ingest_claims(kg, results, keep_noise=keep_noise, strict_phase1=strict_phase1)
    after = kg.stats()

    _append_paper_metadata(paths["papers_csv"], papers, results, label)
    _append_claims_with_paper_year(paths["claims"], results, label)
    save_graph(kg, paths["graph"])

    errors = sum(1 for r in results if r and r.error)
    zero = sum(1 for r in results if r and not r.error and len(r.claims) == 0)
    summary = {
        "preset": preset,
        "source": source,
        "total_papers": len(papers),
        "total_claims": raw_claims,
        "extraction_errors": errors,
        "zero_claim_papers": zero,
        "skipped_seen": skipped_seen,
        "query_hits": query_hits,
        "concepts_added": after["n_concepts"] - before["n_concepts"],
        "edges_added": after["n_edges"] - before["n_edges"],
        "graph_concepts": after["n_concepts"],
        "graph_edges": after["n_edges"],
    }

    logger.info("")
    logger.info("  CASE-TARGETED SUMMARY for %s source=%s", preset, source)
    logger.info("    papers extracted:   %d", summary["total_papers"])
    logger.info("    raw claims:         %d", summary["total_claims"])
    logger.info("    extraction errors:  %d", summary["extraction_errors"])
    logger.info("    zero-claim papers:  %d", summary["zero_claim_papers"])
    logger.info("    skipped seen pmids: %d", summary["skipped_seen"])
    logger.info("    graph delta:        %+d concepts, %+d edges",
                summary["concepts_added"], summary["edges_added"])
    return summary

