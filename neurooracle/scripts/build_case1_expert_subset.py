"""Build a blinded, pairwise-ready Case Study 1 expert-study subset.

The output keeps outcome labels at the top level.  ``UserStudyService`` loads
only the ``hypotheses`` list, so experts never receive those labels in either
the manual or assisted condition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
_SERVICE_PATH = ROOT / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurodiscovery_user_study_subset", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_SERVICE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SERVICE_MODULE)
build_progressive_pair_schedule = _SERVICE_MODULE.build_progressive_pair_schedule
candidate_comparison_task = _SERVICE_MODULE.candidate_comparison_task
candidate_semantic_key = _SERVICE_MODULE.candidate_semantic_key

_DIRECTION_PATH = ROOT / "neurooracle" / "src" / "case1_hypothesis.py"
_DIRECTION_SPEC = importlib.util.spec_from_file_location(
    "neurodiscovery_case1_hypothesis", _DIRECTION_PATH
)
assert _DIRECTION_SPEC and _DIRECTION_SPEC.loader
_DIRECTION_MODULE = importlib.util.module_from_spec(_DIRECTION_SPEC)
_DIRECTION_SPEC.loader.exec_module(_DIRECTION_MODULE)
case1_directional_statement = _DIRECTION_MODULE.case1_directional_statement
case1_directional_title = _DIRECTION_MODULE.case1_directional_title
propose_case1_direction = _DIRECTION_MODULE.propose_case1_direction


CASE_ROOT = ROOT / "neurooracle" / "data" / "cs_runs" / "case1_transdiagnostic"
DEFAULT_CURRENT = CASE_ROOT / "atlas_specific_smoke_20260615" / "hypotheses_raw.json"
DEFAULT_VALIDATED = CASE_ROOT / "single_disease_smoke_20260615" / "hypotheses_raw.json"
DEFAULT_VALIDATION = (
    CASE_ROOT / "atlas_all_20260610_0418" / "pilot_validation" / "pilot_validation_results.json"
)
DEFAULT_CLAIMS = ROOT / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
DEFAULT_OUTPUT = ROOT / "neurooracle" / "data" / "user_study" / "case1_expert_subset_v1.json"
DEFAULT_STUDY_ABSTRACT_CACHE = (
    ROOT / "neurooracle" / "data" / "user_study" / "case1_reference_abstract_cache.jsonl"
)
DEFAULT_STUDY_LINK_CACHE = (
    ROOT / "neurooracle" / "data" / "user_study" / "case1_reference_link_cache.jsonl"
)
DEFAULT_ABSTRACT_SOURCES = (
    ROOT / "neurooracle" / "data" / "full_v2" / "abstract_cache.jsonl",
    ROOT
    / "neurooracle"
    / "data"
    / "full_v2"
    / "scope_reaudit_all_20260720"
    / "abstract_backfill_20260720"
    / "abstract_backfill.jsonl",
    DEFAULT_STUDY_ABSTRACT_CACHE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidates_from(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload if isinstance(payload, list) else payload.get("hypotheses", [])
    return [dict(item) for item in rows if isinstance(item, dict) and item.get("id")]


def _case1_candidate_components(candidate: dict[str, Any]) -> tuple[list[str], str, str]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    candidate_tuple = (
        metadata.get("candidate_tuple")
        if isinstance(metadata.get("candidate_tuple"), dict)
        else {}
    )
    diseases = candidate_tuple.get("disease_ids") or candidate_tuple.get("diseases") or []
    if isinstance(diseases, str):
        diseases = [diseases]
    disease_values = [str(item).strip() for item in diseases if str(item).strip()]
    region = str(candidate_tuple.get("region_id") or candidate_tuple.get("region_name") or "").strip()
    feature = str(candidate_tuple.get("feature_id") or candidate_tuple.get("feature_name") or "").strip()
    return disease_values, region, feature


def single_disease_case1_candidates_from(path: Path) -> list[dict[str, Any]]:
    """Load only the native Case 1 unit: one disease x one ROI x one feature.

    Cross-disease clusters are downstream summaries over validated candidates;
    they are not hypotheses emitted by the current Case 1 generator.
    """
    rows = candidates_from(path)
    invalid: list[tuple[str, int, str, str]] = []
    for candidate in rows:
        diseases, region, feature = _case1_candidate_components(candidate)
        if (
            candidate.get("hypothesis_type") != "case1_candidate"
            or len(diseases) != 1
            or not region
            or not feature
        ):
            invalid.append((str(candidate.get("id") or ""), len(diseases), region, feature))
    if invalid:
        preview = ", ".join(f"{item[0]} ({item[1]} diseases)" for item in invalid[:5])
        raise ValueError(
            f"{path} contains {len(invalid)} non-native Case 1 candidates; expected exactly "
            f"one disease x one ROI x one feature. Examples: {preview}"
        )
    return rows


def collapse_semantic_duplicates(
    candidates: list[dict[str, Any]],
    *,
    rank_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge generator records that encode the same scientific proposition."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = candidate_semantic_key(candidate)
        # Native Case 1 candidates always have a key; retain an ID fallback so
        # malformed records are never silently merged together.
        if not key:
            key = ("candidate-id", str(candidate.get("id") or ""))
        groups.setdefault(key, []).append(candidate)

    collapsed: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_records = 0
    for rows in groups.values():
        rows.sort(key=rank_key)
        representative = dict(rows[0])
        metadata = dict(representative.get("metadata") or {})
        aliases = sorted({str(item.get("id") or "") for item in rows if item.get("id")})
        methods = sorted(
            {
                str((item.get("metadata") or {}).get("generation_method") or "").strip()
                for item in rows
                if str((item.get("metadata") or {}).get("generation_method") or "").strip()
            }
        )
        metadata["semantic_candidate_ids"] = aliases
        metadata["semantic_duplicate_count"] = len(rows)
        if methods:
            metadata["generation_methods"] = methods
        representative["metadata"] = metadata
        collapsed.append(representative)
        if len(rows) > 1:
            duplicate_groups += 1
            duplicate_records += len(rows) - 1

    return collapsed, {
        "input_records": len(candidates),
        "unique_hypotheses": len(collapsed),
        "duplicate_groups": duplicate_groups,
        "duplicate_records_removed": duplicate_records,
    }


def paper_count(candidate: dict[str, Any]) -> int:
    seen: set[str] = set()
    for edge in candidate.get("path") or []:
        if not isinstance(edge, dict):
            continue
        paper = edge.get("source_paper") if isinstance(edge.get("source_paper"), dict) else {}
        key = str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "").strip()
        if key:
            seen.add(key)
    return len(seen)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _disease_aliases(name: str) -> set[str]:
    normalized = _norm(name)
    aliases = {normalized} if normalized else set()
    groups = {
        "anorexia": {"anorexia nervosa", "anorexia"},
        "attention deficit": {"attention deficit hyperactivity disorder", "attention deficit disorder", "adhd"},
        "bipolar": {"bipolar disorder", "bipolar"},
        "major depressive": {"major depressive disorder", "depressive disorder", "depression", "mdd"},
        "obsessive compulsive": {"obsessive compulsive disorder", "obsessive compulsive", "ocd"},
        "psychotic": {"psychotic disorder", "psychotic disorders", "psychosis"},
        "schizophrenia": {"schizophrenia", "schizophrenic"},
        "post traumatic": {"post traumatic stress disorder", "posttraumatic stress disorder", "ptsd"},
        "posttraumatic": {"post traumatic stress disorder", "posttraumatic stress disorder", "ptsd"},
        "anxiety": {"anxiety disorder", "anxiety disorders", "anxiety"},
        "substance": {
            "substance related disorders",
            "substance use disorder",
            "substance use disorders",
            "substance dependence",
            "addiction",
        },
    }
    for marker, values in groups.items():
        if marker in normalized:
            aliases.update(values)
    return {_norm(alias) for alias in aliases if alias}


def _region_aliases(name: str) -> set[str]:
    normalized = _norm(str(name or "").split("|")[0])
    aliases = {normalized} if normalized else set()
    groups = {
        "amygdala": {"amygdala", "amygdalar"},
        "hippocamp": {"hippocampus", "hippocampal"},
        "insula": {"insula", "insular cortex", "insular"},
        "thalam": {"thalamus", "thalamic"},
        "posterior cingulate": {"posterior cingulate", "posterior cingulate cortex", "pcc"},
        "fusiform": {"fusiform gyrus", "fusiform"},
        "middle temporal": {"middle temporal gyrus", "middle temporal"},
        "inferior temporal": {"inferior temporal gyrus", "inferior temporal"},
        "orbitofrontal": {"orbitofrontal cortex", "orbitofrontal"},
        "frontal pole": {"frontal pole", "frontopolar cortex", "frontopolar"},
    }
    for marker, values in groups.items():
        if marker in normalized:
            aliases.update(values)
    return {_norm(alias) for alias in aliases if alias and len(_norm(alias)) >= 3}


def _feature_aliases(name: str) -> set[str]:
    normalized = _norm(name)
    aliases = {normalized} if normalized else set()
    groups = {
        "cortthick": {"cortical thickness", "cortical thinning", "cortthick"},
        "cortical thickness": {"cortical thickness", "cortical thinning"},
        "alff": {"alff", "amplitude of low frequency fluctuation"},
        "reho": {"reho", "regional homogeneity"},
        "functional connectivity": {"functional connectivity", "resting state connectivity"},
        "participation coefficient": {"participation coefficient", "network participation"},
        "temporal variance": {"temporal variance", "signal variance"},
        "volume": {"volume", "volumetry", "volumetric"},
    }
    for marker, values in groups.items():
        if marker in normalized:
            aliases.update(values)
    return {_norm(alias) for alias in aliases if alias and len(_norm(alias)) >= 3}


def _sentences(text: Any, limit: int = 2) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", value) if piece.strip()]
    return pieces[:limit] if pieces else [value]


def _paper_key(paper: dict[str, Any]) -> str:
    return _norm(paper.get("pmid") or paper.get("doi") or paper.get("title"))


def _candidate_literature_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    candidate_tuple = metadata.get("candidate_tuple") if isinstance(metadata.get("candidate_tuple"), dict) else {}
    path = [edge for edge in candidate.get("path") or [] if isinstance(edge, dict)]
    diseases = [str(item).strip() for item in candidate_tuple.get("diseases") or [] if str(item).strip()]
    if not diseases:
        diseases = list(dict.fromkeys(str(edge.get("from_name") or "").strip() for edge in path if edge.get("from_name")))
    region = str(
        candidate_tuple.get("region_name")
        or metadata.get("cluster_region_name")
        or (path[0].get("to_name") if path else "")
        or candidate.get("target_name")
        or ""
    ).split("|")[0].strip()
    feature = str(
        candidate_tuple.get("feature_name")
        or metadata.get("cluster_modality")
        or (path[0].get("evidence", {}).get("candidate_feature_id") if path and isinstance(path[0].get("evidence"), dict) else "")
        or candidate.get("hypothesis_type")
        or ""
    ).strip()
    return {
        "candidate": candidate,
        "diseases": diseases,
        "disease_aliases": {disease: _disease_aliases(disease) for disease in diseases},
        "region_aliases": _region_aliases(region),
        "feature_aliases": _feature_aliases(feature),
        "papers": {},
    }


def _add_literature_record(
    profile: dict[str, Any],
    paper: dict[str, Any],
    excerpt: Any,
    *,
    score: float,
    disease: str = "",
    direct: bool = False,
) -> None:
    title = str(paper.get("title") or "").strip()
    key = _paper_key(paper)
    if not title or not key:
        return
    rows: dict[str, dict[str, Any]] = profile["papers"]
    row = rows.setdefault(
        key,
        {
            "title": title,
            "authors": paper.get("authors") or "",
            "year": paper.get("year") or "",
            "journal": paper.get("journal") or "",
            "pmid": paper.get("pmid") or "",
            "doi": paper.get("doi") or "",
            "url": paper.get("url") or paper.get("link") or "",
            "excerpts": [],
            "_score": 0.0,
            "_direct": False,
            "_diseases": set(),
        },
    )
    row["_score"] = max(float(row["_score"]), float(score))
    row["_direct"] = bool(row["_direct"] or direct)
    if disease:
        row["_diseases"].add(disease)
    for sentence in _sentences(excerpt):
        if sentence not in row["excerpts"] and len(row["excerpts"]) < 2:
            row["excerpts"].append(sentence)


def enrich_literature(
    candidates: list[dict[str, Any]],
    claims_path: Path | None,
    *,
    max_references: int = 5,
) -> dict[str, Any]:
    profiles = [_candidate_literature_profile(candidate) for candidate in candidates]
    for profile in profiles:
        for edge in profile["candidate"].get("path") or []:
            if not isinstance(edge, dict):
                continue
            paper = edge.get("source_paper") if isinstance(edge.get("source_paper"), dict) else {}
            _add_literature_record(
                profile,
                paper,
                edge.get("raw_text") or paper.get("abstract") or "",
                score=100.0 + abs(float(edge.get("confidence") or 0.0)),
                disease=str(edge.get("from_name") or ""),
                direct=True,
            )

    claims_scanned = 0
    claims_used = 0
    if claims_path and claims_path.exists():
        with claims_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                claims_scanned += 1
                try:
                    claim = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper = claim.get("source_paper") if isinstance(claim.get("source_paper"), dict) else {}
                if not str(paper.get("title") or "").strip():
                    continue
                raw_text = str(claim.get("raw_text") or "").strip()
                if not raw_text:
                    continue
                evidence = claim.get("evidence") if isinstance(claim.get("evidence"), dict) else {}
                claim_disease = _norm(claim.get("disease") or claim.get("disease_query"))
                claim_text = _norm(
                    " ".join(
                        str(value or "")
                        for value in (
                            claim.get("disease"),
                            claim.get("subject_name"),
                            claim.get("object_name"),
                            raw_text,
                            evidence.get("methodology"),
                            evidence.get("study_type"),
                        )
                    )
                )
                neuroimaging_hit = any(
                    term in claim_text
                    for term in ("mri", "fmri", "neuroimaging", "brain", "cortical", "connectivity", "volume")
                )
                matched_any = False
                for profile in profiles:
                    matched_diseases = [
                        disease
                        for disease, aliases in profile["disease_aliases"].items()
                        if any(
                            alias and (
                                (claim_disease and (
                                    alias == claim_disease
                                    or alias in claim_disease
                                    or claim_disease in alias
                                ))
                                or (not claim_disease and alias in claim_text)
                            )
                            for alias in aliases
                        )
                    ]
                    if not matched_diseases:
                        continue
                    region_hit = any(alias in claim_text for alias in profile["region_aliases"])
                    feature_hit = any(alias in claim_text for alias in profile["feature_aliases"])
                    if not (region_hit or feature_hit or neuroimaging_hit):
                        continue
                    score = 20.0 + (14.0 if region_hit else 0.0) + (5.0 if feature_hit else 0.0) + (1.0 if neuroimaging_hit else 0.0)
                    try:
                        score += max(0.0, float(claim.get("confidence") or 0.0)) * 2.0
                    except (TypeError, ValueError):
                        pass
                    for disease in matched_diseases:
                        _add_literature_record(profile, paper, raw_text, score=score, disease=disease)
                    matched_any = True
                if matched_any:
                    claims_used += 1

    counts: list[int] = []
    for profile in profiles:
        pool = list(profile["papers"].values())
        selected: list[dict[str, Any]] = []
        covered: set[str] = set()
        remaining = list(pool)
        while remaining and len(selected) < max_references:
            remaining.sort(
                key=lambda row: (
                    bool(set(row["_diseases"]) - covered),
                    bool(row["_direct"]),
                    float(row["_score"]),
                    int(row["year"] or 0) if str(row["year"] or "").isdigit() else 0,
                    row["title"],
                ),
                reverse=True,
            )
            row = remaining.pop(0)
            selected.append(row)
            covered.update(row["_diseases"])
        public_rows = []
        for row in selected:
            public_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
        profile["candidate"]["literature"] = public_rows
        counts.append(len(public_rows))
    return {
        "claims_source": safe_source_path(claims_path),
        "claims_scanned": claims_scanned,
        "claims_used": claims_used,
        "max_references": max_references,
        "candidates_with_five": sum(count >= max_references for count in counts),
        "minimum_references": min(counts) if counts else 0,
    }


def _abstract_keys(paper: dict[str, Any]) -> list[str]:
    pmid = str(paper.get("pmid") or "").strip()
    doi = str(paper.get("doi") or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    title = _norm(paper.get("title"))
    keys = []
    if pmid:
        keys.append(f"pmid:{pmid}")
    if doi:
        keys.append(f"doi:{doi}")
    if title:
        keys.append(f"title:{title}")
    return keys


def load_abstract_index(paths: tuple[Path, ...] | list[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
                merged = {**paper, **{key: row.get(key) for key in ("pmid", "doi", "title") if row.get(key)}}
                abstract = re.sub(r"\s+", " ", str(row.get("abstract") or "")).strip()
                if not abstract:
                    continue
                record = {"abstract": abstract, "source": safe_source_path(path)}
                for key in _abstract_keys(merged):
                    index.setdefault(key, record)
    return index


def fetch_pubmed_abstracts(pmids: list[str], *, timeout: float = 45.0) -> tuple[dict[str, str], str]:
    """Fetch official PubMed abstracts in batches; failures remain non-fatal."""
    recovered: dict[str, str] = {}
    try:
        for start in range(0, len(pmids), 100):
            batch = pmids[start : start + 100]
            if not batch:
                continue
            query = urllib.parse.urlencode(
                {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            )
            request = urllib.request.Request(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}",
                headers={
                    "User-Agent": "NeuroDiscovery/0.2.2 abstract-backfill contact=research@example.com"
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                root = ET.fromstring(response.read())
            for article in root.findall(".//PubmedArticle"):
                pmid = str(article.findtext(".//MedlineCitation/PMID") or "").strip()
                parts = []
                for element in article.findall(".//Article/Abstract/AbstractText"):
                    text = re.sub(r"\s+", " ", "".join(element.itertext())).strip()
                    label = str(element.attrib.get("Label") or "").strip()
                    if text:
                        parts.append(f"{label}: {text}" if label else text)
                if pmid and parts:
                    recovered[pmid] = " ".join(parts)
        return recovered, ""
    except Exception as exc:  # network is a best-effort enrichment step
        return recovered, str(exc)


def fetch_pubmed_identifiers(
    pmids: list[str], *, timeout: float = 45.0
) -> tuple[dict[str, dict[str, str]], str]:
    """Resolve DOI and PMC identifiers from official PubMed records."""
    recovered: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for start in range(0, len(pmids), 100):
        batch = pmids[start : start + 100]
        if not batch:
            continue
        last_error = ""
        root = None
        for _attempt in range(2):
            try:
                query = urllib.parse.urlencode(
                    {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
                )
                request = urllib.request.Request(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}",
                    headers={
                        "User-Agent": "NeuroDiscovery/0.2.2 reference-link-backfill contact=research@example.com"
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    root = ET.fromstring(response.read())
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
        if root is None:
            errors.append(f"batch {start // 100 + 1}: {last_error}")
            continue
        for article in root.findall(".//PubmedArticle"):
            pmid = str(article.findtext(".//MedlineCitation/PMID") or "").strip()
            if not pmid:
                continue
            identifiers = {"pmid": pmid, "doi": "", "pmcid": ""}
            for element in article.findall("./PubmedData/ArticleIdList/ArticleId"):
                kind = str(element.attrib.get("IdType") or "").strip().lower()
                value = str(element.text or "").strip()
                if kind == "doi" and value:
                    identifiers["doi"] = value
                elif kind == "pmc" and value:
                    identifiers["pmcid"] = value.upper()
            recovered[pmid] = identifiers
    return recovered, "; ".join(errors)


def load_reference_link_index(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not path.exists() or not path.is_file():
        return index
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(row.get("pmid") or "").strip()
            if pmid:
                index[pmid] = {
                    "pmid": pmid,
                    "doi": str(row.get("doi") or "").strip(),
                    "pmcid": str(row.get("pmcid") or "").strip().upper(),
                }
    return index


def _valid_http_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urllib.parse.urlparse(text)
    except ValueError:
        return ""
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def enrich_reference_links(
    candidates: list[dict[str, Any]],
    *,
    fetch_missing: bool = True,
    output_path: Path = DEFAULT_STUDY_LINK_CACHE,
) -> dict[str, Any]:
    """Attach direct PMC, DOI, and PubMed destinations to every reference."""
    references = [
        paper
        for candidate in candidates
        for paper in candidate.get("literature") or []
        if isinstance(paper, dict)
    ]
    index = load_reference_link_index(output_path)
    pmids = sorted({str(paper.get("pmid") or "").strip() for paper in references} - {""})
    missing_pmids = [pmid for pmid in pmids if pmid not in index]
    network_recovered: dict[str, dict[str, str]] = {}
    network_error = ""
    if fetch_missing and missing_pmids:
        network_recovered, network_error = fetch_pubmed_identifiers(missing_pmids)
        if network_recovered:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                for pmid, row in sorted(network_recovered.items()):
                    if pmid in index:
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    index[pmid] = row

    direct = 0
    doi_count = 0
    pmc_count = 0
    pubmed_count = 0
    for paper in references:
        pmid = str(paper.get("pmid") or "").strip()
        resolved = index.get(pmid, {})
        doi = str(paper.get("doi") or resolved.get("doi") or "").strip()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        pmcid = str(paper.get("pmcid") or resolved.get("pmcid") or "").strip().upper()
        if pmcid and not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        source_url = _valid_http_url(paper.get("url") or paper.get("link"))
        doi_url = f"https://doi.org/{urllib.parse.quote(doi, safe='/():;._-')}" if doi else ""
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{urllib.parse.quote(pmid)}/" if pmid else ""
        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{urllib.parse.quote(pmcid)}/" if pmcid else ""
        direct_url = pmc_url or doi_url or source_url or pubmed_url
        paper["doi"] = doi
        paper["pmcid"] = pmcid
        paper["doi_url"] = doi_url
        paper["pubmed_url"] = pubmed_url
        paper["pmc_url"] = pmc_url
        paper["direct_url"] = direct_url
        paper["url"] = direct_url
        direct += bool(direct_url)
        doi_count += bool(doi_url)
        pmc_count += bool(pmc_url)
        pubmed_count += bool(pubmed_url)

    return {
        "references": len(references),
        "references_with_direct_url": direct,
        "references_with_doi_url": doi_count,
        "references_with_pmc_full_text": pmc_count,
        "references_with_pubmed_url": pubmed_count,
        "unique_pmids": len(pmids),
        "network_requested": len(missing_pmids) if fetch_missing else 0,
        "network_recovered": len(network_recovered),
        "network_error": network_error,
        "cache": safe_source_path(output_path),
    }


def enrich_reference_abstracts(
    candidates: list[dict[str, Any]],
    *,
    abstract_sources: tuple[Path, ...] | list[Path] = DEFAULT_ABSTRACT_SOURCES,
    fetch_missing: bool = True,
    output_path: Path = DEFAULT_STUDY_ABSTRACT_CACHE,
) -> dict[str, Any]:
    references = [
        paper
        for candidate in candidates
        for paper in candidate.get("literature") or []
        if isinstance(paper, dict)
    ]
    index = load_abstract_index(abstract_sources)

    def lookup(paper: dict[str, Any]) -> dict[str, str] | None:
        return next((index[key] for key in _abstract_keys(paper) if key in index), None)

    missing_pmids = sorted(
        {
            str(paper.get("pmid") or "").strip()
            for paper in references
            if not lookup(paper) and str(paper.get("pmid") or "").strip()
        }
    )
    network_recovered: dict[str, str] = {}
    network_error = ""
    if fetch_missing and missing_pmids:
        network_recovered, network_error = fetch_pubmed_abstracts(missing_pmids)
        if network_recovered:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing = load_abstract_index([output_path])
            with output_path.open("a", encoding="utf-8") as handle:
                for pmid, abstract in sorted(network_recovered.items()):
                    if f"pmid:{pmid}" in existing:
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "pmid": pmid,
                                "abstract": abstract,
                                "source": "pubmed_efetch",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    index[f"pmid:{pmid}"] = {
                        "abstract": abstract,
                        "source": safe_source_path(output_path),
                    }

    source_counts: Counter[str] = Counter()
    unique_available: set[str] = set()
    unique_all: set[str] = set()
    for paper in references:
        keys = _abstract_keys(paper)
        paper_key = keys[0] if keys else f"anonymous:{id(paper)}"
        unique_all.add(paper_key)
        match = lookup(paper)
        if match:
            paper["abstract"] = match["abstract"]
            paper["abstract_source"] = match["source"]
            paper["abstract_status"] = "available"
            source_counts[match["source"]] += 1
            unique_available.add(paper_key)
        else:
            paper.pop("abstract", None)
            paper["abstract_source"] = ""
            paper["abstract_status"] = "missing"

    return {
        "references": len(references),
        "references_with_abstract": sum(
            paper.get("abstract_status") == "available" for paper in references
        ),
        "unique_papers": len(unique_all),
        "unique_papers_with_abstract": len(unique_available),
        "missing_unique_papers": len(unique_all - unique_available),
        "network_requested": len(missing_pmids) if fetch_missing else 0,
        "network_recovered": len(network_recovered),
        "network_error": network_error,
        "source_counts": dict(sorted(source_counts.items())),
    }


def support_score(candidate: dict[str, Any]) -> float:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    try:
        return float(metadata.get("support_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def validation_score(row: dict[str, Any]) -> float:
    verdict_weight = 2.0 if row.get("pilot_verdict") == "region_signal_present" else 1.0
    null = row.get("random_roi_null") if isinstance(row.get("random_roi_null"), dict) else {}
    percentile = max(
        float(null.get("roi_std_abs_d_percentile") or 0.0),
        float(null.get("fc_abs_d_percentile") or 0.0),
    )
    return verdict_weight + percentile


def annotate_comparison_task(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    metadata = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
    out["metadata"] = metadata
    task_id = candidate_comparison_task(out)
    out["comparison_task_id"] = task_id
    metadata["comparison_task_id"] = task_id
    return out


def annotate_directional_hypothesis(candidate: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy direction-neutral Case 1 records to explicit predictions."""
    out = dict(candidate)
    metadata = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
    candidate_tuple = (
        dict(metadata.get("candidate_tuple"))
        if isinstance(metadata.get("candidate_tuple"), dict)
        else {}
    )
    disease = {
        "id": candidate_tuple.get("disease_id") or out.get("source_id"),
        "name": candidate_tuple.get("disease_name") or out.get("source_name"),
    }
    region = {
        "id": candidate_tuple.get("region_id"),
        "name": candidate_tuple.get("region_name") or str(out.get("target_name") or "").split("|")[0].strip(),
    }
    feature = {
        "id": candidate_tuple.get("feature_id"),
        "name": candidate_tuple.get("feature_name") or str(out.get("target_name") or "").split("|")[-1].strip(),
    }
    proposal = propose_case1_direction(disease, region, feature, out.get("path") or [])
    direction = proposal["direction"]
    disease_name = str(disease["name"] or "Disease")
    region_name = str(region["name"] or "the selected ROI")
    feature_name = str(feature["name"] or "the selected imaging feature")

    candidate_tuple["direction"] = direction
    metadata["candidate_tuple"] = candidate_tuple
    metadata["direction_assumption"] = direction
    metadata["direction_source"] = proposal["source"]
    metadata["directional_evidence_votes"] = proposal["directional_evidence_votes"]
    title = case1_directional_title(disease_name, region_name, feature_name, direction)
    statement = case1_directional_statement(disease_name, region_name, feature_name, direction)
    metadata["display_title"] = title
    out["metadata"] = metadata
    out["title"] = title
    out["explanation"] = statement
    out["summary"] = statement
    return out


def safe_source_path(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def execution_labels(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = load_json(path)
    rows = payload if isinstance(payload, list) else payload.get("execution_results", payload.get("results", []))
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        hypothesis_id = str(row.get("hypothesis_id") or row.get("id") or "").strip()
        status = str(row.get("status") or row.get("outcome") or "").strip().lower()
        if hypothesis_id and status:
            labels[hypothesis_id] = status
    return labels


def build_subset(
    current_path: Path,
    validated_path: Path,
    validation_path: Path,
    *,
    output_path: Path,
    execution_path: Path | None = None,
    claims_path: Path | None = DEFAULT_CLAIMS,
    abstract_sources: tuple[Path, ...] | list[Path] = DEFAULT_ABSTRACT_SOURCES,
    fetch_missing_abstracts: bool = True,
    abstract_output_path: Path = DEFAULT_STUDY_ABSTRACT_CACHE,
    fetch_missing_links: bool = True,
    link_output_path: Path = DEFAULT_STUDY_LINK_CACHE,
    target_size: int = 320,
    target_pairs: int = 300,
    max_references: int = 5,
    seed: int = 20260615,
) -> dict[str, Any]:
    current = [
        annotate_comparison_task(annotate_directional_hypothesis(item))
        for item in single_disease_case1_candidates_from(current_path)
    ]
    validated = [
        annotate_comparison_task(annotate_directional_hypothesis(item))
        for item in single_disease_case1_candidates_from(validated_path)
    ]
    all_candidates = {str(item["id"]): item for item in [*current, *validated]}
    strict_labels = execution_labels(execution_path)
    validation_payload = load_json(validation_path)
    validation_rows = [
        row for row in validation_payload.get("results", []) if isinstance(row, dict) and row.get("hypothesis_id")
    ]

    strict_confirmed = {
        hypothesis_id
        for hypothesis_id, status in strict_labels.items()
        if status == "confirmed" and hypothesis_id in all_candidates
    }
    pilot_scores = {
        str(row["hypothesis_id"]): validation_score(row)
        for row in validation_rows
        if str(row["hypothesis_id"]) in all_candidates
    }
    pilot_positive = {
        hypothesis_id
        for hypothesis_id, _score in sorted(
            pilot_scores.items(), key=lambda item: item[1], reverse=True
        )[:8]
    }

    def semantic_candidate_ids(candidate: dict[str, Any]) -> set[str]:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        aliases = metadata.get("semantic_candidate_ids") or []
        return {str(candidate["id"]), *(str(item) for item in aliases if str(item))}

    def outcome_label(candidate: dict[str, Any]) -> str:
        hypothesis_ids = semantic_candidate_ids(candidate)
        if hypothesis_ids & strict_confirmed:
            return "confirmed"
        if hypothesis_ids & pilot_positive:
            return "pilot_proxy_positive"
        if paper_count(candidate) > 0:
            return "literature_supported"
        statuses = {strict_labels[item] for item in hypothesis_ids if item in strict_labels}
        return sorted(statuses)[0] if statuses else "unverified"

    label_priority = {
        "confirmed": 4,
        "pilot_proxy_positive": 3,
        "literature_supported": 2,
        "unverified": 1,
    }
    def candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
        label = outcome_label(candidate)
        hypothesis_ids = semantic_candidate_ids(candidate)
        return (
            -label_priority.get(label, 0),
            -max((pilot_scores.get(item, 0.0) for item in hypothesis_ids), default=0.0),
            -paper_count(candidate),
            -support_score(candidate),
            hashlib.sha256(f"{seed}:{candidate['id']}".encode()).hexdigest(),
        )

    unique_candidates, deduplication = collapse_semantic_duplicates(
        list(all_candidates.values()),
        rank_key=candidate_rank,
    )
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in unique_candidates:
        task_groups.setdefault(candidate_comparison_task(candidate), []).append(candidate)
    task_groups = {task_id: rows for task_id, rows in task_groups.items() if len(rows) >= 2}
    if len(task_groups) < 2:
        raise ValueError("Expert subset requires at least two comparison tasks with two hypotheses each")

    for rows in task_groups.values():
        rows.sort(key=candidate_rank)
    ordered_tasks = sorted(
        task_groups,
        key=lambda task_id: (
            candidate_rank(task_groups[task_id][0]),
            -len(task_groups[task_id]),
            task_id,
        ),
    )

    # Cover multiple scientific questions, while keeping at least two
    # alternatives per question so every selected hypothesis is rankable.
    chosen_tasks: list[str] = []
    available = 0
    minimum_tasks = min(8, max(2, target_size // 4), len(ordered_tasks))
    for task_id in ordered_tasks:
        chosen_tasks.append(task_id)
        available += len(task_groups[task_id])
        if len(chosen_tasks) >= minimum_tasks and available >= target_size:
            break

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    labels: dict[str, str] = {}
    cursors = {task_id: 0 for task_id in chosen_tasks}

    def add_from_task(task_id: str) -> bool:
        rows = task_groups[task_id]
        cursor = cursors[task_id]
        while cursor < len(rows):
            candidate = rows[cursor]
            cursor += 1
            cursors[task_id] = cursor
            hypothesis_id = str(candidate["id"])
            if hypothesis_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(hypothesis_id)
            labels[hypothesis_id] = outcome_label(candidate)
            return True
        return False

    for task_id in chosen_tasks:
        add_from_task(task_id)
        add_from_task(task_id)
    while len(selected) < target_size:
        progressed = False
        for task_id in chosen_tasks:
            if len(selected) >= target_size:
                break
            progressed = add_from_task(task_id) or progressed
        if not progressed:
            break

    literature_summary = enrich_literature(
        selected,
        claims_path,
        max_references=max(1, int(max_references)),
    )
    abstract_summary = enrich_reference_abstracts(
        selected,
        abstract_sources=abstract_sources,
        fetch_missing=fetch_missing_abstracts,
        output_path=abstract_output_path,
    )
    link_summary = enrich_reference_links(
        selected,
        fetch_missing=fetch_missing_links,
        output_path=link_output_path,
    )
    pairs = build_progressive_pair_schedule(selected, max_pairs=max(1, int(target_pairs)), seed=seed)
    if len(pairs) < target_pairs:
        raise ValueError(
            f"Selected {len(selected)} hypotheses yield only {len(pairs)} valid comparisons; "
            f"increase --target-size to reach --target-pairs={target_pairs}."
        )
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    difficulty_counts: dict[str, int] = {}
    for pair in pairs:
        difficulty_counts[pair["difficulty"]] = difficulty_counts.get(pair["difficulty"], 0) + 1
    expected_per_tier = target_pairs // 3
    expected_difficulty_counts = {
        "easy": expected_per_tier + (1 if target_pairs % 3 >= 1 else 0),
        "medium": expected_per_tier + (1 if target_pairs % 3 >= 2 else 0),
        "hard": expected_per_tier,
    }
    if difficulty_counts != expected_difficulty_counts:
        raise ValueError(
            "Could not build a balanced difficulty bank: "
            f"expected {expected_difficulty_counts}, got {difficulty_counts}. "
            "Increase --target-size or improve candidate diversity."
        )

    payload = {
        "schema_version": "case1-expert-subset-v6",
        "hypothesis_unit": "single_disease_x_roi_x_imaging_feature",
        "blinding": "Only the hypotheses list is served to expert clients; outcome_labels remain server-side.",
        "confirmation_basis": (
            "strict_execution_results"
            if strict_confirmed
            else "provisional_pilot_and_literature"
            if pilot_positive
            else "literature_support_only"
        ),
        "n_hypotheses": len(selected),
        "n_pairs": len(pairs),
        "n_comparison_tasks": len({candidate_comparison_task(item) for item in selected}),
        "comparison_task_counts": {
            task_id: sum(candidate_comparison_task(item) == task_id for item in selected)
            for task_id in sorted({candidate_comparison_task(item) for item in selected})
        },
        "label_counts": counts,
        "pair_difficulty_counts": difficulty_counts,
        "literature_enrichment": literature_summary,
        "abstract_enrichment": abstract_summary,
        "reference_link_enrichment": link_summary,
        "semantic_deduplication": deduplication,
        "source_candidate_space": max(
            (int((item.get("metadata") or {}).get("total_candidate_space") or 0) for item in selected),
            default=0,
        ),
        "provenance": {
            "current_candidates": safe_source_path(current_path),
            "validated_candidates": safe_source_path(validated_path),
            "pilot_validation": safe_source_path(validation_path),
            "execution_results": safe_source_path(execution_path),
            "seed": seed,
        },
        "outcome_labels": labels,
        "pair_schedule": pairs,
        "hypotheses": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--execution-results", type=Path)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument(
        "--abstract-source",
        action="append",
        type=Path,
        dest="abstract_sources",
        help="Additional JSONL abstract cache. Repeat to provide multiple caches.",
    )
    parser.add_argument("--abstract-output", type=Path, default=DEFAULT_STUDY_ABSTRACT_CACHE)
    parser.add_argument("--link-output", type=Path, default=DEFAULT_STUDY_LINK_CACHE)
    parser.add_argument(
        "--fetch-missing-abstracts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Best-effort PubMed backfill for selected references not found in local caches.",
    )
    parser.add_argument(
        "--fetch-missing-links",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve DOI and PMC identifiers from official PubMed records.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-size",
        type=int,
        default=320,
        help="Number of unique hypotheses retained to support the comparison questions.",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=300,
        help="Exact number of pairwise expert questions to generate.",
    )
    parser.add_argument("--max-references", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()
    payload = build_subset(
        args.current,
        args.validated,
        args.validation,
        output_path=args.output,
        execution_path=args.execution_results,
        claims_path=args.claims,
        abstract_sources=tuple(args.abstract_sources or DEFAULT_ABSTRACT_SOURCES),
        fetch_missing_abstracts=args.fetch_missing_abstracts,
        abstract_output_path=args.abstract_output,
        fetch_missing_links=args.fetch_missing_links,
        link_output_path=args.link_output,
        target_size=max(2, args.target_size),
        target_pairs=max(1, args.target_pairs),
        max_references=max(1, args.max_references),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_hypotheses": payload["n_hypotheses"],
                "n_pairs": payload["n_pairs"],
                "confirmation_basis": payload["confirmation_basis"],
                "label_counts": payload["label_counts"],
                "pair_difficulty_counts": payload["pair_difficulty_counts"],
                "literature_enrichment": payload["literature_enrichment"],
                "abstract_enrichment": payload["abstract_enrichment"],
                "reference_link_enrichment": payload["reference_link_enrichment"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
