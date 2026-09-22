"""Re-curate abstract evidence and credibility metadata for Expert Study papers.

The input bank already contains hypothesis-specific, manually reviewed relevance
reasons. This postprocessor uses those reasons to select one to four verbatim
abstract sentences for each hypothesis-paper association. It also records only
credibility metadata that can be supported by the local abstract or the checked
journal-metrics registry. Missing values remain explicit instead of being guessed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS = SCRIPT_DIR / "resources" / "case1_journal_metrics_v1.json"
DEFAULT_GOLD = SCRIPT_DIR / "resources" / "case1_evidence_sentence_gold_v1.json"
DEFAULT_CREDIBILITY = SCRIPT_DIR / "resources" / "case1_paper_credibility_overrides_v1.json"

SECTION_LABELS = (
    "BACKGROUND AND HYPOTHESIS",
    "BACKGROUND",
    "OBJECTIVE",
    "OBJECTIVES",
    "HYPOTHESIS",
    "STUDY DESIGN",
    "DESIGN",
    "AIM",
    "AIMS",
    "METHODS",
    "METHOD",
    "STUDY RESULTS",
    "RESULTS",
    "RESULT",
    "CONCLUSIONS",
    "CONCLUSION",
)
SECTION_PATTERN = "|".join(re.escape(value) for value in SECTION_LABELS)

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "and", "are",
    "because", "been", "before", "being", "between", "both", "but", "can",
    "compared", "could", "did", "does", "each", "for", "from", "had", "has",
    "have", "here", "into", "its", "more", "most", "not", "only", "other",
    "our", "over", "paper", "patients", "relative", "reported", "reports",
    "showed", "shows", "study", "than", "that", "the", "their", "these",
    "they", "this", "those", "through", "using", "was", "were", "what",
    "when", "where", "which", "while", "with", "within", "would",
}

RESULT_TERMS = re.compile(
    r"\b(results?|found|showed|exhibited|revealed|demonstrated|associated|"
    r"correlat|higher|lower|increase|decrease|reduc|significant|"
    r"hyperactiv|hypoactiv|greater|smaller|predicted|implicated)\w*\b",
    re.IGNORECASE,
)
BACKGROUND_TERMS = re.compile(
    r"\b(remain(?:s|ed)? unclear|frequently used|little is known|"
    r"we aimed|this study (?:aimed|was conducted)|objective:)\b",
    re.IGNORECASE,
)
METHOD_TERMS = re.compile(
    r"\b(calculated|computed|measured|acquired|analy[sz]ed|utili[sz]ed|"
    r"resting-state|functional magnetic resonance|fMRI|fNIRS|meta-analysis)\b",
    re.IGNORECASE,
)
COHORT_TERMS = re.compile(
    r"\b(participants?|patients?|controls?|subjects?|individuals?|cohorts?|"
    r"TDCs?|HCs?|SSDs?|SCZ|BD|ADHD)\b",
    re.IGNORECASE,
)
P_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:FDR\s+)?(?:p(?:[-_ ]?(?:voxel|cluster|fdr))?|q)"
    r"\s*(?:-?value\s*)?(?:=|<|>|≤|≥)\s*"
    r"(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKD", compact(value)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def paper_id(paper: dict[str, Any]) -> str:
    return compact(paper.get("pmid") or paper.get("doi") or paper.get("title")).casefold()


def split_sentences(abstract: str) -> list[str]:
    text = compact(abstract)
    if not text:
        return []
    text = re.sub(
        rf"\s+(?=(?:{SECTION_PATTERN}):\s*)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    chunks: list[str] = []
    for section in text.split("\n"):
        chunks.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", section))
    output: list[str] = []
    seen: set[str] = set()
    for sentence in chunks:
        sentence = compact(sentence)
        key = normalized(sentence)
        if len(sentence) < 24 or not key or key in seen:
            continue
        seen.add(key)
        output.append(sentence)
    return output


def reason_sections(reason: str) -> dict[str, str]:
    text = str(reason or "").strip()
    patterns = {
        "studied": r"What this paper studied:\s*(.*?)(?=\s*Specific support for this hypothesis:|$)",
        "support": r"Specific support for this hypothesis:\s*(.*?)(?=\s*Evidence strength:|$)",
        "strength": r"Evidence strength:\s*(.*)$",
    }
    return {
        key: compact(match.group(1)) if (match := re.search(pattern, text, re.S)) else ""
        for key, pattern in patterns.items()
    }


def reason_sections_zh(reason: str) -> dict[str, str]:
    text = str(reason or "").strip()
    patterns = {
        "studied": r"本文研究的是：\s*(.*?)(?=\s*对当前 hypothesis 的具体支持：|$)",
        "support": r"对当前 hypothesis 的具体支持：\s*(.*?)(?=\s*证据强度：|$)",
        "strength": r"证据强度：\s*(.*)$",
    }
    return {
        key: compact(match.group(1)) if (match := re.search(pattern, text, re.S)) else ""
        for key, pattern in patterns.items()
    }


def tokens(value: Any) -> list[str]:
    return [
        token
        for token in normalized(value).split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]


def candidate_terms(hypothesis: dict[str, Any]) -> dict[str, list[str] | str]:
    candidate = dict(hypothesis.get("metadata", {}).get("candidate_tuple", {}))
    disease = compact(candidate.get("disease_name") or hypothesis.get("source_name"))
    region = compact(candidate.get("region_name"))
    feature = compact(candidate.get("feature_name"))
    direction = compact(candidate.get("direction"))

    disease_aliases = [disease]
    disease_key = normalized(disease)
    if "schizophrenia" in disease_key or "psychosis" in disease_key:
        disease_aliases.extend(
            ["schizophrenia", "schizophrenia spectrum", "psychosis", "psychotic", "SCZ", "SZ", "SSD"]
        )
    if "bipolar" in disease_key:
        disease_aliases.extend(["bipolar disorder", "BD"])
    if "attention deficit" in disease_key or "adhd" in disease_key:
        disease_aliases.extend(["attention deficit hyperactivity disorder", "ADHD"])

    feature_aliases = [feature]
    feature_key = normalized(feature)
    if "falff" in feature_key:
        feature_aliases.extend(["fALFF", "fractional amplitude of low-frequency fluctuation", "ALFF"])
    elif "alff" in feature_key:
        feature_aliases.extend(["ALFF", "amplitude of low-frequency fluctuation", "fALFF"])
    if "connect" in feature_key or "corr" in feature_key or "degree" in feature_key:
        feature_aliases.extend(["functional connectivity", "connectivity", "FC", "network"])
    if "volume" in feature_key:
        feature_aliases.extend(["volume", "volumetric", "gray matter"])

    region_aliases = [region]
    region_key = normalized(region)
    expansions = {
        "ofc": "orbitofrontal cortex",
        "dlpfc": "dorsolateral prefrontal cortex",
        "vlpfc": "ventrolateral prefrontal cortex",
        "acc": "anterior cingulate cortex",
        "pcc": "posterior cingulate cortex",
        "stg": "superior temporal gyrus",
        "mtg": "middle temporal gyrus",
        "itg": "inferior temporal gyrus",
    }
    for abbreviation, expansion in expansions.items():
        if re.search(rf"\b{abbreviation}\b", region_key):
            region_aliases.append(expansion)
    region_families = {
        "temporal": "temporal",
        "cingulate": "cingulate",
        "insula": "insula",
        "insular": "insula",
        "hippocamp": "hippocampus",
        "thalam": "thalamus",
        "frontal": "frontal",
        "parietal": "parietal",
        "occipital": "occipital",
        "cerebell": "cerebellum",
        "amygdala": "amygdala",
        "striat": "striatum",
        "putamen": "putamen",
        "caudate": "caudate",
        "precuneus": "precuneus",
    }
    for marker, alias in region_families.items():
        if marker in region_key:
            region_aliases.append(alias)
    meaningful_region = [
        token for token in tokens(region)
        if token not in {"networks", "multiatlas", "roi", "left", "right", "lh", "rh"}
    ]
    if meaningful_region:
        region_aliases.append(" ".join(meaningful_region))
    return {
        "disease": disease_aliases,
        "region": region_aliases,
        "feature": feature_aliases,
        "direction": direction,
    }


def contains_alias(sentence: str, aliases: Iterable[str]) -> bool:
    haystack = f" {normalized(sentence)} "
    for alias in aliases:
        needle = normalized(alias)
        if needle and f" {needle} " in haystack:
            return True
    return False


def sentence_similarity(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def sentence_scores(hypothesis: dict[str, Any], paper: dict[str, Any]) -> list[dict[str, Any]]:
    rows = split_sentences(compact(paper.get("abstract")))
    reason = reason_sections(compact(paper.get("relevance_reason_en")))
    terms = candidate_terms(hypothesis)
    weighted_query: dict[str, float] = {}
    for value, weight in (
        (reason["support"], 4.0),
        (reason["studied"], 2.0),
        (paper.get("title"), 1.0),
        (hypothesis.get("title"), 1.5),
    ):
        for token in tokens(value):
            weighted_query[token] = max(weighted_query.get(token, 0.0), weight)
    document_frequency = Counter(
        token for row in rows for token in set(tokens(row))
    )
    scored: list[dict[str, Any]] = []
    for index, sentence in enumerate(rows):
        sentence_tokens = set(tokens(sentence))
        lexical_raw = sum(
            weight * (1.0 + math.log((len(rows) + 1) / (document_frequency[token] + 1)))
            for token, weight in weighted_query.items()
            if token in sentence_tokens
        )
        lexical = lexical_raw / (1.0 + 0.12 * max(len(sentence_tokens) - 8, 0))
        disease = contains_alias(sentence, terms["disease"])
        region = contains_alias(sentence, terms["region"])
        feature = contains_alias(sentence, terms["feature"])
        result = bool(RESULT_TERMS.search(sentence))
        method = bool(METHOD_TERMS.search(sentence))
        cohort = bool(COHORT_TERMS.search(sentence) and re.search(r"\b\d[\d,]*\b", sentence))
        stats = bool(P_VALUE_PATTERN.search(sentence) or re.search(r"\br\s*=", sentence, re.I))
        background = bool(BACKGROUND_TERMS.search(sentence))
        section_bonus = 0.0
        if re.match(r"(?:STUDY )?RESULTS?:", sentence, re.I):
            section_bonus = 8.0
        elif re.match(r"CONCLUSIONS?:", sentence, re.I):
            section_bonus = 3.0
        elif re.match(r"(?:STUDY )?(?:DESIGN|METHODS?):", sentence, re.I):
            section_bonus = 0.5
        elif re.match(r"BACKGROUND", sentence, re.I):
            section_bonus = -6.0
        score = lexical + section_bonus
        score += 3.0 if disease else 0.0
        score += 6.0 if region else 0.0
        score += 6.0 if feature else 0.0
        score += 9.0 if result else 0.0
        score += 1.0 if method else 0.0
        score += 0.5 if cohort else 0.0
        score += 2.5 if stats else 0.0
        score -= 12.0 if background else 0.0
        scored.append(
            {
                "sentence": sentence,
                "index": index,
                "score": score,
                "lexical": lexical,
                "coverage": {
                    "disease": disease,
                    "region": region,
                    "feature": feature,
                    "result": result,
                    "method": method,
                    "cohort": cohort,
                    "stats": stats,
                    "background": background,
                },
            }
        )
    return scored


def select_evidence_sentences(
    hypothesis: dict[str, Any], paper: dict[str, Any]
) -> list[str]:
    scored = sentence_scores(hypothesis, paper)
    if not scored:
        return []
    ranked = sorted(
        scored,
        key=lambda row: (
            row["score"],
            row["coverage"]["result"],
            row["coverage"]["region"],
            row["coverage"]["feature"],
            -row["index"],
        ),
        reverse=True,
    )
    finding_rows = [row for row in ranked if row["coverage"]["result"] and not row["coverage"]["background"]]
    selected: list[dict[str, Any]] = [finding_rows[0] if finding_rows else ranked[0]]
    covered = {key for key, value in selected[0]["coverage"].items() if value}
    top_score = max(selected[0]["score"], 1.0)

    for row in ranked[1:]:
        if len(selected) >= 4:
            break
        if any(sentence_similarity(row["sentence"], item["sentence"]) >= 0.72 for item in selected):
            continue
        if row["coverage"]["background"]:
            continue
        adds = {key for key, value in row["coverage"].items() if value} - covered
        especially_useful = bool(adds & {"region", "feature", "stats"})
        lexical_reference = max(selected[0]["lexical"], 1.0)
        novel_support = (
            row["coverage"]["result"]
            and (row["coverage"]["region"] or row["coverage"]["feature"] or row["coverage"]["stats"])
            and row["lexical"] >= lexical_reference * 0.84
            and row["score"] >= top_score * 0.48
        )
        context_extension = especially_useful and row["score"] >= top_score * 0.38
        if novel_support or context_extension:
            selected.append(row)
            covered.update(key for key, value in row["coverage"].items() if value)

    # Add only context that explains the measurement or a statistic attached to
    # the finding. Cohort size is displayed separately in the credibility row.
    for preferred in ("method", "stats"):
        if len(selected) >= 4 or preferred in covered:
            continue
        candidate = next(
            (
                row for row in ranked
                if row["coverage"][preferred]
                and (
                    (preferred == "method" and (row["coverage"]["feature"] or "feature" not in covered))
                    or (preferred == "stats" and row["coverage"]["result"])
                )
                and row not in selected
                and all(sentence_similarity(row["sentence"], item["sentence"]) < 0.72 for item in selected)
            ),
            None,
        )
        if candidate is not None and not candidate["coverage"]["background"] and candidate["score"] >= top_score * 0.28:
            selected.append(candidate)
            covered.update(key for key, value in candidate["coverage"].items() if value)

    return [row["sentence"] for row in sorted(selected, key=lambda row: row["index"])][:4]


def load_gold(path: Path) -> dict[tuple[str, str], list[str]]:
    payload = load_json(path)
    return {
        (compact(item.get("candidate_id")), compact(item.get("paper_id")).casefold()): [
            compact(sentence) for sentence in item.get("sentences", []) if compact(sentence)
        ]
        for item in payload.get("associations", [])
    }


def load_metrics(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    index: dict[str, dict[str, Any]] = {}
    for item in payload.get("journals", []):
        for alias in [item.get("canonical_name"), *item.get("aliases", [])]:
            if compact(alias):
                index[normalized(alias)] = item
    return index, payload


def publication_override(paper: dict[str, Any]) -> dict[str, Any]:
    identifier = paper_id(paper)
    if identifier == "oa:w7112672006":
        return {
            "journal": "Psychiatry Research: Neuroimaging",
            "year": 2026,
            "publication_status": "repository_copy_of_peer_reviewed",
            "publication_of_record_doi": "10.1016/j.pscychresns.2025.112082",
            "direct_url": "https://doi.org/10.1016/j.pscychresns.2025.112082",
        }
    if identifier == "medrxiv:10.64898/2026.01.10.26343837":
        return {
            "journal": "Brain Stimulation",
            "year": 2026,
            "publication_status": "preprint_with_published_version",
            "publication_of_record_doi": "10.1016/j.brs.2026.103084",
            "direct_url": "https://doi.org/10.1016/j.brs.2026.103084",
        }
    raw_journal = normalized(paper.get("journal"))
    if "biorxiv" in raw_journal or "medrxiv" in raw_journal:
        return {"publication_status": "preprint"}
    if "advances in experimental medicine" in raw_journal:
        return {"publication_status": "peer_reviewed_book_chapter"}
    return {"publication_status": "peer_reviewed"}


def cohort_metadata(abstract: str) -> dict[str, Any]:
    candidates = [
        sentence
        for sentence in split_sentences(abstract)
        if COHORT_TERMS.search(sentence) and re.search(r"\b\d[\d,]*\b", sentence)
    ]
    if not candidates:
        return {
            "status": "not_reported_in_abstract",
            "n_total": None,
            "display": None,
            "source_sentence": None,
        }

    def cohort_score(sentence: str) -> tuple[int, int]:
        score = 0
        if re.search(r"\b(enrolled|recruited|included|obtained from|acquired from|sample of)\b", sentence, re.I):
            score += 5
        if re.search(r"\b\d[\d,]*\s+(?:participants?|patients?|subjects?|individuals?)\b", sentence, re.I):
            score += 5
        if re.search(r"\bcontrols?\b", sentence, re.I):
            score += 2
        if re.search(r"\b(age|year|week|month|frequency bands?)\b", sentence, re.I):
            score -= 1
        return score, -len(sentence)

    source = max(candidates, key=cohort_score)
    explicit_patterns = (
        r"\b(?:from|of|included|enrolled|recruited)\s+(\d[\d,]*)\s+(?:participants?|patients?|subjects?|individuals?)\b",
        r"\b(?:sample|cohort)\s+of\s+(\d[\d,]*)\b",
        r"\b(?:total\s+of|N\s*=)\s*(\d[\d,]*)\b",
        r"\b(\d[\d,]*)\s+(?:participants?|patients?|subjects?|individuals?)\s+(?:were|was)\s+(?:enrolled|recruited|included)",
    )
    n_total: int | None = None
    for pattern in explicit_patterns:
        if match := re.search(pattern, source, re.I):
            n_total = int(match.group(1).replace(",", ""))
            break
    if n_total is None:
        group_pattern = re.compile(
            r"\b(\d[\d,]*)\s+(?:(?:with|healthy|typically developing|schizophrenia|"
            r"bipolar disorder|autistic|control|clinical|SCZ|BD|HC|TDC|SSD)\s+){0,3}"
            r"(?:participants?|patients?|controls?|subjects?|individuals?|TDCs?|HCs?|SSDs?)\b",
            re.I,
        )
        values = [int(match.group(1).replace(",", "")) for match in group_pattern.finditer(source)]
        if len(values) >= 2:
            n_total = sum(values)
        elif len(values) == 1:
            n_total = values[0]
    if n_total is not None and not 2 <= n_total <= 10_000_000:
        n_total = None
    return {
        "status": "reported_in_abstract",
        "n_total": n_total,
        "display_en": f"N = {n_total:,}" if n_total is not None else "Reported; total N not derivable",
        "display_zh": f"N = {n_total:,}" if n_total is not None else "摘要报告了分组人数，但无法可靠合计总样本量",
        "source_sentence": source,
    }


def p_value_metadata(abstract: str, evidence: list[str]) -> dict[str, Any]:
    evidence_keys = {normalized(sentence) for sentence in evidence}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in split_sentences(abstract):
        values = [compact(match.group(0)) for match in P_VALUE_PATTERN.finditer(sentence)]
        if not values:
            continue
        unique_values = list(dict.fromkeys(values))
        key = ("; ".join(unique_values).casefold(), normalized(sentence))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "values": unique_values,
                "context": sentence,
                "relevance": "selected_evidence" if normalized(sentence) in evidence_keys else "abstract_only",
            }
        )
    records.sort(key=lambda item: item["relevance"] != "selected_evidence")
    selected_records = [record for record in records if record["relevance"] == "selected_evidence"]
    flattened = list(dict.fromkeys(value for record in selected_records for value in record["values"]))
    if selected_records:
        status = "reported_for_selected_evidence"
    elif records:
        status = "reported_elsewhere_in_abstract"
    else:
        status = "not_reported_in_abstract"
    return {
        "status": status,
        "display": "; ".join(flattened[:6]) if flattened else None,
        "records": records,
    }


def credibility_metadata(
    paper: dict[str, Any], metrics_index: dict[str, dict[str, Any]],
    credibility_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    override = publication_override(paper)
    journal_raw = compact(override.get("journal") or paper.get("journal"))
    metric_row = metrics_index.get(normalized(journal_raw))
    canonical_journal = compact(metric_row.get("canonical_name")) if metric_row else journal_raw
    impact_factor = dict(metric_row.get("impact_factor", {})) if metric_row else {
        "value": None,
        "metric_year": None,
        "status": "unverified",
        "source_url": None,
    }
    cohort = dict(credibility_index.get(paper_id(paper), {}).get("cohort", {}))
    if not cohort:
        cohort = cohort_metadata(compact(paper.get("abstract")))
    cohort["source"] = "manual_abstract_review" if paper_id(paper) in credibility_index else "abstract_parser"
    return {
        "publication_year": override.get("year") or paper.get("year"),
        "journal": canonical_journal,
        "publication_status": override.get("publication_status", "peer_reviewed"),
        "publication_of_record_doi": override.get("publication_of_record_doi"),
        "journal_impact_factor": impact_factor,
        "cohort": cohort,
    }


def evidence_fit_score(strength: str) -> tuple[int, str]:
    """Map a hypothesis-specific manual evidence conclusion onto a 1-8 fit score.

    This component measures how directly the paper supports the exact hypothesis.
    It intentionally excludes journal prestige and sample size, which are handled
    separately by ``study_credibility_score``. Direct counterevidence is retained
    as such instead of being misrepresented as weak positive support.
    """

    text = normalized(strength)
    if "duplicate record" in text or "not independent evidence" in text:
        return 1, "duplicate"
    if any(marker in text for marker in ("counterevidence", "counter evidence", "opposing direction", "opposite direction")):
        return 1, "counterevidence"
    if "strong direct" in text:
        return 8, "supportive"
    if "regional level direct" in text:
        return 7, "supportive"
    if "relatively strong" in text or "moderately strong" in text:
        return 6, "supportive"
    if "strong" in text and any(
        marker in text
        for marker in (
            "limited metric", "weak metric", "limited direction", "weak hypothesis",
            "limited hypothesis", "regional functional relevance", "unresolved direction",
            "metric mismatch", "not direct", "does not directly", "cannot establish",
        )
    ):
        return 5, "supportive"
    if "strong" in text:
        return 7, "supportive"
    if "low to moderate" in text or "weak to moderate" in text:
        return 3, "supportive"
    if "moderate" in text and any(
        marker in text
        for marker in (
            "directionally aligned", "same region", "same network", "same prefrontal",
            "same temporal", "same insular", "ipsilateral",
        )
    ):
        return 6, "supportive"
    if "moderate" in text and any(
        marker in text
        for marker in (
            "indirect", "adjacent", "neighbor", "system", "network", "contralateral",
            "social", "cingulate", "prefrontal", "insular", "temporal",
        )
    ):
        return 4, "supportive"
    if "moderate" in text:
        return 5, "supportive"
    if "weak" in text or "low" in text:
        return 2, "supportive"
    return 3, "supportive"


def study_credibility_score(
    paper: dict[str, Any], credibility: dict[str, Any], polarity: str,
) -> tuple[int, str, str]:
    if polarity == "duplicate":
        return (
            0,
            "This record duplicates another reference and therefore adds no independent study-level weight.",
            "该记录与另一篇参考记录重复，因此不增加独立的研究层面权重。",
        )

    status = compact(credibility.get("publication_status"))
    cohort = dict(credibility.get("cohort", {}))
    p_values = dict(credibility.get("p_values", {}))
    n_total = cohort.get("n_total")
    published = status in {
        "peer_reviewed", "peer_reviewed_book_chapter",
        "preprint_with_published_version", "repository_copy_of_peer_reviewed",
    }
    abstract_text = normalized(paper.get("abstract"))
    title_text = normalized(paper.get("title"))
    synthesis = "meta analysis" in abstract_text or "systematic review" in abstract_text or "meta analysis" in title_text
    relevant_p = p_values.get("status") == "reported_for_selected_evidence"
    large_cohort = isinstance(n_total, int) and n_total >= 100

    if synthesis:
        score = 2
        basis_en = "This is a systematic review or meta-analysis, so it receives the full study-credibility component."
        basis_zh = "该文属于系统综述或荟萃分析，因此研究可信度部分记满分。"
    elif published and (large_cohort or relevant_p):
        score = 2
        details_en: list[str] = []
        details_zh: list[str] = []
        if large_cohort:
            details_en.append(f"the abstract-supported cohort is N = {n_total:,}")
            details_zh.append(f"摘要可核实样本量为 N = {n_total:,}")
        if relevant_p:
            details_en.append("a p value is reported for the selected evidence")
            details_zh.append("最相关证据报告了 p 值")
        basis_en = "The paper is published/peer reviewed, and " + " and ".join(details_en) + "."
        basis_zh = "该论文已经发表或经过同行评议，并且" + "，且".join(details_zh) + "。"
    elif published:
        score = 1
        basis_en = "The paper is published/peer reviewed, but the abstract does not provide a large verifiable cohort or a p value tied to the selected evidence."
        basis_zh = "该论文已经发表或经过同行评议，但摘要未提供可核实的大样本量，或未给出与所选证据直接对应的 p 值。"
    else:
        score = 0
        basis_en = "The available record is a preprint without a verified published version, so no study-credibility point is added."
        basis_zh = "现有记录是尚无可核实正式发表版本的预印本，因此不增加研究可信度分。"
    return score, basis_en, basis_zh


def support_band(score: int, polarity: str) -> tuple[str, str]:
    if polarity == "duplicate":
        return "Duplicate; not independent evidence", "重复记录，不属于独立证据"
    if polarity == "counterevidence":
        return "Counterevidence", "反向证据"
    if score <= 2:
        return "Little or no direct support", "几乎没有直接支持"
    if score <= 4:
        return "Weak indirect support", "较弱的间接支持"
    if score <= 6:
        return "Moderate support", "中等支持"
    if score <= 8:
        return "Strong but incomplete support", "较强但不完整的支持"
    return "Direct strong support", "直接且较强的支持"


def support_score_metadata(
    paper: dict[str, Any], assessment: dict[str, str], credibility: dict[str, Any],
) -> dict[str, Any]:
    fit_score, polarity = evidence_fit_score(assessment.get("evidence_strength_en", ""))
    credibility_score, credibility_basis_en, credibility_basis_zh = study_credibility_score(
        paper, credibility, polarity
    )
    score = max(1, min(10, fit_score + credibility_score))
    if polarity == "counterevidence":
        score = min(score, 2)
    elif polarity == "duplicate":
        score = 1
    elif fit_score < 8:
        # Reserve 9-10 for evidence that directly matches the exact hypothesis.
        score = min(score, 8)
    label_en, label_zh = support_band(score, polarity)
    strength_en = compact(assessment.get("evidence_strength_en"))
    strength_zh = compact(assessment.get("evidence_strength_zh"))
    explanation_en = (
        f"{score}/10 — {label_en}. Evidence fit: {fit_score}/8; study credibility: "
        f"{credibility_score}/2. {strength_en} {credibility_basis_en} "
        "This is an evidence-support score for this paper–hypothesis pair, not a replication probability."
    )
    explanation_zh = (
        f"{score}/10——{label_zh}。证据匹配：{fit_score}/8；研究可信度："
        f"{credibility_score}/2。{strength_zh}{credibility_basis_zh}"
        "该数值是这篇论文对当前假设的证据支持度，不是复现概率。"
    )
    return {
        "schema_version": "hypothesis-paper-support-score-v1",
        "value": score,
        "max_score": 10,
        "label_en": label_en,
        "label_zh": label_zh,
        "polarity": polarity,
        "evidence_fit": {
            "score": fit_score,
            "max_score": 8,
            "basis_en": strength_en,
            "basis_zh": strength_zh,
        },
        "study_credibility": {
            "score": credibility_score,
            "max_score": 2,
            "basis_en": credibility_basis_en,
            "basis_zh": credibility_basis_zh,
        },
        "explanation_en": explanation_en,
        "explanation_zh": explanation_zh,
        "is_replication_probability": False,
        "scored_from": [
            "hypothesis_specific_manual_relevance_assessment",
            "abstract_verified_evidence_sentences",
            "verified_publication_and_cohort_metadata",
        ],
    }


def validate_gold_sentence(sentence: str, abstract: str, key: tuple[str, str]) -> None:
    if normalized(sentence) not in normalized(abstract):
        raise RuntimeError(f"Gold evidence sentence is absent from abstract: {key}: {sentence}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--credibility", type=Path, default=DEFAULT_CREDIBILITY)
    args = parser.parse_args()

    payload = load_json(args.input)
    metrics_index, metrics_payload = load_metrics(args.metrics)
    gold = load_gold(args.gold)
    credibility_payload = load_json(args.credibility)
    credibility_index = {
        compact(item.get("paper_id")).casefold(): item
        for item in credibility_payload.get("papers", [])
        if compact(item.get("paper_id"))
    }
    counts = Counter()
    sentence_counts = Counter()
    missing_reasons: list[tuple[str, str]] = []

    for hypothesis in payload.get("hypotheses", []):
        candidate_id = compact(hypothesis.get("id"))
        for paper in hypothesis.get("literature", []):
            counts["associations"] += 1
            key = (candidate_id, paper_id(paper))
            abstract = compact(paper.get("abstract"))
            if not abstract:
                raise RuntimeError(f"Missing abstract for {key}")
            selected = gold.get(key)
            selection_status = "expert_gold" if selected else "reason_aligned_v3"
            if selected:
                for sentence in selected:
                    validate_gold_sentence(sentence, abstract, key)
                counts["expert_gold"] += 1
            else:
                selected = select_evidence_sentences(hypothesis, paper)
            if not 1 <= len(selected) <= 4:
                raise RuntimeError(f"Expected 1-4 evidence sentences for {key}; got {len(selected)}")

            reason_en = reason_sections(compact(paper.get("relevance_reason_en")))
            reason_zh = reason_sections_zh(compact(paper.get("relevance_reason_zh")))
            if not all(reason_en.values()) or not all(reason_zh.values()):
                missing_reasons.append(key)

            credibility = credibility_metadata(paper, metrics_index, credibility_index)
            credibility["p_values"] = p_value_metadata(abstract, selected)
            if credibility["journal_impact_factor"].get("value") is not None:
                counts["impact_factor_available"] += 1
            if credibility["cohort"].get("n_total") is not None:
                counts["cohort_total_available"] += 1
            if credibility["p_values"].get("status") == "reported_for_selected_evidence":
                counts["p_value_available"] += 1

            override = publication_override(paper)
            if override.get("direct_url"):
                paper["direct_url"] = override["direct_url"]
            paper["evidence_sentences"] = selected
            paper["excerpts"] = selected
            paper["excerpt"] = " ".join(selected)
            paper["evidence_selection"] = {
                "schema_version": "hypothesis-paper-evidence-v2",
                "status": selection_status,
                "source": "abstract",
                "sentence_count": len(selected),
            }
            paper["relevance_assessment"] = {
                "studied_en": reason_en["studied"],
                "specific_support_en": reason_en["support"],
                "evidence_strength_en": reason_en["strength"],
                "studied_zh": reason_zh["studied"],
                "specific_support_zh": reason_zh["support"],
                "evidence_strength_zh": reason_zh["strength"],
            }
            paper["credibility"] = credibility
            paper["support_score"] = support_score_metadata(
                paper, paper["relevance_assessment"], credibility
            )
            counts[f"support_score_{paper['support_score']['value']}"] += 1
            counts[f"support_polarity_{paper['support_score']['polarity']}"] += 1
            sentence_counts[len(selected)] += 1

    if missing_reasons:
        raise RuntimeError(f"{len(missing_reasons)} associations lack structured manual reasons")

    payload["literature_evidence_curation"] = {
        "schema_version": "case1-literature-evidence-v2",
        "curated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "hypothesis-specific manual reason aligned abstract selection",
        "evidence_source": "locally stored abstracts",
        "evidence_sentence_min": 1,
        "evidence_sentence_max": 4,
        "associations": counts["associations"],
        "expert_gold_associations": counts["expert_gold"],
        "sentence_count_distribution": {str(key): sentence_counts[key] for key in sorted(sentence_counts)},
        "impact_factor_available": counts["impact_factor_available"],
        "cohort_total_available": counts["cohort_total_available"],
        "p_value_available": counts["p_value_available"],
        "journal_metrics_registry": args.metrics.name,
        "paper_credibility_registry": args.credibility.name,
        "journal_metrics_retrieved_at": metrics_payload.get("retrieved_at"),
        "metric_note_en": metrics_payload.get("metric_note_en"),
        "metric_note_zh": metrics_payload.get("metric_note_zh"),
    }
    payload["literature_support_scoring"] = {
        "schema_version": "case1-literature-support-scoring-v1",
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "associations_scored": counts["associations"],
        "scale_min": 1,
        "scale_max": 10,
        "components": {
            "evidence_fit": {
                "max_score": 8,
                "basis": "hypothesis-specific manually curated disease/region/measure/direction assessment",
            },
            "study_credibility": {
                "max_score": 2,
                "basis": "publication status, evidence synthesis design, abstract-supported cohort size, and p value attached to selected evidence",
            },
        },
        "counterevidence_cap": 2,
        "duplicate_record_score": 1,
        "score_distribution": {
            str(score): counts[f"support_score_{score}"] for score in range(1, 11)
        },
        "polarity_distribution": {
            polarity: counts[f"support_polarity_{polarity}"]
            for polarity in ("supportive", "counterevidence", "duplicate")
        },
        "disclaimer_en": "Scores quantify how strongly one paper supports one exact hypothesis. They are not replication probabilities, posterior probabilities, or substitutes for expert judgment.",
        "disclaimer_zh": "该分数只量化一篇论文对一个具体假设的支持程度，不是复现概率、后验概率，也不能替代专家判断。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["literature_evidence_curation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
