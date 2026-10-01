"""Look up one research topic in the current accepted graph, with its evidence.

Read-only aggregation over the accepted shared catalog and existing original
claim index: tokenise a topic, rank lexical matches, then use the web explorer's
batch query to expand selected records into per-paper observations.
It adds no claim, no literature and no merged
proposition; a lexical hit is a retrieval candidate, never scientific support or
a novelty decision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

MAX_TERMS = 12
MAX_CLAIMS = 50
DEFAULT_LIMIT = 20
DEFAULT_EVIDENCE_CLAIMS = 10
MAX_RENDERED_OBSERVATIONS = 3


class TopicQueryError(ValueError):
    """Invalid user input, distinct from an unavailable evidence revision."""

CJK = "\u4e00-\u9fff"
CJK_SEPARATORS = "\u4e0e\u548c\u7684\u4e2d\u5728\u53ca\u6216\u5bf9\u662f\u6709\u4e3a\u628a\u88ab\u5c06\u4e86\u7740\u4e4b\u5176"
STOPWORDS = {
    "the", "and", "of", "in", "for", "with", "on", "to", "an", "is", "are", "be", "by",
    "from", "that", "this", "as", "at", "or", "vs", "versus", "between", "among", "study",
    "studies", "effect", "effects", "role", "using", "into", "than", "does", "how",
}

# Small, explicit query aliases only. They never change graph identities or
# equate scientific propositions. Unknown vocabulary remains a literal query.
ALIASES = (
    ("hippocampal", "hippocampus", "hippocampi", "海马体", "海马"),
    ("atrophy", "atrophic", "萎缩"),
    ("mci", "mild cognitive impairment", "轻度认知障碍"),
    ("alzheimer", "alzheimer's", "alzheimers", "ad", "阿尔茨海默病", "阿尔茨海默", "阿兹海默症"),
    ("cerebellum", "cerebellar", "小脑"),
    ("adhd", "attention deficit hyperactivity disorder", "attention-deficit/hyperactivity disorder", "多动症", "注意缺陷多动障碍"),
    ("memory", "记忆"),
    ("cognition", "cognitive", "认知"),
    ("volume", "volumes", "volumetric", "体积"),
    ("connectivity", "连接"),
    ("mri", "magnetic resonance imaging", "磁共振"),
    ("conversion", "convert", "converters", "progression", "转归", "转化", "进展"),
    ("schizophrenia", "schizophrenic", "精神分裂症"),
    ("epilepsy", "epileptic", "癫痫"),
    ("parkinson", "parkinson's", "帕金森病", "帕金森"),
)


def _pattern(term: str) -> str:
    # Prevent short abbreviations (AD, MCI) matching unrelated word interiors.
    escaped = re.escape(term)
    return escaped if re.search(f"[{CJK}]", term) else r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"


def query_groups(topic: str, extra_terms: Iterable[str] = ()) -> list[tuple[str, ...]]:
    groups = []
    aliases = {alias: group for group in ALIASES for alias in group}
    alias_pattern = "|".join(_pattern(a) for a in sorted(aliases, key=len, reverse=True))
    for text in [topic, *extra_terms]:
        text = text.casefold().strip()
        end = 0
        for hit in re.finditer(alias_pattern, text):
            groups.extend((term,) for term in _literal_terms(text[end:hit.start()]))
            groups.append(aliases[hit.group()])
            end = hit.end()
        groups.extend((term,) for term in _literal_terms(text[end:]))
    return list(dict.fromkeys(groups))[:MAX_TERMS]


def _literal_terms(topic: str, extra_terms: Iterable[str] = ()) -> list[str]:
    """Casefolded search keys: Latin words plus short CJK n-grams, deduplicated."""
    text = str(topic or "").casefold()
    terms: list[str] = []

    def add(value: str) -> None:
        if value and value not in terms:
            terms.append(value)

    # Keep source order so a mixed "Cerebellum \u5c0f\u8111 timing" topic reads naturally.
    for match in re.finditer(f"[a-z0-9][a-z0-9\\-\\+\\.]*|[{CJK}]+", text):
        token = match.group(0)
        if token[0] in "-+.":
            continue
        if re.fullmatch(f"[{CJK}]+", token) is None:
            token = token.strip("-+.")
            if len(token) >= 2 and token not in STOPWORDS:
                add(token)
            continue
        for part in re.split(f"[{CJK_SEPARATORS}]+", token):
            if len(part) < 2:
                continue
            # No CJK tokenizer is installed and none is added here, so a run is cut
            # into near-equal 2-3 character pieces (海马体萎缩 -> 海马体, 萎缩). The
            # pieces are substring keys, not linguistic words: pass --term for real
            # synonyms when this segmentation misses a term of interest.
            pieces = max(1, math.ceil(len(part) / 3))
            size, remainder = divmod(len(part), pieces)
            start = 0
            for index in range(pieces):
                length = size + (1 if index < remainder else 0)
                add(part[start:start + length])
                start += length
    for extra in extra_terms:
        add(str(extra or "").casefold().strip())
    return terms[:MAX_TERMS]


def topic_terms(topic: str, extra_terms: Iterable[str] = ()) -> list[str]:
    return [group[0] for group in query_groups(topic, extra_terms)]


def _matchers(groups):
    return [(group[0], re.compile("|".join(_pattern(term) for term in group), re.IGNORECASE))
            for group in groups]


def _studies(document: dict[str, Any]) -> list[dict[str, Any]]:
    studies = []
    for paper in document.get("papers") or []:
        bibliography = paper.get("bibliography") or {}
        observations = []
        for observation in paper.get("observations") or []:
            review = observation.get("source_review") or {}
            observations.append({
                "claim_id": observation.get("claim_id"),
                "observation_role": observation.get("observation_role"),
                "proposition_support": observation.get("proposition_support"),
                "negated": observation.get("negated"),
                "population": observation.get("population"),
                "conditions": observation.get("conditions"),
                "evidence": observation.get("evidence"),
                "source_anchor": review.get("source_anchor"),
                "scope_note": review.get("scope_note"),
                "source_text": observation.get("raw_text"),
                "source_reviewed": bool(review),
                "source_review": deepcopy(observation.get("source_review")),
                "metadata": deepcopy(observation.get("original_claim") or observation.get("source_derived_claim") or {}),
                "observation_origin": observation.get("observation_origin", "original_graph"),
                "text_kind": "reviewed_source_anchor" if review.get("source_anchor") else "graph_extraction_text",
            })
        studies.append({
            "paper_key": paper.get("paper_key"),
            "work_key": paper.get("work_key") or paper.get("paper_key"),
            "pmid": bibliography.get("pmid"),
            "doi": bibliography.get("doi"),
            "title": bibliography.get("title"),
            "year": bibliography.get("year"),
            "bibliography": deepcopy(bibliography),
            "source_identity": deepcopy(paper.get("source_identity")),
            "publication_versions": deepcopy(paper.get("publication_versions") or []),
            "publication_review": deepcopy(paper.get("publication_review")),
            "verified": paper.get("verified"),
            "counted_toward_default": paper.get("counted_toward_default"),
            "supports_reviewed_proposition": paper.get("supports_reviewed_proposition"),
            "own_result": paper.get("own_result"),
            "publication_status": (paper.get("publication_review") or {}).get("status"),
            "observations": observations,
        })
    studies.sort(key=lambda study: (not study["verified"], not study["supports_reviewed_proposition"],
                                    str(study["paper_key"])))
    return studies


def _expand(layer, ids: list[str], revision) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    queries = [{"claim_id" if value.startswith("CLM:") else "relation_id": value} for value in ids]
    batch = layer.query_batch(queries=queries)
    documents = batch["results"]
    if (any(batch.get(key) != value for key, value in revision.items()) or
            any(any(doc.get(key) != value for key, value in revision.items()) for doc in documents) or
            len(documents) != len(ids) or
            any((doc.get("requested_claim_id") != value or value not in doc.get("original_claim_ids", []))
                if value.startswith("CLM:") else doc["shared_claim_id"] != value
                for value, doc in zip(ids, documents))):
        from core.web.claim_evidence import EvidenceUnavailable
        raise EvidenceUnavailable("Evidence batch differs from the topic search revision or IDs")
    return dict(zip(ids, documents))


def topic_evidence(
    layer,
    topic: str,
    *,
    limit: int = DEFAULT_LIMIT,
    evidence_claims: int = DEFAULT_EVIDENCE_CLAIMS,
    extra_terms: Iterable[str] = (),
    minimum_papers: int = 0,
    minimum_coverage: float = 0.5,
) -> dict[str, Any]:
    """Rank matched claims for one topic and attach the evidence of the top ones."""
    started = time.perf_counter()
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 500:
        raise TopicQueryError("Provide a topic of 1 to 500 characters")
    extra_terms = list(extra_terms)
    if len(extra_terms) > MAX_TERMS or any(not isinstance(t, str) or not t.strip() or len(t) > 100 for t in extra_terms):
        raise TopicQueryError("Provide at most 12 nonempty extra terms of at most 100 characters")
    if type(limit) is not int or not 1 <= limit <= MAX_CLAIMS:
        raise TopicQueryError(f"limit must be an integer in [1,{MAX_CLAIMS}]")
    if type(evidence_claims) is not int or not 0 <= evidence_claims <= MAX_CLAIMS:
        raise TopicQueryError(f"evidence_claims must be an integer in [0,{MAX_CLAIMS}]")
    if type(minimum_papers) is not int or not 0 <= minimum_papers <= 10000:
        raise TopicQueryError("minimum_papers must be a nonnegative integer")
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise TopicQueryError("minimum_coverage must be in [0,1]")

    groups = query_groups(topic, extra_terms)
    terms = [group[0] for group in groups]
    if not terms:
        raise TopicQueryError("Provide a topic with at least one searchable term.")
    with layer.read_snapshot() as revision:
        result = _retrieve(layer, topic, groups, revision, limit, evidence_claims, minimum_papers, float(minimum_coverage))
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _retrieve(layer, topic, groups, revision, limit, evidence_claims, minimum_papers, minimum_coverage):
    terms = [group[0] for group in groups]
    matchers = _matchers(groups)
    minimum_matches = max(1, math.ceil(len(terms) * minimum_coverage))

    index = layer.claim_index()
    matched = []
    for summary, haystack in index:
        if summary["reviewed_supporting_article_count"] < minimum_papers:
            continue
        hits = [term for term, pattern in matchers if pattern.search(haystack)]
        if len(hits) < minimum_matches:
            continue
        matched.append((len(hits) / len(terms), hits, summary))
    shared_matched = len(matched)
    originals = layer.original_claim_candidates(matchers, minimum_matches=minimum_matches, limit=limit)
    if minimum_papers <= 1:
        for summary in originals["candidates"]:
            hits = summary["matched_terms"]
            matched.append((len(hits) / len(terms), hits, summary))
    matched.sort(key=lambda item: (-item[0], -len(item[1]), -(item[2]["reviewed_supporting_article_count"] or 0),
                                   -(item[2]["article_count"] or 1), item[2]["claim"]["subject_name"].casefold(),
                                   item[2]["shared_claim_id"]))
    selected = matched[:limit]
    documents = _expand(layer, [summary.get("query_id", summary["shared_claim_id"])
                               for _, _, summary in selected[:evidence_claims]], revision)

    claims = []
    for coverage, hits, summary in selected:
        document = documents.get(summary.get("query_id", summary["shared_claim_id"]))
        counts = {name: (document or summary).get(name) for name in ("observation_count", *layer.COUNTS)}
        if minimum_papers and (counts["reviewed_supporting_article_count"] or 0) < minimum_papers:
            continue
        claims.append({
            "shared_claim_id": summary["shared_claim_id"],
            "query_id": summary.get("query_id", summary["shared_claim_id"]),
            "retrieval_origin": summary.get("retrieval_origin", "accepted_shared_catalog"),
            "claim": (document or summary)["claim"],
            "match": {"matched_terms": hits, "coverage": round(coverage, 4)},
            "counts": counts,
            "evidence_expanded": document is not None,
            "studies": _studies(document) if document else [],
        })

    studies = {}
    for claim in claims:
        for study in claim["studies"]:
            key = study["work_key"]
            if key not in studies:
                studies[key] = dict(deepcopy(study), observations=[], related_claims=[], matched_terms=[])
            row = studies[key]
            row["related_claims"].append({"shared_claim_id": claim["shared_claim_id"], "query_id": claim["query_id"],
                                          "retrieval_origin": claim["retrieval_origin"], "claim": claim["claim"], "match": claim["match"]})
            for observation in study["observations"]:
                bound = dict(observation, shared_claim_id=claim["shared_claim_id"])
                if bound not in row["observations"]:
                    row["observations"].append(bound)
            for flag in ("verified", "counted_toward_default", "supports_reviewed_proposition", "own_result"):
                row[flag] = row[flag] or study[flag]
    for study in studies.values():
        text = json.dumps([study["bibliography"], study["observations"]], ensure_ascii=False)
        study["matched_terms"] = [term for term, pattern in matchers if pattern.search(text)]
        study["missing_terms"] = [term for term in terms if term not in study["matched_terms"]]
        study["topic_coverage"] = round(len(study["matched_terms"]) / len(terms), 4)
    # Sharing a graph proposition does not make every member paper relevant to
    # this topic. Require named diagnoses in each paper's own record, as well as
    # the caller's keyword threshold; keep other members in claim-level context.
    diagnoses = set(terms) & {"mci", "alzheimer", "adhd", "schizophrenia", "epilepsy", "parkinson"}
    topical = [s for s in studies.values() if len(s["matched_terms"]) >= minimum_matches
               and diagnoses <= set(s["matched_terms"])]
    ranked_studies = sorted(topical, key=lambda s: (
        -len(s["matched_terms"]), not s["verified"],
        not any(o["source_anchor"] for o in s["observations"]), str(s["work_key"])))
    notes = [
        "Lexical match over shared-claim concept names, predicates and paper bibliography, "
        "plus original-index endpoints, predicates and source IDs; "
        "not semantic, full-text or literature search.",
        "A matched claim is a retrieval candidate, not scientific support or a novelty judgement.",
        "Study records state whether the observation was source-reviewed; unreviewed records stay unreviewed.",
        "The studies list applies the topic threshold and named diagnoses to each paper's own record; "
        "other papers remain only in claim-level context. Missing terms expose partial matches.",
        "Scope: the accepted shared-claim catalog and its published extensions, plus the existing original-claim index when available. "
        "An empty result is not absence of relevant literature. Counts describe this bounded result only.",
        "Automatic aliases cover a small explicit vocabulary; use extra_terms for other translations or synonyms.",
        f"Bounded: {shared_matched} shared matches and {originals['matched']} additional original-observation matches; "
        f"one representative per original source competes for the same {limit}-claim budget; expanded {len(documents)}.",
        "Unexpanded original candidates have unknown paper/support counts; a positive minimum_papers filter "
        "requires measured supporting counts after expansion and may return fewer results.",
    ]
    if not originals["available"]:
        notes.append("No accepted original-claim index is configured; only the shared catalog was searched.")
    if not matched:
        notes.append("No shared claim matched, and no eligible original candidate was found; supply English synonyms via extra_terms "
                     "before concluding the graph holds no relevant work.")
    return {
        "topic": str(topic),
        "terms": terms,
        "expanded_terms": {group[0]: list(group) for group in groups},
        "minimum_matches": minimum_matches,
        **revision,
        "claim_total": len(index) + originals["eligible"],
        "shared_claim_total": len(index),
        "claim_matched": shared_matched + (originals["matched"] if minimum_papers <= 1 else 0),
        "shared_claim_matched": shared_matched,
        "original_retrieval": {k: v for k, v in originals.items() if k != "candidates"},
        "claim_returned": len(claims),
        "study_returned": len(ranked_studies),
        "context_study_count": len(studies) - len(ranked_studies),
        "studies": ranked_studies,
        "claims": claims,
        "notes": notes,
    }


def render_text(result: dict[str, Any]) -> str:
    """A compact console view; the JSON form keeps every returned field."""
    lines = [f"Topic: {result['topic']}",
             "Terms: " + ", ".join(result["terms"]),
             f"Matched {result['shared_claim_matched']} of {result['shared_claim_total']} shared claims plus "
             f"{result['original_retrieval']['matched']} original observations; "
             f"showing {result['claim_returned']} in {result['elapsed_seconds']}s",
             f"Topic-matched papers: {result['study_returned']}; "
             f"other claim-context papers: {result['context_study_count']}",
             f"Graph revision: {result['graph_revision']}"]
    topical = {study['work_key']: study for study in result['studies']}
    for claim in result["claims"]:
        values = claim["claim"]
        counts = claim["counts"]
        lines.append("")
        lines.append(f"[{claim['shared_claim_id']}] {values['subject_name']} {values['predicate']} "
                     f"{values['object_name']}  (coverage {claim['match']['coverage']:.2f}; "
                     f"hits: {', '.join(claim['match']['matched_terms'])})")
        lines.append(f"  observations {counts['observation_count']}; papers {counts['article_count']}; "
                     f"reviewed supporting {counts['reviewed_supporting_article_count']}; "
                     f"own result {counts['own_result_article_count']}")
        if not claim["evidence_expanded"]:
            lines.append("  evidence: not expanded in this bounded query")
            continue
        for study in claim["studies"]:
            if study['work_key'] not in topical:
                continue
            identifier = study["pmid"] or study["doi"] or study["paper_key"]
            lines.append(f"  - {identifier} ({study['year']}) {study['title']} "
                         f"[verified={study['verified']}; reviewed support={study['supports_reviewed_proposition']}]")
            if topical[study['work_key']]['missing_terms']:
                lines.append("      Partial topic match; missing: " + ", ".join(topical[study['work_key']]['missing_terms']))
            for observation in study["observations"][:MAX_RENDERED_OBSERVATIONS]:
                text = observation["source_anchor"] or observation["source_text"] or ""
                text = " ".join(str(text).split())
                lines.append(f"      {observation['observation_role']}/{observation['proposition_support']}"
                             + (" negated" if observation["negated"] else "")
                             + f": {text[:200]}")
            if len(study["observations"]) > MAX_RENDERED_OBSERVATIONS:
                lines.append(f"      ... {len(study['observations']) - MAX_RENDERED_OBSERVATIONS} more observations")
    lines.append("")
    lines.extend(f"note: {note}" for note in result["notes"])
    return "\n".join(lines)


def default_layer(campaign: Path | None = None):
    """The same read-only loader the graph explorer uses; never a rebuild."""
    from core.web.claim_evidence import configured_campaign
    from core.web.claim_layer_v8 import AcceptedClaimLayer

    path = Path(campaign) if campaign else configured_campaign(Path(__file__).resolve().parents[1])
    if path is None:
        raise ValueError("No accepted graph campaign is configured")
    return AcceptedClaimLayer(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Research topic or question")
    parser.add_argument("--term", action="append", default=[],
                        help="Extra search term, e.g. an English synonym; repeatable")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--evidence-claims", type=int, default=DEFAULT_EVIDENCE_CLAIMS)
    parser.add_argument("--minimum-papers", type=int, default=0)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    parser.add_argument("--campaign", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the full JSON result")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON; never overwrites")
    args = parser.parse_args(argv)
    result = topic_evidence(default_layer(args.campaign), args.topic, limit=args.limit,
                            evidence_claims=args.evidence_claims, extra_terms=args.term,
                            minimum_papers=args.minimum_papers,
                            minimum_coverage=args.minimum_coverage)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render_text(result))


if __name__ == "__main__":
    main()
