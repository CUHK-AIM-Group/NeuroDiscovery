"""P0 step 1: verbatim validator for extracted claims.

Problem
-------
Phase 2 LLM extraction sometimes "injects" subject/object concepts that
are not actually mentioned verbatim in the raw sentence. The audit found
~17,613 (20.7%) of claims whose object name tokens are mostly absent from
the raw_text, and ~19,902 (23.4%) where the subject is absent.

However, many of those "injected" flags are false positives: a paper
abstract might read "DLB is characterized by X, Y, Z" and each feature
becomes its own claim with raw_text being just the sentence that mentions
the feature (which doesn't repeat "DLB"). That's a legitimate pattern,
not LLM hallucination.

Design
------
We split the findings into two tiers:

Tier A — auto-delete (high precision, no false positives):
    * self_claim      : subject_id == object_id  (~1,490 claims)
    * raw_too_short   : raw_text shorter than MIN_RAW_LEN (~56 claims)

Tier B — report only (may include many false positives):
    * injected_subject: no surface form of the subject concept appears
                        in raw_text (often inter-sentence context /
                        paraphrased document-level summary — NOT deleted).
    * injected_object : same as above for object side.
    * vague_hub_mapping: subject or object mapped to a known-vague
                        CognitiveAtlas hub (loss, risk, activation,
                        stress, logic, memory) — downstream useless,
                        user should decide whether to prune.

Outputs
-------
1. <out_dir>/bad_claim_ids.json
       Tier A ids (actually safe to delete).
2. <out_dir>/flagged_claim_ids.json
       Tier B ids with reasons (report only).
3. <out_dir>/extracted_claims.filtered.jsonl
       All claims that passed Tier A (Tier B claims are still here).
4. <out_dir>/extracted_claims.flagged.jsonl
       All claims that failed Tier A or Tier B, each with
       `_flag_reasons`. Useful for spot-checking.

Run apply_claim_filter_to_kg.py to delete Tier A claim nodes from the KG.

Usage
-----
    python -m core.knowledge_graph.src.verbatim_validator \
        --claims core/knowledge_graph/data/full/extracted_claims.jsonl \
        --kg     core/knowledge_graph/data/full/knowledge_graph.json \
        --out-dir core/knowledge_graph/data/full
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from .storage import load_graph

log = logging.getLogger("verbatim_validator")

MATCH_THRESHOLD = 0.5        # >= 50% content tokens of some surface form hit
MIN_TOKEN_LEN = 4            # tokens of length < 4 are stopwords-ish
MIN_RAW_LEN = 40             # raw_text shorter than this -> delete (tier A)

# CognitiveAtlas concept ids that act as over-general "hub" nodes in the
# KG. The audit found these in top-degree hubs with thousands of edges
# each; their claims are usually useless for downstream hypothesis
# reasoning (e.g. "X is_biomarker_of memory" / "Y correlates_with loss").
# These do NOT get auto-deleted — just flagged for review.
VAGUE_HUB_IDS = {
    "COGAT_CONCEPT:trm_5159c80c1dd24",   # loss
    "COGAT_CONCEPT:trm_4a3fd79d0afcf",   # risk
    "COGAT_CONCEPT:trm_4a3fd79d09741",   # activation
    "COGAT_CONCEPT:trm_4a3fd79d0b2a8",   # stress
    "COGAT_CONCEPT:trm_4a3fd79d0a80f",   # logic
    "COGAT_CONCEPT:trm_4a3fd79d0a891",   # memory
}


_STOPWORD_TOKENS = {
    "with", "from", "into", "onto", "without", "have", "been", "being",
    "that", "this", "these", "those", "their", "them", "there", "over",
    "under", "about", "while", "after", "before", "during", "between",
    "among", "such", "than", "also", "both", "only", "same", "other",
    "would", "could", "should", "might", "must", "does", "done", "some",
    "many", "most", "much", "very", "like", "into", "each", "which",
    "when", "where", "what", "within", "show", "shows", "showed",
    "study", "studies", "result", "results", "finding", "findings",
}


def _content_tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", s.lower())
            if len(t) >= MIN_TOKEN_LEN and t not in _STOPWORD_TOKENS]


def _surface_forms(kg_index: dict, concept_id: str, fallback_name: str) -> list[str]:
    """Return [preferred_name] + aliases for a concept (lowercased, deduped)."""
    forms = []
    if concept_id and concept_id in kg_index:
        node = kg_index[concept_id]
        if node.preferred_name:
            forms.append(node.preferred_name)
        if node.aliases:
            forms.extend(node.aliases)
    if fallback_name:
        forms.append(fallback_name)
    # lowercase + dedupe while preserving order
    seen = set()
    out = []
    for f in forms:
        f = f.strip().lower()
        if not f or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _side_matches_raw(raw_lower: str, concept_id: str, fallback_name: str,
                      kg_index: dict) -> bool:
    """True if ANY surface form has >=MATCH_THRESHOLD of its content tokens
    present (as substring) in raw_lower. Short / acronym forms (<=4 chars)
    pass via whole-word substring check.
    """
    forms = _surface_forms(kg_index, concept_id, fallback_name)
    if not forms:
        return True   # nothing to check against, don't flag

    for form in forms:
        # short form / acronym: require whole word
        if len(form) <= 4:
            pattern = r"\b" + re.escape(form) + r"\b"
            if re.search(pattern, raw_lower):
                return True
            continue

        # also try full phrase substring first (cheapest positive signal)
        if form in raw_lower:
            return True

        tokens = _content_tokens(form)
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in raw_lower)
        if hits / len(tokens) >= MATCH_THRESHOLD:
            return True
    return False


def validate(claims_path: Path, kg_path: Path, out_dir: Path) -> dict:
    log.info(f"loading KG {kg_path}")
    kg = load_graph(kg_path)
    kg_index = kg._index
    log.info(f"KG has {len(kg_index)} concepts")

    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / "extracted_claims.filtered.jsonl"
    flagged_path = out_dir / "extracted_claims.flagged.jsonl"
    bad_ids_path = out_dir / "bad_claim_ids.json"
    flagged_ids_path = out_dir / "flagged_claim_ids.json"

    stats = Counter()
    by_reason = Counter()
    tier_a_ids: list[str] = []   # will be deleted
    tier_b: list[dict] = []      # flagged only, kept in KG

    with claims_path.open("r", encoding="utf-8") as fin, \
            filtered_path.open("w", encoding="utf-8") as fout_keep, \
            flagged_path.open("w", encoding="utf-8") as fout_drop:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                continue

            stats["total"] += 1
            cid = c.get("id") or f"line_{line_no}"
            raw = c.get("raw_text") or ""
            raw_lower = raw.lower()
            subj_id = c.get("subject_id", "")
            obj_id = c.get("object_id", "")
            subj_name = c.get("subject_name", "")
            obj_name = c.get("object_name", "")

            # Tier A: auto-delete categories
            tier_a: list[str] = []
            if not raw:
                tier_a.append("no_raw_text")
            elif len(raw) < MIN_RAW_LEN:
                tier_a.append("raw_too_short")
            if subj_id and obj_id and subj_id == obj_id:
                tier_a.append("self_claim")

            # Tier B: report-only categories
            tier_b_reasons: list[str] = []
            if subj_id in VAGUE_HUB_IDS:
                tier_b_reasons.append("vague_hub_subject")
            if obj_id in VAGUE_HUB_IDS:
                tier_b_reasons.append("vague_hub_object")
            if raw_lower:
                if not _side_matches_raw(raw_lower, subj_id, subj_name, kg_index):
                    tier_b_reasons.append("injected_subject")
                if not _side_matches_raw(raw_lower, obj_id, obj_name, kg_index):
                    tier_b_reasons.append("injected_object")

            all_reasons = tier_a + tier_b_reasons
            if tier_a:
                stats["tier_a_deleted"] += 1
                tier_a_ids.append(cid)
            elif tier_b_reasons:
                stats["tier_b_flagged"] += 1
            else:
                stats["passed"] += 1

            for r in all_reasons:
                by_reason[r] += 1

            if tier_a:
                # physically drop
                c["_flag_reasons"] = all_reasons
                c["_tier"] = "A"
                fout_drop.write(json.dumps(c, ensure_ascii=False) + "\n")
            elif tier_b_reasons:
                # keep in filtered output, also write to flagged for review
                c_copy = dict(c)
                c_copy["_flag_reasons"] = tier_b_reasons
                c_copy["_tier"] = "B"
                fout_drop.write(json.dumps(c_copy, ensure_ascii=False) + "\n")
                tier_b.append({"id": cid, "reasons": tier_b_reasons})
                fout_keep.write(json.dumps(c, ensure_ascii=False) + "\n")
            else:
                fout_keep.write(json.dumps(c, ensure_ascii=False) + "\n")

            if line_no % 10000 == 0:
                log.info(f"  scanned {line_no} claims; "
                         f"tier_a={stats['tier_a_deleted']} "
                         f"tier_b={stats['tier_b_flagged']}")

    bad_ids_path.write_text(
        json.dumps({
            "match_threshold": MATCH_THRESHOLD,
            "min_raw_len": MIN_RAW_LEN,
            "stats": dict(stats),
            "by_reason": dict(by_reason),
            "tier_a_ids": tier_a_ids,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flagged_ids_path.write_text(
        json.dumps({"tier_b": tier_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"total={stats['total']} passed={stats['passed']} "
             f"tier_a_deleted={stats['tier_a_deleted']} "
             f"tier_b_flagged={stats['tier_b_flagged']}")
    log.info(f"by_reason: {dict(by_reason)}")
    log.info(f"-> {filtered_path}")
    log.info(f"-> {flagged_path}")
    log.info(f"-> {bad_ids_path}  (tier A, safe to delete)")
    log.info(f"-> {flagged_ids_path}  (tier B, review-only)")
    return {"stats": dict(stats), "by_reason": dict(by_reason)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--kg", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    validate(args.claims, args.kg, args.out_dir)


if __name__ == "__main__":
    main()
