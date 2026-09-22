"""Canonicalize reviewed Case Study 2 claims for production KG replay.

This migration keeps the reviewed claim content intact while aligning legacy
manual identifiers, predicates, and paper scopes with the current full_v2
schema. Existing claim IDs can be supplied to make the output idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from neurooracle.src.schema import CLAIM_PREDICATES


PREDICATE_MAP = {
    "does_not_associate_with": "is_associated_with",
    "not_associated_with": "is_associated_with",
    "does_not_predict": "predicts",
    "does_not_outperform": "predicts",
    "identifies": "is_biomarker_of",
    "indicates": "is_biomarker_of",
    "detects": "is_biomarker_of",
    "visualizes": "is_biomarker_of",
    "reveals": "is_biomarker_of",
    "maps": "is_biomarker_of",
    "characterizes": "is_biomarker_of",
    "captures": "is_biomarker_of",
    "measures": "is_biomarker_of",
    "contributes_to": "causes",
    "induces": "causes",
    "drives": "causes",
    "triggers": "causes",
    "disrupts": "causes",
    "accelerates": "causes",
    "supports": "is_associated_with",
    "recapitulates": "is_associated_with",
    "models": "is_associated_with",
    "harmonizes": "is_associated_with",
    "standardizes": "is_associated_with",
    "routes": "is_associated_with",
    "tracks": "is_associated_with",
    "confounds": "is_associated_with",
    "is_implicated_in": "is_associated_with",
    "are_associated_with": "is_associated_with",
    "are_spatially_associated_with": "is_associated_with",
    "is_contested_as": "is_associated_with",
    "is_concordant_with": "is_associated_with",
    "is_equivalent_to": "is_associated_with",
    "has_amyloid_dependent_and_independent_effects_on": "modulates",
    "modifies": "modulates",
    "promotes": "modulates",
    "restores": "modulates",
    "enables": "predicts",
    "improves": "treats",
    "protects_against": "reduces",
    "suppresses": "inhibits",
    "rules_in_or_out": "distinguishes",
    "is_increased_in": "is_biomarker_of",
    "changes_before": "predicts",
    "associated_with": "is_associated_with",
    "associates_with": "is_associated_with",
    "associate_with": "is_associated_with",
    "predict": "predicts",
    "are_predictive_of": "predicts",
}


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def _canonical_id(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if not text.upper().startswith("MANUAL") or ":" not in text:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _stable_anchor_id(name: Any) -> str:
    normalized = str(name or "").strip().lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    slug = "".join(char if char.isalnum() else "_" for char in normalized)
    slug = "_".join(part for part in slug.split("_") if part)[:96] or "unnamed"
    return f"CLM_CONCEPT:{slug}_{digest}"


def _existing_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        str(row.get("id") or "").strip()
        for row in _jsonl(path)
        if row.get("id")
    }


def canonicalize_claim(row: dict[str, Any], injection_source: str) -> dict[str, Any]:
    claim = dict(row)
    claim["id"] = _canonical_id(claim.get("id"), "CLM")
    claim["subject_id"] = _canonical_id(claim.get("subject_id"), "CLM_CONCEPT")
    claim["object_id"] = _canonical_id(claim.get("object_id"), "CLM_CONCEPT")
    if not claim["subject_id"]:
        claim["subject_id"] = _stable_anchor_id(claim.get("subject_name"))
    if not claim["object_id"]:
        claim["object_id"] = _stable_anchor_id(claim.get("object_name"))

    original_predicate = str(claim.get("predicate") or "").strip()
    predicate = PREDICATE_MAP.get(original_predicate, original_predicate)
    if predicate not in CLAIM_PREDICATES:
        predicate = "is_associated_with"
    claim["predicate"] = predicate
    if original_predicate.startswith(("does_not_", "not_")):
        claim["negated"] = True

    claim["paper_case_study_ids"] = ["case2_pathway_mediation"]
    claim["claim_case_study_ids"] = ["case2_pathway_mediation"]
    if not isinstance(claim.get("evidence"), dict):
        evidence_text = str(claim.get("evidence") or "").strip()
        claim["evidence"] = {
            "study_type": "manual_curated_abstract",
            "methodology": evidence_text or "manual abstract curation",
            "p_value": None,
            "effect_size": None,
            "effect_metric": "",
            "sample_size": None,
            "replicability": "single_study",
            "direction": "",
        }
    metadata = dict(claim.get("metadata") or {})
    metadata.update(
        {
            "paper_case_study_ids": ["case2_pathway_mediation"],
            "claim_case_study_ids": ["case2_pathway_mediation"],
            "case_study_membership_schema_version": "case_study_membership.v2",
            "original_predicate": original_predicate,
            "predicate_canonicalized": original_predicate != predicate,
            "kg_injection_source": metadata.get("kg_injection_source") or injection_source,
            "kg_injected": True,
        }
    )
    claim["metadata"] = metadata
    return claim


def prepare(
    input_path: Path,
    output_path: Path,
    existing_path: Path | None,
    injection_source: str,
) -> dict[str, Any]:
    existing = _existing_ids(existing_path)
    seen: set[str] = set()
    predicate_changes: Counter[str] = Counter()
    paper_ids: set[str] = set()
    written = skipped_existing = skipped_duplicate = invalid = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for raw in _jsonl(input_path):
            claim = canonicalize_claim(raw, injection_source)
            claim_id = str(claim.get("id") or "")
            if not claim_id or not claim.get("subject_name") or not claim.get("object_name"):
                invalid += 1
                continue
            if claim_id in existing:
                skipped_existing += 1
                continue
            if claim_id in seen:
                skipped_duplicate += 1
                continue
            seen.add(claim_id)
            original = str((claim.get("metadata") or {}).get("original_predicate") or "")
            if original != claim["predicate"]:
                predicate_changes[f"{original}->{claim['predicate']}"] += 1
            paper = claim.get("source_paper") or {}
            paper_id = str(paper.get("pmid") or paper.get("doi") or "").strip()
            if paper_id:
                paper_ids.add(paper_id)
            handle.write(json.dumps(claim, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "existing_claim_ids": len(existing),
        "written": written,
        "unique_papers": len(paper_ids),
        "skipped_existing": skipped_existing,
        "skipped_duplicate": skipped_duplicate,
        "invalid": invalid,
        "predicate_changes": sum(predicate_changes.values()),
        "predicate_change_top": predicate_changes.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing", type=Path)
    parser.add_argument(
        "--injection-source",
        default="case2_manual_expansion_20260730",
    )
    args = parser.parse_args()
    summary = prepare(
        args.input,
        args.output,
        args.existing,
        args.injection_source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Updated: 2026-07-30 14:45 HKT
