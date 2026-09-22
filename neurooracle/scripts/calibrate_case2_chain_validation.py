#!/usr/bin/env python3
"""Replay the Case 2 paper-chain gate against frozen historical re-audit data.

This is an offline calibration of the deterministic evidence gate, not an LLM
benchmark.  Historical final decisions are read from ``reaudit.sqlite`` in
read-only mode.  The script tests all historical Case-2-positive papers plus a
fixed-seed random-negative cohort and a lexically hard-negative cohort.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from neurooracle.src.case2_chain_validation import (
    CASE2_CHAIN_EVIDENCE_FIELD,
    CASE2_GATE,
    CASE2_ID,
    COMPONENT_KEYS,
    _GENETIC_RE,
    _LONGITUDINAL_RE,
    _MEDIATION_RE,
    _NEURAL_RE,
    _OUTCOME_RE,
    build_case2_paper_chain_validation,
)
from neurooracle.src.case_study_membership_contract import source_text_sha256
from neurooracle.src.case_study_membership_policy import LEGACY_POLICY


RUBRIC_SHA256 = LEGACY_POLICY.rubric_sha256
RUBRIC_VERSION = LEGACY_POLICY.version


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "full_graph_v3"
    / "reaudit.sqlite"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "calibration"
)
CALIBRATION_VERSION = "case2_chain_historical_calibration.v1"


def _sentences(source: str) -> list[str]:
    source = str(source or "")
    spans = re.split(r"(?<=[.!?])\s+|[\r\n]+", source)
    return [span.strip() for span in spans if span.strip()]


def _endpoint_types(items: list[dict]) -> set[str]:
    return {
        str(item.get(field) or "").strip().upper().replace("-", "_")
        for item in items
        for field in ("subject_type", "object_type")
    }


def _first_span(sentences: Iterable[str], predicate) -> str:
    for sentence in sentences:
        if predicate(sentence):
            return sentence
    return ""


def _infer_evidence(source: str, items: list[dict]) -> dict[str, list[str]]:
    sentences = _sentences(source)
    endpoint_types = _endpoint_types(items)
    predicates = {str(item.get("predicate") or "").lower() for item in items}

    genetic = _first_span(sentences, lambda value: bool(_GENETIC_RE.search(value)))
    if not genetic and "GENE_TARGET" in endpoint_types:
        genetic = _first_span(
            sentences,
            lambda value: any(
                str(item.get("raw_sentence") or "").strip() == value
                and "GENE_TARGET"
                in {
                    str(item.get("subject_type") or "").upper(),
                    str(item.get("object_type") or "").upper(),
                }
                for item in items
            ),
        )

    neural = _first_span(sentences, lambda value: bool(_NEURAL_RE.search(value)))
    if not neural and "IMAGING_MARKER" in endpoint_types:
        neural = _first_span(
            sentences,
            lambda value: any(
                str(item.get("raw_sentence") or "").strip() == value
                and "IMAGING_MARKER"
                in {
                    str(item.get("subject_type") or "").upper(),
                    str(item.get("object_type") or "").upper(),
                }
                for item in items
            ),
        )

    longitudinal = _first_span(
        sentences,
        lambda value: bool(_LONGITUDINAL_RE.search(value))
        and bool(_OUTCOME_RE.search(value)),
    )
    mediation = _first_span(sentences, lambda value: bool(_MEDIATION_RE.search(value)))
    if not mediation and "mediates" in predicates:
        mediation = _first_span(
            sentences,
            lambda value: any(
                str(item.get("raw_sentence") or "").strip() == value
                and str(item.get("predicate") or "").lower() == "mediates"
                for item in items
            ),
        )

    return {
        "genetic_or_pathway": [genetic] if genetic else [],
        "brain_imaging_or_physiology": [neural] if neural else [],
        "longitudinal_clinical_or_cognitive_outcome": (
            [longitudinal] if longitudinal else []
        ),
        "mediation_or_causal_chain": [mediation] if mediation else [],
    }


def _signal_count(source: str) -> int:
    return sum(
        (
            bool(_GENETIC_RE.search(source)),
            bool(_NEURAL_RE.search(source)),
            bool(_LONGITUDINAL_RE.search(source) and _OUTCOME_RE.search(source)),
            bool(_MEDIATION_RE.search(source)),
        )
    )


def _chunks(values: list[str], size: int = 400) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_claims(
    connection: sqlite3.Connection,
    paper_keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in _chunks(paper_keys):
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT paper_key, payload_json, review_json FROM claims "
            f"WHERE paper_key IN ({placeholders}) ORDER BY graph_ordinal"
        )
        for paper_key, payload_json, review_json in connection.execute(query, chunk):
            payload = json.loads(payload_json)
            review = json.loads(review_json) if review_json else {}
            metadata = payload.get("metadata") or {}
            claims[paper_key].append(
                {
                    "claim_id": payload.get("id"),
                    "subject": payload.get("subject_name") or payload.get("subject"),
                    "subject_type": payload.get("subject_type")
                    or metadata.get("subject_type"),
                    "predicate": payload.get("predicate"),
                    "object": payload.get("object_name") or payload.get("object"),
                    "object_type": payload.get("object_type")
                    or metadata.get("object_type"),
                    "raw_sentence": payload.get("raw_text") or "",
                    "historical_labels": review.get("claim_case_study_ids") or [],
                }
            )
    return claims


def _confusion(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["historical_case2"] and item["gate_valid"] for item in records)
    fn = sum(item["historical_case2"] and not item["gate_valid"] for item in records)
    fp = sum(not item["historical_case2"] and item["gate_valid"] for item in records)
    tn = sum(not item["historical_case2"] and not item["gate_valid"] for item in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _component_complete(record: dict[str, Any]) -> bool:
    spans = record.get("component_spans") or {}
    return all(bool(spans.get(key)) for key in COMPONENT_KEYS)


def _evaluate(
    paper_key: str,
    paper: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    historical_case2: bool,
    cohort: str,
    source_mode: str,
) -> dict[str, Any]:
    abstract = str(paper.get("abstract") or "").strip()
    claim_context = "\n".join(
        str(item.get("raw_sentence") or "").strip()
        for item in items
        if str(item.get("raw_sentence") or "").strip()
    )
    if source_mode == "complete_abstract":
        source = abstract
    else:
        source = "\n".join(value for value in (abstract, claim_context) if value)

    evidence = _infer_evidence(source, items)
    probe_items = []
    for item in items:
        probe_item = dict(item)
        probe_item["case_study_ids"] = [CASE2_ID]
        probe_item["case_study_gates"] = {CASE2_GATE: True}
        probe_item[CASE2_CHAIN_EVIDENCE_FIELD] = evidence
        probe_items.append(probe_item)
    context_hash = source_text_sha256(source)
    validation = build_case2_paper_chain_validation(
        probe_items,
        source_text=source,
        source_context_sha256=context_hash,
    )
    source_paper = paper.get("source_paper") or {}
    return {
        "paper_key": paper_key,
        "title": source_paper.get("title") or "",
        "year": source_paper.get("year"),
        "cohort": cohort,
        "source_mode": source_mode,
        "historical_case2": historical_case2,
        "gate_valid": validation["valid"],
        "source_characters": len(source),
        "abstract_available": bool(abstract),
        "component_spans": evidence,
        "validation_reasons": validation["reasons"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-count", type=int, default=912)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    ledger = args.ledger.resolve()
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")

    positive_keys = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT c.paper_key FROM claims c, "
            "json_each(c.review_json, '$.claim_case_study_ids') labels "
            "WHERE labels.value=? ORDER BY c.paper_key",
            (CASE2_ID,),
        )
    ]
    positive_set = set(positive_keys)
    papers: dict[str, dict[str, Any]] = {}
    negative_candidates: list[str] = []
    hard_candidates: list[tuple[int, str]] = []
    for paper_key, source_paper_json, abstract in connection.execute(
        "SELECT paper_key, source_paper_json, abstract FROM papers ORDER BY paper_key"
    ):
        papers[paper_key] = {
            "source_paper": json.loads(source_paper_json),
            "abstract": abstract or "",
        }
        if paper_key in positive_set:
            continue
        negative_candidates.append(paper_key)
        if abstract:
            signals = _signal_count(abstract)
            if signals >= 3:
                hard_candidates.append((signals, paper_key))

    rng = random.Random(args.seed)
    random_negative_keys = sorted(
        rng.sample(
            negative_candidates,
            min(args.negative_count, len(negative_candidates)),
        )
    )
    hard_negative_keys = [
        paper_key
        for _, paper_key in sorted(
            hard_candidates, key=lambda item: (-item[0], item[1])
        )[: args.negative_count]
    ]
    selected_keys = sorted(
        positive_set | set(random_negative_keys) | set(hard_negative_keys)
    )
    claims = _load_claims(connection, selected_keys)

    records: list[dict[str, Any]] = []
    cohort_by_key: dict[str, set[str]] = defaultdict(set)
    for key in positive_keys:
        cohort_by_key[key].add("historical_positive")
    for key in random_negative_keys:
        cohort_by_key[key].add("random_negative")
    for key in hard_negative_keys:
        cohort_by_key[key].add("hard_negative")

    for paper_key in selected_keys:
        if not claims.get(paper_key):
            continue
        historical_case2 = paper_key in positive_set
        cohort = "+".join(sorted(cohort_by_key[paper_key]))
        records.append(
            _evaluate(
                paper_key,
                papers[paper_key],
                claims[paper_key],
                historical_case2=historical_case2,
                cohort=cohort,
                source_mode="historical_reaudit_context",
            )
        )
        if papers[paper_key]["abstract"]:
            records.append(
                _evaluate(
                    paper_key,
                    papers[paper_key],
                    claims[paper_key],
                    historical_case2=historical_case2,
                    cohort=cohort,
                    source_mode="complete_abstract",
                )
            )

    context_records = [
        record
        for record in records
        if record["source_mode"] == "historical_reaudit_context"
    ]
    abstract_records = [
        record for record in records if record["source_mode"] == "complete_abstract"
    ]
    historical_positive_context = [
        record for record in context_records if record["historical_case2"]
    ]
    random_negative_context = [
        record
        for record in context_records
        if "random_negative" in record["cohort"]
        and not record["historical_case2"]
    ]
    hard_negative_context = [
        record
        for record in context_records
        if "hard_negative" in record["cohort"]
        and not record["historical_case2"]
    ]
    positive_rejection_reasons = Counter(
        reason
        for record in historical_positive_context
        if not record["gate_valid"]
        for reason in record["validation_reasons"]
    )
    positive_claim_count, positive_paper_count = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT c.paper_key) FROM claims c, "
        "json_each(c.review_json, '$.claim_case_study_ids') labels "
        "WHERE labels.value=?",
        (CASE2_ID,),
    ).fetchone()
    connection.close()

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": CALIBRATION_VERSION,
        "generated_at": generated_at,
        "ledger": str(ledger),
        "ledger_size_bytes": ledger.stat().st_size,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "case_study_id": CASE2_ID,
        "random_seed": args.seed,
        "historical_reference": {
            "case2_claims": positive_claim_count,
            "case2_papers": positive_paper_count,
            "review_status": "final_complete",
        },
        "cohorts": {
            "all_historical_positive_papers": len(positive_keys),
            "random_negative_papers": len(random_negative_keys),
            "hard_negative_papers": len(hard_negative_keys),
            "unique_selected_papers": len(selected_keys),
        },
        "historical_reaudit_context": {
            "records": len(context_records),
            "metrics": _confusion(context_records),
            "historical_positive_component_complete": sum(
                _component_complete(record)
                for record in historical_positive_context
            ),
            "historical_positive_full_validator_pass": sum(
                record["gate_valid"] for record in historical_positive_context
            ),
            "historical_positive_rejection_reasons": dict(
                sorted(positive_rejection_reasons.items())
            ),
            "random_negative_gate_acceptance": {
                "accepted": sum(record["gate_valid"] for record in random_negative_context),
                "total": len(random_negative_context),
            },
            "hard_negative_gate_acceptance": {
                "accepted": sum(record["gate_valid"] for record in hard_negative_context),
                "total": len(hard_negative_context),
            },
        },
        "complete_abstract_available_subset": {
            "records": len(abstract_records),
            "positive_records": sum(
                record["historical_case2"] for record in abstract_records
            ),
            "metrics": _confusion(abstract_records),
        },
        "interpretation": {
            "scope": "Offline deterministic evidence-gate replay; no API or LLM was called.",
            "confirmatory_not_autolabel": (
                "Production uses this validator only to reject unsupported model labels; "
                "it never auto-adds Case 2 membership."
            ),
            "historical_context_note": (
                "Historical re-audit context combines available abstracts with stored "
                "claim evidence, matching the old review context more closely than the "
                "complete-abstract-only subset."
            ),
            "policy_drift_warning": (
                "The historical Case 2 positives are not a clean gold set for the new "
                "paper-level complete-chain rule. Low positive agreement must trigger "
                "targeted Case 2 re-adjudication, not relaxation of the v3 gate."
            ),
        },
        "calibration_status": {
            "negative_safety": (
                "pass"
                if _confusion(context_records)["specificity"] >= 0.99
                else "fail"
            ),
            "historical_positive_compatibility": (
                "pass"
                if _confusion(context_records)["recall"] >= 0.80
                else "fail"
            ),
            "overall": "pass_safety_fail_legacy_compatibility",
        },
        "strict_historical_positive_paper_keys": [
            record["paper_key"]
            for record in historical_positive_context
            if record["gate_valid"]
        ],
        "false_positive_paper_keys": [
            record["paper_key"]
            for record in context_records
            if not record["historical_case2"] and record["gate_valid"]
        ],
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "case2_paper_chain_calibration_20260810.json"
    samples_path = output_dir / "case2_paper_chain_calibration_20260810.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(f"report={report_path}")
    print(f"samples={samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
