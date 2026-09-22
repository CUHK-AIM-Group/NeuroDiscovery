"""Propagate final reviews only to exact semantic duplicates in the audit ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    RUBRIC_VERSION,
    compact_json,
    connect,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    claim_contract_fields,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def semantic_fingerprint(paper_key: str, payload: dict[str, Any]) -> str:
    """Fingerprint only exact claim semantics within the same canonical paper."""
    canonical = {
        "paper_key": paper_key,
        "subject": _norm(payload.get("subject_name") or payload.get("subject")),
        "predicate": _norm(payload.get("predicate")),
        "object": _norm(payload.get("object_name") or payload.get("object")),
        "negated": bool(payload.get("negated", False)),
        "raw_text": _norm(payload.get("raw_text")),
        "conditions": sorted(_norm(value) for value in payload.get("conditions") or []),
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def review_decision_key(review: dict[str, Any]) -> str:
    decision = {
        "claim_case_study_ids": review.get("claim_case_study_ids") or [],
        "gates": review.get("gates") or {},
    }
    return json.dumps(decision, sort_keys=True, separators=(",", ":"))


def plan_propagation(connection: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchors: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for claim_id, paper_key, payload_json, review_json in connection.execute(
        """
        SELECT claim_id, paper_key, payload_json, review_json FROM claims
        WHERE review_status='final_complete' AND review_json IS NOT NULL
        ORDER BY graph_ordinal
        """
    ):
        payload = json.loads(payload_json)
        review = json.loads(review_json)
        anchors[semantic_fingerprint(str(paper_key), payload)].append(
            (str(claim_id), review)
        )

    consensus: dict[str, tuple[str, dict[str, Any]]] = {}
    conflicting_fingerprints = 0
    for fingerprint, rows in anchors.items():
        decisions = {review_decision_key(review) for _, review in rows}
        if len(decisions) != 1:
            conflicting_fingerprints += 1
            continue
        consensus[fingerprint] = rows[0]

    candidates: list[dict[str, Any]] = []
    pending_counts: Counter[str] = Counter()
    pending_examples: dict[str, str] = {}
    for claim_id, paper_key, payload_json in connection.execute(
        """
        SELECT claim_id, paper_key, payload_json FROM claims
        WHERE review_status IN ('pending', 'primary_failed')
        ORDER BY graph_ordinal
        """
    ):
        payload = json.loads(payload_json)
        fingerprint = semantic_fingerprint(str(paper_key), payload)
        pending_counts[fingerprint] += 1
        pending_examples.setdefault(fingerprint, str(claim_id))
        anchor = consensus.get(fingerprint)
        if anchor is None:
            continue
        anchor_id, review = anchor
        candidates.append(
            {
                "claim_id": str(claim_id),
                "anchor_claim_id": anchor_id,
                "fingerprint": fingerprint,
                "review": review,
                "paper_key": str(paper_key),
                "payload": payload,
            }
        )

    report = {
        "final_anchor_claims": sum(len(rows) for rows in anchors.values()),
        "unique_anchor_fingerprints": len(anchors),
        "conflicting_anchor_fingerprints": conflicting_fingerprints,
        "eligible_pending_exact_duplicates": len(candidates),
        "pending_claims": sum(pending_counts.values()),
        "pending_unique_semantic_fingerprints": len(pending_counts),
        "pending_exact_duplicate_surplus": sum(
            count - 1 for count in pending_counts.values() if count > 1
        ),
        "pending_multi_claim_fingerprint_groups": sum(
            count > 1 for count in pending_counts.values()
        ),
        "largest_pending_groups": [
            {"fingerprint": fingerprint, "count": count,
             "example_claim_id": pending_examples[fingerprint]}
            for fingerprint, count in pending_counts.most_common(10)
            if count > 1
        ],
        "sample_matches": [
            {
                "claim_id": row["claim_id"],
                "anchor_claim_id": row["anchor_claim_id"],
                "claim_case_study_ids": row["review"].get(
                    "claim_case_study_ids"
                )
                or [],
            }
            for row in candidates[:20]
        ],
    }
    return candidates, report


def apply_propagation(connection: Any, candidates: list[dict[str, Any]]) -> int:
    reviewed_at = datetime.now(timezone.utc).isoformat()
    for candidate in candidates:
        anchor_review = candidate["review"]
        stored = {
            "claim_id": candidate["claim_id"],
            "claim_case_study_ids": anchor_review.get("claim_case_study_ids") or [],
            "confidence": min(float(anchor_review.get("confidence", 1.0)), 0.99),
            "reason": (
                f"Exact semantic duplicate of {candidate['anchor_claim_id']} in the "
                "same canonical paper; inherited its finalized adjudication."
            ),
            "needs_secondary_review": False,
            "gates": anchor_review.get("gates") or {},
            "rubric_version": RUBRIC_VERSION,
            "review_stage": "exact_duplicate_propagation",
            "reviewer_id": "deterministic:exact_semantic_duplicate",
            "reasoning_effort": "exact_same_paper_claim_equivalence",
            "reviewed_at": reviewed_at,
            "anchor_claim_id": candidate["anchor_claim_id"],
            "semantic_fingerprint": candidate["fingerprint"],
        }
        stored.update(
            claim_contract_fields(
                paper_key=candidate["paper_key"],
                payload=candidate["payload"],
                labels=stored["claim_case_study_ids"],
                gates=stored["gates"],
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            )
        )
        cursor = connection.execute(
            """
            UPDATE claims SET review_status='final_complete', review_json=?, reviewed_at=?
            WHERE claim_id=? AND review_status IN ('pending', 'primary_failed')
            """,
            (compact_json(stored), reviewed_at, candidate["claim_id"]),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"claim status changed: {candidate['claim_id']}")
    connection.commit()
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    connection = connect(args.output_dir.resolve() / "reaudit.sqlite")
    try:
        candidates, report = plan_propagation(connection)
        report["mode"] = "apply" if args.apply else "dry_run"
        report["claims_applied"] = (
            apply_propagation(connection, candidates) if args.apply else 0
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
