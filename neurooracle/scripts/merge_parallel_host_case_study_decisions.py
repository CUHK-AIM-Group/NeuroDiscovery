"""Merge disjoint host-review shards and expand a validated import result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neurooracle.scripts.host_review_full_graph_case_study_reaudit import (
    GATE_NAMES,
    expand_compact_decisions,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def task_semantic_fingerprints(task: dict[str, Any]) -> dict[str, str]:
    claims = task["audit_payload"]["claims_to_review"]
    output: dict[str, str] = {}
    for claim in claims:
        canonical = {
            "paper_ref": claim.get("paper_ref"),
            "subject": normalized_text(claim.get("subject")),
            "predicate": normalized_text(claim.get("predicate")),
            "object": normalized_text(claim.get("object")),
            "negated": bool(claim.get("negated", False)),
            "raw_text": normalized_text(claim.get("raw_text")),
            "conditions": sorted(
                normalized_text(value) for value in claim.get("conditions") or []
            ),
        }
        output[str(claim["claim_id"])] = hashlib.sha256(
            canonical_json(canonical).encode("utf-8")
        ).hexdigest()
    return output


def validate_compact_review(review: object) -> dict[str, Any]:
    if not isinstance(review, dict):
        raise ValueError("every shard review must be an object")
    claim_id = str(review.get("claim_id") or "")
    if not claim_id:
        raise ValueError("shard review is missing claim_id")
    labels = review.get("claim_case_study_ids")
    if not isinstance(labels, list) or any(not isinstance(value, str) for value in labels):
        raise ValueError(f"{claim_id}: claim_case_study_ids must be a string list")
    unknown = set(labels) - set(CASE_STUDY_IDS)
    if unknown:
        raise ValueError(f"{claim_id}: unknown labels {sorted(unknown)}")
    normalized_labels = [value for value in CASE_STUDY_IDS if value in set(labels)]
    if normalized_labels != labels:
        raise ValueError(f"{claim_id}: labels must be unique and in registry order")
    confidence = review.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"{claim_id}: confidence must be numeric")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError(f"{claim_id}: confidence outside 0..1")
    reason = str(review.get("reason") or "").strip()
    if len(reason) < 12:
        raise ValueError(f"{claim_id}: evidence-specific reason is missing")
    overrides = review.get("gate_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{claim_id}: gate_overrides must be an object")
    unknown_gates = set(overrides) - set(GATE_NAMES)
    if unknown_gates:
        raise ValueError(f"{claim_id}: unknown gate overrides {sorted(unknown_gates)}")
    if any(not isinstance(value, bool) for value in overrides.values()):
        raise ValueError(f"{claim_id}: gate overrides must be JSON booleans")
    needs_secondary = review.get("needs_secondary_review", False)
    if not isinstance(needs_secondary, bool):
        raise ValueError(f"{claim_id}: needs_secondary_review must be a JSON boolean")
    return {
        "claim_id": claim_id,
        "claim_case_study_ids": normalized_labels,
        "confidence": confidence,
        "reason": reason,
        **({"needs_secondary_review": True} if needs_secondary else {}),
        **({"gate_overrides": overrides} if overrides else {}),
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    task = json.loads(args.task.read_text(encoding="utf-8"))
    task_ids = [str(value) for value in task["claim_ids"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task claim IDs are not unique")
    merged_by_id: dict[str, dict[str, Any]] = {}
    source_by_id: dict[str, str] = {}
    shard_counts: dict[str, int] = {}
    for shard_path in args.shard:
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        reviews = payload.get("reviews")
        if not isinstance(reviews, list):
            raise ValueError(f"{shard_path}: reviews must be a list")
        shard_counts[str(shard_path.resolve())] = len(reviews)
        for raw_review in reviews:
            review = validate_compact_review(raw_review)
            claim_id = review["claim_id"]
            if claim_id not in set(task_ids):
                raise ValueError(f"{shard_path}: unknown task claim {claim_id}")
            if claim_id in merged_by_id:
                raise ValueError(
                    f"duplicate shard decision for {claim_id}: "
                    f"{source_by_id[claim_id]} and {shard_path}"
                )
            merged_by_id[claim_id] = review
            source_by_id[claim_id] = str(shard_path.resolve())
    missing = [claim_id for claim_id in task_ids if claim_id not in merged_by_id]
    if missing:
        raise ValueError(f"shards are missing {len(missing)} task claims: {missing[:5]}")

    fingerprints = task_semantic_fingerprints(task)
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for claim_id, fingerprint in fingerprints.items():
        by_fingerprint[fingerprint].append(claim_id)
    divergent_groups: list[dict[str, Any]] = []
    for fingerprint, claim_ids in by_fingerprint.items():
        if len(claim_ids) < 2:
            continue
        signatures = {
            canonical_json(
                {
                    "labels": merged_by_id[claim_id]["claim_case_study_ids"],
                    "gate_overrides": merged_by_id[claim_id].get("gate_overrides") or {},
                }
            )
            for claim_id in claim_ids
        }
        if len(signatures) > 1:
            divergent_groups.append(
                {"fingerprint": fingerprint, "claim_ids": claim_ids}
            )
    if divergent_groups:
        raise ValueError(
            f"exact same-paper semantic decisions diverge in {len(divergent_groups)} groups: "
            f"{divergent_groups[:3]}"
        )

    ordered = [merged_by_id[claim_id] for claim_id in task_ids]
    args.merged.parent.mkdir(parents=True, exist_ok=True)
    args.merged.write_text(
        json.dumps({"reviews": ordered}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    expanded = expand_compact_decisions(
        SimpleNamespace(task=args.task, decisions=args.merged, output=args.result)
    )
    return {
        "task_id": task["task_id"],
        "claims": len(ordered),
        "shards": shard_counts,
        "exact_semantic_duplicate_groups": sum(
            len(values) > 1 for values in by_fingerprint.values()
        ),
        "divergent_exact_semantic_groups": 0,
        "merged": str(args.merged.resolve()),
        "expanded": expanded,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(merge(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
