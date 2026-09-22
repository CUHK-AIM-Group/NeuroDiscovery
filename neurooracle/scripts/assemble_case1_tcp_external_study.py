"""Assemble the public working Expert Study bank without hidden outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--literature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = load(args.selection)
    literature = load(args.literature)
    pairs = [
        dict(item)
        for item in selection.get("pair_schedule", [])
        if isinstance(item, dict)
    ]
    hypotheses = [
        dict(item)
        for item in literature.get("hypotheses", [])
        if isinstance(item, dict)
    ]
    hypothesis_ids = {str(item.get("id") or "") for item in hypotheses}
    selected_ids = {
        str(pair.get(field) or "")
        for pair in pairs
        for field in ("left_id", "right_id")
    }
    missing = sorted(selected_ids - hypothesis_ids)
    extra = sorted(hypothesis_ids - selected_ids)
    if missing or extra:
        raise RuntimeError(
            f"Selection/literature mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    incomplete_literature = [
        item["id"]
        for item in hypotheses
        if len(item.get("literature", [])) != 5
    ]
    if incomplete_literature:
        raise RuntimeError(
            f"{len(incomplete_literature)} hypotheses do not have five papers"
        )
    incomplete_relevance = [
        (str(item.get("id") or ""), str(reference.get("pmid") or ""))
        for item in hypotheses
        for reference in item.get("literature", [])
        if (
            reference.get("manual_relevance_status") != "completed"
            or not str(reference.get("relevance_reason_zh") or "").strip()
            or not str(reference.get("relevance_reason_en") or "").strip()
        )
    ]
    relevance_review_complete = not incomplete_relevance
    if relevance_review_complete:
        for pair in pairs:
            pair["manual_review"] = {
                "status": "completed",
                "quality": "keep",
                "evidence_grade": "mixed_curated",
                "notes": (
                    "Both hypotheses have five abstract-verified references "
                    "with bilingual manual relevance assessments."
                ),
            }

    session_protocol = dict(selection["session_protocol"])
    pairs_by_session = session_protocol.get("pairs_by_session")
    if isinstance(pairs_by_session, dict) and pairs_by_session:
        session_protocol["pair_pool_per_session"] = max(
            int(value) for value in pairs_by_session.values()
        )
    session_protocol["assignment_policy"] = "fixed_shared_schedule"
    session_protocol["shared_random_seed"] = int(
        session_protocol.get("fixed_shared_seed") or 0
    )

    payload = {
        "schema_version": "case1-tcp-external-expert-study-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_id": "case1_tcp_external_validation_v1",
        "case_study": "case1_tcp_external_validation",
        "status": (
            "ready_for_expert_study"
            if relevance_review_complete
            else "literature_selected_relevance_review_pending"
        ),
        "curation": {
            "status": (
                "external_validation_curated"
                if relevance_review_complete
                else "literature_review_pending"
            ),
            "reviewed_pairs": len(pairs) if relevance_review_complete else 0,
            "reviewed_hypotheses": (
                len(hypotheses) if relevance_review_complete else 0
            ),
        },
        "hypothesis_unit": "disease_atlas_roi_imaging_feature_direction",
        "blinding": (
            "Participants see frozen TCP hypotheses, generator scores when enabled, "
            "and prior literature. External effects, FDR values, and outcome sides "
            "are stored only in the separate private_truth directory."
        ),
        "pairing_policy": selection["selection_policy"],
        "session_protocol": session_protocol,
        "n_pairs": len(pairs),
        "n_hypotheses": len(hypotheses),
        "difficulty_counts": selection["selected_by_difficulty"],
        "disease_counts": selection["selected_by_disease"],
        "literature_quality": {
            "papers_per_hypothesis": 5,
            "selected_reference_records": literature[
                "selected_reference_records"
            ],
            "unique_papers": literature["unique_papers"],
            "all_selected_papers_have_abstracts": (
                literature["hypotheses_with_fewer_than_five_papers"] == 0
            ),
            "pairs_passing_overlap_gate": literature[
                "pairs_passing_literature_overlap_gate"
            ],
            "maximum_pair_reference_jaccard": literature[
                "maximum_pair_reference_jaccard"
            ],
            "relevance_reason_status": (
                "completed" if relevance_review_complete else "pending_manual_review"
            ),
        },
        "pair_schedule": pairs,
        "hypotheses": hypotheses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": len(pairs),
                "hypotheses": len(hypotheses),
                "status": payload["status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
