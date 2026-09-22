"""Apply manually reviewed literature relevance reasons to the study JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def paper_id(item: dict[str, Any]) -> str:
    return str(
        item.get("pmid")
        or item.get("doi")
        or item.get("title")
        or ""
    ).strip().casefold()


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def compose_reason(row: pd.Series, language: str) -> str:
    if language == "zh":
        studied = text(row.get("what_the_paper_studied_zh"))
        support = text(row.get("specific_support_zh"))
        strength = text(row.get("evidence_strength_zh"))
        boundary = text(row.get("boundary_zh"))
        return (
            f"本文研究的是：{studied}\n"
            f"对当前 hypothesis 的具体支持：{support}\n"
            f"证据强度：{strength}"
            + (f"——{boundary}" if boundary else "")
        )
    studied = text(row.get("what_the_paper_studied_en"))
    support = text(row.get("specific_support_en"))
    strength = text(row.get("evidence_strength_en"))
    boundary = text(row.get("boundary_en"))
    return (
        f"What this paper studied: {studied}\n"
        f"Specific support for this hypothesis: {support}\n"
        f"Evidence strength: {strength}"
        + (f" — {boundary}" if boundary else "")
    )


def row_paper_id(row: pd.Series) -> str:
    for field in ("pmid", "doi", "paper_title"):
        value = text(row.get(field))
        if value:
            return value.casefold()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--reviewed-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.hypotheses.read_text(encoding="utf-8-sig"))
    reviewed = pd.read_csv(args.reviewed_queue, low_memory=False)
    index = {
        (
            text(row.get("candidate_id")),
            row_paper_id(row),
        ): row
        for _, row in reviewed.iterrows()
    }

    applied = 0
    previously_completed = 0
    missing: list[dict[str, str]] = []
    for hypothesis in payload.get("hypotheses", []):
        candidate_id = str(hypothesis.get("id") or "")
        for paper in hypothesis.get("literature", []):
            key = (candidate_id, paper_id(paper))
            row = index.get(key)
            required = (
                "what_the_paper_studied_zh",
                "specific_support_zh",
                "evidence_strength_zh",
                "what_the_paper_studied_en",
                "specific_support_en",
                "evidence_strength_en",
            )
            if row is None or not all(text(row.get(field)) for field in required):
                if (
                    text(paper.get("manual_relevance_status")) == "completed"
                    and text(paper.get("relevance_reason_zh"))
                    and text(paper.get("relevance_reason_en"))
                ):
                    previously_completed += 1
                    continue
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "paper_id": paper_id(paper),
                        "paper_title": str(paper.get("title") or ""),
                    }
                )
                continue
            paper["relevance_reason_zh"] = compose_reason(row, "zh")
            paper["relevance_reason_en"] = compose_reason(row, "en")
            paper["manual_relevance_status"] = "completed"
            paper["relevance_curation_batch"] = text(row.get("curation_batch"))
            applied += 1

    if args.require_complete and missing:
        raise RuntimeError(
            f"{len(missing)} hypothesis-paper relevance reviews are incomplete"
        )
    payload["manual_relevance_review"] = {
        "newly_applied_references": applied,
        "previously_completed_references": previously_completed,
        "completed_references": applied + previously_completed,
        "missing_references": len(missing),
        "complete": not missing,
    }
    payload["manual_relevance_reasons_completed"] = applied + previously_completed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = args.output.with_name(
        "case1_tcp_external_literature_summary.json"
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        summary["manual_relevance_reasons_completed"] = (
            applied + previously_completed
        )
        summary["manual_relevance_review_complete"] = not missing
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    missing_path = args.output.with_name(
        args.output.stem + "_missing_relevance_reviews.json"
    )
    missing_path.write_text(
        json.dumps({"missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "newly_applied_references": applied,
                "previously_completed_references": previously_completed,
                "completed_references": applied + previously_completed,
                "missing_references": len(missing),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
