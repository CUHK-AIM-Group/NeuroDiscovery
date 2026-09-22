"""Prepare immutable manual-review shards for existing Case 2 memberships.

The formal KG is never mutated.  Inputs come from the completed full-graph
re-audit ledger and are grouped by canonical paper identity so reviewers can
decide the paper-level complete chain before deciding which labelled claims
actually participate in that chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "full_graph_v3"
    / "reaudit.sqlite"
)
DEFAULT_CALIBRATION = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "calibration"
    / "case2_paper_chain_calibration_20260810.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "case2_targeted_manual_20260810"
)
CASE2_ID = "case2_pathway_mediation"
SCHEMA_VERSION = "case2_targeted_manual_reaudit_task.v1"


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if stat.st_size <= 32 * 1024 * 1024
        else None,
    }


def claim_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": str(payload.get("id") or ""),
        "subject": payload.get("subject_name") or payload.get("subject") or "",
        "predicate": payload.get("predicate") or "",
        "object": payload.get("object_name") or payload.get("object") or "",
        "raw_text": payload.get("raw_text") or "",
        "subject_type": payload.get("subject_type") or "",
        "object_type": payload.get("object_type") or "",
        "conditions": payload.get("conditions") or [],
        "evidence": payload.get("evidence") or {},
    }


def load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cohort") == "historical_positive":
                result[str(row.get("paper_key") or "")] = row
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    ledger = args.ledger.resolve()
    calibration_path = args.calibration.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(ledger)
    connection.row_factory = sqlite3.Row
    try:
        selected: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute(
            """
            SELECT claim_id, paper_key, payload_json, review_json, graph_ordinal
            FROM claims WHERE review_status='final_complete'
            ORDER BY graph_ordinal
            """
        ):
            review = json.loads(row["review_json"])
            if CASE2_ID not in (review.get("claim_case_study_ids") or []):
                continue
            payload = json.loads(row["payload_json"])
            selected.setdefault(str(row["paper_key"]), []).append(
                {
                    **claim_view(payload),
                    "graph_ordinal": int(row["graph_ordinal"]),
                    "old_case2_review": {
                        "confidence": review.get("confidence"),
                        "reason": review.get("reason"),
                        "gates": review.get("gates") or {},
                        "reviewer_id": review.get("reviewer_id"),
                        "reviewed_at": review.get("reviewed_at"),
                    },
                }
            )

        calibration = load_calibration(calibration_path)
        papers: list[dict[str, Any]] = []
        for paper_index, paper_key in enumerate(sorted(selected), start=1):
            paper_row = connection.execute(
                """
                SELECT source_paper_json, aliases_json, abstract, abstract_source,
                       claim_count
                FROM papers WHERE paper_key=?
                """,
                (paper_key,),
            ).fetchone()
            if paper_row is None:
                raise RuntimeError(f"missing paper ledger row: {paper_key}")
            context = [
                claim_view(json.loads(row[0]))
                for row in connection.execute(
                    """
                    SELECT payload_json FROM claims WHERE paper_key=?
                    ORDER BY graph_ordinal
                    """,
                    (paper_key,),
                )
            ]
            task = {
                "paper_index": paper_index,
                "paper_key": paper_key,
                "aliases": json.loads(paper_row["aliases_json"]),
                "source_paper": json.loads(paper_row["source_paper_json"]),
                "abstract": str(paper_row["abstract"] or ""),
                "abstract_source": str(paper_row["abstract_source"] or ""),
                "abstract_available": bool(paper_row["abstract"]),
                "paper_claim_count": int(paper_row["claim_count"]),
                "currently_case2_labelled_claims": selected[paper_key],
                "all_available_claim_context": context,
                "deterministic_calibration": calibration.get(paper_key),
                "required_decision": {
                    "paper_decision": "retain | remove | unresolved_source",
                    "component_evidence": {
                        "genetic_or_pathway": "exact supported span(s)",
                        "brain_imaging_or_physiology": "exact supported span(s)",
                        "longitudinal_clinical_or_cognitive_outcome": "exact supported span(s)",
                        "mediation_or_causal_chain": "exact supported span(s)",
                    },
                    "claim_decisions": "one retain/remove decision per currently labelled claim",
                },
            }
            task["audit_input_sha256"] = sha256_json(task)
            papers.append(task)

        if len(papers) != 912:
            raise RuntimeError(f"expected 912 Case 2 papers, found {len(papers)}")
        labelled_claims = sum(
            len(paper["currently_case2_labelled_claims"]) for paper in papers
        )
        if labelled_claims != 939:
            raise RuntimeError(f"expected 939 Case 2 claims, found {labelled_claims}")

        shard_count = int(args.shards)
        if shard_count <= 0:
            raise ValueError("shards must be positive")
        shard_size = (len(papers) + shard_count - 1) // shard_count
        shard_reports: list[dict[str, Any]] = []
        for shard_index in range(shard_count):
            start = shard_index * shard_size
            stop = min(len(papers), start + shard_size)
            shard_papers = papers[start:stop]
            if not shard_papers:
                continue
            payload = {
                "schema_version": SCHEMA_VERSION,
                "case_study_id": CASE2_ID,
                "shard_index": shard_index + 1,
                "paper_index_start": shard_papers[0]["paper_index"],
                "paper_index_end": shard_papers[-1]["paper_index"],
                "review_rule": (
                    "Retain only when the same paper explicitly supports a genetic/"
                    "pathway -> brain imaging/physiology -> later longitudinal clinical/"
                    "cognitive outcome chain and an explicit mediation/causal-chain "
                    "relation. Mere co-occurrence or separate associations are insufficient."
                ),
                "papers": shard_papers,
            }
            payload["task_sha256"] = sha256_json(payload)
            path = output_dir / f"shard_{shard_index + 1:02d}_task.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            shard_reports.append(
                {
                    "shard_index": shard_index + 1,
                    "path": str(path),
                    "papers": len(shard_papers),
                    "case2_claims": sum(
                        len(p["currently_case2_labelled_claims"])
                        for p in shard_papers
                    ),
                    "bytes": path.stat().st_size,
                    "task_sha256": payload["task_sha256"],
                }
            )

        manifest = {
            "schema_version": "case2_targeted_manual_reaudit_manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_study_id": CASE2_ID,
            "sources": {
                "ledger": file_fingerprint(ledger),
                "calibration": file_fingerprint(calibration_path),
            },
            "inventory": {
                "papers": len(papers),
                "currently_case2_labelled_claims": labelled_claims,
                "papers_with_abstract": sum(p["abstract_available"] for p in papers),
                "papers_without_abstract": sum(
                    not p["abstract_available"] for p in papers
                ),
                "shards": len(shard_reports),
            },
            "shards": shard_reports,
            "mutation_policy": "Review outputs are staging-only; formal KG mutation is forbidden.",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
