"""Build an incremental, source-linked Primary/Cross-QA comparison ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.manage_paired_luna_claim_campaign import campaign_shards


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def canonical_value(value: Any) -> str:
    """Return a stable representation for structured claim qualifiers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def relation_signature(claim: dict[str, Any]) -> tuple[Any, ...]:
    from neurooracle.src.kg_bulk_cleanup import legacy_scope_metadata
    metadata = legacy_scope_metadata(claim)
    return (
        norm(claim.get("subject_name")),
        norm(metadata.get("subject_type")),
        norm(metadata.get("subject_canonical_hint")),
        norm(metadata.get("subject_atlas")),
        str(claim.get("predicate") or ""),
        norm(claim.get("object_name")),
        norm(metadata.get("object_type")),
        norm(metadata.get("object_canonical_hint")),
        norm(metadata.get("object_atlas")),
        bool(claim.get("negated")),
        norm(claim.get("raw_text")),
        canonical_value(claim.get("evidence") or {}),
        canonical_value(metadata.get("conditions") or []),
        canonical_value(metadata.get("population")),
        canonical_value(metadata.get("raw_stats") or {}),
        float(claim.get("confidence") or 0.0),
    )


def scope_signature(claim: dict[str, Any]) -> tuple[Any, ...]:
    reaudit = claim.get("scope_reaudit") or {}
    from neurooracle.src.kg_bulk_cleanup import legacy_scope_metadata
    metadata = legacy_scope_metadata(claim)
    gates = reaudit.get("gates") or {}
    return (
        tuple(sorted(str(value) for value in claim.get("paper_case_study_ids") or [])),
        tuple(sorted(str(value) for value in claim.get("claim_case_study_ids") or [])),
        tuple(sorted((str(key), bool(value)) for key, value in gates.items())),
        tuple(
            sorted(norm(value) for value in metadata.get("scope_evidence_spans") or [])
        ),
        float(metadata.get("scope_confidence") or 0.0),
        norm(metadata.get("scope_decision_basis")),
        float(reaudit.get("confidence") or 0.0),
        norm(reaudit.get("decision_basis")),
    )


def paper_scope_signature(paper: dict[str, Any] | None) -> tuple[str, ...]:
    return tuple(
        sorted(str(value) for value in (paper or {}).get("paper_case_study_ids") or [])
    )


def evidence_anchor(claim: dict[str, Any]) -> str:
    """Return the immutable source sentence used to support a claim."""

    return norm(claim.get("raw_text"))


def evidence_signature(claim: dict[str, Any]) -> tuple[str, bool]:
    """Return a source anchor plus the explicitly encoded claim polarity."""

    return (evidence_anchor(claim), bool(claim.get("negated")))


def classify_decision(
    left_claims: list[dict[str, Any]],
    right_claims: list[dict[str, Any]],
    *,
    left_paper: dict[str, Any] | None = None,
    right_paper: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Classify disagreement without treating any non-exact result as accepted.

    The extra categories are routing hints for adjudication.  They deliberately
    do not infer semantic equivalence from similar endpoint wording.
    """

    left_relation = sorted(relation_signature(row) for row in left_claims)
    right_relation = sorted(relation_signature(row) for row in right_claims)
    left_full = sorted(
        (relation_signature(row), scope_signature(row)) for row in left_claims
    )
    right_full = sorted(
        (relation_signature(row), scope_signature(row)) for row in right_claims
    )
    left_evidence = sorted(evidence_signature(row) for row in left_claims)
    right_evidence = sorted(evidence_signature(row) for row in right_claims)
    left_anchor = sorted(evidence_anchor(row) for row in left_claims)
    right_anchor = sorted(evidence_anchor(row) for row in right_claims)
    left_unique = set(left_evidence)
    right_unique = set(right_evidence)
    overlap = left_unique & right_unique
    union = left_unique | right_unique
    left_paper_scope = paper_scope_signature(left_paper)
    right_paper_scope = paper_scope_signature(right_paper)

    if left_full == right_full and left_paper_scope == right_paper_scope:
        decision = "exact_agreement"
    elif left_relation == right_relation:
        decision = "scope_only_disagreement"
    elif not left_claims or not right_claims:
        decision = "zero_claim_disagreement"
    elif left_anchor == right_anchor and left_evidence != right_evidence:
        decision = "explicit_polarity_disagreement"
    elif left_evidence == right_evidence:
        decision = "evidence_aligned_semantic_disagreement"
    elif left_unique == right_unique:
        decision = "claim_granularity_disagreement"
    elif overlap:
        decision = "partial_evidence_overlap_disagreement"
    else:
        decision = "semantic_evidence_disagreement"

    diagnostics = {
        "primary_relation_inventory": left_relation,
        "qa_relation_inventory": right_relation,
        "primary_full_inventory": left_full,
        "qa_full_inventory": right_full,
        "primary_evidence_inventory": left_evidence,
        "qa_evidence_inventory": right_evidence,
        "shared_unique_evidence_count": len(overlap),
        "union_unique_evidence_count": len(union),
        "evidence_jaccard": round(len(overlap) / len(union), 6) if union else 1.0,
        "primary_paper_scope": left_paper_scope,
        "qa_paper_scope": right_paper_scope,
    }
    return decision, diagnostics


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    target = read_json(campaign_dir / "TARGET.json")
    target_by_shard = {
        str(row["shard"]): int(row["paper_count"])
        for row in target.get("shards") or []
    }
    shards = campaign_shards(campaign_dir)
    if tuple(target_by_shard) != shards:
        raise RuntimeError("TARGET shard inventory is not canonical or ordered")
    if sum(target_by_shard.values()) != int(target.get("target_unique_papers") or -1):
        raise RuntimeError("TARGET paper count does not equal its shard inventory")
    output_dir = campaign_dir / "comparison"
    comparisons: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    per_shard: list[dict[str, Any]] = []

    for shard in shards:
        primary_dir = campaign_dir / "primary" / "forks" / shard
        qa_dir = campaign_dir / "qa" / "forks" / shard
        primary_papers = read_jsonl(primary_dir / "paper_results.jsonl")
        qa_papers = read_jsonl(qa_dir / "paper_results.jsonl")
        common = min(
            len(primary_papers), len(qa_papers), target_by_shard[shard]
        )
        primary_claims = {
            str(row["id"]): row for row in read_jsonl(primary_dir / "claims.jsonl")
        }
        qa_claims = {
            str(row["id"]): row for row in read_jsonl(qa_dir / "claims.jsonl")
        }
        tasks = read_jsonl(primary_dir / "task.jsonl")[:common]
        shard_counts: dict[str, int] = {}

        for offset in range(common):
            left_paper = primary_papers[offset]
            right_paper = qa_papers[offset]
            task = tasks[offset]
            identity_fields = ("queue_index", "paper_key", "source_context_sha256")
            for field in identity_fields:
                if left_paper.get(field) != right_paper.get(field):
                    raise RuntimeError(f"{shard}/{offset}: role identity mismatch: {field}")
                if left_paper.get(field) != task.get(field):
                    raise RuntimeError(f"{shard}/{offset}: task identity mismatch: {field}")
            left_claims = [
                primary_claims[str(claim_id)]
                for claim_id in left_paper.get("claim_ids") or []
            ]
            right_claims = [
                qa_claims[str(claim_id)]
                for claim_id in right_paper.get("claim_ids") or []
            ]
            decision, diagnostics = classify_decision(
                left_claims,
                right_claims,
                left_paper=left_paper,
                right_paper=right_paper,
            )
            left_relation = diagnostics["primary_relation_inventory"]
            right_relation = diagnostics["qa_relation_inventory"]
            left_full = diagnostics["primary_full_inventory"]
            right_full = diagnostics["qa_full_inventory"]
            status_counts[decision] = status_counts.get(decision, 0) + 1
            shard_counts[decision] = shard_counts.get(decision, 0) + 1
            comparison = {
                "schema_version": "neurooracle.paired_luna_comparison.v3",
                "shard": shard,
                "queue_index": int(task["queue_index"]),
                "paper_key": str(task["paper_key"]),
                "source_context_sha256": str(task["source_context_sha256"]),
                "decision": decision,
                "primary_claim_count": len(left_claims),
                "qa_claim_count": len(right_claims),
                "primary_relation_inventory_sha256": canonical_hash(left_relation),
                "qa_relation_inventory_sha256": canonical_hash(right_relation),
                "primary_full_inventory_sha256": canonical_hash(left_full),
                "qa_full_inventory_sha256": canonical_hash(right_full),
                "primary_evidence_inventory_sha256": canonical_hash(
                    diagnostics["primary_evidence_inventory"]
                ),
                "qa_evidence_inventory_sha256": canonical_hash(
                    diagnostics["qa_evidence_inventory"]
                ),
                "shared_unique_evidence_count": diagnostics[
                    "shared_unique_evidence_count"
                ],
                "union_unique_evidence_count": diagnostics[
                    "union_unique_evidence_count"
                ],
                "evidence_jaccard": diagnostics["evidence_jaccard"],
                "primary_paper_scope": list(diagnostics["primary_paper_scope"]),
                "qa_paper_scope": list(diagnostics["qa_paper_scope"]),
                "requires_adjudication": decision != "exact_agreement",
            }
            comparisons.append(comparison)
            if decision != "exact_agreement":
                disagreements.append(
                    {
                        **comparison,
                        "paper": task.get("paper") or {},
                        "abstract": str(task.get("abstract") or ""),
                        "primary_claims": left_claims,
                        "qa_claims": right_claims,
                        "adjudication_status": "pending",
                    }
                )
        per_shard.append(
            {
                "shard": shard,
                "paired_papers": common,
                "target_papers": target_by_shard[shard],
                "status_counts": shard_counts,
            }
        )

    write_jsonl(output_dir / "comparisons_working.jsonl", comparisons)
    write_jsonl(output_dir / "disagreements_working.jsonl", disagreements)
    summary = {
        "schema_version": "neurooracle.paired_luna_comparison_summary.v3",
        "updated_at": utc_now(),
        "target_unique_papers": int(target["target_unique_papers"]),
        "paired_papers": len(comparisons),
        "paired_completion_percent": round(
            len(comparisons) * 100 / int(target["target_unique_papers"]), 6
        ),
        "status_counts": status_counts,
        "pending_adjudication_papers": len(disagreements),
        "per_shard": per_shard,
        "artifacts": {
            "comparisons": {
                "path": str(output_dir / "comparisons_working.jsonl"),
                "sha256": sha256_file(output_dir / "comparisons_working.jsonl"),
            },
            "disagreements": {
                "path": str(output_dir / "disagreements_working.jsonl"),
                "sha256": sha256_file(output_dir / "disagreements_working.jsonl"),
            },
        },
        "formal_kg_mutated": False,
    }
    write_json(output_dir / "SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.campaign_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
