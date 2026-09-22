"""Validate and merge the targeted manual Case-2 re-audit results.

This is a staging-only operation.  It validates immutable task identities,
paper/claim coverage, reviewer decisions, and source-grounding of retained
component evidence.  It does not project decisions into any formal KG file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any


MANIFEST_SCHEMA = "case2_targeted_manual_reaudit_manifest.v1"
RESULT_SCHEMA = "case2_targeted_manual_reaudit_result.v1"
MERGED_SCHEMA = "case2_targeted_manual_reaudit_merged.v1"
COMPONENTS = {
    "genetic_or_pathway",
    "brain_imaging_or_physiology",
    "longitudinal_clinical_or_cognitive_outcome",
    "mediation_or_causal_chain",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized(text: object) -> str:
    value = html.unescape(str(text or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(value.lower().split())


def _paper_source_text(paper: dict[str, Any], supplemental_text: str = "") -> str:
    parts: list[str] = []
    source_paper = paper.get("source_paper") or {}
    parts.append(str(source_paper.get("title") or ""))
    parts.append(str(paper.get("abstract") or ""))
    for group in (
        paper.get("currently_case2_labelled_claims") or [],
        paper.get("all_available_claim_context") or [],
    ):
        for claim in group:
            for field in ("subject", "predicate", "object", "raw_text", "raw_sentence"):
                parts.append(str(claim.get(field) or ""))
            parts.append(
                " | ".join(
                    str(claim.get(field) or "")
                    for field in ("subject", "predicate", "object")
                )
            )
    parts.append(supplemental_text)
    return _normalized("\n".join(parts))


def _validate_result(
    *,
    task: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    source_supplements: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    prefix = f"shard {task.get('shard_index')}"
    if result.get("schema_version") != RESULT_SCHEMA:
        errors.append(f"{prefix}: result schema mismatch")
    if result.get("shard_index") != task.get("shard_index"):
        errors.append(f"{prefix}: shard_index mismatch")
    if result.get("task_sha256") != task.get("task_sha256"):
        errors.append(f"{prefix}: task_sha256 mismatch")

    supplied_result_hash = result.get("result_sha256")
    if supplied_result_hash:
        unhashed = {key: value for key, value in result.items() if key != "result_sha256"}
        if supplied_result_hash != _canonical_hash(unhashed):
            errors.append(f"{prefix}: result_sha256 mismatch")

    papers = task.get("papers") or []
    decisions = result.get("decisions") or []
    if len(decisions) != len(papers):
        errors.append(f"{prefix}: paper decision count mismatch")
        return errors

    for paper, decision in zip(papers, decisions):
        paper_index = paper.get("paper_index")
        label = f"{prefix} paper {paper_index}"
        for key in ("paper_index", "paper_key", "audit_input_sha256"):
            if decision.get(key) != paper.get(key):
                errors.append(f"{label}: {key} mismatch")
        paper_decision = decision.get("paper_decision")
        if paper_decision not in {"retain", "remove", "unresolved_source"}:
            errors.append(f"{label}: invalid paper_decision")
        try:
            confidence = float(decision.get("confidence"))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{label}: invalid confidence")
        if len(str(decision.get("rationale") or "").strip()) < 24:
            errors.append(f"{label}: rationale is missing or too short")

        evidence = decision.get("component_evidence") or {}
        if set(evidence) != COMPONENTS:
            errors.append(f"{label}: component evidence keys mismatch")
            evidence = {key: [] for key in COMPONENTS}
        if any(not isinstance(evidence.get(key), list) for key in COMPONENTS):
            errors.append(f"{label}: component evidence values must be lists")
            evidence = {key: [] for key in COMPONENTS}

        task_claim_ids = [
            str(claim.get("claim_id") or "")
            for claim in paper.get("currently_case2_labelled_claims") or []
        ]
        claim_decisions = decision.get("claim_decisions") or []
        result_claim_ids = [str(claim.get("claim_id") or "") for claim in claim_decisions]
        if result_claim_ids != task_claim_ids:
            errors.append(f"{label}: claim order/coverage mismatch")
        retained_claims = 0
        for claim in claim_decisions:
            if claim.get("decision") not in {"retain", "remove"}:
                errors.append(f"{label}: invalid claim decision")
            if claim.get("decision") == "retain":
                retained_claims += 1
            if len(str(claim.get("reason") or "").strip()) < 12:
                errors.append(f"{label}: claim reason is missing or too short")

        if paper_decision == "retain":
            if retained_claims == 0:
                errors.append(f"{label}: retained paper has no retained claim")
            if any(not evidence.get(key) for key in COMPONENTS):
                errors.append(f"{label}: retained paper lacks a chain component")
            source = _paper_source_text(
                paper, source_supplements.get(str(paper.get("paper_key") or ""), "")
            )
            for component, spans in evidence.items():
                for span in spans:
                    if _normalized(span) not in source:
                        errors.append(
                            f"{label}: ungrounded {component} evidence span: {span!r}"
                        )
        elif retained_claims:
            errors.append(f"{label}: non-retained paper has retained claims")

    if errors:
        errors.append(f"{prefix}: invalid result file {result_path}")
    return errors


def merge(root: Path, *, partial: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema mismatch")

    merged_decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    missing_shards: list[int] = []
    result_files: list[dict[str, Any]] = []
    by_shard: list[dict[str, Any]] = []
    reviewer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    supplements_path = root / "source_supplements.json"
    source_supplements: dict[str, str] = {}
    supplements_sha256 = ""
    if supplements_path.exists():
        supplements = _read_json(supplements_path)
        if supplements.get("schema_version") != "case2_manual_source_supplements.v1":
            errors.append("source supplement schema mismatch")
        for entry in supplements.get("entries") or []:
            paper_key = str(entry.get("paper_key") or "")
            source_text = str(entry.get("source_text") or "")
            if not paper_key or not source_text:
                errors.append("malformed source supplement entry")
                continue
            if paper_key in source_supplements:
                errors.append(f"duplicate source supplement: {paper_key}")
                continue
            source_supplements[paper_key] = source_text
        supplements_sha256 = _sha256_file(supplements_path)

    for shard in manifest.get("shards") or []:
        shard_index = int(shard["shard_index"])
        task_path = root / f"shard_{shard_index:02d}_task.json"
        result_path = root / f"shard_{shard_index:02d}_result.json"
        task = _read_json(task_path)
        if task.get("task_sha256") != shard.get("task_sha256"):
            errors.append(f"shard {shard_index}: manifest/task hash mismatch")
        if not result_path.exists():
            missing_shards.append(shard_index)
            continue
        result = _read_json(result_path)
        errors.extend(
            _validate_result(
                task=task,
                result=result,
                result_path=result_path,
                source_supplements=source_supplements,
            )
        )
        result_files.append(
            {
                "shard_index": shard_index,
                "path": str(result_path.resolve()),
                "sha256": _sha256_file(result_path),
                "reviewer_id": result.get("reviewer_id"),
            }
        )
        paper_counter = Counter(
            decision["paper_decision"] for decision in result.get("decisions") or []
        )
        claim_counter = Counter(
            claim["decision"]
            for decision in result.get("decisions") or []
            for claim in decision.get("claim_decisions") or []
        )
        by_shard.append(
            {
                "shard_index": shard_index,
                "papers": dict(paper_counter),
                "claims": dict(claim_counter),
            }
        )
        reviewer_id = str(result.get("reviewer_id") or "unknown")
        reviewer_counts[reviewer_id].update(
            {
                "papers": len(result.get("decisions") or []),
                "claims": sum(
                    len(decision.get("claim_decisions") or [])
                    for decision in result.get("decisions") or []
                ),
            }
        )
        for decision in result.get("decisions") or []:
            merged_decisions.append(
                {
                    "shard_index": shard_index,
                    "reviewer_id": reviewer_id,
                    **decision,
                }
            )

    if missing_shards and not partial:
        errors.append(f"missing result shards: {missing_shards}")

    paper_indices = [int(decision["paper_index"]) for decision in merged_decisions]
    if len(paper_indices) != len(set(paper_indices)):
        errors.append("duplicate paper_index across results")
    if not missing_shards and paper_indices != list(
        range(1, int(manifest["inventory"]["papers"]) + 1)
    ):
        errors.append("merged paper order/coverage does not match manifest inventory")

    claim_ids = [
        claim["claim_id"]
        for decision in merged_decisions
        for claim in decision.get("claim_decisions") or []
    ]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("duplicate claim_id across merged results")
    if not missing_shards and len(claim_ids) != int(
        manifest["inventory"]["currently_case2_labelled_claims"]
    ):
        errors.append("merged claim coverage does not match manifest inventory")

    paper_totals = Counter(
        decision["paper_decision"] for decision in merged_decisions
    )
    claim_totals = Counter(
        claim["decision"]
        for decision in merged_decisions
        for claim in decision.get("claim_decisions") or []
    )
    unresolved = [
        {
            "paper_index": decision["paper_index"],
            "paper_key": decision["paper_key"],
            "rationale": decision["rationale"],
        }
        for decision in merged_decisions
        if decision["paper_decision"] == "unresolved_source"
    ]
    retained = [
        {
            "paper_index": decision["paper_index"],
            "paper_key": decision["paper_key"],
            "retained_claim_ids": [
                claim["claim_id"]
                for claim in decision["claim_decisions"]
                if claim["decision"] == "retain"
            ],
        }
        for decision in merged_decisions
        if decision["paper_decision"] == "retain"
    ]

    merged: dict[str, Any] = {
        "schema_version": MERGED_SCHEMA,
        "case_study_id": manifest["case_study_id"],
        "manifest_sha256": _sha256_file(manifest_path),
        "complete": not missing_shards,
        "missing_shards": missing_shards,
        "result_files": result_files,
        "source_supplements_sha256": supplements_sha256,
        "decisions": merged_decisions,
    }
    merged["merged_decisions_sha256"] = _canonical_hash(merged_decisions)
    summary: dict[str, Any] = {
        "schema_version": "case2_targeted_manual_reaudit_summary.v1",
        "case_study_id": manifest["case_study_id"],
        "complete": not missing_shards,
        "validation_passed": not errors,
        "validation_errors": errors,
        "missing_shards": missing_shards,
        "papers": {"total": len(merged_decisions), **dict(paper_totals)},
        "claims": {"total": len(claim_ids), **dict(claim_totals)},
        "by_shard": by_shard,
        "by_reviewer": {
            reviewer: dict(counts) for reviewer, counts in reviewer_counts.items()
        },
        "retained": retained,
        "unresolved_source": unresolved,
        "merged_decisions_sha256": merged["merged_decisions_sha256"],
        "source_supplements_sha256": supplements_sha256,
        "formal_kg_modified": False,
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    merged, summary = merge(args.root.resolve(), partial=args.partial)
    if summary["validation_errors"]:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    merged_path = args.root / "merged_manual_reaudit.json"
    summary_path = args.root / "summary.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
