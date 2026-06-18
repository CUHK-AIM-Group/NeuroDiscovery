"""Backfill canonical paper_scope labels onto claim nodes in a KG JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from neurooracle.src.paper_scope import infer_paper_scope_from_claim_dict, normalize_paper_scope


def _concepts(data: dict[str, Any]) -> list[dict[str, Any]]:
    concepts = data.get("concepts") or []
    if isinstance(concepts, dict):
        return [node for node in concepts.values() if isinstance(node, dict)]
    return [node for node in concepts if isinstance(node, dict)]


def _is_claim_node(node: dict[str, Any]) -> bool:
    metadata = node.get("metadata")
    return (
        str(node.get("id", "")).startswith("CLM:")
        or "claim" in (node.get("domain_tags") or [])
        or (isinstance(metadata, dict) and "source_paper" in metadata and "subject_name" in metadata)
    )


def _paper_id(claim: dict[str, Any]) -> str:
    paper = claim.get("source_paper")
    if not isinstance(paper, dict):
        return ""
    for key in ("pmid", "doi", "pmcid", "arxiv_id"):
        value = str(paper.get(key) or "").strip()
        if value:
            return value
    return ""


def _legacy_text_scope(claim: dict[str, Any]) -> list[str]:
    """Recover early unscoped case-targeted claims where only text carried the cue."""
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("id", "raw_text", "subject_name", "object_name")
    )
    paper = claim.get("source_paper")
    if isinstance(paper, dict):
        text += " " + str(paper.get("title") or "")
    text = text.lower()
    if "transdiagnostic" in text or "cross-diagnostic" in text or "cross diagnostic" in text:
        return ["case1"]
    return []


def claim_scope(claim: dict[str, Any], *, legacy_text_heuristic: bool) -> list[str]:
    scope = infer_paper_scope_from_claim_dict(claim, default=())
    if scope:
        return scope
    if legacy_text_heuristic:
        scope = _legacy_text_scope(claim)
        if scope:
            return scope
    return ["general"]


def backfill_paper_scope(
    graph_path: Path,
    *,
    dry_run: bool = False,
    legacy_text_heuristic: bool = True,
) -> dict[str, Any]:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    claim_nodes = [
        node
        for node in _concepts(data)
        if _is_claim_node(node) and isinstance(node.get("metadata"), dict)
    ]

    paper_to_scope: dict[str, set[str]] = defaultdict(set)
    claim_rows: list[tuple[dict[str, Any], str]] = []
    claims_without_paper_id = 0

    for node in claim_nodes:
        claim = node["metadata"]
        pid = _paper_id(claim)
        if not pid:
            claims_without_paper_id += 1
            continue
        claim_rows.append((claim, pid))
        for scope in claim_scope(claim, legacy_text_heuristic=legacy_text_heuristic):
            paper_to_scope[pid].add(scope)

    changed_claims = 0
    paper_scope_counts: Counter[str] = Counter()
    for claim, pid in claim_rows:
        scope = normalize_paper_scope(sorted(paper_to_scope[pid])) or ["general"]
        old_scope = normalize_paper_scope(claim.get("paper_scope"))
        if old_scope != scope:
            claim["paper_scope"] = scope
            changed_claims += 1
        for item in scope:
            paper_scope_counts[item] += 1

    unique_paper_scope_counts = Counter()
    for scopes in paper_to_scope.values():
        for scope in normalize_paper_scope(sorted(scopes)) or ["general"]:
            unique_paper_scope_counts[scope] += 1

    summary = {
        "graph": str(graph_path),
        "claim_nodes": len(claim_nodes),
        "claims_with_paper_id": len(claim_rows),
        "claims_without_paper_id": claims_without_paper_id,
        "unique_papers": len(paper_to_scope),
        "changed_claims": changed_claims,
        "claim_scope_counts": dict(sorted(paper_scope_counts.items())),
        "paper_scope_counts": dict(sorted(unique_paper_scope_counts.items())),
        "legacy_text_heuristic": legacy_text_heuristic,
        "dry_run": dry_run,
    }

    data.setdefault("metadata", {})["paper_scope_backfill"] = {
        **summary,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if not dry_run:
        tmp_path = graph_path.with_suffix(graph_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(graph_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to knowledge_graph.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-legacy-text-heuristic",
        action="store_true",
        help="Do not recover old transdiagnostic claims from title/raw text.",
    )
    args = parser.parse_args()

    summary = backfill_paper_scope(
        args.graph,
        dry_run=args.dry_run,
        legacy_text_heuristic=not args.no_legacy_text_heuristic,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
