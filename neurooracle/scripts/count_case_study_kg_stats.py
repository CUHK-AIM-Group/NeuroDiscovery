"""Stream formal paper/claim coverage for all registered case studies."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.streaming_graph_json import is_claim_node, iter_concepts
from neurooracle.src.case_study_scope import (
    CASE_STUDY_IDS,
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_REPORT = REPO / "neurooracle" / "data" / "full_v2" / "case_study_kg_stats.json"


def clean_doi(value: object) -> str:
    doi = str(value or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)


def clean_title(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def paper_aliases(claim: dict[str, Any]) -> list[str]:
    paper = claim.get("source_paper") or {}
    aliases: list[str] = []
    for prefix, value in (
        ("pmid", paper.get("pmid")),
        ("doi", clean_doi(paper.get("doi"))),
        ("pmcid", paper.get("pmcid")),
        ("arxiv", paper.get("arxiv_id")),
        ("openalex", paper.get("openalex_id")),
    ):
        normalized = str(value or "").strip().lower()
        if normalized:
            aliases.append(f"{prefix}:{normalized}")
    title = clean_title(paper.get("title"))
    year = str(paper.get("year") or paper.get("publication_year") or "")
    if title:
        aliases.append(f"title_year:{title}|{year}")
    nested = claim.get("metadata") or {}
    fallback = str(claim.get("paper_id") or nested.get("paper_id") or "").strip()
    if fallback:
        aliases.append(f"paper_id:{fallback.lower()}")
    return list(dict.fromkeys(aliases))


class PaperIdentityIndex:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.case_study_ids: dict[str, set[str]] = {}

    def find(self, alias: str) -> str:
        self.parent.setdefault(alias, alias)
        self.case_study_ids.setdefault(alias, set())
        root = alias
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[alias] != alias:
            next_alias = self.parent[alias]
            self.parent[alias] = root
            alias = next_alias
        return root

    def add(self, aliases: list[str], memberships: list[str]) -> str:
        if not aliases:
            return ""
        root = self.find(aliases[0])
        for alias in aliases[1:]:
            other = self.find(alias)
            if other == root:
                continue
            self.parent[other] = root
            self.case_study_ids[root].update(self.case_study_ids.pop(other, set()))
        self.case_study_ids[root].update(memberships)
        return root

    def roots(self) -> dict[str, set[str]]:
        merged: dict[str, set[str]] = {}
        for alias in self.parent:
            root = self.find(alias)
            merged.setdefault(root, set()).update(self.case_study_ids.get(root, set()))
        return merged


def count_case_studies(graph_path: Path) -> dict[str, Any]:
    claim_counts: Counter[str] = Counter()
    papers = PaperIdentityIndex()
    claim_nodes = 0
    claims_without_membership = 0
    claims_without_paper_identity = 0
    multi_label_claims = 0
    claim_not_subset_of_paper = 0
    legacy_claims = 0

    for node_id, node in iter_concepts(graph_path):
        if not is_claim_node(node_id, node):
            continue
        claim_nodes += 1
        claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        claim.setdefault("id", node_id)
        claim_ids = claim_case_study_ids_from_dict(claim)
        paper_ids = paper_case_study_ids_from_dict(claim)
        if not set(claim_ids).issubset(paper_ids):
            claim_not_subset_of_paper += 1
            paper_ids = [
                case_study_id
                for case_study_id in CASE_STUDY_IDS
                if case_study_id in {*paper_ids, *claim_ids}
            ]
        if not claim_ids:
            claims_without_membership += 1
        if len(claim_ids) > 1:
            multi_label_claims += 1
        if any(
            key in claim or key in (claim.get("metadata") or {})
            for key in ("paper_scope", "case3_tasks", "case3_subtasks")
        ):
            legacy_claims += 1
        for case_study_id in claim_ids:
            claim_counts[case_study_id] += 1
        aliases = paper_aliases(claim)
        if not aliases:
            claims_without_paper_identity += 1
        papers.add(aliases, paper_ids)

    roots = papers.roots()
    paper_counts: Counter[str] = Counter()
    for memberships in roots.values():
        for case_study_id in memberships:
            paper_counts[case_study_id] += 1

    return {
        "schema_version": "case_study_membership.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(graph_path),
        "source_bytes": graph_path.stat().st_size,
        "counting_policy": (
            "Non-exclusive formal case-study membership. Claim counts use "
            "claim_case_study_ids; paper counts use paper_case_study_ids and "
            "identity union by PMID, DOI, PMCID, arXiv, OpenAlex, title/year, "
            "then fallback paper ID."
        ),
        "general": {
            "papers": len(roots),
            "claims": claim_nodes,
        },
        "case_studies": {
            case_study_id: {
                "papers": paper_counts[case_study_id],
                "claims": claim_counts[case_study_id],
            }
            for case_study_id in CASE_STUDY_IDS
        },
        "quality": {
            "claims_without_case_study_membership": claims_without_membership,
            "claims_without_paper_identity": claims_without_paper_identity,
            "multi_label_claims": multi_label_claims,
            "multi_label_papers": sum(len(ids) > 1 for ids in roots.values()),
            "claim_membership_not_subset_of_paper_membership": claim_not_subset_of_paper,
            "claims_with_legacy_scope_fields": legacy_claims,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = count_case_studies(args.graph.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
