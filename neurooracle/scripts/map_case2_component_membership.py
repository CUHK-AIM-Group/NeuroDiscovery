"""Project existing component memberships into Case 2, without semantic re-audit.

The current Case 2 policy is additive: a claim already assigned to
``imaging_genetics``, ``progression_prediction``, or ``prognosis`` also belongs
to ``case2_pathway_mediation``.  This migration applies that deterministic
projection to both canonical stores and all claim-backed graph edges while
preserving the prior semantic ``scope_reaudit`` record unchanged.

Dry-run is the default.  ``--apply`` requires a matching dry-run against the
same byte-for-byte canonical baseline, builds and validates complete temporary
copies, creates a hash-verified rollback snapshot, and only then swaps the
formal files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from neurooracle.scripts.count_case_study_kg_stats import (
    PaperIdentityIndex,
    paper_aliases,
)
from neurooracle.scripts.refresh_full_v2_state import build_state, render_readme
from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    iter_edges,
    rewrite_concepts,
    rewrite_edges,
)
from neurooracle.src.case_study_membership_policy import (
    CASE2_COMPONENT_IDS,
    CASE2_ID,
    RUBRIC_SHA256,
    RUBRIC_VERSION,
)
from neurooracle.src.case_study_scope import (
    CASE_STUDY_IDS,
    normalize_case_study_ids,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_FULL_V2 = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_OUTPUT_DIR = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "case2_component_mapping_20260811"
)
ARCHIVE_ROOT = REPO / "neurooracle" / "data" / "archive" / "kg_mutation_backups"
MAPPING_ID = "case2_component_mapping_20260811"
MAPPING_SCHEMA_VERSION = "neurooracle.case2_component_mapping.v1"
PROJECTION_FIELD = "case_study_membership_projections"
COMPONENT_IDS = tuple(
    case_study_id
    for case_study_id in CASE_STUDY_IDS
    if case_study_id in CASE2_COMPONENT_IDS
)
CANONICAL_FILES = (
    "knowledge_graph.json",
    "extracted_claims.jsonl",
    "CURRENT_STATE.json",
    "README.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def baseline_hashes(full_v2: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_FILES:
        path = full_v2 / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def ordered(values: object) -> list[str]:
    normalized = normalize_case_study_ids(values, strict=True)
    return [case_study_id for case_study_id in CASE_STUDY_IDS if case_study_id in normalized]


def projected_labels(values: object) -> list[str]:
    """Return the additive deterministic Case 2 projection."""

    labels = set(ordered(values))
    if labels.intersection(CASE2_COMPONENT_IDS):
        labels.add(CASE2_ID)
    return [case_study_id for case_study_id in CASE_STUDY_IDS if case_study_id in labels]


def _membership_pair(claim: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    claim_labels = ordered(claim.get("claim_case_study_ids") or [])
    paper_labels = ordered(claim.get("paper_case_study_ids") or [])
    if not set(claim_labels).issubset(paper_labels):
        raise ValueError(
            f"{claim.get('id')}: claim membership is not a paper-membership subset"
        )
    metadata = claim.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{claim.get('id')}: metadata must be an object")
    if ordered(metadata.get("claim_case_study_ids") or []) != claim_labels:
        raise ValueError(f"{claim.get('id')}: nested claim membership differs")
    if ordered(metadata.get("paper_case_study_ids") or []) != paper_labels:
        raise ValueError(f"{claim.get('id')}: nested paper membership differs")
    return claim_labels, paper_labels


def expected_membership(claim: Mapping[str, Any]) -> dict[str, Any]:
    old_claim, old_paper = _membership_pair(claim)
    new_claim = projected_labels(old_claim)
    new_paper = projected_labels(old_paper)
    return {
        "old_claim": old_claim,
        "old_paper": old_paper,
        "new_claim": new_claim,
        "new_paper": new_paper,
        "claim_changed": old_claim != new_claim,
        "paper_changed": old_paper != new_paper,
    }


def projection_record(expected: Mapping[str, Any], *, applied_at: str) -> dict[str, Any]:
    old_claim = list(expected["old_claim"])
    old_paper = list(expected["old_paper"])
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "mapping_id": MAPPING_ID,
        "mode": "additive_deterministic_projection_without_semantic_reaudit",
        "applied_at": applied_at,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "source_case_study_ids": list(COMPONENT_IDS),
        "target_case_study_id": CASE2_ID,
        "claim_components_present": [
            value for value in COMPONENT_IDS if value in old_claim
        ],
        "paper_components_present": [
            value for value in COMPONENT_IDS if value in old_paper
        ],
        "previous_claim_case_study_ids": old_claim,
        "previous_paper_case_study_ids": old_paper,
        "projected_claim_case_study_ids": list(expected["new_claim"]),
        "projected_paper_case_study_ids": list(expected["new_paper"]),
        "semantic_reaudit_deferred": True,
    }


def apply_claim_projection(
    claim: dict[str, Any],
    *,
    applied_at: str,
) -> dict[str, Any]:
    """Mutate one claim and return its before/after membership description."""

    expected = expected_membership(claim)
    if not expected["claim_changed"] and not expected["paper_changed"]:
        return expected

    claim["claim_case_study_ids"] = list(expected["new_claim"])
    claim["paper_case_study_ids"] = list(expected["new_paper"])
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{claim.get('id')}: metadata must be an object")
    metadata["claim_case_study_ids"] = list(expected["new_claim"])
    metadata["paper_case_study_ids"] = list(expected["new_paper"])
    metadata["case_study_membership_schema_version"] = "case_study_membership.v2"
    projections = metadata.setdefault(PROJECTION_FIELD, [])
    if not isinstance(projections, list):
        raise ValueError(f"{claim.get('id')}: {PROJECTION_FIELD} must be a list")
    if any(
        isinstance(item, Mapping) and item.get("mapping_id") == MAPPING_ID
        for item in projections
    ):
        raise ValueError(f"{claim.get('id')}: mapping already recorded")
    projections.append(projection_record(expected, applied_at=applied_at))
    return expected


def _read_graph_inventory(graph: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    papers = PaperIdentityIndex()
    claim_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    strict_without_component = 0

    for node_id, node in iter_concepts(graph):
        if not is_claim_node(node_id, node):
            continue
        claim = node.get("metadata")
        if not isinstance(claim, dict):
            raise ValueError(f"{node_id}: graph claim metadata is invalid")
        claim_id = str(claim.get("id") or node_id)
        if claim_id != node_id:
            raise ValueError(f"{node_id}: embedded claim ID differs")
        if claim_id in inventory:
            raise ValueError(f"duplicate graph claim ID: {claim_id}")
        expected = expected_membership(claim)
        inventory[claim_id] = expected
        counters["claims"] += 1
        counters["claim_memberships_added"] += int(expected["claim_changed"])
        counters["paper_label_rows_changed"] += int(expected["paper_changed"])
        counters["claim_rows_changed"] += int(
            expected["claim_changed"] or expected["paper_changed"]
        )
        if (
            CASE2_ID in expected["old_claim"]
            and not set(expected["old_claim"]).intersection(CASE2_COMPONENT_IDS)
        ):
            strict_without_component += 1
        claim_counts.update(expected["new_claim"])
        aliases = paper_aliases(claim)
        if not aliases:
            raise ValueError(f"{claim_id}: paper identity is missing")
        papers.add(aliases, expected["new_paper"])

    roots = papers.roots()
    paper_counts: Counter[str] = Counter()
    for memberships in roots.values():
        paper_counts.update(memberships)
    projected = {
        "general": {"papers": len(roots), "claims": counters["claims"]},
        "case_studies": {
            case_study_id: {
                "papers": paper_counts[case_study_id],
                "claims": claim_counts[case_study_id],
            }
            for case_study_id in CASE_STUDY_IDS
        },
    }
    summary = {
        **dict(counters),
        "legacy_strict_case2_claims_without_component": strict_without_component,
        "projected_coverage": projected,
    }
    return inventory, summary


def _validate_extracted_baseline(
    extracted: Path,
    inventory: Mapping[str, Mapping[str, Any]],
) -> int:
    seen: set[str] = set()
    with extracted.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if claim_id in seen or claim_id not in inventory:
                raise ValueError(
                    f"invalid extracted claim ID at line {line_number}: {claim_id}"
                )
            seen.add(claim_id)
            expected = expected_membership(row)
            graph_expected = inventory[claim_id]
            for field in ("old_claim", "old_paper", "new_claim", "new_paper"):
                if expected[field] != graph_expected[field]:
                    raise ValueError(
                        f"{claim_id}: extracted/graph membership differs for {field}"
                    )
    if seen != set(inventory):
        raise ValueError("extracted and graph claim inventories differ")
    return len(seen)


def _validate_edge_baseline(
    graph: Path,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    referenced = 0
    changed = 0
    for edge in iter_edges(graph):
        metadata = edge.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        claim_id = str(metadata.get("claim_id") or "")
        expected = inventory.get(claim_id)
        if expected is None:
            continue
        referenced += 1
        if ordered(metadata.get("claim_case_study_ids") or []) != expected["old_claim"]:
            raise ValueError(f"{claim_id}: edge/claim membership differs")
        if ordered(metadata.get("paper_case_study_ids") or []) != expected["old_paper"]:
            raise ValueError(f"{claim_id}: edge/paper membership differs")
        changed += int(expected["claim_changed"] or expected["paper_changed"])
    return {"claim_edges": referenced, "claim_edges_changed": changed}


def build_preflight(full_v2: Path) -> dict[str, Any]:
    graph = full_v2 / "knowledge_graph.json"
    extracted = full_v2 / "extracted_claims.jsonl"
    state = read_json(full_v2 / "CURRENT_STATE.json")
    if state.get("status") != "canonical_current":
        raise ValueError("CURRENT_STATE is not canonical_current")
    inventory, summary = _read_graph_inventory(graph)
    extracted_rows = _validate_extracted_baseline(extracted, inventory)
    edge_summary = _validate_edge_baseline(graph, inventory)
    if extracted_rows != summary["claims"]:
        raise ValueError("formal claim-store row count differs from graph claim count")
    current = state.get("formal_kg_statistics") or {}
    if current.get("general") != {
        "papers": summary["projected_coverage"]["general"]["papers"],
        "claims": summary["claims"],
    }:
        # Paper and claim totals are invariant under this mapping.
        raise ValueError("CURRENT_STATE general totals differ from the formal graph")

    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "mapping_id": MAPPING_ID,
        "status": "ready_for_atomic_mapping",
        "dry_run": True,
        "generated_at": utc_now(),
        "formal_kg_mutated": False,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "source_case_study_ids": list(COMPONENT_IDS),
        "target_case_study_id": CASE2_ID,
        "mapping_mode": "additive_deterministic_projection_without_semantic_reaudit",
        "baseline": baseline_hashes(full_v2),
        "baseline_coverage": current,
        **summary,
        **edge_summary,
        "extracted_rows": extracted_rows,
    }


def validate_matching_dry_run(output_dir: Path, preflight: Mapping[str, Any]) -> None:
    path = output_dir / "MAPPING_DRY_RUN.json"
    if not path.is_file():
        raise FileNotFoundError("matching MAPPING_DRY_RUN.json is required")
    previous = read_json(path)
    if previous.get("status") != "ready_for_atomic_mapping":
        raise ValueError("dry-run is not ready_for_atomic_mapping")
    for field in (
        "mapping_id",
        "rubric_sha256",
        "source_case_study_ids",
        "target_case_study_id",
        "claim_memberships_added",
        "paper_label_rows_changed",
        "claim_rows_changed",
        "projected_coverage",
    ):
        if previous.get(field) != preflight.get(field):
            raise ValueError(f"dry-run/apply mismatch: {field}")
    for name in CANONICAL_FILES:
        if (
            previous.get("baseline", {}).get(name, {}).get("sha256")
            != preflight.get("baseline", {}).get(name, {}).get("sha256")
        ):
            raise ValueError(f"formal baseline changed since dry-run: {name}")


def _stage_projection(
    full_v2: Path,
    output_dir: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    applied_at: str,
) -> tuple[Path, Path]:
    stage = output_dir / "projection"
    stage.mkdir(parents=True, exist_ok=True)
    concepts_temp = stage / "knowledge_graph.concepts.tmp.json"
    graph_projected = stage / "knowledge_graph.json"
    extracted_projected = stage / "extracted_claims.jsonl"
    for path in (concepts_temp, graph_projected, extracted_projected):
        if path.exists():
            path.unlink()

    def transform_concept(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
        if not is_claim_node(node_id, node):
            return node
        claim = node.get("metadata")
        if not isinstance(claim, dict):
            raise ValueError(f"{node_id}: graph claim metadata is invalid")
        observed = apply_claim_projection(claim, applied_at=applied_at)
        expected = inventory[node_id]
        if observed["new_claim"] != expected["new_claim"] or observed["new_paper"] != expected["new_paper"]:
            raise ValueError(f"{node_id}: staged graph projection differs from preflight")
        return node

    rewrite_concepts(
        full_v2 / "knowledge_graph.json",
        concepts_temp,
        transform_concept,
    )

    def transform_edge(edge: dict[str, Any]) -> dict[str, Any]:
        metadata = edge.get("metadata")
        if not isinstance(metadata, dict):
            return edge
        expected = inventory.get(str(metadata.get("claim_id") or ""))
        if expected is None:
            return edge
        metadata["claim_case_study_ids"] = list(expected["new_claim"])
        metadata["paper_case_study_ids"] = list(expected["new_paper"])
        metadata["case_study_membership_schema_version"] = "case_study_membership.v2"
        return edge

    rewrite_edges(concepts_temp, graph_projected, transform_edge)
    concepts_temp.unlink()

    seen: set[str] = set()
    with (full_v2 / "extracted_claims.jsonl").open(
        "r", encoding="utf-8"
    ) as source, extracted_projected.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if claim_id in seen or claim_id not in inventory:
                raise ValueError(f"invalid extracted claim during staging: {claim_id}")
            seen.add(claim_id)
            observed = apply_claim_projection(row, applied_at=applied_at)
            expected = inventory[claim_id]
            if observed["new_claim"] != expected["new_claim"] or observed["new_paper"] != expected["new_paper"]:
                raise ValueError(f"{claim_id}: staged extracted projection differs")
            destination.write(compact_json(row) + "\n")
    if seen != set(inventory):
        raise ValueError("staged extracted inventory differs from the graph")
    return graph_projected, extracted_projected


def _validate_staged_claim(
    claim: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    claim_labels, paper_labels = _membership_pair(claim)
    if claim_labels != expected["new_claim"] or paper_labels != expected["new_paper"]:
        raise ValueError(f"{claim.get('id')}: staged membership mismatch")
    if expected["claim_changed"] or expected["paper_changed"]:
        metadata = claim.get("metadata") or {}
        records = [
            item
            for item in metadata.get(PROJECTION_FIELD) or []
            if isinstance(item, Mapping) and item.get("mapping_id") == MAPPING_ID
        ]
        if len(records) != 1:
            raise ValueError(f"{claim.get('id')}: mapping provenance is missing/duplicated")


def validate_projection(
    graph: Path,
    extracted: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    projected_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    graph_seen: set[str] = set()
    for node_id, node in iter_concepts(graph):
        if not is_claim_node(node_id, node):
            continue
        claim = node.get("metadata") or {}
        _validate_staged_claim(claim, inventory[node_id])
        graph_seen.add(node_id)
    if graph_seen != set(inventory):
        raise ValueError("staged graph claim inventory differs")

    edge_count = 0
    for edge in iter_edges(graph):
        metadata = edge.get("metadata") or {}
        expected = inventory.get(str(metadata.get("claim_id") or ""))
        if expected is None:
            continue
        if ordered(metadata.get("claim_case_study_ids") or []) != expected["new_claim"]:
            raise ValueError("staged edge claim membership mismatch")
        if ordered(metadata.get("paper_case_study_ids") or []) != expected["new_paper"]:
            raise ValueError("staged edge paper membership mismatch")
        edge_count += 1

    extracted_seen: set[str] = set()
    with extracted.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if claim_id in extracted_seen or claim_id not in inventory:
                raise ValueError(f"invalid staged extracted claim: {claim_id}")
            _validate_staged_claim(row, inventory[claim_id])
            extracted_seen.add(claim_id)
    if extracted_seen != set(inventory):
        raise ValueError("staged extracted claim inventory differs")

    state = build_state(graph.parent)
    actual = state["formal_kg_statistics"]
    if actual["general"] != projected_coverage["general"]:
        raise ValueError("staged general coverage differs from projection")
    if actual["case_studies"] != projected_coverage["case_studies"]:
        raise ValueError("staged Case Study coverage differs from projection")
    return {
        "graph_claims": len(graph_seen),
        "extracted_rows": len(extracted_seen),
        "claim_edges_checked": edge_count,
        "state": state,
    }


def copy_with_hash_check(source: Path, target: Path, expected_sha256: str) -> None:
    shutil.copy2(source, target)
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise ValueError(f"backup hash mismatch for {source.name}")


def apply_mapping(
    full_v2: Path,
    output_dir: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    validate_matching_dry_run(output_dir, preflight)
    inventory, _ = _read_graph_inventory(full_v2 / "knowledge_graph.json")
    applied_at = utc_now()
    graph_projected, extracted_projected = _stage_projection(
        full_v2,
        output_dir,
        inventory,
        applied_at=applied_at,
    )
    validation = validate_projection(
        graph_projected,
        extracted_projected,
        inventory,
        preflight["projected_coverage"],
    )
    projected_hashes = {
        "knowledge_graph.json": sha256_file(graph_projected),
        "extracted_claims.jsonl": sha256_file(extracted_projected),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = ARCHIVE_ROOT / f"{MAPPING_ID}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in CANONICAL_FILES:
        copy_with_hash_check(
            full_v2 / name,
            backup_dir / name,
            preflight["baseline"][name]["sha256"],
        )
    atomic_write_json(
        backup_dir / "backup_manifest.json",
        {
            "schema_version": "neurooracle.kg_mutation_backup.v1",
            "mapping_id": MAPPING_ID,
            "created_at": utc_now(),
            "source_full_v2": str(full_v2.resolve()),
            "files": preflight["baseline"],
        },
    )

    swapped = False
    try:
        os.replace(graph_projected, full_v2 / "knowledge_graph.json")
        os.replace(extracted_projected, full_v2 / "extracted_claims.jsonl")
        swapped = True
        if sha256_file(full_v2 / "knowledge_graph.json") != projected_hashes["knowledge_graph.json"]:
            raise ValueError("live graph differs from the validated projection")
        if sha256_file(full_v2 / "extracted_claims.jsonl") != projected_hashes["extracted_claims.jsonl"]:
            raise ValueError("live claim store differs from the validated projection")

        state = build_state(full_v2)
        if state["formal_kg_statistics"]["general"] != preflight["projected_coverage"]["general"]:
            raise ValueError("post-mapping general coverage mismatch")
        if state["formal_kg_statistics"]["case_studies"] != preflight["projected_coverage"]["case_studies"]:
            raise ValueError("post-mapping Case Study coverage mismatch")
        mapping_summary = {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "mapping_id": MAPPING_ID,
            "mode": preflight["mapping_mode"],
            "applied_at": applied_at,
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": RUBRIC_SHA256,
            "source_case_study_ids": list(COMPONENT_IDS),
            "target_case_study_id": CASE2_ID,
            "claims_added_to_case2": preflight["claim_memberships_added"],
            "legacy_strict_case2_claims_preserved": preflight[
                "legacy_strict_case2_claims_without_component"
            ],
            "semantic_reaudit_deferred_until_v3_complete": True,
        }
        state["case2_component_mapping"] = mapping_summary
        atomic_write_json(full_v2 / "CURRENT_STATE.json", state)
        atomic_write_text(full_v2 / "README.md", render_readme(state))

        post_hashes = baseline_hashes(full_v2)
        report = {
            **preflight,
            "status": "mapped_and_validated",
            "dry_run": False,
            "formal_kg_mutated": True,
            "applied_at": applied_at,
            "completed_at": utc_now(),
            "backup_dir": str(backup_dir.resolve()),
            "validated_projection_hashes": projected_hashes,
            "post_mapping_hashes": post_hashes,
            "final_coverage": state["formal_kg_statistics"],
            "validation": {
                key: value for key, value in validation.items() if key != "state"
            },
            "mapping_summary": mapping_summary,
        }
        atomic_write_json(output_dir / "MAPPING_COMPLETE.json", report)
        atomic_write_json(backup_dir / "MAPPING_COMPLETE.json", report)
        return report
    except Exception:
        if swapped:
            for name in CANONICAL_FILES:
                shutil.copyfile(backup_dir / name, full_v2 / name)
            for name in CANONICAL_FILES:
                if sha256_file(full_v2 / name) != preflight["baseline"][name]["sha256"]:
                    raise RuntimeError(f"rollback hash validation failed for {name}")
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_v2 = args.full_v2.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(full_v2)
    if not args.apply:
        atomic_write_json(output_dir / "MAPPING_DRY_RUN.json", preflight)
        return preflight
    return apply_mapping(full_v2, output_dir, preflight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-v2", type=Path, default=DEFAULT_FULL_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        failure = {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "mapping_id": MAPPING_ID,
            "status": "failed",
            "failed_at": utc_now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output_dir.resolve() / "MAPPING_FAILED.json", failure)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

