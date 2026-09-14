"""Build a reversible metadata/alias repair on the frozen compact candidate.

Only 480 reviewed semantic-type lists and two reviewed alias lists may change.
Retired lexical pairs move to a history table; actual UMLS mapping edges and
all raw UMLS detail payloads remain untouched. Formal files are read-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_umls_simplification_candidate import (
    DEFAULT_OUTPUT as BASE_DIR, FORMAL, REPO, UMLS_ROOT, atomic_json, cheap,
    compact, digest_update, evidence_scores, file_sha, hashed_reader,
    kge_input_regression, progress, read_json, utc_now, walk_graph,
)
from review_umls_existing_alignment import (
    DEFAULT_OUTPUT as REVIEW_ROOT, POLICY_VERSION, check_source_fingerprints,
    checked_jsonl, read_jsonl, write_jsonl,
)
from node_alias_safety import RETIRED_GLOBAL_ALIASES
from umls_audit_store import UmlsAuditStore
from umls_mention_mapping import normalize_term


REVIEW_DIR = REVIEW_ROOT / POLICY_VERSION
DEFAULT_OUTPUT = UMLS_ROOT / "simplification_candidate_v2_node_repairs_20260906"
REPAIR_ID = "umls_2026AA_node_metadata_alias_repair_v1"
SCHEMA = "umls.node_metadata_alias_repair.v1"
ALLOWED_NODE_FIELDS = {"semantic_types", "aliases"}


def check_baselines() -> dict:
    review = read_json(REVIEW_DIR / "REVIEW_FREEZE.json")
    if review.get("status") != "READ_ONLY_ALIGNMENT_REVIEW_VALIDATED":
        raise ValueError("alignment review is not accepted")
    for group in ("artifacts", "context_inputs", "implementation"):
        for fp in review[group].values():
            if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
                raise ValueError(f"review baseline changed: {fp['path']}")
    candidate = check_source_fingerprints(BASE_DIR)
    if candidate != review["source_baseline"]:
        raise ValueError("review belongs to a different candidate")
    return {"review": review, "candidate": candidate}


def load_repair_plan(baseline: dict) -> dict:
    review = baseline["review"]
    proposals = checked_jsonl(review["artifacts"]["METADATA_RECONCILIATION_PROPOSALS.jsonl"])
    semantic_rows = checked_jsonl(review["artifacts"]["SEMANTIC_METADATA_REVIEW.jsonl"])
    semantic_by_id = {row["target_id"]: row for row in semantic_rows}
    refs = {row["node_id"]: row["record"] for row in checked_jsonl(review["context_inputs"]["REFERENCE_NODES.jsonl"])}
    mrsty = review["semantic_metadata_audit"]["source"]
    if file_sha(Path(mrsty["path"])) != mrsty["sha256"]:
        raise ValueError("MRSTY reference changed")
    patches = {}
    for proposal in proposals:
        node_id = proposal["target_id"]
        before = refs[node_id]
        if sorted(before.get("semantic_types") or []) != proposal["before_semantic_types"]:
            raise ValueError("semantic proposal old value does not match source")
        if proposal["proposed_semantic_types"] != semantic_by_id[node_id]["umls_2026AA_semantic_types"] or not proposal["proposed_semantic_types"]:
            raise ValueError("semantic proposal differs from MRSTY evidence")
        if proposal["source_mrsty_sha256"] != mrsty["sha256"] or node_id in patches:
            raise ValueError("duplicate or unbound semantic proposal")
        after = deepcopy(before)
        after["semantic_types"] = list(proposal["proposed_semantic_types"])
        patches[node_id] = {"node_id": node_id, "fields": ["semantic_types"], "before": before, "after": after,
                            "reason": "reconcile_same_recorded_CUI_with_frozen_MRSTY_2026AA"}
    for node_id, aliases in RETIRED_GLOBAL_ALIASES.items():
        if node_id in patches:
            raise ValueError("unexpected overlap between semantic and alias targets")
        before = refs[node_id]
        if any(alias not in before.get("aliases", []) for alias in aliases):
            raise ValueError("reviewed legacy alias is absent or changed")
        after = deepcopy(before)
        retired = {normalize_term(alias) for alias in aliases}
        after["aliases"] = [alias for alias in before.get("aliases", []) if normalize_term(alias) not in retired]
        patches[node_id] = {"node_id": node_id, "fields": ["aliases"], "before": before, "after": after,
                            "retired_aliases": list(aliases), "reason": "retire_reviewed_ambiguous_or_non_equivalent_global_alias"}
    if len(proposals) != 480 or len(patches) != 482:
        raise ValueError("unexpected repair scope; this migration is bound to 480+2 reviewed nodes")
    retired_pairs = []
    with UmlsAuditStore(BASE_DIR / "umls_details.sqlite") as store:
        for node_id in RETIRED_GLOBAL_ALIASES:
            after = patches[node_id]["after"]
            allowed = {normalize_term(value) for value in [after["preferred_name"], *after.get("aliases", [])]}
            for row in store.connection.execute(
                "SELECT c.atom_id,c.target_id,c.mapping_status,c.shared_domains,c.decision,a.payload_json "
                "FROM alignment_candidates c JOIN atoms a ON a.record_id=c.atom_id WHERE c.target_id=? ORDER BY c.atom_id", (node_id,)
            ):
                atom = json.loads(row[5])
                if normalize_term(atom["preferred_name"]) not in allowed:
                    retired_pairs.append({"source_row": list(row[:5]), "atom_name": atom["preferred_name"],
                                          "reason": "retired_global_alias", "repair_id": REPAIR_ID})
    if len(retired_pairs) != 488:
        raise ValueError("retired pair count differs from reviewed 485 CT + 3 fALFF scope")
    return {"patches": patches, "retired_pairs": retired_pairs, "mrsty_source": mrsty}


def validate_patch_shape(patch: dict) -> None:
    before, after = patch["before"], patch["after"]
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if not changed or changed != set(patch["fields"]) or not changed <= ALLOWED_NODE_FIELDS:
        raise ValueError("patch changes an unapproved field")
    if before.get("id") != patch["node_id"] or after.get("id") != patch["node_id"]:
        raise ValueError("repair must not change node identity")
    if (before.get("metadata") or {}).get("audit_ref"):
        raise ValueError("this migration may not change a compact UMLS detail-backed node")


def write_repaired_graph(source: Path, output: Path, patches: dict, expected_source: dict, retired_pair_count: int) -> dict:
    for patch in patches.values():
        validate_patch_shape(patch)
    before_fp = cheap(source)
    counts, fields = Counter(), defaultdict(set)
    digests = {key: hashlib.sha256() for key in ("source_nodes", "repaired_nodes", "unchanged_nodes", "claims", "edges")}
    seen, metadata = set(), None
    affected_claims = []
    graph_path = output / "knowledge_graph.candidate.json"
    writer_digest = hashlib.sha256()
    with graph_path.open("xb", buffering=1024 * 1024) as handle:
        def emit(text):
            value = text.encode("utf-8")
            handle.write(value)
            writer_digest.update(value)
        first_node, first_edge, edge_section = True, True, False
        with hashed_reader(source) as (reader, source_digest):
            for kind, node_id, record in walk_graph(reader):
                if kind == "metadata":
                    if metadata is not None:
                        raise ValueError("duplicate metadata")
                    metadata = deepcopy(record)
                    metadata["umls_node_metadata_alias_repair"] = {
                        "schema_version": SCHEMA, "repair_id": REPAIR_ID, "status": "CANDIDATE_ONLY",
                        "base_candidate_sha256": expected_source["sha256"],
                        "semantic_type_nodes_corrected": sum("semantic_types" in row["fields"] for row in patches.values()),
                        "alias_nodes_corrected": sum("aliases" in row["fields"] for row in patches.values()),
                        "retired_lexical_pairs": retired_pair_count,
                        "reversible_node_patch_log": "NODE_REPAIRS.jsonl",
                        "retired_pair_table": "retired_alignment_candidates",
                        "actual_mapping_edges_changed": False,
                    }
                    emit('{"metadata":' + compact(metadata) + ',"concepts":{')
                    continue
                if metadata is None:
                    raise ValueError("metadata must precede graph data")
                if kind == "node":
                    if edge_section:
                        raise ValueError("node after edge section")
                    digest_update(digests["source_nodes"], record)
                    patched = record
                    if node_id in patches:
                        if record != patches[node_id]["before"] or node_id in seen:
                            raise ValueError(f"repair precondition failed: {node_id}")
                        seen.add(node_id)
                        patched = patches[node_id]["after"]
                    else:
                        digest_update(digests["unchanged_nodes"], record)
                    digest_update(digests["repaired_nodes"], patched)
                    counts["nodes"] += 1
                    fields["node_metadata"].update(patched.get("metadata") or {})
                    if node_id.startswith("CLM:"):
                        counts["claims"] += 1
                        digest_update(digests["claims"], record)
                        md = record.get("metadata") or {}
                        if md.get("subject_id") in patches or md.get("object_id") in patches:
                            affected_claims.append(record)
                    if not first_node:
                        emit(",")
                    first_node = False
                    emit(compact(node_id) + ":" + compact(patched))
                    if counts["nodes"] % 500000 == 0:
                        progress(output, "REPAIR_GRAPH_NODES", nodes=counts["nodes"], changed=len(seen))
                else:
                    if not edge_section:
                        emit('},"edges":[')
                        edge_section = True
                    counts["edges"] += 1
                    digest_update(digests["edges"], record)
                    fields["edge_metadata"].update(record.get("metadata") or {})
                    if not first_edge:
                        emit(",")
                    first_edge = False
                    emit(compact(record))
                    if counts["edges"] % 1000000 == 0:
                        progress(output, "REPAIR_GRAPH_EDGES", edges=counts["edges"])
            source_sha = source_digest.hexdigest()
        if not edge_section:
            emit('},"edges":[')
        emit("]}")
        handle.flush()
        os.fsync(handle.fileno())
    if source_sha != expected_source["sha256"] or cheap(source) != before_fp:
        raise ValueError("base graph integrity check failed")
    if seen != set(patches):
        raise ValueError("not every repair target was found")
    if counts["nodes"] != metadata["stats"]["n_concepts"] or counts["edges"] != metadata["stats"]["n_edges"]:
        raise ValueError("graph counts changed unexpectedly")
    patches_fp = write_jsonl(output / "NODE_REPAIRS.jsonl", (patches[key] for key in sorted(patches)))
    affected_fp = write_jsonl(output / "AFFECTED_CLAIMS.jsonl", affected_claims)
    return {"source": {**before_fp, "sha256": source_sha},
            "candidate": {**cheap(graph_path), "sha256": writer_digest.hexdigest()},
            "counts": dict(counts), "digests": {key: value.hexdigest() for key, value in digests.items()},
            "metadata_fields": {key: sorted(value) for key, value in fields.items()},
            "changed_nodes": len(seen), "patch_log": patches_fp, "affected_claims": affected_fp,
            "connected_components": metadata["stats"]["connected_components"]}


def copy_and_repair_details(source: Path, output: Path, retired_pairs: list, expected_sha: str) -> dict:
    before = cheap(source)
    path = output / "umls_details.sqlite"
    digest = hashlib.sha256()
    progress(output, "COPY_AND_VERIFY_DETAIL_STORE")
    with source.open("rb") as src, path.open("xb") as dst:
        for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    if digest.hexdigest() != expected_sha or cheap(source) != before:
        raise ValueError("detail source differs from frozen baseline")
    progress(output, "RETIRE_INVALIDATED_LEXICAL_PAIRS", pairs=len(retired_pairs))
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE retired_alignment_candidates(atom_id TEXT NOT NULL,target_id TEXT NOT NULL,mapping_status TEXT NOT NULL,shared_domains TEXT NOT NULL,decision TEXT NOT NULL,retirement_reason TEXT NOT NULL,repair_id TEXT NOT NULL,PRIMARY KEY(atom_id,target_id))")
        before_count = connection.execute("SELECT COUNT(*) FROM alignment_candidates").fetchone()[0]
        for retired in retired_pairs:
            old = tuple(retired["source_row"])
            observed = connection.execute("SELECT * FROM alignment_candidates WHERE atom_id=? AND target_id=?", old[:2]).fetchone()
            if observed != old:
                raise ValueError("lexical pair changed before retirement")
            connection.execute("INSERT INTO retired_alignment_candidates VALUES (?,?,?,?,?,?,?)", (*old, retired["reason"], REPAIR_ID))
            if connection.execute("DELETE FROM alignment_candidates WHERE atom_id=? AND target_id=?", old[:2]).rowcount != 1:
                raise ValueError("lexical pair retirement did not affect exactly one row")
        after_count = connection.execute("SELECT COUNT(*) FROM alignment_candidates").fetchone()[0]
        if before_count - after_count != len(retired_pairs):
            raise ValueError("unexpected lexical pair count delta")
        connection.execute("INSERT INTO info VALUES (?,?)", ("node_metadata_alias_repair", compact({
            "repair_id": REPAIR_ID, "base_detail_sha256": expected_sha,
            "active_alignment_candidates": after_count, "retired_alignment_candidates": len(retired_pairs),
            "raw_atoms_cuis_mappings_preserved": True,
        })))
    history_fp = write_jsonl(output / "RETIRED_ALIGNMENT_PAIRS.jsonl", retired_pairs)
    return {"source": {**before, "sha256": expected_sha}, "candidate_path": str(path.resolve()),
            "alignment_pairs_before": before_count, "alignment_pairs_active": after_count,
            "alignment_pairs_retired": len(retired_pairs), "retirement_log": history_fp}


def build_candidate(output: Path) -> dict:
    output = output.resolve()
    if output.is_relative_to(FORMAL.resolve()) or output.is_relative_to(BASE_DIR.resolve()) or output.is_relative_to(REVIEW_ROOT.resolve()):
        raise ValueError("repair output must be separate from protected sources")
    if output.exists():
        raise FileExistsError(output)
    baseline = check_baselines()
    plan = load_repair_plan(baseline)
    output.mkdir(parents=True, exist_ok=False)
    progress(output, "BUILD_NODE_REPAIR_CANDIDATE", semantic_nodes=480, alias_nodes=2)
    base = baseline["candidate"]
    graph = write_repaired_graph(BASE_DIR / "knowledge_graph.candidate.json", output, plan["patches"],
                                 base["artifacts"]["knowledge_graph.candidate.json"], len(plan["retired_pairs"]))
    detail = copy_and_repair_details(BASE_DIR / "umls_details.sqlite", output, plan["retired_pairs"],
                                     base["artifacts"]["umls_details.sqlite"]["sha256"])
    if check_baselines() != baseline:
        raise ValueError("protected inputs changed during repair construction")
    result = {"schema_version": SCHEMA, "repair_id": REPAIR_ID, "status": "BUILT_AWAITING_VALIDATION",
              "created_at": utc_now(), "baseline": baseline, "graph": graph, "details": detail,
              "mrsty_source": plan["mrsty_source"], "formal_files_modified": False,
              "base_candidate_modified": False, "node_identity_changes": 0, "edge_changes": 0}
    atomic_json(output / "BUILD_COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_candidate(args.output)


if __name__ == "__main__":
    main()
