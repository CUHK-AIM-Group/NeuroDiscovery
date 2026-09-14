"""Graph-centric quality audit. No KGE fitting/splits, no graph or source edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_umls_experiment_metadata import CANDIDATE, check_candidate
from build_umls_simplification_candidate import (
    REPO, UMLS_ROOT, atomic_json, cheap, compact, file_sha, hashed_reader, progress, read_json, utc_now, walk_graph,
)
from neurooracle.src.kg_quality_checks import claim_record_checks, entity_identifier_checks, is_number
from neurooracle.src.metadata_field_audit import node_class, nonempty
from neurooracle.src.umls_mention_mapping import normalize_term


DEFAULT_OUTPUT = UMLS_ROOT / "graph_integrity_audit_v1_20260907"
SOURCE_FILES = (
    "neurooracle/src/graph_manager.py", "neurooracle/src/storage.py", "neurooracle/src/schema.py",
    "neurooracle/src/claim_semantics.py", "neurooracle/src/case_study_scope.py", "neurooracle/src/atoms.py",
    "neurooracle/src/hypothesis_engine.py", "neurooracle/src/claim_ingestion.py",
    "neurooracle/src/metadata_field_audit.py", "neurooracle/src/umls_mention_mapping.py",
    "neurooracle/src/kg_quality_checks.py", "neurooracle/scripts/audit_kg_integrity.py",
)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path, self.rows = path, 0
        self.handle = path.open("xb", buffering=1024 * 1024)
        self.digest = hashlib.sha256()

    def add(self, row):
        data = (compact(row) + "\n").encode("utf-8")
        self.handle.write(data)
        self.digest.update(data)
        self.rows += 1

    def close(self) -> dict:
        self.handle.close()
        return {**cheap(self.path), "sha256": self.digest.hexdigest(), "rows": self.rows}


def readonly_db(output: Path):
    path = (output / "audit_index.sqlite").resolve(strict=True)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-131072")
    return db


def snapshot_sources() -> dict:
    return {relative: {**cheap(REPO / relative), "sha256": file_sha(REPO / relative)} for relative in SOURCE_FILES}


def unchanged_inputs(inputs: dict) -> None:
    if check_candidate() != inputs["baseline"]:
        raise ValueError("candidate/protected formal baseline changed")
    for fp in inputs["implementation"].values():
        if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError(f"audit dependency changed: {fp['path']}")


def claim_summary(node_id: str, record: dict) -> dict:
    md = record.get("metadata") or {}
    nested = md.get("metadata") or {}
    summary = {key: md.get(key) for key in ("subject_id", "object_id", "subject_name", "object_name", "predicate", "confidence", "negated")}
    for key in ("subject_type", "object_type"):
        summary[key] = md.get(key) or (nested.get(key) if isinstance(nested, dict) else None)
    summary["id"] = node_id
    paper = md.get("source_paper") or {}
    if not isinstance(paper, dict):
        paper = {}
    summary["source_paper"] = {key: paper.get(key) for key in ("pmid", "doi", "title", "year")}
    summary["raw_text_excerpt"] = str(md.get("raw_text") or "")[:900]
    return summary


def scan(output: Path) -> dict:
    if (output / "INPUTS.json").exists():
        raise ValueError("scan already started; use later phases instead of overwriting")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"started_at": utc_now(), "baseline": check_candidate(), "implementation": snapshot_sources(),
              "scope": "full repaired candidate; graph structure, identity records, claim connectivity and real reader behavior; no KGE"}
    atomic_json(output / "INPUTS.json", inputs)
    db = sqlite3.connect(output / "audit_index.sqlite")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-131072")
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,kind TEXT NOT NULL,name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,cui TEXT NOT NULL,parent_id TEXT,brief_json TEXT NOT NULL);
        CREATE TABLE claims(id TEXT PRIMARY KEY,s TEXT,t TEXT,p TEXT,negated TEXT,paper_key TEXT,summary_json TEXT NOT NULL);
        CREATE TABLE edges(ordinal INTEGER PRIMARY KEY,s TEXT,t TEXT,r TEXT,confidence REAL,
            claim_id TEXT,negated TEXT,source TEXT,record_sha256 TEXT,payload_json TEXT NOT NULL);
    """)
    buffers = defaultdict(list)
    counts, classes, issue_counts, issue_fields, shapes, relation_counts = (Counter() for _ in range(6))
    claim_issues = JsonlWriter(output / "CLAIM_RECORD_ISSUES.jsonl")
    entity_issues = JsonlWriter(output / "ENTITY_RECORD_ISSUES.jsonl")
    edge_issues = JsonlWriter(output / "EDGE_RECORD_ISSUES.jsonl")
    witnesses, witness_counts = [], Counter()

    def flush(table: str):
        if buffers[table]:
            placeholders = ",".join("?" for _ in buffers[table][0])
            db.executemany(f"INSERT INTO {table} VALUES({placeholders})", buffers[table])
            buffers[table].clear()

    def insert(table: str, row):
        buffers[table].append(row)
        if len(buffers[table]) >= 5000:
            flush(table)

    progress(output, "SCAN_GRAPH_AND_CLAIM_READER")
    with hashed_reader(Path(inputs["baseline"]["graph"]["path"])) as (reader, digest):
        for kind, record_id, record in walk_graph(reader):
            if kind == "metadata":
                continue
            counts[kind] += 1
            if kind == "node":
                category = node_class(record_id, record)
                classes[category] += 1
                md = record.get("metadata") or {}
                name = str(record.get("preferred_name") or "")
                external = record.get("external_ids") or {}
                cui = str(external.get("UMLS_CUI") or "").removeprefix("CUI:")
                if not cui and record_id.startswith("CUI:"):
                    cui = record_id[4:]
                brief = {"id": record_id, "preferred_name": name, "aliases": record.get("aliases") or [],
                         "domain_tags": record.get("domain_tags") or [], "semantic_types": record.get("semantic_types") or [],
                         "source_vocab": record.get("source_vocab"), "external_ids": external,
                         "metadata": {"atom_types": md.get("atom_types") or []}}
                insert("nodes", (record_id, category, name, normalize_term(name), cui,
                                 md.get("source_mention_id"), compact(brief)))
                issues = []
                if not name.strip():
                    issues.append({"code": "empty_node_preferred_name"})
                if not record.get("domain_tags"):
                    issues.append({"code": "empty_node_domain_tags"})
                if category == "claim":
                    summary = claim_summary(record_id, record)
                    paper = summary["source_paper"]
                    paper_key = ("pmid:" + str(paper["pmid"])) if paper.get("pmid") else (("doi:" + str(paper["doi"])) if paper.get("doi") else "")
                    insert("claims", (record_id, summary["subject_id"], summary["object_id"], summary["predicate"],
                                      compact(summary.get("negated", False)), paper_key, compact(summary)))
                    issues.extend(claim_record_checks(record_id, record))
                    shapes["evidence:" + type(md.get("evidence")).__name__] += 1
                    if issues:
                        claim_issues.add({"node_id": record_id, "issues": issues})
                    for code in {row["code"] for row in issues}:
                        if witness_counts[code] < 5:
                            witnesses.append({"issue_code": code, "node_id": record_id, "record": record})
                            witness_counts[code] += 1
                else:
                    issues.extend(entity_identifier_checks(record_id, record))
                    if issues:
                        entity_issues.add({"node_id": record_id, "node_class": category, "issues": issues})
                issue_counts.update({row["code"] for row in issues})
                issue_fields.update(row["code"] + ":" + row["field"] for row in issues if row.get("field"))
                if counts[kind] % 250000 == 0:
                    for table in ("nodes", "claims"):
                        flush(table)
                    db.commit()
                    progress(output, "SCAN_NODES", nodes=counts[kind], claims=classes["claim"])
            else:
                md = record.get("metadata") or {}
                s, t, relation = (record.get(key) for key in ("source_id", "target_id", "relation_type"))
                confidence = record.get("confidence")
                payload = compact(record)
                insert("edges", (int(record_id), s, t, relation, confidence if is_number(confidence) else None,
                                 md.get("claim_id"), compact(md.get("negated")) if "negated" in md else None,
                                 record.get("source"), hashlib.sha256(payload.encode("utf-8")).hexdigest(), payload))
                relation_counts[relation] += 1
                issues = []
                if s == t:
                    issues.append({"code": "edge_self_loop_skipped_by_reader"})
                if not is_number(confidence) or not 0 <= confidence <= 1:
                    issues.append({"code": "edge_confidence_not_finite_unit_number", "value": confidence})
                if not s or not t or not relation:
                    issues.append({"code": "edge_required_field_empty"})
                if issues:
                    edge_issues.add({"edge_ordinal": int(record_id), "issues": issues})
                    issue_counts.update({row["code"] for row in issues})
                if counts[kind] % 500000 == 0:
                    flush("edges")
                    db.commit()
                    progress(output, "SCAN_EDGES", edges=counts[kind])
        graph_sha = digest.hexdigest()
    if graph_sha != inputs["baseline"]["graph"]["sha256"]:
        raise ValueError("graph digest differs from accepted candidate")
    for table in ("nodes", "claims", "edges"):
        flush(table)
    db.commit()
    progress(output, "BUILD_READ_ONLY_AUDIT_INDEXES")
    db.executescript("""
        CREATE INDEX edges_pair ON edges(s,t,r);
        CREATE INDEX edges_claim ON edges(claim_id);
        CREATE INDEX claims_pair ON claims(s,t,p);
        CREATE INDEX nodes_cui ON nodes(cui);
        CREATE INDEX nodes_name ON nodes(normalized_name);
    """)
    db.close()
    witness_writer = JsonlWriter(output / "CLAIM_READER_WITNESSES.jsonl")
    for row in witnesses:
        witness_writer.add(row)
    artifacts = {"claim_record_issues": claim_issues.close(), "entity_record_issues": entity_issues.close(),
                 "edge_record_issues": edge_issues.close(), "claim_reader_witnesses": witness_writer.close()}
    unchanged_inputs(inputs)
    result = {"status": "GRAPH_SCANNED", "completed_at": utc_now(), "graph_sha256": graph_sha,
              "counts": dict(counts), "node_classes": dict(classes), "relation_counts": dict(relation_counts),
              "issue_node_or_edge_counts": dict(issue_counts), "issue_field_counts": dict(issue_fields),
              "claim_value_shapes": dict(shapes), "artifacts": artifacts,
              "source_graph_modified": False, "kge_checked": False}
    if counts != {"node": inputs["baseline"]["counts"]["nodes"], "edge": inputs["baseline"]["counts"]["edges"]}:
        raise ValueError("record count mismatch")
    atomic_json(output / "SCAN_COMPLETE.json", result)
    progress(output, "SCAN_COMPLETE", counts=dict(counts))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    scan(args.output.resolve())


if __name__ == "__main__":
    main()
