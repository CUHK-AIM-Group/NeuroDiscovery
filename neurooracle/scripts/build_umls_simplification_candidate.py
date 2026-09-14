"""Build a lossless UMLS simplification candidate without changing formal KG files.

Unmapped, edge-free CLM_ATOM records move to an indexed SQLite detail store.
Retained UMLS records keep runtime metadata and a resolvable audit reference.
Existing concepts, claims, edges and their evidence are verified exactly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import io
import json
import os
import sqlite3
import sys
from array import array
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "neurooracle/scripts"))
sys.path.insert(0, str(REPO / "neurooracle/src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from streaming_graph_json import IncrementalJsonReader
from umls_audit_store import UmlsAuditStore, compact_record
from umls_mention_mapping import normalize_term

FORMAL = REPO / "neurooracle/data/full_v2"
UMLS_ROOT = REPO / "neurooracle/data/umls_mapping/umls_2026AA_atomic_mentions_v1_20260906"
COMPLETION = UMLS_ROOT / "formal_apply_v1_20260906/FORMAL_APPLY_COMPLETE.json"
DEFAULT_OUTPUT = UMLS_ROOT / "simplification_candidate_v1_20260906"
MAPPING_SOURCE = "UMLS_2026AA_normalized_exact_atomic_mention_alignment"
TARGET_SOURCE = "UMLS_2026AA"
SCHEMA = "umls.simplification_candidate.v1"
DIGEST_GROUPS = (
    "original_nodes", "claim_nodes", "core_atoms", "offloaded_atoms", "new_cuis",
    "original_edges", "umls_edges",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest_update(digest, value: object) -> None:
    digest.update(compact(value).encode("utf-8"))
    digest.update(b"\n")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cheap(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def progress(output: Path, phase: str, **details) -> None:
    value = {"status": "RUNNING", "phase": phase, "updated_at": utc_now(), **details}
    atomic_json(output / "RUN_STATE.json", value)
    print(compact(value), flush=True)


class HashingRaw(io.RawIOBase):
    def __init__(self, path: Path):
        super().__init__()
        self.handle = path.open("rb", buffering=0)
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        length = self.handle.readinto(buffer)
        if length:
            self.digest.update(memoryview(buffer)[:length])
        return length

    def close(self):
        self.handle.close()
        super().close()


@contextmanager
def hashed_reader(path: Path):
    raw = HashingRaw(path)
    with io.TextIOWrapper(io.BufferedReader(raw, 1024 * 1024), encoding="utf-8") as handle:
        yield IncrementalJsonReader(handle), raw.digest


def walk_graph(reader: IncrementalJsonReader):
    reader.expect("{")
    seen = set()
    while reader.peek() != "}":
        key = reader.value()
        if key in seen:
            raise ValueError(f"duplicate top-level key: {key}")
        seen.add(key)
        reader.expect(":")
        if key == "metadata":
            yield "metadata", "", reader.value()
        elif key == "concepts":
            reader.expect("{")
            while reader.peek() != "}":
                node_id = reader.value()
                reader.expect(":")
                node = reader.value()
                if node.get("id") != node_id:
                    raise ValueError(f"node key/id mismatch: {node_id}")
                yield "node", node_id, node
                if reader.peek() == "}":
                    break
                reader.expect(",")
            reader.expect("}")
        elif key == "edges":
            reader.expect("[")
            ordinal = 0
            while reader.peek() != "]":
                ordinal += 1
                yield "edge", str(ordinal), reader.value()
                if reader.peek() == "]":
                    break
                reader.expect(",")
            reader.expect("]")
        else:
            raise ValueError(f"unexpected top-level key: {key}")
        if reader.peek() == "}":
            break
        reader.expect(",")
    reader.expect("}")
    reader.skip_whitespace()
    if seen != {"metadata", "concepts", "edges"} or not reader.eof or reader.position != len(reader.buffer):
        raise ValueError("incomplete graph or trailing content")


class Components:
    def __init__(self):
        self.ids = {}
        self.parents = array("I")
        self.sizes = array("I")
        self.count = 0

    def add(self, node_id):
        if node_id in self.ids:
            raise ValueError(f"duplicate core node: {node_id}")
        index = len(self.parents)
        self.ids[node_id] = index
        self.parents.append(index)
        self.sizes.append(1)
        self.count += 1

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, source, target):
        if source not in self.ids or target not in self.ids:
            raise ValueError(f"edge touches an offloaded or absent node: {source} -> {target}")
        left, right = self.root(self.ids[source]), self.root(self.ids[target])
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parents[right] = left
        self.sizes[left] += self.sizes[right]
        self.count -= 1


class DetailWriter:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript("""
            CREATE TABLE atoms(ordinal INTEGER PRIMARY KEY,record_id TEXT NOT NULL,
                source_mention_id TEXT NOT NULL,mapping_status TEXT NOT NULL,
                in_core INTEGER NOT NULL,payload_json TEXT NOT NULL);
            CREATE TABLE cuis(ordinal INTEGER PRIMARY KEY,record_id TEXT NOT NULL,payload_json TEXT NOT NULL);
            CREATE TABLE mappings(ordinal INTEGER PRIMARY KEY,record_id TEXT NOT NULL,
                source_id TEXT NOT NULL,target_id TEXT NOT NULL,review_status TEXT NOT NULL,
                payload_json TEXT NOT NULL);
            CREATE TABLE alignment_candidates(atom_id TEXT NOT NULL,target_id TEXT NOT NULL,
                mapping_status TEXT NOT NULL,shared_domains TEXT NOT NULL,
                decision TEXT NOT NULL DEFAULT 'needs_semantic_review');
            CREATE TABLE info(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
            PRAGMA user_version=1;
        """)
        self.buffers = defaultdict(list)
        self.ordinals = Counter()

    def add(self, table: str, row: tuple) -> None:
        self.buffers[table].append(row)
        if len(self.buffers[table]) >= 5000:
            self.flush(table)

    def flush(self, table: str) -> None:
        rows = self.buffers[table]
        if not rows:
            return
        placeholders = ",".join("?" for _ in rows[0])
        self.connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        self.connection.commit()
        rows.clear()

    def finish(self, metadata: dict) -> None:
        for table in list(self.buffers):
            self.flush(table)
        self.connection.executescript("""
            CREATE UNIQUE INDEX atoms_id ON atoms(record_id);
            CREATE INDEX atoms_parent ON atoms(source_mention_id);
            CREATE UNIQUE INDEX cuis_id ON cuis(record_id);
            CREATE UNIQUE INDEX mappings_id ON mappings(record_id);
            CREATE INDEX mappings_source ON mappings(source_id,review_status);
            CREATE INDEX mappings_target ON mappings(target_id);
            CREATE UNIQUE INDEX alignment_pair ON alignment_candidates(atom_id,target_id);
        """)
        self.connection.execute("INSERT INTO info VALUES (?,?)", ("manifest", compact(metadata)))
        self.connection.commit()
        self.connection.close()


def add_fixture(heaps, node_id, node, per_scope=16):
    metadata = node.get("metadata") or {}
    rank = int.from_bytes(hashlib.sha256(("umls-simplification-v1:" + node_id).encode()).digest()[:8], "big")
    payload = compact(node)
    for scope in metadata.get("claim_case_study_ids") or ["general"]:
        heap = heaps[scope]
        item = (-rank, node_id, payload)
        if len(heap) < per_scope:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def build_candidate(source: Path, output: Path, expected_sha: str | None = None) -> dict:
    source, output = source.resolve(strict=True), output.resolve()
    if output == source.parent or output.is_relative_to(FORMAL.resolve()):
        raise ValueError("candidate must be separate from the formal graph directory")
    output.mkdir(parents=True, exist_ok=False)
    writer = DetailWriter(output / "umls_details.sqlite")
    try:
        return _build_candidate(source, output, expected_sha, writer)
    finally:
        writer.connection.close()


def _build_candidate(source: Path, output: Path, expected_sha: str | None, writer: DetailWriter) -> dict:
    start_fingerprint = cheap(source)
    progress(output, "BUILD_CORE_AND_DETAIL_STORE")
    counts, domains, sources, relations = (Counter() for _ in range(4))
    digests = {key: hashlib.sha256() for key in DIGEST_GROUPS}
    field_sets = defaultdict(set)
    old_names, old_domains = defaultdict(set), {}
    alignment_counts, atom_statuses = Counter(), Counter()
    same_parent = Counter()
    components = Components()
    fixture_heaps = defaultdict(list)
    original_metadata = None
    first_node, first_edge, edges_started, atomic_started = True, True, False, False
    body_path = output / "core_body.tmp"
    with body_path.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as body:
        body.write('"concepts":{')
        with hashed_reader(source) as (reader, source_hash):
            for kind, node_id, record in walk_graph(reader):
                if kind == "metadata":
                    original_metadata = record
                    continue
                metadata = record.get("metadata") or {}
                if kind == "node":
                    if edges_started:
                        raise ValueError("source concepts must precede edges")
                    counts["source_nodes"] += 1
                    field_sets["source_node_metadata"].update(metadata)
                    is_atom = node_id.startswith("CLM_ATOM:")
                    is_cui = record.get("source_vocab") == TARGET_SOURCE
                    if is_atom:
                        atomic_started = True
                        status = metadata.get("mapping_status")
                        if status not in {"unmapped", "auto_accepted_exact", "needs_review"}:
                            raise ValueError(f"unsupported mapping status: {node_id}")
                        if metadata.get("biomedical_eligible") is not True:
                            raise ValueError(f"ineligible atom in formal source: {node_id}")
                        in_core = status != "unmapped"
                        if in_core != (int(metadata.get("mapping_count", 0)) > 0):
                            raise ValueError(f"inconsistent mapping count: {node_id}")
                        parent = str(metadata.get("source_mention_id") or "")
                        if parent not in components.ids or not parent.startswith("CLM_CONCEPT:"):
                            raise ValueError(f"atom source mention absent from core: {node_id}")
                        atom_statuses[status] += 1
                        table = "atoms"
                        writer.ordinals[table] += 1
                        writer.add(table, (writer.ordinals[table], node_id, parent, status, int(in_core), compact(record)))
                        group = "core_atoms" if in_core else "offloaded_atoms"
                        counts[group] += 1
                        digest_update(digests[group], record)
                        label = normalize_term(record.get("preferred_name") or "")
                        if label == normalize_term(metadata.get("source_text") or ""):
                            same_parent[status] += 1
                        matched = False
                        for target_id in sorted(old_names.get(label, ())):
                            shared = sorted(set(record.get("domain_tags") or []) & old_domains[target_id])
                            if shared:
                                writer.add("alignment_candidates", (node_id, target_id, status, compact(shared), "needs_semantic_review"))
                                counts["alignment_pairs"] += 1
                                matched = True
                        if matched:
                            alignment_counts[status] += 1
                        if not in_core:
                            if counts["source_nodes"] % 250000 == 0:
                                progress(output, "BUILD_NODES", nodes=counts["source_nodes"], offloaded=counts["offloaded_atoms"])
                            continue
                        projected = compact_record(record, "atoms", node_id)
                    elif is_cui:
                        if atomic_started:
                            raise ValueError("CUI targets unexpectedly follow atomic records")
                        writer.ordinals["cuis"] += 1
                        writer.add("cuis", (writer.ordinals["cuis"], node_id, compact(record)))
                        counts["new_cuis"] += 1
                        digest_update(digests["new_cuis"], record)
                        projected = compact_record(record, "cuis", node_id)
                    else:
                        if atomic_started:
                            raise ValueError("original concepts unexpectedly follow atomic records")
                        counts["original_nodes"] += 1
                        digest_update(digests["original_nodes"], record)
                        if node_id.startswith("CLM:"):
                            counts["claim_nodes"] += 1
                            digest_update(digests["claim_nodes"], record)
                            add_fixture(fixture_heaps, node_id, record)
                        elif node_id.startswith("CLM_CONCEPT:"):
                            counts["original_mentions"] += 1
                        else:
                            old_domains[node_id] = set(record.get("domain_tags") or [])
                            for label in [record.get("preferred_name") or "", *(record.get("aliases") or [])]:
                                label = normalize_term(label)
                                if label:
                                    old_names[label].add(node_id)
                        projected = record
                    components.add(node_id)
                    counts["core_nodes"] += 1
                    domains.update(projected.get("domain_tags") or [])
                    sources[str(projected.get("source_vocab") or "")] += 1
                    field_sets["core_node_metadata"].update(projected.get("metadata") or {})
                    if not first_node:
                        body.write(",")
                    first_node = False
                    body.write(compact(node_id) + ":" + compact(projected))
                    if counts["source_nodes"] % 250000 == 0:
                        progress(output, "BUILD_NODES", nodes=counts["source_nodes"], offloaded=counts["offloaded_atoms"])
                else:
                    if not edges_started:
                        body.write('},"edges":[')
                        edges_started = True
                    counts["source_edges"] += 1
                    field_sets["source_edge_metadata"].update(metadata)
                    source_id, target_id = record["source_id"], record["target_id"]
                    components.union(source_id, target_id)
                    if record.get("source") == MAPPING_SOURCE:
                        counts["umls_edges"] += 1
                        ordinal = counts["umls_edges"]
                        writer.ordinals["mappings"] = ordinal
                        record_id = f"UMLS_MAP:{ordinal:09d}"
                        writer.add("mappings", (ordinal, record_id, source_id, target_id, str(metadata.get("review_status") or ""), compact(record)))
                        digest_update(digests["umls_edges"], record)
                        projected = compact_record(record, "mappings", record_id)
                    else:
                        counts["original_edges"] += 1
                        digest_update(digests["original_edges"], record)
                        projected = record
                    field_sets["core_edge_metadata"].update(projected.get("metadata") or {})
                    relations[str(projected.get("relation_type") or "")] += 1
                    if not first_edge:
                        body.write(",")
                    first_edge = False
                    body.write(compact(projected))
                    if counts["source_edges"] % 500000 == 0:
                        progress(output, "BUILD_EDGES", edges=counts["source_edges"])
            source_sha256 = source_hash.hexdigest()
        if not edges_started:
            body.write('},"edges":[')
        body.write("]}")
    if cheap(source) != start_fingerprint:
        raise ValueError("formal source changed during candidate construction")
    if expected_sha and source_sha256 != expected_sha:
        raise ValueError("formal source SHA-256 differs from the trusted completed release")
    if original_metadata is None:
        raise ValueError("source metadata missing")
    original_stats = original_metadata.get("stats") or {}
    if original_stats.get("n_concepts") != counts["source_nodes"] or original_stats.get("n_edges") != counts["source_edges"]:
        raise ValueError("source node/edge statistics are stale")
    if components.count + counts["offloaded_atoms"] != original_stats.get("connected_components"):
        raise ValueError("independent component count disagrees with formal source baseline")
    core_components = components.count
    del components, old_names, old_domains
    gc.collect()
    source_fp = {**start_fingerprint, "sha256": source_sha256}
    build = {
        "schema_version": SCHEMA,
        "status": "BUILT_AWAITING_VALIDATION",
        "created_at": utc_now(),
        "source": source_fp,
        "counts": dict(counts),
        "source_connected_components": original_stats["connected_components"],
        "core_connected_components": core_components,
        "digests": {key: value.hexdigest() for key, value in digests.items()},
        "metadata_fields": {key: sorted(value) for key, value in field_sets.items()},
        "atom_statuses": dict(atom_statuses),
        "same_surface_as_parent": dict(same_parent),
        "alignment_candidate_atoms_by_status": dict(alignment_counts),
        "semantically_approved_merges": 0,
        "formal_files_modified": False,
    }
    progress(output, "INDEX_DETAIL_STORE", records=sum(writer.ordinals.values()))
    writer.finish({"schema_version": SCHEMA, "source": source_fp, "counts": dict(counts)})
    metadata = dict(original_metadata)
    metadata["stats"] = {
        **original_stats,
        "n_concepts": counts["core_nodes"],
        "n_edges": counts["source_edges"],
        "domains": dict(sorted(domains.items())),
        "sources": dict(sorted(sources.items())),
        "relations": dict(sorted(relations.items())),
        "connected_components": core_components,
    }
    metadata["umls_simplification_candidate"] = {
        "schema_version": SCHEMA,
        "status": "READ_ONLY_CANDIDATE",
        "source_sha256": source_sha256,
        "detail_store": "umls_details.sqlite",
        "offloaded_unmapped_atoms": counts["offloaded_atoms"],
        "audit_reference_format": "table/record_id",
        "detail_store_reader": "neurooracle.src.umls_audit_store.UmlsAuditStore",
        "all_edge_payloads_recoverable": True,
        "semantic_merges_applied": 0,
    }
    progress(output, "ASSEMBLE_CORE_GRAPH", nodes=counts["core_nodes"], edges=counts["source_edges"])
    graph_path = output / "knowledge_graph.candidate.json"
    write_hash = hashlib.sha256()
    with graph_path.open("xb", buffering=1024 * 1024) as target:
        header = ('{"metadata":' + compact(metadata) + ",").encode("utf-8")
        target.write(header)
        write_hash.update(header)
        with body_path.open("rb") as body:
            for chunk in iter(lambda: body.read(16 * 1024 * 1024), b""):
                target.write(chunk)
                write_hash.update(chunk)
        target.flush()
        os.fsync(target.fileno())
    body_path.unlink()  # Only the temporary body created in this candidate directory.
    build["candidate"] = {**cheap(graph_path), "sha256": write_hash.hexdigest()}
    fixtures = {}
    for scope, heap in fixture_heaps.items():
        for _, node_id, payload in heap:
            fixture = fixtures.setdefault(node_id, {"node": json.loads(payload), "scopes": []})
            fixture["scopes"].append(scope)
    atomic_json(output / "EVIDENCE_FIXTURE.json", {
        "selection": "16 lowest SHA-256 ranks per nonexclusive claim task, seed umls-simplification-v1",
        "source_sha256": source_sha256,
        "scopes": sorted(fixture_heaps),
        "records": [fixtures[key] for key in sorted(fixtures)],
    })
    atomic_json(output / "BUILD_COMPLETE.json", build)
    return build


def validate_candidate(output: Path, build: dict) -> dict:
    progress(output, "VALIDATE_DETAIL_STORE")
    expected = build["digests"]
    observed = {key: hashlib.sha256() for key in DIGEST_GROUPS}
    projections = {key: hashlib.sha256() for key in ("atoms", "cuis", "mappings")}
    db_counts = Counter()
    with UmlsAuditStore(output / "umls_details.sqlite") as store:
        if store.connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed")
        for table in ("atoms", "cuis", "mappings"):
            for record_id, payload in store.connection.execute(f"SELECT record_id,payload_json FROM {table} ORDER BY ordinal"):
                record = json.loads(payload)
                db_counts[table] += 1
                if table == "atoms":
                    in_core = (record.get("metadata") or {}).get("mapping_status") != "unmapped"
                    group = "core_atoms" if in_core else "offloaded_atoms"
                    if record.get("id") != record_id:
                        raise ValueError("atom detail identity mismatch")
                elif table == "cuis":
                    in_core, group = True, "new_cuis"
                    if record.get("id") != record_id:
                        raise ValueError("CUI detail identity mismatch")
                else:
                    in_core, group = True, "umls_edges"
                digest_update(observed[group], record)
                if in_core:
                    digest_update(projections[table], compact_record(record, table, record_id))
                if db_counts[table] % 500000 == 0:
                    progress(output, "VALIDATE_DETAILS", table=table, records=db_counts[table])
        # Query columns must agree with the lossless payload as well as its hashes.
        query_column_errors = store.connection.execute("""
            SELECT COUNT(*) FROM atoms WHERE record_id!=json_extract(payload_json,'$.id')
              OR source_mention_id!=json_extract(payload_json,'$.metadata.source_mention_id')
              OR mapping_status!=json_extract(payload_json,'$.metadata.mapping_status')
              OR in_core!=(mapping_status!='unmapped')
        """).fetchone()[0]
        query_column_errors += store.connection.execute("""
            SELECT COUNT(*) FROM mappings WHERE source_id!=json_extract(payload_json,'$.source_id')
              OR target_id!=json_extract(payload_json,'$.target_id')
              OR review_status!=json_extract(payload_json,'$.metadata.review_status')
        """).fetchone()[0]
        if query_column_errors:
            raise ValueError("detail query columns disagree with payloads")
    for group in ("core_atoms", "offloaded_atoms", "new_cuis", "umls_edges"):
        if observed[group].hexdigest() != expected[group]:
            raise ValueError(f"lossless detail digest mismatch: {group}")
    progress(output, "VALIDATE_CORE_GRAPH")
    actual_projections = {key: hashlib.sha256() for key in projections}
    node_ids, counts = set(), Counter()
    domains, sources, relations = Counter(), Counter(), Counter()
    graph_metadata = None
    fixture = read_json(output / "EVIDENCE_FIXTURE.json")
    fixture_expected = {row["node"]["id"]: row["node"] for row in fixture["records"]}
    fixture_found = {}
    with hashed_reader(output / "knowledge_graph.candidate.json") as (reader, graph_hash):
        for kind, node_id, record in walk_graph(reader):
            if kind == "metadata":
                graph_metadata = record
                continue
            reference = (record.get("metadata") or {}).get("audit_ref")
            if kind == "node":
                if node_id in node_ids:
                    raise ValueError(f"duplicate candidate node: {node_id}")
                node_ids.add(node_id)
                counts["nodes"] += 1
                domains.update(record.get("domain_tags") or [])
                sources[str(record.get("source_vocab") or "")] += 1
                if node_id.startswith("CLM_ATOM:"):
                    if (record.get("metadata") or {}).get("mapping_status") == "unmapped":
                        raise ValueError("unmapped atom remains in candidate graph")
                    if reference != "atoms/" + node_id:
                        raise ValueError("invalid atom audit reference")
                    digest_update(actual_projections["atoms"], record)
                    counts["core_atoms"] += 1
                elif record.get("source_vocab") == TARGET_SOURCE:
                    if reference != "cuis/" + node_id:
                        raise ValueError("invalid CUI audit reference")
                    digest_update(actual_projections["cuis"], record)
                    counts["new_cuis"] += 1
                else:
                    digest_update(observed["original_nodes"], record)
                    counts["original_nodes"] += 1
                    if node_id.startswith("CLM:"):
                        digest_update(observed["claim_nodes"], record)
                        counts["claim_nodes"] += 1
                    if node_id in fixture_expected:
                        if record != fixture_expected[node_id]:
                            raise ValueError("fixed evidence fixture changed")
                        fixture_found[node_id] = record
                if counts["nodes"] % 500000 == 0:
                    progress(output, "VALIDATE_CORE_NODES", nodes=counts["nodes"])
            else:
                if record["source_id"] not in node_ids or record["target_id"] not in node_ids:
                    raise ValueError("dangling candidate edge")
                counts["edges"] += 1
                relations[str(record.get("relation_type") or "")] += 1
                if record.get("source") == MAPPING_SOURCE:
                    digest_update(actual_projections["mappings"], record)
                    counts["umls_edges"] += 1
                else:
                    digest_update(observed["original_edges"], record)
                    counts["original_edges"] += 1
        graph_sha = graph_hash.hexdigest()
    for group in ("original_nodes", "claim_nodes", "original_edges"):
        if observed[group].hexdigest() != expected[group]:
            raise ValueError(f"preserved source content changed: {group}")
    for table in projections:
        if actual_projections[table].digest() != projections[table].digest():
            raise ValueError(f"core projection does not match recoverable details: {table}")
    stats = graph_metadata["stats"]
    checks = {
        "source_nodes_partition_exact": counts["nodes"] + build["counts"]["offloaded_atoms"] == build["counts"]["source_nodes"],
        "node_count": counts["nodes"] == stats["n_concepts"] == build["counts"]["core_nodes"],
        "edge_count": counts["edges"] == stats["n_edges"] == build["counts"]["source_edges"],
        "claim_count": counts["claim_nodes"] == build["counts"]["claim_nodes"],
        "atom_store_complete": db_counts["atoms"] == build["counts"]["core_atoms"] + build["counts"]["offloaded_atoms"],
        "cui_store_complete": db_counts["cuis"] == counts["new_cuis"],
        "mapping_store_complete": db_counts["mappings"] == counts["umls_edges"],
        "domains_exact": dict(domains) == stats["domains"],
        "sources_exact": dict(sources) == stats["sources"],
        "relations_exact": dict(relations) == stats["relations"],
        "connected_components": stats["connected_components"] == build["core_connected_components"],
        "candidate_bytes_hash": graph_sha == build["candidate"]["sha256"],
        "fixed_evidence_fixture": set(fixture_expected) == set(fixture_found),
    }
    # Every source mention and matching candidate target remains resolvable.
    with UmlsAuditStore(output / "umls_details.sqlite") as store:
        for (parent,) in store.connection.execute("SELECT DISTINCT source_mention_id FROM atoms"):
            if parent not in node_ids:
                raise ValueError("detail source mention is absent from core")
        for (target,) in store.connection.execute("SELECT DISTINCT target_id FROM alignment_candidates"):
            if target not in node_ids:
                raise ValueError("alignment candidate target is absent from core")
    if not all(checks.values()):
        raise ValueError(f"candidate checks failed: {checks}")
    result = {
        "status": "PASS", "checks": checks, "counts": dict(counts),
        "candidate_sha256": graph_sha, "core_connected_components": stats["connected_components"],
        "detail_payloads_exact": True, "all_audit_references_resolve": True,
        "original_nodes_and_edges_exact": True,
        "fixed_evidence_fixture_count": len(fixture_found),
        "fixed_evidence_fixture_scopes": fixture["scopes"],
    }
    atomic_json(output / "VALIDATION.json", result)
    atomic_json(output / "EVIDENCE_FIXTURE_CANDIDATE.json", {
        **fixture, "records": [{**row, "node": fixture_found[row["node"]["id"]]} for row in fixture["records"]],
    })
    return result


def summarize_triples(triples):
    digest = hashlib.sha256()
    relations = Counter()
    for triple in triples:
        digest_update(digest, triple.as_tuple())
        relations[triple.relation_type] += 1
    return {"count": len(triples), "ordered_sha256": digest.hexdigest(), "relations": dict(relations)}


def kge_input_regression(graph_path: Path, output: Path, label: str) -> dict:
    from neurooracle.src.kge.triple_loader import load_triples_from_kg, split_triples
    progress(output, "KGE_READER_REGRESSION", graph=label)
    triples, domains = load_triples_from_kg(graph_path)
    result = {"all_triples": summarize_triples(triples), "node_domain_entries": len(domains), "splits": {}}
    for seed in (0, 42):
        progress(output, "KGE_FIXED_SPLIT_REGRESSION", graph=label, seed=seed)
        train, val, test = split_triples(triples, domains, seed=seed)
        result["splits"][str(seed)] = {
            "train": summarize_triples(train), "validation": summarize_triples(val), "test": summarize_triples(test),
        }
        del train, val, test
    atomic_json(output / f"KGE_{label.upper()}_INPUTS.json", result)
    return result


def evidence_scores(fixture: dict) -> dict:
    from neurooracle.src.graph_manager import KnowledgeGraph
    from neurooracle.src.hypothesis_engine import HypothesisEngine, HypothesisLink
    from neurooracle.src.schema import ConceptNode
    kg = KnowledgeGraph()
    for row in fixture["records"]:
        kg.add_concept(ConceptNode.from_dict(row["node"]))
    engine = HypothesisEngine(kg)
    scores = {}
    for row in fixture["records"]:
        node = row["node"]
        md = node.get("metadata") or {}
        link = HypothesisLink(
            from_id=str(md.get("subject_id") or ""), from_name=str(md.get("subject_name") or ""),
            to_id=str(md.get("object_id") or ""), to_name=str(md.get("object_name") or ""),
            relation_type=str(md.get("predicate") or ""), confidence=float(md.get("confidence") or 0.0),
            claim_id=node["id"], raw_text=str(md.get("raw_text") or ""),
            evidence=md.get("evidence") or {}, source_paper=md.get("source_paper") or {},
        )
        scores[node["id"]] = {
            "evidence_score": engine._compute_evidence_score([link]),
            "confidence_score": engine._compute_confidence_score([link]),
            "temporal_decay": engine.compute_temporal_decay(md),
        }
    return scores


def run_regression(source: Path, output: Path) -> dict:
    before = kge_input_regression(source, output, "source")
    gc.collect()
    after = kge_input_regression(output / "knowledge_graph.candidate.json", output, "candidate")
    checks = {
        "all_kge_triples_exact": before["all_triples"] == after["all_triples"],
        "fixed_seed_train_val_test_exact": before["splits"] == after["splits"],
    }
    progress(output, "FIXED_EVIDENCE_SCORER_REGRESSION")
    source_fixture = read_json(output / "EVIDENCE_FIXTURE.json")
    candidate_fixture = read_json(output / "EVIDENCE_FIXTURE_CANDIDATE.json")
    before_scores, after_scores = evidence_scores(source_fixture), evidence_scores(candidate_fixture)
    checks["fixed_evidence_scores_exact"] = before_scores == after_scores
    if not all(checks.values()):
        raise ValueError(f"experiment reader regression failed: {checks}")
    with UmlsAuditStore(output / "umls_details.sqlite") as store:
        policy_counts = dict(store.connection.execute("SELECT review_status,COUNT(*) FROM mappings GROUP BY review_status"))
    result = {
        "status": "PASS", "checks": checks,
        "kge_triples": before["all_triples"]["count"], "kge_split_seeds": [0, 42],
        "source_domain_index_entries": before["node_domain_entries"],
        "candidate_domain_index_entries": after["node_domain_entries"],
        "evidence_fixture_claims": len(before_scores), "evidence_fixture_scopes": source_fixture["scopes"],
        "evidence_scores": before_scores,
        "mapping_status_counts": policy_counts,
        "audit_store_default_mapping_policy": "auto_accepted_exact_only",
        "existing_kge_loader_review_filter_added": False,
        "full_model_training_rerun": False,
        "limit": "Actual KGE reader and two fixed full-input splits; fixed claim evidence/confidence scores. This does not assert identical global task sampling or retrained model performance.",
    }
    atomic_json(output / "EXPERIMENT_REGRESSION.json", result)
    return result


def formal_baseline() -> dict:
    complete = read_json(COMPLETION)
    if complete.get("status") != "APPLIED_AND_POST_VERIFIED":
        raise ValueError("UMLS formal completion is not verified")
    records = complete["formal_sources_after"]
    for name, record in records.items():
        actual = cheap(Path(record["path"]))
        if any(actual[key] != record[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError(f"formal baseline changed: {name}")
    return records


def make_report(output: Path, build: dict, validation: dict, regression: dict, artifacts: dict) -> str:
    n = lambda value: f"{value:,}"
    c = build["counts"]
    fields = build["metadata_fields"]
    return f"""# UMLS 精简候选 v1：验收完成

状态：只读候选已生成并冻结。正式图及 claims、CURRENT_STATE、README 均保持原指纹。

| 项目 | 正式图 | 精简候选 |
|---|---:|---:|
| 节点 | {n(c['source_nodes'])} | {n(c['core_nodes'])} |
| 边 | {n(c['source_edges'])} | {n(c['source_edges'])} |
| 连通分量 | {n(build['source_connected_components'])} | {n(build['core_connected_components'])} |
| 节点 metadata 一级字段种类 | {len(fields['source_node_metadata'])} | {len(fields['core_node_metadata'])} |
| 边 metadata 一级字段种类 | {len(fields['source_edge_metadata'])} | {len(fields['core_edge_metadata'])} |
| 主图字节数 | {n(build['source']['bytes'])} | {n(build['candidate']['bytes'])} |

## 精简内容

- {n(c['offloaded_atoms'])} 个未映射、无图边的原子及其完整来源记录移到 umls_details.sqlite。
- {n(c['core_atoms'])} 个已映射原子、{n(c['new_cuis'])} 个新增 CUI、{n(c['umls_edges'])} 条映射边保留在主图；详细 metadata 可经 audit_ref 精确恢复。
- 原始节点、claim 与非 UMLS 边逐条语义摘要一致；全部映射边的完整原始内容也可恢复。
- 新旧名称/别名与共同领域的对应候选包含 {n(c['alignment_pairs'])} 对；尚未实施任何语义合并或待审核映射晋级。
- 原子相对原 mention 的规范化文本相同数量为 {n(sum(build['same_surface_as_parent'].values()))}；完整对应记录保留用于下一轮语义核对。
- 核心原子 metadata 保留 source_mention_id、mapping_status、mapping_count、evidence_span、audit_ref；映射边保留 method、review_status、semantic_compatibility、audit_ref。
- SQLite 是候选的必要组成部分。主图体积减少是记录布局调整，不代表总存储量减少或审计数据被删除。

## 验收

- 完整正式源 SHA-256 与 UMLS 完成证据一致；候选完整读取、哈希与 SQLite integrity_check 均通过。
- 全图 {n(validation['counts']['edges'])} 条边无悬空端点；源/候选内容分区与可恢复详细记录逐项一致。
- 实际 KGE 读取器得到 {n(regression['kge_triples'])} 条完全相同的有序三元组；种子 0、42 的全量训练/验证/测试划分完全一致。
- 固定 {n(regression['evidence_fixture_claims'])} 条 claim、{len(regression['evidence_fixture_scopes'])} 个任务/通用范围的原文和证据、置信度评分完全一致。
- 附属表读取 API 默认仅返回 auto_accepted_exact 映射；needs_review 必须显式请求。现有 KGE 读取器的默认审核策略未改动，待审核边仍可能被它读取。
- 未重跑完整模型训练，也未验证所有全局候选抽样器；不得把本次数据与评分回归扩写为全任务最终指标完全一致。

## 文件

- knowledge_graph.candidate.json — 精简主图。
- umls_details.sqlite — 完整 UMLS 记录及新旧节点对应候选，必需。
- VALIDATION.json — 全量结构与可恢复性校验。
- EXPERIMENT_REGRESSION.json — 实际读取器、固定划分及评分回归。
- CANDIDATE_FREEZE.json — 各文件哈希、正式源基线和只读状态。

主图 SHA-256：{artifacts['knowledge_graph.candidate.json']['sha256']}

详细表 SHA-256：{artifacts['umls_details.sqlite']['sha256']}
"""


def finish_candidate(source: Path, output: Path, baseline: dict) -> dict:
    build = read_json(output / "BUILD_COMPLETE.json")
    validation = validate_candidate(output, build)
    regression = run_regression(source, output)
    if formal_baseline() != baseline:
        raise ValueError("formal files changed during candidate validation")
    progress(output, "FREEZE_CANDIDATE")
    artifacts = {}
    for name in (
        "knowledge_graph.candidate.json", "umls_details.sqlite", "BUILD_COMPLETE.json", "VALIDATION.json",
        "EVIDENCE_FIXTURE.json", "EVIDENCE_FIXTURE_CANDIDATE.json", "KGE_SOURCE_INPUTS.json",
        "KGE_CANDIDATE_INPUTS.json", "EXPERIMENT_REGRESSION.json",
    ):
        path = output / name
        sha = validation["candidate_sha256"] if name == "knowledge_graph.candidate.json" else file_sha(path)
        artifacts[name] = {**cheap(path), "sha256": sha}
    report = make_report(output, build, validation, regression, artifacts)
    (output / "CANDIDATE_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    artifacts["CANDIDATE_REPORT.md"] = {
        **cheap(output / "CANDIDATE_REPORT.md"), "sha256": file_sha(output / "CANDIDATE_REPORT.md"),
    }
    freeze = {
        "schema_version": SCHEMA, "status": "READ_ONLY_CANDIDATE_VALIDATED",
        "frozen_at": utc_now(), "formal_sources": baseline,
        "formal_files_modified": False, "formal_apply_performed": False,
        "counts": build["counts"], "metadata_fields": build["metadata_fields"],
        "validation": {key: value for key, value in validation.items() if key != "counts"},
        "experiment_regression": {key: value for key, value in regression.items() if key != "evidence_scores"},
        "artifacts": artifacts,
    }
    atomic_json(output / "CANDIDATE_FREEZE.json", freeze)
    atomic_json(output / "RUN_STATE.json", {
        "status": "COMPLETE", "phase": "READ_ONLY_CANDIDATE_VALIDATED", "updated_at": utc_now(),
        "freeze_path": str(output / "CANDIDATE_FREEZE.json"),
    })
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "lookup"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", help="atoms/ID, cuis/ID or mappings/ID for lookup")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "lookup":
        with UmlsAuditStore(output / "umls_details.sqlite") as store:
            print(json.dumps(store.fetch(args.reference or ""), ensure_ascii=False, indent=2))
        return
    baseline = formal_baseline()
    source = Path(baseline["knowledge_graph"]["path"])
    if output == source.parent or output.is_relative_to(FORMAL.resolve()):
        raise ValueError("candidate must be separate from the formal graph directory")
    if args.command == "build" and output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    try:
        if args.command == "build":
            build_candidate(source, output, baseline["knowledge_graph"]["sha256"])
        else:
            if (output / "CANDIDATE_FREEZE.json").exists():
                raise ValueError("candidate is already frozen; do not rewrite its evidence")
            if read_json(output / "BUILD_COMPLETE.json")["source"]["sha256"] != baseline["knowledge_graph"]["sha256"]:
                raise ValueError("built candidate belongs to a different formal source")
        result = finish_candidate(source, output, baseline)
        print(compact({"status": result["status"], "counts": result["counts"], "output": str(output)}), flush=True)
    except Exception as exc:
        if output.is_dir() and not (output / "CANDIDATE_FREEZE.json").exists():
            atomic_json(output / "RUN_STATE.json", {"status": "FAILED", "error": repr(exc), "updated_at": utc_now()})
        raise


if __name__ == "__main__":
    main()
