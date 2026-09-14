"""Versioned compatibility repair; frozen source data is always read-only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_kg_integrity import DEFAULT_OUTPUT as AUDIT, JsonlWriter
from build_umls_simplification_candidate import REPO, UMLS_ROOT, atomic_json, cheap, compact, digest_update, file_sha, hashed_reader, progress, read_json, utc_now, walk_graph
from neurooracle.src.claim_compatibility import SCIENTIFIC_FIELDS, has_value, normalize_claim_payload
from neurooracle.src.schema import Claim, Evidence

OUTPUT = UMLS_ROOT / "claim_compatibility_repair_v1_20260907"
ALLOWED_CHANGED = {"neurooracle/src/schema.py", "neurooracle/tests/test_kg_quality_checks.py"}


def fingerprint(path):
    return {**cheap(path), "sha256": file_sha(path)}


def guard(fp):
    current = cheap(Path(fp["path"]))
    if any(current[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
        raise ValueError(f"protected input changed: {fp['path']}")


def check_inputs(output):
    prepared = read_json(output / "INPUTS.json")
    guard(prepared["audit_freeze"])
    audit = read_json(Path(prepared["audit_freeze"]["path"]))
    for group in ("artifacts", "formal_sources", "reused_metadata_sources"):
        for fp in audit[group].values():
            guard(fp)
    for key in ("pointer", "acceptance", "graph", "details"):
        guard(audit["candidate_baseline"][key])
    for name, fp in audit["implementation"].items():
        if name not in ALLOWED_CHANGED:
            guard(fp)
    guard(prepared["schema_before"])
    return prepared, audit


def prepare(output):
    from audit_kg_integrity import unchanged_inputs
    if output.exists():
        raise ValueError("repair directory already exists; never overwrite")
    audit = read_json(AUDIT / "AUDIT_FREEZE.json")
    unchanged_inputs(read_json(AUDIT / "INPUTS.json"))
    for group in ("artifacts", "implementation", "reused_metadata_sources", "formal_sources"):
        for fp in audit[group].values():
            guard(fp)
    output.mkdir()
    # Archive the exact pre-edit implementation as a data artifact for regression.
    shutil.copyfile(REPO / "neurooracle/src/schema.py", output / "schema_before.py")
    before = fingerprint(output / "schema_before.py")
    if before["sha256"] != audit["implementation"]["neurooracle/src/schema.py"]["sha256"]:
        raise ValueError("pre-edit schema differs from audited implementation")
    result = {"prepared_at": utc_now(), "audit_freeze": fingerprint(AUDIT / "AUDIT_FREEZE.json"),
              "schema_before": before, "allowed_changed_existing_files": sorted(ALLOWED_CHANGED),
              "scope": "Claim/Evidence compatibility and separately justified reference corrections in a new candidate only"}
    atomic_json(output / "INPUTS.json", result)
    print(compact(result), flush=True)


def runtime_sources():
    return {name: fingerprint(REPO / name) for name in
            ("neurooracle/src/schema.py", "neurooracle/src/claim_compatibility.py",
             "neurooracle/scripts/repair_kg_claim_compatibility.py")}


def before_claim_class(output):
    name = "neurooracle.src._schema_before_compatibility_repair"
    spec = importlib.util.spec_from_file_location(name, output / "schema_before.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Claim


def node_patch(before, after):
    changes = []
    for key in set(before) | set(after):
        if key != "metadata" and before.get(key) != after.get(key):
            raise ValueError("compatibility repair changed a non-metadata node field")
    left, right = before["metadata"], after["metadata"]
    if set(left) - {"evidence", "metadata"} != set(right) - {"evidence", "metadata"}:
        raise ValueError("unexpected claim field change")
    for key in set(left) | set(right):
        if key not in {"evidence", "metadata"} and left.get(key) != right.get(key):
            raise ValueError("unexpected claim payload change")
    if left.get("evidence") != right.get("evidence"):
        changes.append({"path": ["metadata", "evidence"], "before_present": "evidence" in left,
                        "before": left.get("evidence"), "after": right.get("evidence")})
    old_nested, new_nested = left.get("metadata"), right.get("metadata")
    if old_nested != new_nested:
        if not isinstance(old_nested, dict):
            changes.append({"path": ["metadata", "metadata"], "before_present": "metadata" in left,
                            "before": old_nested, "after": new_nested})
        else:
            for key in set(old_nested) | set(new_nested):
                if old_nested.get(key) != new_nested.get(key) or (key in old_nested) != (key in new_nested):
                    if key not in SCIENTIFIC_FIELDS or has_value(old_nested.get(key)):
                        raise ValueError("repair overwrites a populated nested field")
                    changes.append({"path": ["metadata", "metadata", key], "before_present": key in old_nested,
                                    "before": old_nested.get(key), "after": new_nested.get(key)})
    return {"node_id": before["id"], "before_sha256": hashlib.sha256(compact(before).encode()).hexdigest(),
            "after_sha256": hashlib.sha256(compact(after).encode()).hexdigest(), "changes": changes}


def undo_patch(record, patch):
    restored = deepcopy(record)
    for change in reversed(patch["changes"]):
        parent = restored
        for key in change["path"][:-1]:
            parent = parent[key]
        leaf = change["path"][-1]
        if parent.get(leaf) != change["after"]:
            raise ValueError("patch postimage mismatch")
        if change["before_present"]:
            parent[leaf] = deepcopy(change["before"])
        else:
            del parent[leaf]
    return restored


def assert_compatible_roundtrip(payload):
    original = deepcopy(payload)
    obj = Claim.from_dict(payload)
    encoded = obj.to_dict()
    if payload != original or Claim.from_dict(encoded).to_dict() != encoded:
        raise ValueError("claim reader mutates input or is not stable on roundtrip")
    for key in SCIENTIFIC_FIELDS:
        if key in payload and encoded.get(key) != payload[key]:
            raise ValueError("direct scientific field was not preserved")
        if has_value(payload.get(key)) and not has_value(obj.metadata.get(key)):
            raise ValueError("direct scientific field not carried into object")
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        if any(encoded["evidence"].get(key) != value or key not in encoded["evidence"] for key, value in evidence.items()):
            raise ValueError("structured evidence field was lost or changed")
    elif "evidence" in payload and encoded["evidence"].get("legacy_value") != evidence:
        raise ValueError("legacy evidence was not preserved")
    for key in ("id", "subject_id", "object_id", "subject_name", "object_name", "predicate", "negated", "confidence", "raw_text"):
        if key in payload and encoded[key] != payload[key]:
            raise ValueError("scientific claim content changed")
    for key, value in (payload.get("source_paper") or {}).items():
        if key not in encoded["source_paper"] or encoded["source_paper"][key] != value:
            raise ValueError("source-paper evidence changed")
    return obj, encoded


def build(output):
    _, audit = check_inputs(output)
    code = runtime_sources()
    source = audit["candidate_baseline"]["graph"]
    before_claim = before_claim_class(output)
    plan = read_json(output / "REFERENCE_PLAN.json")
    if plan["plan"]["rows"] != 0:
        raise ValueError("this compatibility build may not apply edge repairs")
    graph_path = output / "knowledge_graph.candidate.json"
    partial = output / "knowledge_graph.candidate.json.partial"
    if graph_path.exists() or partial.exists():
        raise ValueError("candidate graph already exists")
    writer = JsonlWriter(output / "CLAIM_COMPAT_PATCHES.jsonl")
    counts, field_counts = Counter(), Counter()
    digests = {key: hashlib.sha256() for key in ("source_nodes", "candidate_nodes", "edges", "source_claims")}
    graph_digest = hashlib.sha256()
    progress(output, "BUILDING_COMPATIBILITY_CANDIDATE")
    with partial.open("xb", buffering=1024 * 1024) as handle:
        def emit(value):
            data = value.encode("utf-8")
            handle.write(data)
            graph_digest.update(data)
        first_node, first_edge, edges_started = True, True, False
        with hashed_reader(Path(source["path"])) as (reader, source_digest):
            for kind, record_id, record in walk_graph(reader):
                if kind == "metadata":
                    emit('{"metadata":' + compact(record) + ',"concepts":{')
                    continue
                counts[kind] += 1
                if kind == "node":
                    digest_update(digests["source_nodes"], record)
                    repaired = record
                    if record_id.startswith("CLM:"):
                        counts["claims_checked"] += 1
                        md = record["metadata"]
                        digest_update(digests["source_claims"], record)
                        old = before_claim.from_dict(deepcopy(md))
                        try:
                            old.to_dict()
                        except (AttributeError, TypeError):
                            counts["old_reader_reencoder_failures"] += 1
                        assert_compatible_roundtrip(md)
                        patched_md = normalize_claim_payload(md)
                        assert_compatible_roundtrip(patched_md)
                        if patched_md != md:
                            repaired = {**record, "metadata": patched_md}
                            patch = node_patch(record, repaired)
                            if undo_patch(repaired, patch) != record:
                                raise ValueError("in-memory rollback is not exact")
                            writer.add(patch)
                            counts["claim_nodes_changed"] += 1
                        if patched_md.get("evidence") != md.get("evidence"):
                            counts["legacy_evidence_wrapped"] += 1
                        carried = [key for key in SCIENTIFIC_FIELDS if has_value(md.get(key)) and not has_value(old.metadata.get(key))]
                        if carried:
                            counts["claims_scientific_fields_carried"] += 1
                            field_counts.update(carried)
                        evidence = md.get("evidence")
                        if isinstance(evidence, dict):
                            known = set(Evidence.__dataclass_fields__) - {"extra_fields"}
                            if any(has_value(value) and key not in known for key, value in evidence.items()):
                                counts["claims_evidence_extensions_preserved"] += 1
                    digest_update(digests["candidate_nodes"], repaired)
                    emit(("" if first_node else ",") + compact(record_id) + ":" + compact(repaired))
                    first_node = False
                    if counts["node"] % 250000 == 0:
                        progress(output, "CANDIDATE_NODE_PROGRESS", **dict(counts))
                else:
                    if not edges_started:
                        emit('},"edges":[')
                        edges_started = True
                    digest_update(digests["edges"], record)
                    emit(("" if first_edge else ",") + compact(record))
                    first_edge = False
                    if counts["edge"] % 500000 == 0:
                        progress(output, "COPYING_UNCHANGED_EDGES", edges=counts["edge"])
            if not edges_started:
                emit('},"edges":[')
            emit("]}\n")
            source_sha = source_digest.hexdigest()
    if source_sha != source["sha256"]:
        raise ValueError("source graph SHA differs from audited graph")
    expected = audit["candidate_counts"]
    if counts["node"] != expected["nodes"] or counts["edge"] != expected["edges"] or counts["claims_checked"] != expected["claims"]:
        raise ValueError("candidate count mismatch")
    if counts["old_reader_reencoder_failures"] != 124297 or counts["claims_scientific_fields_carried"] != 150551:
        raise ValueError("before/after defect scope differs from frozen audit")
    patch_artifact = writer.close()
    for fp in code.values():
        guard(fp)
    check_inputs(output)
    partial.rename(graph_path)
    copied = {}
    for name in ("umls_details.sqlite", "NODE_REPAIRS.jsonl"):
        path = Path(source["path"]).parent / name
        target = output / name
        if target.exists():
            raise ValueError("candidate companion already exists")
        progress(output, "COPYING_UNCHANGED_COMPANION", name=name)
        expected_sha = audit["candidate_baseline"]["details"]["sha256"] if name.endswith("sqlite") else file_sha(path)
        shutil.copyfile(path, target)
        copied[name] = fingerprint(target)
        if copied[name]["sha256"] != expected_sha:
            raise ValueError("companion copy differs from source")
    check_inputs(output)
    result = {"status": "COMPATIBILITY_CANDIDATE_BUILT_NOT_APPLIED", "completed_at": utc_now(),
              "counts": dict(counts), "scientific_field_counts": dict(field_counts),
              "digests": {key: digest.hexdigest() for key, digest in digests.items()},
              "source_graph_sha256": source_sha, "source_graph_full_verified_at": utc_now(),
              "graph": {**cheap(graph_path), "sha256": graph_digest.hexdigest()}, "companions": copied,
              "patches": patch_artifact, "runtime_implementation": code, "new_reader_failures": 0,
              "edges_changed": 0, "node_ids_changed": 0, "node_merges": 0, "formal_apply_performed": False}
    atomic_json(output / "BUILD_COMPLETE.json", result)
    print(compact({key: result[key] for key in ("status", "counts", "scientific_field_counts")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "build"), required=True)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else build)(args.output.resolve())


if __name__ == "__main__":
    main()
