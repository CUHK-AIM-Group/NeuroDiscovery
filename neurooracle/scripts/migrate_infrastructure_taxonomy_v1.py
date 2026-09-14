"""Atomically migrate the formal KG infrastructure vocabulary to v2.

The migration is intentionally narrow:

* rename the persisted ``atlas`` domain tag to ``spatial_reference`` while
  retaining stable ``ATLAS:*`` identifiers and source-level query aliases;
* synchronize the curated spatial-reference and ML-model registries;
* append only missing model-to-modality infrastructure edges;
* leave every claim, paper and extracted-claim row untouched.

The multi-gigabyte graph is rewritten as a stream.  ``stage`` performs a full
structural validation and hashes the candidate.  ``apply`` creates a verified
hard-link rollback backup before atomically replacing the canonical graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from streaming_graph_json import (  # noqa: E402
    IncrementalJsonReader,
    is_claim_node,
    iter_concepts,
    iter_edges,
)

from neurooracle.scripts.refresh_full_v2_state import render_readme  # noqa: E402
from neurooracle.src.ingestion.experiment_infra import (  # noqa: E402
    ML_MODELS,
    SPATIAL_REFERENCES,
    _build_model_node,
    _build_spatial_reference_node,
)
from neurooracle.src.schema import Edge, normalize_domain_tags  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "neurooracle" / "data" / "full_v2"
GRAPH = DATA_DIR / "knowledge_graph.json"
EXTRACTED = DATA_DIR / "extracted_claims.jsonl"
STATE = DATA_DIR / "CURRENT_STATE.json"
README = DATA_DIR / "README.md"
PRIMARY = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "low7_single_pass_full_20260826"
)
CONTROLLER = PRIMARY / "CONTROLLER_STATE.json"
ARCHIVE_ROOT = REPO / "neurooracle" / "data" / "archive" / "kg_mutation_backups"
DEFAULT_OUTPUT = (
    REPO
    / "neurooracle"
    / "data"
    / "infrastructure_taxonomy"
    / "spatial_reference_ml_model_v1_20260903"
)

MIGRATION_ID = "spatial_reference_ml_model_v1_20260903"
AUTHORIZATION_TOKEN = (
    "explicit-user-authorization:spatial-reference-ml-model-expansion:2026-09-03"
)
LEGACY_SPATIAL_NAMES = frozenset(
    {
        "Schaefer100",
        "Schaefer200",
        "Schaefer400",
        "Schaefer1000",
        "AAL90",
        "AAL116",
        "Desikan",
        "Destrieux",
        "HarvardOxford_sub",
        "Glasser",
        "voxel",
        "EEG_10_20",
        "EEG_10_10",
        "EEG_SEED_62",
        "EEG_SEED_VIG_17",
        "EEG_BCI_32",
    }
)
LEGACY_MODEL_NAMES = frozenset(
    {
        "BrainGNN",
        "SwiFT",
        "3D-CNN",
        "XGBoost",
        "SVM",
        "EEGNet",
        "ShallowConvNet",
        "DeepConvNet",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cheap_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def full_fingerprint(path: Path) -> dict[str, Any]:
    result = cheap_fingerprint(path)
    result["sha256"] = file_sha256(path)
    return result


def require_authorization(value: str) -> None:
    if value != AUTHORIZATION_TOKEN:
        raise RuntimeError("exact infrastructure-migration authorization is required")


def require_lease(owner: str) -> dict[str, Any]:
    controller = read_json(CONTROLLER)
    lease = dict(controller.get("lease") or {})
    if lease.get("owner") != owner:
        raise RuntimeError("migration requires the owned primary controller lease")
    if datetime.fromisoformat(str(lease.get("expires_at"))) <= datetime.now(timezone.utc):
        raise RuntimeError("primary controller lease expired")
    return lease


def desired_nodes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, spec in SPATIAL_REFERENCES.items():
        node = _build_spatial_reference_node(name, spec)
        result[node.id] = node.to_dict()
    for name, spec in ML_MODELS.items():
        if not spec.get("kg_node", True):
            continue
        node = _build_model_node(name, spec)
        result[node.id] = node.to_dict()
    if len(result) != 72:
        raise RuntimeError(f"expected 72 registry nodes, found {len(result)}")
    return result


def desired_model_edges() -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model, spec in ML_MODELS.items():
        if not spec.get("kg_node", True):
            continue
        for modality in spec["modalities"]:
            edge = Edge(
                source_id=f"MODEL:{model}",
                target_id=f"MODALITY:{modality}",
                relation_type="supports_modality",
                source="experiment_infra",
                confidence=1.0,
            ).to_dict()
            result[(edge["source_id"], edge["target_id"])] = edge
    if len(result) != 168:
        raise RuntimeError(f"expected 168 model-modality edges, found {len(result)}")
    return result


def merge_registry_node(existing: dict[str, Any], desired: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged["id"] = desired["id"]
    merged["preferred_name"] = desired["preferred_name"]
    merged["source_vocab"] = desired["source_vocab"]
    merged["definition"] = desired["definition"]
    merged["semantic_types"] = list(
        dict.fromkeys([*(existing.get("semantic_types") or []), *(desired.get("semantic_types") or [])])
    )
    merged["domain_tags"] = normalize_domain_tags(
        [*(existing.get("domain_tags") or []), *(desired.get("domain_tags") or [])]
    )
    merged["aliases"] = list(
        dict.fromkeys([*(existing.get("aliases") or []), *(desired.get("aliases") or [])])
    )
    external_ids = dict(existing.get("external_ids") or {})
    external_ids.update(desired.get("external_ids") or {})
    merged["external_ids"] = external_ids
    if not merged.get("spatial_mapping") and desired.get("spatial_mapping"):
        merged["spatial_mapping"] = desired["spatial_mapping"]
    metadata = dict(existing.get("metadata") or {})
    metadata.update(desired.get("metadata") or {})
    merged["metadata"] = metadata
    return merged


def migrated_metadata(
    old: dict[str, Any], *, migrated_at: str, added_nodes: int, added_edges: int
) -> dict[str, Any]:
    metadata = dict(old)
    if metadata.get("infrastructure_taxonomy_migration"):
        raise RuntimeError("infrastructure taxonomy migration is already recorded")
    stats = dict(metadata.get("stats") or {})
    domains = dict(stats.get("domains") or {})
    sources = dict(stats.get("sources") or {})
    relations = dict(stats.get("relations") or {})
    if domains.get("atlas") != 16 or domains.get("spatial_reference", 0) != 0:
        raise RuntimeError("unexpected pre-migration atlas domain counts")
    if domains.get("ml_model") != 8:
        raise RuntimeError("unexpected pre-migration ml_model count")
    if relations.get("supports_modality") != 25:
        raise RuntimeError("unexpected pre-migration supports_modality count")
    if sources.get("experiment_infra") != 64:
        raise RuntimeError("unexpected pre-migration experiment_infra source count")

    domains.pop("atlas")
    domains["spatial_reference"] = len(SPATIAL_REFERENCES)
    domains["ml_model"] = len(ML_MODELS)
    sources["experiment_infra"] = int(sources["experiment_infra"]) + added_nodes
    relations["supports_modality"] = int(relations["supports_modality"]) + added_edges
    stats["n_concepts"] = int(stats["n_concepts"]) + added_nodes
    stats["n_edges"] = int(stats["n_edges"]) + added_edges
    stats["domains"] = domains
    stats["sources"] = sources
    stats["relations"] = relations
    # Every new model is attached to existing modality infrastructure. The 20
    # newly registered spatial references intentionally remain standalone until
    # their ROI crosswalks are curated.
    stats["connected_components"] = int(stats["connected_components"]) + 20
    metadata["stats"] = stats
    metadata["infrastructure_taxonomy_version"] = "experiment_infra.v2"
    metadata["infrastructure_taxonomy_migration"] = {
        "migration_id": MIGRATION_ID,
        "migrated_at": migrated_at,
        "legacy_query_alias": {"atlas": "spatial_reference"},
        "stable_identifier_namespace_retained": "ATLAS",
        "claim_or_paper_membership_changed": False,
        "spatial_reference": {"before": 16, "added": 20, "after": 36},
        "ml_model": {"before": 8, "added": 28, "after": 36},
        "supports_modality": {"before": 25, "added": added_edges, "after": relations["supports_modality"]},
    }
    return metadata


def rewrite_graph(source_path: Path, output_path: Path, migrated_at: str) -> dict[str, Any]:
    desired = desired_nodes()
    desired_edges = desired_model_edges()
    legacy_node_ids = {
        *(f"ATLAS:{name}" for name in LEGACY_SPATIAL_NAMES),
        *(f"MODEL:{name}" for name in LEGACY_MODEL_NAMES),
    }
    added_node_ids = set(desired) - legacy_node_ids
    expected_existing_model_edges = {
        pair for pair in desired_edges if pair[0] in {f"MODEL:{name}" for name in LEGACY_MODEL_NAMES}
    }
    if len(added_node_ids) != 48 or len(expected_existing_model_edges) != 24:
        raise RuntimeError("registry delta differs from the frozen migration contract")
    added_edges = len(desired_edges) - len(expected_existing_model_edges)

    found_registry: set[str] = set()
    found_model_edges: set[tuple[str, str]] = set()
    concept_count_before = 0
    edge_count_before = 0
    legacy_tags_replaced = 0
    modality_ids: set[str] = set()
    original_metadata: dict[str, Any] | None = None

    with source_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        reader = IncrementalJsonReader(source)
        reader.expect("{")
        output.write("{")
        first_top = True
        seen_top: set[str] = set()
        while True:
            if reader.peek() == "}":
                reader.expect("}")
                break
            key = reader.value()
            if not isinstance(key, str):
                raise RuntimeError("top-level graph key is not a string")
            if key in seen_top:
                raise RuntimeError(f"duplicate top-level graph key: {key}")
            seen_top.add(key)
            reader.expect(":")
            if not first_top:
                output.write(",")
            first_top = False
            output.write(compact_json(key))
            output.write(":")

            if key == "metadata":
                value = reader.value()
                if not isinstance(value, dict):
                    raise RuntimeError("graph metadata is not an object")
                original_metadata = value
                output.write(
                    compact_json(
                        migrated_metadata(
                            value,
                            migrated_at=migrated_at,
                            added_nodes=len(added_node_ids),
                            added_edges=added_edges,
                        )
                    )
                )
            elif key == "concepts":
                reader.expect("{")
                output.write("{")
                first = True
                while True:
                    if reader.peek() == "}":
                        reader.expect("}")
                        break
                    node_id = reader.value()
                    if not isinstance(node_id, str):
                        raise RuntimeError("concept key is not a string")
                    reader.expect(":")
                    node = reader.value()
                    if not isinstance(node, dict):
                        raise RuntimeError(f"concept {node_id!r} is not an object")
                    concept_count_before += 1
                    if node_id.startswith("MODALITY:"):
                        modality_ids.add(node_id)
                    tags = list(node.get("domain_tags") or [])
                    if "atlas" in tags:
                        legacy_tags_replaced += 1
                    node["domain_tags"] = normalize_domain_tags(tags)
                    if node_id in desired:
                        found_registry.add(node_id)
                        node = merge_registry_node(node, desired[node_id])
                    if not first:
                        output.write(",")
                    first = False
                    output.write(compact_json(node_id))
                    output.write(":")
                    output.write(compact_json(node))
                    delimiter = reader.peek()
                    if delimiter == ",":
                        reader.expect(",")
                    elif delimiter != "}":
                        raise RuntimeError(f"unexpected concept delimiter {delimiter!r}")
                if found_registry != legacy_node_ids:
                    missing = sorted(legacy_node_ids - found_registry)
                    unexpected = sorted(found_registry - legacy_node_ids)
                    raise RuntimeError(
                        f"formal registry baseline mismatch: missing={missing}, unexpected={unexpected}"
                    )
                for node_id in sorted(added_node_ids):
                    if not first:
                        output.write(",")
                    first = False
                    output.write(compact_json(node_id))
                    output.write(":")
                    output.write(compact_json(desired[node_id]))
                output.write("}")
            elif key == "edges":
                missing_modalities = {
                    target for _, target in desired_edges if target not in modality_ids
                }
                if missing_modalities:
                    raise RuntimeError(f"missing modality nodes: {sorted(missing_modalities)}")
                reader.expect("[")
                output.write("[")
                first = True
                while True:
                    if reader.peek() == "]":
                        reader.expect("]")
                        break
                    edge = reader.value()
                    if not isinstance(edge, dict):
                        raise RuntimeError("edge is not an object")
                    edge_count_before += 1
                    pair = (str(edge.get("source_id") or ""), str(edge.get("target_id") or ""))
                    if pair in desired_edges:
                        if edge.get("relation_type") != "supports_modality":
                            raise RuntimeError(f"model-modality pair has conflicting relation: {pair}")
                        if pair in found_model_edges:
                            raise RuntimeError(f"duplicate model-modality edge: {pair}")
                        found_model_edges.add(pair)
                    if not first:
                        output.write(",")
                    first = False
                    output.write(compact_json(edge))
                    delimiter = reader.peek()
                    if delimiter == ",":
                        reader.expect(",")
                    elif delimiter != "]":
                        raise RuntimeError(f"unexpected edge delimiter {delimiter!r}")
                if found_model_edges != expected_existing_model_edges:
                    raise RuntimeError(
                        "existing model-modality edges differ from the migration baseline"
                    )
                for pair in sorted(set(desired_edges) - found_model_edges):
                    if not first:
                        output.write(",")
                    first = False
                    output.write(compact_json(desired_edges[pair]))
                output.write("]")
            else:
                output.write(compact_json(reader.value()))

            delimiter = reader.peek()
            if delimiter == ",":
                reader.expect(",")
            elif delimiter != "}":
                raise RuntimeError(f"unexpected top-level delimiter {delimiter!r}")
        output.write("}")
        reader.skip_whitespace()
        if reader.buffer[reader.position :].strip() or source.read(1):
            raise RuntimeError("unexpected trailing data after graph object")

    if seen_top != {"metadata", "concepts", "edges"}:
        raise RuntimeError(f"unexpected top-level graph keys: {sorted(seen_top)}")
    if original_metadata is None:
        raise RuntimeError("graph metadata was not found")
    if legacy_tags_replaced != 16:
        raise RuntimeError(f"expected 16 legacy atlas tags, found {legacy_tags_replaced}")
    original_stats = original_metadata["stats"]
    if concept_count_before != int(original_stats["n_concepts"]):
        raise RuntimeError("source concept count differs from source metadata")
    if edge_count_before != int(original_stats["n_edges"]):
        raise RuntimeError("source edge count differs from source metadata")
    return {
        "concepts_before": concept_count_before,
        "concepts_added": len(added_node_ids),
        "concepts_after": concept_count_before + len(added_node_ids),
        "edges_before": edge_count_before,
        "edges_added": added_edges,
        "edges_after": edge_count_before + added_edges,
        "legacy_atlas_tags_replaced": legacy_tags_replaced,
        "spatial_reference_before": 16,
        "spatial_reference_after": len(SPATIAL_REFERENCES),
        "ml_model_before": 8,
        "ml_model_after": len(ML_MODELS),
        "model_modality_edges_after": len(desired_edges),
    }


def graph_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        reader = IncrementalJsonReader(source)
        reader.expect("{")
        key = reader.value()
        reader.expect(":")
        if key != "metadata":
            raise RuntimeError("metadata must be the first graph field")
        value = reader.value()
        if not isinstance(value, dict):
            raise RuntimeError("graph metadata is not an object")
        return value


def validate_staged(path: Path, rewrite: dict[str, Any]) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    registry_seen: set[str] = set()
    model_edges_seen: set[tuple[str, str]] = set()
    concepts = 0
    claims = 0
    desired = desired_nodes()
    expected_edges = desired_model_edges()
    for node_id, node in iter_concepts(path):
        concepts += 1
        if is_claim_node(node_id, node):
            claims += 1
        for tag in node.get("domain_tags") or []:
            domains[str(tag)] += 1
        sources[str(node.get("source_vocab") or "")] += 1
        if node_id in desired:
            registry_seen.add(node_id)
            if node.get("domain_tags") != desired[node_id].get("domain_tags"):
                raise RuntimeError(f"registry node has unexpected domain tag: {node_id}")
    edges = 0
    for edge in iter_edges(path):
        edges += 1
        relation = str(edge.get("relation_type") or "unknown")
        relations[relation] += 1
        pair = (str(edge.get("source_id") or ""), str(edge.get("target_id") or ""))
        if pair in expected_edges and relation == "supports_modality":
            model_edges_seen.add(pair)

    metadata = graph_metadata(path)
    stats = metadata["stats"]
    errors: list[str] = []
    if concepts != rewrite["concepts_after"] or concepts != int(stats["n_concepts"]):
        errors.append("concept count mismatch")
    if edges != rewrite["edges_after"] or edges != int(stats["n_edges"]):
        errors.append("edge count mismatch")
    if claims != 905274:
        errors.append(f"claim-node count changed: {claims}")
    if domains.get("atlas", 0):
        errors.append("legacy atlas tag remains")
    if domains.get("spatial_reference") != 36:
        errors.append("spatial_reference count is not 36")
    if domains.get("ml_model") != 36:
        errors.append("ml_model count is not 36")
    if registry_seen != set(desired):
        errors.append("not every registry node is present")
    if model_edges_seen != set(expected_edges):
        errors.append("not every registered model-modality edge is present")
    if dict(domains) != dict(stats["domains"]):
        errors.append("domain counts differ from graph metadata")
    if dict(sources) != dict(stats["sources"]):
        errors.append("source counts differ from graph metadata")
    if dict(relations) != dict(stats["relations"]):
        errors.append("relation counts differ from graph metadata")
    migration = metadata.get("infrastructure_taxonomy_migration") or {}
    if migration.get("migration_id") != MIGRATION_ID:
        errors.append("migration metadata missing")
    if errors:
        raise RuntimeError(f"staged graph validation failed: {errors}")
    return {
        "status": "PASS",
        "concepts": concepts,
        "claim_nodes": claims,
        "non_claim_concepts": concepts - claims,
        "edges": edges,
        "domains": dict(domains),
        "sources": dict(sources),
        "relations": dict(relations),
        "registered_model_modality_edges": len(model_edges_seen),
        "claim_or_paper_membership_changed": False,
    }


def stage(args: argparse.Namespace) -> dict[str, Any]:
    require_authorization(args.authorization)
    lease = require_lease(args.lease_owner)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "STAGE_REPORT.json"
    if report_path.exists():
        raise RuntimeError(f"stage report already exists: {report_path}")
    staged_graph = output_dir / "knowledge_graph.staged.json"
    if staged_graph.exists():
        raise RuntimeError(f"staged graph already exists: {staged_graph}")

    state = read_json(STATE)
    expected_graph = dict(state["canonical_files"]["knowledge_graph"])
    expected_extracted = dict(state["canonical_files"]["extracted_claims"])
    baseline = {
        "knowledge_graph": full_fingerprint(GRAPH),
        "extracted_claims": {
            **cheap_fingerprint(EXTRACTED),
            "sha256": str(expected_extracted["sha256"]),
        },
        "current_state": full_fingerprint(STATE),
        "readme": full_fingerprint(README),
    }
    if baseline["knowledge_graph"]["sha256"] != str(expected_graph["sha256"]):
        raise RuntimeError("formal graph hash differs from CURRENT_STATE")
    if baseline["knowledge_graph"]["bytes"] != int(expected_graph["bytes"]):
        raise RuntimeError("formal graph size differs from CURRENT_STATE")
    if baseline["extracted_claims"]["bytes"] != int(expected_extracted["bytes"]):
        raise RuntimeError("extracted-claim store size differs from CURRENT_STATE")

    migrated_at = utc_now()
    try:
        rewrite = rewrite_graph(GRAPH, staged_graph, migrated_at)
        if cheap_fingerprint(GRAPH) != {
            k: baseline["knowledge_graph"][k] for k in ("path", "bytes", "mtime_ns")
        }:
            raise RuntimeError("formal graph changed while staging")
        validation = validate_staged(staged_graph, rewrite)
        staged = full_fingerprint(staged_graph)
        report = {
            "schema_version": "neurooracle.infrastructure_taxonomy.stage.v1",
            "status": "STAGED_VALIDATED_READY_TO_APPLY",
            "migration_id": MIGRATION_ID,
            "staged_at": migrated_at,
            "authorization": args.authorization,
            "lease": lease,
            "formal_source_baseline": baseline,
            "rewrite": rewrite,
            "validation": validation,
            "staged_graph": staged,
        }
        report["stage_report_sha256"] = canonical_hash(report)
        write_json(report_path, report)
        return report
    except Exception:
        if staged_graph.exists():
            staged_graph.unlink()
        failure = {
            "status": "STAGE_FAILED_NO_FORMAL_MUTATION",
            "failed_at": utc_now(),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "STAGE_FAILED.json", failure)
        raise


def hardlink_backup(source: Path, destination: Path) -> None:
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"hard-link backup verification failed: {source}")


def restore_from_hardlink(backup: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".rollback-link")
    if temporary.exists():
        temporary.unlink()
    os.link(backup, temporary)
    os.replace(temporary, destination)


def new_state(
    old: dict[str, Any], report: dict[str, Any], graph_fp: dict[str, Any], backup_dir: Path
) -> dict[str, Any]:
    state = dict(old)
    applied_at = utc_now()
    state["generated_at"] = applied_at
    state["infrastructure_taxonomy_version"] = "experiment_infra.v2"
    extracted_fp = dict(report["formal_source_baseline"]["extracted_claims"])
    extracted_stat = EXTRACTED.stat()
    state["canonical_files"] = {
        "knowledge_graph": {
            "path": str(GRAPH.resolve()),
            "bytes": graph_fp["bytes"],
            "last_write_utc": datetime.fromtimestamp(
                GRAPH.stat().st_mtime, timezone.utc
            ).isoformat(),
            "sha256": graph_fp["sha256"],
        },
        "extracted_claims": {
            "path": str(EXTRACTED.resolve()),
            "bytes": extracted_fp["bytes"],
            "last_write_utc": datetime.fromtimestamp(
                extracted_stat.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": extracted_fp["sha256"],
        },
    }
    audit = dict(state.get("full_graph_taxonomy_audit") or {})
    audit["created_at"] = applied_at
    audit["source"] = str(GRAPH.resolve())
    audit["source_bytes"] = graph_fp["bytes"]
    audit["entities_scanned"] = {
        "concept_nodes": report["validation"]["concepts"],
        "claim_nodes": report["validation"]["claim_nodes"],
        "non_claim_concept_nodes": report["validation"]["non_claim_concepts"],
        "edges": report["validation"]["edges"],
    }
    audit["is_canonical"] = True
    state["full_graph_taxonomy_audit"] = audit
    state["infrastructure_taxonomy_update"] = {
        "schema_version": "neurooracle.infrastructure_taxonomy.update.v1",
        "status": "APPLIED_AND_POST_VERIFIED",
        "migration_id": MIGRATION_ID,
        "applied_at": applied_at,
        "legacy_query_alias": {"atlas": "spatial_reference"},
        "stable_identifier_namespace_retained": "ATLAS",
        "spatial_reference": {"before": 16, "added": 20, "after": 36},
        "ml_model": {"before": 8, "added": 28, "after": 36},
        "supports_modality_edges_added": report["rewrite"]["edges_added"],
        "claim_or_paper_membership_changed": False,
        "backup_dir": str(backup_dir.resolve()),
    }
    return state


def update_controller(
    owner: str,
    completion_path: Path,
    completion_sha256: str,
    graph_fp: dict[str, Any],
    backup_dir: Path,
) -> None:
    require_lease(owner)
    controller = read_json(CONTROLLER)
    controller["updated_at"] = utc_now()
    controller["last_deep_verify_at"] = controller["updated_at"]
    controller["last_deep_verify_valid"] = True
    controller["last_deep_verify_errors"] = []
    controller["formal_kg_mutation_permitted"] = False
    controller["infrastructure_taxonomy_control"] = {
        "schema_version": "neurooracle.infrastructure_taxonomy.control.v1",
        "status": "COMPLETE",
        "migration_id": MIGRATION_ID,
        "completion_evidence": str(completion_path.resolve()),
        "completion_evidence_sha256": completion_sha256,
        "formal_knowledge_graph": graph_fp,
        "extracted_claims_unchanged": True,
        "backup_dir": str(backup_dir.resolve()),
        "further_mutation_requires_new_user_authorization": True,
    }
    write_json(CONTROLLER, controller)


def apply(args: argparse.Namespace) -> dict[str, Any]:
    require_authorization(args.authorization)
    require_lease(args.lease_owner)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "STAGE_REPORT.json"
    report = read_json(report_path)
    claimed_report_hash = str(report.get("stage_report_sha256") or "")
    unsigned = dict(report)
    unsigned.pop("stage_report_sha256", None)
    if canonical_hash(unsigned) != claimed_report_hash:
        raise RuntimeError("stage report hash mismatch")
    if report.get("status") != "STAGED_VALIDATED_READY_TO_APPLY":
        raise RuntimeError("stage is not ready to apply")
    if report.get("migration_id") != MIGRATION_ID:
        raise RuntimeError("stage migration id mismatch")

    staged_graph = Path(str(report["staged_graph"]["path"]))
    staged_cheap = cheap_fingerprint(staged_graph)
    for key in ("bytes", "mtime_ns"):
        if staged_cheap[key] != report["staged_graph"][key]:
            raise RuntimeError(f"staged graph {key} changed")
    current_graph = cheap_fingerprint(GRAPH)
    baseline_graph = report["formal_source_baseline"]["knowledge_graph"]
    if current_graph != {
        key: baseline_graph[key] for key in ("path", "bytes", "mtime_ns")
    }:
        raise RuntimeError("formal graph changed after staging")
    if cheap_fingerprint(EXTRACTED) != {
        k: report["formal_source_baseline"]["extracted_claims"][k]
        for k in ("path", "bytes", "mtime_ns")
    }:
        raise RuntimeError("extracted-claim store changed after staging")

    lock_path = DATA_DIR / ".formal_kg_mutation.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, f"{args.lease_owner}\n".encode("utf-8"))
    os.close(lock_fd)
    applied_at = utc_now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = ARCHIVE_ROOT / f"infrastructure_taxonomy_v1_{stamp}"
    swapped = False
    committed = False
    try:
        backup_dir.mkdir(parents=True, exist_ok=False)
        hardlink_backup(GRAPH, backup_dir / GRAPH.name)
        shutil.copy2(STATE, backup_dir / STATE.name)
        shutil.copy2(README, backup_dir / README.name)
        write_json(
            backup_dir / "BACKUP_MANIFEST.json",
            {
                "schema_version": "neurooracle.infrastructure_taxonomy.backup.v1",
                "created_at": applied_at,
                "migration_id": MIGRATION_ID,
                "source_fingerprints": report["formal_source_baseline"],
                "large_graph_is_verified_hard_link": True,
            },
        )
        os.replace(staged_graph, GRAPH)
        swapped = True
        graph_fp = full_fingerprint(GRAPH)
        if graph_fp["sha256"] != report["staged_graph"]["sha256"]:
            raise RuntimeError("post-apply graph hash differs from validated stage")
        if graph_fp["bytes"] != report["staged_graph"]["bytes"]:
            raise RuntimeError("post-apply graph size differs from validated stage")

        old_state = read_json(backup_dir / STATE.name)
        state = new_state(old_state, report, graph_fp, backup_dir)
        write_json(STATE, state)
        README.write_text(render_readme(state), encoding="utf-8")
        extracted_after = {
            **cheap_fingerprint(EXTRACTED),
            "sha256": report["formal_source_baseline"]["extracted_claims"]["sha256"],
        }
        formal_after = {
            "knowledge_graph": graph_fp,
            "extracted_claims": extracted_after,
            "current_state": full_fingerprint(STATE),
            "readme": full_fingerprint(README),
        }
        if extracted_after != report["formal_source_baseline"]["extracted_claims"]:
            raise RuntimeError("extracted-claim store changed during apply")
        completion = {
            "schema_version": "neurooracle.infrastructure_taxonomy.complete.v1",
            "status": "APPLIED_AND_POST_VERIFIED",
            "migration_id": MIGRATION_ID,
            "applied_at": applied_at,
            "authorization": args.authorization,
            "stage_report_sha256": claimed_report_hash,
            "formal_sources_before": report["formal_source_baseline"],
            "formal_sources_after": formal_after,
            "rewrite": report["rewrite"],
            "validation": report["validation"],
            "backup_dir": str(backup_dir.resolve()),
            "backup_recoverable": True,
            "claim_or_paper_membership_changed": False,
        }
        completion["completion_sha256"] = canonical_hash(completion)
        completion_path = output_dir / "FORMAL_APPLY_COMPLETE.json"
        write_json(completion_path, completion)
        update_controller(
            args.lease_owner,
            completion_path,
            file_sha256(completion_path),
            graph_fp,
            backup_dir,
        )
        committed = True
        return completion
    except Exception:
        if not committed and swapped:
            restore_from_hardlink(backup_dir / GRAPH.name, GRAPH)
            shutil.copy2(backup_dir / STATE.name, STATE)
            shutil.copy2(backup_dir / README.name, README)
        write_json(
            output_dir / "FORMAL_APPLY_FAILED.json",
            {
                "status": "APPLY_FAILED_ROLLBACK_ATTEMPTED",
                "failed_at": utc_now(),
                "traceback": traceback.format_exc(),
                "backup_dir": str(backup_dir.resolve()),
            },
        )
        raise
    finally:
        if lock_path.exists():
            lock_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("stage", "apply"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        child.add_argument("--lease-owner", required=True)
        child.add_argument("--authorization", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = stage(args) if args.command == "stage" else apply(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
