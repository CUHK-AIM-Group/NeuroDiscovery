"""Collect and review new-atom/old-node matches; never mutate either graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_umls_simplification_candidate import (
    DEFAULT_OUTPUT as CANDIDATE_DIR,
    FORMAL,
    UMLS_ROOT,
    REPO,
    atomic_json,
    cheap,
    compact,
    file_sha,
    hashed_reader,
    progress,
    read_json,
    utc_now,
    walk_graph,
)
from umls_audit_store import UmlsAuditStore, compact_record
from umls_existing_alignment_review import DECISION_LABELS, POLICY_VERSION, context_summary, recorded_cuis, review_pair, semantic_types


DEFAULT_OUTPUT = UMLS_ROOT / "existing_alignment_review_v1_20260906"
SCHEMA = "umls.existing_alignment_review.v1"
REUSE_DECISIONS = {"verified_existing_reuse", "existing_reuse_metadata_mismatch"}


def check_source_fingerprints(candidate_dir: Path) -> dict:
    freeze = read_json(candidate_dir / "CANDIDATE_FREEZE.json")
    if freeze.get("status") != "READ_ONLY_CANDIDATE_VALIDATED":
        raise ValueError("candidate source has not passed validation")
    for group in ("formal_sources", "artifacts"):
        for name, record in freeze[group].items():
            actual = cheap(Path(record["path"]))
            if any(actual[key] != record[key] for key in ("path", "bytes", "mtime_ns")):
                raise ValueError(f"protected source changed: {group}/{name}")
    return freeze


def write_jsonl(path: Path, records) -> dict:
    digest = hashlib.sha256()
    count = 0
    with path.open("xb") as handle:
        for record in records:
            line = (compact(record) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
    return {**cheap(path), "rows": count, "sha256": digest.hexdigest()}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_context(candidate_dir: Path, output: Path) -> dict:
    candidate_dir, output = candidate_dir.resolve(strict=True), output.resolve()
    if output.is_relative_to(FORMAL.resolve()) or output.is_relative_to(candidate_dir):
        raise ValueError("review output must not be inside a protected source")
    baseline = check_source_fingerprints(candidate_dir)
    output.mkdir(parents=True, exist_ok=False)
    progress(output, "COLLECT_INDEXED_ALIGNMENT_INPUTS")
    atoms, pairs, mappings = {}, [], {}
    mappings_by_atom = defaultdict(list)
    with UmlsAuditStore(candidate_dir / "umls_details.sqlite") as store:
        for atom_id, target_id, status, shared in store.connection.execute(
            "SELECT atom_id,target_id,mapping_status,shared_domains "
            "FROM alignment_candidates ORDER BY atom_id,target_id"
        ):
            pairs.append({"atom_id": atom_id, "target_id": target_id,
                          "mapping_status": status, "shared_domains": json.loads(shared)})
        for record_id, payload, in_core in store.connection.execute(
            "SELECT a.record_id,a.payload_json,a.in_core FROM atoms a "
            "JOIN (SELECT DISTINCT atom_id FROM alignment_candidates) c ON c.atom_id=a.record_id "
            "ORDER BY a.record_id"
        ):
            atoms[record_id] = {"record": json.loads(payload), "in_core": bool(in_core)}
        for record_id, payload in store.connection.execute(
            "SELECT m.record_id,m.payload_json FROM mappings m "
            "JOIN (SELECT DISTINCT atom_id FROM alignment_candidates) c ON c.atom_id=m.source_id "
            "ORDER BY m.ordinal"
        ):
            record = json.loads(payload)
            mappings[record_id] = record
            mappings_by_atom[record["source_id"]].append(record_id)
    if len(pairs) != baseline["counts"]["alignment_pairs"]:
        raise ValueError("alignment pair count differs from frozen candidate")
    if set(atoms) != {row["atom_id"] for row in pairs}:
        raise ValueError("alignment atom references are not complete")
    for row in pairs:
        if row["mapping_status"] != atoms[row["atom_id"]]["record"]["metadata"]["mapping_status"]:
            raise ValueError("alignment status differs from original atom")
    parents = {row["record"]["metadata"]["source_mention_id"] for row in atoms.values()}
    old_targets = {row["target_id"] for row in pairs}
    required_nodes = parents | old_targets | {row["target_id"] for row in mappings.values()}
    nodes, claims, linked_claims = {}, {}, defaultdict(list)
    observed_atoms, observed_mappings = set(), set()
    counts = Counter()
    progress(output, "COLLECT_GRAPH_CONTEXT", atoms=len(atoms), pairs=len(pairs), required_nodes=len(required_nodes))
    with hashed_reader(candidate_dir / "knowledge_graph.candidate.json") as (reader, digest):
        for kind, node_id, record in walk_graph(reader):
            if kind == "metadata":
                continue
            counts[kind] += 1
            if kind == "node":
                if node_id in required_nodes:
                    if node_id in nodes:
                        raise ValueError("duplicate reference node")
                    nodes[node_id] = record
                if node_id in atoms:
                    if not atoms[node_id]["in_core"]:
                        raise ValueError("offloaded atom unexpectedly appears in core")
                    if compact_record(atoms[node_id]["record"], "atoms", node_id) != record:
                        raise ValueError("core atom and detail payload differ")
                    observed_atoms.add(node_id)
                if node_id.startswith("CLM:"):
                    counts["all_claims"] += 1
                    md = record.get("metadata") or {}
                    related = {md.get("subject_id"), md.get("object_id")} & parents
                    if related:
                        claims[node_id] = record
                        for parent in related:
                            linked_claims[parent].append(node_id)
                if counts[kind] % 500000 == 0:
                    progress(output, "COLLECT_GRAPH_NODES", nodes=counts[kind], selected_claims=len(claims))
            else:
                if record["source_id"] in atoms or record["target_id"] in atoms:
                    reference = (record.get("metadata") or {}).get("audit_ref") or ""
                    if not reference.startswith("mappings/"):
                        raise ValueError("unexpected non-UMLS edge touches a reviewed atom")
                    mapping_id = reference.partition("/")[2]
                    if mapping_id not in mappings or compact_record(mappings[mapping_id], "mappings", mapping_id) != record:
                        raise ValueError("core mapping and detail payload differ")
                    if mapping_id in observed_mappings:
                        raise ValueError("duplicate mapping reference")
                    observed_mappings.add(mapping_id)
                if counts[kind] % 1000000 == 0:
                    progress(output, "COLLECT_GRAPH_EDGES", edges=counts[kind])
        candidate_sha = digest.hexdigest()
    checks = {
        "candidate_sha256": candidate_sha == baseline["artifacts"]["knowledge_graph.candidate.json"]["sha256"],
        "candidate_node_count": counts["node"] == baseline["counts"]["core_nodes"],
        "candidate_edge_count": counts["edge"] == baseline["counts"]["source_edges"],
        "all_required_nodes_found": set(nodes) == required_nodes,
        "retained_atoms_match_details": observed_atoms == {key for key, row in atoms.items() if row["in_core"]},
        "all_mapping_edges_match_details": observed_mappings == set(mappings),
    }
    if not all(checks.values()):
        raise ValueError(f"context collection checks failed: {checks}")
    progress(output, "VERIFY_DETAIL_SOURCE_HASH")
    detail_sha = file_sha(candidate_dir / "umls_details.sqlite")
    if detail_sha != baseline["artifacts"]["umls_details.sqlite"]["sha256"]:
        raise ValueError("detail source hash differs from frozen candidate")
    if check_source_fingerprints(candidate_dir) != baseline:
        raise ValueError("source freeze changed while collecting context")
    artifacts = {
        "ATOMS.jsonl": write_jsonl(output / "ATOMS.jsonl", (
            {"atom_id": key, **row, "mapping_refs": mappings_by_atom[key]} for key, row in sorted(atoms.items())
        )),
        "PAIRS.jsonl": write_jsonl(output / "PAIRS.jsonl", pairs),
        "MAPPINGS.jsonl": write_jsonl(output / "MAPPINGS.jsonl", (
            {"mapping_ref": key, "record": record} for key, record in sorted(mappings.items())
        )),
        "REFERENCE_NODES.jsonl": write_jsonl(output / "REFERENCE_NODES.jsonl", (
            {"node_id": key, "record": record, "linked_claim_ids": sorted(linked_claims.get(key, []))}
            for key, record in sorted(nodes.items())
        )),
        "CLAIMS.jsonl": write_jsonl(output / "CLAIMS.jsonl", (claims[key] for key in sorted(claims))),
    }
    result = {
        "schema_version": SCHEMA, "status": "CONTEXT_COLLECTED_AND_VERIFIED", "created_at": utc_now(),
        "candidate_directory": str(candidate_dir),
        "source_freeze": {**cheap(candidate_dir / "CANDIDATE_FREEZE.json"),
                          "sha256": file_sha(candidate_dir / "CANDIDATE_FREEZE.json")},
        "source_baseline": baseline,
        "full_source_hashes_verified": {"candidate_graph": candidate_sha, "detail_store": detail_sha},
        "checks": checks,
        "counts": {"atoms": len(atoms), "pairs": len(pairs), "old_targets": len(old_targets),
                   "parents": len(parents), "reference_nodes": len(nodes), "mappings": len(mappings),
                   "linked_claims": len(claims), "parents_with_claim_context": len(linked_claims)},
        "claim_context_scope": "All CLM nodes whose canonical subject_id or object_id equals a selected parent mention.",
        "formal_files_modified": False, "candidate_files_modified": False,
        "artifacts": artifacts,
    }
    atomic_json(output / "CONTEXT_COMPLETE.json", result)
    progress(output, "CONTEXT_READY_FOR_REVIEW", **result["counts"])
    return result


def check_context_artifacts(output: Path) -> dict:
    complete = read_json(output / "CONTEXT_COMPLETE.json")
    if complete.get("status") != "CONTEXT_COLLECTED_AND_VERIFIED":
        raise ValueError("context inputs are not complete")
    for name, fp in complete["artifacts"].items():
        path = output / name
        if any(cheap(path)[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError(f"context fingerprint changed: {name}")
        if file_sha(path) != fp["sha256"]:
            raise ValueError(f"context content changed: {name}")
    return complete


def audit_recorded_semantics(nodes: dict, target_ids: set, result_dir: Path) -> dict:
    manifest_path = REPO / "neurooracle/data/raw/umls/2026AA/UMLS_RELEASE_MANIFEST.json"
    manifest = read_json(manifest_path)
    impact = read_json(UMLS_ROOT / "UMLS_2026AA_IMPACT_FREEZE.json")
    manifest_sha = file_sha(manifest_path)
    if manifest_sha != impact["release_manifest_sha256"] or manifest.get("release") != "2026AA":
        raise ValueError("semantic reference release manifest differs from UMLS impact baseline")
    expected = manifest["files"]["MRSTY.RRF"]
    path = Path(expected["path"])
    before = cheap(path)
    requested_cuis = {cui for key in target_ids for cui in recorded_cuis(nodes[key]["record"])}
    types_by_cui = defaultdict(list)
    digest = hashlib.sha256()
    row_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row_count += 1
            fields = line.decode("utf-8").rstrip("\r\n").split("|")
            if len(fields) < 4:
                raise ValueError("malformed MRSTY record")
            if fields[0] in requested_cuis:
                types_by_cui[fields[0]].append({"tui": fields[1], "tree_number": fields[2], "name": fields[3]})
    if digest.hexdigest() != expected["sha256"] or row_count != expected["rows"] or before["bytes"] != expected["bytes"] or cheap(path) != before:
        raise ValueError("MRSTY full integrity check failed")
    records = []
    for node_id in sorted(target_ids):
        node = nodes[node_id]["record"]
        cuis = recorded_cuis(node)
        if not cuis:
            continue
        old_types = semantic_types(node)
        current_types = {row["tui"] for cui in cuis for row in types_by_cui[cui]}
        if len(cuis) != 1:
            status = "multiple_recorded_cuis_needs_review"
        elif not current_types:
            status = "recorded_cui_not_found_in_2026AA_mrsty"
        elif old_types == current_types:
            status = "matches_2026AA_mrsty"
        elif not old_types:
            status = "old_semantic_types_missing"
        else:
            status = "old_semantic_types_differ"
        records.append({
            "target_id": node_id, "target_name": node.get("preferred_name"), "recorded_cuis": sorted(cuis),
            "old_semantic_types": sorted(old_types), "umls_2026AA_semantic_types": sorted(current_types),
            "reference_semantic_type_records": {cui: types_by_cui[cui] for cui in sorted(cuis)},
            "status": status, "metadata_update_applied": False,
            "scope": "Compare existing node metadata with the same recorded identifier in frozen MRSTY; no diagnosis or identity merge.",
        })
    artifact = write_jsonl(result_dir / "SEMANTIC_METADATA_REVIEW.jsonl", records)
    return {
        "status": "PASS", "source": {**before, "sha256": digest.hexdigest(), "rows": row_count},
        "release_manifest": {**cheap(manifest_path), "sha256": manifest_sha},
        "target_nodes_audited": len(records), "counts": dict(Counter(row["status"] for row in records)),
        "artifact": artifact,
    }


def triage_context(output: Path) -> dict:
    if (output / "REVIEW_FREEZE.json").exists():
        raise ValueError("review already frozen; do not rewrite evidence")
    complete = check_context_artifacts(output)
    result_dir = output / POLICY_VERSION
    result_dir.mkdir(exist_ok=False)
    progress(output, "TRIAGE_ALL_ALIGNMENT_PAIRS")
    atoms = {row["atom_id"]: row for row in read_jsonl(output / "ATOMS.jsonl")}
    mappings = {row["mapping_ref"]: row for row in read_jsonl(output / "MAPPINGS.jsonl")}
    nodes = {row["node_id"]: row for row in read_jsonl(output / "REFERENCE_NODES.jsonl")}
    claims = {row["id"]: row for row in read_jsonl(output / "CLAIMS.jsonl")}
    pairs = list(read_jsonl(output / "PAIRS.jsonl"))
    target_counts = Counter(row["atom_id"] for row in pairs)
    contexts = {}
    for atom in atoms.values():
        parent_id = atom["record"]["metadata"]["source_mention_id"]
        if parent_id not in contexts:
            parent = nodes[parent_id]
            contexts[parent_id] = context_summary(parent["record"], [claims[key] for key in parent["linked_claim_ids"]])
    decisions, by_atom = [], defaultdict(list)
    for pair in pairs:
        atom = atoms[pair["atom_id"]]
        parent_id = atom["record"]["metadata"]["source_mention_id"]
        result = review_pair(
            atom["record"], nodes[pair["target_id"]]["record"], nodes[parent_id]["record"],
            [mappings[key] for key in atom["mapping_refs"]], claim_context=contexts[parent_id],
            candidate_target_count=target_counts[pair["atom_id"]], in_core=atom["in_core"],
        )
        if result["shared_domains"] != sorted(pair["shared_domains"]):
            raise ValueError("shared domain evidence differs from frozen candidate")
        legacy_sources = (nodes[pair["target_id"]]["record"].get("metadata") or {}).get("standardized_clm_sources") or []
        result["recorded_legacy_parent_lineage"] = [row for row in legacy_sources if isinstance(row, dict) and row.get("id") == parent_id]
        if result["recorded_legacy_parent_lineage"]:
            result["caution_flags"] = sorted(set(result["caution_flags"]) | {"old_node_documents_this_parent_in_standardization_history"})
        decisions.append(result)
        by_atom[pair["atom_id"]].append(result)
    atom_results = []
    for atom_id, rows in sorted(by_atom.items()):
        verified = [row["target_id"] for row in rows if row["decision"] in REUSE_DECISIONS]
        if len(verified) > 1:
            raise ValueError("an atom has more than one verified unique identity")
        status = atoms[atom_id]["record"]["metadata"]["mapping_status"]
        if verified:
            resolution = "already_reused_existing_node"
        elif status == "unmapped":
            resolution = "unmapped_local_candidates_need_review"
        elif status == "needs_review":
            resolution = "umls_pending_candidates_need_review"
        else:
            resolution = "accepted_umls_but_lexical_candidates_not_verified"
        atom_results.append({
            "atom_id": atom_id, "atom_name": rows[0]["atom_name"], "source_mention_id": rows[0]["source_mention_id"],
            "mapping_status": status, "in_core": atoms[atom_id]["in_core"], "resolution": resolution,
            "verified_existing_target": verified[0] if verified else None,
            "target_decisions": [{"target_id": row["target_id"], "decision": row["decision"]} for row in rows],
            "claim_context_count": rows[0]["context"]["claim_count"],
            "merge_authorized": False, "parent_redirect_authorized": False,
        })
    decision_counts = Counter(row["decision"] for row in decisions)
    target_groups = defaultdict(list)
    for row in decisions:
        if row["decision"] != "verified_existing_reuse":
            target_groups[row["target_id"]].append(row)
    queue = []
    for target_id, rows in sorted(target_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        queue.append({
            "target_id": target_id, "target_name": rows[0]["target_name"], "target_definition": rows[0]["target_definition"],
            "atoms": len(rows), "decisions": dict(Counter(row["decision"] for row in rows)),
            "surface_counts": dict(Counter(row["atom_name"] for row in rows)),
            "atom_ids": sorted(row["atom_id"] for row in rows),
            "sample_parent_texts": list(dict.fromkeys(row["parent_text"] for row in rows))[:5],
        })
    quality = {
        "invalid_source_trace_pairs": sum(not row["source_trace"]["valid"] for row in decisions),
        "atoms_without_claim_context": sum(not row["claim_context_count"] for row in atom_results),
        "parents_without_claim_context": sum(not row["claim_count"] for row in contexts.values()),
        "atoms_with_multiple_lexical_targets": sum(count > 1 for count in target_counts.values()),
        "pairs_with_recorded_legacy_parent_lineage": sum(bool(row["recorded_legacy_parent_lineage"]) for row in decisions),
        "known_alias_conflict_pairs": decision_counts["blocked_known_alias_conflation"],
    }
    progress(output, "AUDIT_OLD_SEMANTIC_METADATA_AGAINST_MRSTY")
    semantics = audit_recorded_semantics(nodes, {row["target_id"] for row in pairs}, result_dir)
    # Fixed filenames are written once. Re-triage requires a fresh output directory.
    artifacts = {
        "PAIR_REVIEW.jsonl": write_jsonl(result_dir / "PAIR_REVIEW.jsonl", decisions),
        "ATOM_RESOLUTION.jsonl": write_jsonl(result_dir / "ATOM_RESOLUTION.jsonl", atom_results),
        "REVIEW_QUEUE.jsonl": write_jsonl(result_dir / "REVIEW_QUEUE.jsonl", queue),
        "SEMANTIC_METADATA_REVIEW.jsonl": semantics["artifact"],
    }
    summary = {
        "schema_version": SCHEMA, "policy_version": POLICY_VERSION, "status": "TRIAGED_AWAITING_ACCEPTANCE",
        "created_at": utc_now(), "counts": complete["counts"],
        "result_directory": str(result_dir.resolve()), "semantic_metadata_audit": semantics,
        "pair_decisions": dict(decision_counts), "decision_labels": DECISION_LABELS,
        "atom_resolutions": dict(Counter(row["resolution"] for row in atom_results)),
        "verified_existing_canonical_targets": len({row["verified_existing_target"] for row in atom_results if row["verified_existing_target"]}),
        "quality_checks": quality,
        "new_nodes_added": 0, "new_edges_added": 0, "nodes_merged": 0,
        "claim_endpoints_redirected": 0, "review_statuses_promoted": 0,
        "formal_files_modified": False, "candidate_files_modified": False,
        "artifacts": artifacts,
    }
    atomic_json(result_dir / "TRIAGE_SUMMARY.json", summary)
    progress(output, "TRIAGE_READY_FOR_ACCEPTANCE", atom_resolutions=summary["atom_resolutions"], quality_checks=quality)
    return summary


def checked_jsonl(fp: dict) -> list[dict]:
    path = Path(fp["path"])
    before = cheap(path)
    digest, records = hashlib.sha256(), []
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            records.append(json.loads(line))
    if digest.hexdigest() != fp["sha256"] or len(records) != fp["rows"] or cheap(path) != before:
        raise ValueError(f"review artifact changed: {path.name}")
    if any(before[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
        raise ValueError(f"review artifact fingerprint changed: {path.name}")
    return records


def contextual_alias_checks(rows: list[dict], claims: dict) -> list[dict]:
    """Store actual text cues; do not infer unobserved acronym expansions."""
    records = []
    cue_patterns = {
        "computed_tomography_in_claim": re.compile(r"\bcomputed\s+tomograph\w*\b", re.I),
        "explicit_CT_expansion": re.compile(r"\bcomputed\s+tomograph\w*\s*\(\s*CT\s*\)", re.I),
        "cortical_thickness_in_claim": re.compile(r"\bcortical\s+thickness\b", re.I),
    }
    for row in rows:
        if row["target_id"] != "IF:cortical_thickness" or row["atom_name"].casefold() != "ct":
            continue
        matches = []
        present = set()
        for claim_id in row["context"]["claim_ids"]:
            md = claims[claim_id].get("metadata") or {}
            text = str(md.get("raw_text") or "")
            cues = [key for key, pattern in cue_patterns.items() if pattern.search(text)]
            present.update(cues)
            if cues:
                matches.append({"claim_id": claim_id, "raw_text": text, "cues": cues,
                                "source_paper": md.get("source_paper") or {}})
        records.append({
            "atom_id": row["atom_id"], "target_id": row["target_id"], "source_mention_id": row["source_mention_id"],
            "parent_text": row["parent_text"], "target_definition": row["target_definition"],
            "cues": sorted(present), "claim_evidence": matches,
            "disposition": "block_automatic_alias_alignment_pending_context_review",
            "scope": "A phrase in a linked claim is a review cue; no claim is made that every CT token shares that expansion.",
            "graph_change_applied": False,
        })
    return records


def render_review_report(summary: dict, validation: dict, contextual: list[dict], proposals: list[dict]) -> str:
    a, p, q = summary["atom_resolutions"], summary["pair_decisions"], summary["quality_checks"]
    sem = summary["semantic_metadata_audit"]
    ct_cues = sum("computed_tomography_in_claim" in row["cues"] for row in contextual)
    ct_explicit = sum("explicit_CT_expansion" in row["cues"] for row in contextual)
    lines = [
        "# 新原子与旧节点对应核对：已验收的只读结果", "",
        "范围：19,198个原子、21,221对名称/领域候选、1,870个旧节点。不是全图所有潜在同义词的语义审校。",
        "方法：全量确定性分流与原始记录核对，并对具体高风险别名做重点核对；不等于逐条人工医学语义认证。",
        "正式图、精简候选主图及附属数据库均未修改；没有新建节点/边、节点合并、claim端点重定向或待审核映射晋级。", "",
        "## 原子级结论（互斥，合计19,198）", "",
        "| 分类 | 原子数 | 处理 |", "|---|---:|---|",
        f"| 已复用同一旧CUI，已有类型未见交集冲突 | {p.get('verified_existing_reuse', 0):,} | 沿用已有映射，不重复建规范节点 |",
        f"| 已复用同一旧CUI，但新旧类型无交集 | {p.get('existing_reuse_metadata_mismatch', 0):,} | 核对旧节点属性，不另建实体 |",
        f"| UMLS映射仍待审核 | {a.get('umls_pending_candidates_need_review', 0):,} | 保留needs_review，不自动晋级 |",
        f"| UMLS已映射，但本轮旧节点对应未确认 | {a.get('accepted_umls_but_lexical_candidates_not_verified', 0):,} | 区分本地概念、标识差异及泛化范围 |",
        f"| UMLS未映射，本地名称/别名候选 | {a.get('unmapped_local_candidates_need_review', 0):,} | 继续保存在附属表中核对 |", "",
        f"合计{a.get('already_reused_existing_node', 0):,}个原子已通过既有单一已接受边复用{summary['verified_existing_canonical_targets']:,}个旧节点。",
        "这里确认的是已有记录的标识与映射一致性，不意味着复合父mention可整体替换为CUI。原子仍承担来源角色。", "",
        "## 旧节点语义类型属性：直接对账UMLS 2026AA MRSTY", "",
        f"对{sem['target_nodes_audited']:,}个具有CUI前缀或记录标识的旧节点对账：",
        f"- 与MRSTY一致：{sem['counts'].get('matches_2026AA_mrsty', 0):,}个。",
        f"- 有类型但不同：{sem['counts'].get('old_semantic_types_differ', 0):,}个。",
        f"- 类型缺失：{sem['counts'].get('old_semantic_types_missing', 0):,}个。",
        f"- 未在MRSTY找到：{sem['counts'].get('recorded_cui_not_found_in_2026AA_mrsty', 0):,}个。", "",
        f"已输出{len(proposals):,}项仅供审核的类型对账提案，均包含旧值、2026AA值及来源哈希，尚未应用。",
        "1,662个同CUI但类型不一致的原子，其类型集合均与当前MRSTY一致；问题落在213个旧目标节点的属性上。",
        "两个未找到的标识为`CUI:multiple_sclerosis`、`CUI:multiple_sclerosis_pathogenesis`；不能仅凭CUI前缀将文字占位ID视为标准UMLS编号。",
        "本节semantic_types是节点顶层属性，不是metadata对象中新增加的字段；对账不会天然要求增加节点或字段种类。", "",
        "## 重点别名与范围风险", "",
        "1. `IF:alff`的别名含`fALFF`，产生3对候选。fALFF是比值指标，不能按同义词并入ALFF。已阻断这些候选的自动对应建议，旧图未改。[Zou等，2008](https://pubmed.ncbi.nlm.nih.gov/18501969/)",
        f"2. {len(contextual):,}个`CT`原子匹配`IF:cortical_thickness`；{ct_cues}个的关联claim出现computed tomography，其中{ct_explicit}个出现明确的括号展开。只是文本语境风险证据，不把余下CT一概判为同一含义。全部保留为待核对，不自动对齐。",
        f"3. {p.get('review_dataset_or_parcellation_scope', 0):,}对涉及数据字段/分区，{p.get('review_method_vs_measurement', 0):,}对涉及方法与定量指标，{p.get('review_broad_anchor_scope', 0):,}对涉及泛化锚点。不能以共同领域标签取代范围核对。", "",
        "## 验收与限制", "",
        f"- 全部{summary['counts']['pairs']:,}对与冻结候选一一对应，原文边界异常{q['invalid_source_trace_pairs']}对。",
        f"- {q['atoms_with_multiple_lexical_targets']:,}个原子有多个旧节点名称候选；{q['atoms_without_claim_context']}个原子、{q['parents_without_claim_context']}个父mention未找到规范端点关联claim。",
        f"- 读取并保留{summary['counts']['linked_claims']:,}条相关claim的完整记录；34对有旧节点标准化历史中的父ID线索，单独保留而未据此替换复合mention。",
        f"- {validation['tests']['tests']}项相关测试通过，完整结果校验、候选与附属表哈希、MRSTY哈希均通过。",
        "- 没有进行新的完整实验训练。本轮不改图，沿用上一轮实验输入回归证据，不声称新增对应候选已获得实验收益。",
        "- 现有KGE读取器仍未增加needs_review过滤；本轮复核标记和提案不会自动改变训练准入。", "",
        "## 交付与下一步", "",
        "以本目录REVIEW_FREEZE.json为唯一验收清单；上级目录的首版TRIAGE草稿未被采用，不是最终结论。",
        "- PAIR_REVIEW.jsonl：21,221对的逐项依据、父原文、CUI、类型、claim引用与禁止合并标记。",
        "- ATOM_RESOLUTION.jsonl：19,198个原子的互斥结论。",
        "- SEMANTIC_METADATA_REVIEW.jsonl / METADATA_RECONCILIATION_PROPOSALS.jsonl：旧类型对账及待审核提案。",
        "- ACRONYM_CONTEXT_REVIEW.jsonl：CT语境风险及原文依据。",
        "- REVIEW_QUEUE.jsonl：按旧目标聚合的后续核对清单。", "",
        "建议先处理480项旧语义类型属性提案、两个占位式CUI及高风险别名，再核对本地映射；不要把提案或候选清单直接当作合并表。",
        "完整性检查遵循[incremental-integrity-checks](C:/Users/45846/.codex/skills/incremental-integrity-checks/SKILL.md)：普通进度复用指纹，新结果冻结边界做完整核验。", "",
    ]
    return "\n".join(lines)


def accept_review(output: Path) -> dict:
    result_dir = output / POLICY_VERSION
    if (result_dir / "REVIEW_FREEZE.json").exists():
        raise ValueError("review is already frozen")
    complete = read_json(output / "CONTEXT_COMPLETE.json")
    summary = read_json(result_dir / "TRIAGE_SUMMARY.json")
    if summary["policy_version"] != POLICY_VERSION:
        raise ValueError("review policy version changed")
    for fp in complete["artifacts"].values():
        if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError("verified context input changed before acceptance")
    progress(output, "ACCEPT_REVIEW_RESULTS")
    reviewed = {name: checked_jsonl(fp) for name, fp in summary["artifacts"].items()}
    rows = reviewed["PAIR_REVIEW.jsonl"]
    atom_results = reviewed["ATOM_RESOLUTION.jsonl"]
    semantic_rows = reviewed["SEMANTIC_METADATA_REVIEW.jsonl"]
    pairs = list(read_jsonl(output / "PAIRS.jsonl"))
    atoms = {row["atom_id"]: row for row in read_jsonl(output / "ATOMS.jsonl")}
    mappings = {row["mapping_ref"]: row for row in read_jsonl(output / "MAPPINGS.jsonl")}
    nodes = {row["node_id"]: row for row in read_jsonl(output / "REFERENCE_NODES.jsonl")}
    claims = {row["id"]: row for row in read_jsonl(output / "CLAIMS.jsonl")}
    expected_pairs = {(row["atom_id"], row["target_id"]) for row in pairs}
    seen = {(row["atom_id"], row["target_id"]) for row in rows}
    by_atom = defaultdict(list)
    for row in rows:
        by_atom[row["atom_id"]].append(row)
        atom = atoms[row["atom_id"]]
        parent = nodes[row["source_mention_id"]]
        context = context_summary(parent["record"], [claims[key] for key in parent["linked_claim_ids"]])
        repeat = review_pair(atom["record"], nodes[row["target_id"]]["record"], parent["record"],
                             [mappings[key] for key in atom["mapping_refs"]], claim_context=context,
                             candidate_target_count=row["candidate_target_count"], in_core=atom["in_core"])
        for key, value in repeat.items():
            if key == "caution_flags":
                if not set(value).issubset(row[key]):
                    raise ValueError("a recomputed caution flag is missing")
            elif row.get(key) != value:
                raise ValueError(f"review evidence is not reproducible: {row['atom_id']} / {key}")
        if any(row[key] is not False for key in ("merge_authorized", "parent_redirect_authorized", "review_promotion_authorized")):
            raise ValueError("unexpected mutation authorization in review results")
    for result in atom_results:
        expected_targets = {row["target_id"] for row in by_atom[result["atom_id"]] if row["decision"] in REUSE_DECISIONS}
        if expected_targets != ({result["verified_existing_target"]} if result["verified_existing_target"] else set()):
            raise ValueError("atom resolution differs from its pair evidence")
    semantic_by_id = {row["target_id"]: row for row in semantic_rows}
    mismatches = [row for row in rows if row["decision"] == "existing_reuse_metadata_mismatch"]
    checks = {
        "all_pairs_accounted_for_once": seen == expected_pairs and len(seen) == len(rows),
        "all_atoms_accounted_for_once": set(atoms) == {row["atom_id"] for row in atom_results} and len(atom_results) == len(atoms),
        "pair_counts_exact": dict(Counter(row["decision"] for row in rows)) == summary["pair_decisions"],
        "atom_counts_exact": dict(Counter(row["resolution"] for row in atom_results)) == summary["atom_resolutions"],
        "same_cui_new_types_match_mrsty": all(row["atom_semantic_types"] == semantic_by_id[row["target_id"]]["umls_2026AA_semantic_types"] for row in mismatches),
        "semantic_metadata_counts_exact": dict(Counter(row["status"] for row in semantic_rows)) == summary["semantic_metadata_audit"]["counts"],
        "all_authorizations_false": all(not row["merge_authorized"] and not row["parent_redirect_authorized"] for row in atom_results),
    }
    if not all(checks.values()):
        raise ValueError(f"review acceptance failed: {checks}")
    test_path = result_dir / "TEST_RESULTS.xml"
    test_root = ET.parse(test_path).getroot()
    suites = list(test_root.iter("testsuite"))
    tests = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    if not tests["tests"] or tests["failures"] or tests["errors"] or tests["skipped"]:
        raise ValueError(f"test acceptance failed: {tests}")
    contextual = contextual_alias_checks(rows, claims)
    proposals = [{
        "target_id": row["target_id"], "target_name": row["target_name"], "recorded_cui": row["recorded_cuis"][0],
        "before_semantic_types": row["old_semantic_types"], "proposed_semantic_types": row["umls_2026AA_semantic_types"],
        "source_mrsty_sha256": summary["semantic_metadata_audit"]["source"]["sha256"],
        "requires_review_before_apply": True, "applied": False, "nodes_added": 0,
    } for row in semantic_rows if row["status"] in {"old_semantic_types_differ", "old_semantic_types_missing"}]
    supplemental = {
        "ACRONYM_CONTEXT_REVIEW.jsonl": write_jsonl(result_dir / "ACRONYM_CONTEXT_REVIEW.jsonl", contextual),
        "METADATA_RECONCILIATION_PROPOSALS.jsonl": write_jsonl(result_dir / "METADATA_RECONCILIATION_PROPOSALS.jsonl", proposals),
    }
    validation = {
        "status": "PASS", "checks": checks, "tests": tests,
        "source_trace_invalid_pairs": summary["quality_checks"]["invalid_source_trace_pairs"],
        "ct_candidates": len(contextual),
        "ct_candidates_with_computed_tomography_cue": sum("computed_tomography_in_claim" in row["cues"] for row in contextual),
        "ct_candidates_with_explicit_expansion": sum("explicit_CT_expansion" in row["cues"] for row in contextual),
        "metadata_reconciliation_proposals": len(proposals),
    }
    atomic_json(result_dir / "REVIEW_VALIDATION.json", validation)
    (result_dir / "ALIGNMENT_REVIEW_REPORT.md").write_text(render_review_report(summary, validation, contextual, proposals), encoding="utf-8", newline="\n")
    candidate_dir = Path(complete["candidate_directory"])
    if check_source_fingerprints(candidate_dir) != complete["source_baseline"]:
        raise ValueError("protected baseline changed before review freeze")
    for name in ("source", "release_manifest"):
        fp = summary["semantic_metadata_audit"][name]
        if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError("MRSTY reference changed before review freeze")
    artifacts = {**summary["artifacts"], **supplemental}
    for name in ("TRIAGE_SUMMARY.json", "REVIEW_VALIDATION.json", "TEST_RESULTS.xml", "ALIGNMENT_REVIEW_REPORT.md"):
        path = result_dir / name
        artifacts[name] = {**cheap(path), "sha256": file_sha(path)}
    code = {}
    for path in (Path(__file__), REPO / "neurooracle/src/umls_existing_alignment_review.py",
                 REPO / "neurooracle/tests/test_umls_existing_alignment_review.py",
                 REPO / "neurooracle/tests/test_umls_alignment_context.py"):
        code[str(path.relative_to(REPO))] = {**cheap(path), "sha256": file_sha(path)}
    freeze = {
        "schema_version": SCHEMA, "policy_version": POLICY_VERSION, "status": "READ_ONLY_ALIGNMENT_REVIEW_VALIDATED",
        "frozen_at": utc_now(), "counts": summary["counts"], "atom_resolutions": summary["atom_resolutions"],
        "pair_decisions": summary["pair_decisions"], "semantic_metadata_audit": summary["semantic_metadata_audit"],
        "validation": validation, "context_inputs": complete["artifacts"], "source_baseline": complete["source_baseline"],
        "artifacts": artifacts, "implementation": code, "formal_files_modified": False, "candidate_files_modified": False,
        "nodes_merged": 0, "claim_endpoints_redirected": 0, "review_statuses_promoted": 0,
        "not_asserted": ["full_human_semantic_certification", "full_model_retraining", "equivalence_of_parent_mention_and_atom", "automatic_training_use_of_review_proposals"],
    }
    atomic_json(result_dir / "REVIEW_FREEZE.json", freeze)
    atomic_json(output / "RUN_STATE.json", {"status": "COMPLETE", "phase": freeze["status"],
                                             "updated_at": utc_now(), "active_freeze": str(result_dir / "REVIEW_FREEZE.json")})
    print(compact({"status": freeze["status"], "atom_resolutions": freeze["atom_resolutions"], "validation": validation}), flush=True)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "triage", "accept"))
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "collect":
        collect_context(args.candidate, args.output)
    elif args.command == "triage":
        triage_context(args.output.resolve(strict=True))
    else:
        accept_review(args.output.resolve(strict=True))


if __name__ == "__main__":
    main()
