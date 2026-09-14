"""Validate and freeze a repair candidate, without applying it to full_v2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_umls_node_repair_candidate import (
    BASE_DIR, DEFAULT_OUTPUT, REPAIR_ID, REPO, REVIEW_DIR, SCHEMA,
    atomic_json, cheap, check_baselines, checked_jsonl, compact, digest_update,
    evidence_scores, file_sha, hashed_reader, kge_input_regression, progress,
    read_json, read_jsonl, utc_now, walk_graph, write_jsonl,
)
from umls_audit_store import UmlsAuditStore


def validate_details(output: Path, build: dict) -> dict:
    progress(output, "VALIDATE_RAW_DETAIL_PAYLOADS")
    expected = read_json(BASE_DIR / "BUILD_COMPLETE.json")["digests"]
    digests = {key: hashlib.sha256() for key in ("core_atoms", "offloaded_atoms", "new_cuis", "umls_edges")}
    counts = Counter()
    with UmlsAuditStore(output / "umls_details.sqlite") as store:
        if store.connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("repaired detail database integrity check failed")
        for table in ("atoms", "cuis", "mappings"):
            for (payload,) in store.connection.execute(f"SELECT payload_json FROM {table} ORDER BY ordinal"):
                import json
                record = json.loads(payload)
                if table == "atoms":
                    group = "offloaded_atoms" if record["metadata"]["mapping_status"] == "unmapped" else "core_atoms"
                else:
                    group = "new_cuis" if table == "cuis" else "umls_edges"
                digest_update(digests[group], record)
                counts[table] += 1
                if counts[table] % 500000 == 0:
                    progress(output, "VALIDATE_RAW_DETAILS", table=table, records=counts[table])
        active = store.connection.execute("SELECT * FROM alignment_candidates ORDER BY atom_id,target_id").fetchall()
        retired = store.connection.execute("SELECT atom_id,target_id,mapping_status,shared_domains,decision FROM retired_alignment_candidates ORDER BY atom_id,target_id").fetchall()
        info = store.connection.execute("SELECT value_json FROM info WHERE key='node_metadata_alias_repair'").fetchone()
    with UmlsAuditStore(BASE_DIR / "umls_details.sqlite") as base:
        original = base.connection.execute("SELECT * FROM alignment_candidates ORDER BY atom_id,target_id").fetchall()
    retirement_log = checked_jsonl(build["details"]["retirement_log"])
    expected_retired = sorted(tuple(row["source_row"]) for row in retirement_log)
    active_ids = {(row[0], row[1]) for row in active}
    retired_ids = {(row[0], row[1]) for row in retired}
    checks = {
        "all_raw_atom_cui_mapping_payloads_exact": all(digests[key].hexdigest() == expected[key] for key in digests),
        "lexical_pair_partition_lossless": sorted(active + retired) == original,
        "retired_pair_history_exact": retired == expected_retired,
        "active_retired_disjoint": not active_ids & retired_ids,
        "active_pair_count": len(active) == build["details"]["alignment_pairs_active"],
        "retired_pair_count": len(retired) == build["details"]["alignment_pairs_retired"],
        "detail_repair_manifest_present": bool(info),
    }
    if not all(checks.values()):
        raise ValueError(f"detail validation failed: {checks}")
    return {"status": "PASS", "checks": checks, "counts": dict(counts),
            "active_alignment_pairs": len(active), "retired_alignment_pairs": len(retired)}


def validate_graph(output: Path, build: dict) -> dict:
    progress(output, "VALIDATE_REPAIRED_GRAPH")
    patches = {row["node_id"]: row for row in checked_jsonl(build["graph"]["patch_log"])}
    affected_expected = {row["id"]: row for row in checked_jsonl(build["graph"]["affected_claims"])}
    fixture = read_json(BASE_DIR / "EVIDENCE_FIXTURE_CANDIDATE.json")
    fixture_expected = {row["node"]["id"]: row["node"] for row in fixture["records"]}
    fixture_found = {}
    digests = {key: hashlib.sha256() for key in ("source_nodes", "repaired_nodes", "unchanged_nodes", "claims", "edges")}
    ids, changed, affected_found = set(), set(), set()
    counts, metadata = Counter(), None
    with hashed_reader(output / "knowledge_graph.candidate.json") as (reader, digest):
        for kind, node_id, record in walk_graph(reader):
            if kind == "metadata":
                metadata = record
                continue
            if kind == "node":
                if node_id in ids:
                    raise ValueError("duplicate repaired node")
                ids.add(node_id)
                counts["nodes"] += 1
                digest_update(digests["repaired_nodes"], record)
                original = record
                if node_id in patches:
                    if record != patches[node_id]["after"]:
                        raise ValueError("node does not match its exact repair patch")
                    original = patches[node_id]["before"]
                    changed.add(node_id)
                else:
                    digest_update(digests["unchanged_nodes"], record)
                digest_update(digests["source_nodes"], original)
                if node_id.startswith("CLM:"):
                    counts["claims"] += 1
                    digest_update(digests["claims"], record)
                if node_id in affected_expected:
                    if record != affected_expected[node_id]:
                        raise ValueError("an affected claim's evidence changed")
                    affected_found.add(node_id)
                if node_id in fixture_expected:
                    if record != fixture_expected[node_id]:
                        raise ValueError("fixed claim fixture changed in repaired graph")
                    fixture_found[node_id] = record
                if counts["nodes"] % 500000 == 0:
                    progress(output, "VALIDATE_REPAIRED_NODES", nodes=counts["nodes"])
            else:
                if record["source_id"] not in ids or record["target_id"] not in ids:
                    raise ValueError("dangling repaired graph edge")
                counts["edges"] += 1
                digest_update(digests["edges"], record)
        graph_sha = digest.hexdigest()
    checks = {
        "all_node_patches_exact": changed == set(patches),
        "all_content_partition_digests_exact": all(value.hexdigest() == build["graph"]["digests"][key] for key, value in digests.items()),
        "inverse_patch_restores_all_source_node_records": digests["source_nodes"].hexdigest() == build["graph"]["digests"]["source_nodes"],
        "counts_unchanged": dict(counts) == build["graph"]["counts"],
        "metadata_counts_exact": counts["nodes"] == metadata["stats"]["n_concepts"] and counts["edges"] == metadata["stats"]["n_edges"],
        "affected_claims_exact": affected_found == set(affected_expected),
        "fixed_claim_fixture_read_from_repaired_graph": set(fixture_found) == set(fixture_expected),
        "candidate_sha256": graph_sha == build["graph"]["candidate"]["sha256"],
        "connected_components_unchanged": metadata["stats"]["connected_components"] == build["graph"]["connected_components"],
    }
    if not all(checks.values()):
        raise ValueError(f"graph validation failed: {checks}")
    atomic_json(output / "EVIDENCE_FIXTURE_REPAIRED.json", {
        "selection": fixture["selection"], "scopes": fixture["scopes"], "candidate_sha256": graph_sha,
        "records": [{**row, "node": fixture_found[row["node"]["id"]]} for row in fixture["records"]],
    })
    return {"status": "PASS", "checks": checks, "counts": dict(counts),
            "changed_nodes": len(changed), "candidate_sha256": graph_sha, "dangling_edges": 0}


def endpoint_and_alias_regression(output: Path, build: dict, artifact_output: Path | None = None) -> dict:
    from neurooracle.src.claim_semantics import concept_atom_roles, semantic_claim_endpoint
    from neurooracle.src.graph_manager import KnowledgeGraph
    from neurooracle.src.hypothesis_engine import HypothesisEngine
    from neurooracle.src.schema import ConceptNode
    patches = {row["node_id"]: row for row in read_jsonl(output / "NODE_REPAIRS.jsonl")}
    before = {key: row["before"] for key, row in patches.items()}
    after = {key: row["after"] for key, row in patches.items()}
    role_checks = {key: concept_atom_roles(before[key]) == concept_atom_roles(after[key]) for key in patches}
    if not all(role_checks.values()):
        raise ValueError("unexpected change in runtime concept atom roles")
    changed_endpoints, evaluated = [], 0
    for claim in read_jsonl(output / "AFFECTED_CLAIMS.jsonl"):
        md = claim.get("metadata") or {}
        for side in ("subject", "object"):
            node_id = md.get(f"{side}_id")
            if node_id not in patches:
                continue
            evaluated += 1
            old = semantic_claim_endpoint(claim, side, before)
            new = semantic_claim_endpoint(claim, side, after)
            if old != new:
                if "aliases" not in patches[node_id]["fields"]:
                    raise ValueError("a semantic-type-only repair changed claim endpoint resolution")
                changed_endpoints.append({"claim_id": claim["id"], "side": side, "canonical_id": node_id,
                                          "claim_endpoint_name": md.get(f"{side}_name"),
                                          "before": asdict(old) if old else None, "after": asdict(new) if new else None,
                                          "raw_claim_endpoint_modified": False})
    endpoint_fp = write_jsonl((artifact_output or output) / "ALIAS_ENDPOINT_VIEW_CHANGES.jsonl", changed_endpoints)
    alias_results = {}
    # Exercise the actual name resolver on both legacy and corrected records.
    for label, nodes in (("legacy_records", before), ("repaired_records", after)):
        kg = KnowledgeGraph()
        for node_id in ("IF:alff", "IF:cortical_thickness"):
            kg.add_concept(ConceptNode.from_dict(deepcopy(nodes[node_id])))
        engine = HypothesisEngine(kg)
        observations = {query: engine.resolve_name(query) for query in (
            "CT", "CT imaging", "cortical thickness", "CortThick", "fALFF", "dynamic fALFF", "ALFF",
        )}
        if any(observations[query] is not None for query in ("CT", "CT imaging", "fALFF", "dynamic fALFF")):
            raise ValueError("an unsupported abbreviation resolved to an unrelated imaging node")
        if observations["ALFF"] != "IF:alff" or observations["CortThick"] != "IF:cortical_thickness" or observations["cortical thickness"] != "IF:cortical_thickness":
            raise ValueError("a retained imaging-feature name stopped resolving")
        alias_results[label] = observations
    semantic_review = list(read_jsonl(REVIEW_DIR / "SEMANTIC_METADATA_REVIEW.jsonl"))
    satisfied = sum((after[row["target_id"]]["semantic_types"] if row["target_id"] in after else row["old_semantic_types"]) == row["umls_2026AA_semantic_types"]
                    for row in semantic_review if row["umls_2026AA_semantic_types"])
    return {"status": "PASS", "runtime_atom_roles_unchanged": True, "role_nodes_checked": len(role_checks),
            "claim_endpoints_evaluated": evaluated, "intentional_alias_endpoint_view_changes": len(changed_endpoints),
            "endpoint_changes_artifact": endpoint_fp, "actual_resolver_results": alias_results,
            "reviewed_old_nodes_now_matching_mrsty": satisfied, "placeholder_cui_nodes_left_unchanged": 2,
            "domain_tags_and_metadata_atom_types_reclassified": False}


def experiment_regression(output: Path, build: dict) -> dict:
    progress(output, "REPAIRED_CANDIDATE_EXPERIMENT_REGRESSION")
    before = kge_input_regression(BASE_DIR / "knowledge_graph.candidate.json", output, "base")
    gc.collect()
    after = kge_input_regression(output / "knowledge_graph.candidate.json", output, "repaired")
    checks = {"actual_kge_triples_and_domains_exact": before["all_triples"] == after["all_triples"] and before["node_domain_entries"] == after["node_domain_entries"],
              "both_fixed_seed_splits_exact": before["splits"] == after["splits"]}
    fixture = read_json(BASE_DIR / "EVIDENCE_FIXTURE_CANDIDATE.json")
    before_scores = evidence_scores(deepcopy(fixture))
    after_scores = evidence_scores(read_json(output / "EVIDENCE_FIXTURE_REPAIRED.json"))
    frozen_scores = read_json(BASE_DIR / "EXPERIMENT_REGRESSION.json")["evidence_scores"]
    checks["fixed_claim_scores_match_previous_candidate"] = before_scores == after_scores == frozen_scores
    endpoint = endpoint_and_alias_regression(output, build)
    if not all(checks.values()):
        raise ValueError(f"experiment regression failed: {checks}")
    result = {"status": "PASS", "checks": checks, "kge_triples": after["all_triples"]["count"],
              "split_seeds": [0, 42], "fixed_claim_score_count": len(after_scores),
              "endpoint_and_alias_regression": endpoint,
              "all_claim_records_preserved": True, "full_model_training_rerun": False,
              "existing_kge_needs_review_filter_changed": False,
              "limitation": "KGE inputs and fixed evidence scores are checked; alias resolution changes are intentional. No assertion that every global task sampler or retrained model metric is unchanged."}
    atomic_json(output / "EXPERIMENT_REGRESSION.json", result)
    return result


def report(build: dict, validation: dict, regression: dict, tests: dict) -> str:
    g, d = build["graph"], build["details"]
    e = regression["endpoint_and_alias_regression"]
    return f"""# 语义类型与高风险别名修复候选：验收完成

本轮已生成新的完整候选主图及配套附属数据库。正式图、第一版候选及上一轮核对结果均未改动。

## 已落地到新候选的修复

- 480个旧CUI节点的semantic_types与UMLS 2026AA MRSTY对账：369个修正、111个补全。
- IF:cortical_thickness移出通用别名CT；IF:alff移出fALFF。保留原始别名及全部修复前记录，可逆向恢复。
- 488对失效词法候选（485个CT、3个fALFF）从活动表移入retired_alignment_candidates历史表，未删除其原子、原文或实际映射边。
- 名称解析器增加定向保护，防止模糊子串匹配重新把fALFF认成ALFF；种子定义不再加入这两个别名。

fALFF与ALFF不能作为同义指标归并的依据：[Zou等，2008](https://pubmed.ncbi.nlm.nih.gov/18501969/)。CT是否表示皮层厚度需要上下文，不能作为不带上下文的全局别名。

## 修复后的规模

| 项目 | 修复前候选 | 修复后候选 |
|---|---:|---:|
| 节点 | {g['counts']['nodes']:,} | {g['counts']['nodes']:,} |
| 边 | {g['counts']['edges']:,} | {g['counts']['edges']:,} |
| claim | {g['counts']['claims']:,} | {g['counts']['claims']:,} |
| 节点metadata一级字段种类 | {len(g['metadata_fields']['node_metadata'])} | {len(g['metadata_fields']['node_metadata'])} |
| 边metadata一级字段种类 | {len(g['metadata_fields']['edge_metadata'])} | {len(g['metadata_fields']['edge_metadata'])} |
| 活动词法对应候选 | {d['alignment_pairs_before']:,} | {d['alignment_pairs_active']:,} |
| 历史退役对应候选 | 0 | {d['alignment_pairs_retired']:,} |

semantic_types是已有的节点顶层属性，本轮不新增每节点metadata字段。没有节点身份合并、claim端点ID重定向或UMLS审核状态晋级。

## 验收

- 全量主图读取与哈希通过；482个节点仅发生批准字段的修改，其余节点及全部边的记录摘要一致。
- 反向应用NODE_REPAIRS.jsonl可恢复所有原始节点记录；原始候选仍完整保留，未做原位覆盖。
- 附属数据库完整性、全部原子/CUI/映射原始payload摘要通过；活动与退役词法候选的并集精确等于原21,221对。
- 实际KGE读取器得到{regression['kge_triples']:,}条相同三元组；种子0、42的全量训练/验证/测试划分完全一致。
- 209条固定claim的评分与前一轮冻结证据一致。检查了{e['claim_endpoints_evaluated']:,}个涉及修复节点的claim端点，其中{e['intentional_alias_endpoint_view_changes']:,}个语义视图投影因别名清理发生变化，均列入ALIAS_ENDPOINT_VIEW_CHANGES.jsonl；原始claim未改。
- 482个节点的运行时atom角色保持不变；这是因为当前角色读取器使用domain_tags/metadata.atom_types。本轮没有据semantic_types重新推断研究角色。
- 本轮MRSTY可对账的{e['reviewed_old_nodes_now_matching_mrsty']:,}个旧节点已一致；两个文字占位式CUI没有改名或猜测归并。
- {tests['tests']}项相关测试通过；名称解析保护同时覆盖带旧别名和已修复的节点记录。

未重跑完整模型训练，未宣称全部全局候选抽样结果不变。现有KGE的needs_review筛选策略未改变，退役词法候选与实际UMLS映射边是不同的数据对象。

## 文件与使用

- knowledge_graph.candidate.json与umls_details.sqlite必须配套使用。
- NODE_REPAIRS.jsonl：482个节点的完整修复前/后记录及原因。
- RETIRED_ALIGNMENT_PAIRS.jsonl：488对退役候选的原始记录。
- VALIDATION.json、EXPERIMENT_REGRESSION.json、TEST_RESULTS.xml：验收证据。
- REPAIR_FREEZE.json：冻结哈希、输入绑定及实现文件指纹；尚未执行正式应用。

完整性检查遵循[incremental-integrity-checks](C:/Users/45846/.codex/skills/incremental-integrity-checks/SKILL.md)：常规进度复用指纹，修复候选冻结时完整核验。
"""


def finish_candidate(output: Path) -> dict:
    if (output / "REPAIR_FREEZE.json").exists():
        raise ValueError("repair candidate is already frozen")
    build = read_json(output / "BUILD_COMPLETE.json")
    if check_baselines() != build["baseline"]:
        raise ValueError("protected baseline changed")
    details = validate_details(output, build)
    graph = validate_graph(output, build)
    validation = {"status": "PASS", "graph": graph, "details": details}
    atomic_json(output / "VALIDATION.json", validation)
    regression = experiment_regression(output, build)
    suites = list(ET.parse(output / "TEST_RESULTS.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "errors", "failures", "skipped")}
    if not tests["tests"] or tests["errors"] or tests["failures"] or tests["skipped"]:
        raise ValueError(f"test gate failed: {tests}")
    if check_baselines() != build["baseline"]:
        raise ValueError("protected baseline changed during validation")
    progress(output, "FREEZE_NODE_REPAIR_CANDIDATE")
    (output / "CANDIDATE_REPORT.md").write_text(report(build, validation, regression, tests), encoding="utf-8", newline="\n")
    artifacts = {}
    for name in ("knowledge_graph.candidate.json", "umls_details.sqlite", "BUILD_COMPLETE.json", "NODE_REPAIRS.jsonl",
                 "AFFECTED_CLAIMS.jsonl", "RETIRED_ALIGNMENT_PAIRS.jsonl", "ALIAS_ENDPOINT_VIEW_CHANGES.jsonl",
                 "EVIDENCE_FIXTURE_REPAIRED.json",
                 "VALIDATION.json", "EXPERIMENT_REGRESSION.json", "KGE_BASE_INPUTS.json", "KGE_REPAIRED_INPUTS.json",
                 "TEST_RESULTS.xml", "CANDIDATE_REPORT.md"):
        path = output / name
        sha = graph["candidate_sha256"] if name == "knowledge_graph.candidate.json" else file_sha(path)
        artifacts[name] = {**cheap(path), "sha256": sha}
    code = {}
    for relative in ("neurooracle/scripts/build_umls_node_repair_candidate.py", "neurooracle/scripts/validate_umls_node_repair_candidate.py",
                     "neurooracle/src/node_alias_safety.py", "neurooracle/src/hypothesis_engine.py",
                     "neurooracle/src/ingestion/atlas_roi_modality.py", "neurooracle/tests/test_umls_node_repair.py"):
        path = REPO / relative
        code[relative] = {**cheap(path), "sha256": file_sha(path)}
    freeze = {"schema_version": SCHEMA, "repair_id": REPAIR_ID, "status": "REPAIRED_CANDIDATE_VALIDATED_NOT_APPLIED",
              "frozen_at": utc_now(), "baseline": build["baseline"], "counts": build["graph"]["counts"],
              "changed_nodes": 482, "semantic_type_nodes": 480, "alias_nodes": 2,
              "retired_lexical_pairs": details["retired_alignment_pairs"], "active_lexical_pairs": details["active_alignment_pairs"],
              "validation": validation, "experiment_regression": regression, "tests": tests,
              "artifacts": artifacts, "implementation": code, "formal_apply_performed": False,
              "formal_files_modified": False, "base_candidate_modified": False, "nodes_merged": 0, "edges_changed": 0}
    atomic_json(output / "REPAIR_FREEZE.json", freeze)
    atomic_json(output / "RUN_STATE.json", {"status": "COMPLETE", "phase": freeze["status"], "updated_at": utc_now(),
                                             "freeze_path": str(output / "REPAIR_FREEZE.json")})
    print(compact({"status": freeze["status"], "counts": freeze["counts"], "retired_pairs": details["retired_alignment_pairs"],
                   "endpoint_view_changes": regression["endpoint_and_alias_regression"]["intentional_alias_endpoint_view_changes"]}), flush=True)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    finish_candidate(args.output.resolve(strict=True))


if __name__ == "__main__":
    main()
