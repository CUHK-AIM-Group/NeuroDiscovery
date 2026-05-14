"""One-shot audit: scan KG + claims + hypotheses for quality issues.

Usage:
    python -m core.knowledge_graph.src.audit_kg \
        --kg core/knowledge_graph/data/full/knowledge_graph.json \
        --claims core/knowledge_graph/data/full/extracted_claims.jsonl \
        --hyp-dir core/knowledge_graph/data/quick \
        --out core/knowledge_graph/data/audit_report.md
"""
from __future__ import annotations
import argparse
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


VAGUE_PREDICATES = {"is_associated_with", "correlates_with", "related_to"}
VAGUE_NAME_TOKENS = {
    "activity", "function", "effect", "result", "outcome", "factor",
    "mechanism", "process", "change", "response", "level", "measure",
    "condition", "status", "state", "pattern", "value", "score",
    "finding", "group", "study", "patient", "patients", "subject",
    "subjects", "control", "controls", "cohort", "sample", "samples",
    "the", "this", "these", "those", "our", "their", "its", "various",
    "multiple", "several", "different", "some", "many",
    "various", "specific", "general", "common", "main", "key",
    "important", "significant",
}
GENERIC_HUB_NAMES = {
    "brain", "neural activity", "neurons", "function", "cognition",
    "disease", "disorder", "treatment", "patients", "outcome",
    "stress", "risk", "performance",
}


def load_kg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def audit_nodes(kg: dict) -> dict:
    concepts = kg.get("concepts", {})
    edges = kg.get("edges", [])

    report = {
        "total_nodes": len(concepts),
        "by_source_vocab": Counter(),
        "by_domain": Counter(),
        "clm_concepts": 0,
        "phase1_resolved": 0,
        "no_aliases_count": 0,
        "no_definition_count": 0,
        "no_external_ids_count": 0,
        "short_name": [],
        "vague_name": [],
        "single_word_generic": [],
        "numeric_only_name": [],
        "empty_name": [],
        "duplicate_pref_name_examples": {},
        "orphan_nodes": 0,
        "high_degree_hubs": [],   # >500 incident edges
        "clm_isolated_count": 0,  # CLM_CONCEPT with degree <= 1
    }

    # build degree
    degree = Counter()
    for e in edges:
        s, t = e.get("source_id"), e.get("target_id")
        if s: degree[s] += 1
        if t: degree[t] += 1

    name_to_ids = defaultdict(list)
    for cid, c in concepts.items():
        if not isinstance(c, dict):
            continue
        name = (c.get("preferred_name") or "").strip()
        sv = c.get("source_vocab") or "unknown"
        domains = c.get("domain_tags") or []
        report["by_source_vocab"][sv] += 1
        for d in domains:
            report["by_domain"][d] += 1

        if cid.startswith("CLM_CONCEPT"):
            report["clm_concepts"] += 1
            if degree[cid] <= 1:
                report["clm_isolated_count"] += 1
        else:
            report["phase1_resolved"] += 1

        if not c.get("aliases"):
            report["no_aliases_count"] += 1
        if not c.get("definition"):
            report["no_definition_count"] += 1
        if not c.get("external_ids"):
            report["no_external_ids_count"] += 1

        if not name:
            report["empty_name"].append(cid)
            continue
        # short single-word lowercase (e.g., "il6", "fc")
        if len(name) <= 3 and name.lower() == name:
            report["short_name"].append({"id": cid, "name": name, "vocab": sv})
        # vague (only generic tokens)
        tokens = re.findall(r"[a-zA-Z]+", name.lower())
        if tokens and all(t in VAGUE_NAME_TOKENS for t in tokens):
            report["vague_name"].append({"id": cid, "name": name, "vocab": sv})
        # single-word generic hub
        if name.lower() in GENERIC_HUB_NAMES:
            report["single_word_generic"].append({"id": cid, "name": name,
                                                  "vocab": sv, "degree": degree[cid]})
        if re.fullmatch(r"[\d\W]+", name):
            report["numeric_only_name"].append({"id": cid, "name": name})

        name_to_ids[name.lower()].append(cid)

        if degree[cid] == 0:
            report["orphan_nodes"] += 1
        if degree[cid] >= 500:
            report["high_degree_hubs"].append({"id": cid, "name": name,
                                               "vocab": sv, "degree": degree[cid]})

    # duplicate names
    duplicate_count = 0
    duplicate_examples = []
    for nm, ids in name_to_ids.items():
        if len(ids) > 1:
            duplicate_count += 1
            duplicate_examples.append((nm, ids))
    duplicate_examples.sort(key=lambda kv: -len(kv[1]))
    report["duplicate_pref_name_count"] = duplicate_count
    report["duplicate_pref_name_examples"] = {
        nm: ids[:6] for nm, ids in duplicate_examples[:25]
    }

    report["high_degree_hubs"] = sorted(
        report["high_degree_hubs"], key=lambda x: -x["degree"]
    )[:20]
    report["by_source_vocab"] = dict(report["by_source_vocab"].most_common())
    report["by_domain"] = dict(report["by_domain"].most_common())
    for k in ("short_name", "vague_name", "single_word_generic",
              "numeric_only_name", "empty_name"):
        report[f"{k}_count"] = len(report[k])
        report[k] = report[k][:25]

    return report


def audit_edges(kg: dict) -> dict:
    edges = kg.get("edges", [])
    report = {
        "total_edges": len(edges),
        "by_relation": Counter(),
        "vague_predicate_count": 0,
        "self_loops": 0,
        "low_confidence_lt_0_3": 0,
        "no_evidence_ref": 0,
        "duplicate_triples": 0,
        "negated_count": 0,
    }
    triple_counts = Counter()
    for e in edges:
        rel = e.get("relation_type") or "unknown"
        report["by_relation"][rel] += 1
        if rel in VAGUE_PREDICATES:
            report["vague_predicate_count"] += 1
        s, t = e.get("source_id"), e.get("target_id")
        if s and s == t:
            report["self_loops"] += 1
        conf = e.get("confidence")
        if conf is not None and float(conf) < 0.3:
            report["low_confidence_lt_0_3"] += 1
        if not e.get("evidence_ref"):
            report["no_evidence_ref"] += 1
        if (e.get("metadata") or {}).get("negated"):
            report["negated_count"] += 1
        triple_counts[(s, rel, t)] += 1
    report["duplicate_triples"] = sum(1 for c in triple_counts.values() if c > 1)
    report["by_relation"] = dict(report["by_relation"].most_common(30))
    return report


def audit_claims(claims_path: Path) -> dict:
    report = {
        "total_claims": 0,
        "by_predicate": Counter(),
        "raw_text_missing": 0,
        "raw_text_short_lt_60": 0,
        "predicate_keyword_absent_in_raw": 0,
        "object_tokens_absent_from_raw": 0,
        "subject_tokens_absent_from_raw": 0,
        "self_claim_subj_eq_obj": 0,
        "no_p_value_no_effect_size": 0,
        "narrative_review_count": 0,
        "same_pmid_same_pair_diff_predicate": 0,
        "examples_object_injected": [],
        "examples_subject_injected": [],
        "examples_vague_plus_precise_pair": [],
        "examples_self_claim": [],
    }
    pmid_pair_preds = defaultdict(list)

    for c in iter_jsonl(claims_path):
        report["total_claims"] += 1
        pred = c.get("predicate", "")
        report["by_predicate"][pred] += 1

        raw = (c.get("raw_text") or "").lower()
        if not raw:
            report["raw_text_missing"] += 1
            continue
        if len(raw) < 60:
            report["raw_text_short_lt_60"] += 1

        subj_name = (c.get("subject_name") or "").lower()
        obj_name = (c.get("object_name") or "").lower()
        subj_id = c.get("subject_id") or ""
        obj_id = c.get("object_id") or ""

        if subj_id and obj_id and subj_id == obj_id:
            report["self_claim_subj_eq_obj"] += 1
            if len(report["examples_self_claim"]) < 10:
                report["examples_self_claim"].append({
                    "pmid": (c.get("source_paper") or {}).get("pmid"),
                    "subject_name": c.get("subject_name"),
                    "object_name": c.get("object_name"),
                    "predicate": pred,
                    "raw_preview": (c.get("raw_text") or "")[:160],
                })

        # predicate keyword absence (skip generic vague predicates)
        if pred and pred not in VAGUE_PREDICATES and pred != "is_a":
            pred_tokens = [t for t in pred.lower().replace("_", " ").split()
                           if len(t) >= 4 and t not in {"with", "from"}]
            if pred_tokens and not any(t in raw for t in pred_tokens):
                report["predicate_keyword_absent_in_raw"] += 1

        # object token presence
        if obj_name:
            obj_tokens = re.findall(r"[a-z]{4,}", obj_name)
            if obj_tokens:
                hits = sum(1 for t in obj_tokens if t in raw)
                if hits / len(obj_tokens) < 0.4:
                    report["object_tokens_absent_from_raw"] += 1
                    if len(report["examples_object_injected"]) < 15:
                        report["examples_object_injected"].append({
                            "pmid": (c.get("source_paper") or {}).get("pmid"),
                            "predicate": pred,
                            "subject": c.get("subject_name"),
                            "object": c.get("object_name"),
                            "raw_preview": (c.get("raw_text") or "")[:200],
                        })

        # subject token presence
        if subj_name:
            subj_tokens = re.findall(r"[a-z]{4,}", subj_name)
            if subj_tokens:
                hits = sum(1 for t in subj_tokens if t in raw)
                if hits / len(subj_tokens) < 0.4:
                    report["subject_tokens_absent_from_raw"] += 1
                    if len(report["examples_subject_injected"]) < 10:
                        report["examples_subject_injected"].append({
                            "pmid": (c.get("source_paper") or {}).get("pmid"),
                            "predicate": pred,
                            "subject": c.get("subject_name"),
                            "object": c.get("object_name"),
                            "raw_preview": (c.get("raw_text") or "")[:200],
                        })

        ev = c.get("evidence") or {}
        if not ev.get("p_value") and not ev.get("effect_size"):
            report["no_p_value_no_effect_size"] += 1
        if (ev.get("study_type") or "").lower() in {"narrative_review", "review"}:
            report["narrative_review_count"] += 1

        pmid = (c.get("source_paper") or {}).get("pmid")
        if pmid and subj_id and obj_id:
            pmid_pair_preds[(pmid, subj_id, obj_id)].append(pred)

    for key, preds in pmid_pair_preds.items():
        if len(set(preds)) > 1:
            report["same_pmid_same_pair_diff_predicate"] += 1
            has_vague = any(p in VAGUE_PREDICATES for p in preds)
            has_precise = any(p not in VAGUE_PREDICATES for p in preds)
            if has_vague and has_precise and \
                    len(report["examples_vague_plus_precise_pair"]) < 12:
                report["examples_vague_plus_precise_pair"].append({
                    "pmid": key[0], "subject_id": key[1], "object_id": key[2],
                    "predicates": sorted(set(preds)),
                })
    report["by_predicate"] = dict(report["by_predicate"].most_common(25))
    return report


def audit_hypotheses(hyp_dir: Path) -> dict:
    report = {"files": {}}
    for f in sorted(hyp_dir.glob("hypotheses_*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            report["files"][f.name] = {"error": str(e)}
            continue

        if isinstance(data, dict) and "hypotheses" in data:
            items = data["hypotheses"]
        elif isinstance(data, list):
            items = data
        else:
            report["files"][f.name] = {"error": "unknown structure"}
            continue

        rec = {
            "count": len(items),
            "vague_predicate_in_path": 0,
            "single_node_path": 0,
            "duplicate_path": 0,
            "low_score_lt_0_2": 0,
            "hub_to_hub": 0,
            "single_node_endpoint_unresolved_clm": 0,
            "target_domain_counter": Counter(),
            "path_length_counter": Counter(),
            "endpoint_examples": [],
        }
        seen_paths = set()
        for h in items:
            path = h.get("path") or h.get("full_path") or h.get("nodes") or []
            if isinstance(path, str):
                path = path.split(" -> ")
            norm_path = []
            for p in path:
                if isinstance(p, dict):
                    norm_path.append(str(p.get("preferred_name") or p.get("name")
                                       or p.get("id") or p))
                else:
                    norm_path.append(str(p))
            path = norm_path
            if len(path) <= 1:
                rec["single_node_path"] += 1
            rec["path_length_counter"][len(path)] += 1
            key = tuple(path)
            if key in seen_paths:
                rec["duplicate_path"] += 1
            else:
                seen_paths.add(key)
            edges = h.get("edges") or h.get("edge_details") or h.get("relations") or []
            for e in edges:
                if isinstance(e, dict):
                    rel = e.get("relation_type") or e.get("relation") or ""
                    if rel in VAGUE_PREDICATES:
                        rec["vague_predicate_in_path"] += 1
                        break
            if path:
                a = path[0].lower()
                b = path[-1].lower()
                if a in GENERIC_HUB_NAMES and b in GENERIC_HUB_NAMES:
                    rec["hub_to_hub"] += 1
                # unresolved CLM concept endpoint
                ids = h.get("path_ids") or h.get("node_ids") or []
                if ids and isinstance(ids, list):
                    if any(str(x).startswith("CLM_CONCEPT") for x in [ids[0], ids[-1]]):
                        rec["single_node_endpoint_unresolved_clm"] += 1
                        if len(rec["endpoint_examples"]) < 5:
                            rec["endpoint_examples"].append({
                                "path": " -> ".join(path),
                                "endpoint_ids": [ids[0], ids[-1]] if len(ids) >= 2 else ids,
                            })
            score = h.get("composite_score") or h.get("score") or h.get("confidence")
            if score is not None and float(score) < 0.2:
                rec["low_score_lt_0_2"] += 1
            td = h.get("target_domain") or h.get("target_type") or "unknown"
            rec["target_domain_counter"][td] += 1

        rec["target_domain_counter"] = dict(rec["target_domain_counter"])
        rec["path_length_counter"] = dict(rec["path_length_counter"])
        report["files"][f.name] = rec
    return report


def format_md(nodes_r, edges_r, claims_r, hyp_r) -> str:
    out = ["# KG & Hypothesis Audit Report",
           "",
           "_Source: `core/knowledge_graph/data/full/`_",
           ""]

    out.append("## Phase 1-2: 节点质量\n")
    out.append(f"- 总节点数: **{nodes_r['total_nodes']:,}**")
    out.append(f"  - Phase 1 已解析（NN/MeSH/CognitiveAtlas/etc.）: {nodes_r['phase1_resolved']:,}")
    out.append(f"  - CLM_CONCEPT（Phase 2 LLM 新建）: **{nodes_r['clm_concepts']:,}**")
    out.append(f"- 孤立节点（degree=0）: **{nodes_r['orphan_nodes']:,}**")
    out.append(f"- CLM_CONCEPT degree ≤ 1（孤立或单一引用，价值低）: **{nodes_r['clm_isolated_count']:,}**")
    out.append(f"- 无 aliases: {nodes_r['no_aliases_count']:,}")
    out.append(f"- 无 definition: {nodes_r['no_definition_count']:,}")
    out.append(f"- 无 external_ids: {nodes_r['no_external_ids_count']:,}")
    out.append(f"- 短名字（≤3 chars 全小写）: **{nodes_r['short_name_count']:,}**")
    out.append(f"- 模糊名字（仅由 'activity/function/effect' 等通用词构成）: **{nodes_r['vague_name_count']:,}**")
    out.append(f"- 单词通用 hub（'brain', 'function' 等）: **{nodes_r['single_word_generic_count']:,}**")
    out.append(f"- 纯数字/符号名字: {nodes_r['numeric_only_name_count']:,}")
    out.append(f"- 空 preferred_name: {nodes_r['empty_name_count']:,}")
    out.append(f"- 同名不同 id（疑似未合并别名）: **{nodes_r['duplicate_pref_name_count']:,}**\n")

    out.append("### Source vocab 分布\n```json")
    out.append(json.dumps(nodes_r["by_source_vocab"], indent=2, ensure_ascii=False))
    out.append("```\n")
    out.append("### Domain 分布\n```json")
    out.append(json.dumps(nodes_r["by_domain"], indent=2, ensure_ascii=False))
    out.append("```\n")

    out.append("### High-degree hubs（degree ≥ 500，通用 hub 嫌疑）")
    for ex in nodes_r["high_degree_hubs"]:
        out.append(f"- `{ex['id']}` **{ex['name']}** ({ex['vocab']}, degree={ex['degree']})")

    if nodes_r["short_name"]:
        out.append("\n### 短名字示例")
        for ex in nodes_r["short_name"][:15]:
            out.append(f"- `{ex['id']}` → **{ex['name']}** ({ex['vocab']})")

    if nodes_r["vague_name"]:
        out.append("\n### 模糊名字示例")
        for ex in nodes_r["vague_name"][:15]:
            out.append(f"- `{ex['id']}` → **{ex['name']}** ({ex['vocab']})")

    if nodes_r["single_word_generic"]:
        out.append("\n### 单词通用 hub 示例")
        for ex in nodes_r["single_word_generic"][:15]:
            out.append(f"- `{ex['id']}` → **{ex['name']}** ({ex['vocab']}, degree={ex['degree']})")

    if nodes_r["duplicate_pref_name_examples"]:
        out.append("\n### 同名不同 id（top 20）")
        for nm, ids in list(nodes_r["duplicate_pref_name_examples"].items())[:20]:
            out.append(f"- **{nm}** → {len(ids)} ids: {ids[:4]}")

    # ===== Edges =====
    out.append("\n## Phase 2: 边质量\n")
    out.append(f"- 总边数: **{edges_r['total_edges']:,}**")
    out.append(f"- 模糊谓词 ({sorted(VAGUE_PREDICATES)}): **{edges_r['vague_predicate_count']:,}** "
               f"({100*edges_r['vague_predicate_count']/max(1,edges_r['total_edges']):.1f}%)")
    out.append(f"- 自环边: **{edges_r['self_loops']:,}**")
    out.append(f"- confidence < 0.3 的边: **{edges_r['low_confidence_lt_0_3']:,}**")
    out.append(f"- 无 evidence_ref: {edges_r['no_evidence_ref']:,}")
    out.append(f"- negated edges: {edges_r['negated_count']:,}")
    out.append(f"- 重复 triple (s, rel, t): **{edges_r['duplicate_triples']:,}**\n")
    out.append("### Relation 分布 top 30\n```json")
    out.append(json.dumps(edges_r["by_relation"], indent=2, ensure_ascii=False))
    out.append("```\n")

    # ===== Claims =====
    out.append("\n## Phase 2: Claims 内容质量\n")
    n = max(1, claims_r["total_claims"])
    out.append(f"- 总 claims: **{claims_r['total_claims']:,}**")
    out.append(f"- raw_text 缺失: {claims_r['raw_text_missing']:,}")
    out.append(f"- raw_text < 60 chars（过短，证据弱）: {claims_r['raw_text_short_lt_60']:,}")
    out.append(f"- subject_id == object_id（自指 claim）: **{claims_r['self_claim_subj_eq_obj']:,}**")
    out.append(f"- predicate 关键词不在 raw_text 中: **{claims_r['predicate_keyword_absent_in_raw']:,}** "
               f"({100*claims_r['predicate_keyword_absent_in_raw']/n:.1f}%)")
    out.append(f"- object 名称多数 token 不在 raw_text（LLM 注入嫌疑）: **{claims_r['object_tokens_absent_from_raw']:,}** "
               f"({100*claims_r['object_tokens_absent_from_raw']/n:.1f}%)")
    out.append(f"- subject 名称多数 token 不在 raw_text: {claims_r['subject_tokens_absent_from_raw']:,}")
    out.append(f"- 无 p_value 无 effect_size（弱量化证据）: {claims_r['no_p_value_no_effect_size']:,} "
               f"({100*claims_r['no_p_value_no_effect_size']/n:.1f}%)")
    out.append(f"- 来自 narrative_review/review 类研究: {claims_r['narrative_review_count']:,} "
               f"({100*claims_r['narrative_review_count']/n:.1f}%)")
    out.append(f"- 同 PMID 同 (s, o) 多谓词（待 dedup）: **{claims_r['same_pmid_same_pair_diff_predicate']:,}**\n")

    out.append("### Predicate 分布\n```json")
    out.append(json.dumps(claims_r["by_predicate"], indent=2, ensure_ascii=False))
    out.append("```\n")

    if claims_r["examples_self_claim"]:
        out.append("\n### Subject == Object 自指 claim 示例")
        for ex in claims_r["examples_self_claim"][:6]:
            out.append(f"- PMID {ex['pmid']} | {ex['predicate']} | **{ex['subject_name']}** = **{ex['object_name']}**")
            out.append(f"  > {ex['raw_preview']}")

    if claims_r["examples_object_injected"]:
        out.append("\n### Object token 缺失（LLM 注入嫌疑）示例")
        for ex in claims_r["examples_object_injected"][:8]:
            out.append(f"- PMID {ex['pmid']} | {ex['subject']} `{ex['predicate']}` → **{ex['object']}**")
            out.append(f"  > {ex['raw_preview']}")

    if claims_r["examples_subject_injected"]:
        out.append("\n### Subject token 缺失示例")
        for ex in claims_r["examples_subject_injected"][:6]:
            out.append(f"- PMID {ex['pmid']} | **{ex['subject']}** `{ex['predicate']}` → {ex['object']}")
            out.append(f"  > {ex['raw_preview']}")

    if claims_r["examples_vague_plus_precise_pair"]:
        out.append("\n### 模糊+精确谓词同源 claim 示例（重复 claim）")
        for ex in claims_r["examples_vague_plus_precise_pair"][:8]:
            out.append(f"- PMID {ex['pmid']} | {ex['subject_id']} → {ex['object_id']}: {ex['predicates']}")

    # ===== Hypotheses =====
    out.append("\n## Phase 3-4: 假设质量\n")
    out.append("| 文件 | 总数 | 重复 path | 含模糊谓词 | hub-to-hub | 含未解析 CLM 端点 | 单节点 path | path 长度分布 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for fname, rec in hyp_r["files"].items():
        if "error" in rec:
            out.append(f"| {fname} | ❌ {rec['error']} | | | | | | |")
            continue
        out.append(f"| {fname} | {rec['count']} | {rec['duplicate_path']} "
                   f"| {rec['vague_predicate_in_path']} | {rec['hub_to_hub']} "
                   f"| {rec['single_node_endpoint_unresolved_clm']} "
                   f"| {rec['single_node_path']} | {rec['path_length_counter']} |")

    out.append("\n### 各假设文件细节")
    for fname, rec in hyp_r["files"].items():
        if "error" in rec:
            continue
        out.append(f"#### {fname}")
        out.append(f"- 假设数: {rec['count']}")
        out.append(f"- 重复 path: {rec['duplicate_path']}")
        out.append(f"- 含模糊谓词: {rec['vague_predicate_in_path']}")
        out.append(f"- hub-to-hub 路径: {rec['hub_to_hub']}")
        out.append(f"- 含未解析 CLM_CONCEPT 端点: {rec['single_node_endpoint_unresolved_clm']}")
        out.append(f"- score < 0.2: {rec['low_score_lt_0_2']}")
        out.append(f"- target_domain: {rec['target_domain_counter']}")
        if rec.get("endpoint_examples"):
            out.append("- 含未解析端点示例:")
            for ex in rec["endpoint_examples"]:
                out.append(f"  - {ex['path']}  (ids={ex['endpoint_ids']})")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--hyp-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading KG from {args.kg} ...", flush=True)
    kg = load_kg(Path(args.kg))
    print(f"  concepts={len(kg.get('concepts', {}))}, edges={len(kg.get('edges', []))}",
          flush=True)

    print("Auditing nodes ...", flush=True)
    nodes_r = audit_nodes(kg)
    print("Auditing edges ...", flush=True)
    edges_r = audit_edges(kg)
    print("Auditing claims ...", flush=True)
    claims_r = audit_claims(Path(args.claims))
    print(f"  total claims: {claims_r['total_claims']}", flush=True)
    print("Auditing hypotheses ...", flush=True)
    hyp_r = audit_hypotheses(Path(args.hyp_dir))

    md = format_md(nodes_r, edges_r, claims_r, hyp_r)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
