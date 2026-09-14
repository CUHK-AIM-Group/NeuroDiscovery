"""Conservative, non-mutating triage of atomic mention/canonical-node pairs.

The only verified resolution is a previously accepted, unique mapping to the
same recorded canonical identity. Lexical matches never authorize a merge,
new edge, review promotion, or redirection of a parent mention's claim edges.
"""

from __future__ import annotations

import re
from collections import Counter

try:
    from .umls_mention_mapping import normalize_term
except ImportError:  # Direct scripts use the source directory on sys.path.
    from umls_mention_mapping import normalize_term


POLICY_VERSION = "existing-alignment-triage-v2"
ALFF_SOURCE = "https://pubmed.ncbi.nlm.nih.gov/18501969/"
DECISION_LABELS = {
    "verified_existing_reuse": "已存在且身份一致的单一已接受映射",
    "existing_reuse_metadata_mismatch": "已经复用同一CUI，但新旧语义类型不一致",
    "existing_mapping_needs_review": "已有对应边，但仍属待审核映射",
    "review_source_trace": "来源文本或原子边界需复核",
    "blocked_known_alias_conflation": "已发现别名混用，阻断该对应候选",
    "review_dataset_or_parcellation_scope": "数据字段或分区范围不同，不能按同名合并",
    "review_method_vs_measurement": "检查方法与定量指标的层级需区分",
    "review_broad_anchor_scope": "泛化类别或宽泛锚点需核对范围",
    "review_umls_identity_conflict": "与当前UMLS候选标识不一致",
    "review_semantic_type_mismatch": "双方已有语义类型无交集，需复核",
    "review_umls_ambiguity": "仍有多个或待审核UMLS解释",
    "review_missing_claim_context": "未找到规范端点关联的claim上下文",
    "review_local_acronym": "本地缩写对应候选，需确认展开及语境",
    "review_local_name": "本地名称/别名对应候选，尚缺身份确证",
    "review_lexical_only": "仅有名称和领域线索，尚缺身份确证",
}

LEXICAL_FLAGS = {
    "laterality_in_parent": re.compile(r"\b(?:left|right|bilateral)\b", re.I),
    "change_or_direction_in_parent": re.compile(r"\b(?:increased?|decreased?|reduced?|higher|lower|impaired|improved|change[sd]?)\b", re.I),
    "time_or_stage_in_parent": re.compile(r"\b(?:baseline|follow[- ]?up|longitudinal|early|late|acute|chronic|months?|years?)\b", re.I),
    "nonhuman_cue_in_parent": re.compile(r"\b(?:mice|mouse|rats?|murine|zebrafish|monkeys?)\b", re.I),
}


def recorded_cuis(node: dict) -> set[str]:
    """Compare recorded identifiers, without guessing a CUI from a label."""
    result = set()
    node_id = str(node.get("id") or "")
    if node_id.startswith("CUI:") and node_id[4:]:
        result.add(node_id[4:])
    for key, value in (node.get("external_ids") or {}).items():
        if re.sub(r"[^a-z]", "", str(key).lower()) not in {"umlscui", "cui"}:
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item.strip():
                result.add(item.strip().removeprefix("CUI:"))
    return result


def semantic_types(node: dict) -> set[str]:
    result = set()
    for value in node.get("semantic_types") or []:
        tui = value.get("tui") if isinstance(value, dict) else value
        if isinstance(tui, str) and tui:
            result.add(tui)
    return result


def is_acronym_like(surface: str, target: dict) -> bool:
    letters = [char for char in surface if char.isalpha()]
    expanded = normalize_term(surface) != normalize_term(target.get("preferred_name") or "")
    return expanded and bool(letters) and len(surface.split()) == 1 and (
        len(letters) <= 3 or (len(letters) <= 10 and sum(char.isupper() for char in letters) >= 2)
    )


def source_trace(atom: dict, parent: dict) -> dict:
    md = atom.get("metadata") or {}
    span = md.get("evidence_span") or {}
    source = md.get("source_text")
    start, end = span.get("start"), span.get("end")
    valid_bounds = (
        isinstance(source, str) and type(start) is int and type(end) is int
        and 0 <= start < end <= len(source)
    )
    checks = {
        "parent_id_matches": md.get("source_mention_id") == parent.get("id"),
        "source_text_matches_parent": source == parent.get("preferred_name"),
        "span_field_matches": span.get("field") == "preferred_name",
        "span_bounds_valid": valid_bounds,
        "span_text_matches": bool(valid_bounds and source[start:end] == span.get("text") == atom.get("preferred_name")),
    }
    return {"valid": all(checks.values()), "checks": checks, "span": span,
            "full_parent_surface": normalize_term(source or "") == normalize_term(atom.get("preferred_name") or "")}


def context_summary(parent: dict, claims: list[dict]) -> dict:
    parent_name = parent.get("preferred_name") or ""
    flags = [key for key, regex in LEXICAL_FLAGS.items() if regex.search(parent_name)]
    negated = sum((claim.get("metadata") or {}).get("negated") is True for claim in claims)
    if negated:
        flags.append("negated_linked_claim_present")
    if not claims:
        flags.append("missing_canonical_endpoint_claim_context")
    scopes = Counter()
    pmids = set()
    for claim in claims:
        md = claim.get("metadata") or {}
        scopes.update(md.get("claim_case_study_ids") or ["general"])
        paper = md.get("source_paper") or {}
        if isinstance(paper, dict) and paper.get("pmid"):
            pmids.add(str(paper["pmid"]))
    return {
        "claim_ids": sorted(claim["id"] for claim in claims), "claim_count": len(claims),
        "distinct_pmids": len(pmids), "negated_claim_count": negated,
        "task_scope_counts": dict(sorted(scopes.items())), "lexical_caution_flags": sorted(flags),
        "flag_scope": "Lexical cues in the complete parent/claim, not asserted modifiers of this atom.",
    }


def review_pair(atom: dict, target: dict, parent: dict, mappings: list[dict], *,
                claim_context: dict, candidate_target_count: int, in_core: bool) -> dict:
    md = atom.get("metadata") or {}
    surface = str(atom.get("preferred_name") or "")
    normalized = normalize_term(surface)
    matches = []
    for field, text in [("preferred_name", target.get("preferred_name") or ""),
                        *((f"aliases[{index}]", text) for index, text in enumerate(target.get("aliases") or []))]:
        if normalize_term(text) == normalized:
            matches.append({"field": field, "text": text})
    shared_domains = sorted(set(atom.get("domain_tags") or []) & set(target.get("domain_tags") or []))
    if not matches or not shared_domains:
        raise ValueError("pair no longer satisfies its frozen lexical/domain candidate definition")
    trace = source_trace(atom, parent)
    target_cuis = recorded_cuis(target)
    mapped_targets = {row["record"]["target_id"] for row in mappings}
    mapped_cuis = {target_id[4:] for target_id in mapped_targets if target_id.startswith("CUI:")}
    direct = [row for row in mappings if row["record"]["target_id"] == target["id"]]
    accepted = [row for row in direct if (row["record"].get("metadata") or {}).get("review_status") == "auto_accepted_exact"]
    direct_review = [row for row in direct if (row["record"].get("metadata") or {}).get("review_status") == "needs_review"]
    atom_tuis, target_tuis = semantic_types(atom), semantic_types(target)
    flags = list(claim_context["lexical_caution_flags"])
    if candidate_target_count > 1:
        flags.append("multiple_lexical_targets")
    if md.get("atomization_rule") == "top_level_composite_split":
        flags.append("atom_is_component_not_entire_parent")
    if atom_tuis and target_tuis and not atom_tuis & target_tuis:
        flags.append("disjoint_recorded_semantic_types")
    acronym = is_acronym_like(surface, target)
    if acronym:
        flags.append("acronym_or_short_alias")
    prefix = target["id"].split(":", 1)[0]
    definition = str(target.get("definition") or "")
    tmd = target.get("metadata") or {}
    rationale, citations = [], []
    valid_accepted_identity = (
        md.get("mapping_status") == "auto_accepted_exact" and md.get("mapping_count") == 1
        and len(mappings) == 1 and len(accepted) == 1 and len(mapped_targets) == 1
        and len(target_cuis) == 1 and target_cuis == mapped_cuis
        and accepted[0]["record"].get("metadata", {}).get("semantic_compatibility") == "compatible"
        and not accepted[0]["record"].get("metadata", {}).get("candidate_overflow")
    )
    # A concrete pre-existing alias error, checked against the primary paper.
    if target["id"] == "IF:alff" and normalized == "falff":
        decision = "blocked_known_alias_conflation"
        rationale.append("fALFF is a fractional ratio measure; the stored target defines ALFF amplitude. Existing alias membership is not identity evidence.")
        citations.append(ALFF_SOURCE)
    elif not trace["valid"]:
        decision = "review_source_trace"
        rationale.append("Parent identity, verbatim source, or exact span requires inspection.")
    elif valid_accepted_identity and "disjoint_recorded_semantic_types" in flags:
        decision = "existing_reuse_metadata_mismatch"
        rationale.append("The accepted edge already reuses the same recorded CUI, but old/new TUI sets disagree. Reconcile metadata against MRSTY; do not create another entity.")
    elif valid_accepted_identity:
        decision = "verified_existing_reuse"
        rationale.append("One previously accepted compatible mapping already points to this exact existing CUI. No new canonical node or edge is needed.")
    elif direct_review:
        decision = "existing_mapping_needs_review"
        rationale.append("This exact target already has a maps_to edge, but its status is needs_review; it is not promoted here.")
    elif prefix in {"UKB", "ADNI", "HCP", "ATLAS", "VROI"} or target["id"].startswith(("NN:NN_TAL:", "NN:NN_HO:")) or tmd.get("parent_dataset"):
        decision = "review_dataset_or_parcellation_scope"
        rationale.append("A dataset field/category or atlas-specific representation is not made identical to a general mention by a label match.")
    elif target["id"] == "IF:fdg_uptake" and normalized in {"fdg pet", "fdg-pet"}:
        decision = "review_method_vs_measurement"
        rationale.append("The alias names an imaging method while the target is a quantitative readout; inspect the claim's intended measurement.")
    elif re.search(r"\b(?:broad|generic)\b", definition, re.I) or ("modality" in target.get("domain_tags", []) and prefix.startswith("NCL")):
        decision = "review_broad_anchor_scope"
        rationale.append("The target's stored definition marks a broad/generic category; preserve the distinction from a specific construct or measurement.")
    elif target_cuis and mapped_cuis and target_cuis.isdisjoint(mapped_cuis):
        decision = "review_umls_identity_conflict"
        rationale.append("The candidate's recorded CUI is absent from this atom's current UMLS mappings; shared spelling is insufficient to change identity.")
    elif "disjoint_recorded_semantic_types" in flags:
        decision = "review_semantic_type_mismatch"
        rationale.append("Existing TUI sets do not overlap. This is a review flag, not proof that the biological entities differ.")
    elif md.get("mapping_status") == "needs_review":
        decision = "review_umls_ambiguity"
        rationale.append("The source atom still has unresolved UMLS candidates; the local label match does not resolve them automatically.")
    elif not claim_context["claim_count"]:
        decision = "review_missing_claim_context"
        rationale.append("No CLM node uses this parent as its canonical subject or object in the current graph.")
    elif not target_cuis and acronym:
        decision = "review_local_acronym"
        rationale.append("The old node records this abbreviation, but context must confirm its expansion and measurement scope.")
    elif not target_cuis:
        decision = "review_local_name"
        rationale.append("A local canonical name/alias and domain agree; this is a mapping proposal, not an approved entity merge.")
    else:
        decision = "review_lexical_only"
        rationale.append("No unique previously accepted mapping verifies this pair; retain it for evidence-based review.")
    return {
        "policy_version": POLICY_VERSION, "atom_id": atom["id"], "target_id": target["id"],
        "source_mention_id": parent["id"], "atom_name": surface, "target_name": target.get("preferred_name"),
        "parent_text": parent.get("preferred_name"), "target_definition": definition,
        "mapping_status": md.get("mapping_status"), "in_core": in_core,
        "decision": decision, "decision_label": DECISION_LABELS[decision], "rationale": rationale,
        "matched_labels": matches, "shared_domains": shared_domains, "source_trace": trace,
        "target_recorded_cuis": sorted(target_cuis), "current_umls_cuis": sorted(mapped_cuis),
        "atom_semantic_types": sorted(atom_tuis), "target_semantic_types": sorted(target_tuis),
        "existing_direct_mapping_refs": [row["mapping_ref"] for row in direct],
        "all_mapping_refs": [row["mapping_ref"] for row in mappings],
        "candidate_target_count": candidate_target_count, "context": claim_context,
        "caution_flags": sorted(set(flags)), "citations": citations,
        "merge_authorized": False, "parent_redirect_authorized": False, "review_promotion_authorized": False,
        "proposed_action": "reuse_existing_mapping_without_new_graph_records" if decision in {"verified_existing_reuse", "existing_reuse_metadata_mismatch"} else "hold_pair_without_graph_mutation",
    }
