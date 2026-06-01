"""UMLS-based KG canonicalization (post-ingest migration).

Run AFTER all ingestors have populated the KG with native ids
(MSH:Dxxx, GENE:APOE, DISGENET:Cxxx, ATC:N06DA02, COGAT_DISORDER:dso_*,
NN:..., OUTCOME:..., INDIVIDUAL_DATA:..., etc.). This module:

  1. Streams MRCONSO.RRF once and builds three lookup tables:
       (a) (SAB, source_code) -> CUI        - exact source-id match
       (b) (domain, name_lower) -> CUI      - SAB-restricted name match
       (c) CUI -> {preferred_name, aliases, source_codes}
  2. For every node in the KG that is in scope (see ELIGIBLE_PREFIXES),
     resolve it to a CUI. Hit -> rename to "CUI:Cxxxxxxx", merge aliases
     and external_ids into the CUI node. Miss -> keep native id.
  3. Update every edge's source_id / target_id to the new node ids.

Out-of-scope prefixes (ATLAS, MODALITY, MODEL, DATASET, IF, VROI,
COGAT_TASK, COGAT_CONCEPT, UKB, ADNI, HCP, INDIVIDUAL_DATA,
CLM_CONCEPT, NN_TAL/NN_HO atlas-specific NN children) keep their
native ids.

Atlas-specific NN children (NN:NN_TAL:* / NN:NN_HO:*) are intentionally
NOT canonicalized: they share names with the NeuroNames main hierarchy
but represent a different parcellation, and UMLS would collapse them.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode

logger = logging.getLogger(__name__)

# MRCONSO.RRF columns (0-indexed)
COL_CUI = 0
COL_LAT = 1
COL_TS = 2          # P=preferred name within concept
COL_ISPREF = 6      # Y/N preferred-in-source
COL_SAB = 11
COL_CODE = 13       # source vocabulary code (HGNC:xxxx, ATC code, MeSH UI, OMIM id, ...)
COL_STR = 14
COL_SUPPRESS = 16

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_MRCONSO = DEFAULT_DATA_DIR / "MRCONSO.RRF"
DEFAULT_CACHE = DEFAULT_DATA_DIR / "_umls_canonicalize_cache.json"

# Domains we attempt to canonicalize. Other prefixes keep native ids.
# Map: id_prefix -> (id_strategy, name_match_sabs, domain_label)
#   id_strategy: how to derive the SAB-code lookup key from the node id
#   name_match_sabs: SAB allow-list when falling back to name match
#   domain_label: used to bucket the (domain, name) lookup table
ELIGIBLE_PREFIXES: dict[str, dict] = {
    "MSH": {
        "sab_code": ("MSH", "external_ids:MeSH_UI"),
        "name_sabs": ("MSH", "SNOMEDCT_US"),
        "domain": "biomed",
    },
    "DISGENET": {
        # DISGENET id is itself a CUI - direct adoption
        "sab_code": ("CUI", "id_suffix"),
        "name_sabs": ("MSH", "SNOMEDCT_US"),
        "domain": "disease",
    },
    "GENE": {
        "sab_code": ("HGNC", "preferred_name_as_hgnc_symbol"),
        "name_sabs": ("HGNC",),
        "domain": "gene",
    },
    "ATC": {
        # only leaf ATC codes (>=5 chars, alphanumeric chemical-substance level)
        "sab_code": ("ATC", "id_suffix_if_leaf"),
        "name_sabs": ("ATC", "RXNORM", "MSH"),
        "domain": "drug",
    },
    "COGAT_DISORDER": {
        # CogAt has no source-id mapping in UMLS; rely on name match against MSH
        "sab_code": None,
        "name_sabs": ("MSH",),
        "domain": "disease",
    },
    "OUTCOME": {
        # Clinical scales: name match against MSH (T058 Health Care Activity)
        "sab_code": None,
        "name_sabs": ("MSH",),
        "domain": "scale",
    },
    "INDIVIDUAL_DATA": {
        "sab_code": None,
        "name_sabs": ("MSH",),
        "domain": "individual_data",
    },
    "NN": {
        # Only canonicalize main-hierarchy NN nodes. Atlas-specific
        # children (NN:NN_TAL:*, NN:NN_HO:*) keep native ids - skipped at
        # node selection time below.
        "sab_code": None,
        "name_sabs": ("MSH", "FMA", "NCI"),
        "domain": "neuroanatomy",
    },
    "CLM_CONCEPT": {
        "sab_code": None,
        "name_sabs": ("MSH",),
        "domain": "individual_data",
    },
}

# Prefixes that NEVER canonicalize (research-only entities).
SKIP_PREFIXES = {
    "ATLAS", "MODALITY", "MODEL", "DATASET",
    "IF", "VROI", "COGAT_TASK", "COGAT_CONCEPT",
    "UKB", "ADNI", "HCP",
}


# ── MRCONSO scan ──────────────────────────────────────────────────────


def _scan_mrconso(
    mrconso_path: Path,
    target_codes: dict[str, set[str]],
    target_names: dict[str, set[str]],
) -> dict:
    """Single streaming pass of MRCONSO. Returns:

      {
        "code_to_cui":   {(sab, code): cui},
        "name_to_cui":   {(sab, name_lower): cui},
        "cui_info":      {cui: {"preferred_name", "aliases", "source_codes",
                                "semantic_types"}},
      }
    Only retains rows that match target_codes (by sab) or target_names
    (by sab) to keep memory bounded.
    """
    code_to_cui: dict[tuple[str, str], str] = {}
    name_to_cui: dict[tuple[str, str], tuple[str, bool]] = {}  # (sab, name_lower) -> (cui, is_pref)
    cui_info: dict[str, dict] = {}

    sabs_we_need = set(target_codes.keys()) | set(target_names.keys())

    line_count = 0
    with open(mrconso_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if line_count % 5_000_000 == 0:
                logger.info(
                    f"  MRCONSO scan: {line_count/1e6:.0f}M lines, "
                    f"codes={len(code_to_cui):,} names={len(name_to_cui):,}"
                )
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            if parts[COL_LAT] != "ENG":
                continue
            if parts[COL_SUPPRESS] in ("O", "Y"):
                continue
            sab = parts[COL_SAB]
            if sab not in sabs_we_need:
                continue

            cui = parts[COL_CUI]
            code = parts[COL_CODE]
            name = parts[COL_STR].strip()
            if not name:
                continue
            is_pref = parts[COL_TS] == "P" or parts[COL_ISPREF] == "Y"

            # source-code match: only for SABs we declared interest in
            if sab in target_codes and code in target_codes[sab]:
                key = (sab, code)
                if key not in code_to_cui:
                    code_to_cui[key] = cui

            # name match: only if this SAB is a name-match source
            if sab in target_names:
                name_lower = name.lower()
                if name_lower in target_names[sab]:
                    nkey = (sab, name_lower)
                    existing = name_to_cui.get(nkey)
                    if existing is None or (is_pref and not existing[1]):
                        name_to_cui[nkey] = (cui, is_pref)

            # cui_info: collect for every relevant SAB row
            info = cui_info.setdefault(cui, {
                "preferred_name": "",
                "aliases": [],
                "source_codes": {},
                "_pref_score": -1,
            })
            score = (1 if parts[COL_TS] == "P" else 0) + (1 if parts[COL_ISPREF] == "Y" else 0)
            if score > info["_pref_score"]:
                if info["preferred_name"] and info["preferred_name"] != name:
                    info["aliases"].append(info["preferred_name"])
                info["preferred_name"] = name
                info["_pref_score"] = score
            elif name != info["preferred_name"] and name not in info["aliases"]:
                info["aliases"].append(name)
            info["source_codes"].setdefault(sab, set()).add(code)

    logger.info(
        f"MRCONSO scan complete: {line_count:,} lines, "
        f"{len(code_to_cui):,} code matches, {len(name_to_cui):,} name matches, "
        f"{len(cui_info):,} CUIs collected"
    )

    # finalize: drop scratch fields; convert sets to sorted lists
    name_to_cui_final = {k: v[0] for k, v in name_to_cui.items()}
    for cui, info in cui_info.items():
        info.pop("_pref_score", None)
        info["source_codes"] = {sab: sorted(codes) for sab, codes in info["source_codes"].items()}
        info["aliases"] = sorted(set(info["aliases"]))[:50]  # cap aliases

    return {
        "code_to_cui": {f"{sab}|{code}": cui for (sab, code), cui in code_to_cui.items()},
        "name_to_cui": {f"{sab}|{n}": cui for (sab, n), cui in name_to_cui_final.items()},
        "cui_info": cui_info,
    }


# ── KG-side helpers ──────────────────────────────────────────────────


def _is_atlas_specific_nn(node_id: str) -> bool:
    """NN:NN_TAL:* / NN:NN_HO:* - atlas-specific children, skip."""
    return node_id.startswith("NN:NN_TAL:") or node_id.startswith("NN:NN_HO:")


def _node_lookup_keys(node: ConceptNode, prefix: str, spec: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build (sab,code) and (sab,name) lookup keys for a single node.

    Returns (code_keys, name_keys). Each list is in priority order.
    """
    code_keys: list[tuple[str, str]] = []
    name_keys: list[tuple[str, str]] = []

    sab_code = spec.get("sab_code")
    if sab_code:
        sab, strategy = sab_code
        if strategy == "external_ids:MeSH_UI":
            ui = node.external_ids.get("MeSH_UI", "")
            if ui:
                code_keys.append((sab, ui))
        elif strategy == "id_suffix":
            suffix = node.id.split(":", 1)[1] if ":" in node.id else ""
            if suffix:
                code_keys.append((sab, suffix))
        elif strategy == "preferred_name_as_hgnc_symbol":
            sym = node.preferred_name.strip()
            if sym:
                code_keys.append((sab, f"HGNC:{sym}"))
        elif strategy == "id_suffix_if_leaf":
            suffix = node.id.split(":", 1)[1] if ":" in node.id else ""
            # ATC leaf level == 7 chars (e.g. N06DA02). Skip tree-level codes
            # (1/3/4/5 chars) and our hand-coined N06D_LECANEMAB.
            if len(suffix) == 7 and "_" not in suffix:
                code_keys.append((sab, suffix))

    name_sabs = spec.get("name_sabs", ())
    name_lower = node.preferred_name.lower().strip()
    if name_lower:
        for sab in name_sabs:
            name_keys.append((sab, name_lower))
        # also try aliases
        for alias in node.aliases[:10]:
            al = alias.lower().strip()
            if al and al != name_lower:
                for sab in name_sabs:
                    name_keys.append((sab, al))

    return code_keys, name_keys


def _resolve_cui(
    node: ConceptNode,
    prefix: str,
    spec: dict,
    code_to_cui: dict[str, str],
    name_to_cui: dict[str, str],
) -> Optional[str]:
    """Return CUI for this node, or None if no match."""
    code_keys, name_keys = _node_lookup_keys(node, prefix, spec)

    # special: DISGENET id is already a CUI - just adopt it (validated by cui_info)
    if prefix == "DISGENET" and node.id.startswith("DISGENET:C"):
        return node.id.split(":", 1)[1]

    for sab, code in code_keys:
        cui = code_to_cui.get(f"{sab}|{code}")
        if cui:
            return cui
    for sab, name in name_keys:
        cui = name_to_cui.get(f"{sab}|{name}")
        if cui:
            return cui
    return None


# ── public entry ─────────────────────────────────────────────────────


def build_or_load_lookup(
    kg: KnowledgeGraph,
    mrconso_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> dict:
    """Build (or load cached) MRCONSO lookup tables targeted at this KG."""
    mrconso_path = Path(mrconso_path) if mrconso_path else DEFAULT_MRCONSO
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE

    if not mrconso_path.exists():
        raise FileNotFoundError(f"MRCONSO.RRF not found at {mrconso_path}")

    if cache_path.exists() and not force_rebuild:
        try:
            cache_mtime = cache_path.stat().st_mtime
            mrconso_mtime = mrconso_path.stat().st_mtime
            if cache_mtime > mrconso_mtime:
                with open(cache_path, "r", encoding="utf-8") as f:
                    logger.info(f"loading UMLS lookup cache from {cache_path.name}")
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.info("UMLS lookup cache unreadable; rebuilding")

    # Collect targets from KG
    target_codes: dict[str, set[str]] = defaultdict(set)
    target_names: dict[str, set[str]] = defaultdict(set)

    for nid, node in kg._index.items():
        prefix = nid.split(":", 1)[0]
        if prefix not in ELIGIBLE_PREFIXES:
            continue
        if prefix == "NN" and _is_atlas_specific_nn(nid):
            continue
        spec = ELIGIBLE_PREFIXES[prefix]
        code_keys, name_keys = _node_lookup_keys(node, prefix, spec)
        for sab, code in code_keys:
            target_codes[sab].add(code)
        for sab, name in name_keys:
            target_names[sab].add(name)

    logger.info(
        f"MRCONSO targets: codes={ {s: len(v) for s, v in target_codes.items()} }, "
        f"names={ {s: len(v) for s, v in target_names.items()} }"
    )

    result = _scan_mrconso(
        mrconso_path,
        {sab: codes for sab, codes in target_codes.items()},
        {sab: names for sab, names in target_names.items()},
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    logger.info(f"cached UMLS lookup tables -> {cache_path}")
    return result


def canonicalize_kg(
    kg: KnowledgeGraph,
    mrconso_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
    dry_run: bool = False,
) -> dict:
    """Migrate KG node ids to CUI:Cxxxxxxx where UMLS resolves them.

    In-place rewrite of:
      - kg._index keys
      - kg.G node names (rebuild graph by relabel_nodes equivalent)
      - every edge's source/target

    Skipped prefixes (atlas/modality/model/dataset/IF/VROI/COGAT_TASK/
    COGAT_CONCEPT/UKB/ADNI/HCP/CLM_CONCEPT non-canonicalizable) keep
    their native ids.
    """
    lookup = build_or_load_lookup(kg, mrconso_path, cache_path, force_rebuild)
    code_to_cui = lookup["code_to_cui"]
    name_to_cui = lookup["name_to_cui"]
    cui_info = lookup["cui_info"]

    # Phase 1: build remap (old_id -> new_id)
    remap: dict[str, str] = {}
    by_prefix_resolved = defaultdict(int)
    by_prefix_unresolved = defaultdict(int)
    by_prefix_skipped = defaultdict(int)

    for nid, node in list(kg._index.items()):
        prefix = nid.split(":", 1)[0]
        if prefix in SKIP_PREFIXES:
            by_prefix_skipped[prefix] += 1
            continue
        if prefix not in ELIGIBLE_PREFIXES:
            by_prefix_skipped[prefix] += 1
            continue
        if prefix == "NN" and _is_atlas_specific_nn(nid):
            by_prefix_skipped["NN(atlas-specific)"] += 1
            continue

        spec = ELIGIBLE_PREFIXES[prefix]
        cui = _resolve_cui(node, prefix, spec, code_to_cui, name_to_cui)
        if cui:
            new_id = f"CUI:{cui}"
            if new_id != nid:
                remap[nid] = new_id
            by_prefix_resolved[prefix] += 1
        else:
            by_prefix_unresolved[prefix] += 1

    logger.info(f"resolved by prefix: {dict(by_prefix_resolved)}")
    logger.info(f"unresolved by prefix: {dict(by_prefix_unresolved)}")
    logger.info(f"skipped by prefix: {dict(by_prefix_skipped)}")
    logger.info(f"total remappings (id changes): {len(remap)}")

    if dry_run:
        return {
            "remap": remap,
            "resolved": dict(by_prefix_resolved),
            "unresolved": dict(by_prefix_unresolved),
            "skipped": dict(by_prefix_skipped),
        }

    # Phase 2: apply remap to _index, merging duplicates.
    # Priority for "which node keeps its preferred_name when multiple
    # natives map to the same CUI": MSH > NN > GENE > DISGENET > others.
    # Sort kg._index entries so the first one inserted into new_index for
    # a given CUI is the most authoritative-name source.
    _NAME_PRIORITY = {
        "MSH": 0, "NN": 1, "GENE": 2, "OUTCOME": 3,
        "INDIVIDUAL_DATA": 4, "COGAT_DISORDER": 5,
        "DISGENET": 6, "ATC": 7, "CLM_CONCEPT": 8,
    }
    def _prio(item):
        nid = item[0]
        prefix = nid.split(":", 1)[0]
        return _NAME_PRIORITY.get(prefix, 99)

    merged_count = 0
    new_index: dict[str, ConceptNode] = {}
    for nid, node in sorted(kg._index.items(), key=_prio):
        new_id = remap.get(nid, nid)
        cui_for_enrich = new_id.split(":", 1)[1] if new_id.startswith("CUI:") else None

        if new_id in new_index:
            # merge into existing
            existing = new_index[new_id]
            existing.aliases = sorted(set(existing.aliases + [node.preferred_name] + node.aliases))
            existing.external_ids.update(node.external_ids)
            for tag in node.domain_tags:
                if tag not in existing.domain_tags:
                    existing.domain_tags.append(tag)
            for st in node.semantic_types:
                if st not in existing.semantic_types:
                    existing.semantic_types.append(st)
            if not existing.definition and node.definition:
                existing.definition = node.definition
            if not existing.atlas_mapping and node.atlas_mapping:
                existing.atlas_mapping = node.atlas_mapping
            existing.metadata.setdefault("merged_from", []).append(nid)
            merged_count += 1
        else:
            node.id = new_id
            if cui_for_enrich:
                # Keep the node's existing preferred_name (often MSH/HGNC -
                # more readable than UMLS canonical strings like "Structure
                # of X"). Just append UMLS preferred_name + UMLS aliases as
                # aliases, and propagate source codes to external_ids.
                info = cui_info.get(cui_for_enrich)
                if info:
                    umls_pref = info.get("preferred_name", "")
                    if umls_pref and umls_pref != node.preferred_name and umls_pref not in node.aliases:
                        node.aliases.append(umls_pref)
                    for al in info.get("aliases", [])[:30]:
                        if al not in node.aliases and al != node.preferred_name:
                            node.aliases.append(al)
                    for sab, codes in info.get("source_codes", {}).items():
                        for code in codes[:1]:
                            node.external_ids.setdefault(sab, code)
                node.external_ids.setdefault("UMLS_CUI", cui_for_enrich)
            new_index[new_id] = node

    kg._index = new_index

    # Phase 3: rebuild kg.G with relabeled nodes + edges
    import networkx as nx
    new_G = nx.DiGraph()
    for nid, node in kg._index.items():
        new_G.add_node(nid, **node.to_dict())
    edges_dropped_self_loop = 0
    edges_kept = 0
    for u, v, data in kg.G.edges(data=True):
        nu = remap.get(u, u)
        nv = remap.get(v, v)
        if nu == nv:
            edges_dropped_self_loop += 1
            continue
        if nu not in kg._index or nv not in kg._index:
            continue
        # update embedded source_id/target_id in edge data
        data = dict(data)
        data["source_id"] = nu
        data["target_id"] = nv
        if new_G.has_edge(nu, nv):
            existing = new_G.edges[nu, nv]
            if data.get("confidence", 0) > existing.get("confidence", 0):
                new_G.edges[nu, nv].update(data)
        else:
            new_G.add_edge(nu, nv, **data)
            edges_kept += 1
    kg.G = new_G
    kg.invalidate_semantic_view()

    summary = {
        "total_nodes_after": len(kg._index),
        "remapped": len(remap),
        "merged_into_existing": merged_count,
        "edges_after": kg.G.number_of_edges(),
        "edges_dropped_self_loop_after_merge": edges_dropped_self_loop,
        "resolved_by_prefix": dict(by_prefix_resolved),
        "unresolved_by_prefix": dict(by_prefix_unresolved),
        "skipped_by_prefix": dict(by_prefix_skipped),
    }
    logger.info(f"canonicalize_kg done: {summary}")
    return summary
