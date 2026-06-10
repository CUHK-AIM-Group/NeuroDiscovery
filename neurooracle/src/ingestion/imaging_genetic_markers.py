"""Inject IM/GM nodes and their connecting edges into the v2 KG.

Both `imaging_markers.json` (88 IMs) and `genetic_markers.json` (73 GMs)
are operationalised marker objects that NeuroClaw's atom-aware hypothesis
engine treats as the IMAGING_MARKER and GENE_TARGET surfaces. The v2 KG
predates the marker registries and contains zero IM:*/GM:* nodes — this
module folds them in, plus 17 GENESET:* group nodes for gene-set bookkeeping.

Node creation
-------------
- IM:im_NNNN   (88) — domain_tags=["imaging_feature"] (+["paradigm"] when conditioned)
- GM:gm_NNNN   (73) — domain_tags=["gene"] (+["disease"] when GM has disease)
- GENESET:<slug> (17) — domain_tags=["gene"]

Edge creation (only existing RELATION_TYPES)
--------------------------------------------
- IM --is_a--> IF:<operation>                        (operation_id direct)
- IM --measured_by_modality--> MODALITY:<m>          (modality_id direct)
- IM --is_imaging_feature_of--> region                (id-first, name fallback)
- IM --is_associated_with--> COGAT_TASK / COGAT_CONCEPT (when conditioned)
- GM --gene_associated_with_disease--> CUI:disease    (name resolver)
- GM --is_associated_with--> trait-like anchor / cognitive concept
  (fallback when `disease` names a non-disease GWAS trait such as
  Cognitive Performance / Educational Attainment)
- GM --is_a--> IndividualDataAnchor: Polygenic Risk Score
  (for all PRS-family markers so trait PRS nodes never become isolated)
- GM --is_associated_with--> CUI:gene                 (name resolver on gene_symbols/gene_ids)
- GM --is_associated_with--> GENESET:<slug>           (direct)
- gene --part_of--> GENESET                           (membership union of GMs)

Idempotent — re-running adds zero new nodes/edges. Endpoints absent from
the KG are silently skipped, mirroring outcome_im_bridges.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, Edge
from ..storage import load_graph, save_graph
from .outcome_im_bridges import IM_TO_SCALE_EDGES

logger = logging.getLogger(__name__)

REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data" / "full_snapshot_v2"
DEFAULT_IM = REPO_DATA / "imaging_markers.json"
DEFAULT_GM = REPO_DATA / "genetic_markers.json"


# ── name index built once per run ─────────────────────────────────────────

def _build_name_index(kg: KnowledgeGraph) -> dict[str, list[ConceptNode]]:
    """Lowercased name → list of ConceptNodes (preferred_name + aliases)."""
    idx: dict[str, list[ConceptNode]] = defaultdict(list)
    for node in kg._index.values():
        names = {node.preferred_name} | set(node.aliases or [])
        for n in names:
            if not n:
                continue
            idx[n.strip().lower()].append(node)
    return idx


def _resolve_name(
    name: str,
    name_idx: dict[str, list[ConceptNode]],
    prefer_domain: Optional[str] = None,
    id_priority: tuple[str, ...] = ("CUI:", "NN:", "VROI:", "GENE:"),
) -> Optional[str]:
    """Pick the best concept id for `name`. None if no match."""
    if not name:
        return None
    cands = name_idx.get(name.strip().lower())
    if not cands:
        return None
    if prefer_domain:
        filtered = [c for c in cands if prefer_domain in (c.domain_tags or [])]
        if filtered:
            cands = filtered
    # tie-break: id-prefix priority, then shortest id
    def _key(c: ConceptNode):
        for i, p in enumerate(id_priority):
            if c.id.startswith(p):
                return (i, len(c.id))
        return (len(id_priority), len(c.id))
    cands.sort(key=_key)
    return cands[0].id


def _resolve_name_strict_domain(
    name: str,
    name_idx: dict[str, list[ConceptNode]],
    prefer_domain: str,
) -> Optional[str]:
    """Resolve `name` only if a candidate actually matches `prefer_domain`."""
    cid = _resolve_name(name, name_idx, prefer_domain=prefer_domain)
    if not cid:
        return None
    cands = name_idx.get(name.strip().lower()) or []
    for cand in cands:
        if cand.id == cid and prefer_domain in (cand.domain_tags or []):
            return cid
    return None


# ── edge helper (mirrors outcome_im_bridges.py) ───────────────────────────

def _add_edge_if_new(
    kg: KnowledgeGraph,
    source_id: str,
    target_id: str,
    relation_type: str,
    source: str,
    confidence: float = 0.9,
) -> bool:
    if not source_id or not target_id or source_id == target_id:
        return False
    if not (kg.has_concept(source_id) and kg.has_concept(target_id)):
        return False
    if kg.G.has_edge(source_id, target_id):
        existing = kg.G[source_id][target_id]
        if existing.get("relation_type") == relation_type:
            return False
    kg.add_edge(Edge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        source=source,
        confidence=confidence,
    ))
    return True


# ── IM injection ──────────────────────────────────────────────────────────

def _inject_ims(
    kg: KnowledgeGraph,
    ims: list[dict],
    name_idx: dict[str, list[ConceptNode]],
) -> dict:
    counts = Counter()
    for im in ims:
        im_id = f"IM:{im['id']}"
        tags = ["imaging_feature"]
        if im.get("conditioning"):
            tags.append("paradigm")
        meta = {
            "family":       im.get("family"),
            "modality":     im.get("modality"),
            "operation":    im.get("operation"),
            "formula":      im.get("formula"),
            "rationale":    im.get("rationale"),
            "regions":      im.get("regions") or [],
            "region_names": im.get("region_names") or [],
            "conditioning": im.get("conditioning"),
            "atoms":        im.get("atoms") or [],
            "llm_model":    im.get("llm_model"),
        }
        before = im_id in kg._index
        kg.add_concept(ConceptNode(
            id=im_id,
            preferred_name=im.get("formula") or im.get("name") or im_id,
            domain_tags=tags,
            source_vocab="NeuroClaw-IM",
            aliases=[im.get("name")] if im.get("name") else [],
            metadata=meta,
        ))
        if not before:
            counts["im_nodes_added"] += 1

        # IM → IF (is_a)
        op_id = im.get("operation_id")
        if op_id and _add_edge_if_new(kg, im_id, op_id, "is_a", "NeuroClaw-IM"):
            counts["im_to_if"] += 1

        # IM → MODALITY (measured_by_modality)
        mod_id = im.get("modality_id")
        if mod_id and _add_edge_if_new(kg, im_id, mod_id, "measured_by_modality", "NeuroClaw-IM"):
            counts["im_to_modality"] += 1

        # IM → region (is_imaging_feature_of)
        regions = im.get("regions") or []
        region_names = im.get("region_names") or []
        for i, rid in enumerate(regions):
            target = rid if kg.has_concept(rid) else None
            if not target:
                rname = region_names[i] if i < len(region_names) else ""
                target = _resolve_name(rname, name_idx, prefer_domain="neuroanatomy")
            if target and _add_edge_if_new(kg, im_id, target, "is_imaging_feature_of", "NeuroClaw-IM"):
                counts["im_to_region"] += 1

        # IM → COGAT_TASK / COGAT_CONCEPT (is_associated_with)
        cond = im.get("conditioning") or {}
        if isinstance(cond, dict):
            for key in ("task", "concept"):
                v = cond.get(key)
                if not v:
                    continue
                tid = _resolve_name(v, name_idx)
                if tid and _add_edge_if_new(kg, im_id, tid, "is_associated_with", "NeuroClaw-IM"):
                    counts[f"im_to_cogat_{key}"] += 1
    return dict(counts)


# ── GM + GENESET injection ────────────────────────────────────────────────

def _inject_gms_and_genesets(
    kg: KnowledgeGraph,
    gms: list[dict],
    name_idx: dict[str, list[ConceptNode]],
) -> dict:
    counts = Counter()

    # Pass 1: build the universe of gene_set slugs and the genes referenced
    # under each (union across all GMs that name that set).
    geneset_to_genes: dict[str, set[str]] = defaultdict(set)
    geneset_slugs: set[str] = set()
    for gm in gms:
        slug = gm.get("gene_set")
        if slug:
            geneset_slugs.add(slug)
            for sym in (gm.get("gene_symbols") or []):
                geneset_to_genes[slug].add(sym)
            for gid in (gm.get("gene_ids") or []):
                # gene_ids in the JSON are GENE:<symbol> form; strip prefix
                if gid.startswith("GENE:"):
                    geneset_to_genes[slug].add(gid.split(":", 1)[1])
                else:
                    geneset_to_genes[slug].add(gid)

    # Create GENESET nodes
    for slug in sorted(geneset_slugs):
        gs_id = f"GENESET:{slug}"
        before = gs_id in kg._index
        kg.add_concept(ConceptNode(
            id=gs_id,
            preferred_name=slug,
            domain_tags=["gene"],
            source_vocab="NeuroClaw-GeneSet",
            aliases=[slug.replace("_", " ")],
            metadata={"members_count": len(geneset_to_genes.get(slug, set()))},
        ))
        if not before:
            counts["geneset_nodes_added"] += 1

    # gene CUI → GENESET (part_of)
    for slug, gene_syms in geneset_to_genes.items():
        gs_id = f"GENESET:{slug}"
        for sym in gene_syms:
            cid = _resolve_name(sym, name_idx, prefer_domain="gene")
            if cid and _add_edge_if_new(kg, cid, gs_id, "part_of", "NeuroClaw-GeneSet"):
                counts["gene_to_geneset"] += 1

    # Pass 2: inject GM nodes + their edges
    for gm in gms:
        gm_id = f"GM:{gm['id']}"
        tags = ["gene"]
        if gm.get("disease"):
            tags.append("disease")
        meta = {
            "family":       gm.get("family"),
            "operation":    gm.get("operation"),
            "data_type":    gm.get("data_type"),
            "gene_symbols": gm.get("gene_symbols") or [],
            "gene_ids":     gm.get("gene_ids") or [],
            "gene_set":     gm.get("gene_set"),
            "tissue":       gm.get("tissue"),
            "tissues":      gm.get("tissues") or [],
            "clock":        gm.get("clock"),
            "disease":      gm.get("disease"),
            "formula":      gm.get("formula"),
            "rationale":    gm.get("rationale"),
            "atoms":        gm.get("atoms") or [],
            "llm_model":    gm.get("llm_model"),
        }
        before = gm_id in kg._index
        kg.add_concept(ConceptNode(
            id=gm_id,
            preferred_name=gm.get("name") or gm_id,
            domain_tags=tags,
            source_vocab="NeuroClaw-GM",
            aliases=[gm["name"]] if gm.get("name") else [],
            metadata=meta,
        ))
        if not before:
            counts["gm_nodes_added"] += 1

        # GM → disease / trait-like target
        disease = gm.get("disease")
        if disease:
            did = _resolve_name_strict_domain(disease, name_idx, prefer_domain="disease")
            if did and _add_edge_if_new(kg, gm_id, did, "gene_associated_with_disease", "NeuroClaw-GM"):
                counts["gm_to_disease"] += 1
            else:
                # Some GWAS-backed PRS targets are traits rather than diseases
                # (for example Cognitive Performance or Educational Attainment).
                # Those should still attach to the conceptual side of the KG.
                for trait_domain in ("individual_data_anchor", "cognitive_function"):
                    tid = _resolve_name(disease, name_idx, prefer_domain=trait_domain)
                    if tid and _add_edge_if_new(kg, gm_id, tid, "is_associated_with", "NeuroClaw-GM"):
                        counts["gm_to_trait"] += 1
                        break

        # Every PRS marker is a specialized instance of the subject-level
        # polygenic-risk-score anchor, which keeps trait PRS nodes connected
        # even when their GWAS label is not a disease concept in the KG.
        if gm.get("family") == "polygenic_risk":
            prs_anchor = _resolve_name(
                "Polygenic Risk Score",
                name_idx,
                prefer_domain="individual_data_anchor",
            )
            if prs_anchor and _add_edge_if_new(kg, gm_id, prs_anchor, "is_a", "NeuroClaw-GM"):
                counts["gm_to_prs_anchor"] += 1

        # GM → gene CUIs (collect both gene_symbols and gene_ids, dedup by symbol)
        symbols: set[str] = set()
        for s in (gm.get("gene_symbols") or []):
            if s:
                symbols.add(s)
        for gid in (gm.get("gene_ids") or []):
            if gid.startswith("GENE:"):
                symbols.add(gid.split(":", 1)[1])
            else:
                symbols.add(gid)
        for sym in symbols:
            cid = _resolve_name(sym, name_idx, prefer_domain="gene")
            if cid and _add_edge_if_new(kg, gm_id, cid, "is_associated_with", "NeuroClaw-GM"):
                counts["gm_to_gene"] += 1

        # GM → GENESET
        slug = gm.get("gene_set")
        if slug:
            gs_id = f"GENESET:{slug}"
            if _add_edge_if_new(kg, gm_id, gs_id, "is_associated_with", "NeuroClaw-GM"):
                counts["gm_to_geneset"] += 1

    return dict(counts)


# ── bridges that close Case Study 1/2 directed traversal ─────────────────

def _bridge_region_to_im(kg: KnowledgeGraph) -> dict:
    """Mirror IM --is_imaging_feature_of--> region as region --has_imaging_feature--> IM.

    Without this, disease/gene → region paths (ENIGMA, AHBA, HPO) hit a
    dead end at the region node — there is no out-edge into the IM atom
    surface. Mirror is dynamic so any future is_imaging_feature_of edge
    auto-bridges on the next run. Idempotent.
    """
    counts = Counter()
    pairs = [
        (u, v) for u, v, d in kg.G.edges(data=True)
        if d.get("relation_type") == "is_imaging_feature_of" and u.startswith("IM:")
    ]
    for im_id, region_id in pairs:
        if _add_edge_if_new(kg, region_id, im_id, "has_imaging_feature", "NeuroClaw-IM-Reverse"):
            counts["region_to_im"] += 1
    return dict(counts)


def _bridge_im_to_outcome(kg: KnowledgeGraph) -> dict:
    """Re-anchor outcome_im_bridges.IM_TO_SCALE_EDGES on the IM:* atom layer.

    The original table targets pre-canonicalisation NN:* / CLM_CONCEPT:*
    region ids that UMLS folds into CUI:* — most of those region rows now
    no-op. We re-emit (IM:* --predicts--> OUTCOME:scale) for every IM
    whose region set overlaps the bridge's region anchor (resolved by id
    or name). Without this, no IM:* node has an out-edge to OUTCOME:*,
    blocking Case Study 1's D -> IM -> O closure. Idempotent.
    """
    counts = Counter()

    # Build region-anchor -> {IM ids that touch it} (by both raw region id
    # and post-canon target id of the is_imaging_feature_of edge).
    region_to_ims: dict[str, set[str]] = defaultdict(set)
    name_to_ims: dict[str, set[str]] = defaultdict(set)
    for nid, node in kg._index.items():
        if not nid.startswith("IM:"):
            continue
        # raw region ids from the IM metadata (pre-canon NN:* keys still
        # appear in IM_TO_SCALE_EDGES even when the KG no longer has them)
        for r in node.metadata.get("regions") or []:
            region_to_ims[r].add(nid)
        for rn in node.metadata.get("region_names") or []:
            if rn:
                name_to_ims[rn.strip().lower()].add(nid)
        # also the canonicalised region the is_imaging_feature_of edge points at
        for _, tgt, d in kg.G.out_edges(nid, data=True):
            if d.get("relation_type") == "is_imaging_feature_of":
                region_to_ims[tgt].add(nid)

    for region_anchor, scale_id in IM_TO_SCALE_EDGES:
        ims = set(region_to_ims.get(region_anchor) or set())
        # if anchor is an NN:* / CLM_CONCEPT:* the KG dropped, fall back to
        # name-resolved IMs via the region's preferred_name (best-effort).
        if not ims and region_anchor in kg._index:
            anchor_name = kg._index[region_anchor].preferred_name
            ims |= name_to_ims.get(anchor_name.strip().lower(), set())
        for im_id in ims:
            if _add_edge_if_new(kg, im_id, scale_id, "predicts", "NeuroClaw-IM-Outcome"):
                counts["im_to_outcome"] += 1
    return dict(counts)


# ── public entry point ────────────────────────────────────────────────────

def ingest_imaging_genetic_markers(
    kg: KnowledgeGraph,
    im_path: Path = DEFAULT_IM,
    gm_path: Path = DEFAULT_GM,
) -> dict:
    """Inject IM/GM/GENESET nodes and edges into `kg`. Idempotent."""
    im_data = json.loads(Path(im_path).read_text(encoding="utf-8"))
    gm_data = json.loads(Path(gm_path).read_text(encoding="utf-8"))
    ims = im_data.get("imaging_markers") or []
    gms = gm_data.get("genetic_markers") or []
    logger.info(f"loaded {len(ims)} IMs from {im_path}")
    logger.info(f"loaded {len(gms)} GMs from {gm_path}")

    # Build name index against the *current* KG (no IMs/GMs/GENESETs yet on
    # first pass, so we won't accidentally resolve back into our own nodes).
    name_idx = _build_name_index(kg)
    logger.info(f"name index: {len(name_idx)} unique names")

    im_counts = _inject_ims(kg, ims, name_idx)
    gm_counts = _inject_gms_and_genesets(kg, gms, name_idx)
    bridge_region_counts = _bridge_region_to_im(kg)
    bridge_outcome_counts = _bridge_im_to_outcome(kg)

    counts = {**im_counts, **gm_counts, **bridge_region_counts, **bridge_outcome_counts}
    counts["total_edges_added"] = sum(
        v for k, v in counts.items()
        if k.endswith("_to_if") or k.endswith("_to_modality")
        or k.endswith("_to_region") or k.startswith("im_to_cogat_")
        or k == "gm_to_disease" or k == "gm_to_gene"
        or k == "gm_to_geneset" or k == "gene_to_geneset"
        or k == "gm_to_trait" or k == "gm_to_prs_anchor"
        or k == "region_to_im" or k == "im_to_outcome"
    )
    counts["total_nodes_added"] = (
        counts.get("im_nodes_added", 0)
        + counts.get("gm_nodes_added", 0)
        + counts.get("geneset_nodes_added", 0)
    )

    # The semantic_view cache (if any) is now stale.
    kg.invalidate_semantic_view()

    logger.info(f"IM/GM ingestion complete: {counts}")
    return counts


__all__ = ["ingest_imaging_genetic_markers"]


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Inject IM/GM/GENESET into KG")
    ap.add_argument("--kg", type=Path, default=REPO_DATA / "knowledge_graph.json")
    ap.add_argument("--im", type=Path, default=DEFAULT_IM)
    ap.add_argument("--gm", type=Path, default=DEFAULT_GM)
    ap.add_argument("--in-place", action="store_true",
                    help="write back to --kg path (default: also writes to --kg)")
    ap.add_argument("--output", type=Path, default=None,
                    help="optional explicit output path; overrides --in-place")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    kg = load_graph(args.kg)
    pre_n_concepts = len(kg._index)
    pre_n_edges = kg.G.number_of_edges()

    counts = ingest_imaging_genetic_markers(kg, args.im, args.gm)

    out = args.output or args.kg
    save_graph(kg, out)

    print()
    print("=" * 70)
    print("IM/GM INGESTION RESULTS")
    print("=" * 70)
    print(f"input KG  : {pre_n_concepts:>6} concepts, {pre_n_edges:>6} edges")
    print(f"output KG : {len(kg._index):>6} concepts, {kg.G.number_of_edges():>6} edges  -> {out}")
    print()
    for k, v in sorted(counts.items()):
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
