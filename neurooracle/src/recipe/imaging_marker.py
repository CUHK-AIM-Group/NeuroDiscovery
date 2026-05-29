"""IM (imaging marker) brainstorming: structured composition over KG primitives.

Goal: produce a Phase 1 catalogue of imaging-derived markers — each one a
scalar (or vector) quantity that an analysis pipeline could compute from a
single subject's data — by composing the KG's existing IM primitives:

  * 15 imaging-feature operations  (IF:* nodes — thickness, FA, FC, SUVR, …)
  *  6 imaging modalities          (MODALITY:{sMRI,dMRI,fMRI,PET,EEG,MEG})
  * region pool                    (NN:* + VROI:*)
  * conditioning pool              (COGAT_TASK + COGAT_CONCEPT)

The LLM never invents an operation or modality; it picks them from the
palette and validates against a static modality<->operation compatibility
table. Atoms are assigned by construction: a passing IM is by definition
IMAGING_MARKER (+ COGNITIVE_TASK when task-evoked / decoding).

This module replaces the older free-form recipe brainstormer. The previous
generator predated the atom alphabet and produced a mix of IMs, gene/
biomarker quantities, and meta-data tokens; that scope is no longer useful
once we have explicit atoms. IM brainstorming is now a single, focused job.

Output is `imaging_markers.json`. Markers are NOT written into the KG —
disease/gene -> IM edges come from Phase 2 paper extraction.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# ── IM families and modality<->operation compatibility ─────────────────────

IM_FAMILIES: tuple[str, ...] = (
    "univariate",       # one operation on one region
    "asymmetry",        # left - right of paired regions
    "ratio",            # operation(A) / operation(B) within one modality
    "network_summary",  # mean/variance over a region set (e.g. DMN FC)
    "task_evoked",      # GLM beta or evoked response under a task/stimulus
    "cross_modal",      # ratio/sum of operations across two modalities (e.g. tau / cortical thickness)
    "longitudinal",     # change over time of any of the above
    "contrastive",      # operation(A) - operation(B) between two regions same modality
)

# Operation -> {valid modality}. Keys are IF:* slugs (and a small set of
# composite ops introduced for IM families). Single-modality ops are listed
# explicitly; composite ops (asymmetry/ratio/network_summary/contrast) inherit
# the atomic operation's compatibility — so they don't appear here directly.
OP_TO_MODALITIES: dict[str, frozenset[str]] = {
    # sMRI
    "cortical_thickness":     frozenset({"sMRI"}),
    "cortical_surface_area":  frozenset({"sMRI"}),
    "regional_volume":        frozenset({"sMRI"}),
    "gray_matter_density":    frozenset({"sMRI"}),
    # dMRI
    "fractional_anisotropy":  frozenset({"dMRI"}),
    "mean_diffusivity":       frozenset({"dMRI"}),
    "radial_diffusivity":     frozenset({"dMRI"}),
    "axial_diffusivity":      frozenset({"dMRI"}),
    # fMRI
    "functional_connectivity": frozenset({"fMRI"}),
    "alff":                    frozenset({"fMRI"}),
    "reho":                    frozenset({"fMRI"}),
    "bold_amplitude":          frozenset({"fMRI"}),
    # PET
    "amyloid_suvr":            frozenset({"PET"}),
    "tau_suvr":                frozenset({"PET"}),
    "fdg_uptake":              frozenset({"PET"}),
    # EEG/MEG operations are not yet IF:* nodes; we still allow the LLM to
    # propose `power_band` / `coherence` / `evoked_potential` and treat them
    # as soft tokens (no IF:* link) so EEG/MEG IMs are not all rejected.
    "power_band":              frozenset({"EEG", "MEG"}),
    "coherence":               frozenset({"EEG", "MEG"}),
    "evoked_potential":        frozenset({"EEG", "MEG"}),
}

ALL_OPERATIONS: frozenset[str] = frozenset(OP_TO_MODALITIES.keys())

IMAGING_MODALITIES: frozenset[str] = frozenset({"sMRI", "dMRI", "fMRI", "PET", "EEG", "MEG"})

# Families that REQUIRE a single (atomic) operation; composite families
# (asymmetry / ratio / network_summary / contrastive) wrap an atomic op.
ATOMIC_OP_FAMILIES: frozenset[str] = frozenset({"univariate", "task_evoked", "longitudinal"})

# Families that MUST carry conditioning (task or stimulus) for the IM to
# make physical sense. Resting-state fMRI ops (FC/ALFF/ReHo) do NOT need
# conditioning; only task-evoked BOLD or EEG/MEG event-related does.
CONDITIONING_REQUIRED_FAMILIES: frozenset[str] = frozenset({"task_evoked"})

# Family -> minimum number of regions the LLM must cite.
FAMILY_MIN_REGIONS: dict[str, int] = {
    "univariate":       1,
    "asymmetry":        1,   # one canonical region (left/right both implied)
    "ratio":            2,
    "network_summary":  3,
    "task_evoked":      1,
    "cross_modal":      1,
    "longitudinal":     1,
    "contrastive":      2,
}


# ── Palette construction ───────────────────────────────────────────────────


@dataclass
class IMPalette:
    """KG-grounded primitives shown to the LLM for IM composition."""
    modalities:     list[dict]   # [{id, name}]
    operations:     list[dict]   # [{slug, name, modality, definition, aliases, if_id}]
    core_regions:   list[dict]   # [{id, name, aliases}] — high-degree canonical
    visual_rois:    list[dict]   # [{id, name, aliases}] — VROI:* functional
    enigma_regions: list[dict]   # [{id, name, aliases}] — DK + Aseg
    tasks:          list[dict]   # [{id, name}] — top-degree COGAT_TASK
    concepts:       list[dict]   # [{id, name}] — top-degree COGAT_CONCEPT

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def all_region_ids(self) -> set[str]:
        out: set[str] = set()
        for bucket in (self.core_regions, self.visual_rois, self.enigma_regions):
            out.update(r["id"] for r in bucket)
        return out

    @property
    def name_to_region_id(self) -> dict[str, str]:
        """Lower-cased name (preferred + aliases) -> region KG id."""
        idx: dict[str, str] = {}
        for bucket in (self.core_regions, self.visual_rois, self.enigma_regions):
            for r in bucket:
                for nm in [r["name"], *r.get("aliases", [])]:
                    if nm:
                        idx.setdefault(nm.lower().strip(), r["id"])
        return idx


# Source vocabularies whose cognitive_function entries are real cognitive
# constructs (not MSH-tagged clinical syndromes like "Depression" / "Pain"
# that flooded the previous inventory).
_REAL_COGNITIVE_SOURCES: frozenset[str] = frozenset({"CognitiveAtlas"})


def _domain_of(c: dict) -> str:
    tags = c.get("domain_tags") or []
    return tags[0] if tags else c.get("domain", "")


def _degree(edges: Iterable[dict]) -> dict[str, int]:
    deg: dict[str, int] = collections.Counter()
    for e in edges:
        sid = e.get("source_id")
        tid = e.get("target_id")
        if sid:
            deg[sid] += 1
        if tid:
            deg[tid] += 1
    return deg


def build_im_palette(
    concepts: dict[str, dict],
    edges: Optional[Iterable[dict]] = None,
    n_core_regions: int = 80,
    n_tasks: int = 40,
    n_concepts: int = 40,
) -> IMPalette:
    """Extract IM-relevant primitives from the KG.

    The palette is purposely tight. Region pool is capped by edge degree so
    the LLM sees canonical anatomy (hippocampus, ACC, thalamus) rather than
    the long tail of NN_TAL labels with no downstream evidence. Task /
    concept pools are pulled from CognitiveAtlas only (MSH cognitive-function
    entries are mostly clinical signs, not cognitive constructs).
    """
    edges = list(edges or [])
    deg = _degree(edges)

    # Modalities — only imaging ones; drop genetics/clinical/environment/etc.
    modalities: list[dict] = []
    for cid, c in concepts.items():
        if not cid.startswith("MODALITY:"):
            continue
        slug = cid.split(":", 1)[1]
        if slug in IMAGING_MODALITIES:
            modalities.append({"id": cid, "name": slug})
    modalities.sort(key=lambda m: list(IMAGING_MODALITIES).index(m["name"])
                    if m["name"] in IMAGING_MODALITIES else 99)

    # Operations — every IF:* node, plus EEG/MEG soft ops.
    operations: list[dict] = []
    for cid, c in concepts.items():
        if not cid.startswith("IF:"):
            continue
        slug = cid.split(":", 1)[1]
        md = c.get("metadata") or {}
        operations.append({
            "slug":       slug,
            "name":       c.get("preferred_name") or slug,
            "modality":   md.get("modality") or "",
            "definition": (c.get("definition") or "")[:160],
            "aliases":    list(c.get("aliases") or []),
            "if_id":      cid,
        })
    for slug in ("power_band", "coherence", "evoked_potential"):
        if slug in OP_TO_MODALITIES:
            mods = sorted(OP_TO_MODALITIES[slug])
            operations.append({
                "slug":       slug,
                "name":       slug.replace("_", " "),
                "modality":   "/".join(mods),
                "definition": {
                    "power_band":       "spectral power in canonical EEG/MEG band per channel/source",
                    "coherence":        "phase coherence between two channels/sources in a band",
                    "evoked_potential": "event-related EEG/MEG amplitude (e.g. P300, MMN)",
                }.get(slug, ""),
                "aliases":    [],
                "if_id":      None,
            })

    # Regions — three buckets so the LLM sees diverse anatomy.
    nn_items = [
        (cid, c, deg.get(cid, 0))
        for cid, c in concepts.items()
        if cid.startswith("NN:") and _domain_of(c) == "neuroanatomy"
    ]
    nn_items.sort(key=lambda t: -t[2])
    core_regions: list[dict] = []
    seen_names: set[str] = set()
    for cid, c, _d in nn_items:
        if len(core_regions) >= n_core_regions:
            break
        nm = (c.get("preferred_name") or "").strip()
        if not nm or nm.lower() in seen_names:
            continue
        # Skip pure container nodes ("Cerebral Cortex", "Telencephalon",
        # "Brainstem") — they are not measurable IM targets in any single
        # subject scan; ENIGMA / atlas entries below cover the real ROIs.
        if nm.lower() in {
            "telencephalon", "cerebral cortex", "cerebral white matter",
            "ventricular system", "lateral ventricle", "brainstem",
            "cerebellum", "cerebellar cortex",
            "anterior lobe", "posterior lobe",
        }:
            continue
        seen_names.add(nm.lower())
        core_regions.append({
            "id":      cid,
            "name":    nm,
            "aliases": list(c.get("aliases") or [])[:3],
        })

    visual_rois: list[dict] = []
    for cid, c in concepts.items():
        if not cid.startswith("VROI:"):
            continue
        visual_rois.append({
            "id":      cid,
            "name":    (c.get("preferred_name") or cid.split(":", 1)[1]),
            "aliases": list(c.get("aliases") or [])[:3],
        })

    # ENIGMA-canonical DK + Aseg targets are already in the KG via NN:*
    # ids, but they're a deliberate subset (only the 41 DK/Aseg ROIs ENIGMA
    # reports on). Surface them as a separate slot so the LLM sees them
    # as the "case-control comparable" pool.
    try:
        from ..ingestion.enigma_disease_im import ASEG_ROI_TO_NN, DK_ROI_TO_NN
        enigma_ids = sorted({*DK_ROI_TO_NN.values(), *ASEG_ROI_TO_NN.values()})
    except ImportError:
        enigma_ids = []
    enigma_regions: list[dict] = []
    for cid in enigma_ids:
        if cid in concepts and cid not in {r["id"] for r in core_regions}:
            c = concepts[cid]
            nm = (c.get("preferred_name") or "").strip()
            if nm:
                enigma_regions.append({
                    "id":      cid,
                    "name":    nm,
                    "aliases": list(c.get("aliases") or [])[:3],
                })

    tasks: list[dict] = []
    task_items = [
        (cid, c, deg.get(cid, 0))
        for cid, c in concepts.items()
        if cid.startswith("COGAT_TASK:")
    ]
    task_items.sort(key=lambda t: (-t[2], len(t[1].get("preferred_name") or "")))
    for cid, c, _d in task_items[:n_tasks]:
        tasks.append({"id": cid, "name": c.get("preferred_name") or cid})

    concepts_pool: list[dict] = []
    cf_items = [
        (cid, c, deg.get(cid, 0))
        for cid, c in concepts.items()
        if cid.startswith("COGAT_CONCEPT:")
        and c.get("source_vocab", "") in _REAL_COGNITIVE_SOURCES
    ]
    cf_items.sort(key=lambda t: (-t[2], len(t[1].get("preferred_name") or "")))
    for cid, c, _d in cf_items[:n_concepts]:
        concepts_pool.append({"id": cid, "name": c.get("preferred_name") or cid})

    return IMPalette(
        modalities=modalities,
        operations=operations,
        core_regions=core_regions,
        visual_rois=visual_rois,
        enigma_regions=enigma_regions,
        tasks=tasks,
        concepts=concepts_pool,
    )


def _render_palette(palette: IMPalette,
                    op_filter: Optional[set[str]] = None,
                    region_cap: int = 40) -> str:
    """Serialise the palette into a compact prompt block.

    Per batch the caller may pass an `op_filter` to nudge the LLM toward a
    specific modality family (e.g. only sMRI ops in this batch). Region
    cap keeps the prompt short for batched runs.
    """
    lines: list[str] = []
    mod_names = ", ".join(m["name"] for m in palette.modalities)
    lines.append(f"Modalities: {mod_names}")
    lines.append("")
    lines.append("Operations (each fixed to a modality):")
    for op in palette.operations:
        if op_filter is not None and op["slug"] not in op_filter:
            continue
        lines.append(f"  - {op['slug']:24s} [{op['modality']}] {op['name']}"
                     + (f" — {op['definition']}" if op['definition'] else ""))
    lines.append("")
    if palette.core_regions:
        names = [r["name"] for r in palette.core_regions[:region_cap]]
        lines.append(f"Canonical regions ({len(names)}): " + ", ".join(names))
    if palette.enigma_regions:
        names = [r["name"] for r in palette.enigma_regions]
        lines.append(f"ENIGMA DK/Aseg ROIs ({len(names)}): " + ", ".join(names))
    if palette.visual_rois:
        names = [r["name"] for r in palette.visual_rois]
        lines.append(f"Visual functional ROIs: " + ", ".join(names))
    if palette.tasks:
        names = [t["name"] for t in palette.tasks[:25]]
        lines.append(f"Tasks (Cognitive Atlas, top {len(names)}): " + ", ".join(names))
    if palette.concepts:
        names = [c["name"] for c in palette.concepts[:25]]
        lines.append(f"Cognitive concepts (Cognitive Atlas, top {len(names)}): " + ", ".join(names))
    return "\n".join(lines)


# ── ImagingMarker dataclass ────────────────────────────────────────────────


@dataclass
class ImagingMarker:
    """A single brainstormed imaging marker."""
    id: str
    name: str
    family: str
    modality: str
    operation: str
    regions: list[str] = field(default_factory=list)         # KG ids after linking
    region_names: list[str] = field(default_factory=list)    # raw LLM-side names
    conditioning: Optional[dict] = None                       # {"task": "...", "concept": "..."}
    formula: str = ""
    rationale: str = ""
    operation_id: Optional[str] = None                        # IF:* if resolved
    modality_id: Optional[str] = None                         # MODALITY:* if resolved
    atoms: list[str] = field(default_factory=list)
    llm_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── LLM brainstorm ─────────────────────────────────────────────────────────


_SYSTEM_PROMPT = (
    "You design imaging-derived markers (IMs) for a neuroimaging knowledge "
    "graph. An IM is a scalar (or vector) quantity that an MRI / PET / EEG / "
    "MEG analysis pipeline can compute from one subject's data — e.g. "
    "'cortical thickness of left entorhinal cortex from sMRI' or "
    "'mean BOLD response to faces in FFA from fMRI'. "
    "You receive a fixed palette of modalities, operations, regions, tasks, "
    "and cognitive concepts pulled from the knowledge graph; you may only "
    "compose IMs from those tokens. Do not invent new modalities or "
    "operations. Output strictly valid JSON, no prose."
)


_RULES = (
    "RULES (any IM violating these is invalid):\n"
    "  1. `modality` must be one of the listed imaging modalities.\n"
    "  2. `operation` must be a slug from the palette AND must be valid for "
    "the chosen modality (FA/MD/RD/AD only on dMRI; SUVR/uptake only on PET; "
    "FC/ALFF/ReHo/BOLD only on fMRI; thickness/surface_area/volume/GM_density "
    "only on sMRI; power_band/coherence/evoked_potential only on EEG or MEG).\n"
    "  3. `regions` must reference brain regions BY NAME from the palette "
    "lists — copy names verbatim. Do NOT invent acronyms or new regions.\n"
    "  4. `family` must be one of: univariate, asymmetry, ratio, "
    "network_summary, task_evoked, cross_modal, longitudinal, contrastive.\n"
    "  5. `task_evoked` IMs MUST set `conditioning` to a task or cognitive "
    "concept from the palette. Resting-state ops (FC/ALFF/ReHo) do NOT set "
    "conditioning.\n"
    "  6. Region count: ratio and contrastive need >=2 regions; "
    "network_summary needs >=3; others need >=1.\n"
    "  7. Each IM must be physically computable from a single subject's "
    "scan(s). No data-availability indicators, no cross-subject means, "
    "no behavioural-only scores.\n"
)


def _build_prompt(palette: IMPalette,
                  n: int,
                  family_focus: Optional[list[str]] = None,
                  op_filter: Optional[set[str]] = None,
                  existing_names: Optional[list[str]] = None) -> str:
    pal = _render_palette(palette, op_filter=op_filter)
    fam = (
        f"\nFamily focus for this batch (mix freely among these): "
        f"{', '.join(family_focus)}\n" if family_focus else ""
    )
    avoid = ""
    if existing_names:
        avoid = "\nAlready proposed (do NOT repeat or trivially rephrase):\n"
        for nm in existing_names:
            avoid += f"  - {nm}\n"
    return (
        f"Palette:\n{pal}\n"
        f"{fam}"
        f"\n{_RULES}\n"
        f"{avoid}"
        f"\nPropose {n} distinct imaging markers. Return a JSON array; each "
        "element MUST follow this schema exactly:\n"
        '  {"name": "<short slug>",\n'
        '   "family": "<one of the 8 families>",\n'
        '   "modality": "<sMRI|dMRI|fMRI|PET|EEG|MEG>",\n'
        '   "operation": "<palette operation slug>",\n'
        '   "regions": ["<palette region name>", ...],\n'
        '   "conditioning": {"task": "<palette task name>", "concept": "<palette concept name>"} | null,\n'
        '   "formula": "<one-line definition referencing palette tokens>",\n'
        '   "rationale": "<one short sentence: why this IM is a meaningful neural quantity>"}\n'
    )


def brainstorm_ims(
    palette: IMPalette,
    n: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
    family_focus: Optional[list[str]] = None,
    op_filter: Optional[set[str]] = None,
    existing_names: Optional[list[str]] = None,
) -> list[ImagingMarker]:
    """Single LLM call -> list of raw ImagingMarker (pre-validation)."""
    prompt = _build_prompt(palette, n, family_focus, op_filter, existing_names)
    raw = llm_call(prompt, _SYSTEM_PROMPT)
    data = _extract_json(raw)
    if not isinstance(data, list):
        logger.warning("LLM did not return a JSON array; got %r", type(data))
        return []
    out: list[ImagingMarker] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        family = str(item.get("family") or "").strip().lower()
        modality = str(item.get("modality") or "").strip()
        operation = str(item.get("operation") or "").strip()
        if not (name and family and modality and operation):
            continue
        regions_raw = item.get("regions") or []
        if not isinstance(regions_raw, list):
            continue
        region_names = [str(r).strip() for r in regions_raw if str(r).strip()]
        cond = item.get("conditioning")
        if cond is not None and not isinstance(cond, dict):
            cond = None
        out.append(ImagingMarker(
            id=f"im_{i+1:04d}",
            name=name,
            family=family,
            modality=modality,
            operation=operation,
            regions=[],          # filled by link_to_kg
            region_names=region_names,
            conditioning=cond,
            formula=str(item.get("formula") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
            llm_model=model_name,
        ))
    return out


# ── batched brainstorm ─────────────────────────────────────────────────────


# Each batch family-tuple drives op_filter via _ops_for_family_focus, so the
# LLM sees only the relevant operations for that batch.
_FAMILY_ROTATION: tuple[tuple[str, ...], ...] = (
    ("univariate",),
    ("asymmetry", "contrastive"),
    ("ratio",),
    ("network_summary",),
    ("task_evoked",),
    ("cross_modal",),
    ("longitudinal",),
)


def _ops_for_family_focus(focus: tuple[str, ...]) -> Optional[set[str]]:
    """Some batches benefit from a narrower op slate.

    For task_evoked we only want bold_amplitude / evoked_potential. For
    network_summary we steer toward FC and structural-network-friendly ops.
    Other families see the full op pool.
    """
    if focus == ("task_evoked",):
        return {"bold_amplitude", "evoked_potential", "power_band", "coherence"}
    if focus == ("network_summary",):
        return {"functional_connectivity", "coherence", "fractional_anisotropy",
                "alff", "reho"}
    return None


def _normalise_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", s.strip().lower())


def brainstorm_ims_batched(
    palette: IMPalette,
    n_total: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
    batch_size: int = 30,
    seed: int = 0,
) -> list[ImagingMarker]:
    """Generate `n_total` raw IMs via repeated LLM calls with rotating family focus."""
    rng = random.Random(seed)
    accepted: list[ImagingMarker] = []
    seen_keys: set[str] = set()
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        if len(accepted) >= n_total:
            break
        focus = _FAMILY_ROTATION[batch_idx % len(_FAMILY_ROTATION)]
        op_filter = _ops_for_family_focus(focus)
        sample_existing = (
            [r.name for r in rng.sample(accepted, k=min(20, len(accepted)))]
            if accepted else None
        )
        try:
            batch = brainstorm_ims(
                palette, n=batch_size, llm_call=llm_call,
                model_name=model_name,
                family_focus=list(focus),
                op_filter=op_filter,
                existing_names=sample_existing,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch %d LLM call failed: %s", batch_idx, exc)
            continue
        added = 0
        for im in batch:
            key = _normalise_name(im.name)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            accepted.append(im)
            added += 1
            if len(accepted) >= n_total:
                break
        logger.info("batch %d/%d focus=%s -> +%d (total %d/%d)",
                    batch_idx + 1, n_batches, focus, added,
                    len(accepted), n_total)

    accepted = accepted[:n_total]
    for i, im in enumerate(accepted):
        im.id = f"im_{i+1:04d}"
    return accepted


def _extract_json(text: str) -> Any:
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    for s, e in (("[", "]"), ("{", "}")):
        i, j = text.find(s), text.rfind(e)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ── validation ─────────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    accepted: list[ImagingMarker]
    rejected: list[tuple[ImagingMarker, str]]

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def reject_reasons(self) -> dict[str, int]:
        c: dict[str, int] = collections.Counter()
        for _, reason in self.rejected:
            c[reason] += 1
        return dict(c)


def validate_ims(ims: Iterable[ImagingMarker], palette: IMPalette) -> ValidationReport:
    """Apply structural rules; mark each IM as accepted or rejected with reason."""
    accepted: list[ImagingMarker] = []
    rejected: list[tuple[ImagingMarker, str]] = []
    name_to_rid = palette.name_to_region_id
    task_names = {t["name"].lower() for t in palette.tasks}
    concept_names = {c["name"].lower() for c in palette.concepts}

    for im in ims:
        if im.modality not in IMAGING_MODALITIES:
            rejected.append((im, "modality_invalid"))
            continue
        if im.family not in IM_FAMILIES:
            rejected.append((im, "family_invalid"))
            continue
        op = im.operation
        if op not in ALL_OPERATIONS:
            rejected.append((im, "operation_unknown"))
            continue
        if im.modality not in OP_TO_MODALITIES[op]:
            rejected.append((im, "operation_modality_incompatible"))
            continue
        min_regions = FAMILY_MIN_REGIONS.get(im.family, 1)
        if len(im.region_names) < min_regions:
            rejected.append((im, f"region_count<{min_regions}"))
            continue
        # Resolve regions to KG ids; require strict match (case-insensitive).
        resolved: list[str] = []
        for rn in im.region_names:
            rid = name_to_rid.get(rn.lower().strip())
            if rid:
                resolved.append(rid)
        if len(resolved) < min_regions:
            rejected.append((im, "region_resolution_failed"))
            continue
        im.regions = resolved
        # Conditioning gate.
        if im.family in CONDITIONING_REQUIRED_FAMILIES:
            cond = im.conditioning or {}
            t = (cond.get("task") or "").lower().strip()
            cn = (cond.get("concept") or "").lower().strip()
            if not (t in task_names or cn in concept_names):
                rejected.append((im, "conditioning_missing_or_unknown"))
                continue
        accepted.append(im)
    return ValidationReport(accepted=accepted, rejected=rejected)


# ── KG linking + atom tagging ──────────────────────────────────────────────


def link_ims_to_kg(ims: Iterable[ImagingMarker], palette: IMPalette) -> None:
    """Attach modality_id and operation_id from the palette."""
    op_to_id = {op["slug"]: op["if_id"] for op in palette.operations}
    mod_to_id = {m["name"]: m["id"] for m in palette.modalities}
    for im in ims:
        im.operation_id = op_to_id.get(im.operation)
        im.modality_id = mod_to_id.get(im.modality)


def tag_atoms(ims: Iterable[ImagingMarker]) -> None:
    """Assign atoms by construction.

    Every validated IM is IMAGING_MARKER. Task-evoked IMs additionally get
    COGNITIVE_TASK because the conditioning makes them composite. Other
    families do not infer atoms from token matches — that's exactly what
    made the previous recipe pipeline noisy.
    """
    for im in ims:
        atoms = ["IMAGING_MARKER"]
        if im.family == "task_evoked" and im.conditioning:
            atoms.append("COGNITIVE_TASK")
        im.atoms = atoms


__all__ = [
    "ImagingMarker",
    "IMPalette",
    "ValidationReport",
    "IM_FAMILIES",
    "OP_TO_MODALITIES",
    "ALL_OPERATIONS",
    "IMAGING_MODALITIES",
    "build_im_palette",
    "brainstorm_ims",
    "brainstorm_ims_batched",
    "validate_ims",
    "link_ims_to_kg",
    "tag_atoms",
]
