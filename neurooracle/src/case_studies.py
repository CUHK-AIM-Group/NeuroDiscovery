"""Case-study registry for the NeuroClaw Nature paper.

The autoresearch CLI runs a four-stage cycle (batch -> novelty -> critic ->
plausibility) over the canonical task / chain registry in :mod:`atoms`. The
three case studies that anchor the Nature paper do not all map cleanly onto a
single canonical task, so this module records the extra routing metadata the
generic engine needs:

- Case Study 1 uses a candidate-space generator over disease x ROI x feature
  hypotheses, with exhaustive, random-walk, LLM-brainstorm, and NeuroDiscovery
  strategies.
- Case Study 2 uses a canonical chain plus case-specific atom-pool restrictions.
- Case Study 3 is reserved for hindcasting over a frozen historical KG snapshot.

This module declares each case study as a :class:`CaseStudy` config: the
generator family it dispatches to, the underlying canonical task / chain
(when any), per-stage parameter overrides, and pre/post hooks that adapt
the generic engine to case-specific constraints. The CLI's ``case-study``
subcommand reads this registry and routes execution accordingly.

At the current rollout state, Case Study 1 and Case Study 2 are wired into the
case-study orchestrator. Case Study 3 remains a reserved generator family until the
historical claim / snapshot pipeline is finalized.
"""

from __future__ import annotations

import re
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .atoms import (
    Atom,
    Task,
    TaskChain,
    chain_by_name,
    task_by_name,
)


# ── Generator families ────────────────────────────────────────────────────────
# Identifies which code path inside the case-study orchestrator handles
# stage [1/4] (raw hypothesis generation) for a given case study.
GENERATOR_TASK              = "task"               # CANONICAL_TASKS via batch_generate_for_task
GENERATOR_CHAIN             = "chain"              # CANONICAL_CHAINS via batch_generate_for_chain
GENERATOR_CASE1_CANDIDATE   = "case1_candidate_space"  # CS1 disease x ROI x feature search
GENERATOR_ATOM_SUBSTITUTION = "atom_substitution"  # Case Study 3 hindcasting (reserved)

_KNOWN_GENERATORS = frozenset({
    GENERATOR_TASK,
    GENERATOR_CHAIN,
    GENERATOR_CASE1_CANDIDATE,
    GENERATOR_ATOM_SUBSTITUTION,
})

NEUROSTORM_ATLAS_ROOT = Path(r"C:\Users\45846\Documents\Code\NeuroSTORM\datasets\atlas")


def _clean_atlas_label_cell(cell: str) -> str:
    cell = (cell or "").strip()
    if not cell:
        return ""
    matches = re.findall(r'\["([^"]+)":\s*[0-9.]+\]', cell)
    candidates = [m.strip() for m in matches] if matches else [cell]
    for candidate in candidates:
        folded = candidate.casefold().strip()
        if not folded or folded in {"none", "background", "volume", "center of mass"}:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", folded):
            continue
        if re.fullmatch(r"\(?\s*-?\d+(\.\d+)?\s*;\s*-?\d+(\.\d+)?\s*;\s*-?\d+(\.\d+)?\s*\)?", folded):
            continue
        if re.fullmatch(r"roi[_\s-]*\d+", folded):
            continue
        return candidate
    return ""


def _fallback_atlas_roi_label(atlas_name: str, roi_index: str, row: list[str]) -> str:
    for cell in row[1:]:
        value = (cell or "").strip()
        folded = value.casefold()
        if not value or folded in {"none", "background", "volume", "center of mass"}:
            continue
        if re.fullmatch(r"\(?\s*-?\d+(\.\d+)?\s*;\s*-?\d+(\.\d+)?\s*;\s*-?\d+(\.\d+)?\s*\)?", folded):
            continue
        if re.fullmatch(r"\d+(\.\d+)?", folded) or re.fullmatch(r"roi[_\s-]*\d+", folded):
            break
        return value
    return f"{atlas_name} ROI {roi_index}"


def _load_neurostorm_atlas_labels() -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    tuple[dict[str, str], ...],
]:
    labels: set[str] = set()
    label_sources: dict[str, set[str]] = {}
    atlas_names: list[str] = []
    atlas_rois: list[dict[str, str]] = []
    if not NEUROSTORM_ATLAS_ROOT.exists():
        return tuple(), {}, tuple(), tuple()
    for atlas_dir in sorted(NEUROSTORM_ATLAS_ROOT.iterdir()):
        if not atlas_dir.is_dir() or not (atlas_dir / "atlas.nii.gz").is_file():
            continue
        atlas_names.append(atlas_dir.name)
        labels_csv = atlas_dir / "labels.csv"
        if not labels_csv.is_file():
            continue
        with labels_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
            for raw_line in f:
                line = raw_line.strip()
                folded_line = line.casefold()
                if (
                    not line
                    or line.startswith("#")
                    or folded_line.startswith("index,")
                    or folded_line.startswith("roi number,")
                ):
                    continue
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    continue
                if len(row) < 2:
                    continue
                label = ""
                for cell in row[1:]:
                    label = _clean_atlas_label_cell(cell)
                    if label:
                        break
                roi_index = row[0].strip()
                if not roi_index:
                    continue
                if not label:
                    label = _fallback_atlas_roi_label(atlas_dir.name, roi_index, row)
                if not label or label.casefold() == "background":
                    continue
                fallback_name = f"{atlas_dir.name} ROI {roi_index}"
                display_name = label if label == fallback_name else f"{fallback_name}: {label}"
                labels.add(label)
                label_sources.setdefault(label, set()).add(atlas_dir.name)
                atlas_rois.append({
                    "atlas_name": atlas_dir.name,
                    "roi_index": roi_index,
                    "label": label,
                    "name": display_name,
                })
    return (
        tuple(sorted(labels)),
        {label: tuple(sorted(srcs)) for label, srcs in sorted(label_sources.items())},
        tuple(atlas_names),
        tuple(atlas_rois),
    )


(
    CASE1_ATLAS_LABELS,
    CASE1_ATLAS_LABEL_SOURCES,
    CASE1_ATLAS_NAMES,
    CASE1_ATLAS_ROIS,
) = _load_neurostorm_atlas_labels()


# Case Study 1 is framed as a search over an executable hypothesis space:
# disease x atlas/region x feature. Direction is not part of the generator;
# validation estimates whether the disease group is higher or lower, and
# cross-disease clusters are summarized after validation.
CASE1_FEATURE_SPACE: tuple[dict[str, Any], ...] = (
    {
        "id": "roi_alff",
        "name": "ROI ALFF",
        "family": "roi_activity",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("roi_timeseries",),
        "primary": True,
    },
    {
        "id": "roi_falff",
        "name": "ROI fALFF",
        "family": "roi_activity",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("roi_timeseries",),
        "primary": True,
    },
    {
        "id": "roi_temporal_variance",
        "name": "ROI temporal variance",
        "family": "roi_activity",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("roi_timeseries",),
        "primary": True,
    },
    {
        "id": "roi_mean_whole_brain_fc",
        "name": "ROI mean whole-brain FC",
        "family": "seed_fc",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix",),
        "primary": True,
    },
    {
        "id": "roi_within_network_fc",
        "name": "ROI within-network FC",
        "family": "seed_fc",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix", "network_labels"),
        "primary": True,
    },
    {
        "id": "roi_between_network_fc",
        "name": "ROI between-network FC",
        "family": "seed_fc",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix", "network_labels"),
        "primary": True,
    },
    {
        "id": "roi_node_strength",
        "name": "ROI node strength",
        "family": "graph",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix",),
        "primary": True,
    },
    {
        "id": "roi_node_degree",
        "name": "ROI node degree",
        "family": "graph",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix", "edge_threshold"),
        "primary": True,
    },
    {
        "id": "roi_participation_coefficient",
        "name": "ROI participation coefficient",
        "family": "graph",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix", "network_labels"),
        "primary": True,
    },
    {
        "id": "roi_local_efficiency",
        "name": "ROI local efficiency",
        "family": "graph",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix", "edge_threshold"),
        "primary": True,
    },
    {
        "id": "roi_fc_variability",
        "name": "ROI FC variability",
        "family": "dynamic_fc",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("roi_timeseries", "sliding_window_fc"),
        "primary": True,
    },
    {
        "id": "subject_state_occupancy",
        "name": "Subject state occupancy",
        "family": "dynamic_fc",
        "modality": "fMRI",
        "level": "subject",
        "requires": ("roi_timeseries", "dynamic_state_model"),
        "primary": True,
    },
    {
        "id": "roi_cortical_thickness",
        "name": "ROI cortical thickness",
        "family": "structural",
        "modality": "sMRI",
        "level": "ROI",
        "requires": ("T1w", "FreeSurfer_or_equivalent"),
        "primary": False,
    },
    {
        "id": "roi_surface_area",
        "name": "ROI surface area",
        "family": "structural",
        "modality": "sMRI",
        "level": "ROI",
        "requires": ("T1w", "FreeSurfer_or_equivalent"),
        "primary": False,
    },
    {
        "id": "roi_gray_matter_volume",
        "name": "ROI gray-matter volume",
        "family": "structural",
        "modality": "sMRI",
        "level": "ROI",
        "requires": ("T1w", "FreeSurfer_or_equivalent"),
        "primary": False,
    },
)


# ── Per-stage parameter blocks ───────────────────────────────────────────────
# Each stage's params mirror the kwargs of the corresponding cmd_* in
# hypothesis_cli.py / kge.cli — the orchestrator forwards them verbatim.

@dataclass(frozen=True)
class BatchParams:
    """Parameters for stage [1/4] — raw hypothesis generation."""
    max_hops: int = 3
    min_hops: int = 2
    metapath_min_domains: int = 2
    max_paths: int = 4
    max_seeds: int = 30
    target_per_task: int = 100
    max_retries: int = 4
    retry_scale: float = 2.0
    prefer_longer_paths: bool = True


@dataclass(frozen=True)
class NoveltyParams:
    """Parameters for stage [2/4] — PubMed + Semantic Scholar novelty check."""
    top: int = 200
    alpha: float = 0.5
    skip_pubmed: bool = False
    skip_semantic: bool = False


@dataclass(frozen=True)
class CriticParams:
    """Parameters for stage [3/4] — three-perspective Critic Agent."""
    top: int = 100
    max_rounds: int = 2
    threshold: float = 0.55
    max_workers: int = 12


@dataclass(frozen=True)
class PlausibilityParams:
    """Parameters for stage [4/4] — KGE ComplEx + PubMed attestation."""
    top: int = 100
    no_pubmed: bool = False
    skip_existing: bool = True
    enable_surprise: bool = False
    surprise_alpha: float = 0.1
    evo_surprise_min: Optional[float] = None
    device: Optional[str] = None


@dataclass(frozen=True)
class StageParams:
    """Bundle of stage configs — one block per pipeline stage."""
    batch: BatchParams = field(default_factory=BatchParams)
    novelty: NoveltyParams = field(default_factory=NoveltyParams)
    critic: CriticParams = field(default_factory=CriticParams)
    plausibility: PlausibilityParams = field(default_factory=PlausibilityParams)


# ── Case-study record ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseStudy:
    """A single Nature-paper case study and its autoresearch config.

    Fields
    ------
    name:
        Stable slug used as CLI argument and output-directory key.
    chinese_name / english_name:
        Frozen copy of the names finalized in the project memory; the CLI
        echoes both in the run log so figure / paper text stays in sync
        with whatever the orchestrator actually generated.
    generator:
        One of GENERATOR_TASK / GENERATOR_CHAIN / GENERATOR_CASE1_CANDIDATE
        / GENERATOR_ATOM_SUBSTITUTION.
    task / chain:
        The canonical Task or TaskChain backing the generator. Exactly one
        is set for GENERATOR_TASK / GENERATOR_CHAIN. Case Study 1 still pins a
        Task for downstream tagging; Case Study 3 pins a chain as its schema anchor.
    stage_params:
        Per-stage parameter overrides. Defaults match run_cycle.sh.
    pre_hooks / post_hooks:
        Optional callables invoked around stage [1/4]. Each receives the
        active :class:`HypothesisEngine` and the case study itself; they
        may mutate engine state (e.g. _path_ignore_ids for Case Study 2's
        pathway-only GENE pool) or rewrap generated hypotheses.
    extras:
        Generator-specific config dict that doesn't fit anywhere else
        (snapshot paths for Case Study 3, cluster-mining knobs for Case Study 1, ...).
    """
    name: str
    chinese_name: str
    english_name: str
    generator: str
    task: Optional[Task] = None
    chain: Optional[TaskChain] = None
    stage_params: StageParams = field(default_factory=StageParams)
    pre_hooks: tuple[Callable[..., None], ...] = ()
    post_hooks: tuple[Callable[..., None], ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CaseStudy.name must be non-empty")
        if self.generator not in _KNOWN_GENERATORS:
            raise ValueError(
                f"CaseStudy '{self.name}': unknown generator {self.generator!r} "
                f"(valid: {sorted(_KNOWN_GENERATORS)})"
            )
        if self.generator == GENERATOR_TASK and self.task is None:
            raise ValueError(f"CaseStudy '{self.name}': generator='task' requires .task")
        if self.generator == GENERATOR_CHAIN and self.chain is None:
            raise ValueError(f"CaseStudy '{self.name}': generator='chain' requires .chain")


# ── Pre_hooks ────────────────────────────────────────────────────────────────

_CASE2_IMAGING_MARKER_RE = re.compile(
    r"("
    r"\b(PET|MRI|fMRI|SPECT|DTI|FDG|SUVR|ALFF|ReHo)\b|"
    r"amyloid|tau|hypometabolism|atrophy|volume|thickness|surface area|"
    r"fractional anisotropy|diffusivity|perfusion|cerebral blood flow|"
    r"connectivity|network|white matter|gray matter|grey matter|"
    r"cortical|hippocamp|entorhinal|parahippocamp|amygdala|cingulate|"
    r"frontal|temporal|parietal|occipital|striatal|thalam"
    r")",
    re.I,
)

_CASE2_NON_IMAGING_BIOMARKER_RE = re.compile(
    r"\b(CSF|blood|plasma|serum|gut|microbiome|bile|cytokine|proteomic|"
    r"inflammatory|metabolic|short-chain fatty acids?)\b",
    re.I,
)

_CASE2_OUTCOME_RE = re.compile(
    r"("
    r"ADAS|MMSE|MoCA|CDR|HAMD|HAM-D|MADRS|UPDRS|PACC|CBI|inventory|"
    r"scale|score|decline|impairment|deficit|performance|cognition|"
    r"cognitive|memory|executive|attention|language|behavior|behaviour|"
    r"depression|symptom|severity|conversion|progression|response"
    r")",
    re.I,
)


def _case2_is_claim_backed_clm(nid: str, node, claim_incident: Counter) -> bool:
    return (
        node is not None
        and nid.startswith("CLM_CONCEPT:")
        and claim_incident.get(nid, 0) > 0
    )


def _case2_pin_atom_pools(engine, case) -> None:
    """Case Study 2: route the chain through claim-dense G -> IM -> O anchors.

    Without this, ``batch_generate_for_chain`` accepts any node whose
    ``domain_tags`` overlap the atom's domain set — for IMAGING_MARKER that
    includes ``neuroanatomy``, so raw region CUIs (Hippocampus, etc.) end up
    serving as the marker anchor and the new IM:* atom layer is never
    reached. Case Study 2 now keeps those curated IM:*/OUTCOME:* anchors, but also
    admits high-confidence CLM_CONCEPT claim entities when they look like
    concrete imaging markers or clinical/cognitive outcomes. Gene seeds stay
    pathway-aware, with Phase-2 claim-backed genes ranked first.
    """
    # Genes that participate in any GENESET (the "pathway_aggregated" pool).
    pathway_genes: set[str] = set()
    claim_incident: Counter[str] = Counter()
    gene_claim_scores: Counter[str] = Counter()
    imaging_domains = {"biomarker", "connectivity", "imaging_feature", "neuroanatomy"}
    outcome_domains = {"treatment_outcome", "dataset_variable", "cognitive_function"}

    for u, v, d in engine.G.edges(data=True):
        if d.get("relation_type") == "part_of" and v.startswith("GENESET:"):
            pathway_genes.add(u)
        if not d.get("metadata", {}).get("claim_id"):
            continue
        claim_incident[u] += 1
        claim_incident[v] += 1
        u_node = engine._index.get(u)
        v_node = engine._index.get(v)
        u_domains = set(u_node.domain_tags or []) if u_node else set()
        v_domains = set(v_node.domain_tags or []) if v_node else set()
        if "gene" in u_domains:
            gene_claim_scores[u] += 1
            if v_domains & imaging_domains:
                gene_claim_scores[u] += 3
            if v_domains & outcome_domains:
                gene_claim_scores[u] += 2
        if "gene" in v_domains:
            gene_claim_scores[v] += 1
            if u_domains & imaging_domains:
                gene_claim_scores[v] += 3
            if u_domains & outcome_domains:
                gene_claim_scores[v] += 2

    def _im_filter(nid, node):
        if nid.startswith("IM:"):
            return True
        if not _case2_is_claim_backed_clm(nid, node, claim_incident):
            return False
        domains = set(node.domain_tags or [])
        if not (domains & imaging_domains):
            return False
        name = node.preferred_name or ""
        if domains & {"connectivity", "imaging_feature"}:
            return True
        if _CASE2_NON_IMAGING_BIOMARKER_RE.search(name) and not re.search(
            r"\b(PET|MRI|fMRI|SPECT|DTI|FDG|SUVR)\b|amyloid|tau|brain|cortical|"
            r"hippocamp|connectivity|atrophy|hypometabolism",
            name,
            re.I,
        ):
            return False
        return bool(_CASE2_IMAGING_MARKER_RE.search(name))

    def _gene_filter(nid, node):
        if node is None:
            return False
        if nid.startswith("GENESET:") or nid.startswith("CLM_CONCEPT:"):
            return False
        if "gene" not in (node.domain_tags or []):
            return False
        # Keep pathway genes as the Case Study 2 backbone, but allow claim-backed genes
        # into the seed pool too. The ranker below tries claim-backed genes
        # first, so the run starts in the dense case-study evidence region.
        return ((nid in pathway_genes) if pathway_genes else True) or nid in gene_claim_scores

    def _outcome_filter(nid, node):
        if nid.startswith("OUTCOME:"):
            return True
        if not _case2_is_claim_backed_clm(nid, node, claim_incident):
            return False
        domains = set(node.domain_tags or [])
        if not (domains & outcome_domains):
            return False
        return bool(_CASE2_OUTCOME_RE.search(node.preferred_name or ""))

    def _gene_ranker(nid, node):
        return gene_claim_scores.get(nid, 0)

    engine._chain_atom_filters = {
        Atom.IMAGING_MARKER: _im_filter,
        Atom.GENE_TARGET:    _gene_filter,
        Atom.OUTCOME:        _outcome_filter,
    }
    engine._chain_atom_extra_domains = {
        Atom.OUTCOME: frozenset({"cognitive_function"}),
    }
    engine._chain_atom_rankers = {
        Atom.GENE_TARGET: _gene_ranker,
    }
    engine._chain_prefer_claim_backed_paths = True
    engine._chain_claimless_path_fraction = 0.15


# ── Concrete case studies (frozen for the Nature paper) ──────────────────────

CASE1 = CaseStudy(
    name="case1_transdiagnostic",
    chinese_name="跨诊断精神疾病脑影像图谱",
    english_name="Transdiagnostic Brain Atlas of Psychiatric Disorders",
    generator=GENERATOR_CASE1_CANDIDATE,
    task=task_by_name("transdiagnostic_clustering"),
    stage_params=StageParams(
        batch=BatchParams(max_paths=6, max_seeds=60, target_per_task=80),
        novelty=NoveltyParams(top=120, alpha=0.5),
        critic=CriticParams(top=60, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=60),
    ),
    extras={
        "generation_methods": (
            "exhaustive",
            "random_walk",
            "llm_brainstorm",
            "neurodiscovery",
        ),
        "max_hypotheses_per_method": {
            "exhaustive": 40,
            "random_walk": 40,
            "llm_brainstorm": 40,
            "neurodiscovery": 40,
        },
        "candidate_unit": "disease_region_feature",
        "random_seed": 20260615,
        "disease_include_names": (
            "Anorexia Nervosa",
            "Attention Deficit Disorder with Hyperactivity",
            "Bipolar Disorder",
            "Major Depressive Disorder",
            "Obsessive-Compulsive Disorder",
            "Schizophrenia",
            "Schizoaffective Disorder",
            "Post-Traumatic Stress Disorder",
            "Anxiety Disorders",
            "Generalized Anxiety Disorder",
            "Substance Use Disorders",
        ),
        "atlas_names": CASE1_ATLAS_NAMES,
        "atlas_rois": CASE1_ATLAS_ROIS,
        "atlas_label_names": CASE1_ATLAS_LABELS,
        "atlas_label_sources": CASE1_ATLAS_LABEL_SOURCES,
        "feature_space": CASE1_FEATURE_SPACE,
    },
)

CASE2 = CaseStudy(
    name="case2_pathway_mediation",
    chinese_name="多基因通路影像中介",
    english_name="Pathway-Level Polygenic Mediation through Brain Imaging",
    generator=GENERATOR_CHAIN,
    chain=chain_by_name("pathway_polygenic_mediation"),
    stage_params=StageParams(
        batch=BatchParams(
            max_hops=3, min_hops=2, max_paths=4, max_seeds=30,
            target_per_task=100, max_retries=4, retry_scale=2.0,
        ),
        novelty=NoveltyParams(top=200, alpha=0.5),
        critic=CriticParams(top=100, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=100),
    ),
    pre_hooks=(_case2_pin_atom_pools,),
    extras={
        "gene_pool_filter": "pathway_aggregated",
    },
)

CASE3 = CaseStudy(
    name="case3_hindcasting",
    chinese_name="假设回溯预测",
    english_name="Hypothesis Hindcasting",
    generator=GENERATOR_ATOM_SUBSTITUTION,
    chain=chain_by_name("genetic_imaging_disease"),
    stage_params=StageParams(
        batch=BatchParams(max_paths=4, max_seeds=30, target_per_task=100),
        novelty=NoveltyParams(top=200, alpha=0.5),
        critic=CriticParams(top=100, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=100),
    ),
    extras={
        "snapshot_2022_kg":         "neurooracle/data/snapshots/kg_2022.json",
        "snapshot_2022_kge":        "neurooracle/data/snapshots/kge_2022.pt",
        "substitution_axes":        ("GENE_TARGET", "IMAGING_MARKER", "DISEASE"),
        "max_substitutions_per_seed": 5,
    },
)


CASE_STUDIES: tuple[CaseStudy, ...] = (CASE1, CASE2, CASE3)


def case_study_by_name(name: str) -> CaseStudy:
    """Look up a case study by its registry slug. Raises KeyError."""
    for cs in CASE_STUDIES:
        if cs.name == name:
            return cs
    valid = ", ".join(list_case_study_names())
    raise KeyError(f"unknown case study: {name!r} (valid: {valid})")


def list_case_study_names() -> tuple[str, ...]:
    return tuple(cs.name for cs in CASE_STUDIES)


__all__ = [
    "BatchParams",
    "NoveltyParams",
    "CriticParams",
    "PlausibilityParams",
    "StageParams",
    "CaseStudy",
    "CASE1",
    "CASE2",
    "CASE3",
    "CASE_STUDIES",
    "case_study_by_name",
    "list_case_study_names",
    "GENERATOR_TASK",
    "GENERATOR_CHAIN",
    "GENERATOR_CASE1_CANDIDATE",
    "GENERATOR_ATOM_SUBSTITUTION",
]
