"""Formal case-study registry for NeuroOracle autoresearch.

The autoresearch CLI runs a four-stage cycle (batch -> novelty -> critic ->
plausibility) over the canonical task / chain registry in :mod:`atoms`. The
formal registry separates research scope from evaluation protocol. A case
study describes a scientific research scope and its optional hypothesis
generator; hindcasting is a validation protocol declared separately in
``validation_protocols.py`` and can be applied to every registered case study.

The two implemented manuscript experiments do not both map cleanly onto a
single canonical task, so this module records the extra routing metadata the
generic engine needs:

- Case Study 1 uses a candidate-space generator over disease x ROI x feature
  hypotheses, with exhaustive, random-walk, LLM-brainstorm, and NeuroDiscovery
  strategies.
- Case Study 2 uses a canonical chain plus case-specific atom-pool restrictions.
- Fifteen task-backed research scopes are registered as first-class case
  studies. They are not nested under another case study and are not
  automatically scheduled for primary experiments.

This module declares each case study as a :class:`CaseStudy` config: the
generator family it dispatches to, the underlying canonical task / chain
(when any), per-stage parameter overrides, and pre/post hooks that adapt
the generic engine to case-specific constraints. The CLI's ``case-study``
subcommand reads this registry and routes execution accordingly.

The registry therefore contains seventeen peer case-study IDs: Case Study 1,
Case Study 2, and the fifteen task-backed scopes. ``case3_hindcasting`` is not
a case-study ID.
"""

from __future__ import annotations

import os
import re
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .atoms import (
    Atom,
    CANONICAL_TASKS,
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

_KNOWN_GENERATORS = frozenset({
    GENERATOR_TASK,
    GENERATOR_CHAIN,
    GENERATOR_CASE1_CANDIDATE,
})

NEUROSTORM_ATLAS_ROOT = Path(
    os.environ.get(
        "NEUROSTORM_ATLAS_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "atlas",
    )
).expanduser()


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
        One of GENERATOR_TASK / GENERATOR_CHAIN / GENERATOR_CASE1_CANDIDATE.
    task / chain:
        The canonical Task or TaskChain backing the generator. Exactly one
        is set for GENERATOR_TASK / GENERATOR_CHAIN. Case Study 1 still pins a
        Task for downstream tagging.
    stage_params:
        Per-stage parameter overrides. Defaults match run_cycle.sh.
    pre_hooks / post_hooks:
        Optional callables invoked around stage [1/4]. Each receives the
        active :class:`HypothesisEngine` and the case study itself; they
        may mutate engine state (e.g. _path_ignore_ids for Case Study 2's
        pathway-only GENE pool) or rewrap generated hypotheses.
    extras:
        Generator-specific config dict that doesn't fit anywhere else
        (cluster-mining knobs for Case Study 1, scheduling metadata, ...).
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

_CASE2_IMAGING_MEASUREMENT_RE = re.compile(
    r"(PET|MRI|fMRI|SPECT|DTI|FDG|SUVR|hypometabolism|atrophy|volume|"
    r"thickness|surface area|fractional anisotropy|diffusivity|perfusion|"
    r"cerebral blood flow|connectivity|activation|activity|signal|ALFF|ReHo|"
    r"homogeneity|network topology|density|binding|uptake)",
    re.I,
)

_CASE2_GENETIC_ENTITY_RE = re.compile(
    r"\b(gene|genetic|genotype|allele|variant|polygenic|PRS)\w*\b",
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


def _case2_endpoint_atom_roles(name: str, declared_type: str) -> set[Atom]:
    """Assign one unambiguous Case 2 atom role unless the type is explicitly mixed."""
    name = str(name or "").strip()
    declared = str(declared_type or "").strip().upper()

    if "IMAGING_AND_OUTCOME" in declared:
        return set()
    if "IMAGING" in declared or declared in {
        "NEUROIMAGING", "CONNECTIVITY", "PET_MARKER"
    }:
        return {Atom.IMAGING_MARKER}
    if declared == "BRAIN_REGION":
        return {Atom.IMAGING_MARKER} if _CASE2_IMAGING_MEASUREMENT_RE.search(name) else set()
    if "OUTCOME" in declared or declared in {
        "CLINICAL_EVENT", "COGNITIVE_FUNCTION", "SYMPTOM", "RATING_SCALE"
    }:
        return {Atom.OUTCOME}
    if any(
        token in declared
        for token in (
            "GENE", "GENETIC", "POLYGENIC", "PATHWAY", "MOLECULAR", "CELLULAR"
        )
    ):
        return {Atom.GENE_TARGET}

    imaging_named = bool(
        _CASE2_IMAGING_MARKER_RE.search(name)
        and _CASE2_IMAGING_MEASUREMENT_RE.search(name)
        and not _CASE2_GENETIC_ENTITY_RE.search(name)
    ) and not (
        _CASE2_NON_IMAGING_BIOMARKER_RE.search(name)
        and not re.search(
            r"\b(PET|MRI|fMRI|SPECT|DTI|FDG|SUVR)\b|amyloid|tau|brain|"
            r"cortical|hippocamp|connectivity|atrophy|hypometabolism",
            name,
            re.I,
        )
    )
    outcome_named = bool(_CASE2_OUTCOME_RE.search(name))
    if imaging_named == outcome_named:
        return set()
    return {Atom.IMAGING_MARKER if imaging_named else Atom.OUTCOME}


def _case2_is_claim_backed_node(nid: str, node, claim_incident: Counter) -> bool:
    return node is not None and claim_incident.get(nid, 0) > 0


def _case2_pin_atom_pools(engine, case) -> None:
    """Case Study 2: route the chain through claim-dense G -> IM -> O anchors.

    Without this, ``batch_generate_for_chain`` accepts any node whose
    ``domain_tags`` overlap the atom's domain set — for IMAGING_MARKER that
    includes ``neuroanatomy``, so raw region CUIs (Hippocampus, etc.) end up
    serving as the marker anchor and the new IM:* atom layer is never
    reached. Case Study 2 now keeps those curated IM:*/OUTCOME:* anchors, but also
    admits high-confidence CLM_CONCEPT claim entities when they look like
    concrete imaging markers or clinical/cognitive outcomes. When scoped
    Case Study 2 claims are present, their genes and anchors are ranked ahead
    of equally dense nodes from unrelated parts of the full community KG.
    """
    # Genes that participate in any GENESET (the "pathway_aggregated" pool).
    pathway_genes: set[str] = set()
    claim_incident: Counter[str] = Counter()
    case2_claim_incident: Counter[str] = Counter()
    gene_claim_scores: Counter[str] = Counter()
    case2_gene_claim_scores: Counter[str] = Counter()
    case2_imaging_scores: Counter[str] = Counter()
    case2_outcome_scores: Counter[str] = Counter()
    case2_endpoint_names: dict[str, set[str]] = {}
    case2_imaging_ids: set[str] = set()
    case2_outcome_ids: set[str] = set()
    case2_gene_ids: set[str] = set()
    case2_claim_endpoint_roles: dict[tuple[str, str], set[Atom]] = {}
    imaging_domains = {"biomarker", "connectivity", "imaging_feature", "neuroanatomy"}
    outcome_domains = {"treatment_outcome", "dataset_variable", "cognitive_function"}

    def _claim_scopes(claim_id: str) -> set[str]:
        claim_node = engine._index.get(claim_id)
        if claim_node is None:
            return set()
        claim_meta = claim_node.metadata or {}
        nested_meta = claim_meta.get("metadata") or {}
        scopes: set[str] = set()
        for holder in (claim_meta, nested_meta):
            values = holder.get("claim_case_study_ids") or []
            if isinstance(values, str):
                values = [values]
            scopes.update(str(value).strip() for value in values if value)
        return scopes

    def _score_gene(
        scores: Counter[str],
        nid: str,
        node_domains: set[str],
        other_domains: set[str],
    ) -> None:
        if "gene" not in node_domains:
            return
        scores[nid] += 1
        if other_domains & imaging_domains:
            scores[nid] += 3
        if other_domains & outcome_domains:
            scores[nid] += 2

    def _score_anchor(
        scores: Counter[str],
        nid: str,
        node_domains: set[str],
        other_domains: set[str],
        anchor_domains: set[str],
    ) -> None:
        if not (node_domains & anchor_domains):
            return
        scores[nid] += 1
        if other_domains & {"gene"}:
            scores[nid] += 3
        if other_domains & imaging_domains:
            scores[nid] += 2
        if other_domains & outcome_domains:
            scores[nid] += 2

    def _remember_endpoint(
        nid: str,
        name: str,
        declared_type: str,
    ) -> set[Atom]:
        if not nid:
            return set()
        name = str(name or "").strip()
        if name:
            case2_endpoint_names.setdefault(nid, set()).add(name)
        roles = _case2_endpoint_atom_roles(name, declared_type)
        if Atom.IMAGING_MARKER in roles:
            case2_imaging_ids.add(nid)
        if Atom.OUTCOME in roles:
            case2_outcome_ids.add(nid)
        if Atom.GENE_TARGET in roles:
            case2_gene_ids.add(nid)
        return roles

    # Scope membership is authoritative on the claim nodes. Building the
    # endpoint pools from those nodes also survives DiGraph edge coalescing,
    # where a different claim may be retained as the display edge.
    for claim_id, claim_node in engine._index.items():
        if "claim" not in (claim_node.domain_tags or []):
            continue
        if "case2_pathway_mediation" not in _claim_scopes(claim_id):
            continue
        claim_meta = claim_node.metadata or {}
        nested_meta = claim_meta.get("metadata") or {}
        subject_id = str(claim_meta.get("subject_id") or "")
        object_id = str(claim_meta.get("object_id") or "")
        for endpoint_id in (subject_id, object_id):
            if endpoint_id:
                case2_claim_incident[endpoint_id] += 1
        case2_claim_endpoint_roles[(claim_id, "subject")] = _remember_endpoint(
            subject_id,
            claim_meta.get("subject_name", ""),
            nested_meta.get("subject_type", ""),
        )
        case2_claim_endpoint_roles[(claim_id, "object")] = _remember_endpoint(
            object_id,
            claim_meta.get("object_name", ""),
            nested_meta.get("object_type", ""),
        )
    for u, v, d in engine.G.edges(data=True):
        if d.get("relation_type") == "part_of" and v.startswith("GENESET:"):
            pathway_genes.add(u)
        claim_id = d.get("metadata", {}).get("claim_id")
        if not claim_id:
            continue
        claim_incident[u] += 1
        claim_incident[v] += 1
        u_node = engine._index.get(u)
        v_node = engine._index.get(v)
        u_domains = set(u_node.domain_tags or []) if u_node else set()
        v_domains = set(v_node.domain_tags or []) if v_node else set()
        _score_gene(gene_claim_scores, u, u_domains, v_domains)
        _score_gene(gene_claim_scores, v, v_domains, u_domains)

        if "case2_pathway_mediation" not in _claim_scopes(claim_id):
            continue
        _score_gene(case2_gene_claim_scores, u, u_domains, v_domains)
        _score_gene(case2_gene_claim_scores, v, v_domains, u_domains)
        _score_anchor(
            case2_imaging_scores, u, u_domains, v_domains, imaging_domains
        )
        _score_anchor(
            case2_imaging_scores, v, v_domains, u_domains, imaging_domains
        )
        _score_anchor(
            case2_outcome_scores, u, u_domains, v_domains, outcome_domains
        )
        _score_anchor(
            case2_outcome_scores, v, v_domains, u_domains, outcome_domains
        )

    active_claim_incident = (
        case2_claim_incident if case2_claim_incident else claim_incident
    )
    active_gene_scores = (
        case2_gene_claim_scores if case2_gene_claim_scores else gene_claim_scores
    )

    def _im_filter(nid, node):
        if not _case2_is_claim_backed_node(nid, node, active_claim_incident):
            return False
        if nid in case2_imaging_ids:
            return True
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
        if nid.startswith("GENESET:"):
            return False
        if nid in case2_gene_ids:
            return True
        if nid in case2_imaging_ids or nid in case2_outcome_ids:
            return False
        if nid.startswith("CLM_CONCEPT:"):
            return False
        if "gene" not in (node.domain_tags or []):
            return False
        # Keep pathway genes as the Case Study 2 backbone, but allow claim-backed genes
        # into the seed pool too. The ranker below tries claim-backed genes
        # first, so the run starts in the dense case-study evidence region.
        return ((nid in pathway_genes) if pathway_genes else True) or nid in active_gene_scores

    def _outcome_filter(nid, node):
        if not _case2_is_claim_backed_node(nid, node, active_claim_incident):
            return False
        if nid in case2_outcome_ids:
            return True
        domains = set(node.domain_tags or [])
        if not (domains & outcome_domains):
            return False
        return bool(_CASE2_OUTCOME_RE.search(node.preferred_name or ""))

    def _gene_ranker(nid, node):
        return active_gene_scores.get(nid, 0)

    def _im_ranker(nid, node):
        return case2_imaging_scores.get(nid, active_claim_incident.get(nid, 0))

    def _outcome_ranker(nid, node):
        return case2_outcome_scores.get(nid, active_claim_incident.get(nid, 0))

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
        Atom.IMAGING_MARKER: _im_ranker,
        Atom.OUTCOME: _outcome_ranker,
    }
    engine._chain_atom_explicit_pools = {
        Atom.GENE_TARGET: case2_gene_ids,
        Atom.IMAGING_MARKER: case2_imaging_ids,
        Atom.OUTCOME: case2_outcome_ids,
    }
    engine._chain_claim_endpoint_roles = case2_claim_endpoint_roles
    engine._chain_claim_first_scope = case.name
    engine._chain_forbidden_bridge_ids = (
        set(engine._path_ignore_ids) | set(engine._intermediate_only_ignore_ids)
    )
    engine._chain_prefer_claim_backed_paths = True
    engine._chain_claimless_path_fraction = 0.0
    engine._chain_require_claim_backed_paths = True
    engine._chain_required_claim_scope = case.name


# ── Formal case studies ──────────────────────────────────────────────────────

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

_TASK_CASE_STUDY_NAMES: dict[str, tuple[str, str]] = {
    "biomarker_discovery": ("生物标志物发现", "Biomarker Discovery"),
    "disease_subtyping": ("疾病亚型划分", "Disease Subtyping"),
    "progression_prediction": ("疾病进展预测", "Progression Prediction"),
    "imaging_genetics": ("影像遗传学", "Imaging Genetics"),
    "differential_diagnosis": ("鉴别诊断", "Differential Diagnosis"),
    "drug_response_prediction": ("药物反应预测", "Drug Response Prediction"),
    "personalised_treatment": ("个体化治疗", "Personalised Treatment"),
    "drug_repurposing": ("药物重定位", "Drug Repurposing"),
    "adverse_event_prediction": ("不良事件预测", "Adverse Event Prediction"),
    "neuromodulation_target": ("神经调控靶点", "Neuromodulation Target"),
    "functional_localization": ("功能定位", "Functional Localisation"),
    "cognitive_decoding": ("认知解码", "Cognitive Decoding"),
    "connectome_behavior": ("连接组—行为关联", "Connectome–Behaviour"),
    "brain_age": ("脑龄", "Brain Age"),
    "prognosis": ("预后预测", "Prognosis"),
}


def _build_task_case_studies() -> tuple[CaseStudy, ...]:
    studies: list[CaseStudy] = []
    for task in CANONICAL_TASKS:
        if task.name == "transdiagnostic_clustering":
            continue
        try:
            chinese_name, english_name = _TASK_CASE_STUDY_NAMES[task.name]
        except KeyError as exc:
            raise RuntimeError(
                f"canonical task {task.name!r} has no formal case-study name"
            ) from exc
        studies.append(
            CaseStudy(
                name=task.name,
                chinese_name=chinese_name,
                english_name=english_name,
                generator=GENERATOR_TASK,
                task=task,
                stage_params=StageParams(
                    batch=BatchParams(max_paths=4, max_seeds=30, target_per_task=100),
                    novelty=NoveltyParams(top=200, alpha=0.5),
                    critic=CriticParams(top=100, max_rounds=2, threshold=0.55),
                    plausibility=PlausibilityParams(top=100),
                ),
                extras={
                    "primary_experiment_scheduled": False,
                    "hindcasting_supported": True,
                },
            )
        )
    return tuple(studies)


TASK_CASE_STUDIES = _build_task_case_studies()
CASE_STUDIES: tuple[CaseStudy, ...] = (CASE1, CASE2, *TASK_CASE_STUDIES)
CASE_STUDY_BY_NAME = {case.name: case for case in CASE_STUDIES}

if len(CASE_STUDIES) != 17 or len(CASE_STUDY_BY_NAME) != 17:
    raise RuntimeError("the formal case-study registry must contain 17 unique IDs")
if "case3_hindcasting" in CASE_STUDY_BY_NAME:
    raise RuntimeError("hindcasting is a validation protocol, not a case-study ID")

# Display numbering is deliberately derived from the frozen registry order.
# It is a UI convenience only: the slug remains the stable identifier used by
# the CLI, graph membership, audit contracts, and persisted experiment files.
CASE_STUDY_DISPLAY_NUMBERS: dict[str, int] = {
    case.name: index
    for index, case in enumerate(CASE_STUDIES, start=1)
}


def case_study_display_number(name: str) -> int:
    """Return the one-based UI number for a formal Case Study ID."""

    try:
        return CASE_STUDY_DISPLAY_NUMBERS[name]
    except KeyError as exc:
        valid = ", ".join(list_case_study_names())
        raise KeyError(f"unknown case study: {name!r} (valid: {valid})") from exc


def list_case_study_catalog() -> tuple[dict[str, object], ...]:
    """Return renderer-facing metadata without exposing mutable registry state."""

    return tuple(
        {
            "number": CASE_STUDY_DISPLAY_NUMBERS[case.name],
            "id": case.name,
            "name": case.english_name,
            "chinese_name": case.chinese_name,
            "english_name": case.english_name,
        }
        for case in CASE_STUDIES
    )


def case_study_by_name(name: str) -> CaseStudy:
    """Look up a case study by its registry slug. Raises KeyError."""
    case = CASE_STUDY_BY_NAME.get(name)
    if case is not None:
        return case
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
    "TASK_CASE_STUDIES",
    "CASE_STUDIES",
    "CASE_STUDY_BY_NAME",
    "CASE_STUDY_DISPLAY_NUMBERS",
    "case_study_by_name",
    "case_study_display_number",
    "list_case_study_catalog",
    "list_case_study_names",
    "GENERATOR_TASK",
    "GENERATOR_CHAIN",
    "GENERATOR_CASE1_CANDIDATE",
]
