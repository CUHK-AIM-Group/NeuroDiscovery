"""Curated grounding from literature outcome names to dataset metadata columns.

Phase 2 claims often preserve the wording used by the abstract. That means
targets such as "cognitive performance" or "executive functions" can be valid
literature outcomes, but Phase 3 still needs a concrete dataset column before an
experiment can run. This module keeps that bridge explicit and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class OutcomeGrounding:
    """One candidate dataset column for a normalized outcome concept."""

    outcome: str
    aliases: tuple[str, ...]
    dataset: str
    dataset_node_id: str
    domain_node_id: str
    metadata_column: str
    label_key: str | None = None
    label_file: str | None = None
    status: str = "domain_only"
    task_type: str = "regression"
    direction: str | None = None
    priority: int = 100
    notes: str = ""

    @property
    def has_local_label(self) -> bool:
        return self.status == "local_label" and bool(self.label_file)


def _hcp(
    outcome: str,
    aliases: Iterable[str],
    domain_node_id: str,
    metadata_column: str,
    label_key: str,
    priority: int,
    notes: str = "",
) -> OutcomeGrounding:
    return OutcomeGrounding(
        outcome=outcome,
        aliases=tuple(aliases),
        dataset="HCP_YA",
        dataset_node_id="DATASET:HCP_YA",
        domain_node_id=domain_node_id,
        metadata_column=metadata_column,
        label_key=label_key,
        label_file=f"data/hcp_{label_key}_labels.csv",
        status="local_label",
        priority=priority,
        notes=notes,
    )


OUTCOME_GROUNDINGS: tuple[OutcomeGrounding, ...] = (
    # Broad cognition: keep several HCP choices because abstracts often do not
    # say whether "cognitive performance" means total, fluid, or task-specific
    # cognition. Phase 3 can choose the top available candidate or keep all.
    _hcp(
        "general cognition",
        ("general cognition", "global cognition", "cognitive performance", "cognition"),
        "HCP:DOM_COG_TOTAL",
        "CogTotalComp_Unadj",
        "cogfluidcomp",
        10,
        "Local HCP label is the closest available composite cognition target.",
    ),
    _hcp(
        "fluid cognition",
        ("fluid cognition", "fluid intelligence", "reasoning", "cognitive ability"),
        "HCP:DOM_COG_FLUID",
        "CogFluidComp_Unadj",
        "cogfluid",
        20,
    ),
    _hcp(
        "matrix reasoning",
        ("pmat", "ravens progressive matrices", "raven's progressive matrices"),
        "HCP:DOM_COG_FLUID",
        "PMAT24_A_CR",
        "pmat24",
        25,
    ),
    _hcp(
        "crystallized cognition",
        ("crystallized cognition", "verbal cognition", "language cognition"),
        "HCP:DOM_COG_CRYST",
        "CogCrystalComp_Unadj",
        "cogcrystal",
        30,
    ),
    _hcp(
        "vocabulary",
        ("vocabulary", "picture vocabulary"),
        "HCP:DOM_COG_CRYST",
        "PicVocab_Unadj",
        "picvocab",
        35,
    ),
    _hcp(
        "reading",
        ("reading", "oral reading"),
        "HCP:DOM_COG_CRYST",
        "ReadEng_Unadj",
        "readeng",
        36,
    ),
    # Executive / attention / memory targets.
    _hcp(
        "executive function",
        ("executive function", "executive functions", "executive functioning"),
        "HCP:DOM_COG_FLUID",
        "Flanker_Unadj",
        "flanker",
        10,
        "Primary HCP proxy for inhibitory control / attention.",
    ),
    _hcp(
        "cognitive flexibility",
        ("cognitive flexibility", "set shifting", "executive control", "executive function", "executive functions"),
        "HCP:DOM_COG_FLUID",
        "CardSort_Unadj",
        "cardsort",
        15,
    ),
    _hcp(
        "working memory",
        ("working memory", "n-back", "n back", "executive function", "executive functions"),
        "HCP:DOM_COG_FLUID",
        "WM_2BK_Acc",
        "wm_2bk_acc",
        20,
    ),
    _hcp(
        "list sorting working memory",
        ("list sorting", "list sort", "working memory capacity", "executive function", "executive functions"),
        "HCP:DOM_COG_FLUID",
        "ListSort_Unadj",
        "listsort",
        25,
    ),
    _hcp(
        "episodic memory",
        ("episodic memory", "memory", "memory performance"),
        "HCP:DOM_COG_FLUID",
        "PicSeq_Unadj",
        "listsort",
        45,
        "HCP local label set lacks a dedicated PicSeq file; ListSort is the ready local fallback.",
    ),
    _hcp(
        "processing speed",
        ("processing speed", "psychomotor speed"),
        "HCP:DOM_COG_FLUID",
        "ProcSpeed_Unadj",
        "procspeed",
        30,
    ),
    # Clinical / behavioral HCP outcomes.
    _hcp(
        "depression severity",
        ("depression severity", "depressive symptoms", "depressive disorder"),
        "HCP:DOM_PSYCH",
        "NEOFAC_N / NIH Toolbox negative affect proxy",
        "neg_affect",
        20,
    ),
    _hcp(
        "psychological distress",
        ("psychological distress", "stress", "perceived stress"),
        "HCP:DOM_PSYCH",
        "PercStress_Unadj",
        "percstress",
        10,
    ),
    _hcp(
        "sleep quality",
        ("sleep quality", "insomnia", "sleep disturbance"),
        "HCP:DOM_PSYCH",
        "PSQI_Score",
        "psqi",
        10,
    ),
    _hcp(
        "social cognition",
        ("social cognition", "theory of mind", "social interaction"),
        "HCP:DOM_SOCIAL",
        "Social_Task_TOM",
        "social_tom_perc_tom",
        10,
    ),
    # ADNI: current KG has domain nodes and scale bridges, but local case-study
    # labels are not yet materialized as per-scale CSV files in data/labels.
    OutcomeGrounding(
        "MMSE",
        ("mmse", "mini mental state examination", "mini-mental state examination"),
        "ADNI",
        "DATASET:ADNI",
        "ADNI:DOM_NEUROPSYCH",
        "MMSE",
        label_key="mmse",
        status="domain_only",
        direction="higher_better",
        priority=10,
        notes="KG bridge exists; local ADNI MMSE label CSV still needs materialization.",
    ),
    OutcomeGrounding(
        "ADAS-Cog",
        ("adas-cog", "adas cog", "alzheimer disease assessment scale cognitive"),
        "ADNI",
        "DATASET:ADNI",
        "ADNI:DOM_NEUROPSYCH",
        "ADAS-Cog11/ADAS-Cog13",
        label_key="adas_cog",
        status="domain_only",
        direction="lower_better",
        priority=10,
    ),
    OutcomeGrounding(
        "CDR-SB",
        ("cdr-sb", "cdr sum of boxes", "clinical dementia rating sum of boxes"),
        "ADNI",
        "DATASET:ADNI",
        "ADNI:DOM_NEUROPSYCH",
        "CDRSB",
        label_key="cdr_sb",
        status="domain_only",
        direction="lower_better",
        priority=10,
    ),
    OutcomeGrounding(
        "MCI/AD progression",
        ("progression", "conversion", "mci conversion", "mci to ad conversion"),
        "ADNI",
        "DATASET:ADNI",
        "ADNI:DOM_PROGRESSION",
        "DX progression / time-to-conversion",
        label_key="adni_dx_3way",
        label_file="data/labels/adni2_dx_3way_labels.csv",
        status="local_label",
        task_type="classification",
        priority=20,
    ),
    # UKB: extractor mappings exist; local raw UKB-derived labels are not present.
    OutcomeGrounding(
        "fluid intelligence",
        ("ukb fluid intelligence", "fluid intelligence score"),
        "UKB",
        "DATASET:UKB",
        "UKB:CAT110011",
        "p20016_i0",
        label_key="fluid_intelligence",
        status="extractor_mapping",
        direction="higher_better",
        priority=40,
        notes="Prepared in the UKB extractor; requires local UKB raw phenotype export.",
    ),
    OutcomeGrounding(
        "reaction time",
        ("reaction time", "mean reaction time"),
        "UKB",
        "DATASET:UKB",
        "UKB:CAT110011",
        "p20023_i0",
        label_key="reaction_time",
        status="extractor_mapping",
        direction="lower_better",
        priority=45,
    ),
    OutcomeGrounding(
        "trail making",
        ("trail making", "trail making test", "executive function"),
        "UKB",
        "DATASET:UKB",
        "UKB:CAT110011",
        "p20156_i0 / p20157_i0",
        label_key="trail_making",
        status="extractor_mapping",
        direction="lower_better",
        priority=45,
    ),
)


_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_outcome_name(name: str) -> str:
    """Normalize claim target names for alias lookup."""

    text = name.lower().replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\bfunctions\b", "function", text)
    text = re.sub(r"\bperformances\b", "performance", text)
    return " ".join(text.split())


def _aliases(g: OutcomeGrounding) -> set[str]:
    return {normalize_outcome_name(g.outcome), *(normalize_outcome_name(a) for a in g.aliases)}


def ground_outcome(
    name: str,
    *,
    dataset: str | None = None,
    require_local_label: bool = False,
    max_results: int | None = None,
) -> list[OutcomeGrounding]:
    """Return candidate dataset columns for an outcome mention."""

    needle = normalize_outcome_name(name)
    dataset_norm = dataset.upper().replace("-", "_") if dataset else None
    hits: list[OutcomeGrounding] = []
    for grounding in OUTCOME_GROUNDINGS:
        if dataset_norm and grounding.dataset.upper().replace("-", "_") != dataset_norm:
            continue
        if require_local_label and not grounding.has_local_label:
            continue
        aliases = _aliases(grounding)
        if needle in aliases or any(needle in a or a in needle for a in aliases):
            hits.append(grounding)
    hits.sort(key=lambda g: (not g.has_local_label, g.priority, g.dataset, g.label_key or ""))
    if max_results is not None:
        return hits[:max_results]
    return hits


def best_grounding(
    name: str,
    *,
    dataset: str | None = None,
    require_local_label: bool = False,
) -> OutcomeGrounding | None:
    """Return the highest-priority grounding for an outcome mention."""

    hits = ground_outcome(name, dataset=dataset, require_local_label=require_local_label, max_results=1)
    return hits[0] if hits else None


def hcp_label_for_target(target_name: str) -> str | None:
    """Return the primary ready HCP label key for a hypothesis target."""

    grounding = best_grounding(target_name, dataset="HCP_YA", require_local_label=True)
    return grounding.label_key if grounding else None


def target_to_hcp_label_mapping() -> dict[str, str]:
    """Build a backward-compatible target-name -> HCP label-key map."""

    mapping: dict[str, str] = {}
    for grounding in OUTCOME_GROUNDINGS:
        if grounding.dataset != "HCP_YA" or not grounding.has_local_label or not grounding.label_key:
            continue
        for name in (grounding.outcome, *grounding.aliases):
            mapping.setdefault(name, grounding.label_key)
    return mapping


__all__ = [
    "OutcomeGrounding",
    "OUTCOME_GROUNDINGS",
    "best_grounding",
    "ground_outcome",
    "hcp_label_for_target",
    "normalize_outcome_name",
    "target_to_hcp_label_mapping",
]
