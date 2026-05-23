"""Tests for the path-specificity filter."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from neurooracle.src.kge.specificity import (
    HUB_BLACKLIST_IDS,
    GENERIC_NODE_NAMES,
    path_specificity,
)


@dataclass
class _Link:
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    relation_type: str = "rel"


@dataclass
class _Hyp:
    path: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def test_clean_path_scores_one():
    h = _Hyp(path=[
        _Link("MSH:D000544", "Alzheimer Disease", "MSH:D006625", "Hippocampus"),
        _Link("MSH:D006625", "Hippocampus", "GENE:APOE", "APOE"),
    ])
    score, issues = path_specificity(h)
    assert score == 1.0
    assert issues == []


def test_hub_blacklist_id_flagged():
    """An MSH ID in HUB_BLACKLIST_IDS triggers a hub_blacklist issue."""
    nervous = next(iter(HUB_BLACKLIST_IDS))
    h = _Hyp(path=[
        _Link(nervous, "Nervous System Diseases", "MSH:D006625", "Hippocampus"),
    ])
    score, issues = path_specificity(h)
    assert score < 1.0
    assert any("hub_blacklist" in i for i in issues)


def test_generic_name_flagged_even_without_id_match():
    """A node whose name is in GENERIC_NODE_NAMES is flagged regardless of ID."""
    h = _Hyp(path=[
        _Link("FOO:1", "Cerebrum", "MSH:D006625", "Hippocampus"),
    ])
    score, issues = path_specificity(h)
    # "Cerebrum" hits both HUB_BLACKLIST_IDS (MSH:D054022) by name and
    # GENERIC_NODE_NAMES — only one issue per unique node either way.
    assert "Cerebrum" in GENERIC_NODE_NAMES
    assert any("Cerebrum" in i for i in issues)
    assert score == 0.5  # 1 issue / 2 unique nodes


def test_clm_concept_cohort_phrase_flagged():
    h = _Hyp(path=[
        _Link("CLM_CONCEPT:250_healthy_controls", "250 healthy controls",
              "MSH:D005625", "Frontal Lobe"),
    ])
    score, issues = path_specificity(h)
    assert any("vague_clm" in i for i in issues)
    assert score == 0.5


def test_clm_concept_methodology_phrase_flagged():
    h = _Hyp(path=[
        _Link("CLM_CONCEPT:much_heterogeneity_among_studies",
              "much heterogeneity among studies",
              "MSH:D010146", "Pain"),
    ])
    score, issues = path_specificity(h)
    assert any("vague_clm" in i for i in issues)


def test_clm_concept_pure_gene_name_not_flagged():
    """Real gene-like CLM_CONCEPT nodes (e.g. P-glycoprotein) must not trigger."""
    h = _Hyp(path=[
        _Link("CLM_CONCEPT:P-glycoprotein", "P-glycoprotein",
              "MSH:D001812", "Blood-Brain Barrier"),
    ])
    score, issues = path_specificity(h)
    assert score == 1.0
    assert issues == []


def test_visual_stimulus_outside_visual_task_flagged():
    h = _Hyp(
        path=[_Link("VS:coco:person", "person",
                    "MSH:D005625", "Frontal Lobe")],
        metadata={"task_kind": "imaging-disease"},
    )
    score, issues = path_specificity(h)
    assert any("misplaced_visual_stimulus" in i for i in issues)


def test_visual_stimulus_inside_visual_task_ok():
    h = _Hyp(
        path=[_Link("VS:coco:person", "person",
                    "MSH:D005625", "Frontal Lobe")],
        metadata={"task_kind": "visual-decoding"},
    )
    score, issues = path_specificity(h)
    assert score == 1.0


def test_extreme_hub_with_umbrella_name_flagged():
    """A node with degree > 5000 AND an umbrella-style name is flagged."""
    h = _Hyp(path=[
        _Link("CUSTOM:big_umbrella", "Movement Disorders",
              "MSH:D006625", "Hippocampus"),
    ])
    degrees = {"CUSTOM:big_umbrella": 9999, "MSH:D006625": 100}
    # "Movement Disorders" is also in GENERIC_NODE_NAMES so it should be flagged
    # via the generic-name rule before the extreme-hub rule fires.
    score, issues = path_specificity(h, degrees=degrees)
    assert any("Movement Disorders" in i for i in issues)


def test_extreme_hub_specific_name_not_flagged():
    """High-degree node with a specific name (e.g. Alzheimer Disease) stays clean.

    The earlier extreme-hub regex rule was removed because words like
    'Disease' inside legitimate names ("Alzheimer Disease", "Parkinson
    Disease") triggered false positives. Hub umbrella protection now relies
    on the explicit ``HUB_BLACKLIST_IDS`` / ``GENERIC_NODE_NAMES`` lists.
    """
    h = _Hyp(path=[
        _Link("MSH:D000544", "Alzheimer Disease",
              "MSH:D006625", "Hippocampus"),
    ])
    degrees = {"MSH:D000544": 26559, "MSH:D006625": 5000}
    score, issues = path_specificity(h, degrees=degrees)
    assert score == 1.0
    assert issues == []


def test_score_drops_proportional_to_issue_count():
    """3 unique nodes, 2 flagged → score = 1/3."""
    h = _Hyp(path=[
        _Link("CLM_CONCEPT:interventions", "interventions",
              "MSH:D010146", "Pain"),
        _Link("MSH:D010146", "Pain",
              "CLM_CONCEPT:much_heterogeneity_among_studies",
              "much heterogeneity among studies"),
    ])
    score, issues = path_specificity(h)
    assert len(issues) == 2
    assert score == pytest.approx(1.0 / 3.0)


def test_dedup_repeated_node():
    """A node appearing twice in the path counts once."""
    h = _Hyp(path=[
        _Link("CLM_CONCEPT:interventions", "interventions",
              "MSH:D010146", "Pain"),
        _Link("MSH:D010146", "Pain",
              "CLM_CONCEPT:interventions", "interventions"),
    ])
    score, issues = path_specificity(h)
    # 2 unique nodes; "interventions" flagged once
    assert len(issues) == 1
    assert score == 0.5


def test_empty_path_returns_one():
    h = _Hyp(path=[])
    score, issues = path_specificity(h)
    assert score == 1.0
    assert issues == []
