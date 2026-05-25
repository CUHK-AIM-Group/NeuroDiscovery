"""Tests for recipe.generator (Input Recipe revival)."""

from __future__ import annotations

import json

from neurooracle.src.recipe.generator import (
    Recipe,
    brainstorm_recipes,
    build_inventory,
    link_to_concepts,
)


def _concept(cid: str, name: str, domain: str, aliases=None) -> tuple[str, dict]:
    return cid, {
        "id": cid,
        "preferred_name": name,
        "domain_tags": [domain],
        "aliases": aliases or [],
    }


def test_build_inventory_caps_and_dedupes():
    concepts = dict([
        _concept("M1", "fMRI", "modality"),
        _concept("M2", "sMRI", "modality"),
        _concept("M3", "fMRI", "modality"),  # dup
        _concept("G1", "APOE", "gene"),
        _concept("R1", "Hippocampus", "neuroanatomy"),
        _concept("X1", "irrelevant claim", "claim"),  # dropped: not in buckets
    ])
    inv = build_inventory(concepts)
    assert inv["modality"] == ["fMRI", "sMRI"]
    assert "claim" not in inv
    assert inv["gene"] == ["APOE"]


def test_brainstorm_recipes_parses_json_array():
    inv = {"modality": ["fMRI"], "neuroanatomy": ["Hippocampus"]}
    payload = [
        {"name": "hippocampal volume",
         "formula": "volume(Hippocampus) from sMRI",
         "inputs_used": ["sMRI", "Hippocampus"]},
        {"name": "FC of DMN",
         "formula": "mean pairwise FC of DMN nodes from fMRI",
         "inputs_used": ["fMRI"]},
    ]
    captured = {}

    def stub(prompt: str, system_prompt: str) -> str:
        captured["prompt"] = prompt
        captured["sys"] = system_prompt
        return json.dumps(payload)

    recipes = brainstorm_recipes(inv, n=2, llm_call=stub, model_name="stub-1")
    assert len(recipes) == 2
    assert recipes[0].id == "recipe_0001"
    assert recipes[0].name == "hippocampal volume"
    assert recipes[0].llm_model == "stub-1"
    # Spec: prompt must NOT positively frame any downstream task
    low = captured["prompt"].lower()
    for forbidden in ("predict ", "diagnose", "classify"):
        assert forbidden not in low, f"prompt leaked task word: {forbidden}"


def test_brainstorm_recipes_handles_code_block():
    inv = {"modality": ["fMRI"]}

    def stub(prompt, sys):
        return '```json\n[{"name": "x", "formula": "y", "inputs_used": ["fMRI"]}]\n```'

    recipes = brainstorm_recipes(inv, n=1, llm_call=stub)
    assert len(recipes) == 1
    assert recipes[0].name == "x"


def test_brainstorm_recipes_skips_malformed_items():
    def stub(prompt, sys):
        return json.dumps([
            {"name": "ok", "formula": "f"},
            {"name": "", "formula": "f"},      # empty name
            {"formula": "f"},                   # no name
            "not a dict",
        ])

    recipes = brainstorm_recipes({"modality": ["fMRI"]}, n=4, llm_call=stub)
    assert [r.name for r in recipes] == ["ok"]


def test_link_to_concepts_matches_substrings():
    concepts = dict([
        _concept("R1", "Hippocampus", "neuroanatomy"),
        _concept("R2", "Prefrontal Cortex", "neuroanatomy"),
        _concept("G1", "APOE epsilon4 allele", "gene"),
        _concept("D1", "Alzheimer", "disease"),  # disease not in linkable defaults
    ])
    recipes = [
        Recipe(id="r1", name="hippocampal volume",
               formula="volume(Hippocampus) from sMRI", inputs_used=[]),
        Recipe(id="r2", name="cortex thickness",
               formula="mean cortical thickness of Prefrontal Cortex",
               inputs_used=[]),
    ]
    link_to_concepts(recipes, concepts)
    assert "R1" in recipes[0].linked_concept_ids
    assert "R2" in recipes[1].linked_concept_ids
    # Disease not in default linkable_domains
    for r in recipes:
        assert "D1" not in r.linked_concept_ids
