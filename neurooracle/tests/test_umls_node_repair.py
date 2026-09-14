from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.ingestion.atlas_roi_modality import IMAGING_FEATURES, _build_imaging_feature_node
from neurooracle.src.node_alias_safety import blocked_canonical_ids
from neurooracle.scripts.build_umls_simplification_candidate import build_candidate as build_base, file_sha
from neurooracle.scripts.build_umls_node_repair_candidate import copy_and_repair_details, validate_patch_shape, write_repaired_graph
from neurooracle.tests.test_umls_simplification_candidate import source_graph


@pytest.mark.parametrize("query", ["fALFF", "FALFF", "dynamic fALFF", "fractional ALFF", "fractional amplitude of low-frequency fluctuation"])
def test_falff_cannot_resolve_to_alff_even_with_legacy_alias(query):
    kg = KnowledgeGraph()
    spec = dict(next(spec for spec in IMAGING_FEATURES if spec["id"] == "IF:alff"))
    spec["aliases"] = ["ALFF", "fALFF"]
    kg.add_concept(_build_imaging_feature_node(spec))
    engine = HypothesisEngine(kg)
    assert engine.resolve_name(query) is None
    assert engine.resolve_name("ALFF") == "IF:alff"


def test_ct_guard_preserves_explicit_cortical_thickness_name():
    kg = KnowledgeGraph()
    spec = dict(next(spec for spec in IMAGING_FEATURES if spec["id"] == "IF:cortical_thickness"))
    spec["aliases"] = [*spec["aliases"], "CT"]
    kg.add_concept(_build_imaging_feature_node(spec))
    engine = HypothesisEngine(kg)
    assert engine.resolve_name("CT") is None
    assert engine.resolve_name("CT imaging") is None
    assert engine.resolve_name("cortical thickness") == "IF:cortical_thickness"
    assert engine.resolve_name("cortical thickness (CT)") == "IF:cortical_thickness"
    assert engine.resolve_name("CortThick") == "IF:cortical_thickness"


def test_new_seed_does_not_reintroduce_retired_aliases():
    nodes = {spec["id"]: _build_imaging_feature_node(spec) for spec in IMAGING_FEATURES}
    assert "CT" not in nodes["IF:cortical_thickness"].aliases
    assert "fALFF" not in nodes["IF:alff"].aliases
    assert "ALFF" in nodes["IF:alff"].aliases


@pytest.mark.parametrize("query", ["CT", "ct", " CT ", "ＣＴ"])
def test_ct_cannot_match_letters_inside_fluctuation(query):
    kg = KnowledgeGraph()
    for spec in IMAGING_FEATURES:
        if spec["id"] in {"IF:alff", "IF:cortical_thickness"}:
            kg.add_concept(_build_imaging_feature_node(spec))
    assert HypothesisEngine(kg).resolve_name(query) is None


def test_exact_ct_alias_for_a_distinct_entity_remains_eligible():
    kg = KnowledgeGraph()
    spec = {"id": "MOD:computed_tomography", "name": "Computed Tomography",
            "aliases": ["CT"], "desc": "tomographic imaging", "modality": "CT"}
    kg.add_concept(_build_imaging_feature_node(spec))
    assert HypothesisEngine(kg).resolve_name("CT") == "MOD:computed_tomography"


def test_other_names_and_legitimate_falff_node_are_not_blocked():
    assert blocked_canonical_ids("ACTN1") == frozenset()
    assert blocked_canonical_ids("ALFF") == frozenset()
    assert "IF:falff" not in blocked_canonical_ids("fALFF")
    kg = KnowledgeGraph()
    spec = {"id": "IF:falff", "name": "fractional amplitude of low-frequency fluctuation",
            "aliases": ["fALFF"], "desc": "ratio", "modality": "fMRI"}
    kg.add_concept(_build_imaging_feature_node(spec))
    assert HypothesisEngine(kg).resolve_name("fALFF") == "IF:falff"


def test_repair_is_exact_reversible_and_preserves_real_edges(source_graph, tmp_path):
    source, source_data = source_graph
    base, output = tmp_path / "base", tmp_path / "repaired"
    base_build = build_base(source, base)
    output.mkdir()
    before = source_data["concepts"]["IF:fa"]
    after = copy.deepcopy(before)
    after["semantic_types"] = ["T047"]
    after["aliases"] = []
    patch = {"node_id": "IF:fa", "fields": ["semantic_types", "aliases"], "before": before, "after": after}
    graph_result = write_repaired_graph(base / "knowledge_graph.candidate.json", output, {"IF:fa": patch}, base_build["candidate"], 2)
    base_graph = json.loads((base / "knowledge_graph.candidate.json").read_text(encoding="utf-8"))
    repaired = json.loads((output / "knowledge_graph.candidate.json").read_text(encoding="utf-8"))
    assert repaired["concepts"]["IF:fa"] == after
    repaired["concepts"]["IF:fa"] = before
    assert repaired["concepts"] == base_graph["concepts"]
    assert repaired["edges"] == base_graph["edges"]
    assert graph_result["changed_nodes"] == 1
    assert graph_result["counts"]["claims"] == 1
    with sqlite3.connect(base / "umls_details.sqlite") as connection:
        old = connection.execute("SELECT * FROM alignment_candidates ORDER BY atom_id,target_id").fetchall()
        raw_before = {table: connection.execute(f"SELECT * FROM {table} ORDER BY ordinal").fetchall() for table in ("atoms", "cuis", "mappings")}
    retired = [{"source_row": list(row), "reason": "retired_global_alias", "atom_name": "FA"} for row in old]
    base_sha = file_sha(base / "umls_details.sqlite")
    result = copy_and_repair_details(base / "umls_details.sqlite", output, retired, base_sha)
    assert result["alignment_pairs_active"] == 0
    assert result["alignment_pairs_retired"] == 2
    assert file_sha(base / "umls_details.sqlite") == base_sha
    with sqlite3.connect(output / "umls_details.sqlite") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        history = connection.execute("SELECT atom_id,target_id,mapping_status,shared_domains,decision FROM retired_alignment_candidates ORDER BY atom_id,target_id").fetchall()
        assert history == old
        for table, old_rows in raw_before.items():
            assert connection.execute(f"SELECT * FROM {table} ORDER BY ordinal").fetchall() == old_rows


def test_repair_rejects_identity_domain_and_metadata_changes():
    before = {"id": "CUI:C0000001", "semantic_types": ["T047"], "domain_tags": ["disease"], "metadata": {}}
    for field, value in (("id", "OTHER"), ("domain_tags", ["drug"]), ("metadata", {"new": True})):
        after = {**before, field: value}
        with pytest.raises(ValueError):
            validate_patch_shape({"node_id": before["id"], "fields": [field], "before": before, "after": after})
