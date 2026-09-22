from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurooracle.src.feedback_state import FeedbackRecord, FeedbackState

from neurooracle.scripts.aggregate_case_study_hindcasting import (
    build_coverage_rows,
    collect_metrics,
)
from neurooracle.scripts.case_study_hindcasting_eval import (
    _benchmark_status,
    _aggregate,
    _claim_case_study_ids,
    _future_indexes,
    _historical_pairs,
    _random_baseline,
    _score_hypothesis,
    _validate_hindcasting_hypotheses,
    load_future_claim_records,
)
from neurooracle.scripts.audit_case_study_hindcasting_executability import (
    component_summary,
)
from neurooracle.scripts.audit_hindcasting_candidate_coverage import (
    _candidate_indexes,
)
from neurooracle.scripts.generate_case_study_frozen_baselines import (
    ENDPOINT_CANONICAL_QUALITY_POLICY,
    EndpointEvidence,
    FrozenEdge,
    FrozenGraphIndex,
    MultiInputCandidate,
    _atom_route,
    _endpoint_canonical_quality,
    _endpoint_evidence_pool,
    _coverage_first_endpoint_rows,
    _diverse_multi_input_top,
    _endpoint_rank_band,
    _generation_endpoint_atom_routes,
    _interleave_neurodiscovery_exploration,
    _interleave_evidence_frontier,
    _blend_kge_score,
    _kge_retrieval_quota_rows,
    _rank_stratified_kge_anchor_ids,
    _rank_stratified_kge_rows,
    _is_feedback_endpoint_variant,
    _neurodiscovery_endpoint_pair_rows,
    _ordered_neurodiscovery_exploration_buckets,
    _ranked_endpoint_compositions,
    _select_endpoint_frontier,
    _semantic_context_score,
    _select_endpoint_evidence_head_tail,
    _stratified_anchor_fanout_rows,
    generate_case_hypotheses,
    load_semantic_claim_adjacencies,
)
from neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates import (
    _apply_candidate_canonical_quality,
    _atomic_json_temporary_path,
    _feedback_conditioned_endpoint_mutations,
    _make_frontier_candidate_executable,
    _frontier_index_for_case,
    _merge_neurodiscovery_frontier_payload,
    _merge_hybrid_candidate_payloads,
    _scoped_frontier_seed_mode,
    _augment_scoped_frontier,
    _tag,
    merge_generation_manifests,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    _enforce_fixed_budget,
    _generation_command,
    _is_complete_fixed_budget_payload,
    parse_window,
)
from neurooracle.scripts.run_case_study_hindcasting_matrix import (
    command_for_case_study,
    select_case_studies,
)
from neurooracle.scripts.screen_neurodiscovery_endpoint_quality import (
    _candidate_node_ids,
    _method_label,
    _select_source_runs,
)
from neurooracle.scripts.screen_neurodiscovery_hybrid_prefix import (
    _branch_paths,
    _method_label as hybrid_prefix_method_label,
)
from neurooracle.scripts.screen_neurodiscovery_fixed_candidate_ablation import (
    METHOD_GENERAL_LEGACY,
    METHOD_HYBRID_LEGACY,
    METHOD_SCOPED_LEGACY,
    _legacy_pool_variants,
)
from neurooracle.src.atoms import Atom, task_by_name
from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.hypothesis_cli import cmd_batch
from neurooracle.src.hypothesis_engine import Hypothesis, HypothesisEngine, HypothesisLink

import neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates as replicate_generator
import neurooracle.scripts.run_case_study_hindcasting as hindcast_runner


def test_atomic_json_temporary_path_does_not_repeat_long_target_name(
    tmp_path: Path,
) -> None:
    target = tmp_path / ("candidate_branch_" + "x" * 180 + ".json")
    temporary = _atomic_json_temporary_path(target)

    assert temporary.parent == target.parent
    assert target.name not in temporary.name
    assert len(temporary.name) < 20


def test_incomplete_fixed_budget_payload_is_not_reused() -> None:
    assert not _is_complete_fixed_budget_payload(
        {"hypotheses": [], "metadata": {}},
        case_study_id="case2_pathway_mediation",
        target_per_case_study=1000,
    )


def test_complete_fixed_budget_payload_is_reusable() -> None:
    payload = {
        "hypotheses": [{"id": f"H:{index}"} for index in range(1000)],
        "metadata": {
            "uses_future_outcomes": False,
            "fixed_budget": {
                "case_study_id": "case2_pathway_mediation",
                "requested": 1000,
                "requested_evaluation_prefix": 1000,
                "failure_slots_in_evaluation_prefix": 0,
                "ranked_pool_after_padding": 1000,
            },
        },
    }

    assert _is_complete_fixed_budget_payload(
        payload,
        case_study_id="case2_pathway_mediation",
        target_per_case_study=1000,
    )
    payload["metadata"]["fixed_budget"]["case_study_id"] = "imaging_genetics"
    assert not _is_complete_fixed_budget_payload(
        payload,
        case_study_id="case2_pathway_mediation",
        target_per_case_study=1000,
    )


def test_hindcasting_resume_regenerates_incomplete_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    output = output_dir / "hypotheses_raw.json"
    output.write_text(
        json.dumps({"hypotheses": [], "metadata": {}}),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, *, check, env):
        calls.append(command)
        assert check is True
        assert env["PYTHONHASHSEED"] == "7"
        output.write_text(
            json.dumps({"hypotheses": [{"id": "HYP:regenerated"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(hindcast_runner.subprocess, "run", fake_run)
    result = hindcast_runner.generate_hypotheses(
        case_study_id="brain_age",
        kg_path=tmp_path / "knowledge_graph.json",
        output_dir=output_dir,
        target_per_case_study=1,
        generation_pool_size=1,
        seed=7,
        force=False,
    )

    assert len(calls) == 1
    assert result["hypotheses"][0]["id"] == "HYP:regenerated"
    assert result["metadata"]["fixed_budget"]["valid_before_padding"] == 1


def test_case_study_selection_preserves_order_and_exclusions() -> None:
    selected = select_case_studies(
        ["prognosis", "brain_age", "prognosis"],
        ["prognosis"],
    )
    assert [case.name for case in selected] == ["brain_age"]
    with pytest.raises(KeyError):
        select_case_studies(["case3_hindcasting"], [])


def test_endpoint_frontier_receives_independent_replicate_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "frontier.json"
    output.write_text(
        json.dumps(
            {
                "n_hypotheses": 0,
                "hypotheses": [],
                "metadata": {
                    "generation_mode": "dynamic_scoped_evidence_frontier"
                },
            }
        ),
        encoding="utf-8",
    )
    index = SimpleNamespace()
    captured: dict[str, int] = {}
    monkeypatch.setattr(
        replicate_generator,
        "_frontier_index_for_case",
        lambda _engine, _case: (index, {}, 1),
    )

    def fake_generate_case_hypotheses(**kwargs):
        captured["replicate_index"] = int(kwargs["replicate_index"])
        return {"n_hypotheses": 0, "hypotheses": [], "metadata": {}}

    monkeypatch.setattr(
        replicate_generator,
        "generate_case_hypotheses",
        fake_generate_case_hypotheses,
    )
    monkeypatch.setattr(
        replicate_generator,
        "_merge_neurodiscovery_frontier_payload",
        lambda *_args, **_kwargs: {
            "n_hypotheses": 0,
            "hypotheses": [],
            "metadata": {"neurodiscovery_evidence_frontier": {}},
        },
    )

    _augment_scoped_frontier(
        SimpleNamespace(),
        output,
        case=case_study_by_name("biomarker_discovery"),
        freeze_year=2016,
        target=10,
        seed=123,
        replicate_index=7,
        generation_round=2,
        excluded_semantic_keys=frozenset(),
        evidence_frontier_fraction=1.0,
    )

    assert captured["replicate_index"] == 7


def test_frontier_index_keeps_claim_local_readouts_and_canonical_names() -> None:
    concepts = {
        "CUI:CEREBELLUM": SimpleNamespace(
            id="CUI:CEREBELLUM",
            preferred_name="Cerebellum",
            aliases=["Cerebellar"],
            domain_tags=["neuroanatomy"],
            metadata={"atom_types": ["imaging_marker"]},
        ),
        "D:MDD": SimpleNamespace(
            id="D:MDD",
            preferred_name="Major depressive disorder",
            aliases=["Depressive disorder"],
            domain_tags=["disease"],
            metadata={"atom_types": ["disease"]},
        ),
    }
    engine = SimpleNamespace(
        _index=concepts,
        kg=SimpleNamespace(),
        G=SimpleNamespace(edges=lambda data=True: []),
    )
    local_readout = SimpleNamespace(
        entity_id="CLAIM_ENTITY:READOUT",
        name="cerebellar network homogeneity",
        atoms=("imaging_marker",),
        uses_canonical_id=False,
    )
    canonical_disease = SimpleNamespace(
        entity_id="D:MDD",
        name="major depressive disorder psychopathology",
        atoms=("disease",),
        uses_canonical_id=True,
    )
    records = [{
        "subject": local_readout,
        "object": canonical_disease,
        "predicate": "correlates_with",
        "confidence": 0.8,
        "claim_id": "CLM:1",
        "raw_text": "",
        "source_paper": {"year": 2014},
    }]
    engine._semantic_claim_records_for_scope = lambda _scope: records

    index, adjacency, count = _frontier_index_for_case(
        engine,
        case_study_by_name("case1_transdiagnostic"),
    )

    assert count == 1
    assert index.names["CLAIM_ENTITY:READOUT"] == "cerebellar network homogeneity"
    assert index.names["D:MDD"] == "Major depressive disorder"
    assert set(adjacency) == {"CLAIM_ENTITY:READOUT", "D:MDD"}


def test_semantic_context_prefers_a_matching_disease_neighbour() -> None:
    names = {
        "IM:SOURCE": "hippocampal volume",
        "D:AD": "Alzheimer disease",
        "D:SCZ": "schizophrenia",
        "O:AD": "Alzheimer disease conversion",
        "O:SCZ": "schizophrenia onset",
    }
    edge = FrozenEdge(
        source_id="IM:SOURCE",
        target_id="D:AD",
        relation="is_biomarker_of",
        confidence=0.9,
    )
    adjacency = {"IM:SOURCE": [edge], "D:AD": [edge]}
    index = FrozenGraphIndex(
        concepts={node_id: {} for node_id in names},
        names=names,
        atoms={},
        adjacency=adjacency,
        direct_pairs=set(),
    )
    cache: dict[str, tuple[frozenset[str], frozenset[str]]] = {}

    matching = _semantic_context_score(
        index, adjacency, "IM:SOURCE", "O:AD", cache
    )
    mismatching = _semantic_context_score(
        index, adjacency, "IM:SOURCE", "O:SCZ", cache
    )

    assert matching > mismatching


def test_neurodiscovery_frontier_uses_frozen_global_context_without_replacing_scope() -> None:
    concepts = {
        "IM:SCOPED": {
            "preferred_name": "hippocampal volume",
            "domain_tags": ["imaging_feature"],
        },
        "IM:GLOBAL": {
            "preferred_name": "cortical thickness",
            "domain_tags": ["imaging_feature"],
        },
        "D:AD": {
            "preferred_name": "Alzheimer disease",
            "domain_tags": ["disease"],
        },
        "D:SCZ": {
            "preferred_name": "schizophrenia",
            "domain_tags": ["disease"],
        },
        "CTX:AD": {
            "preferred_name": "Alzheimer conversion",
            "domain_tags": ["clinical_outcome"],
        },
        "CTX:SCZ": {
            "preferred_name": "psychosis onset",
            "domain_tags": ["clinical_outcome"],
        },
    }
    atoms = {
        "IM:SCOPED": frozenset({"imaging_marker"}),
        "IM:GLOBAL": frozenset({"imaging_marker"}),
        "D:AD": frozenset({"disease"}),
        "D:SCZ": frozenset({"disease"}),
        "CTX:AD": frozenset({"outcome"}),
        "CTX:SCZ": frozenset({"outcome"}),
    }
    scoped_edge = FrozenEdge(
        "IM:SCOPED",
        "CTX:AD",
        "is_associated_with",
        0.92,
        claim_id="CLM:SCOPED",
        source_paper={"pmid": "1", "year": 2016},
        year=2016,
    )
    global_edges = [
        FrozenEdge(
            "IM:SCOPED",
            "CTX:AD",
            "is_associated_with",
            0.92,
            claim_id="CLM:GLOBAL:1",
            source_paper={"pmid": "1", "year": 2016},
            year=2016,
        ),
        FrozenEdge(
            "D:AD",
            "CTX:AD",
            "is_associated_with",
            0.91,
            claim_id="CLM:GLOBAL:2",
            source_paper={"pmid": "2", "year": 2016},
            year=2016,
        ),
        FrozenEdge(
            "IM:GLOBAL",
            "CTX:SCZ",
            "is_associated_with",
            0.90,
            claim_id="CLM:GLOBAL:3",
            source_paper={"pmid": "3", "year": 2016},
            year=2016,
        ),
        FrozenEdge(
            "D:SCZ",
            "CTX:SCZ",
            "is_associated_with",
            0.89,
            claim_id="CLM:GLOBAL:4",
            source_paper={"pmid": "4", "year": 2016},
            year=2016,
        ),
    ]
    scoped_adjacency = {
        "IM:SCOPED": [scoped_edge],
        "CTX:AD": [scoped_edge],
    }
    global_adjacency: dict[str, list[FrozenEdge]] = {}
    for edge in global_edges:
        global_adjacency.setdefault(edge.source_id, []).append(edge)
        global_adjacency.setdefault(edge.target_id, []).append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names={node_id: row["preferred_name"] for node_id, row in concepts.items()},
        atoms=atoms,
        adjacency={},
        direct_pairs=set(),
    )
    setattr(index, "_global_context_adjacency", global_adjacency)
    setattr(
        index,
        "_catalog_nodes_by_atom",
        {
            "imaging_marker": ("IM:GLOBAL", "IM:SCOPED"),
            "disease": ("D:AD", "D:SCZ"),
        },
    )

    endpoint_pool = _endpoint_evidence_pool(
        index=index,
        adjacency=scoped_adjacency,
        context_adjacency=global_adjacency,
        atoms=(Atom.IMAGING_MARKER,),
        freeze_year=2016,
        limit=8,
    )
    endpoint_scores = {row.node_id: row.score for row in endpoint_pool}
    assert "IM:GLOBAL" in endpoint_scores
    assert endpoint_scores["IM:SCOPED"] > endpoint_scores["IM:GLOBAL"]

    ranked = _ranked_endpoint_compositions(
        method="neurodiscovery",
        case=case_study_by_name("biomarker_discovery"),
        index=index,
        adjacency=scoped_adjacency,
        freeze_year=2016,
        target_count=8,
        seed=11,
    )
    pair_scores = {(row[1], row[2]): row[0] for row in ranked}
    assert pair_scores[("IM:SCOPED", "D:AD")] > pair_scores[("IM:SCOPED", "D:SCZ")]
    assert pair_scores[("IM:GLOBAL", "D:SCZ")] > pair_scores[("IM:GLOBAL", "D:AD")]


def test_endpoint_quality_prior_prefers_canonical_metadata_without_merging_ids() -> None:
    concepts = {
        "CUI:CANONICAL": {
            "preferred_name": "canonical gene",
            "source_vocab": "DisGeNET",
            "aliases": ["GENE1"],
            "external_ids": {"hgnc": "HGNC:1"},
            "domain_tags": ["gene"],
            "metadata": {"atom_types": ["gene_target"]},
        },
        "CLM_CONCEPT:REPLAY": {
            "preferred_name": "replay gene",
            "source_vocab": "replay_anchor_mint",
            "domain_tags": ["gene"],
        },
        "D:CONTEXT": {
            "preferred_name": "context disease",
            "domain_tags": ["disease"],
        },
    }
    canonical_edge = FrozenEdge(
        "CUI:CANONICAL",
        "D:CONTEXT",
        "is_associated_with",
        0.8,
        claim_id="CLM:1",
        source_paper={"pmid": "1", "year": 2016},
        year=2016,
    )
    replay_edge = FrozenEdge(
        "CLM_CONCEPT:REPLAY",
        "D:CONTEXT",
        "is_associated_with",
        0.8,
        claim_id="CLM:2",
        source_paper={"pmid": "2", "year": 2016},
        year=2016,
    )
    adjacency = {
        "CUI:CANONICAL": [canonical_edge],
        "CLM_CONCEPT:REPLAY": [replay_edge],
        "D:CONTEXT": [canonical_edge, replay_edge],
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names={node_id: row["preferred_name"] for node_id, row in concepts.items()},
        atoms={
            "CUI:CANONICAL": frozenset({"gene_target"}),
            "CLM_CONCEPT:REPLAY": frozenset({"gene_target"}),
            "D:CONTEXT": frozenset({"disease"}),
        },
        adjacency={},
        direct_pairs=set(),
    )

    unweighted = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(Atom.GENE_TARGET,),
        freeze_year=2016,
        limit=8,
        include_catalog=False,
    )
    weighted = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(Atom.GENE_TARGET,),
        freeze_year=2016,
        limit=8,
        include_catalog=False,
        canonical_quality_weight=0.20,
    )

    unweighted_scores = {row.node_id: row.score for row in unweighted}
    weighted_scores = {row.node_id: row.score for row in weighted}
    assert unweighted_scores["CUI:CANONICAL"] == pytest.approx(
        unweighted_scores["CLM_CONCEPT:REPLAY"]
    )
    assert weighted_scores["CUI:CANONICAL"] > weighted_scores["CLM_CONCEPT:REPLAY"]
    assert {row.node_id for row in weighted} == {
        "CUI:CANONICAL",
        "CLM_CONCEPT:REPLAY",
    }
    assert weighted[0].canonical_quality > weighted[1].canonical_quality


def test_endpoint_quality_prior_rejects_invalid_weight() -> None:
    index = FrozenGraphIndex({}, {}, {}, {}, set())
    with pytest.raises(ValueError, match="canonical_quality_weight"):
        _endpoint_evidence_pool(
            index=index,
            adjacency={},
            atoms=(Atom.GENE_TARGET,),
            freeze_year=2016,
            limit=1,
            canonical_quality_weight=1.01,
        )


def test_endpoint_pool_keeps_claim_local_identity_with_frozen_evidence() -> None:
    edge = FrozenEdge(
        "CLAIM_ENTITY:LOCAL",
        "D:CONTEXT",
        "is_associated_with",
        0.9,
        claim_id="CLM:1",
        source_paper={"pmid": "1", "year": 2016},
        year=2016,
    )
    index = FrozenGraphIndex(
        concepts={
            "D:CONTEXT": {
                "preferred_name": "context disease",
                "domain_tags": ["disease"],
                "metadata": {"atom_types": ["disease"]},
            },
        },
        names={
            "CLAIM_ENTITY:LOCAL": "local cortical thickness",
            "D:CONTEXT": "context disease",
        },
        atoms={
            "CLAIM_ENTITY:LOCAL": frozenset({"imaging_marker"}),
            "D:CONTEXT": frozenset({"disease"}),
        },
        adjacency={},
        direct_pairs=set(),
    )
    rows = _endpoint_evidence_pool(
        index=index,
        adjacency={"CLAIM_ENTITY:LOCAL": [edge], "D:CONTEXT": [edge]},
        atoms=(Atom.IMAGING_MARKER,),
        freeze_year=2016,
        limit=8,
        include_catalog=False,
    )
    assert [row.node_id for row in rows] == ["CLAIM_ENTITY:LOCAL"]


def test_endpoint_pool_rejects_unindexed_claim_local_even_with_adjacency() -> None:
    edge = FrozenEdge(
        "CLAIM_ENTITY:ORPHAN",
        "D:CONTEXT",
        "is_associated_with",
        0.9,
        claim_id="CLM:1",
    )
    index = FrozenGraphIndex(
        concepts={
            "D:CONTEXT": {
                "preferred_name": "context disease",
                "domain_tags": ["disease"],
            },
        },
        names={"D:CONTEXT": "context disease"},
        atoms={"D:CONTEXT": frozenset({"disease"})},
        adjacency={},
        direct_pairs=set(),
    )

    rows = _endpoint_evidence_pool(
        index=index,
        adjacency={"CLAIM_ENTITY:ORPHAN": [edge], "D:CONTEXT": [edge]},
        atoms=(Atom.IMAGING_MARKER,),
        freeze_year=2016,
        limit=8,
        include_catalog=False,
    )

    assert rows == []


def test_endpoint_pool_excludes_catalog_only_claim_local_identity() -> None:
    index = FrozenGraphIndex(
        concepts={
            "CLAIM_ENTITY:LOCAL": {
                "preferred_name": "local imaging marker",
                "domain_tags": ["imaging_feature"],
                "metadata": {"atom_types": ["imaging_marker"]},
            },
        },
        names={"CLAIM_ENTITY:LOCAL": "local imaging marker"},
        atoms={"CLAIM_ENTITY:LOCAL": frozenset({"imaging_marker"})},
        adjacency={},
        direct_pairs=set(),
    )
    setattr(
        index,
        "_catalog_nodes_by_atom",
        {"imaging_marker": ("CLAIM_ENTITY:LOCAL",)},
    )

    rows = _endpoint_evidence_pool(
        index=index,
        adjacency={},
        atoms=(Atom.IMAGING_MARKER,),
        freeze_year=2016,
        limit=8,
        include_catalog=True,
    )

    assert rows == []


def test_neurodiscovery_endpoint_tail_sampling_is_reproducible_and_seeded() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    ranked = [
        EndpointEvidence(f"E:{index:03d}", 1.0 - index / 200.0, support, ())
        for index in range(100)
    ]
    first = _select_endpoint_evidence_head_tail(
        ranked,
        limit=64,
        tail_fraction=0.25,
        seed=7,
        case_study_id="brain_age",
        pool_label="target",
    )
    repeated = _select_endpoint_evidence_head_tail(
        ranked,
        limit=64,
        tail_fraction=0.25,
        seed=7,
        case_study_id="brain_age",
        pool_label="target",
    )
    alternate = _select_endpoint_evidence_head_tail(
        ranked,
        limit=64,
        tail_fraction=0.25,
        seed=8,
        case_study_id="brain_age",
        pool_label="target",
    )
    assert [row.node_id for row in first[:48]] == [
        row.node_id for row in ranked[:48]
    ]
    assert [row.node_id for row in first] == [row.node_id for row in repeated]
    assert {row.node_id for row in first[48:]} != {
        row.node_id for row in alternate[48:]
    }
    assert any(int(row.node_id.split(":")[1]) >= 64 for row in first)


def test_neurodiscovery_endpoint_tail_sampling_spans_rank_bands() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    ranked = [
        EndpointEvidence(f"E:{index:05d}", 1.0 - index / 50000.0, support, ())
        for index in range(20_000)
    ]

    selections = [
        _select_endpoint_evidence_head_tail(
            ranked,
            limit=64,
            tail_fraction=0.25,
            seed=seed,
            case_study_id="imaging_genetics",
            pool_label="source",
        )
        for seed in range(4)
    ]

    assert all(
        [row.node_id for row in selection[:48]]
        == [row.node_id for row in ranked[:48]]
        for selection in selections
    )
    tail_ranks = {
        int(row.node_id.split(":")[1])
        for selection in selections
        for row in selection[48:]
    }
    assert len(tail_ranks) == 64
    assert min(tail_ranks) > 48
    assert max(tail_ranks) > 18_000


def test_kge_anchor_selection_keeps_head_and_shards_rank_bands() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    ranked = [
        EndpointEvidence(f"E:{index:04d}", 1.0 - index / 2000.0, support, ())
        for index in range(1000)
    ]
    first = _rank_stratified_kge_anchor_ids(
        ranked,
        limit=256,
        case_study_id="imaging_genetics",
        anchor_side="source",
        replicate_index=0,
        generation_round=0,
    )
    repeated = _rank_stratified_kge_anchor_ids(
        ranked,
        limit=256,
        case_study_id="imaging_genetics",
        anchor_side="source",
        replicate_index=0,
        generation_round=0,
    )
    next_round = _rank_stratified_kge_anchor_ids(
        ranked,
        limit=256,
        case_study_id="imaging_genetics",
        anchor_side="source",
        replicate_index=0,
        generation_round=1,
    )

    assert first[:128] == [row.node_id for row in ranked[:128]]
    assert first == repeated
    assert len(first) == len(set(first)) == 256
    assert set(first[128:]) != set(next_round[128:])
    assert max(int(node_id.split(":")[1]) for node_id in first[128:]) >= 990


def test_kge_tail_anchor_selection_reaches_deep_frozen_evidence() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    ranked = [
        EndpointEvidence(f"E:{index:05d}", 1.0 - index / 30000.0, support, ())
        for index in range(20_000)
    ]

    anchors = _rank_stratified_kge_anchor_ids(
        ranked,
        limit=2048,
        case_study_id="imaging_genetics",
        anchor_side="tail_source",
        replicate_index=0,
        generation_round=0,
    )

    assert anchors[:128] == [row.node_id for row in ranked[:128]]
    assert len(anchors) == len(set(anchors)) == 2048
    assert max(int(node_id.split(":")[1]) for node_id in anchors[128:]) > 19_000


def test_kge_anchor_rounds_do_not_repeat_when_rank_bands_have_size_ten() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    ranked = [
        EndpointEvidence(f"E:{index:04d}", 1.0 - index / 2000.0, support, ())
        for index in range(1408)
    ]

    first_band_choices = [
        _rank_stratified_kge_anchor_ids(
            ranked,
            limit=256,
            case_study_id="imaging_genetics",
            anchor_side="source",
            replicate_index=0,
            generation_round=round_index,
        )[128]
        for round_index in range(10)
    ]

    assert len(set(first_band_choices)) == 10


def test_kge_neighbour_rounds_cover_every_member_of_a_size_sixteen_band() -> None:
    rows = [(1.0 - index / 100.0, "SOURCE", f"TARGET:{index:02d}") for index in range(16)]

    first_band_choices = [
        _rank_stratified_kge_rows(
            rows,
            anchor_side="source",
            seed=0,
            case_study_id="imaging_genetics",
            replicate_index=0,
            generation_round=round_index,
        )[4][2]
        for round_index in range(16)
    ]

    assert len(set(first_band_choices)) == 16


def test_neurodiscovery_anchor_fanout_reaches_deep_counterpart() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    sources = [
        EndpointEvidence(f"S:{index}", 1.0 - index / 1000, support, ())
        for index in range(80)
    ]
    targets = [
        EndpointEvidence(f"T:{index}", 1.0 - index / 1000, support, ())
        for index in range(400)
    ]
    pairs = list(
        _neurodiscovery_endpoint_pair_rows(
            sources,
            targets,
            pool_cap=5000,
            seed=11,
            case_study_id="brain_age",
        )
    )
    assert (sources[33], targets[291]) in pairs
    assert (sources[70], targets[20]) in pairs


def test_neurodiscovery_feedback_anchor_fanout_reaches_both_replacement_sides() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    sources = [
        EndpointEvidence(f"S:{index}", 1.0 - index / 1000, support, ())
        for index in range(80)
    ]
    targets = [
        EndpointEvidence(f"T:{index}", 1.0 - index / 1000, support, ())
        for index in range(400)
    ]
    pairs = list(
        _neurodiscovery_endpoint_pair_rows(
            sources,
            targets,
            pool_cap=5000,
            seed=11,
            case_study_id="brain_age",
            priority_source_ids=("S:70",),
            priority_target_ids=("T:291",),
        )
    )
    assert (sources[70], targets[177]) in pairs
    assert (sources[51], targets[291]) in pairs


def test_feedback_endpoint_frontier_reserves_mutations_and_exploration() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)

    def row(score: float, source: str, target: str) -> tuple:
        return (score, source, target, support, support, ())

    selected = _select_endpoint_frontier(
        [
            row(0.9, "A", "C"),
            row(0.8, "D", "B"),
            row(0.7, "X", "Y"),
            row(0.6, "Q", "R"),
        ],
        3,
        feedback_anchor_pairs=(("A", "B"),),
        feedback_anchor_fraction=2 / 3,
    )
    assert len(selected) == 3
    assert sum(
        _is_feedback_endpoint_variant(row, (("A", "B"),))
        for row in selected
    ) == 2
    assert any(row[1:3] == ("X", "Y") for row in selected)


def test_neurodiscovery_endpoint_frontier_preserves_ranked_prefix() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)

    def row(score: float, source: str, target: str) -> tuple:
        return (score, source, target, support, support, ())

    ranked = [
        row(0.9, "A", "W"),
        row(0.8, "A", "X"),
        row(0.7, "A", "Y"),
        row(0.6, "A", "Z"),
        row(0.5, "B", "Q"),
    ]
    selected = _select_endpoint_frontier(
        ranked,
        4,
        preserve_ranked_order=True,
    )

    assert [candidate[1:3] for candidate in selected] == [
        ("A", "W"),
        ("A", "X"),
        ("A", "Y"),
        ("A", "Z"),
    ]


def test_endpoint_pool_preserves_required_feedback_anchor_from_deep_tail() -> None:
    class Concept:
        def __init__(self, name: str) -> None:
            self.preferred_name = name
            self.domain_tags = ["disease"]
            self.metadata = {"atom_types": ["disease"]}

    concepts = {
        f"D:{index}": Concept(f"disease {index}") for index in range(20)
    }
    names = {node_id: concept.preferred_name for node_id, concept in concepts.items()}
    atoms = {node_id: frozenset({"disease"}) for node_id in concepts}
    adjacency = {
        node_id: [
            FrozenEdge(
                node_id,
                "CTX",
                "is_associated_with",
                0.95 - offset / 100,
                year=2016,
            )
        ]
        for offset, node_id in enumerate(concepts)
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names=names,
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )
    pool = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(Atom.DISEASE,),
        freeze_year=2016,
        limit=5,
        include_catalog=False,
        required_node_ids=("D:19",),
    )
    assert pool[0].node_id == "D:19"
    assert len(pool) == 5


def test_endpoint_pool_preserves_required_catalog_only_feedback_anchor() -> None:
    concepts = {
        f"MESH:D{index:07d}": {
            "preferred_name": f"disease {index}",
            "domain_tags": ["disease"],
            "metadata": {"atom_types": ["disease"]},
        }
        for index in range(100)
    }
    names = {
        node_id: str(concept["preferred_name"])
        for node_id, concept in concepts.items()
    }
    atoms = {node_id: frozenset({"disease"}) for node_id in concepts}
    index = FrozenGraphIndex(
        concepts=concepts,
        names=names,
        atoms=atoms,
        adjacency={},
        direct_pairs=set(),
    )
    pool = _endpoint_evidence_pool(
        index=index,
        adjacency={},
        atoms=(Atom.DISEASE,),
        freeze_year=2016,
        limit=5,
        required_node_ids=("MESH:D0000099",),
        seed=5,
    )
    assert pool[0].node_id == "MESH:D0000099"
    assert len(pool) == 5


def test_neurodiscovery_exploration_uses_fixed_rank_bins() -> None:
    assert _endpoint_rank_band(0) == 0
    assert _endpoint_rank_band(15) == 0
    assert _endpoint_rank_band(16) == 1
    assert _endpoint_rank_band(383) == 23


def test_neurodiscovery_exploration_round_robins_strong_endpoints() -> None:
    buckets = [
        (side, anchor_rank, band)
        for side in ("source", "target")
        for anchor_rank in range(3)
        for band in range(5)
    ]
    ordered = _ordered_neurodiscovery_exploration_buckets(
        buckets,
        seed=9,
        case_study_id="brain_age",
    )
    first_sweep = ordered[:6]
    assert len({(side, rank) for side, rank, _ in first_sweep}) == 6
    assert sorted(ordered) == sorted(buckets)


def test_neurodiscovery_stratified_fanout_crosses_opposite_rank_bands() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    buckets = {
        ("target", anchor_rank, band): [(
            0.9 - band / 100.0,
            f"S:{band * 16}",
            f"T:{anchor_rank}",
            band,
            (
                0.9 - band / 100.0,
                f"S:{band * 16}",
                f"T:{anchor_rank}",
                support,
                support,
                (),
            ),
        )]
        for anchor_rank in range(2)
        for band in range(32)
    }

    selected = _stratified_anchor_fanout_rows(
        buckets,
        anchor_side="target",
        max_anchors=2,
        rows_per_anchor=8,
    )

    assert len(selected) == 16
    for target_id in ("T:0", "T:1"):
        source_ranks = [
            int(row[1].split(":")[1])
            for row in selected
            if row[2] == target_id
        ]
        assert min(source_ranks) == 0
        assert max(source_ranks) >= 448


def test_neurodiscovery_executable_prefix_reserves_stratified_fanout() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    exploitation = [
        (1.0 - index / 1000.0, f"S:base:{index}", "T:base", support, support, ())
        for index in range(100)
    ]
    fanout = [
        (0.6 - index / 1000.0, f"S:tail:{index}", "T:anchor", support, support, ())
        for index in range(30)
    ]

    rows = _interleave_neurodiscovery_exploration(
        exploitation,
        [],
        target_count=100,
        pool_cap=130,
        stratified_fanout_rows=fanout,
    )

    prefix_pairs = {(row[1], row[2]) for row in rows[:100]}
    assert len(prefix_pairs & {(row[1], row[2]) for row in fanout}) == 10


def test_neurodiscovery_exploration_deduplicates_endpoint_pairs() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    row_a = (0.9, "S:1", "T:1", support, support, ())
    row_a_duplicate = (0.8, "T:1", "S:1", support, support, ())
    row_b = (0.7, "S:2", "T:2", support, support, ())
    rows = _interleave_neurodiscovery_exploration(
        [row_a, row_b],
        [row_a, row_a_duplicate, row_b],
        target_count=2,
        pool_cap=4,
    )
    assert len(rows) == 2
    assert {
        tuple(sorted((str(row[1]), str(row[2])))) for row in rows
    } == {("S:1", "T:1"), ("S:2", "T:2")}


def test_neurodiscovery_exploration_reserves_bounded_kge_prefix() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    exploitation = [
        (1.0 - index / 100.0, f"S:base:{index}", "T:base", support, support, ())
        for index in range(20)
    ]
    exploration = [
        (0.7, "S:explore", f"T:explore:{index}", support, support, ())
        for index in range(10)
    ]
    kge_rows = [
        (0.95 - index / 100.0, f"S:kge:{index}", f"T:kge:{index}", support, support, ())
        for index in range(10)
    ]

    rows = _interleave_neurodiscovery_exploration(
        exploitation,
        exploration,
        target_count=10,
        pool_cap=20,
        kge_rows=kge_rows,
    )

    kge_pairs = {(row[1], row[2]) for row in kge_rows}
    assert len(rows) == 20
    assert len({(row[1], row[2]) for row in rows[:10]} & kge_pairs) == 1
    assert ("S:kge:0", "T:kge:0") in {(row[1], row[2]) for row in rows[:10]}


def test_neurodiscovery_exploration_prioritizes_directed_endpoint_coverage() -> None:
    support = FrozenEdge("A", "CTX", "is_associated_with", 0.8)
    rows = [
        (0.99, "S:1", "T:1", support, support, ()),
        (0.98, "S:1", "T:2", support, support, ()),
        (0.80, "S:2", "T:3", support, support, ()),
    ]

    selected = _coverage_first_endpoint_rows(rows, 2)

    assert {(row[1], row[2]) for row in selected} == {
        ("S:1", "T:1"),
        ("S:2", "T:3"),
    }


def test_kge_score_is_a_weak_blended_prior() -> None:
    assert _blend_kge_score(0.60, 0.80, 0.15) == pytest.approx(0.63)
    assert _blend_kge_score(0.60, None, 0.15) == pytest.approx(0.60)
    assert _blend_kge_score(0.60, 0.80, 0.0) == pytest.approx(0.60)


def test_kge_retrieval_preserves_high_score_and_stratified_coverage() -> None:
    high = [(1.0 - index / 100.0, "S:head", f"T:{index}") for index in range(20)]
    stratified = [
        (0.50 - index / 100.0, f"S:{index}", f"T:tail:{index}")
        for index in range(20)
    ]

    selected_high, selected_stratified = _kge_retrieval_quota_rows(
        high,
        stratified,
        top_k=10,
    )

    assert len(selected_high) == 8
    assert len(selected_stratified) == 2
    assert selected_high[0] == high[0]
    assert len({row[1] for row in selected_stratified}) == 2


def test_multi_input_exploration_prioritizes_role_specific_coverage() -> None:
    support = FrozenEdge("A", "D:1", "predicts", 0.8)

    def candidate(
        score: float,
        imaging: str,
        individual: str,
        output: str,
    ) -> MultiInputCandidate:
        return MultiInputCandidate(
            score=score,
            output_id=output,
            input_bindings=(
                ("imaging_marker", imaging),
                ("individual_data", individual),
            ),
            edges=(support, support),
            paper_keys=(),
        )

    ranked = [
        candidate(0.99, "IM:1", "IDV:1", "D:1"),
        candidate(0.98, "IM:1", "IDV:2", "D:1"),
        candidate(0.80, "IM:2", "IDV:3", "D:2"),
    ]

    selected = _diverse_multi_input_top(
        ranked,
        2,
        coverage_fraction=1.0,
    )

    assert {candidate.output_id for candidate in selected} == {"D:1", "D:2"}
    assert {
        node_id
        for candidate in selected
        for _, node_id in candidate.input_bindings
    } == {"IM:1", "IM:2", "IDV:1", "IDV:3"}


def test_candidate_quality_prior_reranks_connected_candidates_without_future_data() -> None:
    class Node:
        def __init__(
            self,
            name: str,
            *,
            source_vocab: str,
            aliases: list[str] | None = None,
            external_ids: dict[str, str] | None = None,
        ) -> None:
            self.preferred_name = name
            self.source_vocab = source_vocab
            self.aliases = aliases or []
            self.external_ids = external_ids or {}
            self.domain_tags = ["gene"]
            self.metadata = {"atom_types": ["gene_target"]}

    engine = object.__new__(HypothesisEngine)
    engine._index = {
        "CUI:CANONICAL": Node(
            "canonical gene",
            source_vocab="DisGeNET",
            aliases=["GENE1"],
            external_ids={"hgnc": "HGNC:1"},
        ),
        "CLM_CONCEPT:REPLAY": Node(
            "replay gene",
            source_vocab="replay_anchor_mint",
        ),
    }
    payload = {
        "hypotheses": [
            {
                "id": "H:REPLAY",
                "source_id": "CLM_CONCEPT:REPLAY",
                "target_id": "CLM_CONCEPT:REPLAY",
                "composite_score": 0.80,
                "metadata": {"path_node_ids": ["CLM_CONCEPT:REPLAY"]},
            },
            {
                "id": "H:CANONICAL",
                "source_id": "CUI:CANONICAL",
                "target_id": "CUI:CANONICAL",
                "composite_score": 0.79,
                "metadata": {"path_node_ids": ["CUI:CANONICAL"]},
            },
        ]
    }

    adjusted = _apply_candidate_canonical_quality(
        engine._index,
        payload,
        weight=0.20,
    )

    assert [row["id"] for row in adjusted["hypotheses"]] == [
        "H:CANONICAL",
        "H:REPLAY",
    ]
    audit = adjusted["hypotheses"][0]["metadata"]["endpoint_canonical_quality"]
    assert audit["uses_future_outcomes"] is False
    assert audit["original_composite_score"] == pytest.approx(0.79)
    assert adjusted["metadata"]["endpoint_canonical_quality"]["policy"] == (
        ENDPOINT_CANONICAL_QUALITY_POLICY
    )


def test_candidate_quality_prior_zero_weight_preserves_order() -> None:
    engine = object.__new__(HypothesisEngine)
    engine._index = {}
    payload = {
        "hypotheses": [
            {"id": "H:1", "composite_score": 0.1},
            {"id": "H:2", "composite_score": 0.9},
        ]
    }

    unchanged = _apply_candidate_canonical_quality(
        engine._index,
        payload,
        weight=0.0,
    )

    assert [row["id"] for row in unchanged["hypotheses"]] == ["H:1", "H:2"]
    assert unchanged["metadata"]["endpoint_canonical_quality"]["policy"] == "disabled"


def test_endpoint_quality_screen_selects_exact_development_matrix() -> None:
    windows = (
        parse_window("2016:2017:2017"),
        parse_window("2017:2018:2018"),
    )
    manifest = {
        "runs": [
            {
                "method": "neurodiscovery",
                "seed": seed,
                "case_study_id": case_id,
                "freeze_year": window.freeze_year,
                "hypotheses_path": "candidate.json",
            }
            for window in windows
            for case_id in ("case1_transdiagnostic", "biomarker_discovery")
            for seed in (0, 1)
        ]
    }

    selected = _select_source_runs(
        manifest,
        case_ids=("case1_transdiagnostic", "biomarker_discovery"),
        windows=windows,
        seeds=(0, 1),
    )

    assert len(selected) == 8
    assert _method_label(0.10) == "neurodiscovery_quality_w010"


def test_endpoint_quality_screen_collects_all_formal_candidate_nodes() -> None:
    payload = {
        "hypotheses": [
            {
                "source_id": "S",
                "target_id": "T",
                "metadata": {
                    "path_node_ids": ["S", "M", "T"],
                    "input_entity_ids": ["I1", "I2"],
                    "mediator_ids": ["M2"],
                },
            }
        ]
    }

    assert _candidate_node_ids(payload) == {"S", "T", "M", "I1", "I2", "M2"}


def test_endpoint_quality_accepts_in_memory_concept_nodes() -> None:
    class ConceptNode:
        preferred_name = "APOE"
        source_vocab = "HGNC"
        aliases = ["APOE4"]
        external_ids = {"hgnc": "613"}
        domain_tags = ["gene"]
        metadata = {"atom_types": ["gene_target"]}

    index = FrozenGraphIndex(
        concepts={"HGNC:613": ConceptNode()},
        names={"HGNC:613": "APOE"},
        atoms={"HGNC:613": frozenset({"gene_target"})},
        adjacency={},
        direct_pairs=set(),
    )

    assert _endpoint_canonical_quality(index, "HGNC:613") > 0.5


def test_neurodiscovery_global_context_is_built_only_from_loaded_snapshot() -> None:
    class Node:
        def __init__(
            self,
            name: str,
            domains: list[str],
            metadata: dict | None = None,
        ) -> None:
            self.preferred_name = name
            self.domain_tags = domains
            self.metadata = metadata or {}

    claim_meta = {
        "subject_id": "IM:HIST",
        "subject_name": "hippocampal volume",
        "subject_type": "imaging_marker",
        "predicate": "is_biomarker_of",
        "object_id": "D:HIST",
        "object_name": "Alzheimer disease",
        "object_type": "disease",
        "confidence": 0.9,
        "source_paper": {"pmid": "10", "year": 2016},
        "claim_case_study_ids": ["biomarker_discovery"],
    }
    index = {
        "IM:HIST": Node("hippocampal volume", ["imaging_feature"]),
        "D:HIST": Node("Alzheimer disease", ["disease"]),
        "CLM:HIST": Node("claim", ["claim"], claim_meta),
    }
    engine = object.__new__(HypothesisEngine)
    engine._index = index
    engine._semantic_scope_claim_cache = {}
    engine._semantic_historical_pairs = None
    engine.kg = type("KG", (), {})()
    engine.G = type(
        "Graph",
        (),
        {"edges": lambda self, data=True: []},
    )()

    frontier_index, scoped, scoped_count = _frontier_index_for_case(
        engine,
        case_study_by_name("biomarker_discovery"),
    )
    context = getattr(frontier_index, "_global_context_adjacency")

    assert scoped_count == 1
    assert "IM:HIST" in scoped
    assert "IM:HIST" in context
    assert "D:FUTURE" not in context
    assert ("D:HIST", "IM:HIST") in frontier_index.direct_pairs
    assert getattr(frontier_index, "_global_context_policy") == (
        "frozen_snapshot_semantic_context.v1"
    )
    audit = getattr(frontier_index, "_global_context_audit")
    assert audit["claim_context_edges"] == 1


def test_neurodiscovery_tag_preserves_top_level_audit_metadata(tmp_path: Path) -> None:
    output = tmp_path / "hypotheses.json"
    output.write_text(
        json.dumps(
            {
                "n_hypotheses": 1,
                "hypotheses": [
                    {
                        "id": "H:1",
                        "source_id": "G:1",
                        "target_id": "IM:1",
                        "metadata": {"generation_mode": "frontier"},
                    }
                ],
                "metadata": {
                    "neurodiscovery_evidence_frontier": {
                        "context_policy": "frozen_snapshot_semantic_context.v1"
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    _tag(
        object.__new__(HypothesisEngine),
        output,
        case_study_by_name("imaging_genetics"),
    )
    tagged = json.loads(output.read_text(encoding="utf-8"))

    assert tagged["metadata"]["neurodiscovery_evidence_frontier"][
        "context_policy"
    ] == "frozen_snapshot_semantic_context.v1"
    assert tagged["hypotheses"][0]["metadata"]["case_study_id"] == (
        "imaging_genetics"
    )


def test_neurodiscovery_payload_records_endpoint_quality_policy() -> None:
    concepts = {
        "G:1": {
            "preferred_name": "gene one",
            "domain_tags": ["gene"],
        },
        "IM:1": {
            "preferred_name": "cortical thickness",
            "domain_tags": ["imaging_feature"],
        },
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names={node_id: row["preferred_name"] for node_id, row in concepts.items()},
        atoms={
            "G:1": frozenset({"gene_target"}),
            "IM:1": frozenset({"imaging_marker"}),
        },
        adjacency={},
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="neurodiscovery",
        case=case_study_by_name("imaging_genetics"),
        index=index,
        adjacency={},
        freeze_year=2016,
        target_count=1,
        seed=0,
        evidence_frontier_fraction=1.0,
    )

    assert payload["metadata"]["endpoint_canonical_quality_policy"] == (
        ENDPOINT_CANONICAL_QUALITY_POLICY
    )
    assert payload["metadata"]["endpoint_canonical_quality_weight"] == pytest.approx(
        0.20
    )


def test_proposed_chain_link_round_trips_through_formal_hypothesis_schema() -> None:
    from neurooracle.scripts.generate_case_study_frozen_baselines import (
        _proposed_link,
    )

    index = FrozenGraphIndex(
        concepts={},
        names={"G:A": "gene A", "IM:B": "cortical thickness"},
        atoms={},
        adjacency={},
        direct_pairs=set(),
    )
    link = _proposed_link(index, "G:A", "IM:B", "associated_with", 0.7)
    hypothesis = Hypothesis.from_dict(
        {
            "id": "H:CHAIN",
            "hypothesis_type": "evidence_frontier_chain",
            "source_id": "G:A",
            "source_name": "gene A",
            "target_id": "IM:B",
            "target_name": "cortical thickness",
            "path": [link],
        }
    )

    assert hypothesis.path[0].evidence["status"] == "proposed_relation"
    assert "metadata" not in link


def test_matrix_command_uses_formal_case_study_id_without_case3_task_flags(
    tmp_path: Path,
) -> None:
    command = command_for_case_study(
        case=case_study_by_name("brain_age"),
        input_dir=tmp_path / "input",
        snapshot_root=tmp_path / "snapshots",
        output_root=tmp_path / "outputs",
        windows=[],
        target_per_case_study=100,
        random_trials=10,
        seed=3,
        generate_only=True,
        force_generation=False,
        force_evaluation=False,
    )
    assert "brain_age" in command
    assert "--tasks" not in command
    assert "--chains" not in command
    assert command[command.index("--target-per-case-study") + 1] == "100"


def test_future_evaluation_filters_by_claim_case_study_membership(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "CLM:future",
            "year": 2021,
            "case_study_ids": frozenset({"brain_age", "prognosis"}),
            "subject_id": "IM:1",
            "subject_name": "cortical thickness",
            "subject_type": "IMAGING_MARKER",
            "object_id": "D:1",
            "object_name": "predicted brain age",
            "object_type": "INDIVIDUAL_DATA",
            "predicate": "predicts",
            "raw_text": "future support",
        },
        {
            "id": "CLM:negated-future",
            "year": 2021,
            "case_study_ids": frozenset({"brain_age"}),
            "subject_id": "IM:1",
            "subject_name": "cortical thickness",
            "subject_type": "IMAGING_MARKER",
            "object_id": "D:1",
            "object_name": "predicted brain age",
            "object_type": "INDIVIDUAL_DATA",
            "predicate": "predicts",
            "raw_text": "no future support",
            "negated": True,
        },
    ]
    concepts = {
        "IM:1": {
            "preferred_name": "cortical thickness",
            "domain_tags": ["imaging_feature"],
        },
        "D:1": {
            "preferred_name": "predicted brain age",
            "domain_tags": ["dataset_variable"],
        },
    }
    future, stats = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        set(),
        2021,
        2025,
        case_study_id="brain_age",
        future_records=records,
    )
    assert stats["future_unique_pairs"] == 1
    assert stats["negated_claims_excluded"] == 1
    assert ("D:1", "IM:1") in future["novel_pair_year"]

    _, wrong = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        set(),
        2021,
        2025,
        case_study_id="drug_repurposing",
        future_records=records,
    )
    assert wrong["future_unique_pairs"] == 0


def test_future_claim_record_loader_filters_tasks_before_caching(
    tmp_path: Path,
) -> None:
    claims = [
        {
            "id": "CLM:brain-age",
            "year": 2021,
            "claim_case_study_ids": ["brain_age"],
            "subject_id": "IM:1",
            "subject_name": "brain age gap",
            "object_id": "OUT:1",
            "object_name": "chronological age",
            "predicate": "predicts",
            "metadata": {
                "subject_type": "IMAGING_MARKER",
                "object_type": "AGE_VARIABLE",
            },
        },
        {
            "id": "CLM:prognosis",
            "year": 2022,
            "claim_case_study_ids": ["prognosis"],
            "subject_id": "IM:2",
            "object_id": "OUT:2",
            "predicate": "predicts",
        },
        {
            "id": "CLM:outside-window",
            "year": 2026,
            "claim_case_study_ids": ["brain_age"],
            "subject_id": "IM:3",
            "object_id": "OUT:3",
            "predicate": "predicts",
        },
        {
            "id": "CLM:negated",
            "year": 2021,
            "claim_case_study_ids": ["brain_age"],
            "subject_id": "IM:4",
            "object_id": "OUT:4",
            "predicate": "predicts",
            "negated": True,
        },
    ]
    path = tmp_path / "claims.jsonl"
    path.write_text(
        "".join(json.dumps(claim) + "\n" for claim in claims),
        encoding="utf-8",
    )

    records = load_future_claim_records(
        path,
        min_year=2021,
        max_year=2025,
        case_study_ids={"brain_age"},
    )

    assert [record["id"] for record in records] == ["CLM:brain-age"]
    assert records[0]["case_study_ids"] == frozenset({"brain_age"})
    assert records[0]["subject_type"] == "IMAGING_MARKER"
    assert records[0]["object_type"] == "AGE_VARIABLE"


def test_functional_localization_future_index_enforces_relation_contract(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "CLM:localization",
            "year": 2021,
            "case_study_ids": frozenset({"functional_localization"}),
            "subject_id": "TASK:WM",
            "subject_name": "working memory task",
            "subject_type": "COGNITIVE_TASK",
            "object_id": "REGION:DLPFC",
            "object_name": "dorsolateral prefrontal cortex",
            "object_type": "BRAIN_REGION",
            "predicate": "activates",
            "raw_text": "Working memory activates dorsolateral prefrontal cortex.",
        },
        {
            "id": "CLM:wrong-relation",
            "year": 2021,
            "case_study_ids": frozenset({"functional_localization"}),
            "subject_id": "IM:THICKNESS",
            "subject_name": "cortical thickness",
            "subject_type": "IMAGING_MARKER",
            "object_id": "DISEASE:SCZ",
            "object_name": "schizophrenia",
            "object_type": "DISEASE",
            "predicate": "distinguishes",
            "raw_text": "Cortical thickness distinguishes schizophrenia.",
        },
    ]
    concepts = {
        "TASK:WM": {
            "preferred_name": "working memory task",
            "domain_tags": ["cognitive_function"],
        },
        "REGION:DLPFC": {
            "preferred_name": "dorsolateral prefrontal cortex",
            "domain_tags": ["neuroanatomy"],
        },
        "IM:THICKNESS": {
            "preferred_name": "cortical thickness",
            "domain_tags": ["imaging_feature"],
        },
        "DISEASE:SCZ": {
            "preferred_name": "schizophrenia",
            "domain_tags": ["disease"],
        },
    }

    future, stats = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        set(),
        2021,
        2025,
        case_study_id="functional_localization",
        future_records=records,
    )

    assert stats["future_claims_total"] == 2
    assert stats["case_study_relation_contract_rejected"] == 1
    assert stats["future_unique_pairs"] == 1
    assert ("REGION:DLPFC", "TASK:WM") in future["novel_pair_year"]


def test_future_index_excludes_endpoints_unavailable_at_freeze(tmp_path: Path) -> None:
    records = [
        {
            "id": "CLM:future-only-endpoint",
            "year": 2021,
            "case_study_ids": frozenset({"brain_age"}),
            "subject_id": "IM:FUTURE",
            "subject_name": "future-only cortical thickness",
            "subject_type": "IMAGING_MARKER",
            "object_id": "OUTCOME:AGE",
            "object_name": "chronological age",
            "object_type": "OUTCOME",
            "predicate": "predicts",
            "raw_text": "Future-only cortical thickness predicts chronological age.",
        }
    ]
    concepts = {
        "OUTCOME:AGE": {
            "preferred_name": "chronological age",
            "domain_tags": ["dataset_variable"],
        },
    }

    future, stats = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        set(),
        2021,
        2025,
        case_study_id="brain_age",
        future_records=records,
    )

    assert stats["future_claims_unavailable_endpoint"] == 1
    assert stats["future_claims_unavailable_subject_endpoint"] == 1
    assert stats["future_unique_pairs"] == 0
    assert not future["novel_pair_year"]


def test_future_index_recovers_claim_local_endpoint_seen_before_freeze(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "CLM:future-existing-local-endpoint",
            "year": 2021,
            "case_study_ids": frozenset({"imaging_genetics"}),
            "subject_id": "GENE:KL",
            "subject_name": "KL",
            "subject_type": "GENE_TARGET",
            "object_id": "CUI:BROAD-BRAIN",
            "object_name": "total brain volume",
            "object_type": "IMAGING_MARKER",
            "predicate": "correlates_with",
            "raw_text": "KL correlates with total brain volume.",
        }
    ]
    concepts = {
        "GENE:KL": {
            "preferred_name": "KL",
            "domain_tags": ["gene"],
        },
        "CUI:BROAD-BRAIN": {
            "preferred_name": "Brain",
            "domain_tags": ["neuroanatomy"],
        },
    }
    local_id = "CLAIM_ENTITY:" + hashlib.sha1(
        "total brain volume".encode("utf-8")
    ).hexdigest()[:20]
    unrelated_historical_pair = (local_id, "IM:HISTORICAL-CONTEXT")

    future, stats = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        {unrelated_historical_pair},
        2021,
        2025,
        case_study_id="imaging_genetics",
        future_records=records,
        historical_endpoint_atoms={
            "GENE:KL": {"gene_target"},
            local_id: {"imaging_marker"},
            "IM:HISTORICAL-CONTEXT": {"imaging_marker"},
        },
    )

    assert stats["claim_local_objects"] == 1
    assert stats["frozen_claim_local_objects_recovered"] == 1
    assert stats.get("future_claims_unavailable_endpoint", 0) == 0
    assert stats["future_unique_pairs"] == 1
    assert (local_id, "GENE:KL") in future["novel_pair_year"]


def test_future_index_rejects_role_available_only_in_future_claim(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "CLM:future-retyped-endpoint",
            "year": 2021,
            "case_study_ids": frozenset({"imaging_genetics"}),
            "subject_id": "CLM_CONCEPT:PRS",
            "subject_name": "schizophrenia polygenic risk score",
            "subject_type": "GENETIC_MARKER",
            "object_id": "IM:THICKNESS",
            "object_name": "cortical thickness",
            "object_type": "IMAGING_MARKER",
            "predicate": "correlates_with",
            "raw_text": "Schizophrenia polygenic risk correlates with cortical thickness.",
        }
    ]
    concepts = {
        "IM:THICKNESS": {
            "preferred_name": "cortical thickness",
            "domain_tags": ["imaging_feature"],
        },
    }
    prs_local_id = "CLAIM_ENTITY:" + hashlib.sha1(
        "schizophrenia polygenic risk score".encode("utf-8")
    ).hexdigest()[:20]

    future, stats = _future_indexes(
        tmp_path / "unused.jsonl",
        concepts,
        set(),
        2021,
        2025,
        case_study_id="imaging_genetics",
        future_records=records,
        historical_endpoint_atoms={
            prs_local_id: {"individual_data"},
            "IM:THICKNESS": {"imaging_marker"},
        },
    )

    assert stats["future_claims_frozen_atom_contract_rejected"] == 1
    assert stats["future_unique_pairs"] == 0
    assert not future["novel_pair_year"]


def test_benchmark_status_distinguishes_empty_sparse_and_executable() -> None:
    assert _benchmark_status(100, 0)[0] == "non_executable"
    assert _benchmark_status(100, 5)[0] == "sparse"
    assert _benchmark_status(100, 10) == ("executable", "")
    assert _benchmark_status(0, 10)[0] == "non_executable"


def test_historical_pairs_replace_claim_backed_canonical_collisions(
    tmp_path: Path,
) -> None:
    claims_path = tmp_path / "extracted_claims.jsonl"
    claims_path.write_text(
        json.dumps({
            "id": "CLM:collision",
            "subject_id": "TASK:FEAR",
            "subject_name": "diminished EEG response amplitude",
            "subject_type": "ELECTROPHYSIOLOGY_MARKER",
            "object_id": "D:PSYCHOSIS",
            "object_name": "negative symptoms across psychosis probands",
            "object_type": "OUTCOME",
        }) + "\n",
        encoding="utf-8",
    )
    concepts = {
        "TASK:FEAR": {
            "preferred_name": "Fear",
            "domain_tags": ["emotion"],
        },
        "D:PSYCHOSIS": {
            "preferred_name": "Psychosis",
            "domain_tags": ["disease"],
        },
        "CURATED:A": {"preferred_name": "A", "domain_tags": ["gene"]},
        "CURATED:B": {"preferred_name": "B", "domain_tags": ["disease"]},
    }
    edges = [
        {
            "source_id": "TASK:FEAR",
            "target_id": "D:PSYCHOSIS",
            "relation_type": "predicts",
            "source": "claim:collision",
            "metadata": {"claim_id": "CLM:collision"},
        },
        {
            "source_id": "CURATED:A",
            "target_id": "CURATED:B",
            "relation_type": "associated_with",
            "source": "curated_db",
            "metadata": {},
        },
    ]

    pairs = _historical_pairs(
        edges,
        claims_path=claims_path,
        concepts=concepts,
    )

    assert ("CURATED:A", "CURATED:B") in pairs
    assert ("D:PSYCHOSIS", "TASK:FEAR") not in pairs
    assert any(any(node.startswith("CLAIM_ENTITY:") for node in pair) for pair in pairs)


def test_claim_scope_does_not_borrow_the_parent_paper_scope() -> None:
    claim = {
        "paper_case_study_ids": ["brain_age"],
        "claim_case_study_ids": ["prognosis"],
    }

    assert _claim_case_study_ids(claim) == frozenset({"prognosis"})


def test_case2_partial_path_is_diagnostic_but_not_a_primary_hit() -> None:
    hypothesis = {
        "source_id": "GENE:1",
        "target_id": "OUTCOME:1",
        "path": [
            {"from_id": "GENE:1", "to_id": "IM:1"},
            {"from_id": "IM:1", "to_id": "OUTCOME:1"},
        ],
    }
    partial = {
        "novel_pair_year": {},
        "all_pair_year": {("GENE:1", "IM:1"): 2021},
        "pair_claims": {},
    }
    complete = {
        "novel_pair_year": {},
        "all_pair_year": {
            ("GENE:1", "IM:1"): 2021,
            ("IM:1", "OUTCOME:1"): 2022,
        },
        "pair_claims": {},
    }

    partial_score = _score_hypothesis(
        hypothesis,
        partial,
        2020,
        case_study_id="case2_pathway_mediation",
    )
    complete_score = _score_hypothesis(
        hypothesis,
        complete,
        2020,
        case_study_id="case2_pathway_mediation",
    )

    assert partial_score["any_path_edge_hit"] is True
    assert partial_score["primary_hit"] is False
    assert complete_score["all_path_edges_hit"] is True
    assert complete_score["primary_hit"] is True
    assert complete_score["primary_year"] == 2022


def test_multi_input_evidence_graph_requires_every_branch_for_primary_hit() -> None:
    hypothesis = {
        "source_id": "DRUG:1",
        "target_id": "OUTCOME:1",
        "path": [
            {"from_id": "DRUG:1", "to_id": "OUTCOME:1"},
            {"from_id": "DISEASE:1", "to_id": "OUTCOME:1"},
            {"from_id": "IM:1", "to_id": "OUTCOME:1"},
        ],
    }
    partial = {
        "novel_pair_year": {},
        "all_pair_year": {
            ("DRUG:1", "OUTCOME:1"): 2021,
            ("DISEASE:1", "OUTCOME:1"): 2022,
        },
        "pair_claims": {},
    }
    complete = {
        "novel_pair_year": {},
        "all_pair_year": {
            ("DRUG:1", "OUTCOME:1"): 2021,
            ("DISEASE:1", "OUTCOME:1"): 2022,
            ("IM:1", "OUTCOME:1"): 2023,
        },
        "pair_claims": {},
    }

    partial_score = _score_hypothesis(
        hypothesis,
        partial,
        2020,
        case_study_id="drug_response_prediction",
    )
    complete_score = _score_hypothesis(
        hypothesis,
        complete,
        2020,
        case_study_id="drug_response_prediction",
    )

    assert partial_score["any_path_edge_hit"] is True
    assert partial_score["primary_hit"] is False
    assert complete_score["all_path_edges_hit"] is True
    assert complete_score["primary_hit"] is True
    assert complete_score["primary_year"] == 2023


def test_frozen_baseline_preserves_case2_mediation_atom_order() -> None:
    source, target, mediators, signature = _atom_route(
        case_study_by_name("case2_pathway_mediation")
    )

    assert source == {"gene_target"}
    assert target == "outcome"
    assert mediators == {"imaging_marker"}
    assert signature == "G->IM->O[longitudinal]"


def test_executability_component_requires_distinct_nodes_for_each_atom() -> None:
    concepts = {
        "CUI:D_IM": {
            "preferred_name": "schizophrenia cortical thickness",
            "domain_tags": ["disease", "imaging_feature"],
        },
        "IM:HIPPO": {
            "preferred_name": "hippocampal volume",
            "domain_tags": ["imaging_feature"],
        },
        "OUT:RELAPSE": {
            "preferred_name": "clinical relapse",
            "domain_tags": ["treatment_outcome"],
        },
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names={key: value["preferred_name"] for key, value in concepts.items()},
        atoms={
            "CUI:D_IM": frozenset({"disease", "imaging_marker"}),
            "IM:HIPPO": frozenset({"imaging_marker"}),
            "OUT:RELAPSE": frozenset({"outcome"}),
        },
        adjacency={},
        direct_pairs=set(),
    )
    first = FrozenEdge(
        "CUI:D_IM",
        "OUT:RELAPSE",
        "predicts",
        0.8,
        claim_id="CLM:1",
        source_paper={"pmid": "1"},
    )
    required = (Atom.DISEASE, Atom.IMAGING_MARKER, Atom.OUTCOME)

    collision_only = component_summary(index, [first], required)
    assert collision_only["complete_components"] == 0
    assert collision_only["max_covered_roles"] == 2

    second = FrozenEdge(
        "IM:HIPPO",
        "OUT:RELAPSE",
        "predicts",
        0.8,
        claim_id="CLM:2",
        source_paper={"pmid": "2"},
    )
    complete = component_summary(index, [first, second], required)
    assert complete["complete_components"] == 1
    assert complete["complete_cross_paper_components"] == 1


def test_frozen_baseline_uses_registered_task_atoms() -> None:
    source, target, mediators, signature = _atom_route(
        case_study_by_name("imaging_genetics")
    )

    assert source == {"gene_target"}
    assert target == "imaging_marker"
    assert mediators is None
    assert signature == "{G}->IM"


def test_frozen_generation_routes_follow_the_shared_relation_contract() -> None:
    assert _generation_endpoint_atom_routes(
        case_study_by_name("progression_prediction")
    ) == (
        (Atom.IMAGING_MARKER, Atom.DISEASE),
        (Atom.IMAGING_MARKER, Atom.OUTCOME),
    )
    assert _generation_endpoint_atom_routes(
        case_study_by_name("connectome_behavior")
    ) == (
        (Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA),
        (Atom.IMAGING_MARKER, Atom.OUTCOME),
    )


def test_frozen_baseline_multi_input_graph_covers_distinct_registered_roles() -> None:
    concepts = {
        "MONDO:0002009": {
            "preferred_name": "major depressive disorder",
            "domain_tags": ["disease"],
        },
        "CHEBI:5118": {
            "preferred_name": "fluoxetine",
            "domain_tags": ["drug"],
        },
        "IM:AMYGDALA_FC": {
            "preferred_name": "amygdala functional connectivity",
            "domain_tags": ["imaging_feature", "connectivity"],
        },
        "OUT:REMISSION": {
            "preferred_name": "treatment response and remission",
            "domain_tags": ["treatment_outcome"],
        },
    }
    names = {key: value["preferred_name"] for key, value in concepts.items()}
    atoms = {
        "MONDO:0002009": frozenset({"disease"}),
        "CHEBI:5118": frozenset({"drug"}),
        "IM:AMYGDALA_FC": frozenset({"imaging_marker"}),
        "OUT:REMISSION": frozenset({"outcome"}),
    }
    edges = [
        FrozenEdge(
            "MONDO:0002009", "OUT:REMISSION", "predicts", 0.80,
            claim_id="CLM:D", source_paper={"pmid": "1"}, year=2018,
        ),
        FrozenEdge(
            "CHEBI:5118", "OUT:REMISSION", "predicts", 0.85,
            claim_id="CLM:RX", source_paper={"pmid": "2"}, year=2018,
        ),
        FrozenEdge(
            "IM:AMYGDALA_FC", "OUT:REMISSION", "predicts", 0.90,
            claim_id="CLM:IM", source_paper={"pmid": "3"}, year=2018,
        ),
    ]
    adjacency = {node_id: [] for node_id in concepts}
    for edge in edges:
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names=names,
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="openscholar_rag",
        case=case_study_by_name("drug_response_prediction"),
        index=index,
        adjacency=adjacency,
        freeze_year=2018,
        target_count=2,
        seed=7,
    )

    assert payload["metadata"]["valid"] == 1
    hypothesis = payload["hypotheses"][0]
    assert hypothesis["hypothesis_type"] == "multi_input_evidence_graph"
    assert hypothesis["metadata"]["input_atom_order"] == [
        "disease", "drug", "imaging_marker",
    ]
    assert set(hypothesis["metadata"]["input_entity_ids"]) == {
        "MONDO:0002009", "CHEBI:5118", "IM:AMYGDALA_FC",
    }
    assert {
        frozenset((edge["from_id"], edge["to_id"]))
        for edge in hypothesis["path"]
    } == {
        frozenset(("MONDO:0002009", "OUT:REMISSION")),
        frozenset(("CHEBI:5118", "OUT:REMISSION")),
        frozenset(("IM:AMYGDALA_FC", "OUT:REMISSION")),
    }
    assert hypothesis["metadata"]["cross_paper"] is True
    assert len(hypothesis["metadata"]["source_paper_keys"]) == 3
    assert payload["hypotheses"][1]["hypothesis_type"] == "generation_failure"


def test_frozen_baseline_multi_input_graph_rejects_role_collision() -> None:
    concepts = {
        "CUI:D_IM": {
            "preferred_name": "schizophrenia cortical thickness",
            "domain_tags": ["disease", "imaging_feature"],
        },
        "CHEBI:5118": {
            "preferred_name": "fluoxetine",
            "domain_tags": ["drug"],
        },
        "OUT:REMISSION": {
            "preferred_name": "treatment response and remission",
            "domain_tags": ["treatment_outcome"],
        },
    }
    names = {key: value["preferred_name"] for key, value in concepts.items()}
    atoms = {
        "CUI:D_IM": frozenset({"disease", "imaging_marker"}),
        "CHEBI:5118": frozenset({"drug"}),
        "OUT:REMISSION": frozenset({"outcome"}),
    }
    edges = [
        FrozenEdge(
            "CUI:D_IM", "OUT:REMISSION", "predicts", 0.85,
            claim_id="CLM:1", source_paper={"pmid": "1"}, year=2018,
        ),
        FrozenEdge(
            "CHEBI:5118", "OUT:REMISSION", "predicts", 0.85,
            claim_id="CLM:2", source_paper={"pmid": "2"}, year=2018,
        ),
    ]
    adjacency = {node_id: [] for node_id in concepts}
    for edge in edges:
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names=names,
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="openscholar_rag",
        case=case_study_by_name("drug_response_prediction"),
        index=index,
        adjacency=adjacency,
        freeze_year=2018,
        target_count=1,
        seed=7,
    )

    assert payload["metadata"]["valid"] == 0
    assert payload["hypotheses"][0]["hypothesis_type"] == "generation_failure"


def test_frozen_baseline_frontier_fills_single_input_budget() -> None:
    concepts = {
        "IM:1": {"preferred_name": "hippocampal volume", "domain_tags": ["imaging_feature"]},
        "IM:2": {"preferred_name": "cortical thickness", "domain_tags": ["imaging_feature"]},
        "D:1": {"preferred_name": "Alzheimer disease", "domain_tags": ["disease"]},
        "D:2": {"preferred_name": "Parkinson disease", "domain_tags": ["disease"]},
    }
    atoms = {
        "IM:1": frozenset({"imaging_marker"}),
        "IM:2": frozenset({"imaging_marker"}),
        "D:1": frozenset({"disease"}),
        "D:2": frozenset({"disease"}),
    }
    support_edges = [
        FrozenEdge("IM:1", "IM:2", "associated_with", 0.8, source_paper={"pmid": "1"}),
        FrozenEdge("D:1", "D:2", "associated_with", 0.8, source_paper={"pmid": "2"}),
    ]
    adjacency = {node_id: [] for node_id in concepts}
    for edge in support_edges:
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names={key: value["preferred_name"] for key, value in concepts.items()},
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="openscholar_rag",
        case=case_study_by_name("biomarker_discovery"),
        index=index,
        adjacency=adjacency,
        freeze_year=2020,
        target_count=3,
        seed=3,
    )

    assert payload["metadata"]["valid"] == 3
    assert all(
        hypothesis["hypothesis_type"] == "retrieval_composed_pair"
        for hypothesis in payload["hypotheses"]
    )
    assert all(not hypothesis["path"] for hypothesis in payload["hypotheses"])


def test_evidence_frontier_fraction_reserves_and_spreads_requested_quota() -> None:
    connected = [
        {
            "id": f"CONNECTED:{index}",
            "hypothesis_type": "bridge",
            "metadata": {"generation_mode": "connected_evidence_bridge"},
        }
        for index in range(6)
    ]
    frontier = [
        {
            "id": f"FRONTIER:{index}",
            "hypothesis_type": "retrieval_composed_pair",
            "metadata": {"generation_mode": "evidence_frontier_composition"},
        }
        for index in range(4)
    ]

    mixed = _interleave_evidence_frontier(
        [*connected, *frontier],
        target_count=8,
        evidence_frontier_fraction=0.5,
    )

    assert len(mixed) == 8
    assert sum(row["id"].startswith("FRONTIER") for row in mixed) == 4
    assert any(row["id"].startswith("FRONTIER") for row in mixed[:4])
    assert any(row["id"].startswith("FRONTIER") for row in mixed[4:])


def test_full_static_scoped_frontier_skips_discarded_connected_search() -> None:
    assert _scoped_frontier_seed_mode(
        task_scoped=True,
        dynamic_generation=False,
        evidence_frontier_fraction=1.0,
    ) == "static_scoped_evidence_frontier"
    assert _scoped_frontier_seed_mode(
        task_scoped=True,
        dynamic_generation=False,
        evidence_frontier_fraction=0.999,
    ) is None
    assert _scoped_frontier_seed_mode(
        task_scoped=False,
        dynamic_generation=True,
        evidence_frontier_fraction=1.0,
    ) is None
    assert _scoped_frontier_seed_mode(
        task_scoped=True,
        dynamic_generation=True,
        evidence_frontier_fraction=0.0,
    ) == "dynamic_scoped_evidence_frontier"


def test_hybrid_prefix_screen_labels_and_resolves_fixed_branches(tmp_path: Path) -> None:
    source = tmp_path / "seed_00" / "case" / "window" / "hypotheses_raw.json"
    branch_dir = source.parent / "candidate_branches"
    branch_dir.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    (branch_dir / "general_graph.json").write_text("{}", encoding="utf-8")
    (branch_dir / "task_scoped.json").write_text("{}", encoding="utf-8")

    general, scoped = _branch_paths(source)

    assert hybrid_prefix_method_label(25) == "neurodiscovery_prefix_k025"
    assert general.name == "general_graph.json"
    assert scoped.name == "task_scoped.json"


def test_fixed_candidate_pool_ablation_keeps_branches_independent() -> None:
    def hypothesis(candidate_id: str, source_id: str, target_id: str) -> dict:
        return {
            "id": candidate_id,
            "source_id": source_id,
            "target_id": target_id,
            "path": [{"from_id": source_id, "to_id": target_id}],
        }

    general = {
        "hypotheses": [
            hypothesis("general-1", "G:1", "O:1"),
            hypothesis("duplicate-pair", "G:2", "O:2"),
        ]
    }
    scoped = {
        "hypotheses": [
            hypothesis("scoped-1", "S:1", "O:1"),
            hypothesis("duplicate-pair-scoped", "G:2", "O:2"),
        ]
    }

    variants = _legacy_pool_variants(
        general,
        scoped,
        case_study_id="brain_age",
        max_candidates=4,
        task_scope_fraction=0.5,
    )

    assert [row["id"] for row in variants[METHOD_GENERAL_LEGACY]["hypotheses"]] == [
        "general-1",
        "duplicate-pair",
    ]
    assert [row["id"] for row in variants[METHOD_SCOPED_LEGACY]["hypotheses"]] == [
        "scoped-1",
        "duplicate-pair-scoped",
    ]
    hybrid = variants[METHOD_HYBRID_LEGACY]
    assert len(hybrid["hypotheses"]) == 3
    assert hybrid["metadata"]["candidate_pool"]["protect_general_top_k"] == 0
    variants[METHOD_GENERAL_LEGACY]["hypotheses"][0]["id"] = "mutated"
    assert general["hypotheses"][0]["id"] == "general-1"


def test_frozen_baseline_frontier_does_not_filter_targets_by_first_source() -> None:
    concepts = {
        "IM:BAD": {
            "preferred_name": "MRI",
            "domain_tags": ["imaging_feature"],
        },
        "IM:GOOD": {
            "preferred_name": "hippocampal volume",
            "domain_tags": ["imaging_feature"],
        },
        "D:1": {
            "preferred_name": "Alzheimer disease",
            "domain_tags": ["disease"],
        },
        "D:2": {
            "preferred_name": "Parkinson disease",
            "domain_tags": ["disease"],
        },
    }
    atoms = {
        "IM:BAD": frozenset({"imaging_marker"}),
        "IM:GOOD": frozenset({"imaging_marker"}),
        "D:1": frozenset({"disease"}),
        "D:2": frozenset({"disease"}),
    }
    support_edges = [
        FrozenEdge(
            "IM:BAD", "IM:GOOD", "associated_with", 0.8,
            source_paper={"pmid": "1"}, year=2020,
        ),
        FrozenEdge(
            "D:1", "D:2", "associated_with", 0.8,
            source_paper={"pmid": "2"}, year=2020,
        ),
    ]
    adjacency = {node_id: [] for node_id in concepts}
    for edge in support_edges:
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names={key: value["preferred_name"] for key, value in concepts.items()},
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="openscholar_rag",
        case=case_study_by_name("differential_diagnosis"),
        index=index,
        adjacency=adjacency,
        freeze_year=2020,
        target_count=2,
        seed=3,
    )

    assert payload["metadata"]["valid"] == 2
    assert {row["source_id"] for row in payload["hypotheses"]} == {"IM:GOOD"}
    assert {row["target_id"] for row in payload["hypotheses"]} == {"D:1", "D:2"}


def test_frozen_baseline_frontier_fills_three_node_chain_budget() -> None:
    concepts = {
        "G:1": {"preferred_name": "APOE pathway", "domain_tags": ["gene_pathway"]},
        "G:2": {"preferred_name": "MAPT pathway", "domain_tags": ["gene_pathway"]},
        "IM:1": {"preferred_name": "hippocampal volume", "domain_tags": ["imaging_feature"]},
        "IM:2": {"preferred_name": "cortical thickness", "domain_tags": ["imaging_feature"]},
        "O:1": {"preferred_name": "cognitive decline", "domain_tags": ["treatment_outcome"]},
        "O:2": {"preferred_name": "disability progression", "domain_tags": ["treatment_outcome"]},
    }
    atoms = {
        "G:1": frozenset({"gene_target"}),
        "G:2": frozenset({"gene_target"}),
        "IM:1": frozenset({"imaging_marker"}),
        "IM:2": frozenset({"imaging_marker"}),
        "O:1": frozenset({"outcome"}),
        "O:2": frozenset({"outcome"}),
    }
    support_edges = [
        FrozenEdge("G:1", "G:2", "associated_with", 0.8, source_paper={"pmid": "1"}),
        FrozenEdge("IM:1", "IM:2", "associated_with", 0.8, source_paper={"pmid": "2"}),
        FrozenEdge("O:1", "O:2", "associated_with", 0.8, source_paper={"pmid": "3"}),
    ]
    adjacency = {node_id: [] for node_id in concepts}
    for edge in support_edges:
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
    index = FrozenGraphIndex(
        concepts=concepts,
        names={key: value["preferred_name"] for key, value in concepts.items()},
        atoms=atoms,
        adjacency=adjacency,
        direct_pairs=set(),
    )

    payload = generate_case_hypotheses(
        method="openscholar_rag",
        case=case_study_by_name("case2_pathway_mediation"),
        index=index,
        adjacency=adjacency,
        freeze_year=2020,
        target_count=2,
        seed=3,
    )

    assert payload["metadata"]["valid"] == 2
    assert all(
        hypothesis["hypothesis_type"] == "evidence_frontier_chain"
        for hypothesis in payload["hypotheses"]
    )
    assert all(len(hypothesis["path"]) == 2 for hypothesis in payload["hypotheses"])


def test_frozen_baseline_claim_graph_uses_semantic_endpoint_projection(
    tmp_path: Path,
) -> None:
    concepts = {
        "TASK:FEAR": {
            "preferred_name": "Fear",
            "domain_tags": ["emotion"],
        },
        "D:PSYCHOSIS": {
            "preferred_name": "Psychosis",
            "domain_tags": ["disease"],
        },
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names={"TASK:FEAR": "Fear", "D:PSYCHOSIS": "Psychosis"},
        atoms={
            "TASK:FEAR": frozenset({"cognitive_task"}),
            "D:PSYCHOSIS": frozenset({"disease"}),
        },
        adjacency={},
        direct_pairs=set(),
    )
    claims_path = tmp_path / "claims.jsonl"
    claims_path.write_text(
        json.dumps({
            "id": "CLM:collision",
            "claim_case_study_ids": ["functional_localization"],
            "subject_id": "TASK:FEAR",
            "subject_name": "fear-conditioning task",
            "subject_type": "COGNITIVE_TASK",
            "predicate": "activates",
            "object_id": "D:PSYCHOSIS",
            "object_name": "diminished EEG response amplitude",
            "object_type": "ELECTROPHYSIOLOGY_MARKER",
        }) + "\n",
        encoding="utf-8",
    )

    all_claims, scoped, audit = load_semantic_claim_adjacencies(
        claims_path,
        index,
        ["functional_localization"],
    )

    assert all_claims
    assert scoped["functional_localization"]
    assert ("D:PSYCHOSIS", "TASK:FEAR") not in index.direct_pairs
    assert any(
        any(node.startswith("CLAIM_ENTITY:") for node in pair)
        for pair in index.direct_pairs
    )
    assert audit["semantic_claim_edges"] == 1
    assert audit["scoped_claim_contract_accepted::functional_localization"] == 1


def test_frozen_scoped_baseline_rejects_claims_outside_relation_contract(
    tmp_path: Path,
) -> None:
    concepts = {
        "TASK:FEAR": {"preferred_name": "fear task", "domain_tags": ["paradigm"]},
        "D:PSYCHOSIS": {"preferred_name": "psychosis", "domain_tags": ["disease"]},
    }
    index = FrozenGraphIndex(
        concepts=concepts,
        names={"TASK:FEAR": "fear task", "D:PSYCHOSIS": "psychosis"},
        atoms={
            "TASK:FEAR": frozenset({"cognitive_task"}),
            "D:PSYCHOSIS": frozenset({"disease"}),
        },
        adjacency={},
        direct_pairs=set(),
    )
    claims_path = tmp_path / "claims.jsonl"
    claims_path.write_text(
        json.dumps({
            "id": "CLM:wrong-scope-shape",
            "claim_case_study_ids": ["functional_localization"],
            "subject_id": "TASK:FEAR",
            "subject_name": "fear task",
            "subject_type": "COGNITIVE_TASK",
            "predicate": "predicts",
            "object_id": "D:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        }) + "\n",
        encoding="utf-8",
    )

    all_claims, scoped, audit = load_semantic_claim_adjacencies(
        claims_path,
        index,
        ["functional_localization"],
    )

    assert all_claims
    assert scoped["functional_localization"] == {}
    assert audit["scoped_claim_contract_rejected::functional_localization"] == 1


def test_fixed_budget_records_case_study_id(tmp_path: Path) -> None:
    output = tmp_path / "hypotheses_raw.json"
    padded = _enforce_fixed_budget(
        {"hypotheses": [{"id": "HYP:0"}]},
        output_path=output,
        target_per_case_study=3,
        case_study_id="brain_age",
    )
    assert len(padded["hypotheses"]) == 3
    assert padded["hypotheses"][0]["metadata"]["case_study_id"] == "brain_age"
    assert padded["hypotheses"][-1]["metadata"]["case_study_id"] == "brain_age"
    assert padded["hypotheses"][-1]["metadata"]["generation_failure"] is True


def test_fixed_budget_is_idempotent_for_generation_failure_padding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "hypotheses_raw.json"
    first = _enforce_fixed_budget(
        {"hypotheses": [{"id": "HYP:0"}]},
        output_path=output,
        target_per_case_study=3,
        case_study_id="brain_age",
    )

    second = _enforce_fixed_budget(
        json.loads(output.read_text(encoding="utf-8")),
        output_path=output,
        target_per_case_study=3,
        case_study_id="brain_age",
    )

    assert len(first["hypotheses"]) == len(second["hypotheses"]) == 3
    assert second["metadata"]["fixed_budget"]["valid_before_padding"] == 1
    assert sum(
        hypothesis["metadata"].get("generation_failure", False)
        for hypothesis in second["hypotheses"]
    ) == 2


def test_fixed_budget_preserves_ranked_tail_for_random_baseline(tmp_path: Path) -> None:
    output = tmp_path / "hypotheses_raw.json"
    payload = {"hypotheses": [{"id": f"HYP:{idx}"} for idx in range(5)]}

    prepared = _enforce_fixed_budget(
        payload,
        output_path=output,
        target_per_case_study=3,
        case_study_id="brain_age",
    )

    assert [hypothesis["id"] for hypothesis in prepared["hypotheses"]] == [
        "HYP:0",
        "HYP:1",
        "HYP:2",
        "HYP:3",
        "HYP:4",
    ]
    assert prepared["metadata"]["fixed_budget"]["ranked_pool_after_padding"] == 5
    assert prepared["metadata"]["fixed_budget"]["requested_evaluation_prefix"] == 3
    assert prepared["metadata"]["fixed_budget"]["valid_in_evaluation_prefix"] == 3
    assert prepared["metadata"]["fixed_budget"]["failure_slots_in_evaluation_prefix"] == 0
    assert prepared["metadata"]["fixed_budget"]["candidate_tail_retained_beyond_budget"] == 2


def test_fixed_budget_removes_semantic_path_duplicates_before_padding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "hypotheses_raw.json"
    shared_path = [
        {"from_id": "A", "to_id": "B"},
        {"from_id": "B", "to_id": "C"},
    ]
    prepared = _enforce_fixed_budget(
        {
            "hypotheses": [
                {
                    "id": "HYP:1",
                    "hypothesis_type": "bridge",
                    "source_id": "A",
                    "target_id": "C",
                    "path": shared_path,
                },
                {
                    "id": "HYP:2",
                    "hypothesis_type": "bridge",
                    "source_id": "A",
                    "target_id": "C",
                    "path": shared_path,
                },
            ]
        },
        output_path=output,
        target_per_case_study=3,
        case_study_id="biomarker_discovery",
    )

    assert prepared["hypotheses"][0]["id"] == "HYP:1"
    assert len(prepared["hypotheses"]) == 3
    assert prepared["metadata"]["fixed_budget"]["valid_before_semantic_dedup"] == 2
    assert prepared["metadata"]["fixed_budget"]["semantic_duplicates_removed"] == 1
    assert prepared["metadata"]["fixed_budget"]["valid_before_padding"] == 1


def test_neurodiscovery_manifest_shards_merge_deduplicate_and_sort(
    tmp_path: Path,
) -> None:
    run_a = {
        "method": "neurodiscovery",
        "seed": 1,
        "case_study_id": "brain_age",
        "freeze_year": 2018,
        "future_start_year": 2019,
        "future_end_year": 2023,
        "hypotheses_path": "seed_01/brain_age/hypotheses_raw.json",
        "n_hypotheses": 1000,
        "valid_before_padding": 900,
    }
    run_b = {
        **run_a,
        "seed": 0,
        "case_study_id": "prognosis",
        "hypotheses_path": "seed_00/prognosis/hypotheses_raw.json",
    }
    common = {
        "schema_version": "neurodiscovery-hindcasting-replicates.v1",
        "method": "neurodiscovery",
        "target_per_case_study": 1000,
        "generation_pool_size": 1200,
    }
    (tmp_path / "generation_manifest_shard_a.json").write_text(
        json.dumps({**common, "runs": [run_a]}), encoding="utf-8"
    )
    (tmp_path / "generation_manifest_shard_b.json").write_text(
        json.dumps({**common, "runs": [run_b, run_a]}), encoding="utf-8"
    )

    manifest = merge_generation_manifests(tmp_path)

    assert manifest["schema_version"] == "neurodiscovery-hindcasting-replicates.v2"
    assert [(row["seed"], row["case_study_id"]) for row in manifest["runs"]] == [
        (0, "prognosis"),
        (1, "brain_age"),
    ]
    assert manifest["seeds"] == [0, 1]
    assert len(manifest["shards"]) == 2
    assert manifest["static_score_family"] == "legacy"
    assert manifest["endpoint_canonical_quality_policy"] == (
        "frozen_node_canonical_quality.v1"
    )
    assert manifest["endpoint_canonical_quality_weight"] == 0.20
    assert manifest["engine_reuse_scope"] == "per_run"
    assert manifest["python_hash_seed"] == "unrecorded"
    assert manifest["python_hash_seed_recording_complete"] is False
    written = json.loads((tmp_path / "generation_manifest.json").read_text())
    assert written["runs"] == manifest["runs"]
    assert not (tmp_path / ".generation_manifest.lock").exists()


def test_neurodiscovery_manifest_merge_rejects_mixed_score_families(
    tmp_path: Path,
) -> None:
    common = {
        "target_per_case_study": 1000,
        "generation_pool_size": 1200,
        "runs": [],
    }
    (tmp_path / "generation_manifest_shard_a.json").write_text(
        json.dumps({**common, "static_score_family": "legacy"}), encoding="utf-8"
    )
    (tmp_path / "generation_manifest_shard_b.json").write_text(
        json.dumps({**common, "static_score_family": "relation_aware"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different generation settings"):
        merge_generation_manifests(tmp_path)


def test_neurodiscovery_manifest_merge_rejects_mixed_endpoint_quality_weights(
    tmp_path: Path,
) -> None:
    common = {
        "target_per_case_study": 1000,
        "generation_pool_size": 1200,
        "runs": [],
    }
    (tmp_path / "generation_manifest_shard_a.json").write_text(
        json.dumps({**common, "endpoint_canonical_quality_weight": 0.0}),
        encoding="utf-8",
    )
    (tmp_path / "generation_manifest_shard_b.json").write_text(
        json.dumps({**common, "endpoint_canonical_quality_weight": 0.2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different generation settings"):
        merge_generation_manifests(tmp_path)


def test_neurodiscovery_manifest_merge_marks_partially_recorded_hash_seed(
    tmp_path: Path,
) -> None:
    common = {
        "target_per_case_study": 1000,
        "generation_pool_size": 1200,
        "runs": [],
    }
    (tmp_path / "generation_manifest_shard_a.json").write_text(
        json.dumps({**common, "python_hash_seed": "0"}), encoding="utf-8"
    )
    (tmp_path / "generation_manifest_shard_b.json").write_text(
        json.dumps(common), encoding="utf-8"
    )

    manifest = merge_generation_manifests(tmp_path)

    assert manifest["python_hash_seed"] == "0"
    assert manifest["python_hash_seed_recording_complete"] is False


def test_neurodiscovery_manifest_merge_rejects_conflicting_rows(
    tmp_path: Path,
) -> None:
    run = {
        "method": "neurodiscovery",
        "seed": 0,
        "case_study_id": "brain_age",
        "freeze_year": 2018,
        "future_start_year": 2019,
        "future_end_year": 2023,
        "hypotheses_path": "first.json",
        "n_hypotheses": 1000,
    }
    common = {
        "target_per_case_study": 1000,
        "generation_pool_size": 1200,
    }
    (tmp_path / "generation_manifest_shard_a.json").write_text(
        json.dumps({**common, "runs": [run]}), encoding="utf-8"
    )
    (tmp_path / "generation_manifest_shard_b.json").write_text(
        json.dumps(
            {**common, "runs": [{**run, "hypotheses_path": "second.json"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting hindcasting manifest rows"):
        merge_generation_manifests(tmp_path)
    assert not (tmp_path / ".generation_manifest.lock").exists()


def test_hybrid_candidate_pool_preserves_prefix_and_adds_scoped_quota() -> None:
    def candidate(prefix: str, index: int) -> dict[str, object]:
        return {
            "id": f"{prefix}:{index}",
            "hypothesis_type": "bridge",
            "source_id": f"{prefix}:S{index}",
            "target_id": f"{prefix}:T{index}",
            "path": [{"from_id": f"{prefix}:S{index}", "to_id": f"{prefix}:T{index}"}],
            "metadata": {},
        }

    general = {"hypotheses": [candidate("G", index) for index in range(10)]}
    duplicate = candidate("G", 0)
    duplicate["id"] = "SCOPED:DUPLICATE"
    scoped = {
        "hypotheses": [duplicate] + [candidate("S", index) for index in range(5)]
    }

    merged = _merge_hybrid_candidate_payloads(
        general,
        scoped,
        case_study_id="functional_localization",
        max_candidates=12,
        task_scope_fraction=0.2,
        protect_general_top_k=3,
    )

    hypotheses = merged["hypotheses"]
    assert [row["source_id"] for row in hypotheses[:3]] == ["G:S0", "G:S1", "G:S2"]
    assert sum(row["metadata"]["candidate_pool_branch"] == "task_scoped" for row in hypotheses) == 2
    assert {row["source_id"] for row in hypotheses if row["source_id"].startswith("G:")} == {
        f"G:S{index}" for index in range(10)
    }
    assert len({row["id"] for row in hypotheses}) == len(hypotheses)
    assert merged["metadata"]["candidate_pool"]["scoped_included"] == 2


def test_candidate_coverage_separates_endpoint_path_and_node_cooccurrence() -> None:
    hypotheses = [
        {
            "hypothesis_type": "bridge",
            "source_id": "A",
            "target_id": "B",
            "path": [
                {"from_id": "A", "to_id": "M"},
                {"from_id": "M", "to_id": "B"},
            ],
            "metadata": {},
        },
        {
            "hypothesis_type": "bridge",
            "source_id": "X",
            "target_id": "Y",
            "path": [
                {"from_id": "X", "to_id": "C"},
                {"from_id": "C", "to_id": "D"},
                {"from_id": "D", "to_id": "Y"},
            ],
            "metadata": {},
        },
        {
            "hypothesis_type": "generation_failure",
            "source_id": "FAIL",
            "target_id": "PAD",
            "path": [{"from_id": "FAIL", "to_id": "PAD"}],
            "metadata": {"generation_failure": True},
        },
    ]

    indexes = _candidate_indexes(hypotheses)

    assert indexes["valid_hypotheses"] == 2
    assert indexes["endpoint_rank"][("A", "B")] == 1
    assert ("A", "B") not in indexes["path_edge_rank"]
    assert indexes["path_edge_rank"][("C", "D")] == 2
    assert indexes["node_hypotheses"]["A"] & indexes["node_hypotheses"]["B"] == {1}
    assert "FAIL" not in indexes["node_rank"]


def test_neurodiscovery_frontier_pair_is_executable_and_endpoint_unique() -> None:
    case = case_study_by_name("biomarker_discovery")
    existing = {
        "id": "EXISTING",
        "hypothesis_type": "claim_bridge",
        "source_id": "IM:THICKNESS",
        "source_name": "cortical thickness",
        "target_id": "MSH:SCHIZOPHRENIA",
        "target_name": "schizophrenia",
        "path": [
            {"from_id": "IM:THICKNESS", "to_id": "MEDIATOR"},
            {"from_id": "MEDIATOR", "to_id": "MSH:SCHIZOPHRENIA"},
        ],
        "metadata": {},
    }
    duplicate_pair = {
        "id": "FRONTIER:DUPLICATE",
        "hypothesis_type": "retrieval_composed_pair",
        "source_id": "IM:THICKNESS",
        "source_name": "cortical thickness",
        "target_id": "MSH:SCHIZOPHRENIA",
        "target_name": "schizophrenia",
        "path": [],
        "confidence_score": 0.7,
        "metadata": {},
    }
    novel_pair = {
        **duplicate_pair,
        "id": "FRONTIER:NOVEL",
        "target_id": "MSH:BIPOLAR",
        "target_name": "bipolar disorder",
    }

    executable = _make_frontier_candidate_executable(
        novel_pair,
        case=case,
        generation_round=2,
    )
    assert len(executable["path"]) == 1
    assert executable["path"][0]["claim_id"] == ""
    assert executable["metadata"]["proposed_endpoint_relation"] is True
    assert executable["metadata"]["frontier_uses_future_outcomes"] is False

    merged = _merge_neurodiscovery_frontier_payload(
        {"hypotheses": [existing], "metadata": {}},
        {"hypotheses": [duplicate_pair, novel_pair]},
        case=case,
        target=2,
        generation_round=2,
        excluded_semantic_keys=frozenset(),
    )
    assert len(merged["hypotheses"]) == 2
    assert merged["hypotheses"][1]["target_id"] == "MSH:BIPOLAR"
    audit = merged["metadata"]["neurodiscovery_evidence_frontier"]
    assert audit["frontier_candidates_added"] == 1
    assert audit["frontier_duplicates_or_previous_rounds_rejected"] == 1


def test_neurodiscovery_frontier_feedback_keeps_mutations_and_exploration() -> None:
    case = case_study_by_name("biomarker_discovery")

    def pair(source: str, target: str) -> dict[str, Any]:
        return {
            "id": f"{source}:{target}",
            "hypothesis_type": "retrieval_composed_pair",
            "source_id": source,
            "source_name": source,
            "target_id": target,
            "target_name": target,
            "path": [],
            "confidence_score": 0.7,
            "metadata": {},
        }

    feedback = FeedbackState(
        [
            FeedbackRecord.from_dict(
                {
                    "status": "supported",
                    "hypothesis_id": "SUPPORTED:A:B",
                    "source_id": "A",
                    "target_id": "B",
                    "path_node_ids": ["A", "B"],
                }
            )
        ]
    )
    merged = _merge_neurodiscovery_frontier_payload(
        {"hypotheses": [], "metadata": {}},
        {
            "hypotheses": [
                pair("A", "C"),
                pair("D", "B"),
                pair("X", "Y"),
                pair("Q", "R"),
            ]
        },
        case=case,
        target=3,
        generation_round=1,
        excluded_semantic_keys=frozenset(),
        feedback_state=feedback,
        feedback_mutation_fraction=2 / 3,
    )
    rows = merged["hypotheses"]
    assert len(rows) == 3
    assert sum(bool((row.get("metadata") or {}).get("feedback_mutation")) for row in rows) == 2
    assert any(
        not bool((row.get("metadata") or {}).get("feedback_mutation"))
        for row in rows
    )
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("feedback_mutation"):
            assert (row["source_id"] == "A") != (row["target_id"] == "B")
            assert metadata["feedback_mutation_affinity"] == 1.0
            assert metadata["feedback_mutation_anchor_ids"] == ["SUPPORTED:A:B"]
    audit = merged["metadata"]["neurodiscovery_evidence_frontier"]
    assert audit["feedback_mutation_candidates_available"] >= 3
    assert audit["feedback_mutation_candidates_added"] == 2
    assert audit["feedback_conditioned_candidates_generated"] == 6
    assert audit["feedback_records_available"] == 1


def test_feedback_conditioned_endpoint_expansion_is_bidirectional() -> None:
    case = case_study_by_name("biomarker_discovery")
    feedback = FeedbackState(
        [
            FeedbackRecord.from_dict(
                {
                    "status": "supported",
                    "hypothesis_id": "SUPPORTED:IM:OLD",
                    "source_id": "IM:SUPPORTED",
                    "target_id": "D:OLD",
                }
            )
        ]
    )
    raw = [
        {
            "hypothesis_type": "retrieval_composed_pair",
            "source_id": "IM:OTHER",
            "source_name": "other marker",
            "target_id": "D:NEW",
            "target_name": "new disease",
            "path": [],
            "confidence_score": 0.6,
            "evidence_score": 0.8,
            "supporting_claims": ["SOURCE:CLAIM", "TARGET:CLAIM"],
            "metadata": {
                "source_endpoint_claim_ids": ["SOURCE:CLAIM"],
                "source_endpoint_paper_keys": ["PMID:SOURCE"],
                "source_endpoint_support_confidence": 0.7,
                "target_endpoint_claim_ids": ["TARGET:CLAIM"],
                "target_endpoint_paper_keys": ["PMID:TARGET"],
                "target_endpoint_support_confidence": 0.8,
            },
        }
    ]

    mutations = _feedback_conditioned_endpoint_mutations(
        raw,
        case=case,
        feedback_state=feedback,
        generation_round=2,
        max_candidates=10,
    )

    assert len(mutations) == 2
    by_direction = {
        mutation["metadata"]["feedback_mutation_direction"]: mutation
        for mutation in mutations
    }
    preserve_input = by_direction["preserve_task_input"]
    assert preserve_input["source_id"] == "IM:SUPPORTED"
    assert preserve_input["target_id"] == "D:NEW"
    assert preserve_input["supporting_claims"] == ["TARGET:CLAIM"]
    assert preserve_input["path"][0]["from_id"] == "IM:SUPPORTED"
    assert preserve_input["metadata"]["feedback_source_is_experiment_result"] is True
    assert preserve_input["metadata"]["feedback_target_uses_frozen_evidence"] is True

    preserve_output = by_direction["preserve_task_output"]
    assert preserve_output["source_id"] == "IM:OTHER"
    assert preserve_output["target_id"] == "D:OLD"
    assert preserve_output["supporting_claims"] == ["SOURCE:CLAIM"]
    assert preserve_output["path"][0]["to_id"] == "D:OLD"
    assert preserve_output["metadata"]["feedback_target_is_experiment_result"] is True
    assert preserve_output["metadata"]["feedback_source_uses_frozen_evidence"] is True
    assert all(
        mutation["metadata"]["feedback_uses_terminal_outcomes"] is False
        for mutation in mutations
    )


def test_random_baseline_is_not_reported_when_k_exhausts_pool() -> None:
    scored = [
        {"primary_hit": True, "any_future_hit": True, "endpoint_hit": True},
        {"primary_hit": False, "any_future_hit": False, "endpoint_hit": False},
    ]

    summary, trials = _random_baseline(scored, 2, 20, random.Random(1))

    assert summary["applicable"] is False
    assert summary["candidate_pool_size"] == 2
    assert trials == []


def test_unique_discovery_metrics_do_not_reward_duplicate_hypotheses() -> None:
    duplicate_rows = [
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "primary_lead_time": 1,
            "endpoint_hit": True,
            "endpoint_discovery_key": "endpoint:A|B",
            "any_path_edge_hit": False,
            "all_path_edges_hit": False,
            "any_future_hit": True,
            "path_edge_hit_rate": 0.0,
        },
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "primary_lead_time": 1,
            "endpoint_hit": True,
            "endpoint_discovery_key": "endpoint:A|B",
            "any_path_edge_hit": False,
            "all_path_edges_hit": False,
            "any_future_hit": True,
            "path_edge_hit_rate": 0.0,
        },
    ]

    result = _aggregate(duplicate_rows, future_pair_total=4)

    assert result["primary_hits"] == 2
    assert result["unique_primary_discoveries"] == 1
    assert result["recovered_future_pairs"] == 1
    assert result["future_pair_recall"] == 0.25


def test_random_baseline_deduplicates_discovery_groups() -> None:
    scored = [
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "endpoint_hit": True,
            "endpoint_discovery_key": "endpoint:A|B",
            "any_future_hit": True,
        },
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "endpoint_hit": True,
            "endpoint_discovery_key": "endpoint:A|B",
            "any_future_hit": True,
        },
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:C|D",
            "endpoint_hit": True,
            "endpoint_discovery_key": "endpoint:C|D",
            "any_future_hit": True,
        },
        {
            "primary_hit": False,
            "primary_discovery_key": "",
            "endpoint_hit": False,
            "endpoint_discovery_key": "",
            "any_future_hit": False,
        },
    ]

    summary, trials = _random_baseline(scored, 2, 100, random.Random(7))

    assert summary["applicable"] is True
    assert all(row["unique_primary_discoveries"] <= row["primary_hits"] for row in trials)
    assert any(
        row["primary_hits"] == 2 and row["unique_primary_discoveries"] == 1
        for row in trials
    )
    assert summary["mean_unique_primary_discoveries"] < summary["mean_primary_hits"]


def test_aggregate_keeps_inapplicable_random_baseline_missing(tmp_path: Path) -> None:
    metrics_path = (
        tmp_path
        / "brain_age"
        / "kg2016_to_2017_2021"
        / "hindcasting"
        / "metrics.json"
    )
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "case_study_id": "brain_age",
                "freeze_year": 2016,
                "future_start_year": 2017,
                "future_end_year": 2021,
                "n_hypotheses": 100,
                "benchmark_status": "non_executable",
                "benchmark_reason": "no future relations",
                "topk": {
                    "100": {
                        "observed": {"primary_hits": 3},
                        "random_same_hypothesis_pool": {
                            "applicable": False,
                            "candidate_pool_size": 100,
                            "reason": "full pool",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    row = collect_metrics(tmp_path)[0]
    assert row["random_applicable"] is False
    assert row["random_mean_primary_hits"] is None
    assert row["random_candidate_pool_size"] == 100
    assert row["benchmark_status"] == "non_executable"


def test_cs1_hindcasting_uses_kg_entity_task_generation(tmp_path: Path) -> None:
    command = _generation_command(
        case_study_id="case1_transdiagnostic",
        kg_path=tmp_path / "knowledge_graph.json",
        output_dir=tmp_path / "run",
        target_per_case_study=100,
        seed=31,
    )

    assert "batch" in command
    assert "case-study" not in command
    assert command[command.index("--tasks") + 1] == "transdiagnostic_clustering"
    assert command[command.index("--claim-scope") + 1] == "case1_transdiagnostic"
    assert command[command.index("--target-per-task") + 1] == "100"
    assert command[command.index("--generation-seed") + 1] == "31"




def test_non_cs1_hindcasting_keeps_registered_case_study_generator(tmp_path: Path) -> None:
    command = _generation_command(
        case_study_id="brain_age",
        kg_path=tmp_path / "knowledge_graph.json",
        output_dir=tmp_path / "run",
        target_per_case_study=100,
        seed=31,
    )

    assert "case-study" in command
    assert "brain_age" in command
    assert "--tasks" not in command


def test_generation_pool_size_is_separate_from_evaluation_budget(tmp_path: Path) -> None:
    command = _generation_command(
        case_study_id="brain_age",
        kg_path=tmp_path / "knowledge_graph.json",
        output_dir=tmp_path / "run",
        target_per_case_study=100,
        generation_pool_size=300,
        seed=31,
    )

    assert command[command.index("--target-per-task") + 1] == "300"

    matrix_command = command_for_case_study(
        case=case_study_by_name("brain_age"),
        input_dir=tmp_path / "input",
        snapshot_root=tmp_path / "snapshots",
        output_root=tmp_path / "outputs",
        windows=[],
        target_per_case_study=100,
        random_trials=10,
        seed=3,
        generate_only=True,
        force_generation=False,
        force_evaluation=False,
        generation_pool_size=300,
    )
    assert matrix_command[matrix_command.index("--generation-pool-size") + 1] == "300"


def test_cs1_hindcasting_rejects_synthetic_experiment_candidates() -> None:
    hypotheses = [
        {
            "id": "case1_exhaustive_1",
            "hypothesis_type": "case1_candidate",
            "source_id": "CUI:D1",
            "target_id": "CASE1:CANDIDATE:roi:feature",
        }
    ]

    with pytest.raises(ValueError, match="synthetic disease x ROI x feature"):
        _validate_hindcasting_hypotheses(hypotheses, "case1_transdiagnostic")


def test_task_domain_expansion_is_stably_sorted() -> None:
    class CapturingEngine:
        domain_pairs: list[tuple[str, str]] = []

        def batch_generate(self, **kwargs):
            self.domain_pairs = kwargs["domain_pairs"]
            return []

        @staticmethod
        def post_process(hypotheses):
            return hypotheses

    engine = CapturingEngine()
    HypothesisEngine.batch_generate_for_task(
        engine,
        task_by_name("transdiagnostic_clustering"),
    )

    assert engine.domain_pairs == sorted(engine.domain_pairs)


def test_cmd_batch_saves_ranked_not_generation_order(tmp_path: Path) -> None:
    low = Hypothesis(id="HYP:low", composite_score=0.1)
    high = Hypothesis(id="HYP:high", composite_score=0.9)

    class RankingEngine:
        saved: list[Hypothesis] = []

        @staticmethod
        def batch_generate_for_task(*_args, **_kwargs):
            return [low, high]

        @staticmethod
        def rank_hypotheses(_hypotheses, *, top_n):
            assert top_n == 2
            return [high, low]

        def save_hypotheses(self, hypotheses, _output):
            self.saved = list(hypotheses)

    engine = RankingEngine()
    cmd_batch(
        engine,
        tmp_path / "hypotheses.json",
        task_filter="brain_age",
        chain_filter="",
        max_retries=1,
    )

    assert [hypothesis.id for hypothesis in engine.saved] == ["HYP:high", "HYP:low"]


def test_cmd_batch_retry_deduplicates_paths_not_reused_local_ids(
    tmp_path: Path,
) -> None:
    def hypothesis(source: str, target: str) -> Hypothesis:
        return Hypothesis(
            id="HYP:000001",
            hypothesis_type="bridge",
            source_id=source,
            target_id=target,
            path=[
                HypothesisLink(
                    from_id=source,
                    from_name=source,
                    to_id=target,
                    to_name=target,
                    relation_type="associated_with",
                    confidence=0.8,
                )
            ],
            composite_score=0.5,
        )

    batches = [[hypothesis("A", "B")], [hypothesis("C", "D")]]

    class RetryEngine:
        saved: list[Hypothesis] = []
        calls = 0

        def batch_generate_for_task(self, *_args, **_kwargs):
            result = batches[min(self.calls, len(batches) - 1)]
            self.calls += 1
            return result

        @staticmethod
        def rank_hypotheses(hypotheses, *, top_n):
            return list(hypotheses[:top_n])

        def save_hypotheses(self, hypotheses, _output):
            self.saved = list(hypotheses)

    engine = RetryEngine()
    cmd_batch(
        engine,
        tmp_path / "hypotheses.json",
        task_filter="brain_age",
        chain_filter="",
        target_per_task=2,
        max_retries=2,
    )

    assert [(hyp.source_id, hyp.target_id) for hyp in engine.saved] == [
        ("A", "B"),
        ("C", "D"),
    ]
    assert len({hyp.id for hyp in engine.saved}) == 2


def test_coverage_reports_formal_case_study_statuses(tmp_path: Path) -> None:
    manifest = {
        "windows": [{"freeze_year": year} for year in range(2016, 2021)],
        "case_studies": [
            {"id": name}
            for name in ("prognosis", "brain_age", "cognitive_decoding", "imaging_genetics")
        ],
        "runs": [
            {"case_study_id": "prognosis", "status": "completed", "returncode": 0},
            {"case_study_id": "brain_age", "status": "completed", "returncode": 0},
            {"case_study_id": "cognitive_decoding", "status": "failed", "returncode": 1},
        ],
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metrics = [
        {"case_study_id": "prognosis", "freeze_year": year}
        for year in range(2016, 2021)
    ] + [
        {"case_study_id": "brain_age", "freeze_year": 2016},
        {"case_study_id": "brain_age", "freeze_year": 2017},
    ]

    rows = {row["case_study_id"]: row for row in build_coverage_rows(tmp_path, metrics)}
    assert rows["prognosis"]["status"] == "completed"
    assert rows["brain_age"]["status"] == "partial"
    assert rows["cognitive_decoding"]["status"] == "failed"
    assert rows["imaging_genetics"]["status"] == "planned"


# Updated: 2026-08-12 19:18 HKT - cover balanced, deduplicated endpoint exploration.
# Updated: 2026-08-12 21:54 HKT - cover endpoint-first and multi-input exploration quotas.
# Updated: 2026-08-12 19:29 HKT - cover reproducible frozen-evidence endpoint tail sampling.
# Updated: 2026-08-12 19:55 HKT - cover bounded endpoint-frontier feedback mutation.
# Updated: 2026-08-12 20:34 HKT - cover supported-input expansion with frozen target provenance.
# Updated: 2026-08-12 22:28 HKT - cover the weak frozen-KGE blend used before candidate truncation.
# Updated: 2026-08-13 02:02 HKT - cover bounded high-score and rank-stratified KGE retrieval quotas.
# Updated: 2026-08-13 02:56 HKT - cover frozen-reachable claim-local future endpoints.
# Updated: 2026-08-13 03:01 HKT - cover frozen-supported versus catalog-only claim-local generation.
# Updated: 2026-08-13 03:23 HKT - cover complementary rank-banded KGE anchor selection.
# Updated: 2026-08-13 04:20 HKT - cover rank-stratified evidence shards and bidirectional frozen feedback mutations.
# Updated: 2026-08-13 05:04 HKT - cover wide shallow-KGE anchor selection across deep frozen evidence.
# Updated: 2026-08-13 05:19 HKT - cover bounded frozen-KGE preservation in the executable prefix.
# Updated: 2026-08-13 05:42:01 HKT - cover semantic identity contract cache invalidation.
# Updated: 2026-08-13 05:43:26 HKT - keep semantic tests independent of the heavy dynamic-runner import graph.
# Updated: 2026-08-13 05:53:47 HKT - cover evidence-backed claim-local generation without admitting orphan identities.
# Updated: 2026-08-13 05:56:00 HKT - require a concrete readout name in the claim-local endpoint regression fixture.
# Updated: 2026-08-13 06:24:23 HKT - cover frozen atom availability and reject future-only endpoint retyping.
# Updated: 2026-08-13 06:25:28 HKT - exercise retyping on a stable claim-local endpoint identity.
# Updated: 2026-08-13 06:39:12 HKT - cover non-repeating closed-loop shards for common rank-band sizes.
# Updated: 2026-08-13 07:27:39 HKT - regress short atomic temporary names on deep Windows paths.
# Updated: 2026-08-13 08:56:00 HKT - cover bounded two-dimensional frozen-rank fan-out.
# Updated: 2026-08-13 09:05:00 HKT - exercise the real exploration-heap wrapper shape.
# Updated: 2026-08-13 09:25:00 HKT - cover lossless NeuroDiscovery ranked-prefix selection.
