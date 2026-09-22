from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts.case_study_closed_loop import (
    ClosedLoopConfig,
    RankingRecord,
    closed_loop_neurodiscovery_order,
    compile_policy_order,
    evaluate_rankings,
    freeze_rankings,
    load_frozen_rankings,
    load_precomputed_baseline_rankings,
    paired_randomization_p_value,
    random_walk_order,
    recovery_curve_auc_by_trial,
    run_benchmark,
    validate_hidden_outcomes,
    validate_public_registry,
    write_kg_delta,
)
from core.scripts.case_study_closed_loop_engine import (
    ExperimentalOverlayGraph,
    run_closed_loop_order,
)
from core.scripts.case_study_search_policy import PolicyRule, SearchPolicy
from core.scripts.case_study_closed_loop_specs import (
    COMPLETED_EXPERIMENT_LINES,
    EVALUATION_PROFILES,
    EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES,
    EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES,
    EXPERIMENT_LINE_IMPLEMENTATIONS,
    GENERIC_EXTERNAL_RESULTS_ROLE,
    TASK_PROTOCOLS,
)
from core.scripts.case_study_feedback_adapters import (
    COMPLETED_CASE_STUDY_LINES,
    adapter_for,
)
from core.scripts.case_study_score_components import (
    SCORE_COMPONENT_SCHEMA,
    candidate_id_sha256,
)


FACTORS = ("disease", "anatomy", "feature")


def test_hierarchical_factor_rule_breaks_full_conjunction_ties() -> None:
    public = validate_public_registry(registry(12), factor_fields=FACTORS)
    public["disease"] = ["target"] * 6 + ["other"] * 6
    public["anatomy"] = ["a", "b", "c", "d", "e", "f"] * 2
    policy = SearchPolicy(
        method="baseline",
        trial=0,
        schema_version="case-study-search-policy.v1",
        rules=(
            PolicyRule(
                weight=0.9,
                when={"disease": "target", "anatomy": "a"},
            ),
        ),
        metadata={"rule_matching": "hierarchical_partial_plus_joint"},
    )
    order = compile_policy_order(public, policy, factor_fields=FACTORS)

    assert public.iloc[order[0]]["disease"] == "target"
    assert set(public.iloc[order[:6]]["disease"]) == {"target"}


def test_closed_loop_config_accepts_hierarchical_utility() -> None:
    config = ClosedLoopConfig(feedback_model="hierarchical_utility")

    config.validate()


def test_surrogate_ridge_alpha_must_be_positive() -> None:
    with pytest.raises(ValueError, match="surrogate_ridge_alpha"):
        ClosedLoopConfig(surrogate_ridge_alpha=0.0).validate()


def test_completed_experiment_registry_has_ten_unique_lines() -> None:
    assert COMPLETED_EXPERIMENT_LINES == COMPLETED_CASE_STUDY_LINES
    assert len(COMPLETED_EXPERIMENT_LINES) == 10
    assert len(set(COMPLETED_EXPERIMENT_LINES)) == 10
    assert EXPERIMENT_LINE_IMPLEMENTATIONS["case1_transdiagnostic"] == (
        "case1_transdiagnostic"
    )
    assert EXPERIMENT_LINE_IMPLEMENTATIONS["biomarker_discovery"] == (
        "biomarker_discovery"
    )
    generic_implementations = {
        implementation
        for line, implementation in EXPERIMENT_LINE_IMPLEMENTATIONS.items()
        if line not in {"case1_transdiagnostic", "case2_pathway_mediation"}
    }
    assert generic_implementations == set(TASK_PROTOCOLS)


def registry(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"c{index:03d}" for index in range(n)],
            "disease": [f"d{index % 3}" for index in range(n)],
            "anatomy": [f"r{index % 4}" for index in range(n)],
            "feature": [f"f{index % 2}" for index in range(n)],
            "score_neurodiscovery": np.linspace(1.0, 0.0, n),
        }
    )


def outcomes(public: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": [index % 5 == 0 for index in range(len(public))],
        }
    )


def test_public_registry_rejects_hidden_fields() -> None:
    public = registry()
    public["q_fdr"] = 0.01
    with pytest.raises(ValueError, match="hidden outcome"):
        validate_public_registry(public, factor_fields=FACTORS)


def test_internal_outcomes_require_exact_candidate_coverage() -> None:
    public = validate_public_registry(registry(), factor_fields=FACTORS)
    hidden = outcomes(public).iloc[:-1]
    with pytest.raises(ValueError, match="must label every"):
        validate_hidden_outcomes(public, hidden)


def test_precomputed_baselines_exclude_old_neurodiscovery_orders(
    tmp_path: Path,
) -> None:
    public = validate_public_registry(registry(12), factor_fields=FACTORS)
    records = [
        RankingRecord("official_baseline", trial, np.arange(len(public)))
        for trial in range(2)
    ] + [
        RankingRecord("neurodiscovery", trial, np.arange(len(public))[::-1])
        for trial in range(2)
    ]
    frozen = freeze_rankings(
        public,
        records,
        output_dir=tmp_path / "source",
        task="old_task",
        factor_fields=FACTORS,
        input_files={},
    )

    loaded, audit = load_precomputed_baseline_rankings(
        Path(frozen["manifest_path"]),
        public,
        methods=["random_walk", "official_baseline", "neurodiscovery"],
        n_trials=2,
    )

    assert {(item.method, item.trial) for item in loaded} == {
        ("official_baseline", 0),
        ("official_baseline", 1),
    }
    assert audit["neurodiscovery_ranking_reused"] is False


def test_streamed_overlay_is_compressed_and_delta_uses_compact_index(
    tmp_path: Path,
) -> None:
    public = validate_public_registry(registry(4), factor_fields=FACTORS)
    adapter = adapter_for("biomarker_discovery", factor_fields=FACTORS)
    path = tmp_path / "overlay.jsonl.gz"
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter,
        factor_fields=FACTORS,
        seed=7,
        trial=0,
        stream_path=path,
    )
    hidden = outcomes(public)
    for index, row in hidden.iterrows():
        overlay.append(candidate_index=index, outcome=row.to_dict(), round_index=0)
    manifest = overlay.write(path)

    assert overlay.records == []
    assert manifest["storage_format"] == "jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == len(public)

    delta = write_kg_delta(
        public,
        hidden,
        [],
        path=tmp_path / "delta.jsonl",
        task="biomarker_discovery",
        overlay_manifests=[manifest],
        factor_fields=FACTORS,
    )
    assert delta["storage_mode"] == "compressed_overlay_index"
    assert delta["records"] == len(public)
    assert delta["ledger_records"] == 1


def test_ridge_feedback_uses_only_available_executed_utilities() -> None:
    public = validate_public_registry(registry(6), factor_fields=FACTORS)
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("biomarker_discovery", factor_fields=FACTORS),
        factor_fields=FACTORS,
        seed=17,
        trial=0,
        feedback_model="ridge_surrogate",
    )
    overlay.append(
        candidate_index=0,
        outcome={
            "candidate_id": "c000",
            "validated": True,
            "feedback_status": "supported",
            "feedback_utility": 0.9,
            "feedback_available": True,
        },
        round_index=0,
    )
    overlay.append(
        candidate_index=1,
        outcome={
            "candidate_id": "c001",
            "validated": False,
            "feedback_status": "inconclusive",
            "feedback_utility": float("nan"),
            "feedback_available": False,
        },
        round_index=1,
    )

    scores, audit = overlay.score_candidates(
        feedback_weight=2.0,
        pair_feedback_weight=0.15,
        exploration_weight=0.03,
    )
    manifest = overlay.manifest()

    assert np.isfinite(scores).all()
    assert audit["feedback_model"] == "ridge_surrogate"
    assert audit["utility_records"] == 1
    assert manifest["records"] == 2
    assert manifest["ranking_feedback_records"] == 1
    assert manifest["utility_feedback_records"] == 1


def test_surrogate_design_drops_interactions_with_constant_factors() -> None:
    public = pd.DataFrame(
        [
            {
                "candidate_id": f"c{index:03d}",
                "modality": "structural_mri",
                "atlas": atlas,
                "feature": feature,
                "model": model,
                "score_neurodiscovery": 0.0,
            }
            for index, (atlas, feature, model) in enumerate(
                itertools.product(("a", "b"), ("x", "y"), ("m", "n"))
            )
        ]
    )
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("brain_age"),
        factor_fields=("modality", "atlas", "feature", "model"),
        seed=23,
        trial=0,
        feedback_model="ridge_surrogate",
    )

    audit = overlay.manifest()["surrogate_design"]
    pair_fields = [
        block["fields"]
        for block in audit["categorical_blocks"]
        if block["kind"] == "factor_pair"
    ]

    assert audit["columns"] == 18
    assert all(["modality"] not in fields for fields in pair_fields)
    assert len(audit["skipped_constant_factor_pairs"]) == 3


def test_factor_ucb_rewards_an_unobserved_factor_level() -> None:
    public = pd.DataFrame(
        [
            {
                "candidate_id": f"c{index:03d}",
                "modality": "structural_mri",
                "atlas": atlas,
                "feature": feature,
                "model": model,
                "score_neurodiscovery": 0.0,
            }
            for index, (atlas, feature, model) in enumerate(
                itertools.product(("a", "b"), ("x", "y"), ("m", "n"))
            )
        ]
    )
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("brain_age"),
        factor_fields=("modality", "atlas", "feature", "model"),
        seed=29,
        trial=0,
        feedback_model="factor_ucb_ridge_surrogate",
    )
    overlay.append(
        candidate_index=0,
        outcome={
            "candidate_id": "c000",
            "validated": True,
            "feedback_status": "supported",
            "feedback_utility": 0.9,
            "feedback_available": True,
        },
        round_index=0,
    )

    scores, audit = overlay.score_candidates(
        feedback_weight=0.0,
        pair_feedback_weight=0.0,
        exploration_weight=1.0,
    )

    assert audit["exploration_mode"] == "max_factor_ucb"
    assert scores[4] > scores[0]


def test_hierarchical_feedback_uses_only_available_executed_utilities() -> None:
    public = validate_public_registry(registry(6), factor_fields=FACTORS)
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("biomarker_discovery", factor_fields=FACTORS),
        factor_fields=FACTORS,
        seed=19,
        trial=0,
        feedback_model="hierarchical_utility",
    )
    overlay.append(
        candidate_index=0,
        outcome={
            "candidate_id": "c000",
            "validated": True,
            "feedback_status": "supported",
            "feedback_utility": 0.9,
            "feedback_available": True,
        },
        round_index=0,
    )
    overlay.append(
        candidate_index=1,
        outcome={
            "candidate_id": "c001",
            "validated": False,
            "feedback_status": "inconclusive",
            "feedback_utility": float("nan"),
            "feedback_available": False,
        },
        round_index=1,
    )

    scores, audit = overlay.score_candidates(
        feedback_weight=2.0,
        pair_feedback_weight=0.15,
        exploration_weight=0.03,
    )
    manifest = overlay.manifest()

    assert np.isfinite(scores).all()
    assert audit["feedback_model"] == "hierarchical_utility"
    assert audit["utility_records"] == 1
    assert manifest["records"] == 2
    assert manifest["ranking_feedback_records"] == 1
    assert manifest["utility_feedback_records"] == 1


def test_feedback_status_is_four_state_and_consistent_with_validation() -> None:
    public = validate_public_registry(registry(4), factor_fields=FACTORS)
    hidden = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": [True, False, False, False],
            "feedback_status": [
                "supported",
                "contradicted",
                "inconclusive",
                "execution_failed",
            ],
        }
    )
    normalized = validate_hidden_outcomes(public, hidden)
    assert normalized["feedback_status"].tolist() == [
        "supported",
        "contradicted",
        "inconclusive",
        "execution_failed",
    ]

    hidden.loc[1, "validated"] = True
    with pytest.raises(ValueError, match="validated must be true"):
        validate_hidden_outcomes(public, hidden)


def test_random_walk_is_a_seeded_full_permutation() -> None:
    public = validate_public_registry(registry(), factor_fields=FACTORS)
    first = random_walk_order(
        public, factor_fields=FACTORS, rng=np.random.default_rng(41)
    )
    second = random_walk_order(
        public, factor_fields=FACTORS, rng=np.random.default_rng(41)
    )
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(public)))


def test_closed_loop_first_batch_is_outcome_blind() -> None:
    public = validate_public_registry(registry(36), factor_fields=FACTORS)
    labels_a = np.zeros(len(public), dtype=bool)
    labels_a[::5] = True
    labels_b = ~labels_a
    config = ClosedLoopConfig(
        batch_size=6,
        max_feedback_rounds=20,
        feedback_horizon=24,
    )
    order_a, trace_a = closed_loop_neurodiscovery_order(
        public,
        labels_a,
        factor_fields=FACTORS,
        rng=np.random.default_rng(7),
        config=config,
    )
    order_b, trace_b = closed_loop_neurodiscovery_order(
        public,
        labels_b,
        factor_fields=FACTORS,
        rng=np.random.default_rng(7),
        config=config,
    )
    assert trace_a[0]["candidate_ids"] == trace_b[0]["candidate_ids"]
    assert sorted(order_a.tolist()) == list(range(len(public)))
    assert sorted(order_b.tolist()) == list(range(len(public)))


def test_closed_loop_commits_each_batch_before_outcome_reveal() -> None:
    public = validate_public_registry(registry(18), factor_fields=FACTORS)
    outcomes = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": np.arange(len(public)) % 4 == 0,
        }
    )
    committed: list[dict[str, object]] = []

    def commit(payload: dict[str, object]) -> dict[str, object]:
        committed.append(dict(payload))
        return {"commit_index": len(committed)}

    order, trace, manifest = run_closed_loop_order(
        public,
        outcomes,
        adapter=adapter_for("case1_transdiagnostic", factor_fields=FACTORS),
        factor_fields=FACTORS,
        rng=np.random.default_rng(17),
        config=ClosedLoopConfig(
            batch_size=3,
            max_feedback_rounds=6,
            feedback_horizon=18,
        ),
        seed=17,
        trial=0,
        batch_commit_callback=commit,
    )

    assert sorted(order.tolist()) == list(range(len(public)))
    assert len(committed) == len(trace) == 6
    assert [row["selection_commit"]["commit_index"] for row in trace] == list(
        range(1, 7)
    )
    assert manifest["batch_selection_commits"] == {
        "enabled": True,
        "count": 6,
        "committed_before_selected_outcome_lookup": True,
    }


def test_generalizable_feedback_projection_uses_repeated_search_factors() -> None:
    public = pd.DataFrame(
        {
            "candidate_id": [f"c{index:03d}" for index in range(36)],
            "phenotype": ["cognition"] * 36,
            "atlas": [f"atlas_{index:02d}" for index in range(12) for _ in range(3)],
            "feature_family": ["fc", "node", "graph"] * 12,
            "model": ["ridge", "svm", "elastic"] * 12,
            "score_neurodiscovery": np.linspace(1.0, 0.0, 36),
        }
    )
    factors = ("phenotype", "atlas", "feature_family", "model")
    public = validate_public_registry(public, factor_fields=factors)
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("connectome_behavior", factor_fields=factors),
        factor_fields=factors,
        seed=3,
        trial=0,
        feedback_projection="generalizable_factors",
    )

    manifest = overlay.manifest()

    assert manifest["feedback_factor_groups"] == [
        ["feature_family"],
        ["model"],
    ]
    assert manifest["feedback_projection_audit"]["cardinality_limit"] == 6
    assert manifest["feedback_projection_audit"]["selected_fields"] == [
        "feature_family",
        "model",
    ]


def test_closed_loop_trials_are_seeded_but_not_identical() -> None:
    public = validate_public_registry(registry(72), factor_fields=FACTORS)
    labels = np.arange(len(public)) % 5 == 0
    config = ClosedLoopConfig(
        batch_size=6,
        max_feedback_rounds=20,
        feedback_horizon=48,
        sampling_temperature=0.01,
    )

    first, _ = closed_loop_neurodiscovery_order(
        public,
        labels,
        factor_fields=FACTORS,
        rng=np.random.default_rng(7),
        config=config,
    )
    repeated, _ = closed_loop_neurodiscovery_order(
        public,
        labels,
        factor_fields=FACTORS,
        rng=np.random.default_rng(7),
        config=config,
    )
    second_seed, _ = closed_loop_neurodiscovery_order(
        public,
        labels,
        factor_fields=FACTORS,
        rng=np.random.default_rng(8),
        config=config,
    )

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second_seed)
    assert sorted(second_seed.tolist()) == list(range(len(public)))


def test_closed_loop_honours_multiple_warmup_batches() -> None:
    public = validate_public_registry(registry(36), factor_fields=FACTORS)
    config = ClosedLoopConfig(
        batch_size=6,
        warmup_batches=2,
        max_feedback_rounds=20,
        feedback_horizon=24,
    )

    _, trace = closed_loop_neurodiscovery_order(
        public,
        np.arange(len(public)) % 4 == 0,
        factor_fields=FACTORS,
        rng=np.random.default_rng(11),
        config=config,
    )

    assert [row["overlay_read"] for row in trace[:2]] == [False, False]
    assert all(row["overlay_deferred_by_warmup"] for row in trace[:2])
    assert any(row["overlay_read"] for row in trace[2:])


def test_closed_loop_can_require_positive_support_before_feedback() -> None:
    public = validate_public_registry(registry(36), factor_fields=FACTORS)
    config = ClosedLoopConfig(
        batch_size=6,
        warmup_batches=1,
        warmup_diversity_penalty=10.0,
        min_supported_before_feedback=2,
        max_feedback_rounds=20,
        feedback_horizon=24,
    )

    _, trace = closed_loop_neurodiscovery_order(
        public,
        np.zeros(len(public), dtype=bool),
        factor_fields=FACTORS,
        rng=np.random.default_rng(11),
        config=config,
    )

    assert trace[0]["overlay_deferred_by_warmup"] is True
    assert all(row["overlay_read"] is False for row in trace)
    assert all(row["overlay_deferred_by_feedback_gate"] is True for row in trace[1:])


def test_inconclusive_feedback_can_preserve_static_order() -> None:
    public = validate_public_registry(registry(36), factor_fields=FACTORS)
    hidden = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": False,
            "feedback_status": "inconclusive",
        }
    )
    config = ClosedLoopConfig(
        batch_size=6,
        warmup_batches=1,
        sampling_temperature=0.5,
        diversity_penalty=0.01,
        max_feedback_rounds=20,
        feedback_horizon=24,
        preserve_static_until_informative_feedback=True,
    )

    first, first_trace, overlay = closed_loop_neurodiscovery_order(
        public,
        np.zeros(len(public), dtype=bool),
        factor_fields=FACTORS,
        rng=np.random.default_rng(11),
        config=config,
        outcomes=hidden,
        return_overlay_manifest=True,
    )
    second, _ = closed_loop_neurodiscovery_order(
        public,
        np.zeros(len(public), dtype=bool),
        factor_fields=FACTORS,
        rng=np.random.default_rng(99),
        config=config,
        outcomes=hidden,
    )

    assert np.array_equal(first, np.arange(len(public)))
    assert np.array_equal(second, first)
    assert all(row["static_order_preserved"] for row in first_trace)
    assert overlay["preserve_static_until_informative_feedback"] is True


def test_informative_feedback_waits_for_warmup_before_adaptation() -> None:
    public = validate_public_registry(registry(36), factor_fields=FACTORS)
    hidden = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": [True] + [False] * (len(public) - 1),
            "feedback_status": ["supported"] + ["inconclusive"] * (len(public) - 1),
        }
    )
    config = ClosedLoopConfig(
        batch_size=6,
        warmup_batches=2,
        sampling_temperature=0.5,
        diversity_penalty=0.01,
        max_feedback_rounds=20,
        feedback_horizon=24,
        preserve_static_until_informative_feedback=True,
    )

    _, trace, overlay = closed_loop_neurodiscovery_order(
        public,
        hidden["validated"].to_numpy(dtype=bool),
        factor_fields=FACTORS,
        rng=np.random.default_rng(13),
        config=config,
        outcomes=hidden,
        return_overlay_manifest=True,
    )

    assert trace[0]["candidate_ids"] == public.iloc[:6]["candidate_id"].tolist()
    assert trace[1]["candidate_ids"] == public.iloc[6:12]["candidate_id"].tolist()
    assert [row["static_order_preserved"] for row in trace[:2]] == [True, True]
    assert trace[2]["informative_feedback_active"] is True
    assert trace[2]["static_order_preserved"] is False
    assert overlay["static_order_guard"]["adaptive_ranking_activated"] is True


def test_relation_endpoint_feedback_excludes_atlas_and_model_qualifiers() -> None:
    public = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "gene_pathway": ["APOE", "APOE", "PRS"],
            "atlas": ["aseg", "aseg", "aseg"],
            "imaging_phenotype": ["hippocampus", "entorhinal", "entorhinal"],
            "model": ["glm", "huber", "glm"],
            "score_neurodiscovery": [1.0, 0.5, 0.0],
        }
    )
    fields = ("gene_pathway", "atlas", "imaging_phenotype", "model")
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("imaging_genetics"),
        factor_fields=fields,
        seed=1,
        trial=0,
        feedback_projection="relation_endpoints",
    )
    overlay.append(
        candidate_index=0,
        outcome={"validated": True},
        round_index=0,
    )

    scores, _ = overlay.score_candidates(
        feedback_weight=1.0,
        pair_feedback_weight=0.0,
        exploration_weight=0.0,
    )
    manifest = overlay.manifest()

    assert scores[1] > scores[2]
    assert manifest["feedback_factor_groups"] == [
        ["gene_pathway"],
        ["imaging_phenotype"],
    ]
    assert manifest["feedback_relation_pairs"] == [
        {
            "subject_fields": ["gene_pathway"],
            "object_fields": ["imaging_phenotype"],
        }
    ]
    assert manifest["qualifier_fields_excluded"] == ["atlas", "model"]


def test_front_loaded_batch_schedule_preserves_early_feedback_resolution() -> None:
    public = validate_public_registry(registry(1000), factor_fields=FACTORS)
    config = ClosedLoopConfig(
        batch_size=2,
        max_feedback_rounds=10,
        feedback_horizon=1000,
        feedback_batch_schedule="front_loaded",
    )

    order, trace, overlay = closed_loop_neurodiscovery_order(
        public,
        np.arange(len(public)) % 7 == 0,
        factor_fields=FACTORS,
        rng=np.random.default_rng(17),
        config=config,
        return_overlay_manifest=True,
    )

    assert trace[0]["end_rank"] == 2
    assert len(trace) == 10
    assert trace[-1]["end_rank"] == 1000
    assert sorted(order.tolist()) == list(range(len(public)))
    assert overlay["feedback_batch_schedule"]["name"] == "front_loaded"


def test_counts_only_tuning_mode_preserves_order_without_retaining_audit_records() -> (
    None
):
    public = validate_public_registry(registry(120), factor_fields=FACTORS)
    labels = np.arange(len(public)) % 7 == 0
    config = ClosedLoopConfig(
        batch_size=3,
        max_feedback_rounds=12,
        feedback_horizon=90,
        feedback_batch_schedule="front_loaded",
    )

    full_order, _full_trace = closed_loop_neurodiscovery_order(
        public,
        labels,
        factor_fields=FACTORS,
        rng=np.random.default_rng(23),
        config=config,
    )
    compact_order, compact_trace, compact_overlay = closed_loop_neurodiscovery_order(
        public,
        labels,
        factor_fields=FACTORS,
        rng=np.random.default_rng(23),
        config=config,
        audit_records=False,
        collect_trace=False,
        return_overlay_manifest=True,
    )

    assert np.array_equal(compact_order, full_order)
    assert compact_trace == []
    assert compact_overlay["records"] == 90
    assert compact_overlay["audit_records_retained"] is False
    assert compact_overlay["semantic_projection_verified"] is False


def test_freeze_roundtrip_verifies_hashes(tmp_path: Path) -> None:
    public = validate_public_registry(registry(), factor_fields=FACTORS)
    records = [
        RankingRecord("random_walk", 0, np.arange(len(public), dtype=np.int64)),
        RankingRecord("neurodiscovery", 0, np.arange(len(public) - 1, -1, -1)),
    ]
    input_path = tmp_path / "input.csv"
    public.to_csv(input_path, index=False)
    manifest = freeze_rankings(
        public,
        records,
        output_dir=tmp_path / "frozen",
        task="biomarker_discovery",
        factor_fields=FACTORS,
        input_files={"public": input_path},
    )
    assert manifest["external_data_read_before_freeze"] is False
    loaded, loaded_manifest = load_frozen_rankings(tmp_path / "frozen", public)
    assert loaded_manifest["reused"] is True
    assert np.array_equal(loaded[1].order, records[1].order)


def test_evaluation_reports_same_budget_and_same_recall() -> None:
    labels = np.asarray([True, False, True, False, False, True])
    records = [
        RankingRecord("neurodiscovery", 0, np.arange(6)),
        RankingRecord("random_walk", 0, np.arange(5, -1, -1)),
    ]
    metrics, costs = evaluate_rankings(
        records,
        labels,
        budgets=(2, 4),
        recall_targets=(1 / 3, 2 / 3),
    )
    row = metrics.query("method == 'neurodiscovery' and experiments == 2").iloc[0]
    assert row["hits"] == 1
    cost = costs.query("method == 'neurodiscovery' and recall_target > 0.6").iloc[0]
    assert cost["experiments_required"] == 3


def test_exact_paired_randomization_is_directional() -> None:
    better = paired_randomization_p_value(
        [10, 11, 12, 13], [1, 2, 3, 4], alternative="greater"
    )
    wrong_direction = paired_randomization_p_value(
        [10, 11, 12, 13], [1, 2, 3, 4], alternative="less"
    )
    assert better == pytest.approx(1 / 16)
    assert wrong_direction == pytest.approx(1.0)


def test_recovery_auc_excludes_shared_exhaustive_endpoint() -> None:
    metrics = pd.DataFrame(
        [
            {
                "scope": "internal",
                "method": method,
                "trial": 0,
                "experiments": budget,
                "recall": recall,
            }
            for method, values in {
                "neurodiscovery": ((10, 0.5), (20, 0.8), (30, 1.0)),
                "baseline": ((10, 0.2), (20, 0.4), (30, 1.0)),
            }.items()
            for budget, recall in values
        ]
    )

    auc = recovery_curve_auc_by_trial(metrics).set_index("method")

    assert auc.loc["neurodiscovery", "auc_horizon"] == 20
    assert auc.loc["neurodiscovery", "auc_points"] == 2
    assert auc.loc["neurodiscovery", "recovery_auc"] == pytest.approx(0.45)
    assert auc.loc["baseline", "recovery_auc"] == pytest.approx(0.20)


def test_supplemental_external_validation_and_five_fold_profiles() -> None:
    assert set(TASK_PROTOCOLS) == {
        "biomarker_discovery",
        "differential_diagnosis",
        "disease_subtyping",
        "progression_prediction",
        "connectome_behavior",
        "brain_age",
        "imaging_genetics",
        "prognosis",
    }
    for protocol in TASK_PROTOCOLS.values():
        assert protocol.minimum_trials == 10
        assert protocol.factor_fields
        assert protocol.discovery_datasets
    assert TASK_PROTOCOLS["biomarker_discovery"].external_datasets == ()
    assert EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES == (
        "differential_diagnosis",
        "connectome_behavior",
        "brain_age",
        "progression_prediction",
        "prognosis",
        "imaging_genetics",
    )
    assert EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES == ("disease_subtyping",)
    assert all(
        TASK_PROTOCOLS[task].external_required
        for task in EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES
    )
    assert TASK_PROTOCOLS["disease_subtyping"].external_required is False
    assert TASK_PROTOCOLS["disease_subtyping"].external_datasets == ()
    assert TASK_PROTOCOLS["brain_age"].discovery_datasets == ("HCP-YA",)
    assert GENERIC_EXTERNAL_RESULTS_ROLE == (
        "required_except_registered_task_exemptions"
    )
    assert len(EVALUATION_PROFILES["development"].seeds) == 5
    assert EVALUATION_PROFILES["development"].outer_folds == 5
    assert EVALUATION_PROFILES["development"].benchmark_trials == 5
    assert len(EVALUATION_PROFILES["final"].seeds) == 10
    assert EVALUATION_PROFILES["final"].outer_folds == 5


def test_full_runner_freezes_before_external_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = registry(20)
    hidden = outcomes(public)
    external = hidden.copy()
    external["executable"] = True
    public_path = tmp_path / "public.csv"
    hidden_path = tmp_path / "internal.csv"
    external_path = tmp_path / "external.csv"
    public.to_csv(public_path, index=False)
    hidden.to_csv(hidden_path, index=False)
    external.to_csv(external_path, index=False)
    out = tmp_path / "run"

    from core.scripts import case_study_closed_loop as module

    original = module._read_csv
    reads: list[str] = []

    def guarded_read(path: Path) -> pd.DataFrame:
        reads.append(path.name)
        if path == external_path:
            assert (
                out / "frozen_discovery_rankings" / "frozen_rankings_manifest.json"
            ).is_file()
        return original(path)

    monkeypatch.setattr(module, "_read_csv", guarded_read)
    args = argparse.Namespace(
        task="biomarker_discovery",
        public_candidates=public_path,
        internal_outcomes=hidden_path,
        external_outcomes=external_path,
        search_policies=None,
        policy_schema="case-study-search-policy.v1",
        output_dir=out,
        factor_fields=list(FACTORS),
        methods=["random_walk", "neurodiscovery"],
        trials=2,
        seed=19,
        batch_size=4,
        warmup_batches=1,
        max_feedback_rounds=10,
        feedback_horizon=12,
        budgets=[5, 10],
        recall_targets=[0.25, 0.5],
    )
    manifest = run_benchmark(args)
    assert reads == ["public.csv", "internal.csv", "external.csv"]
    assert manifest["status"] == "complete"
    assert manifest["external"]["loaded_after_freeze"] is True
    assert manifest["formal_kg_mutated"] is False
    delta = manifest["experimental_kg_delta"]
    assert delta["overlay_count"] == 2
    assert delta["feedback_consumed_during_ranking"] is True
    assert delta["semantic_projection_verified"] is True
    assert delta["per_seed_isolation_verified"] is True
    assert delta["records_by_status"]["inconclusive"] > 0
    assert "contradicted" not in delta["records_by_status"]

    previous = "0" * 64
    delta_path = out / "experimental_kg_delta.jsonl"
    for line in delta_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert payload["previous_hash"] == previous
        record_hash = payload.pop("record_hash")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == record_hash
        previous = record_hash


def test_full_runner_activates_a_frozen_score_component_bundle(
    tmp_path: Path,
) -> None:
    public = registry(20)
    hidden = outcomes(public)
    public_path = tmp_path / "public.csv"
    hidden_path = tmp_path / "internal.csv"
    score_path = tmp_path / "scores.csv"
    score_manifest_path = tmp_path / "scores.manifest.json"
    public.to_csv(public_path, index=False)
    hidden.to_csv(hidden_path, index=False)
    scores = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "score_novelty": np.linspace(0.0, 1.0, len(public)),
        }
    )
    scores.to_csv(score_path, index=False)
    score_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCORE_COMPONENT_SCHEMA,
                "outcome_blind": True,
                "frozen_before_experiment": True,
                "score_table_sha256": hashlib.sha256(
                    score_path.read_bytes()
                ).hexdigest(),
                "candidate_id_sha256": candidate_id_sha256(public["candidate_id"]),
                "components": {
                    "novelty": {
                        "column": "score_novelty",
                        "definition": "frozen synthetic test score",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        task="biomarker_discovery",
        public_candidates=public_path,
        internal_outcomes=hidden_path,
        external_outcomes=None,
        score_components=score_path,
        score_components_manifest=score_manifest_path,
        search_policies=None,
        policy_schema="case-study-search-policy.v1",
        output_dir=tmp_path / "run-with-components",
        factor_fields=list(FACTORS),
        methods=["neurodiscovery"],
        trials=2,
        seed=23,
        batch_size=4,
        warmup_batches=1,
        max_feedback_rounds=10,
        feedback_horizon=12,
        budgets=[5, 10],
        recall_targets=[0.25],
    )

    manifest = run_benchmark(args)

    assert manifest["score_component_bundle"]["columns"] == ["score_novelty"]
    overlay = manifest["experimental_kg_delta"]["source_overlays"][0]
    assert overlay["score_components"]["novelty"]["active"] is True


# Updated: 2026-08-11 20:30 HKT
