from __future__ import annotations

import inspect
import csv
import json

import pandas as pd

import neurooracle.scripts.run_neurodiscovery_dynamic_closed_loop_hindcasting as dynamic_runner

from neurooracle.scripts.run_neurodiscovery_dynamic_closed_loop_hindcasting import (
    DynamicLoopConfig,
    DynamicLoopResult,
    _metrics_rows,
    _audit_generator_closure,
    _load_reusable_run,
    _rank_reservoir,
    _select_dynamic_batch,
    execute_dynamic_loop,
    parse_args as parse_dynamic_args,
)
from neurooracle.scripts.run_neurodiscovery_closed_loop_hindcasting import profile_slate
from neurooracle.scripts.run_neurodiscovery_closed_loop_hindcasting import _static_score
from neurooracle.scripts.run_neurodiscovery_closed_loop_hindcasting import parse_args
from neurooracle.src.hindcasting_static_policy import RELATION_COMPONENT_RAW_FIELDS
from neurooracle.src.feedback_state import FeedbackRecord, FeedbackState


def test_closed_loop_parser_supports_development_only() -> None:
    args = parse_args([
        "--development-only",
        "--profiles",
        "legacy_static",
        "relation_scoped_pair_closed_supported_gate",
    ])

    assert args.development_only is True
    assert args.profiles == [
        "legacy_static",
        "relation_scoped_pair_closed_supported_gate",
    ]


def test_dynamic_config_validates_evidence_frontier_fraction() -> None:
    DynamicLoopConfig(evidence_frontier_fraction=0.5).validate()
    for invalid in (-0.01, 1.01):
        try:
            DynamicLoopConfig(evidence_frontier_fraction=invalid).validate()
        except ValueError as exc:
            assert "evidence_frontier_fraction" in str(exc)
        else:
            raise AssertionError("invalid evidence frontier fraction was accepted")


def test_dynamic_config_and_parser_preserve_endpoint_quality_weight() -> None:
    DynamicLoopConfig(endpoint_canonical_quality_weight=0.0).validate()
    args = parse_dynamic_args(
        ["--endpoint-canonical-quality-weight", "0.0"]
    )
    assert args.endpoint_canonical_quality_weight == 0.0

    for invalid in (-0.01, 1.01):
        try:
            DynamicLoopConfig(
                endpoint_canonical_quality_weight=invalid
            ).validate()
        except ValueError as exc:
            assert "endpoint_canonical_quality_weight" in str(exc)
        else:
            raise AssertionError("invalid endpoint quality weight was accepted")


def test_dynamic_runner_passes_seed_as_frontier_replicate_index() -> None:
    assert "replicate_index=seed" in inspect.getsource(dynamic_runner.run)


def test_dynamic_generator_closure_rejects_future_index_capture() -> None:
    future_index = {"hidden": True}

    def leaking_generator():
        return future_index

    try:
        _audit_generator_closure(leaking_generator)
    except ValueError as exc:
        assert "hidden future outcomes" in str(exc)
    else:
        raise AssertionError("future-index closure capture was accepted")


def test_dynamic_generator_closure_accepts_frozen_graph_capture() -> None:
    graph = {"frozen": True}

    def frozen_generator():
        return graph

    assert _audit_generator_closure(frozen_generator) == ("graph",)


def _write_reusable_run(
    run_dir, *, config, profile, source_bundle, schema="neurodiscovery-dynamic-closed-loop-hindcasting.v4"
):
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": schema,
        "status": "complete",
        "seed": 0,
        "case_study_id": "biomarker_discovery",
        "freeze_year": 2016,
        "future_start_year": 2017,
        "future_end_year": 2021,
        "config": config.__dict__,
        "profile": profile.__dict__,
        "source_bundle": source_bundle,
        "executed_hypotheses": config.max_executions,
        "generation_failure_slots": 0,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hypotheses = [{"id": f"H{rank}"} for rank in range(config.max_executions)]
    (run_dir / "executed_hypotheses.json").write_text(
        json.dumps({"n_hypotheses": len(hypotheses), "hypotheses": hypotheses}),
        encoding="utf-8",
    )
    for name in (
        "feedback_overlay.csv",
        "generation_rounds.csv",
        "hidden_outcomes.csv",
        "proposed_hypotheses.jsonl.gz",
    ):
        (run_dir / name).write_bytes(b"x")
    with (run_dir / "metrics_by_k.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["requested_k", "generation_failure_slots"]
        )
        writer.writeheader()
        writer.writerow({"requested_k": config.max_executions, "generation_failure_slots": 0})
    return manifest


def test_dynamic_resume_accepts_only_exact_v4_checkpoint(tmp_path) -> None:
    config = DynamicLoopConfig(max_unique_proposals=1, max_executions=1)
    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    source_bundle = {"manifest_sha256": "ABC"}
    run_dir = tmp_path / "run"
    expected = _write_reusable_run(
        run_dir, config=config, profile=profile, source_bundle=source_bundle
    )

    reusable = _load_reusable_run(
        run_dir,
        seed=0,
        case_study_id="biomarker_discovery",
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        config=config,
        profile=profile,
        source_bundle=source_bundle,
        budgets=[1],
    )
    assert reusable is not None
    assert reusable[0] == expected


def test_dynamic_resume_rejects_stale_schema(tmp_path) -> None:
    config = DynamicLoopConfig(max_unique_proposals=1, max_executions=1)
    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    run_dir = tmp_path / "run"
    _write_reusable_run(
        run_dir,
        config=config,
        profile=profile,
        source_bundle=None,
        schema="neurodiscovery-dynamic-closed-loop-hindcasting.v3",
    )

    try:
        _load_reusable_run(
            run_dir,
            seed=0,
            case_study_id="biomarker_discovery",
            freeze_year=2016,
            future_start_year=2017,
            future_end_year=2021,
            config=config,
            profile=profile,
            source_bundle=None,
            budgets=[1],
        )
    except ValueError as exc:
        assert "stale or incompatible" in str(exc)
    else:
        raise AssertionError("stale dynamic checkpoint was reused")


def _hypothesis(
    candidate_id: str,
    source: str,
    mediator: str,
    target: str,
    score: float,
) -> dict:
    return {
        "id": candidate_id,
        "hypothesis_type": "bridge",
        "source_id": source,
        "source_name": source,
        "target_id": target,
        "target_name": target,
        "path": [
            {
                "from_id": source,
                "from_name": source,
                "to_id": mediator,
                "to_name": mediator,
                "relation_type": "associated_with",
                "confidence": score,
            },
            {
                "from_id": mediator,
                "from_name": mediator,
                "to_id": target,
                "to_name": target,
                "relation_type": "predicts",
                "confidence": score,
            },
        ],
        "confidence_score": score,
        "evidence_score": score,
        "novelty_score": score,
        "testability_score": score,
        "composite_score": score,
        "metadata": {"domain_a": "gene", "domain_b": "disease"},
    }


def test_dynamic_loop_regenerates_after_feedback_without_terminal_leakage():
    feedback_counts: list[int] = []
    excluded_counts: list[int] = []
    rounds = {
        0: [_hypothesis("raw-1", "A", "M1", "B", 0.99)],
        1: [
            _hypothesis("raw-2", "C", "M2", "D", 0.95),
            _hypothesis("raw-3", "E", "M3", "F", 0.40),
        ],
        2: [
            _hypothesis("raw-duplicate", "A", "M1", "B", 0.99),
            _hypothesis("raw-4", "G", "M4", "H", 0.30),
        ],
    }

    def generate_round(
        round_index, _generation_seed, feedback_state, excluded_semantic_keys
    ):
        feedback_counts.append(len(feedback_state.records))
        excluded_counts.append(len(excluded_semantic_keys))
        return rounds.get(round_index, [])

    def score_candidate(candidate):
        years = {"A": 2017, "C": 2020}
        year = years.get(candidate["source_id"])
        return {
            "primary_hit": year is not None,
            "primary_year": year,
            "any_future_hit": year is not None,
            "first_future_year": year,
        }

    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    result = execute_dynamic_loop(
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        config=DynamicLoopConfig(
            proposal_batch_size=2,
            max_unique_proposals=4,
            max_generation_rounds=3,
            max_stagnant_generation_rounds=2,
            max_executions=4,
            execution_batch_size=1,
            refresh_every_executions=1,
            refill_below=0,
            feedback_years=2,
            feedback_start_budget=1,
            min_supported_before_feedback=1,
        ),
        profile=profile,
        generate_round=generate_round,
        score_candidate=score_candidate,
    )

    assert feedback_counts == [0, 1, 1]
    assert excluded_counts == [0, 1, 3]
    assert [row["feedback_status"] for row in result.hidden[:2]] == [
        "supported",
        "inconclusive",
    ]
    assert result.hidden[1]["terminal_primary_hit"]
    assert len(result.proposed) == 4
    assert len(result.executed) == 4
    assert len({row["id"] for row in result.executed}) == 4
    assert sum(row["status"] == "supported" for row in result.feedback) == 1


def test_dynamic_loop_does_not_refill_after_execution_budget_is_reached():
    generation_rounds: list[int] = []

    def generate_round(
        round_index, _generation_seed, _feedback_state, _excluded_semantic_keys
    ):
        generation_rounds.append(round_index)
        return [
            _hypothesis("raw-1", "A", "M1", "B", 0.9),
            _hypothesis("raw-2", "C", "M2", "D", 0.8),
        ]

    def score_candidate(_candidate):
        return {
            "primary_hit": False,
            "primary_year": None,
            "any_future_hit": False,
            "first_future_year": None,
        }

    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    result = execute_dynamic_loop(
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        config=DynamicLoopConfig(
            proposal_batch_size=2,
            max_unique_proposals=10,
            max_generation_rounds=5,
            max_executions=1,
            execution_batch_size=1,
            refresh_every_executions=1,
            refill_below=10,
            feedback_years=2,
            feedback_start_budget=1,
        ),
        profile=profile,
        generate_round=generate_round,
        score_candidate=score_candidate,
    )

    assert generation_rounds == [0]
    assert len(result.executed) == 1


def test_dynamic_loop_pads_exhausted_candidate_space_with_zero_credit_slots():
    def generate_round(
        round_index, _generation_seed, _feedback_state, _excluded_semantic_keys
    ):
        if round_index == 0:
            return [_hypothesis("raw-1", "A", "M1", "B", 0.9)]
        return []

    def score_candidate(_candidate):
        return {
            "primary_hit": True,
            "primary_year": 2017,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "first_future_year": 2017,
        }

    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    result = execute_dynamic_loop(
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        config=DynamicLoopConfig(
            proposal_batch_size=1,
            max_unique_proposals=3,
            max_generation_rounds=2,
            max_stagnant_generation_rounds=1,
            max_executions=3,
            execution_batch_size=1,
            refresh_every_executions=1,
            refill_below=1,
            feedback_years=1,
            feedback_start_budget=1,
            min_supported_before_feedback=1,
        ),
        profile=profile,
        generate_round=generate_round,
        score_candidate=score_candidate,
    )

    assert result.stop_reason == "candidate_space_exhausted"
    assert result.generation_failure_slots == 2
    assert len(result.executed) == len(result.hidden) == 3
    assert [row["hypothesis_type"] for row in result.executed] == [
        "bridge",
        "generation_failure",
        "generation_failure",
    ]
    assert all(not row["primary_hit"] for row in result.hidden[1:])
    [metrics] = _metrics_rows(
        result=result,
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        budgets=[3],
        future_pair_total=1,
    )
    assert metrics["executed_hypotheses"] == 3
    assert metrics["generation_failure_slots"] == 2
    assert metrics["unique_primary_discoveries"] == 1


def test_dynamic_open_loop_withholds_early_outcomes_from_generation() -> None:
    feedback_counts: list[int] = []

    def generate_round(
        round_index, _generation_seed, feedback_state, _excluded_semantic_keys
    ):
        feedback_counts.append(len(feedback_state.records))
        return [
            _hypothesis(
                f"round-{round_index}",
                f"S{round_index}",
                f"M{round_index}",
                f"T{round_index}",
                0.9,
            )
        ]

    def score_candidate(_candidate):
        return {
            "primary_hit": True,
            "primary_year": 2017,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "first_future_year": 2017,
        }

    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    result = execute_dynamic_loop(
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2018,
        config=DynamicLoopConfig(
            feedback_enabled=False,
            proposal_batch_size=1,
            max_unique_proposals=3,
            max_generation_rounds=3,
            max_stagnant_generation_rounds=2,
            max_executions=3,
            execution_batch_size=1,
            refresh_every_executions=1,
            refill_below=1,
            feedback_years=1,
            feedback_start_budget=1,
            min_supported_before_feedback=1,
        ),
        profile=profile,
        generate_round=generate_round,
        score_candidate=score_candidate,
    )

    assert feedback_counts == [0, 0, 0]
    assert all(not row["available_to_generator"] for row in result.feedback)
    assert all(not row["feedback_active"] for row in result.generation_rounds)
    [metrics] = _metrics_rows(
        result=result,
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2018,
        budgets=[3],
        future_pair_total=1,
        method="neurodiscovery_dynamic_open_loop",
    )
    assert metrics["supported_feedback_records"] == 0
    assert metrics["withheld_supported_outcomes"] == 3


def test_dynamic_loop_preserves_generation_for_delayed_feedback():
    feedback_counts: list[int] = []

    def generate_round(
        round_index, _generation_seed, feedback_state, _excluded_semantic_keys
    ):
        feedback_counts.append(len(feedback_state.records))
        if round_index == 0:
            return [
                _hypothesis("raw-1", "A", "M1", "B", 0.99),
                _hypothesis("raw-2", "C", "M2", "D", 0.90),
                _hypothesis("raw-3", "E", "M3", "F", 0.80),
                _hypothesis("raw-4", "G", "M4", "H", 0.70),
            ]
        return [_hypothesis("feedback-derived", "I", "M5", "J", 0.95)]

    def score_candidate(candidate):
        year = 2017 if candidate["source_id"] == "E" else None
        return {
            "primary_hit": year is not None,
            "primary_year": year,
            "any_future_hit": year is not None,
            "first_future_year": year,
        }

    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    result = execute_dynamic_loop(
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2021,
        config=DynamicLoopConfig(
            proposal_batch_size=4,
            max_unique_proposals=10,
            max_generation_rounds=2,
            max_stagnant_generation_rounds=1,
            max_executions=5,
            execution_batch_size=1,
            refresh_every_executions=100,
            refill_below=10,
            feedback_years=2,
            feedback_start_budget=3,
            min_supported_before_feedback=1,
        ),
        profile=profile,
        generate_round=generate_round,
        score_candidate=score_candidate,
    )

    assert feedback_counts == [0, 1]
    assert result.generation_rounds[1]["feedback_active"] is True
    assert result.generation_rounds[1]["executed_before_refill"] == 3
    assert len(result.proposed) == 5


def test_dynamic_feedback_preserves_supported_input_not_shared_output() -> None:
    profile = next(
        item for item in profile_slate() if item.name == "legacy_closed_supported_gate"
    )
    candidates = [
        _hypothesis("source-variant", "A", "M2", "C", 0.7),
        _hypothesis("target-variant", "D", "M3", "B", 0.7),
        _hypothesis("unrelated", "X", "M4", "Y", 0.7),
        _hypothesis("exact-repeat", "A", "M5", "B", 0.7),
    ]
    feedback = FeedbackState(
        [
            FeedbackRecord.from_dict(
                {
                    "status": "supported",
                    "hypothesis_id": "supported",
                    "source_id": "A",
                    "target_id": "B",
                }
            )
        ]
    )

    public, scores = _rank_reservoir(
        candidates,
        feedback_state=feedback,
        feedback_active=True,
        profile=profile,
    )

    assert scores[0] > scores[1]
    assert scores[1] == scores[2]
    assert scores[3] < scores[2]
    assert public.loc[0, "feedback_source_affinity"] == 1.0
    assert public.loc[1, "feedback_source_affinity"] == 0.0
    assert bool(public.loc[3, "feedback_exact_pair"]) is True


def test_generator_composite_profile_preserves_generator_ranking() -> None:
    profile = next(
        item
        for item in profile_slate()
        if item.name == "generator_composite_closed_supported_gate"
    )
    lower = _hypothesis("lower", "A", "M1", "B", 0.9)
    higher = _hypothesis("higher", "C", "M2", "D", 0.2)
    lower["composite_score"] = 0.25
    higher["composite_score"] = 0.85

    public, scores = _rank_reservoir(
        [lower, higher],
        feedback_state=FeedbackState([]),
        feedback_active=False,
        profile=profile,
    )

    assert public["generator_composite_score"].tolist() == [0.25, 0.85]
    assert scores[1] > scores[0]


def test_dynamic_batch_enforces_cumulative_feedback_mutation_quota() -> None:
    candidates = [
        {
            **_hypothesis(f"mutation-{index}", f"M{index}", "Z", f"T{index}", 0.9),
            "metadata": {"feedback_mutation": True},
        }
        for index in range(8)
    ] + [
        _hypothesis(f"explore-{index}", f"E{index}", "Z", f"U{index}", 0.5)
        for index in range(8)
    ]
    public = pd.DataFrame(
        [
            {
                "source_id": candidate["source_id"],
                "target_id": candidate["target_id"],
                "relation_signature": "predicts",
                "mediator_id": "Z",
                "path_length_bucket": "2",
            }
            for candidate in candidates
        ]
    )
    scores = pd.Series(
        [0.99 - index * 0.01 for index in range(8)]
        + [0.50 - index * 0.01 for index in range(8)]
    ).to_numpy(float)

    selected = _select_dynamic_batch(
        candidates,
        scores,
        public,
        select_n=10,
        feedback_active=True,
        feedback_mutation_fraction=0.2,
        mutations_executed=0,
        executions_after_batch=10,
        diversity_penalty=0.0,
    )

    assert sum(
        bool((candidates[int(index)].get("metadata") or {}).get("feedback_mutation"))
        for index in selected
    ) == 2


def test_dynamic_metrics_count_unique_discoveries_and_recovered_pairs() -> None:
    hidden = [
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "early_primary_hit": True,
            "terminal_primary_hit": False,
            "terminal_any_hit": False,
        },
        {
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "early_primary_hit": True,
            "terminal_primary_hit": False,
            "terminal_any_hit": False,
        },
        {
            "primary_hit": True,
            "primary_discovery_key": "complete_path:A|C;C|D",
            "primary_recovered_pairs": "A|C;C|D",
            "any_future_hit": True,
            "early_primary_hit": False,
            "terminal_primary_hit": True,
            "terminal_any_hit": True,
        },
    ]
    result = DynamicLoopResult(
        proposed=[],
        executed=[],
        hidden=hidden,
        feedback=[],
        generation_rounds=[],
        stop_reason="execution_budget_reached",
        feedback_start_budget=1,
    )

    [row] = _metrics_rows(
        result=result,
        case_study_id="biomarker_discovery",
        seed=0,
        freeze_year=2016,
        future_start_year=2017,
        future_end_year=2018,
        budgets=[3],
        future_pair_total=10,
    )

    assert row["full_primary_hits"] == 3
    assert row["unique_primary_discoveries"] == 2
    assert row["early_unique_primary_discoveries"] == 1
    assert row["terminal_unique_primary_discoveries"] == 1
    assert row["recovered_future_pairs"] == 3
    assert row["future_pair_recall"] == 0.3


def test_relation_static_score_ignores_fixed_budget_failures():
    profile = next(
        item for item in profile_slate() if item.name == "relation_scoped_pair_static"
    )
    valid = {
        "candidate_id": "valid",
        "is_generation_failure": False,
        "is_semantic_duplicate": False,
        **{field: float(index + 1) for index, field in enumerate(RELATION_COMPONENT_RAW_FIELDS)},
    }
    failure = {
        "candidate_id": "failure",
        "is_generation_failure": True,
        "is_semantic_duplicate": False,
        **{field: float("nan") for field in RELATION_COMPONENT_RAW_FIELDS},
    }
    scores = _static_score(pd.DataFrame([valid, failure]), profile)
    assert scores[0] >= 0.0
    assert scores[1] == 0.0


def test_relation_regularized_score_is_legacy_dominant_and_ignores_failures():
    profile = next(
        item
        for item in profile_slate()
        if item.name == "legacy_relation_regularized_static"
    )
    high_legacy = {
        "candidate_id": "high-legacy",
        "is_generation_failure": False,
        "is_semantic_duplicate": False,
        "confidence_score": 0.9,
        "evidence_score": 0.9,
        "novelty_score": 0.9,
        "testability_score": 0.9,
        **{field: 0.0 for field in RELATION_COMPONENT_RAW_FIELDS},
    }
    high_relation = {
        "candidate_id": "high-relation",
        "is_generation_failure": False,
        "is_semantic_duplicate": False,
        "confidence_score": 0.1,
        "evidence_score": 0.1,
        "novelty_score": 0.1,
        "testability_score": 0.1,
        **{field: 1.0 for field in RELATION_COMPONENT_RAW_FIELDS},
    }
    failure = {
        **high_relation,
        "candidate_id": "failure",
        "is_generation_failure": True,
        **{field: float("nan") for field in RELATION_COMPONENT_RAW_FIELDS},
    }

    scores = _static_score(
        pd.DataFrame([high_legacy, high_relation, failure]), profile
    )

    assert scores[0] > scores[1]
    assert scores[1] > 0.0
    assert scores[2] == 0.0


def test_relation_regularized_score_uses_relation_support_as_tie_breaker():
    profile = next(
        item
        for item in profile_slate()
        if item.name == "legacy_relation_regularized_static"
    )
    common = {
        "is_generation_failure": False,
        "is_semantic_duplicate": False,
        "confidence_score": 0.7,
        "evidence_score": 0.7,
        "novelty_score": 0.7,
        "testability_score": 0.7,
    }
    weak = {
        **common,
        "candidate_id": "weak",
        **{field: 0.0 for field in RELATION_COMPONENT_RAW_FIELDS},
    }
    strong = {
        **common,
        "candidate_id": "strong",
        **{field: 1.0 for field in RELATION_COMPONENT_RAW_FIELDS},
    }

    scores = _static_score(pd.DataFrame([weak, strong]), profile)

    assert scores[1] > scores[0]


# Updated: 2026-08-12 14:53 HKT - validate the dynamic evidence-frontier mixture setting.
# Updated: 2026-08-12 20:34 HKT - verify directed input-side feedback ranking and exact-repeat suppression.
# Updated: 2026-08-12 21:02 HKT - verify cumulative execution quota for feedback-derived hypotheses.
# Updated: 2026-08-13 03:42 HKT - verify that dynamic execution can preserve the generator's final composite ranking.
