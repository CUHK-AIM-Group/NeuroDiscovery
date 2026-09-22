from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.summarize_hindcasting_replicates import (
    _holm_adjust_paired_rows,
    _paired_rows,
    _summary_rows,
    _write_csv,
)
from neurooracle.scripts.verify_formal_static_hindcasting import (
    audit_hypothesis_temporal_isolation,
    verify_scored_topk_payload,
    verify_hypothesis_payload,
    verify_summary_tables,
    verify_topk_payload,
)


def _write_hypotheses(path: Path, identifiers: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "id": identifier,
                        "hypothesis_type": "bridge",
                        "metadata": {},
                    }
                    for identifier in identifiers
                ]
            }
        ),
        encoding="utf-8",
    )


def test_hypothesis_payload_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.json"
    _write_hypotheses(path, ["same", "same"])

    with pytest.raises(ValueError, match="duplicate hypothesis id"):
        verify_hypothesis_payload(path, target=2, expected_pool=2)


def test_hypothesis_payload_counts_failures_only_in_execution_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps(
            {
                "hypotheses": [
                    {"id": "valid", "hypothesis_type": "bridge", "metadata": {}},
                    {
                        "id": "failure-prefix",
                        "hypothesis_type": "generation_failure",
                        "metadata": {},
                    },
                    {
                        "id": "failure-tail",
                        "hypothesis_type": "generation_failure",
                        "metadata": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    hypotheses, _digest, failures = verify_hypothesis_payload(
        path, target=2, expected_pool=3
    )

    assert len(hypotheses) == 3
    assert failures == 1


def test_temporal_isolation_accepts_frozen_evidence() -> None:
    counts = audit_hypothesis_temporal_isolation(
        [
            {
                "id": "valid",
                "metadata": {
                    "freeze_year": 2020,
                    "uses_future_outcomes": False,
                },
                "supporting_claims": [{"publication_year": 2019}],
            }
        ],
        freeze_year=2020,
        path=Path("valid.json"),
    )
    assert counts == {"hypotheses": 1, "flags": 1, "evidence_years": 1}


@pytest.mark.parametrize(
    ("hypothesis", "message"),
    [
        (
            {"supporting_claims": [{"year": 2021}]},
            "future evidence year 2021",
        ),
        (
            {"metadata": {"uses_future_outcomes": True}},
            "future outcome flag uses_future_outcomes is true",
        ),
        (
            {"metadata": {"freeze_year": 2019}},
            "embedded freeze_year 2019 differs",
        ),
    ],
)
def test_temporal_isolation_rejects_future_or_mismatched_records(
    hypothesis: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_hypothesis_temporal_isolation(
            [{"id": "invalid", **hypothesis}],
            freeze_year=2020,
            path=Path("invalid.json"),
        )


def _topk_entry(k: int, discoveries: int) -> dict[str, object]:
    return {
        "requested_experiment_slots": k,
        "executed_ranked_prefix_slots": k,
        "observed": {
            "n": k,
            "primary_hits": discoveries,
            "unique_primary_discoveries": discoveries,
            "recovered_future_pairs": discoveries,
            "endpoint_hits": discoveries,
            "unique_endpoint_discoveries": discoveries,
            "any_path_edge_hits": discoveries,
            "any_future_hits": discoveries,
            "primary_hit_rate": discoveries / k,
            "unique_primary_discovery_rate": discoveries / k,
            "future_pair_recall": discoveries / k,
            "endpoint_hit_rate": discoveries / k,
            "any_path_edge_hit_rate": discoveries / k,
            "any_future_hit_rate": discoveries / k,
            "mean_path_edge_hit_rate": discoveries / k,
        },
        "random_same_hypothesis_pool": {"applicable": True, "trials": 50},
    }


def test_topk_verifier_rejects_nonmonotonic_discoveries() -> None:
    payload = {
        "topk": {
            "10": _topk_entry(10, 3),
            "20": _topk_entry(20, 2),
        }
    }

    with pytest.raises(ValueError, match="decreases at K=20"):
        verify_topk_payload(
            payload,
            rank_points=(10, 20),
            candidate_pool_size=20,
            random_trials=50,
        )


def _scored_row(
    rank: int,
    *,
    primary: bool = False,
    endpoint: bool = False,
) -> dict[str, object]:
    return {
        "rank": rank,
        "primary_hit": primary,
        "primary_discovery_key": "primary-a" if primary else "",
        "primary_recovered_pairs": "left|right" if primary else "",
        "endpoint_hit": endpoint,
        "endpoint_discovery_key": "endpoint-a" if endpoint else "",
        "any_path_edge_hit": primary,
        "all_path_edges_hit": primary,
        "any_future_hit": primary or endpoint,
        "path_edge_hit_rate": 1.0 if primary else 0.0,
        "primary_lead_time": 2.0 if primary else "",
    }


def test_scored_topk_verifier_recomputes_observed_metrics() -> None:
    scored = [
        _scored_row(1, primary=True, endpoint=True),
        _scored_row(2),
        _scored_row(3),
    ]
    entry = {
        "requested_experiment_slots": 2,
        "executed_ranked_prefix_slots": 2,
        "observed": {
            "n": 2,
            "primary_hits": 1,
            "primary_hit_rate": 0.5,
            "unique_primary_discoveries": 1,
            "unique_primary_discovery_rate": 0.5,
            "recovered_future_pairs": 1,
            "future_pair_recall": 0.25,
            "endpoint_hits": 1,
            "endpoint_hit_rate": 0.5,
            "unique_endpoint_discoveries": 1,
            "any_path_edge_hits": 1,
            "any_path_edge_hit_rate": 0.5,
            "all_path_edges_hits": 1,
            "any_future_hits": 1,
            "any_future_hit_rate": 0.5,
            "mean_path_edge_hit_rate": 0.5,
            "mean_primary_lead_time": 2.0,
        },
        "random_same_hypothesis_pool": {
            "applicable": True,
            "trials": 2,
            "mean_primary_hits": 0.5,
            "mean_unique_primary_discoveries": 0.5,
            "mean_any_future_hits": 0.5,
            "mean_endpoint_hits": 0.5,
            "mean_unique_endpoint_discoveries": 0.5,
            "sd_primary_hits": 2**-0.5,
            "variance_unique_primary_discoveries": 0.5,
            "sd_any_future_hits": 2**-0.5,
            "sd_endpoint_hits": 2**-0.5,
            "ci95_primary_hits": [0.025, 0.975],
            "ci95_any_future_hits": [0.025, 0.975],
            "ci95_endpoint_hits": [0.025, 0.975],
            "p_primary_hits_ge_observed": 2 / 3,
            "p_unique_primary_discoveries_ge_observed": 2 / 3,
            "p_any_future_hits_ge_observed": 2 / 3,
            "p_endpoint_hits_ge_observed": 2 / 3,
        },
    }
    trials = [
        {
            "trial": 1,
            "k": 2,
            "primary_hits": 0,
            "unique_primary_discoveries": 0,
            "any_future_hits": 0,
            "endpoint_hits": 0,
            "unique_endpoint_discoveries": 0,
        },
        {
            "trial": 2,
            "k": 2,
            "primary_hits": 1,
            "unique_primary_discoveries": 1,
            "any_future_hits": 1,
            "endpoint_hits": 1,
            "unique_endpoint_discoveries": 1,
        },
    ]
    result = verify_scored_topk_payload(
        {"future_stats": {"future_unique_pairs": 4}, "topk": {"2": entry}},
        scored=scored,
        random_trial_rows=trials,
        rank_points=[2],
        random_trials=2,
    )
    assert result == {"rank_points_recomputed": 1, "random_trial_rows_recomputed": 2}

    entry["observed"]["primary_hits"] = 0
    with pytest.raises(ValueError, match="observed top-k metrics.*primary_hits"):
        verify_scored_topk_payload(
            {"future_stats": {"future_unique_pairs": 4}, "topk": {"2": entry}},
            scored=scored,
            random_trial_rows=trials,
            rank_points=[2],
            random_trials=2,
        )


def _observations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, values in {
        "neurodiscovery": (3.0, 5.0),
        "baseline": (1.0, 2.0),
    }.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "case_study_id": "case",
                    "freeze_year": 2016,
                    "future_start_year": 2017,
                    "future_end_year": 2021,
                    "k": 10,
                    "metric": "unique_primary_discoveries",
                    "value": value,
                    "benchmark_status": "executable",
                    "metrics_path": "metrics.json",
                }
            )
    return rows


def _write_valid_summary(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    _write_csv(root / "replicate_observations.csv", rows)
    _write_csv(root / "mean_variance_summary.csv", _summary_rows(rows))
    _write_csv(
        root / "paired_comparisons.csv",
        _holm_adjust_paired_rows(_paired_rows(rows, "neurodiscovery")),
    )


def test_summary_verifier_recomputes_and_rejects_tampering(tmp_path: Path) -> None:
    rows = _observations()
    summary = tmp_path / "summary"
    _write_valid_summary(summary, rows)
    counts = verify_summary_tables(
        summary,
        observations=rows,
        reference_method="neurodiscovery",
    )
    assert counts["observations"] == 4

    path = summary / "mean_variance_summary.csv"
    materialized = path.read_text(encoding="utf-8")
    path.write_text(materialized.replace(",4.0,2.0\n", ",99.0,2.0\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="summary mean differs"):
        verify_summary_tables(
            summary,
            observations=rows,
            reference_method="neurodiscovery",
        )
