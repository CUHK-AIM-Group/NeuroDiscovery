from __future__ import annotations

import argparse
import json

import pytest

from neurooracle.scripts.evaluate_case_study_frozen_baselines import (
    _select_generation_runs,
    _validate_snapshot_root,
)
from neurooracle.scripts.run_case_study_hindcasting import parse_window


def test_select_generation_runs_preserves_sparse_executable_matrix() -> None:
    manifest = {
        "runs": [
            {
                "method": "neurodiscovery",
                "seed": 0,
                "case_study_id": "case1_transdiagnostic",
                "freeze_year": 2016,
                "future_start_year": 2017,
                "future_end_year": 2021,
            },
            {
                "method": "neurodiscovery",
                "seed": 0,
                "case_study_id": "case1_transdiagnostic",
                "freeze_year": 2017,
                "future_start_year": 2018,
                "future_end_year": 2022,
            },
            {
                "method": "neurodiscovery",
                "seed": 0,
                "case_study_id": "case2_pathway_mediation",
                "freeze_year": 2017,
                "future_start_year": 2018,
                "future_end_year": 2022,
            },
        ]
    }

    selected = _select_generation_runs(
        manifest,
        methods=["neurodiscovery"],
        seeds=[0],
        cases=["case1_transdiagnostic", "case2_pathway_mediation"],
        windows=[parse_window("2016:2017:2021"), parse_window("2017:2018:2022")],
    )

    assert [
        (row["freeze_year"], row["case_study_id"]) for row in selected
    ] == [
        (2016, "case1_transdiagnostic"),
        (2017, "case1_transdiagnostic"),
        (2017, "case2_pathway_mediation"),
    ]


def test_snapshot_root_validation_rejects_cross_snapshot_evaluation(tmp_path) -> None:
    generated = tmp_path / "generated_snapshot"
    requested = tmp_path / "requested_snapshot"
    generated.mkdir()
    requested.mkdir()

    try:
        _validate_snapshot_root(
            {"snapshot_root": str(generated)},
            requested,
        )
    except ValueError as exc:
        assert "snapshot mismatch" in str(exc)
    else:
        raise AssertionError("cross-snapshot evaluation must be rejected")


def test_snapshot_root_validation_accepts_declared_snapshot(tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    assert _validate_snapshot_root(
        {"snapshot_root": str(snapshot)},
        snapshot,
    ) == "matched_generation_manifest"


def test_evaluator_rejects_undeclared_ablation_method(tmp_path) -> None:
    from neurooracle.scripts.evaluate_case_study_frozen_baselines import run

    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / "generation_manifest.json").write_text(
        json.dumps(
            {
                "methods": ["neurodiscovery_quality_w010"],
                "case_studies": ["case1_transdiagnostic"],
                "seeds": [0],
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        generation_root=generation_root,
        snapshot_root=tmp_path,
        output_root=tmp_path / "out",
        future_claims=tmp_path / "claims.jsonl",
        case_study_ids=["case1_transdiagnostic"],
        methods=["not_declared"],
        seeds=[0],
        windows=[parse_window("2016:2017:2017")],
        top_k=[10],
        random_trials=1,
        force=False,
    )

    with pytest.raises(ValueError, match="not declared"):
        run(args)
