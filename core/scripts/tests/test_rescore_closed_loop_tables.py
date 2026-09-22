from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.scripts import rescore_closed_loop_tables as module
from core.scripts.case_study_candidate_tables import stable_candidate_id


def test_resume_requires_all_three_canonical_hashes(tmp_path: Path) -> None:
    release = {
        "files": {
            "knowledge_graph": {"sha256": "kg"},
            "extracted_claims": {"sha256": "claims"},
            "current_state": {"sha256": "state"},
        }
    }
    path = tmp_path / "closure.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": "brain_age",
                "canonical_release": release,
            }
        ),
        encoding="utf-8",
    )
    assert module.closure_matches_release(path, release, task="brain_age")
    changed = json.loads(json.dumps(release))
    changed["files"]["extracted_claims"]["sha256"] = "other"
    assert not module.closure_matches_release(path, changed, task="brain_age")


def test_rescore_reuses_hidden_outcomes_without_model_refit(
    tmp_path: Path, monkeypatch
) -> None:
    task = "brain_age"
    task_dir = tmp_path / task
    tables = task_dir / "tables"
    tables.mkdir(parents=True)
    public_path = tables / "public_candidates.csv"
    internal_path = tables / "internal_outcomes.csv"
    external_path = tables / "external_outcomes.csv"
    factor_values = {
        "modality": "resting_state_fmri",
        "atlas": "toy",
        "feature_family": "fc_edge_projection",
        "model": "ridge",
    }
    candidate_id = stable_candidate_id(task, factor_values)
    pd.DataFrame(
        {
            **{key: [value] for key, value in factor_values.items()},
            "candidate_id": [candidate_id],
            "score_neurodiscovery": [0.1],
        }
    ).to_csv(public_path, index=False)
    pd.DataFrame({"candidate_id": [candidate_id], "validated": [True]}).to_csv(
        internal_path, index=False
    )
    pd.DataFrame(
        {"candidate_id": [candidate_id], "executable": [True], "validated": [True]}
    ).to_csv(external_path, index=False)
    closure_path = task_dir / "closure_manifest.json"
    closure_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "table_manifest": {
                    "status": "complete",
                    "factor_fields": ["modality", "atlas", "feature_family", "model"],
                    "kg_scoring": {
                        "case_study_id": task,
                        "semantic_fields": ["modality", "feature_family"],
                    },
                    "provenance": {"source": "test"},
                    "files": {
                        "public_candidates": {"path": str(public_path)},
                        "internal_outcomes": {"path": str(internal_path)},
                        "external_outcomes": {"path": str(external_path)},
                    },
                },
                "benchmark_manifest": {
                    "status": "complete",
                    "methods": ["random_walk", "neurodiscovery"],
                    "method_trial_counts": {"random_walk": 10, "neurodiscovery": 10},
                },
            }
        ),
        encoding="utf-8",
    )
    kg = tmp_path / "kg.json"
    kg.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_score(frame, **kwargs):
        captured["columns"] = list(frame.columns)
        return frame.assign(score_neurodiscovery=0.5), {"kg_sha256": "new"}

    def fake_export(**kwargs):
        captured["internal"] = kwargs["internal"].copy()
        captured["external"] = kwargs["external"].copy()
        captured["provenance"] = kwargs["provenance"]
        return {"status": "complete"}

    monkeypatch.setattr(module, "score_public_candidates", fake_score)
    monkeypatch.setattr(module, "export_table_bundle", fake_export)
    monkeypatch.setattr(
        module,
        "run_benchmark",
        lambda args: {"status": "complete", "methods": args.methods},
    )
    module.rescore_task(tmp_path, task, kg)
    assert captured["columns"] == [
        "modality",
        "atlas",
        "feature_family",
        "model",
        "candidate_id",
    ]
    assert captured["internal"].loc[0, "validated"]
    assert captured["external"].loc[0, "validated"]
    assert captured["provenance"]["rescoring"]["models_refit"] is False


def test_biomarker_refresh_uses_legacy_table_closure_and_frozen_baselines(
    tmp_path: Path, monkeypatch
) -> None:
    task = "biomarker_discovery"
    tables = tmp_path / task / "tables"
    tables.mkdir(parents=True)
    factors = {
        "disease": "ADHD",
        "atlas": "Schaefer-400",
        "roi_index": 17,
        "anatomy": "Default network",
        "feature_family": "ALFF",
        "feature": "roi_alff_proxy",
    }
    candidate_id = stable_candidate_id(task, factors)
    public_path = tables / "public_candidates.csv"
    internal_path = tables / "internal_outcomes.csv"
    external_path = tables / "external_outcomes.csv"
    pd.DataFrame(
        {**{key: [value] for key, value in factors.items()}, "candidate_id": [candidate_id]}
    ).to_csv(public_path, index=False)
    pd.DataFrame({"candidate_id": [candidate_id], "validated": [True]}).to_csv(
        internal_path, index=False
    )
    pd.DataFrame(
        {"candidate_id": [candidate_id], "executable": [True], "validated": [True]}
    ).to_csv(external_path, index=False)
    frozen_records = [
        {
            "array_key": f"order_{trial:03d}",
            "method": "official_baseline",
            "trial": trial,
        }
        for trial in range(10)
    ]
    (tables / "biomarker_closure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "case-study-table-bundle.v1",
                "status": "complete",
                "task": task,
                "factor_fields": list(factors),
                "kg_scoring": {"case_study_id": "case1_transdiagnostic"},
                "provenance": {},
                "files": {
                    "public_candidates": {"path": str(public_path)},
                    "internal_outcomes": {"path": str(internal_path)},
                    "external_outcomes": {"path": str(external_path)},
                },
                "source_ranking_manifest": {
                    "external_data_read_before_freeze": False,
                    "orders": {"records": frozen_records},
                },
            }
        ),
        encoding="utf-8",
    )
    kg = tmp_path / "kg.json"
    kg.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_score(frame, **kwargs):
        captured["score_kwargs"] = kwargs
        return frame.assign(score_neurodiscovery=0.5), {"kg_sha256": "new"}

    monkeypatch.setattr(module, "score_public_candidates", fake_score)
    monkeypatch.setattr(
        module,
        "export_table_bundle",
        lambda **kwargs: {"status": "complete", "files": {}},
    )

    def fake_benchmark(args):
        captured["benchmark_args"] = args
        return {"status": "complete", "methods": args.methods}

    monkeypatch.setattr(module, "run_benchmark", fake_benchmark)
    output_root = tmp_path / "out"
    module.rescore_task(
        tmp_path,
        task,
        kg,
        output_root=output_root,
        policy_run_name="missing-policy-run",
    )

    score_kwargs = captured["score_kwargs"]
    assert score_kwargs["case_study_id"] == task
    assert tuple(score_kwargs["semantic_fields"]) == tuple(
        field for field in factors if field != "roi_index"
    )
    benchmark_args = captured["benchmark_args"]
    assert benchmark_args.precomputed_rankings_manifest.is_file()
    assert benchmark_args.methods == [
        "random_walk",
        "official_baseline",
        "neurodiscovery",
    ]
