from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

import core.scripts.run_supplemental_case_studies_formal as formal_runner
from core.scripts.canonical_kg_release import CURRENT_CANONICAL_SHA256
from core.scripts.case_study_closed_loop_specs import SUPPLEMENTAL_CASE_STUDIES
from core.scripts.run_supplemental_case_studies_formal import (
    DEFAULT_BRAINPILOT,
    DEFAULT_GATEWAY,
    Job,
    audit_closure,
    build_jobs,
    runtime_lock_check,
)


def test_default_gateway_does_not_collide_with_brainpilot_runtime_port() -> None:
    brainpilot_port = urlparse(DEFAULT_BRAINPILOT).port
    gateway_port = urlparse(DEFAULT_GATEWAY).port
    assert brainpilot_port == 18080
    assert gateway_port == 18082
    assert gateway_port not in {brainpilot_port, brainpilot_port + 1}


def test_runtime_lock_check_fails_closed_on_source_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        formal_runner,
        "load_and_verify_lock",
        lambda *args, **kwargs: ({}, ["locked source changed: example.py"]),
    )
    result = runtime_lock_check(
        lock_path=tmp_path / "lock.json",
        phase="development",
        stage="before_job",
        job_id="job",
    )
    assert result["ok"] is False
    assert result["errors"] == ["locked source changed: example.py"]


def test_development_job_matrix_uses_five_seeds_and_robustness(tmp_path: Path) -> None:
    jobs = build_jobs(
        phase="development",
        run_dir=tmp_path,
        cases=SUPPLEMENTAL_CASE_STUDIES,
        quick=False,
        include_robustness=True,
    )

    closure = [job for job in jobs if job.kind == "closure"]
    assert len(closure) == 7 * 5
    assert {job.seed for job in closure} == {
        20260826,
        20260827,
        20260828,
        20260829,
        20260830,
    }
    assert all("--cv-repeats" in job.command for job in closure)
    assert all(job.command[job.command.index("--cv-repeats") + 1] == "1" for job in closure)
    assert all(job.command[job.command.index("--benchmark-trials") + 1] == "5" for job in closure)

    robustness = [job for job in jobs if job.kind != "closure"]
    assert len(robustness) == 7
    differential = next(job for job in jobs if job.job_id == "robustness__differential_diagnosis")
    brain_age = next(job for job in jobs if job.job_id == "robustness__brain_age")
    assert differential.command[differential.command.index("--folds") + 1] == "5"
    assert brain_age.command[brain_age.command.index("--datasets") + 1] == "hcpya"
    assert brain_age.command[brain_age.command.index("--folds") + 1] == "5"


def test_final_job_matrix_uses_ten_seeds(tmp_path: Path) -> None:
    jobs = build_jobs(
        phase="final",
        run_dir=tmp_path,
        cases=("brain_age",),
        quick=False,
        include_robustness=False,
    )
    assert len(jobs) == 10
    assert len({job.seed for job in jobs}) == 10
    assert all(job.command[job.command.index("--benchmark-trials") + 1] == "10" for job in jobs)


def test_closure_audit_requires_latest_kg_external_and_five_folds(tmp_path: Path) -> None:
    output = tmp_path / "differential_diagnosis"
    tables = output / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame({"candidate_id": ["x"]}).to_csv(
        tables / "public_candidates.csv", index=False
    )
    pd.DataFrame({"candidate_id": ["x"], "n_folds": [5]}).to_csv(
        tables / "internal_outcomes.csv", index=False
    )
    pd.DataFrame({"candidate_id": ["x"], "executable": [True]}).to_csv(
        tables / "external_outcomes.csv", index=False
    )
    (tables / "table_manifest.json").write_text(
        json.dumps(
            {
                "kg_scoring": {
                    "kg_sha256": CURRENT_CANONICAL_SHA256["knowledge_graph"]
                },
                "external_executable": 1,
            }
        ),
        encoding="utf-8",
    )
    job = Job(
        job_id="job",
        task="differential_diagnosis",
        seed=1,
        kind="closure",
        output_dir=output,
        command=(),
    )

    result = audit_closure(job, expected_folds=5)
    assert result == {"job_id": "job", "ok": True, "errors": []}

    pd.DataFrame({"candidate_id": ["x"], "n_folds": [3]}).to_csv(
        tables / "internal_outcomes.csv", index=False
    )
    result = audit_closure(job, expected_folds=5)
    assert not result["ok"]
    assert "expected exactly 5 folds" in result["errors"][0]
