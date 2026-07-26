from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.aggregate_case3_taskwise_hindcasting import build_coverage_rows
from neurooracle.scripts.collect_case3_task_literature import _default_tasks
from neurooracle.scripts.run_case3_taskwise_hindcasting import _command_for_subtask
from neurooracle.src.case3_subtasks import (
    CASE3_SUBTASK_BY_NAME,
    CASE3_SUBTASKS,
    case3_subtasks,
    select_case3_subtasks,
)


def test_case3_registry_contains_all_non_case_specific_subtasks() -> None:
    assert len(case3_subtasks("task")) == 15
    assert len(CASE3_SUBTASKS) == 15
    assert "transdiagnostic_clustering" not in CASE3_SUBTASK_BY_NAME
    assert "pathway_polygenic_mediation" not in CASE3_SUBTASK_BY_NAME
    assert "genetic_imaging_disease" not in CASE3_SUBTASK_BY_NAME
    assert _default_tasks() == [subtask.name for subtask in case3_subtasks("task")]
    with pytest.raises(KeyError):
        select_case3_subtasks(["transdiagnostic_clustering"])


def test_case3_subtask_selection_preserves_registry_order_and_exclusions() -> None:
    selected = select_case3_subtasks(
        ["prognosis", "brain_age", "prognosis"],
        exclude=["prognosis"],
    )
    assert [subtask.name for subtask in selected] == ["brain_age"]
    with pytest.raises(KeyError):
        select_case3_subtasks(["not_a_task"])


def test_runner_disables_chain_generation_for_every_case3_task(tmp_path: Path) -> None:
    common = {
        "input_dir": tmp_path / "input",
        "snapshot_root": tmp_path / "snapshots",
        "output_root": tmp_path / "outputs",
        "target_per_task": 1000,
        "random_trials": 1000,
        "seed": 10,
        "force": False,
    }
    flat = _command_for_subtask(subtask=CASE3_SUBTASK_BY_NAME["prognosis"], **common)

    assert flat[flat.index("--tasks") + 1] == "prognosis"
    assert flat[flat.index("--chains") + 1] == ""


def test_coverage_reports_complete_partial_failed_and_missing(tmp_path: Path) -> None:
    manifest = {
        "windows": [{"freeze_year": year} for year in range(2016, 2021)],
        "subtasks": [
            CASE3_SUBTASK_BY_NAME[name].to_dict()
            for name in ("prognosis", "brain_age", "cognitive_decoding", "imaging_genetics")
        ],
        "subtask_runs": [
            {"subtask": "prognosis", "status": "completed", "returncode": 0},
            {"subtask": "brain_age", "status": "completed", "returncode": 0},
            {"subtask": "cognitive_decoding", "status": "failed", "returncode": 1},
        ],
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metrics = [
        {"task": "prognosis", "freeze_year": year} for year in range(2016, 2021)
    ] + [
        {"task": "brain_age", "freeze_year": 2016},
        {"task": "brain_age", "freeze_year": 2017},
    ]

    rows = {row["task"]: row for row in build_coverage_rows(tmp_path, metrics)}
    assert rows["prognosis"]["status"] == "completed"
    assert rows["brain_age"]["status"] == "partial"
    assert rows["cognitive_decoding"]["status"] == "failed"
    assert rows["imaging_genetics"]["status"] == "missing"
