from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_dynamic_generalization_eligibility import (
    freeze_dynamic_generalization_eligibility,
)
from neurooracle.src.experiment_source_bundle import sha256_file
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


def _source_lock(tmp_path: Path) -> Path:
    matrix = tmp_path / "source.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,"
        "analysis_tier,terminal_unique_pairs\n"
        "case1_transdiagnostic,2016,2017,2021,primary,20\n"
        "biomarker_discovery,2016,2017,2021,primary,30\n"
        "connectome_behavior,2018,2019,2023,primary,15\n"
        "differential_diagnosis,2019,2020,2024,primary,25\n"
        "disease_subtyping,2020,2021,2025,exploratory,4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "locked_before_all_task_dynamic_evaluation",
                "feedback_years": 1,
                "min_early_support": 2,
                "min_terminal_pairs": 10,
                "canonical_release": {"release": "test"},
                "snapshot_root": str(tmp_path / "snapshots"),
                "selection_contract": {
                    "performance_columns_consumed": False,
                    "task_or_window_selection_after_method_scoring_permitted": False,
                },
                "primary_case_study_ids": [
                    "case1_transdiagnostic",
                    "biomarker_discovery",
                    "connectome_behavior",
                    "differential_diagnosis",
                ],
                "primary_windows": 4,
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": sha256_file(matrix),
                    "rows": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "status": "frozen_for_confirmatory_application",
                "development_protocol": {
                    "case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_freeze_excludes_only_declared_development_tasks(tmp_path: Path) -> None:
    manifest = freeze_dynamic_generalization_eligibility(
        dynamic_eligibility_manifest=_source_lock(tmp_path),
        frozen_policy_path=_policy(tmp_path),
        output_dir=tmp_path / "out",
    )
    loaded = load_locked_hindcasting_eligibility(
        Path(manifest["locked_matrix"]["path"]).parent
        / "dynamic_generalization_manifest.json"
    )
    assert loaded.primary_case_study_ids == (
        "connectome_behavior",
        "differential_diagnosis",
    )
    assert loaded.primary_windows == frozenset(
        {
            ("connectome_behavior", 2018, 2019, 2023),
            ("differential_diagnosis", 2019, 2020, 2024),
        }
    )
    with loaded.matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["analysis_tier"] == "primary" for row in rows)
    assert manifest["selection_contract"]["performance_columns_consumed"] is False


def test_freeze_rejects_source_selected_with_method_outcomes(tmp_path: Path) -> None:
    source = _source_lock(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["selection_contract"]["performance_columns_consumed"] = True
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="consumed method outcomes"):
        freeze_dynamic_generalization_eligibility(
            dynamic_eligibility_manifest=source,
            frozen_policy_path=_policy(tmp_path),
            output_dir=tmp_path / "out",
        )
