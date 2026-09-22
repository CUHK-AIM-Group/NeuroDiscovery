from __future__ import annotations

import json
from pathlib import Path

from neurooracle.scripts.freeze_dynamic_policy_application import (
    freeze_dynamic_policy_application,
)
from neurooracle.src.experiment_source_bundle import sha256_file


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def test_freeze_dynamic_policy_application_preserves_runtime_and_tracks_exposure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_policy.json"
    selected = {"feedback_enabled": True, "max_executions": 300}
    source.write_text(
        json.dumps(
            {
                "schema_version": "source.v1",
                "status": "frozen_for_confirmatory_application",
                "canonical_release": {
                    "knowledge_graph_sha256": "OLD_KG",
                    "claims_sha256": "OLD_CLAIMS",
                    "state_sha256": "OLD_STATE",
                },
                "development_protocol": {
                    "case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ]
                },
                "selected_policy": selected,
            }
        ),
        encoding="utf-8",
    )
    canonical = {}
    for name in ("knowledge_graph", "extracted_claims", "current_state"):
        path = tmp_path / name
        path.write_text(f"new-{name}", encoding="utf-8")
        canonical[name] = _record(path)
    static = tmp_path / "static.json"
    static.write_text(
        json.dumps(
            {
                "status": "frozen_before_formal_generation",
                "canonical_release": canonical,
            }
        ),
        encoding="utf-8",
    )

    result = freeze_dynamic_policy_application(
        source_policy_path=source,
        static_design_path=static,
        output_path=tmp_path / "application.json",
        workspace_root=tmp_path,
        additional_exposed_case_study_ids=["differential_diagnosis"],
    )

    assert result["selected_policy"] == selected
    assert result["development_protocol"]["case_study_ids"] == [
        "case1_transdiagnostic",
        "biomarker_discovery",
        "differential_diagnosis",
    ]
    assert result["policy_application"]["post_update_retuning"] is False
    assert result["canonical_release"]["knowledge_graph_sha256"] == sha256_file(
        tmp_path / "knowledge_graph"
    )
