from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from core.scripts.case_study_score_components import (
    SCORE_COMPONENT_SCHEMA,
    candidate_id_sha256,
    load_score_component_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, outcome_blind: bool = True) -> tuple[Path, Path]:
    table = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "score_kge": [0.8, 0.2],
            "score_novelty": [0.4, 0.9],
            "score_critic": [0.7, 0.3],
        }
    )
    table_path = tmp_path / "scores.csv"
    table.to_csv(table_path, index=False)
    manifest = {
        "schema_version": SCORE_COMPONENT_SCHEMA,
        "outcome_blind": outcome_blind,
        "frozen_before_experiment": True,
        "score_table_sha256": _sha256(table_path),
        "candidate_id_sha256": candidate_id_sha256(table["candidate_id"]),
        "components": {
            "kge": {
                "column": "score_kge",
                "checkpoint_sha256": "kge-checkpoint-sha",
                "kg_snapshot_sha256": "formal-kg-sha",
                "kg_snapshot_status": "formal_release",
            },
            "novelty": {
                "column": "score_novelty",
                "definition": "high when plausible but weakly attested",
            },
            "critic": {
                "column": "score_critic",
                "model": "frozen-review-model",
                "prompt_sha256": "critic-prompt-sha",
            },
        },
    }
    manifest_path = tmp_path / "scores.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return table_path, manifest_path


def test_frozen_score_bundle_merges_by_candidate_id(tmp_path: Path) -> None:
    public = pd.DataFrame(
        {
            "candidate_id": ["c2", "c1"],
            "score_neurodiscovery": [0.1, 0.9],
        }
    )
    table_path, manifest_path = _bundle(tmp_path)

    merged, audit = load_score_component_bundle(
        public,
        table_path=table_path,
        manifest_path=manifest_path,
    )

    assert merged["score_kge"].tolist() == [0.2, 0.8]
    assert audit["columns"] == ["score_kge", "score_novelty", "score_critic"]
    assert audit["outcome_blind"] is True


def test_score_bundle_rejects_outcome_aware_components(tmp_path: Path) -> None:
    public = pd.DataFrame(
        {"candidate_id": ["c1", "c2"], "score_neurodiscovery": [0.9, 0.1]}
    )
    table_path, manifest_path = _bundle(tmp_path, outcome_blind=False)

    with pytest.raises(ValueError, match="outcome blind"):
        load_score_component_bundle(
            public,
            table_path=table_path,
            manifest_path=manifest_path,
        )
