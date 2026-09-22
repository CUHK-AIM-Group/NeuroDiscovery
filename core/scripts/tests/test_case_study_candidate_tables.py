from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.scripts.case_study_candidate_tables import (
    _terms_for_value,
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
    stable_candidate_id,
)
from core.scripts.case_study_feedback_adapters import adapter_for


def test_runtime_factor_aliases_expand_to_scientific_kg_terms() -> None:
    assert "functional connectivity" in _terms_for_value("fc_edge_projection")
    assert "mild cognitive impairment" in _terms_for_value("mci_to_dementia")
    assert "apoe e4" in _terms_for_value("apoe_e4_dosage")
    assert "attention deficit hyperactivity disorder" in _terms_for_value("ADHD")
    assert "functional connectivity" in _terms_for_value("corr_node_degree_abs_top10")
    assert "default mode network" in _terms_for_value("DMN")


def test_imaging_genetics_treats_atlas_as_a_qualifier() -> None:
    adapter = adapter_for("imaging_genetics")

    assert adapter.relations[0].subject_fields == ("gene_pathway",)
    assert adapter.relations[0].object_fields == ("imaging_phenotype",)
    assert adapter.qualifier_fields == ("atlas", "model")


def test_stable_candidate_id_is_order_independent() -> None:
    left = stable_candidate_id("task", {"atlas": "aal", "model": "svm"})
    right = stable_candidate_id("task", {"model": "svm", "atlas": "aal"})
    changed = stable_candidate_id("task", {"atlas": "aal", "model": "ridge"})
    assert left == right
    assert left != changed


def test_kg_score_is_outcome_blind_and_scoped(tmp_path: Path) -> None:
    graph = tmp_path / "knowledge_graph.json"
    graph.write_text(
        json.dumps(
            {
                "metadata": {},
                "concepts": {
                    "D": {"preferred_name": "Schizophrenia"},
                    "R": {"preferred_name": "Anterior cingulate cortex"},
                    "U": {"preferred_name": "Unmatched region"},
                },
                "edges": [
                    {
                        "source_id": "D",
                        "target_id": "R",
                        "relation_type": "associated_with",
                        "confidence": 0.9,
                        "metadata": {
                            "claim_case_study_ids": ["biomarker_discovery"]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {"disease": "Schizophrenia", "anatomy": "Anterior cingulate cortex"},
            {"disease": "Schizophrenia", "anatomy": "Unmatched region"},
        ]
    )
    scored, audit = score_public_candidates(
        candidates,
        semantic_fields=("disease", "anatomy"),
        case_study_id="biomarker_discovery",
        kg_path=graph,
        seed=7,
    )
    assert scored.loc[0, "score_neurodiscovery"] > scored.loc[1, "score_neurodiscovery"]
    assert audit["scoped_semantic_edges"] == 1
    assert audit["matched_semantic_fields"] == 2
    assert audit["semantic_field_coverage"] == 1.0
    assert audit["uses_experimental_outcomes"] is False
    assert not any("validated" in column for column in scored.columns)


def test_anatomy_auxiliary_metadata_is_used_without_changing_identity(
    tmp_path: Path,
) -> None:
    graph = tmp_path / "knowledge_graph.json"
    graph.write_text(
        json.dumps(
            {
                "metadata": {},
                "concepts": {
                    "D": {"preferred_name": "Attention deficit hyperactivity disorder"},
                    "R": {"preferred_name": "Default mode network"},
                },
                "edges": [
                    {
                        "source_id": "D",
                        "target_id": "R",
                        "relation_type": "associated_with",
                        "confidence": 0.9,
                        "metadata": {"claim_case_study_ids": ["biomarker_discovery"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {"disease": "ADHD", "anatomy": "ROI_1", "network": "DMN"},
            {"disease": "ADHD", "anatomy": "ROI_2", "network": None},
        ]
    )
    scored, audit = score_public_candidates(
        candidates,
        semantic_fields=("disease", "anatomy"),
        case_study_id="biomarker_discovery",
        kg_path=graph,
        seed=7,
    )
    assert scored.loc[0, "kg_relation_scoped_pair_support"] > 0
    assert scored.loc[1, "kg_relation_scoped_pair_support"] == 0
    assert audit["requested_semantic_fields"] == ["disease", "anatomy"]
    assert audit["auxiliary_semantic_fields"] == ["network"]
    assert audit["relation_field_pairs"] == [
        {
            "subject_fields": ["disease"],
            "object_fields": ["anatomy", "network"],
        }
    ]


def test_export_bundle_keeps_outcomes_out_of_public_file(tmp_path: Path) -> None:
    public = attach_candidate_ids(
        pd.DataFrame(
            [
                {"disease": "A", "atlas": "x", "score_neurodiscovery": 0.8},
                {"disease": "B", "atlas": "x", "score_neurodiscovery": 0.2},
            ]
        ),
        task="biomarker_discovery",
        identity_fields=("disease", "atlas"),
    )
    internal = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "validated": [True, False],
            "effect_size": [0.4, 0.01],
        }
    )
    external = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"],
            "executable": [True, False],
            "validated": [True, False],
        }
    )
    manifest = export_table_bundle(
        task="biomarker_discovery",
        public=public,
        internal=internal,
        external=external,
        factor_fields=("disease", "atlas"),
        output_dir=tmp_path / "bundle",
        provenance={"source": "test"},
        kg_audit={"uses_experimental_outcomes": False},
    )
    persisted = pd.read_csv(manifest["files"]["public_candidates"]["path"])
    assert "validated" not in persisted
    assert "effect_size" not in persisted
    assert manifest["internal_validated"] == 1
    assert manifest["external_validated"] == 1
