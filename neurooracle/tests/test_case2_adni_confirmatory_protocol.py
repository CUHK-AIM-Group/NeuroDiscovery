from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import (
    DEFAULT_PROTOCOL,
    REPO_ROOT,
    build_confirmation_registry,
    build_private_replication_registry,
    load_and_validate_protocol,
)


PHASE_PROTOCOL = (
    REPO_ROOT / "neurooracle" / "configs" / "case2_adni_phase_holdout_v3.json"
)


def _subjects_and_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subject: dict[str, object] = {"subject_id": "S1"}
    for index in range(7):
        pathway_id = f"curated__pathway_{index}"
        threshold = "p1em03" if index < 5 else "p5em02"
        score_name = f"pathway_prs__curated__pathway_{index}__{threshold}"
        subject[score_name] = float(index)
        rows.append(
            {
                "score_name": score_name,
                "pathway_id": pathway_id,
                "pathway_source": "NeuroOracle curated",
                "pathway_name": f"Pathway {index}",
                "threshold_label": threshold,
                "gene_count": index + 3,
            }
        )
    return pd.DataFrame([subject]), pd.DataFrame(rows)


def test_protocol_distinguishes_snapshot_pinning_from_temporal_freeze() -> None:
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL)

    assert protocol["freeze_semantics"]["temporal_kg_freeze"] is False
    assert protocol["freeze_semantics"]["kg_snapshot_pinning"] is True
    assert protocol["freeze_semantics"]["initial_ranking_freeze"] is True
    assert protocol["freeze_semantics"]["sequential_batch_commit"] is True
    assert protocol["statistics"]["family_group_columns"] == ["exposure", "outcome"]
    assert protocol["statistics"]["family_size_expected"] == 10


def test_protocol_rejects_development_confirmation_overlap(tmp_path: Path) -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    broken = copy.deepcopy(protocol)
    broken["confirmation_data"]["outcomes"][0] = "MMSE"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        load_and_validate_protocol(path)


def test_phase_protocol_allows_endpoint_overlap_only_across_disjoint_phases() -> None:
    protocol = load_and_validate_protocol(PHASE_PROTOCOL)

    assert protocol["cohort"]["holdout_axis"] == "cohort_phase"
    assert set(protocol["development_data"]["outcomes_previously_accessed"]).issuperset(
        protocol["confirmation_data"]["outcomes"]
    )
    assert not set(protocol["cohort"]["development_values"]).intersection(
        protocol["cohort"]["confirmation_values"]
    )
    assert protocol["generator_evaluation"]["private_labels_visible"] is False


def test_phase_protocol_rejects_phase_overlap(tmp_path: Path) -> None:
    protocol = json.loads(PHASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["cohort"]["confirmation_values"][0] = "ADNI3"
    protocol["cohort"]["include_values"][0] = "ADNI3"
    path = tmp_path / "broken_phase.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="phases overlap"):
        load_and_validate_protocol(path)


def test_public_registry_is_complete_and_outcome_blind() -> None:
    protocol = load_and_validate_protocol(DEFAULT_PROTOCOL)
    subjects, manifest = _subjects_and_manifest()

    registry, exposures = build_confirmation_registry(subjects, manifest, protocol)

    assert len(exposures) == 7
    assert len(registry) == 210
    assert registry["candidate_id"].is_unique
    assert set(registry["outcome"]) == {"mPACCdigit", "FAQ", "LDELTOTAL"}
    assert registry[["modality", "marker"]].drop_duplicates().shape[0] == 10
    forbidden = {"sobel_p", "sobel_q_global", "effect", "rank", "hit"}
    assert not forbidden.intersection(registry.columns)


def test_private_replication_registry_locks_ids_and_signs_without_magnitudes() -> None:
    protocol = load_and_validate_protocol(PHASE_PROTOCOL)
    subjects, manifest = _subjects_and_manifest()
    public, _ = build_confirmation_registry(subjects, manifest, protocol)
    selected = public.iloc[:4].copy()
    selected["nominal_chain_hit"] = True
    selected["a_path_std"] = [0.1, -0.2, 0.3, -0.4]
    selected["a_path_p"] = 0.01
    selected["b_path_std"] = [-0.2, -0.3, 0.4, 0.5]
    selected["b_path_p"] = 0.02
    selected["indirect_effect_std"] = [-0.02, 0.06, 0.12, -0.20]
    selected["sobel_p"] = 0.03

    private = build_private_replication_registry(selected, public, protocol)

    assert len(private) == 4
    assert set(private["expected_a_sign"]) == {-1, 1}
    assert set(private["expected_b_sign"]) == {-1, 1}
    assert set(private["expected_indirect_sign"]) == {-1, 1}
    forbidden_tokens = ("effect", "beta", "p_value", "pvalue", "_p", "_q", "fdr")
    assert not any(
        token in column.lower()
        for column in private.columns
        for token in forbidden_tokens
    )


# Last Updated At: 2026-08-16 12:41 HKT
