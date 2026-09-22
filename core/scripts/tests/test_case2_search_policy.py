from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts.case2_method_comparison import (
    _attach_candidate_ids,
    _build_neurodiscovery_prior,
    _family_chain_labels,
    audit_all_policy_trials,
    compile_blinded_orders,
    compile_closed_loop_orders,
    evaluate_orders,
    paired_method_comparisons,
)
from core.scripts.case2_search_policy import (
    SCHEMA_VERSION,
    PolicyAnchor,
    SearchPolicy,
    build_public_registry,
    compile_policy_order,
)
from core.scripts.case_study_native_output import validate_ranked_hypotheses
from core.scripts.case_study_official_adapter_client import compile_native_policy


def candidate_frame() -> pd.DataFrame:
    rows = []
    rank = 1
    for pathway in ("curated__amyloid", "curated__immune"):
        for marker in ("CENTILOIDS", "Hippocampus"):
            for outcome in ("MMSE", "CDRSB"):
                rows.append(
                    {
                        "kg_rank": rank,
                        "exposure": f"pathway_prs__{pathway}__p1em03",
                        "pathway_id": pathway,
                        "pathway_name": pathway.replace("curated__", ""),
                        "pathway_source": "NeuroOracle curated",
                        "threshold_label": "p1em03",
                        "gene_count": 12,
                        "modality": (
                            "amyloid_pet"
                            if marker == "CENTILOIDS"
                            else "smri_adnimerge"
                        ),
                        "marker": marker,
                        "outcome": outcome,
                        "sobel_p": 0.001 if rank in {1, 3} else 0.5,
                        "a_path_p": 0.001 if rank in {1, 3} else 0.5,
                        "b_path_p": 0.001 if rank in {1, 3} else 0.5,
                        "mapping_score": 1.0 - rank / 100,
                    }
                )
                rank += 1
    return pd.DataFrame(rows)


def test_case2_public_registry_is_result_blind_and_lexical() -> None:
    registry = build_public_registry(candidate_frame())

    assert "sobel_p" not in registry
    assert "kg_rank" not in registry
    assert "mapping_score" not in registry
    assert registry["candidate_id"].tolist() == sorted(registry["candidate_id"])
    assert registry["candidate_id"].is_unique


def test_case2_policy_compiles_to_deterministic_full_order() -> None:
    registry = build_public_registry(candidate_frame())
    anchor_id = registry.iloc[-1]["candidate_id"]
    policy = SearchPolicy(
        method="test",
        trial=0,
        schema_version=SCHEMA_VERSION,
        anchors=(PolicyAnchor(anchor_id, score=0.8),),
    )

    order = compile_policy_order(registry, policy)
    ranked_ids = registry.iloc[order]["candidate_id"].tolist()

    assert ranked_ids[0] == anchor_id
    assert len(order) == len(registry)
    assert len(np.unique(order)) == len(registry)


def test_generic_official_compiler_accepts_case2_ids_only(tmp_path: Path) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    valid_id = registry.iloc[0]["candidate_id"]
    invalid_id = "pathway_prs__missing__p1em03|amyloid_pet|CENTILOIDS|MMSE"

    payload, audit = compile_native_policy(
        method="ai_scientist_v2",
        task={
            "case_study_id": "case2",
            "policy_schema_version": SCHEMA_VERSION,
            "candidate_id_template": "exposure|modality|marker|outcome",
            "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
            "trial": 2,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
        },
        native_result={
            "ideas": [
                {"Experiments": [f"{valid_id}: registered rationale"]},
                f"candidate_id: {invalid_id}",
                "amyloid burden should be studied in general",
            ]
        },
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert [anchor["candidate_id"] for anchor in payload["anchors"]] == [valid_id]
    assert audit["invalid_exact_mentions"] == [invalid_id]


def test_invalid_native_slots_are_not_repaired() -> None:
    registry = build_public_registry(candidate_frame())
    valid_id = registry.iloc[0]["candidate_id"]
    validated = validate_ranked_hypotheses(
        method="biomni_native",
        trial=0,
        payloads=[
            (
                1,
                3,
                {
                    "hypotheses": [
                        {
                            "rank": 1,
                            "candidate_id": valid_id,
                            "rationale": "Exact candidate.",
                            "confidence": 0.8,
                        },
                        {
                            "rank": 2,
                            "candidate_id": "free text",
                            "rationale": "Invalid candidate.",
                            "confidence": 0.8,
                        },
                    ]
                },
                None,
            )
        ],
        candidate_ids=set(registry["candidate_id"]),
    )

    assert validated["generated_rank"].tolist() == [1, 2, 3]
    assert validated["valid"].tolist() == [True, False, False]
    assert "invalid_candidate_id" in validated.loc[1, "validation_status"]
    assert "missing_rank" in validated.loc[2, "validation_status"]


def test_official_compiler_keeps_an_all_invalid_trial_as_zero_anchors(
    tmp_path: Path,
) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)

    payload, audit = compile_native_policy(
        method="virtual_lab",
        task={
            "case_study_id": "case2",
            "policy_schema_version": SCHEMA_VERSION,
            "candidate_id_template": "exposure|modality|marker|outcome",
            "trial": 0,
            "n_anchors": 80,
            "public_registry_path": str(registry_path),
        },
        native_result={"summary": "A broad prose-only hypothesis."},
    )

    assert payload["anchors"] == []
    assert audit["valid_unique_anchors"] == 0
    assert audit["requested_anchors"] == 80


def _public_coordinate_compiler_config() -> dict[str, object]:
    return {
        "enabled": True,
        "schema_version": "public_coordinate_aliases_v1",
        "applicable_methods": ["open_coscientist"],
        "match_fields": ["exposure", "marker", "outcome"],
        "registry_alias_fields": {"exposure": ["pathway_name"]},
        "aliases": {
            "marker": {"Hippocampus": ["hippocampal"]},
            "outcome": {"CDRSB": ["CDR-SB"]},
        },
    }


def test_public_coordinate_compiler_maps_atomic_native_prose(tmp_path: Path) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    immune_hippocampus_cdrsb = registry.loc[
        registry["candidate_id"].str.contains("curated__immune")
        & registry["marker"].eq("Hippocampus")
        & registry["outcome"].eq("CDRSB"),
        "candidate_id",
    ].item()
    amyloid_centiloids_mmse = registry.loc[
        registry["candidate_id"].str.contains("curated__amyloid")
        & registry["marker"].eq("CENTILOIDS")
        & registry["outcome"].eq("MMSE"),
        "candidate_id",
    ].item()

    payload, audit = compile_native_policy(
        method="open_coscientist",
        task={
            "case_study_id": "case2",
            "policy_schema_version": SCHEMA_VERSION,
            "candidate_id_template": "exposure|modality|marker|outcome",
            "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
            "trial": 0,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
            "native_coordinate_compiler": _public_coordinate_compiler_config(),
        },
        native_result={
            "hypotheses": [
                {
                    "text": (
                        "Prioritize an immune pathway PRS with bilateral hippocampal "
                        "volume and subsequent CDR-SB worsening."
                    )
                },
                {
                    "text": (
                        "Prioritize an amyloid pathway PRS with CENTILOIDS and future "
                        "MMSE decline."
                    )
                },
            ]
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [
        immune_hippocampus_cdrsb,
        amyloid_centiloids_mmse,
    ]
    assert audit["mapping_mode"] == "deterministic_exact_or_public_coordinate_aliases_v1"
    assert audit["coordinate_projection_attempts"] == 2
    assert audit["coordinate_projection_mapped"] == 2
    assert len(audit["coordinate_compiler_config_sha256"]) == 64


def test_public_coordinate_compiler_never_repairs_candidate_like_ids(
    tmp_path: Path,
) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    valid_id = registry.iloc[0]["candidate_id"]
    invalid_id = "pathway_prs__missing__p1em03|smri_adnimerge|Hippocampus|CDRSB"

    payload, audit = compile_native_policy(
        method="open_coscientist",
        task={
            "case_study_id": "case2",
            "policy_schema_version": SCHEMA_VERSION,
            "candidate_id_template": "exposure|modality|marker|outcome",
            "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
            "trial": 0,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
            "native_coordinate_compiler": _public_coordinate_compiler_config(),
        },
        native_result={
            "hypotheses": [
                {"text": f"{valid_id} immune hippocampal CDR-SB"},
                {"text": f"{invalid_id} immune hippocampal CDR-SB"},
            ]
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [valid_id]
    assert audit["invalid_exact_mentions"] == [invalid_id]
    assert audit["coordinate_projection_attempts"] == 0


def test_public_coordinate_compiler_rejects_alias_collisions(tmp_path: Path) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    config = _public_coordinate_compiler_config()
    config["aliases"] = {
        "marker": {
            "Hippocampus": ["shared imaging marker"],
            "CENTILOIDS": ["shared imaging marker"],
        }
    }

    with pytest.raises(ValueError, match="ambiguous public coordinate aliases"):
        compile_native_policy(
            method="open_coscientist",
            task={
                "case_study_id": "case2",
                "candidate_id_template": "exposure|modality|marker|outcome",
                "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
                "trial": 0,
                "n_anchors": 5,
                "public_registry_path": str(registry_path),
                "native_coordinate_compiler": config,
            },
            native_result={"hypotheses": [{"text": "shared imaging marker"}]},
        )


def test_public_coordinate_compiler_prefers_longer_specific_alias(tmp_path: Path) -> None:
    exposure = "pathway_prs__curated__immune__p1em03"
    rows = [
        {
            "candidate_id": f"{exposure}|smri_adnimerge|Entorhinal|FAQ",
            "exposure": exposure,
            "pathway_name": "immune",
            "modality": "smri_adnimerge",
            "marker": "Entorhinal",
            "outcome": "FAQ",
        },
        {
            "candidate_id": f"{exposure}|tau_pet|CTX_ENTORHINAL_SUVR|FAQ",
            "exposure": exposure,
            "pathway_name": "immune",
            "modality": "tau_pet",
            "marker": "CTX_ENTORHINAL_SUVR",
            "outcome": "FAQ",
        },
    ]
    registry_path = tmp_path / "registry.jsonl"
    pd.DataFrame(rows).to_json(registry_path, orient="records", lines=True)

    payload, audit = compile_native_policy(
        method="open_coscientist",
        task={
            "case_study_id": "case2",
            "candidate_id_template": "exposure|modality|marker|outcome",
            "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
            "trial": 0,
            "n_anchors": 1,
            "public_registry_path": str(registry_path),
            "native_coordinate_compiler": {
                "enabled": True,
                "schema_version": "public_coordinate_aliases_v1",
                "applicable_methods": ["open_coscientist"],
                "match_fields": ["exposure", "marker", "outcome"],
                "registry_alias_fields": {"exposure": ["pathway_name"]},
                "aliases": {},
            },
        },
        native_result={
            "hypotheses": [
                {"text": "immune PRS with CTX_ENTORHINAL_SUVR and later FAQ"}
            ]
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [
        rows[1]["candidate_id"]
    ]
    assert audit["coordinate_projection_records"][0]["matched_coordinates"]["marker"] == (
        "CTX_ENTORHINAL_SUVR"
    )


def test_case2_comparison_uses_frozen_policy_order() -> None:
    results = _attach_candidate_ids(candidate_frame())
    registry = build_public_registry(results)
    hit_ids = results.loc[results["sobel_p"].lt(0.01), "candidate_id"].tolist()
    policy = SearchPolicy(
        method="native",
        trial=0,
        schema_version=SCHEMA_VERSION,
        anchors=tuple(PolicyAnchor(candidate_id) for candidate_id in hit_ids),
    )
    nd_ids = results.sort_values("kg_rank")["candidate_id"].tolist()
    orders = compile_blinded_orders(registry, [policy], nd_ids, nd_trials=1)
    metrics, rankings = evaluate_orders(
        orders,
        results,
        k_values=(2, len(results)),
        alpha=0.05,
    )

    native_top2 = metrics[(metrics["method"] == "native") & (metrics["k"] == 2)]
    assert int(native_top2.iloc[0]["topk_fdr_chain_hits"]) == 2
    assert float(native_top2.iloc[0]["auprc_nominal_chain"]) == 1.0
    assert int(rankings["nominal_chain_hit"].sum()) == 4
    assert len(rankings) == 2 * len(results)


def test_case2_neurodiscovery_uses_complete_chain_overlay(tmp_path: Path) -> None:
    results = _attach_candidate_ids(candidate_frame())
    registry = build_public_registry(results)
    policy = SearchPolicy(
        method="native",
        trial=0,
        schema_version=SCHEMA_VERSION,
        anchors=(PolicyAnchor(registry.iloc[0]["candidate_id"]),),
    )

    orders, traces, overlays, hidden = compile_closed_loop_orders(
        registry,
        [policy],
        results,
        nd_trials=1,
        batch_size=2,
        seed=11,
        alpha=0.05,
        overlay_dir=tmp_path / "overlays",
    )

    assert len(orders[("neurodiscovery", 0)]) == len(registry)
    assert traces[0]["overlay_read"] is False
    assert any(row["overlay_read"] for row in traces[1:])
    assert set(hidden["feedback_status"]) <= {"supported", "inconclusive"}
    assert overlays[0]["feedback_consumed_during_ranking"] is True
    assert overlays[0]["semantic_assertions"] == 2 * len(registry)
    records = [
        json.loads(line)
        for line in Path(overlays[0]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["requires_complete_chain"] for record in records)
    assert all(len(record["assertions"]) == 2 for record in records)


def test_case2_feedback_uses_prespecified_family_fdr() -> None:
    results = candidate_frame()
    results["sobel_q_global"] = 0.5
    results["sobel_q_family"] = 0.5
    results.loc[0, "sobel_q_family"] = 0.01

    labels = _family_chain_labels(results, alpha=0.05)

    assert labels.tolist() == [True, False, False, False, False, False, False, False]


def test_case2_evidence_consensus_prior_is_outcome_blind() -> None:
    results = _attach_candidate_ids(candidate_frame())
    results["mapping_score"] = 0.5
    results["meaningful_support_count"] = np.arange(1, len(results) + 1)
    registry = build_public_registry(results)
    indexed = results.set_index("candidate_id", drop=False)

    score, audit = _build_neurodiscovery_prior(
        registry,
        indexed,
        prior_name="evidence_consensus_v2",
    )
    perturbed = results.copy()
    perturbed[["sobel_p", "a_path_p", "b_path_p"]] = 1.0 - perturbed[
        ["sobel_p", "a_path_p", "b_path_p"]
    ]
    repeated, _ = _build_neurodiscovery_prior(
        registry,
        perturbed.set_index("candidate_id", drop=False),
        prior_name="evidence_consensus_v2",
    )

    assert np.allclose(score, repeated)
    assert score[-1] > score[0]
    assert audit["experimental_columns_used"] == []
    assert audit["outcome_blind"] is True


def test_case2_policy_audit_covers_every_common_trial() -> None:
    registry = build_public_registry(candidate_frame())
    first_id = registry.iloc[0]["candidate_id"]
    last_id = registry.iloc[-1]["candidate_id"]
    policies = [
        SearchPolicy(
            method="method_a",
            trial=trial,
            schema_version=SCHEMA_VERSION,
            anchors=(PolicyAnchor(first_id),),
        )
        for trial in (0, 1)
    ]
    policies.extend(
        [
            SearchPolicy(
                method="method_b",
                trial=0,
                schema_version=SCHEMA_VERSION,
                anchors=(PolicyAnchor(last_id),),
            ),
            SearchPolicy(
                method="method_b",
                trial=1,
                schema_version=SCHEMA_VERSION,
                anchors=(PolicyAnchor(first_id),),
            ),
        ]
    )

    frame, summary = audit_all_policy_trials(registry, policies)

    assert sorted(frame["trial"].unique().tolist()) == [0, 1]
    assert summary["trials"] == [0, 1]
    assert summary["passed"] is False
    assert summary["collapsed_pairs"] == [
        {"trial": 1, "method_a": "method_a", "method_b": "method_b"}
    ]


def test_case2_paired_comparison_uses_cumulative_family_fdr_hits() -> None:
    metrics = pd.DataFrame(
        [
            {
                "method": method,
                "trial": trial,
                "k": k,
                "family_fdr_chain_hits": hits,
            }
            for method, values in {
                "baseline": {5: 1, 10: 4},
                "neurodiscovery": {5: 3, 10: 4},
            }.items()
            for trial in range(3)
            for k, hits in values.items()
        ]
    )

    result = paired_method_comparisons(metrics)

    top5 = result.loc[result["k"].eq(5)].iloc[0]
    assert top5["mean_difference"] == 2
    assert top5["p_neurodiscovery_greater_exact_sign_flip"] == 0.125
    assert not bool(top5["full_pool_sanity"])
    full = result.loc[result["k"].eq(10)].iloc[0]
    assert full["mean_difference"] == 0
    assert full["p_neurodiscovery_greater_exact_sign_flip"] == 1.0
    assert bool(full["full_pool_sanity"])


# Last Updated At: 2026-08-12 02:45 HKT
