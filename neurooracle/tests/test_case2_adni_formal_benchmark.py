from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.scripts.case2_search_policy import PolicyAnchor, SCHEMA_VERSION, SearchPolicy, build_public_registry
from neurooracle.scripts.case2_formal_deepseek_gateway import _FixedRouter
from neurooracle.scripts.evaluate_case2_adni_formal_benchmark import (
    evaluate_order,
    exact_paired_sign_flip_greater,
    normalized_discounted_cumulative_gain,
    paired_primary_comparisons,
    validate_reference,
)
from neurooracle.scripts.prepare_case2_adni_formal_benchmark import (
    PUBLIC_COLUMNS,
    audit_generator_bundle,
    build_evaluator_reference,
    build_public_registry as build_frozen_public_registry,
    derive_trial_seed,
    load_json,
    validate_config,
)
from neurooracle.scripts.run_case2_adni_formal_benchmark import (
    INVALID_SLOT_PREFIX,
    compile_experiment_stream,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "neurooracle" / "configs" / "case2_adni_formal_benchmark_v1.json"


def _candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pathway in range(7):
        for marker in range(8):
            for outcome in range(3):
                exposure = f"score_{pathway}"
                modality = "pet" if marker < 4 else "smri"
                marker_name = f"marker_{marker}"
                outcome_name = ("ADAS13", "FAQ", "LDELTOTAL")[outcome]
                rows.append(
                    {
                        "candidate_id": f"{exposure}|{modality}|{marker_name}|{outcome_name}",
                        "exposure": exposure,
                        "pathway_id": f"pathway_{pathway}",
                        "pathway_name": f"Pathway {pathway}",
                        "pathway_source": "synthetic",
                        "threshold_label": "p1em03",
                        "gene_count": str(pathway + 2),
                        "modality": modality,
                        "marker": marker_name,
                        "outcome": outcome_name,
                    }
                )
    return rows


def _reference() -> pd.DataFrame:
    rows = sorted(_candidate_rows(), key=lambda row: row["candidate_id"])
    relevance = np.linspace(4.0, 0.01, len(rows))
    return pd.DataFrame(
        {
            "candidate_id": [row["candidate_id"] for row in rows],
            "bootstrap_weakest_link_evidence": relevance,
            "supplemental_family_fdr_hit": [index < 6 for index in range(len(rows))],
            "global_fdr_hit": [False] * len(rows),
            "nominal_bootstrap_hit": [index < 15 for index in range(len(rows))],
        }
    )


def test_protocol_and_seed_derivation_are_locked() -> None:
    config = load_json(CONFIG)
    validate_config(config)
    assert config["trials"]["seeds"] == [
        derive_trial_seed(config["benchmark_id"], trial) for trial in range(10)
    ]


def test_case2_gateway_forces_registered_generation_settings() -> None:
    class FakeRouter:
        def complete(self, **kwargs):
            return kwargs

    fixed = _FixedRouter(FakeRouter(), __import__("threading").Lock())
    result = fixed.complete(
        reasoning_effort="low",
        temperature=0.9,
        max_output_tokens=99999,
        messages=[{"role": "user", "content": "test"}],
    )
    assert result["reasoning_effort"] == "high"
    assert result["temperature"] == 0.0
    assert result["max_output_tokens"] == 8192


def test_public_registry_builder_uses_only_public_fields(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    score_manifest = tmp_path / "scores.csv"
    rows = _candidate_rows()
    master_columns = [
        "candidate_id", "score_name", "pathway_id", "pathway_name",
        "threshold_label", "modality", "marker", "outcome",
    ]
    with master.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "score_name": row["exposure"],
                    "pathway_id": row["pathway_id"],
                    "pathway_name": row["pathway_name"],
                    "threshold_label": row["threshold_label"],
                    "modality": row["modality"],
                    "marker": row["marker"],
                    "outcome": row["outcome"],
                }
            )
    with score_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["score_name", "pathway_source", "gene_count"])
        writer.writeheader()
        for pathway in range(7):
            writer.writerow(
                {
                    "score_name": f"score_{pathway}",
                    "pathway_source": "synthetic",
                    "gene_count": pathway + 2,
                }
            )
    public = build_frozen_public_registry(master, score_manifest)
    assert len(public) == 168
    assert tuple(public[0]) == PUBLIC_COLUMNS
    assert not any("p" in row for row in public if row not in PUBLIC_COLUMNS)


def test_evaluator_reference_uses_bootstrap_weakest_link(tmp_path: Path) -> None:
    reference = _reference()
    formal = tmp_path / "formal.csv"
    rows = []
    for index, candidate_id in enumerate(reference["candidate_id"]):
        rows.append(
            {
                "candidate_id": candidate_id,
                "a_path_hc3_p": 0.001,
                "b_path_hc3_p": 0.01,
                "indirect_bootstrap_p": 0.02 if index < 15 else 0.5,
                "indirect_bootstrap_q_family_8": 0.04 if index < 6 else 0.5,
                "indirect_bootstrap_q_global_168": 0.5,
                "analysis_status": "estimated",
            }
        )
    pd.DataFrame(rows).to_csv(formal, index=False)
    built, counts = build_evaluator_reference(formal, reference["candidate_id"].tolist())
    assert counts == {
        "nominal_bootstrap_hits": 15,
        "supplemental_family_fdr_hits": 6,
        "global_fdr_hits": 0,
    }
    assert np.isclose(float(built[0]["bootstrap_weakest_link_evidence"]), -np.log10(0.02))


def test_failed_slots_are_preserved_in_experiment_stream() -> None:
    registry = build_public_registry(pd.DataFrame(_candidate_rows()))
    anchors = tuple(
        PolicyAnchor(candidate_id=value, score=0.2)
        for value in registry["candidate_id"].head(10)
    )
    policy = SearchPolicy(
        method="ai_scientist_v2",
        trial=0,
        schema_version=SCHEMA_VERSION,
        anchors=anchors,
    )
    stream, failed = compile_experiment_stream(policy, registry, n_anchors=80)
    assert failed == 70
    assert len(stream) == 238
    assert sum(value.startswith(INVALID_SLOT_PREFIX) for value in stream) == 70
    real = [value for value in stream if not value.startswith(INVALID_SLOT_PREFIX)]
    assert len(real) == len(set(real)) == 168
    assert set(real) == set(registry["candidate_id"])


def test_ndcg_and_ranking_metrics_penalize_failed_slots() -> None:
    reference = validate_reference(_reference().astype({
        "supplemental_family_fdr_hit": str,
        "global_fdr_hit": str,
        "nominal_bootstrap_hit": str,
    }))
    perfect = reference.sort_values(
        "bootstrap_weakest_link_evidence", ascending=False, kind="stable"
    )["candidate_id"].tolist()
    perfect_summary, _ = evaluate_order(perfect, reference, budgets=[5, 10, 168])
    assert np.isclose(perfect_summary["bootstrap_weakest_link_evidence_ndcg"], 1.0)
    penalized = [
        *perfect[:10],
        *[f"{INVALID_SLOT_PREFIX}test_0_{index:03d}__" for index in range(70)],
        *perfect[10:],
    ]
    penalized_summary, budgets = evaluate_order(penalized, reference, budgets=[5, 10, 168])
    assert penalized_summary["failed_generation_slots"] == 70
    assert penalized_summary["bootstrap_weakest_link_evidence_ndcg"] < 1.0
    assert budgets[-1]["k"] == 168
    assert np.isclose(normalized_discounted_cumulative_gain([3, 2, 1]), 1.0)


def test_exact_pairwise_statistics_and_holm() -> None:
    rows = []
    baselines = [
        "ai_scientist_v2", "open_coscientist", "sciagents", "virtual_lab",
        "brainpilot_native", "biomni_native",
    ]
    for trial in range(10):
        rows.append({"method": "neurodiscovery", "trial": trial, "metric": 0.9})
        rows.extend({"method": method, "trial": trial, "metric": 0.7} for method in baselines)
    comparisons = paired_primary_comparisons(
        pd.DataFrame(rows),
        metric="metric",
        target="neurodiscovery",
        baselines=baselines,
        bootstrap_resamples=1000,
        bootstrap_seed=1,
    )
    assert np.isclose(exact_paired_sign_flip_greater(np.ones(10)), 1 / 1024)
    assert comparisons["supplemental_superiority_criterion_met"].all()
    assert comparisons["p_holm_six_baselines"].lt(0.05).all()


def test_generator_leakage_audit_fails_on_result_column(tmp_path: Path) -> None:
    generator = tmp_path / "generator_inputs"
    generator.mkdir()
    pd.DataFrame({"candidate_id": ["a"], "indirect_bootstrap_p": [0.1]}).to_csv(
        generator / "bad.csv", index=False
    )
    config = load_json(CONFIG)
    audit = audit_generator_bundle(generator, config)
    assert audit["status"] == "failed"
    assert any("sensitive_csv_columns" in finding for finding in audit["findings"])
