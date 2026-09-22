from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from neurooracle.scripts.verify_formal_dynamic_generalization import (
    _audit_execution_binding,
    _audit_fixed_budget_slots,
    _audit_hypothesis_temporal_fields,
    _expected_metric_rows,
    _split_runtime_contract,
    _verify_arm,
)
from neurooracle.src.experiment_source_bundle import sha256_file


def _hidden_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "H1",
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "early_primary_hit": False,
            "terminal_primary_hit": True,
            "terminal_any_hit": True,
        },
        {
            "candidate_id": "H2",
            "primary_hit": True,
            "primary_discovery_key": "endpoint:A|B",
            "primary_recovered_pairs": "A|B",
            "any_future_hit": True,
            "early_primary_hit": True,
            "terminal_primary_hit": False,
            "terminal_any_hit": False,
        },
        {
            "candidate_id": "H3",
            "primary_hit": True,
            "primary_discovery_key": "endpoint:C|D",
            "primary_recovered_pairs": "C|D",
            "any_future_hit": True,
            "early_primary_hit": False,
            "terminal_primary_hit": True,
            "terminal_any_hit": True,
        },
    ]


def _write_proposals(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_expected_metrics_recompute_prefix_counts_and_unique_discoveries() -> None:
    rows = _expected_metric_rows(
        _hidden_rows(),
        budgets=[1, 2, 3],
        future_pair_total=4,
    )
    assert rows[1]["full_primary_hits"] == 1
    assert rows[2]["full_primary_hits"] == 2
    assert rows[2]["unique_primary_discoveries"] == 1
    assert rows[3]["unique_primary_discoveries"] == 2
    assert rows[3]["terminal_unique_primary_discoveries"] == 2
    assert rows[3]["future_pair_recall"] == 0.5


def test_fixed_budget_audit_accepts_ordered_zero_credit_failure_slots() -> None:
    hypotheses = [
        {"id": "H1", "hypothesis_type": "bridge", "path": [{"to_id": "B"}]},
        {
            "id": "FAILURE-2",
            "hypothesis_type": "generation_failure",
            "source_id": "",
            "target_id": "",
            "path": [],
            "metadata": {
                "generation_failure": True,
                "fixed_budget_slot": True,
                "execution_rank": 2,
            },
        },
    ]
    hidden = [
        {"candidate_id": "H1", "execution_rank": 1, "primary_hit": False},
        {
            "candidate_id": "FAILURE-2",
            "execution_rank": 2,
            "generation_failure": True,
            "feedback_status": "generation_failure",
            "primary_hit": False,
            "any_future_hit": False,
            "early_primary_hit": False,
            "terminal_primary_hit": False,
            "terminal_any_hit": False,
        },
    ]

    assert _audit_fixed_budget_slots(hypotheses, hidden) == 1


def test_fixed_budget_audit_rejects_failure_slot_with_hit_credit() -> None:
    hypotheses = [
        {
            "id": "FAILURE-1",
            "hypothesis_type": "generation_failure",
            "source_id": "",
            "target_id": "",
            "path": [],
            "metadata": {
                "generation_failure": True,
                "fixed_budget_slot": True,
                "execution_rank": 1,
            },
        }
    ]
    hidden = [
        {
            "candidate_id": "FAILURE-1",
            "execution_rank": 1,
            "generation_failure": True,
            "feedback_status": "generation_failure",
            "primary_hit": True,
        }
    ]

    with pytest.raises(ValueError, match="received hit credit"):
        _audit_fixed_budget_slots(hypotheses, hidden)


def test_proposal_audit_rejects_evidence_after_freeze_year(tmp_path: Path) -> None:
    path = tmp_path / "proposed_hypotheses.jsonl.gz"
    _write_proposals(
        path,
        [{"id": "H1", "path": [{"evidence": {"year": 2019}}]}],
    )
    with pytest.raises(ValueError, match="future evidence year"):
        _audit_hypothesis_temporal_fields(path, freeze_year=2018)


def test_proposal_audit_rejects_true_future_outcome_flag(tmp_path: Path) -> None:
    path = tmp_path / "proposed_hypotheses.jsonl.gz"
    _write_proposals(
        path,
        [{"id": "H1", "metadata": {"uses_future_outcomes": True}}],
    )
    with pytest.raises(ValueError, match="future outcome flag is true"):
        _audit_hypothesis_temporal_fields(path, freeze_year=2018)


def test_proposal_audit_accepts_frozen_or_earlier_evidence(tmp_path: Path) -> None:
    path = tmp_path / "proposed_hypotheses.jsonl.gz"
    _write_proposals(
        path,
        [
            {
                "id": "H1",
                "metadata": {"uses_future_outcomes": False, "freeze_year": 2018},
                "path": [{"evidence": {"year": 2017}}],
            }
        ],
    )
    count, digest = _audit_hypothesis_temporal_fields(path, freeze_year=2018)
    assert count == 1
    assert len(digest) == 64


def test_execution_binding_requires_exact_design_and_bundle(tmp_path: Path) -> None:
    design = tmp_path / "design.json"
    bundle = tmp_path / "bundle.json"
    design.write_text("{}\n", encoding="utf-8")
    bundle.write_text("{}\n", encoding="utf-8")
    execution = {
        "design_path": str(design),
        "design_sha256": sha256_file(design),
        "source_bundle_manifest": str(bundle),
        "source_bundle_manifest_sha256": sha256_file(bundle),
    }

    _audit_execution_binding(
        execution,
        design_path=design,
        bundle_manifest=bundle,
    )
    execution["design_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen-design hash mismatch"):
        _audit_execution_binding(
            execution,
            design_path=design,
            bundle_manifest=bundle,
        )


def test_runtime_contract_separates_manifest_profile_from_config() -> None:
    profile, config = _split_runtime_contract(
        {
            "profile": "legacy_closed_supported_gate",
            "max_executions": 300,
            "feedback_years": 1,
        }
    )

    assert profile == "legacy_closed_supported_gate"
    assert config == {"max_executions": 300, "feedback_years": 1}


def test_runtime_contract_requires_profile() -> None:
    with pytest.raises(ValueError, match="lacks a profile"):
        _split_runtime_contract({"max_executions": 300})


def test_verify_arm_accepts_separate_profile_contract(tmp_path: Path) -> None:
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "closed"
    root.mkdir()
    top = {
        "schema_version": "neurodiscovery-dynamic-closed-loop-hindcasting.v4",
        "profile": {"name": "legacy_closed_supported_gate"},
        "config": {
            "feedback_enabled": True,
            "max_executions": 300,
            "feedback_years": 1,
        },
        "budgets": [],
        "eligibility": {"manifest_path": str(eligibility)},
        "run_summaries": [],
        "runs": 0,
    }
    (root / "dynamic_closed_loop_manifest.json").write_text(
        json.dumps(top) + "\n", encoding="utf-8"
    )
    (root / "metrics_by_run.csv").write_text("seed\n", encoding="utf-8")

    observed = _verify_arm(
        root,
        arm="closed",
        expected_runs=set(),
        budgets=[],
        runtime_config={"max_executions": 300, "feedback_years": 1},
        expected_profile="legacy_closed_supported_gate",
        eligibility_manifest=eligibility.resolve(),
    )

    assert observed["runs"] == 0


def test_verify_arm_rejects_profile_drift(tmp_path: Path) -> None:
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "open"
    root.mkdir()
    top = {
        "schema_version": "neurodiscovery-dynamic-closed-loop-hindcasting.v4",
        "profile": {"name": "unexpected_profile"},
        "config": {
            "feedback_enabled": False,
            "max_executions": 300,
            "feedback_years": 1,
        },
        "budgets": [],
        "eligibility": {"manifest_path": str(eligibility)},
        "run_summaries": [],
        "runs": 0,
    }
    (root / "dynamic_closed_loop_manifest.json").write_text(
        json.dumps(top) + "\n", encoding="utf-8"
    )
    (root / "metrics_by_run.csv").write_text("seed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level profile differs"):
        _verify_arm(
            root,
            arm="open",
            expected_runs=set(),
            budgets=[],
            runtime_config={"max_executions": 300, "feedback_years": 1},
            expected_profile="legacy_closed_supported_gate",
            eligibility_manifest=eligibility.resolve(),
        )
