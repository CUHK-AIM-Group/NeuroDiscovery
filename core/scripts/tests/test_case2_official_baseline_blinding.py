from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.scripts.case2_official_baseline_experiment import (
    audit_baseline_rankings,
    freeze_baseline_rankings,
    load_frozen_registry,
)
from core.scripts.case2_search_policy import (
    PUBLIC_COLUMNS,
    SCHEMA_VERSION,
    PolicyAnchor,
    SearchPolicy,
    build_public_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry() -> pd.DataFrame:
    rows = []
    for exposure in ("prs_a", "prs_b"):
        for marker in ("CENTILOIDS", "Hippocampus"):
            rows.append(
                {
                    "exposure": exposure,
                    "pathway_id": exposure,
                    "pathway_name": exposure,
                    "pathway_source": "NeuroOracle curated",
                    "threshold_label": "p1em03",
                    "gene_count": 10,
                    "modality": "amyloid_pet" if marker == "CENTILOIDS" else "smri",
                    "marker": marker,
                    "outcome": "FAQ",
                }
            )
    return build_public_registry(pd.DataFrame(rows))


def _write_frozen_registry(
    tmp_path: Path,
    *,
    status: str = "frozen_before_confirmation_association_access",
) -> tuple[Path, Path, pd.DataFrame]:
    registry = _registry()
    registry_path = tmp_path / "public_candidate_registry.csv"
    registry.loc[:, PUBLIC_COLUMNS].to_csv(registry_path, index=False)
    manifest = {
        "status": status,
        "freeze_id": "test-freeze",
        "lock_material": {
            "registry_sha256": _sha256(registry_path),
            "kg_snapshot": {
                "knowledge_graph": {"sha256": "kg-sha"},
                "extracted_claims": {"sha256": "claims-sha"},
            },
        },
    }
    manifest_path = tmp_path / "protocol_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return registry_path, manifest_path, registry


def test_baseline_registry_loader_reads_only_exact_public_schema(tmp_path: Path) -> None:
    registry_path, manifest_path, expected = _write_frozen_registry(tmp_path)
    observed, manifest = load_frozen_registry(registry_path, manifest_path)

    assert observed.equals(expected)
    assert manifest["freeze_id"] == "test-freeze"
    assert not any("sobel" in column for column in observed.columns)


def test_baseline_registry_loader_accepts_phase_heldout_freeze(tmp_path: Path) -> None:
    registry_path, manifest_path, expected = _write_frozen_registry(
        tmp_path,
        status="frozen_before_phase_heldout_association_access",
    )

    observed, manifest = load_frozen_registry(registry_path, manifest_path)

    assert observed.equals(expected)
    assert manifest["status"] == "frozen_before_phase_heldout_association_access"


def test_baseline_ranking_lock_contains_static_full_permutations(tmp_path: Path) -> None:
    registry_path, manifest_path, registry = _write_frozen_registry(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policies = [
        SearchPolicy(
            method="ai_scientist_v2",
            trial=trial,
            schema_version=SCHEMA_VERSION,
            anchors=(PolicyAnchor(registry.iloc[-1 - trial]["candidate_id"]),),
        )
        for trial in range(2)
    ]

    lock = freeze_baseline_rankings(
        registry,
        policies,
        out_dir=tmp_path,
        protocol_manifest=manifest,
        registry_path=registry_path,
    )
    archive = np.load(lock["orders_path"], allow_pickle=False)

    assert lock["association_results_accessed"] is False
    assert lock["temporal_freeze_year"] is None
    assert lock["kg_snapshot_sha256"] == "kg-sha"
    assert len(lock["orders"]) == 2
    for record in lock["orders"]:
        order = archive[record["array_key"]]
        assert sorted(order.tolist()) == list(range(len(registry)))


def test_ranking_audit_passes_distinct_blinded_policies(tmp_path: Path) -> None:
    registry = _registry()
    policies = [
        SearchPolicy(
            method="ai_scientist_v2",
            trial=0,
            schema_version=SCHEMA_VERSION,
            anchors=(PolicyAnchor(registry.iloc[0]["candidate_id"]),),
        ),
        SearchPolicy(
            method="open_coscientist",
            trial=0,
            schema_version=SCHEMA_VERSION,
            anchors=(PolicyAnchor(registry.iloc[-1]["candidate_id"]),),
        ),
    ]

    audit = audit_baseline_rankings(
        registry,
        policies,
        out_dir=tmp_path,
        expected_methods=["ai_scientist_v2", "open_coscientist"],
        expected_trials=[0],
        n_anchors=1,
    )

    assert audit["status"] == "passed"
    assert audit["association_results_accessed"] is False
    assert audit["expected_policy_count"] == 2


def test_ranking_audit_rejects_private_field_leakage(tmp_path: Path) -> None:
    registry = _registry()
    method = "ai_scientist_v2"
    trial_dir = tmp_path / method / "trial_00"
    trial_dir.mkdir(parents=True)
    (trial_dir / "task.json").write_text(
        json.dumps({"expected_a_sign": 1}), encoding="utf-8"
    )
    policy = SearchPolicy(
        method=method,
        trial=0,
        schema_version=SCHEMA_VERSION,
        anchors=(PolicyAnchor(registry.iloc[0]["candidate_id"]),),
    )

    audit = audit_baseline_rankings(
        registry,
        [policy],
        out_dir=tmp_path,
        expected_methods=[method],
        expected_trials=[0],
        n_anchors=1,
    )

    assert audit["status"] == "failed"
    assert any("private_field_leakage" in issue for issue in audit["critical_issues"])


# Last Updated At: 2026-08-16 15:23 HKT
