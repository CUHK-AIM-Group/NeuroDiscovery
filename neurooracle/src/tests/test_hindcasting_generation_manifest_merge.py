from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from neurooracle.scripts.merge_hindcasting_generation_manifests import (
    merge_generation_manifests,
)


def _write_source(
    root: Path,
    *,
    methods: list[str],
    seeds: tuple[int, ...] = (0, 1),
    omit_last: bool = False,
    target: int = 3,
    eligibility_manifest: Path,
    eligibility_manifest_sha256: str,
    eligibility_matrix_sha256: str,
) -> Path:
    root.mkdir(parents=True)
    runs = []
    for method in methods:
        for seed in seeds:
            if omit_last and seed == seeds[-1]:
                continue
            hypotheses_path = root / method / f"seed_{seed:02d}.json"
            hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
            hypotheses_path.write_text(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "id": f"{method}:{seed}:{rank}",
                                "hypothesis_type": (
                                    "generation_failure" if rank == target - 1 else "bridge"
                                ),
                            }
                            for rank in range(target)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "case_study_id": "case1_transdiagnostic",
                    "freeze_year": 2016,
                    "future_start_year": 2017,
                    "future_end_year": 2021,
                    "hypotheses_path": str(hypotheses_path),
                }
            )
    manifest = {
        "snapshot_root": str((root.parent / "snapshots").resolve()),
        "methods": methods,
        "seeds": list(seeds),
        "target_per_case_study": target,
        "eligibility": {
            "manifest_path": str(eligibility_manifest),
            "manifest_sha256": eligibility_manifest_sha256,
            "matrix_sha256": eligibility_matrix_sha256,
        },
        "runs": runs,
    }
    path = root / "generation_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_eligibility(root: Path) -> tuple[Path, str, str]:
    matrix = root / "eligibility.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,analysis_tier\n"
        "case1_transdiagnostic,2016,2017,2021,primary\n",
        encoding="utf-8",
    )
    matrix_hash = _sha256(matrix)
    manifest = root / "eligibility_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "primary_windows": 1,
                "primary_case_study_ids": ["case1_transdiagnostic"],
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": matrix_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, _sha256(manifest), matrix_hash


def test_merge_generation_manifests_verifies_pairing_and_slots(tmp_path: Path) -> None:
    eligibility, manifest_hash, matrix_hash = _write_eligibility(tmp_path)
    common = {
        "eligibility_manifest": eligibility,
        "eligibility_manifest_sha256": manifest_hash,
        "eligibility_matrix_sha256": matrix_hash,
    }
    baseline = _write_source(tmp_path / "baseline", methods=["sciagents"], **common)
    neuro = _write_source(tmp_path / "neuro", methods=["neurodiscovery"], **common)

    result = merge_generation_manifests(
        [baseline, neuro],
        tmp_path / "merged",
        eligibility_manifest=eligibility,
    )

    assert result["methods"] == ["neurodiscovery", "sciagents"]
    assert len(result["runs"]) == 4
    assert result["paired_case_windows_per_seed"] == 1
    assert result["audit_by_method"]["neurodiscovery"]["failure_slots"] == 2


def test_merge_generation_manifests_rejects_unpaired_matrix(tmp_path: Path) -> None:
    eligibility, manifest_hash, matrix_hash = _write_eligibility(tmp_path)
    common = {
        "eligibility_manifest": eligibility,
        "eligibility_manifest_sha256": manifest_hash,
        "eligibility_matrix_sha256": matrix_hash,
    }
    baseline = _write_source(tmp_path / "baseline", methods=["sciagents"], **common)
    neuro = _write_source(
        tmp_path / "neuro",
        methods=["neurodiscovery"],
        omit_last=True,
        **common,
    )

    with pytest.raises(ValueError, match="unpaired generation matrix"):
        merge_generation_manifests(
            [baseline, neuro],
            tmp_path / "merged",
            eligibility_manifest=eligibility,
        )


def test_merge_generation_manifests_rejects_wrong_fixed_budget(tmp_path: Path) -> None:
    eligibility, manifest_hash, matrix_hash = _write_eligibility(tmp_path)
    common = {
        "eligibility_manifest": eligibility,
        "eligibility_manifest_sha256": manifest_hash,
        "eligibility_matrix_sha256": matrix_hash,
    }
    baseline = _write_source(tmp_path / "baseline", methods=["sciagents"], **common)
    neuro = _write_source(tmp_path / "neuro", methods=["neurodiscovery"], **common)
    payload = json.loads(neuro.read_text(encoding="utf-8"))
    hypotheses_path = Path(payload["runs"][0]["hypotheses_path"])
    hypotheses = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    hypotheses["hypotheses"].pop()
    hypotheses_path.write_text(json.dumps(hypotheses), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed-budget slot mismatch"):
        merge_generation_manifests(
            [baseline, neuro],
            tmp_path / "merged",
            eligibility_manifest=eligibility,
        )


def test_merge_generation_manifests_retains_unexecuted_candidate_tail(
    tmp_path: Path,
) -> None:
    eligibility, manifest_hash, matrix_hash = _write_eligibility(tmp_path)
    common = {
        "eligibility_manifest": eligibility,
        "eligibility_manifest_sha256": manifest_hash,
        "eligibility_matrix_sha256": matrix_hash,
    }
    baseline = _write_source(tmp_path / "baseline", methods=["sciagents"], **common)
    neuro = _write_source(tmp_path / "neuro", methods=["neurodiscovery"], **common)
    source = json.loads(neuro.read_text(encoding="utf-8"))
    for row in source["runs"]:
        hypotheses_path = Path(row["hypotheses_path"])
        payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
        payload["hypotheses"].append(
            {
                "id": f"neurodiscovery:{row['seed']}:tail",
                "hypothesis_type": "generation_failure",
            }
        )
        hypotheses_path.write_text(json.dumps(payload), encoding="utf-8")

    result = merge_generation_manifests(
        [baseline, neuro],
        tmp_path / "merged",
        eligibility_manifest=eligibility,
    )

    neuro_rows = [row for row in result["runs"] if row["method"] == "neurodiscovery"]
    assert {row["n_hypotheses"] for row in neuro_rows} == {3}
    assert {row["candidate_pool_size"] for row in neuro_rows} == {4}
    assert {row["candidate_tail_beyond_budget"] for row in neuro_rows} == {1}
    assert result["audit_by_method"]["neurodiscovery"]["failure_slots"] == 2
