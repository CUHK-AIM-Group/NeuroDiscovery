from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_experiment_source_bundle import freeze_source_bundle
from neurooracle.scripts.freeze_formal_dynamic_source_bundle import (
    DEFAULT_EXTRA_SOURCES,
    freeze_formal_dynamic_source_bundle,
    plan_formal_dynamic_bundle,
)
from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_source_bundle,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _inputs(tmp_path: Path) -> tuple[Path, list[Path], Path, list[Path]]:
    source_a = _write(tmp_path / "pkg/a.py", "A = 1\n")
    source_b = _write(tmp_path / "pkg/b.py", "B = 2\n")
    inherited_config = _write(tmp_path / "data/atlas/labels.csv", "id,name\n1,A\n")
    static_bundle = tmp_path / "base/static"
    dynamic_bundle = tmp_path / "base/dynamic"
    freeze_source_bundle(
        source_paths=[source_a],
        config_paths=[inherited_config],
        reference_paths=[],
        output_dir=static_bundle,
        repo_root=tmp_path,
    )
    freeze_source_bundle(
        source_paths=[source_b],
        config_paths=[],
        reference_paths=[],
        output_dir=dynamic_bundle,
        repo_root=tmp_path,
    )

    canonical = {}
    for name in ("knowledge_graph", "extracted_claims", "current_state"):
        path = _write(tmp_path / f"canonical/{name}.dat", name)
        canonical[name] = _record(path, tmp_path)
    snapshot_root = tmp_path / "snapshots"
    _write(snapshot_root / "canonical_kg_release.json", "{}\n")
    for name in ("knowledge_graph.json", "extracted_claims.jsonl", "manifest.json"):
        _write(snapshot_root / "kg_2019" / name, name)

    static_design = _write(tmp_path / "config/static.json", "{}\n")
    policy = _write(tmp_path / "config/policy.json", "{}\n")
    source_matrix = _write(tmp_path / "config/source_matrix.csv", "x\n1\n")
    source_lock = _write(
        tmp_path / "config/source_lock.json",
        json.dumps({"locked_matrix": {"path": str(source_matrix)}}),
    )
    generalization_matrix = _write(
        tmp_path / "config/generalization_matrix.csv", "x\n1\n"
    )
    generalization = _write(
        tmp_path / "config/generalization.json",
        json.dumps(
            {
                "source_dynamic_eligibility_manifest": str(source_lock),
                "source_dynamic_eligibility_matrix": str(source_matrix),
                "locked_matrix": {"path": str(generalization_matrix)},
            }
        ),
    )
    output = tmp_path / "frozen/formal_dynamic"
    design = _write(
        tmp_path / "config/design.json",
        json.dumps(
            {
                "status": "frozen_before_dynamic_generalization",
                "canonical_release": canonical,
                "source_static_design": _record(static_design, tmp_path),
                "frozen_policy": _record(policy, tmp_path),
                "dynamic_eligibility": {
                    "manifest": _record(generalization, tmp_path),
                    "matrix": _record(generalization_matrix, tmp_path),
                },
                "temporal_inputs": {
                    "snapshot_root": snapshot_root.relative_to(tmp_path).as_posix(),
                    "freeze_years": [2019],
                },
                "reproducibility": {
                    "source_bundle_manifest": (
                        output / "bundle_manifest.json"
                    ).relative_to(tmp_path).as_posix(),
                },
            }
        ),
    )
    extra = [_write(tmp_path / "pkg/formal.py", "FORMAL = True\n")]
    return design, [static_bundle / "bundle_manifest.json", dynamic_bundle / "bundle_manifest.json"], output, extra


def test_plan_and_freeze_formal_dynamic_bundle_are_complete(tmp_path: Path) -> None:
    design, bases, output, extra = _inputs(tmp_path)
    plan = plan_formal_dynamic_bundle(
        design_path=design,
        base_bundle_manifests=bases,
        output_dir=output,
        repo_root=tmp_path,
        extra_source_paths=extra,
        inherit_config_from=[bases[0]],
    )
    assert {path.name for path in plan["sources"]} == {"a.py", "b.py", "formal.py"}
    assert "labels.csv" in {path.name for path in plan["configs"]}
    assert {path.name for path in plan["references"]} == {
        "canonical_kg_release.json",
        "current_state.dat",
        "extracted_claims.dat",
        "extracted_claims.jsonl",
        "knowledge_graph.dat",
        "knowledge_graph.json",
        "manifest.json",
    }

    manifest = freeze_formal_dynamic_source_bundle(
        design_path=design,
        base_bundle_manifests=bases,
        output_dir=output,
        repo_root=tmp_path,
        extra_source_paths=extra,
        inherit_config_from=[bases[0]],
    )
    assert len(manifest["references"]) == 7
    verified = verify_source_bundle(
        output / "bundle_manifest.json",
        require_live_source=True,
        verify_references=True,
    )
    assert verified["status"] == "passed"


def test_default_sources_explicitly_pin_formal_dynamic_runtime() -> None:
    assert {
        "core/scripts/canonical_kg_release.py",
        "core/scripts/case_study_closed_loop_engine.py",
        "neurooracle/scripts/audit_case_study_hindcasting_executability.py",
        "neurooracle/scripts/case_study_hindcasting_eval.py",
        "neurooracle/scripts/compare_dynamic_closed_open_loop.py",
        "neurooracle/scripts/generate_neurodiscovery_hindcasting_replicates.py",
        "neurooracle/scripts/run_case_study_hindcasting.py",
        "neurooracle/scripts/run_neurodiscovery_closed_loop_hindcasting.py",
        "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py",
    }.issubset(DEFAULT_EXTRA_SOURCES)


def test_plan_rejects_design_declaring_another_bundle(tmp_path: Path) -> None:
    design, bases, output, extra = _inputs(tmp_path)
    payload = json.loads(design.read_text(encoding="utf-8"))
    payload["reproducibility"]["source_bundle_manifest"] = "frozen/wrong.json"
    design.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declares a different source bundle"):
        plan_formal_dynamic_bundle(
            design_path=design,
            base_bundle_manifests=bases,
            output_dir=output,
            repo_root=tmp_path,
            extra_source_paths=extra,
        )


def test_plan_rejects_missing_temporal_snapshot(tmp_path: Path) -> None:
    design, bases, output, extra = _inputs(tmp_path)
    (tmp_path / "snapshots/kg_2019/extracted_claims.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="referenced data inputs"):
        plan_formal_dynamic_bundle(
            design_path=design,
            base_bundle_manifests=bases,
            output_dir=output,
            repo_root=tmp_path,
            extra_source_paths=extra,
        )


def test_base_bundle_reference_drift_does_not_block_new_release_bundle(
    tmp_path: Path,
) -> None:
    design, bases, output, extra = _inputs(tmp_path)
    historical_reference = _write(tmp_path / "old_release/data.json", "old\n")
    source = _write(tmp_path / "pkg/historical.py", "HISTORICAL = True\n")
    historical_bundle = tmp_path / "base/historical"
    freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[historical_reference],
        output_dir=historical_bundle,
        repo_root=tmp_path,
    )
    historical_reference.write_text("new-release-bytes\n", encoding="utf-8")

    plan = plan_formal_dynamic_bundle(
        design_path=design,
        base_bundle_manifests=[*bases, historical_bundle / "bundle_manifest.json"],
        output_dir=output,
        repo_root=tmp_path,
        extra_source_paths=extra,
    )
    assert source.resolve() in plan["sources"]
