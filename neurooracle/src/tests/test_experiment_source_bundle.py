from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_experiment_source_bundle import (
    freeze_source_bundle,
)
from neurooracle.scripts.run_neurodiscovery_dynamic_closed_loop_hindcasting import (
    _verified_source_bundle,
)


def _freeze(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "package" / "runner.py"
    config = repo / "config.json"
    reference = tmp_path / "knowledge_graph.json"
    source.parent.mkdir(parents=True)
    source.write_text("print('frozen')\n", encoding="utf-8")
    config.write_text('{"seed": 7}\n', encoding="utf-8")
    reference.write_text('{"large": "input"}\n', encoding="utf-8")
    output = tmp_path / "bundle"
    manifest = freeze_source_bundle(
        source_paths=[source],
        config_paths=[config],
        reference_paths=[reference],
        output_dir=output,
        command="python runner.py --seed 7",
        repo_root=repo,
        registered_at="2026-08-13T00:00:00+00:00",
    )
    return manifest, output, source, reference


def test_source_bundle_copies_code_and_hashes_data_reference(tmp_path: Path) -> None:
    manifest, output, _, reference = _freeze(tmp_path)

    assert manifest["schema_version"].endswith(".v1")
    assert len(manifest["files"]) == 2
    assert len(manifest["references"]) == 1
    assert manifest["references"][0]["path"] == str(reference.resolve())
    assert not (output / "files" / "external" / reference.name).exists()
    assert (output / "files" / "package" / "runner.py").is_file()
    persisted = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["environment"]["environment_variables_captured"] is False


def test_source_bundle_is_idempotent(tmp_path: Path) -> None:
    first, output, source, reference = _freeze(tmp_path)
    second = freeze_source_bundle(
        source_paths=[source],
        config_paths=[source.parents[1] / "config.json"],
        reference_paths=[reference],
        output_dir=output,
        command="python runner.py --seed 7",
        repo_root=source.parents[1],
        registered_at="2099-01-01T00:00:00+00:00",
    )

    assert second == first
    assert second["registered_at"] == "2026-08-13T00:00:00+00:00"


def test_source_bundle_rejects_source_drift(tmp_path: Path) -> None:
    _, output, source, reference = _freeze(tmp_path)
    source.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bytes changed"):
        freeze_source_bundle(
            source_paths=[source],
            config_paths=[source.parents[1] / "config.json"],
            reference_paths=[reference],
            output_dir=output,
            command="python runner.py --seed 7",
            repo_root=source.parents[1],
        )


def test_source_bundle_rejects_archive_tampering(tmp_path: Path) -> None:
    _, output, source, reference = _freeze(tmp_path)
    (output / "files" / "package" / "runner.py").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="modified or removed"):
        freeze_source_bundle(
            source_paths=[source],
            config_paths=[source.parents[1] / "config.json"],
            reference_paths=[reference],
            output_dir=output,
            command="python runner.py --seed 7",
            repo_root=source.parents[1],
        )


def test_dynamic_runner_verifies_live_archive_and_references(tmp_path: Path) -> None:
    _, output, _, _ = _freeze(tmp_path)

    verified = _verified_source_bundle(output / "bundle_manifest.json")

    assert verified is not None
    assert verified["live_source_verified"] is True
    assert verified["archived_source_verified"] is True
    assert verified["referenced_inputs_verified"] is True
    assert verified["bundled_files"] == 2
    assert verified["referenced_files"] == 1


def test_dynamic_runner_rejects_live_source_drift(tmp_path: Path) -> None:
    _, output, source, _ = _freeze(tmp_path)
    source.write_text("print('drifted')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="live source differs"):
        _verified_source_bundle(output / "bundle_manifest.json")


def test_archive_execution_can_ignore_live_source_drift(tmp_path: Path) -> None:
    _, output, source, _ = _freeze(tmp_path)
    source.write_text("print('later workspace edit')\n", encoding="utf-8")

    verified = _verified_source_bundle(
        output / "bundle_manifest.json",
        require_live_source=False,
    )

    assert verified is not None
    assert verified["live_source_match_required"] is False
    assert verified["live_source_verified"] is False
    assert verified["archived_source_verified"] is True


def test_dynamic_runner_rejects_reference_drift(tmp_path: Path) -> None:
    _, output, _, reference = _freeze(tmp_path)
    reference.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="referenced input differs"):
        _verified_source_bundle(output / "bundle_manifest.json")


def test_source_bundle_shortens_deep_windows_archive_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / ("deep_" + "x" * 80) / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('deep')\n", encoding="utf-8")
    output = tmp_path / "bundle"

    manifest = freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[],
        output_dir=output,
        repo_root=repo,
    )

    record = manifest["files"][0]
    assert record["source_path"] == str(source.resolve())
    assert record["bundle_path"].startswith("files/long/")
    assert len(record["bundle_path"]) <= 96
    assert (output / record["bundle_path"]).is_file()


def test_source_bundle_accounts_for_deep_output_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "neurooracle" / "scripts" / (
        "generate_neurodiscovery_hindcasting_replicates.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = (
        tmp_path
        / "neurooracle"
        / "data"
        / "experiments"
        / "hindcasting"
        / "optimization_protocol_20260812"
        / "source_bundles"
        / "formal_static_endpoint_v7_20260813_final_v2"
    )

    manifest = freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[],
        output_dir=output,
        repo_root=repo,
    )

    record = manifest["files"][0]
    assert record["bundle_path"].startswith("files/long/")
    assert (output / record["bundle_path"]).is_file()
