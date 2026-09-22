from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_experiment_source_bundle import freeze_source_bundle
from neurooracle.src.experiment_source_bundle import (
    verify_bundle_member,
    verify_source_bundle,
)


def test_verify_source_bundle_checks_live_archive_and_references(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    reference = tmp_path / "data.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    reference.write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[reference],
        output_dir=bundle,
        repo_root=tmp_path,
    )

    audit = verify_source_bundle(bundle / "bundle_manifest.json")

    assert audit["status"] == "passed"
    assert audit["bundled_files"] == 1
    assert audit["referenced_files"] == 1


def test_verify_source_bundle_accepts_empty_python_source(tmp_path: Path) -> None:
    source = tmp_path / "__init__.py"
    source.write_bytes(b"")
    bundle = tmp_path / "bundle"
    freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[],
        output_dir=bundle,
        repo_root=tmp_path,
    )

    audit = verify_source_bundle(bundle / "bundle_manifest.json")

    assert audit["status"] == "passed"
    assert audit["bundled_files"] == 1


def test_verify_source_bundle_rejects_live_source_change(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    freeze_source_bundle(
        source_paths=[source],
        config_paths=[],
        reference_paths=[],
        output_dir=bundle,
        repo_root=tmp_path,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="live source"):
        verify_source_bundle(bundle / "bundle_manifest.json")


def test_verify_bundle_member_checks_selected_live_config(tmp_path: Path) -> None:
    config = tmp_path / "design.json"
    config.write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    freeze_source_bundle(
        source_paths=[],
        config_paths=[config],
        reference_paths=[],
        output_dir=bundle,
        repo_root=tmp_path,
    )

    assert verify_bundle_member(
        bundle / "bundle_manifest.json", config
    )["roles"] == ["config"]
    config.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bundle member"):
        verify_bundle_member(bundle / "bundle_manifest.json", config)
