from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.prepare_hindcasting_v3_kge import (
    CONFIG,
    FREEZE_YEARS,
    SCHEMA,
    YEAR_SCHEMA,
    sha256_file,
    verify_kge_assets_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_asset_set(tmp_path: Path) -> tuple[Path, dict[int, Path]]:
    release_id = "kg_test_release"
    release_manifest = tmp_path / "release_manifest.json"
    _write_json(release_manifest, {"release_id": release_id})

    snapshots = []
    snapshot_files: dict[int, dict[str, Path]] = {}
    for freeze_year in FREEZE_YEARS:
        snapshot_dir = tmp_path / "snapshots" / f"kg_{freeze_year}"
        snapshot_dir.mkdir(parents=True)
        snapshot_manifest = snapshot_dir / "manifest.json"
        knowledge_graph = snapshot_dir / "knowledge_graph.json"
        extracted_claims = snapshot_dir / "extracted_claims.jsonl"
        _write_json(snapshot_manifest, {"freeze_year": freeze_year})
        _write_json(knowledge_graph, {"concepts": {}, "edges": []})
        extracted_claims.write_text("{}\n", encoding="utf-8")
        snapshots.append(
            {
                "freeze_year": freeze_year,
                "directory": str(snapshot_dir.resolve()),
                "manifest": str(snapshot_manifest.resolve()),
                "manifest_sha256": sha256_file(snapshot_manifest),
            }
        )
        snapshot_files[freeze_year] = {
            "manifest": snapshot_manifest,
            "knowledge_graph": knowledge_graph,
            "extracted_claims": extracted_claims,
        }

    pipeline_path = tmp_path / "eligibility_pipeline_manifest.json"
    pipeline = {
        "schema_version": "neurodiscovery-hindcasting-v3-eligibility-pipeline.v1",
        "status": "locked_method_blind_eligibility",
        "release_id": release_id,
        "release_manifest": str(release_manifest.resolve()),
        "release_manifest_sha256": sha256_file(release_manifest),
        "method_outputs_consumed": False,
        "snapshots": snapshots,
    }
    _write_json(pipeline_path, pipeline)
    _write_json(
        tmp_path / "eligibility_pipeline.lock.json",
        {
            "pipeline_manifest_sha256": sha256_file(pipeline_path),
            "immutable": True,
        },
    )

    output_dir = tmp_path / "kge"
    output_dir.mkdir()
    year_records = []
    checkpoints: dict[int, Path] = {}
    for freeze_year in FREEZE_YEARS:
        checkpoint = output_dir / f"kg_{freeze_year}.pt"
        report = output_dir / f"kg_{freeze_year}_report.json"
        asset_path = output_dir / f"kg_{freeze_year}_asset.json"
        checkpoint.write_bytes(f"checkpoint-{freeze_year}".encode("ascii"))
        _write_json(report, {"freeze_year": freeze_year})
        files = snapshot_files[freeze_year]
        asset = {
            "schema_version": YEAR_SCHEMA,
            "status": "locked",
            "release_id": release_id,
            "eligibility_pipeline_sha256": sha256_file(pipeline_path),
            "freeze_year": freeze_year,
            "config": dict(CONFIG),
            "snapshot": {
                name: _file_record(path) for name, path in files.items()
            },
            "checkpoint": _file_record(checkpoint),
            "report": _file_record(report),
        }
        _write_json(asset_path, asset)
        year_records.append(
            {
                "freeze_year": freeze_year,
                "path": str(asset_path.resolve()),
                "sha256": sha256_file(asset_path),
                "checkpoint_sha256": asset["checkpoint"]["sha256"],
                "snapshot_manifest_sha256": asset["snapshot"]["manifest"]["sha256"],
            }
        )
        checkpoints[freeze_year] = checkpoint

    manifest_path = output_dir / "kge_assets_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA,
            "status": "locked_release_local_kge_assets",
            "release_id": release_id,
            "eligibility_pipeline": str(pipeline_path.resolve()),
            "eligibility_pipeline_sha256": sha256_file(pipeline_path),
            "config": dict(CONFIG),
            "year_assets": year_records,
            "cross_release_reuse_permitted": False,
        },
    )
    _write_json(
        output_dir / "kge_assets.lock.json",
        {
            "release_id": release_id,
            "manifest_sha256": sha256_file(manifest_path),
            "immutable": True,
        },
    )
    return manifest_path, checkpoints


def test_verify_kge_assets_manifest_pins_all_five_years(tmp_path: Path) -> None:
    manifest_path, _ = _build_asset_set(tmp_path)

    verified = verify_kge_assets_manifest(manifest_path, deep=True)

    assert verified["status"] == "locked_release_local_kge_assets"
    assert [row["freeze_year"] for row in verified["year_assets"]] == list(
        FREEZE_YEARS
    )
    assert verified["deep_verified"] is True


def test_deep_verification_detects_same_size_checkpoint_tampering(
    tmp_path: Path,
) -> None:
    manifest_path, checkpoints = _build_asset_set(tmp_path)
    checkpoint = checkpoints[2020]
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    assert verify_kge_assets_manifest(manifest_path, deep=False)["deep_verified"] is False
    with pytest.raises(ValueError, match="KGE checkpoint hash mismatch"):
        verify_kge_assets_manifest(manifest_path, deep=True)
