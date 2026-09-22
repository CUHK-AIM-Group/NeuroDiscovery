"""Train and lock release-local ComplEx checkpoints for hindcasting v3.

The v3 scientific protocol forbids reusing KGE checkpoints across KG releases.
This orchestrator binds every checkpoint to one immutable eligibility pipeline
and to the exact snapshot graph and claims bytes used for its freeze year.
Completed years are independently sealed so an interrupted five-year build can
resume without retraining verified checkpoints.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.src.experiment_source_bundle import sha256_file
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826"
)
DEFAULT_PIPELINE = (
    DEFAULT_V3_ROOT
    / "eligibility/kg_20260825_2c02732582da_705b0799"
    / "eligibility_pipeline_manifest.json"
)

SCHEMA = "neurodiscovery-hindcasting-v3-kge-assets.v1"
YEAR_SCHEMA = "neurodiscovery-hindcasting-v3-kge-year-asset.v1"
CONFIG = {
    "model": "ComplEx",
    "dim": 64,
    "epochs": 10,
    "batch_size": 8192,
    "lr": 0.001,
    "negatives_per_pos": 5,
    "weight_decay": 0.000001,
    "eval_every": 5,
    "early_stop_patience": 0,
    "min_confidence": 0.2,
    "seed_rule": "freeze_year",
}
FREEZE_YEARS = (2016, 2017, 2018, 2019, 2020)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_file_record(
    record: Mapping[str, Any], *, label: str, deep: bool = True
) -> Path:
    path = Path(str(record.get("path") or "")).resolve()
    expected_bytes = int(record.get("bytes") or -1)
    expected_hash = str(record.get("sha256") or "").upper()
    _require(path.is_file(), f"missing {label}: {path}")
    _require(path.stat().st_size == expected_bytes, f"{label} size mismatch: {path}")
    _require(len(expected_hash) == 64, f"invalid {label} SHA-256: {path}")
    if deep:
        _require(sha256_file(path) == expected_hash, f"{label} hash mismatch: {path}")
    return path


def _validate_pipeline(path: Path) -> dict[str, Any]:
    path = path.resolve()
    pipeline = _read_json(path)
    lock_path = path.with_name("eligibility_pipeline.lock.json")
    lock = _read_json(lock_path)
    _require(
        pipeline.get("schema_version")
        == "neurodiscovery-hindcasting-v3-eligibility-pipeline.v1",
        "incompatible eligibility pipeline",
    )
    _require(
        pipeline.get("status") == "locked_method_blind_eligibility",
        "eligibility pipeline is not locked",
    )
    _require(
        sha256_file(path) == str(lock.get("pipeline_manifest_sha256") or "").upper(),
        "eligibility pipeline differs from its lock",
    )
    _require(
        pipeline.get("method_outputs_consumed") is False,
        "eligibility pipeline is not method-blind",
    )
    release_manifest = Path(str(pipeline.get("release_manifest") or "")).resolve()
    _require(release_manifest.is_file(), "eligibility release manifest is missing")
    _require(
        sha256_file(release_manifest)
        == str(pipeline.get("release_manifest_sha256") or "").upper(),
        "eligibility release manifest hash mismatch",
    )
    snapshots = list(pipeline.get("snapshots") or ())
    _require(
        [int(record.get("freeze_year") or 0) for record in snapshots]
        == list(FREEZE_YEARS),
        "eligibility pipeline does not contain the five locked freeze years",
    )
    for record in snapshots:
        manifest_path = Path(str(record.get("manifest") or "")).resolve()
        _require(manifest_path.is_file(), f"missing snapshot manifest: {manifest_path}")
        _require(
            sha256_file(manifest_path)
            == str(record.get("manifest_sha256") or "").upper(),
            f"snapshot manifest hash mismatch: {manifest_path}",
        )
    return pipeline


def _expected_year_paths(output_dir: Path, freeze_year: int) -> dict[str, Path]:
    prefix = f"kg_{freeze_year}_complex_dim64_ep10"
    return {
        "checkpoint": output_dir / f"{prefix}.pt",
        "report": output_dir / f"{prefix}_report.json",
        "asset": output_dir / f"{prefix}_asset.json",
        "checkpoint_tmp": output_dir / f".{prefix}.pt.tmp",
        "report_tmp": output_dir / f".{prefix}_report.json.tmp",
    }


def _verify_year_asset(
    asset_path: Path,
    *,
    release_id: str,
    pipeline_hash: str,
    freeze_year: int,
    snapshot_manifest_hash: str,
    deep: bool = True,
) -> dict[str, Any]:
    asset = _read_json(asset_path)
    _require(asset.get("schema_version") == YEAR_SCHEMA, "incompatible KGE year asset")
    _require(asset.get("status") == "locked", "KGE year asset is not locked")
    _require(asset.get("release_id") == release_id, "KGE year asset release mismatch")
    _require(
        asset.get("eligibility_pipeline_sha256") == pipeline_hash,
        "KGE year asset pipeline mismatch",
    )
    _require(int(asset.get("freeze_year") or 0) == freeze_year, "KGE year mismatch")
    _require(asset.get("config") == CONFIG, "KGE year configuration mismatch")
    snapshot = asset.get("snapshot") or {}
    _require(
        str((snapshot.get("manifest") or {}).get("sha256") or "").upper()
        == snapshot_manifest_hash,
        "KGE snapshot manifest mismatch",
    )
    for key in ("manifest", "knowledge_graph", "extracted_claims"):
        _verify_file_record(
            snapshot.get(key) or {}, label=f"snapshot {key}", deep=deep
        )
    _verify_file_record(
        asset.get("checkpoint") or {}, label="KGE checkpoint", deep=deep
    )
    _verify_file_record(asset.get("report") or {}, label="KGE report", deep=deep)
    return asset


def verify_kge_assets_manifest(
    manifest_path: Path,
    *,
    expected_pipeline_path: Path | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    """Verify one release-local five-year KGE asset set.

    The manifest and each tiny year-asset record are always hashed.  ``deep``
    additionally re-hashes the multi-gigabyte snapshot inputs and checkpoints;
    shallow verification still checks their declared sizes and existence.
    """

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    lock_path = manifest_path.with_name("kge_assets.lock.json")
    lock = _read_json(lock_path)
    manifest_hash = sha256_file(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA, "incompatible KGE manifest")
    _require(
        manifest.get("status") == "locked_release_local_kge_assets",
        "KGE manifest is not locked",
    )
    _require(
        manifest_hash == str(lock.get("manifest_sha256") or "").upper(),
        "KGE manifest differs from its lock",
    )
    _require(lock.get("immutable") is True, "KGE asset lock is not immutable")
    _require(
        lock.get("release_id") == manifest.get("release_id"),
        "KGE asset lock release mismatch",
    )
    _require(manifest.get("config") == CONFIG, "KGE manifest config mismatch")
    _require(
        manifest.get("cross_release_reuse_permitted") is False,
        "KGE manifest permits cross-release reuse",
    )

    pipeline_path = Path(str(manifest.get("eligibility_pipeline") or "")).resolve()
    if expected_pipeline_path is not None:
        _require(
            pipeline_path == expected_pipeline_path.resolve(),
            "KGE manifest references a different eligibility pipeline",
        )
    pipeline = _validate_pipeline(pipeline_path)
    pipeline_hash = sha256_file(pipeline_path)
    pipeline_lock_path = pipeline_path.with_name("eligibility_pipeline.lock.json")
    release_id = str(pipeline.get("release_id") or "")
    _require(manifest.get("release_id") == release_id, "KGE manifest release mismatch")
    _require(
        manifest.get("eligibility_pipeline_sha256") == pipeline_hash,
        "KGE manifest pipeline hash mismatch",
    )
    snapshot_by_year = {
        int(record["freeze_year"]): record for record in pipeline["snapshots"]
    }

    records = list(manifest.get("year_assets") or ())
    _require(
        [int(record.get("freeze_year") or 0) for record in records]
        == list(FREEZE_YEARS),
        "KGE manifest does not contain the five ordered freeze years",
    )
    verified_years: list[dict[str, Any]] = []
    for record in records:
        freeze_year = int(record["freeze_year"])
        expected_snapshot_hash = str(
            snapshot_by_year[freeze_year]["manifest_sha256"]
        ).upper()
        asset_path = Path(str(record.get("path") or "")).resolve()
        _require(asset_path.is_file(), f"missing KGE year asset: {asset_path}")
        asset_hash = sha256_file(asset_path)
        _require(
            asset_hash == str(record.get("sha256") or "").upper(),
            f"KGE year asset hash mismatch: {asset_path}",
        )
        asset = _verify_year_asset(
            asset_path,
            release_id=release_id,
            pipeline_hash=pipeline_hash,
            freeze_year=freeze_year,
            snapshot_manifest_hash=expected_snapshot_hash,
            deep=deep,
        )
        _require(
            str(record.get("checkpoint_sha256") or "").upper()
            == str((asset.get("checkpoint") or {}).get("sha256") or "").upper(),
            f"KGE checkpoint record mismatch: {freeze_year}",
        )
        _require(
            str(record.get("snapshot_manifest_sha256") or "").upper()
            == expected_snapshot_hash,
            f"snapshot record mismatch: {freeze_year}",
        )
        verified_years.append(
            {
                "freeze_year": freeze_year,
                "asset": {
                    "path": str(asset_path),
                    "sha256": asset_hash,
                },
                "snapshot": asset["snapshot"],
                "checkpoint": asset["checkpoint"],
                "report": asset["report"],
            }
        )

    return {
        "status": manifest["status"],
        "release_id": release_id,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
        },
        "lock": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
        },
        "eligibility_pipeline": {
            "path": str(pipeline_path),
            "sha256": pipeline_hash,
            "lock_path": str(pipeline_lock_path),
            "lock_sha256": sha256_file(pipeline_lock_path),
            "release_manifest": str(Path(pipeline["release_manifest"]).resolve()),
            "release_manifest_sha256": str(
                pipeline["release_manifest_sha256"]
            ).upper(),
            "method_outputs_consumed": pipeline["method_outputs_consumed"],
        },
        "config": dict(CONFIG),
        "year_assets": verified_years,
        "deep_verified": deep,
    }


def _train_year(
    *,
    release_id: str,
    pipeline_path: Path,
    pipeline_hash: str,
    snapshot_record: Mapping[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    freeze_year = int(snapshot_record["freeze_year"])
    paths = _expected_year_paths(output_dir, freeze_year)
    snapshot_dir = Path(str(snapshot_record["directory"])).resolve()
    snapshot_manifest = Path(str(snapshot_record["manifest"])).resolve()
    snapshot_manifest_hash = str(snapshot_record["manifest_sha256"]).upper()

    if paths["asset"].is_file():
        print(f"[reuse] KGE_{freeze_year}", flush=True)
        return _verify_year_asset(
            paths["asset"],
            release_id=release_id,
            pipeline_hash=pipeline_hash,
            freeze_year=freeze_year,
            snapshot_manifest_hash=snapshot_manifest_hash,
        )

    _require(
        not paths["checkpoint"].exists() and not paths["report"].exists(),
        f"partial KGE outputs require audit before retry: {freeze_year}",
    )
    paths["checkpoint_tmp"].unlink(missing_ok=True)
    paths["report_tmp"].unlink(missing_ok=True)

    print(f"[hash] snapshot KG_{freeze_year}", flush=True)
    snapshot = {
        "manifest": _file_record(snapshot_manifest),
        "knowledge_graph": _file_record(snapshot_dir / "knowledge_graph.json"),
        "extracted_claims": _file_record(snapshot_dir / "extracted_claims.jsonl"),
    }
    _require(
        snapshot["manifest"]["sha256"] == snapshot_manifest_hash,
        f"snapshot manifest changed before KGE training: {freeze_year}",
    )

    print(f"[train] KGE_{freeze_year} on {device}", flush=True)
    try:
        # Keep verification-only imports lightweight for the cohort manager.
        from neurooracle.src.kge.cli import cmd_kge_train

        cmd_kge_train(
            kg_path=str(snapshot_dir / "knowledge_graph.json"),
            output=str(paths["checkpoint_tmp"]),
            report=str(paths["report_tmp"]),
            dim=int(CONFIG["dim"]),
            epochs=int(CONFIG["epochs"]),
            batch_size=int(CONFIG["batch_size"]),
            lr=float(CONFIG["lr"]),
            negatives_per_pos=int(CONFIG["negatives_per_pos"]),
            weight_decay=float(CONFIG["weight_decay"]),
            eval_every=int(CONFIG["eval_every"]),
            early_stop_patience=int(CONFIG["early_stop_patience"]),
            min_confidence=float(CONFIG["min_confidence"]),
            seed=freeze_year,
            device=device,
        )
        report = _read_json(paths["report_tmp"])
        report["checkpoint"] = str(paths["checkpoint"].resolve())
        report["release_id"] = release_id
        report["freeze_year"] = freeze_year
        report["snapshot_manifest_sha256"] = snapshot_manifest_hash
        _atomic_json(paths["report_tmp"], report)
        paths["checkpoint_tmp"].replace(paths["checkpoint"])
        paths["report_tmp"].replace(paths["report"])
    except BaseException:
        paths["checkpoint_tmp"].unlink(missing_ok=True)
        paths["report_tmp"].unlink(missing_ok=True)
        raise

    asset = {
        "schema_version": YEAR_SCHEMA,
        "status": "locked",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "release_id": release_id,
        "eligibility_pipeline": str(pipeline_path.resolve()),
        "eligibility_pipeline_sha256": pipeline_hash,
        "freeze_year": freeze_year,
        "config": dict(CONFIG),
        "device": device,
        "snapshot": snapshot,
        "checkpoint": _file_record(paths["checkpoint"]),
        "report": _file_record(paths["report"]),
        "test_auroc": report.get("test_auroc"),
    }
    _atomic_json(paths["asset"], asset)
    return _verify_year_asset(
        paths["asset"],
        release_id=release_id,
        pipeline_hash=pipeline_hash,
        freeze_year=freeze_year,
        snapshot_manifest_hash=snapshot_manifest_hash,
    )


def prepare_kge_assets(
    *, pipeline_path: Path, output_dir: Path, device: str
) -> dict[str, Any]:
    pipeline_path = pipeline_path.resolve()
    pipeline = _validate_pipeline(pipeline_path)
    pipeline_hash = sha256_file(pipeline_path)
    release_id = str(pipeline["release_id"])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "kge_assets_manifest.json"
    lock_path = output_dir / "kge_assets.lock.json"

    if manifest_path.is_file():
        verify_kge_assets_manifest(
            manifest_path,
            expected_pipeline_path=pipeline_path,
            deep=True,
        )
        return _read_json(manifest_path)

    year_assets = []
    for snapshot_record in pipeline["snapshots"]:
        asset = _train_year(
            release_id=release_id,
            pipeline_path=pipeline_path,
            pipeline_hash=pipeline_hash,
            snapshot_record=snapshot_record,
            output_dir=output_dir,
            device=device,
        )
        asset_path = _expected_year_paths(
            output_dir, int(snapshot_record["freeze_year"])
        )["asset"]
        year_assets.append(
            {
                "freeze_year": int(asset["freeze_year"]),
                "path": str(asset_path.resolve()),
                "sha256": sha256_file(asset_path),
                "checkpoint_sha256": asset["checkpoint"]["sha256"],
                "snapshot_manifest_sha256": asset["snapshot"]["manifest"]["sha256"],
            }
        )

    manifest = {
        "schema_version": SCHEMA,
        "status": "locked_release_local_kge_assets",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_id": release_id,
        "eligibility_pipeline": str(pipeline_path),
        "eligibility_pipeline_sha256": pipeline_hash,
        "config": dict(CONFIG),
        "year_assets": year_assets,
        "cross_release_reuse_permitted": False,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        lock_path,
        {
            "schema_version": "neurodiscovery-hindcasting-v3-kge-assets-lock.v1",
            "release_id": release_id,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "immutable": True,
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pipeline = _validate_pipeline(args.eligibility_pipeline)
    output_dir = args.output_dir or (
        DEFAULT_V3_ROOT / "execution_assets" / str(pipeline["release_id"]) / "kge"
    )
    result = prepare_kge_assets(
        pipeline_path=args.eligibility_pipeline,
        output_dir=output_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release_id": result["release_id"],
                "freeze_years": [
                    record["freeze_year"] for record in result["year_assets"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
