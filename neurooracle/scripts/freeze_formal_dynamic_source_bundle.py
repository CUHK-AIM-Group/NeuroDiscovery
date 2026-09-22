"""Freeze the source, configuration, and data inputs for formal dynamic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.scripts.freeze_experiment_source_bundle import freeze_source_bundle
from neurooracle.src.experiment_source_bundle import (
    SOURCE_BUNDLE_SCHEMA,
    sha256_file,
    verify_source_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
)
DEFAULT_DESIGN = PROTOCOL_ROOT / "formal_dynamic_generalization_design_20260813.json"
DEFAULT_STATIC_BASE_BUNDLE = (
    ROOT / "neurooracle/.frozen/formal_static_endpoint_v7_v5/bundle_manifest.json"
)
DEFAULT_DYNAMIC_BASE_BUNDLE = (
    PROTOCOL_ROOT
    / "source_bundles/dynamic_dev_repro_final_20260813/bundle_manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "neurooracle/.frozen/formal_dynamic_generalization_v1"
DEFAULT_EXTRA_SOURCES = (
    # Pin the formal execution path explicitly instead of relying only on an
    # inherited development bundle to keep listing these runtime files.
    "core/scripts/canonical_kg_release.py",
    "core/scripts/case_study_closed_loop_engine.py",
    "neurooracle/scripts/audit_case_study_hindcasting_executability.py",
    "neurooracle/scripts/audit_dynamic_hindcasting_eligibility.py",
    "neurooracle/scripts/case_study_hindcasting_eval.py",
    "neurooracle/scripts/compare_dynamic_closed_open_loop.py",
    "neurooracle/scripts/generate_neurodiscovery_hindcasting_replicates.py",
    "neurooracle/scripts/freeze_dynamic_generalization_eligibility.py",
    "neurooracle/scripts/freeze_dynamic_policy_application.py",
    "neurooracle/scripts/freeze_formal_dynamic_generalization_design.py",
    "neurooracle/scripts/freeze_formal_dynamic_source_bundle.py",
    "neurooracle/scripts/run_case_study_hindcasting.py",
    "neurooracle/scripts/run_formal_dynamic_generalization.py",
    "neurooracle/scripts/run_neurodiscovery_closed_loop_hindcasting.py",
    "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py",
    "neurooracle/scripts/verify_formal_dynamic_generalization.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {path.resolve() for path in paths},
        key=lambda value: str(value).casefold(),
    )


def _require_files(paths: Iterable[Path], *, label: str) -> list[Path]:
    resolved = _deduplicate(paths)
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"missing {label}: {preview}")
    return resolved


def _manifest_member_paths(
    manifest_path: Path,
    *,
    role: str,
) -> list[Path]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
        raise ValueError(f"source bundle uses an incompatible schema: {manifest_path}")
    bundle_root = manifest_path.resolve().parent
    for record in manifest.get("files") or ():
        archived = bundle_root / str(record.get("bundle_path") or "")
        expected_size = int(record.get("bytes") or -1)
        expected_hash = str(record.get("sha256") or "").upper()
        if (
            expected_size < 0
            or not expected_hash
            or not archived.is_file()
            or archived.stat().st_size != expected_size
            or sha256_file(archived) != expected_hash
        ):
            raise ValueError(f"archived source member mismatch: {archived}")
    # Base bundles define the audited source inventory. Their historical data
    # references may legitimately differ from the new application release, so
    # only archived code members are verified here; the new bundle separately
    # hashes every current-release reference declared by its frozen design.
    return [
        Path(str(record["source_path"])).resolve()
        for record in manifest.get("files") or ()
        if role in set(record.get("roles") or ())
    ]


def _record_path(record: Mapping[str, Any], root: Path, *, label: str) -> Path:
    value = str(record.get("path") or "")
    if not value:
        raise ValueError(f"{label} has no path")
    return _resolve(value, root)


def _design_config_paths(design: Mapping[str, Any], design_path: Path, root: Path) -> list[Path]:
    paths = [design_path]
    paths.append(_record_path(design["source_static_design"], root, label="static design"))
    paths.append(_record_path(design["frozen_policy"], root, label="frozen policy"))
    policy = _load_json(paths[-1])
    application = policy.get("policy_application") or {}
    if application:
        paths.append(
            _record_path(application["source_policy"], root, label="source policy")
        )
    eligibility = design.get("dynamic_eligibility") or {}
    paths.append(_record_path(eligibility["manifest"], root, label="eligibility manifest"))
    paths.append(_record_path(eligibility["matrix"], root, label="eligibility matrix"))

    generalization_manifest = _load_json(paths[-2])
    for field in (
        "source_dynamic_eligibility_manifest",
        "source_dynamic_eligibility_matrix",
    ):
        value = str(generalization_manifest.get(field) or "")
        if not value:
            raise ValueError(f"dynamic generalization lock has no {field}")
        paths.append(_resolve(value, root))
    return paths


def _design_reference_paths(design: Mapping[str, Any], root: Path) -> list[Path]:
    canonical = design.get("canonical_release") or {}
    references = [
        _record_path(canonical[label], root, label=f"canonical {label}")
        for label in ("knowledge_graph", "extracted_claims", "current_state")
    ]

    temporal = design.get("temporal_inputs") or {}
    snapshot_root = _resolve(str(temporal.get("snapshot_root") or ""), root)
    freeze_years = sorted({int(value) for value in temporal.get("freeze_years") or ()})
    if not freeze_years:
        raise ValueError("formal dynamic design has no freeze years")
    release_record = snapshot_root / "canonical_kg_release.json"
    if release_record.is_file():
        references.append(release_record)
    for year in freeze_years:
        snapshot = snapshot_root / f"kg_{year}"
        references.extend(
            (
                snapshot / "knowledge_graph.json",
                snapshot / "extracted_claims.jsonl",
                snapshot / "manifest.json",
            )
        )
    return references


def plan_formal_dynamic_bundle(
    *,
    design_path: Path,
    base_bundle_manifests: Sequence[Path],
    output_dir: Path,
    repo_root: Path,
    extra_source_paths: Sequence[Path],
    inherit_config_from: Sequence[Path] = (),
) -> dict[str, list[Path]]:
    """Resolve and validate every path before hashing or copying the bundle."""

    repo_root = repo_root.resolve()
    design_path = design_path.resolve()
    output_dir = output_dir.resolve()
    design = _load_json(design_path)
    if design.get("status") != "frozen_before_dynamic_generalization":
        raise ValueError("formal dynamic design is not frozen")
    declared = _resolve(
        str((design.get("reproducibility") or {}).get("source_bundle_manifest") or ""),
        repo_root,
    )
    expected = output_dir / "bundle_manifest.json"
    if declared != expected:
        raise ValueError(
            "formal dynamic design declares a different source bundle: "
            f"declared={declared} expected={expected}"
        )

    base_bundle_manifests = _require_files(
        base_bundle_manifests,
        label="base bundle manifests",
    )
    source_paths: list[Path] = []
    for manifest_path in base_bundle_manifests:
        source_paths.extend(_manifest_member_paths(manifest_path, role="source"))
    source_paths.extend(extra_source_paths)

    config_paths = _design_config_paths(design, design_path, repo_root)
    config_paths.extend(base_bundle_manifests)
    for manifest_path in inherit_config_from:
        config_paths.extend(_manifest_member_paths(manifest_path, role="config"))

    return {
        "sources": _require_files(source_paths, label="source files"),
        "configs": _require_files(config_paths, label="configuration files"),
        "references": _require_files(
            _design_reference_paths(design, repo_root),
            label="referenced data inputs",
        ),
    }


def freeze_formal_dynamic_source_bundle(
    *,
    design_path: Path,
    base_bundle_manifests: Sequence[Path],
    output_dir: Path,
    repo_root: Path,
    extra_source_paths: Sequence[Path],
    inherit_config_from: Sequence[Path] = (),
) -> dict[str, Any]:
    """Create or byte-verify the immutable formal dynamic source bundle."""

    plan = plan_formal_dynamic_bundle(
        design_path=design_path,
        base_bundle_manifests=base_bundle_manifests,
        output_dir=output_dir,
        repo_root=repo_root,
        extra_source_paths=extra_source_paths,
        inherit_config_from=inherit_config_from,
    )
    command = (
        "PYTHONHASHSEED=0; run archived "
        "neurooracle/scripts/run_formal_dynamic_generalization.py --stage all; "
        f"exact options frozen in {design_path.resolve()}"
    )
    return freeze_source_bundle(
        source_paths=plan["sources"],
        config_paths=plan["configs"],
        reference_paths=plan["references"],
        output_dir=output_dir,
        command=command,
        repo_root=repo_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--base-bundle-manifest",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--extra-source", action="append", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    base_manifests = args.base_bundle_manifest or [
        DEFAULT_STATIC_BASE_BUNDLE,
        DEFAULT_DYNAMIC_BASE_BUNDLE,
    ]
    extra_sources = args.extra_source or [ROOT / value for value in DEFAULT_EXTRA_SOURCES]
    manifest = freeze_formal_dynamic_source_bundle(
        design_path=args.design,
        base_bundle_manifests=base_manifests,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        extra_source_paths=extra_sources,
        inherit_config_from=[DEFAULT_STATIC_BASE_BUNDLE],
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "bundled_files": len(manifest["files"]),
                "referenced_files": len(manifest["references"]),
                "referenced_bytes": sum(
                    int(record["bytes"]) for record in manifest["references"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
