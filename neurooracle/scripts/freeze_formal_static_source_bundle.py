"""Freeze source, configuration, and data inputs for formal static runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.scripts.freeze_experiment_source_bundle import freeze_source_bundle
from neurooracle.src.experiment_source_bundle import (
    SOURCE_BUNDLE_SCHEMA,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "optimization_protocol_semantic_v4_20260814"
)
DEFAULT_DESIGN = PROTOCOL_ROOT / "formal_static_design_v4_20260814.json"
DEFAULT_BASE_BUNDLE = (
    ROOT / "neurooracle/.frozen/formal_static_endpoint_v7_v5/bundle_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "neurooracle/.frozen/formal_static_endpoint_v8"
DEFAULT_EXTRA_SOURCES = (
    "core/scripts/canonical_kg_release.py",
    "neurooracle/scripts/freeze_formal_static_design.py",
    "neurooracle/scripts/freeze_formal_static_source_bundle.py",
    "neurooracle/scripts/verify_formal_static_hindcasting.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {path.resolve() for path in paths}, key=lambda value: str(value).casefold()
    )


def _manifest_paths(manifest_path: Path, role: str) -> list[Path]:
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
        raise ValueError("base source bundle uses an incompatible schema")
    bundle_root = manifest_path.resolve().parent
    selected: list[Path] = []
    for record in manifest.get("files") or ():
        archived = (bundle_root / str(record.get("bundle_path") or "")).resolve()
        expected_size = int(record.get("bytes") or -1)
        expected_hash = str(record.get("sha256") or "").upper()
        if (
            not archived.is_file()
            or archived.stat().st_size != expected_size
            or sha256_file(archived) != expected_hash
        ):
            raise ValueError(f"base bundle source member is not intact: {archived}")
        if role in set(record.get("roles") or ()):
            selected.append(Path(str(record["source_path"])).resolve())
    return selected


def _record_path(record: Mapping[str, Any], root: Path) -> Path:
    value = str(record.get("path") or "")
    if not value:
        raise ValueError("frozen design contains an empty path record")
    return _resolve(value, root)


def plan_static_bundle(
    *,
    design_path: Path,
    base_bundle_manifest: Path,
    output_dir: Path,
    repo_root: Path,
    extra_source_paths: Sequence[Path],
) -> dict[str, list[Path]]:
    repo_root = repo_root.resolve()
    design_path = design_path.resolve()
    output_dir = output_dir.resolve()
    design = _load_object(design_path)
    if design.get("status") != "frozen_before_formal_generation":
        raise ValueError("formal static design is not frozen")
    declared = _resolve(
        str((design.get("reproducibility") or {}).get("source_bundle_manifest") or ""),
        repo_root,
    )
    if declared != output_dir / "bundle_manifest.json":
        raise ValueError("formal static design declares a different source bundle")

    sources = _manifest_paths(base_bundle_manifest, "source") + list(extra_source_paths)
    base_configs = _manifest_paths(base_bundle_manifest, "config")
    portable_configs = [
        path
        for path in base_configs
        if (
            "data\\atlas" in str(path).lower()
            or "data/atlas" in str(path).lower()
            or path.name in {"CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md", "RUBRIC.md", "pyproject.toml"}
        )
    ]
    temporal = design["temporal_inputs"]
    configs = portable_configs + [
        design_path,
        _record_path(design["policy_application"]["source_policy"], repo_root),
        _record_path(temporal["eligibility_manifest"], repo_root),
        _record_path(temporal["eligibility_matrix"], repo_root),
    ]

    canonical = design["canonical_release"]
    references = [
        _record_path(canonical[label], repo_root)
        for label in ("knowledge_graph", "extracted_claims", "current_state")
    ]
    snapshot_root = _resolve(str(temporal["snapshot_root"]), repo_root)
    for year in temporal["freeze_years"]:
        snapshot = snapshot_root / f"kg_{int(year)}"
        references.extend(
            snapshot / name
            for name in ("knowledge_graph.json", "extracted_claims.jsonl", "manifest.json")
        )

    result = {
        "sources": _deduplicate(sources),
        "configs": _deduplicate(configs),
        "references": _deduplicate(references),
    }
    for label, paths in result.items():
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {label}: {missing[0]}")
    return result


def freeze_formal_static_source_bundle(
    *,
    design_path: Path,
    base_bundle_manifest: Path,
    output_dir: Path,
    repo_root: Path,
    extra_source_paths: Sequence[Path],
) -> dict[str, Any]:
    plan = plan_static_bundle(
        design_path=design_path,
        base_bundle_manifest=base_bundle_manifest,
        output_dir=output_dir,
        repo_root=repo_root,
        extra_source_paths=extra_source_paths,
    )
    return freeze_source_bundle(
        source_paths=plan["sources"],
        config_paths=plan["configs"],
        reference_paths=plan["references"],
        output_dir=output_dir,
        command=(
            "PYTHONHASHSEED=0; run archived "
            "neurooracle/scripts/run_formal_static_hindcasting.py --stage all; "
            f"exact options frozen in {design_path.resolve()}"
        ),
        repo_root=repo_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--base-bundle-manifest", type=Path, default=DEFAULT_BASE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--extra-source", action="append", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    extras = args.extra_source or [ROOT / value for value in DEFAULT_EXTRA_SOURCES]
    manifest = freeze_formal_static_source_bundle(
        design_path=args.design,
        base_bundle_manifest=args.base_bundle_manifest,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        extra_source_paths=extras,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "bundled_files": len(manifest["files"]),
        "referenced_files": len(manifest["references"]),
        "referenced_bytes": sum(int(row["bytes"]) for row in manifest["references"]),
    }, indent=2))


if __name__ == "__main__":
    main()
