"""Freeze exact code, configuration, snapshots, and KGE inputs for v3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from neurooracle.scripts.freeze_experiment_source_bundle import freeze_source_bundle
from neurooracle.scripts.prepare_hindcasting_v3_execution import (
    V3_ROOT,
    verify_execution_plan,
)
from neurooracle.src.experiment_source_bundle import verify_source_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    V3_ROOT
    / "execution/locked_r1_two_tasks_20260826/execution_plan.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {path.resolve() for path in paths},
        key=lambda value: str(value).casefold(),
    )


def _python_sources(repo_root: Path) -> list[Path]:
    sources: list[Path] = []
    excluded = {"data", ".frozen", "__pycache__", ".pytest_cache"}
    for top in (repo_root / "neurooracle", repo_root / "core"):
        for directory, child_dirs, filenames in os.walk(top, topdown=True):
            directory_path = Path(directory)
            child_dirs[:] = [
                name
                for name in child_dirs
                if name not in excluded
                and not (directory_path / name).is_symlink()
            ]
            sources.extend(
                directory_path / name
                for name in filenames
                if name.endswith(".py")
            )
    return _deduplicate(sources)


def _runtime_package_assets(repo_root: Path) -> list[Path]:
    """Return small non-Python files read by imported runtime modules."""

    source_root = repo_root / "neurooracle" / "src"
    assets = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in {".py", ".pyc", ".pyo"}
        and "__pycache__" not in path.parts
    ]
    assets.append(
        repo_root
        / "neurooracle/data/case_study_reaudit/full_graph_v3/RUBRIC.md"
    )
    return _deduplicate(assets)


def _plan_paths(plan_path: Path) -> dict[str, list[Path]]:
    plan_path = plan_path.resolve()
    verified = verify_execution_plan(plan_path)
    plan = _read_json(plan_path)
    bundle_manifest = Path(verified["source_bundle_manifest"]).resolve()

    configs: list[Path] = [
        plan_path,
        plan_path.with_name("execution_plan.lock.json"),
        Path(plan["matrix"]["path"]),
        Path(plan["protocol"]["path"]),
        Path(plan["protocol"]["lock_path"]),
        Path(plan["llm_api"]["routing_policy_if_required"]["path"]),
        Path(plan["llm_api"]["routing_policy_if_required"]["lock_path"]),
        Path(plan["cohort"]["path"]),
        Path(plan["cohort"]["lock_path"]),
        ROOT / "pyproject.toml",
    ]
    cohort = _read_json(Path(plan["cohort"]["path"]))
    release_manifest = Path(cohort["kg_release"]["manifest_path"]).resolve()
    configs.extend(
        (
            release_manifest,
            release_manifest.with_name("release.lock.json"),
            release_manifest.with_name("CURRENT_STATE.json"),
        )
    )
    selection = cohort.get("method_blind_selection_evidence") or {}
    preflight = selection.get("preflight") or {}
    if preflight.get("path"):
        configs.append(Path(preflight["path"]))
    for record in (selection.get("formal_audits") or {}).values():
        manifest_path = Path(record["path"]).resolve()
        configs.append(manifest_path)
        manifest = _read_json(manifest_path)
        matrix = (manifest.get("locked_matrix") or {}).get("path")
        if matrix:
            configs.append(Path(matrix))

    assets = cohort["execution_assets"]
    pipeline = assets["eligibility_pipeline"]
    configs.extend((Path(pipeline["path"]), Path(pipeline["lock_path"])))
    kge = assets["kge"]
    configs.extend((Path(kge["manifest"]["path"]), Path(kge["lock"]["path"])))

    references: list[Path] = [Path(plan["inputs"]["future_claims"]["path"])]
    for row in kge["year_assets"]:
        snapshot = row["snapshot"]
        configs.extend(
            (
                Path(row["asset"]["path"]),
                Path(row["report"]["path"]),
                Path(snapshot["manifest"]["path"]),
            )
        )
        references.extend(
            (
                Path(snapshot["knowledge_graph"]["path"]),
                Path(snapshot["extracted_claims"]["path"]),
                Path(row["checkpoint"]["path"]),
            )
        )

    result = {
        "sources": _deduplicate(
            [*_python_sources(ROOT), *_runtime_package_assets(ROOT)]
        ),
        "configs": _deduplicate(configs),
        "references": _deduplicate(references),
        "output_dir": [bundle_manifest.parent],
    }
    for label in ("sources", "configs", "references"):
        missing = [path for path in result[label] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {label}: {missing[0]}")
    return result


def freeze_v3_bundle(plan_path: Path) -> dict[str, Any]:
    paths = _plan_paths(plan_path)
    output_dir = paths["output_dir"][0]
    command = (
        "PYTHONHASHSEED=0; run archived "
        "neurooracle/scripts/run_hindcasting_v3.py --mode formal --stage all; "
        f"exact options locked in {plan_path.resolve()}"
    )
    manifest = freeze_source_bundle(
        source_paths=paths["sources"],
        config_paths=paths["configs"],
        reference_paths=paths["references"],
        output_dir=output_dir,
        command=command,
        repo_root=ROOT,
    )
    verification = verify_source_bundle(
        output_dir / "bundle_manifest.json",
        require_live_source=True,
        verify_references=False,
    )
    return {
        "status": "locked_source_bundle",
        "output_dir": str(output_dir),
        "manifest": str((output_dir / "bundle_manifest.json").resolve()),
        "manifest_sha256": verification["manifest_sha256"],
        "bundled_files": len(manifest["files"]),
        "referenced_files": len(manifest["references"]),
        "referenced_bytes": sum(
            int(record["bytes"]) for record in manifest["references"]
        ),
        "archive_and_live_sources_verified": True,
        "references_hashed_during_freeze": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = freeze_v3_bundle(parse_args(argv).plan)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
