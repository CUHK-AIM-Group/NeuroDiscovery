"""Execute the preregistered static hindcasting benchmark with byte guards."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_bundle_member,
    verify_source_bundle,
)
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
)
DEFAULT_DESIGN = (
    DEFAULT_PROTOCOL_ROOT / "formal_static_comparison_design_endpoint_v7_20260813.json"
)
DEFAULT_BUNDLE = (
    ROOT
    / "neurooracle/.frozen/formal_static_endpoint_v7_v5/bundle_manifest.json"
)
STAGES = ("neurodiscovery", "baselines", "merge", "evaluate", "summarize")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_design(design_path: Path, workspace_root: Path) -> dict[str, Any]:
    design_path = design_path.resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("status") != "frozen_before_formal_generation":
        raise ValueError("formal design is not frozen")
    for record in (design.get("canonical_release") or {}).values():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = _resolve(str(record["path"]), workspace_root)
        if sha256_file(path) != str(record["sha256"]).upper():
            raise ValueError(f"canonical release hash mismatch: {path}")
    temporal = design["temporal_inputs"]
    for key in ("eligibility_manifest", "eligibility_matrix"):
        record = temporal[key]
        path = _resolve(str(record["path"]), workspace_root)
        if sha256_file(path) != str(record["sha256"]).upper():
            raise ValueError(f"temporal lock hash mismatch: {path}")
    policy = next(
        method
        for method in design["methods"]["primary"]
        if method["id"] == "neurodiscovery"
    )
    policy_path = _resolve(str(policy["policy_path"]), workspace_root)
    if sha256_file(policy_path) != str(policy["policy_sha256"]).upper():
        raise ValueError("frozen NeuroDiscovery policy hash mismatch")
    return design


def _derive_smoke_lock(
    formal_manifest_path: Path,
    output_root: Path,
) -> Path:
    formal = load_locked_hindcasting_eligibility(formal_manifest_path)
    selected = min(
        (
            key
            for key in formal.primary_windows
            if key[0] == "case1_transdiagnostic"
        ),
        key=lambda key: (-key[1], key),
    )
    smoke_root = output_root / "smoke_lock"
    matrix_path = smoke_root / "eligibility_matrix_locked.csv"
    manifest_path = smoke_root / "eligibility_manifest.json"
    smoke_root.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_study_id",
                "freeze_year",
                "future_start_year",
                "future_end_year",
                "analysis_tier",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_study_id": selected[0],
                "freeze_year": selected[1],
                "future_start_year": selected[2],
                "future_end_year": selected[3],
                "analysis_tier": "primary",
            }
        )
    payload = {
        "schema_version": "neurodiscovery-locked-hindcasting-eligibility.v2",
        "registered_at": _utc_now(),
        "status": "derived_smoke_only_not_formal_analysis",
        "primary_case_study_ids": [selected[0]],
        "primary_windows": 1,
        "locked_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256_file(matrix_path),
            "rows": 1,
        },
        "source_formal_eligibility_manifest": str(formal.manifest_path),
        "source_formal_eligibility_manifest_sha256": sha256_file(
            formal.manifest_path
        ),
        "performance_columns_consumed": False,
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def _run_command(
    *,
    name: str,
    command: Sequence[str],
    output_root: Path,
    bundle_manifest: Path,
    verify_references: bool,
    environment: dict[str, str],
    code_root: Path,
) -> dict[str, Any]:
    before = verify_source_bundle(
        bundle_manifest,
        require_live_source=False,
        verify_references=verify_references,
    )
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    started = _utc_now()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            list(command),
            cwd=code_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}; see {stderr_path}"
        )
    after = verify_source_bundle(
        bundle_manifest,
        require_live_source=False,
        verify_references=verify_references,
    )
    return {
        "stage": name,
        "started_at": started,
        "finished_at": _utc_now(),
        "command": list(command),
        "returncode": result.returncode,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "bundle_before": before,
        "bundle_after": after,
    }


def _commands(
    *,
    python: Path,
    output_root: Path,
    snapshot_root: Path,
    future_claims: Path,
    eligibility_manifest: Path,
    smoke: bool,
    force: bool,
    code_root: Path = ROOT,
) -> dict[str, list[str]]:
    common = [
        "--snapshot-root",
        str(snapshot_root),
        "--eligibility-manifest",
        str(eligibility_manifest),
        "--seeds",
        *(["0"] if smoke else [str(value) for value in range(10)]),
    ]
    target = "100" if smoke else "1000"
    pool = "120" if smoke else "1200"
    window_args: list[str] = []
    case_args: list[str] = []
    if smoke:
        lock = load_locked_hindcasting_eligibility(eligibility_manifest)
        case_id, freeze, start, end = next(iter(lock.primary_windows))
        window_args = ["--windows", f"{freeze}:{start}:{end}"]
        case_args = ["--case-study-ids", case_id]
    force_args = ["--force"] if force else []

    neuro_root = output_root / "neurodiscovery_generation"
    baseline_root = output_root / "frozen_baseline_generation"
    paired_root = output_root / "paired_generation"
    evaluation_root = output_root / "evaluation"
    summary_root = output_root / "summary"
    return {
        "neurodiscovery": [
            str(python),
            str(code_root / "neurooracle/scripts/generate_neurodiscovery_hindcasting_replicates.py"),
            *common,
            *case_args,
            *window_args,
            "--output-root",
            str(neuro_root),
            "--target-per-case-study",
            target,
            "--generation-pool-size",
            pool,
            "--seed-diversity-fraction",
            "0.35",
            "--candidate-pool-mode",
            "hybrid",
            "--task-scope-fraction",
            "0.5",
            "--evidence-frontier-fraction",
            "1.0",
            "--endpoint-canonical-quality-weight",
            "0.0",
            "--protect-general-top-k",
            "0",
            "--static-score-family",
            "legacy",
            *force_args,
        ],
        "baselines": [
            str(python),
            str(code_root / "neurooracle/scripts/generate_case_study_frozen_baselines.py"),
            *common,
            *case_args,
            *window_args,
            "--output-root",
            str(baseline_root),
            "--methods",
            "sciagents",
            "openscholar_rag",
            "--target-per-case-study",
            target,
            *force_args,
        ],
        "merge": [
            str(python),
            str(code_root / "neurooracle/scripts/merge_hindcasting_generation_manifests.py"),
            "--manifest",
            str(neuro_root / "generation_manifest.json"),
            "--manifest",
            str(baseline_root / "generation_manifest.json"),
            "--output-dir",
            str(paired_root),
            "--eligibility-manifest",
            str(eligibility_manifest),
        ],
        "evaluate": [
            str(python),
            str(code_root / "neurooracle/scripts/evaluate_case_study_frozen_baselines.py"),
            "--generation-root",
            str(paired_root),
            "--output-root",
            str(evaluation_root),
            "--snapshot-root",
            str(snapshot_root),
            "--future-claims",
            str(future_claims),
            "--methods",
            "neurodiscovery",
            "sciagents",
            "openscholar_rag",
            "--seeds",
            *(["0"] if smoke else [str(value) for value in range(10)]),
            *case_args,
            *window_args,
            "--top-k",
            *(["10", "20", "50", "100"] if smoke else ["10", "20", "50", "100", "200", "500", "1000"]),
            "--random-trials",
            "50" if smoke else "1000",
            *force_args,
        ],
        "summarize": [
            str(python),
            str(code_root / "neurooracle/scripts/summarize_hindcasting_replicates.py"),
            "--evaluation-root",
            str(evaluation_root),
            "--output-root",
            str(summary_root),
            "--reference-method",
            "neurodiscovery",
            "--metrics",
            "unique_primary_discoveries",
            "future_pair_recall",
            "primary_hits",
            "--eligibility-manifest",
            str(eligibility_manifest),
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    bundle_manifest = args.source_bundle_manifest.resolve()
    archive_root = (bundle_manifest.parent / "files").resolve()
    code_root = Path(__file__).resolve().parents[2]
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    if not executing_from_archive and not args.allow_live_source:
        raise RuntimeError(
            "formal runner must execute from the frozen bundle archive; use "
            "--allow-live-source only for diagnostics"
        )
    if executing_from_archive and code_root != archive_root:
        raise RuntimeError(
            f"archive code root mismatch: expected={archive_root} observed={code_root}"
        )
    verify_bundle_member(bundle_manifest, args.design.resolve())
    design = _verify_design(args.design, workspace_root)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else _resolve(str(design["output_root"]), workspace_root)
    )
    if args.smoke:
        output_root = output_root.parent / f"{output_root.name}_smoke"
    output_root.mkdir(parents=True, exist_ok=True)

    formal_eligibility = _resolve(
        str(design["temporal_inputs"]["eligibility_manifest"]["path"]),
        workspace_root,
    )
    eligibility = (
        _derive_smoke_lock(formal_eligibility, output_root)
        if args.smoke
        else formal_eligibility
    )
    commands = _commands(
        python=args.python.resolve(),
        output_root=output_root,
        snapshot_root=_resolve(
            str(design["temporal_inputs"]["snapshot_root"]), workspace_root
        ),
        future_claims=_resolve(
            str(design["canonical_release"]["extracted_claims"]["path"]),
            workspace_root,
        ),
        eligibility_manifest=eligibility,
        smoke=args.smoke,
        force=args.force,
        code_root=code_root,
    )
    selected_stages = list(STAGES) if args.stage == "all" else [args.stage]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(code_root), python_path) if value
    )
    records = []
    for stage in selected_stages:
        record = _run_command(
            name=stage,
            command=commands[stage],
            output_root=output_root,
            bundle_manifest=bundle_manifest,
            verify_references=not args.skip_reference_rehash,
            environment=environment,
            code_root=code_root,
        )
        records.append(record)
        _atomic_json(
            output_root / "execution_manifest.json",
            {
                "schema_version": "formal-static-hindcasting-execution.v1",
                "status": "running",
                "mode": "smoke" if args.smoke else "formal",
                "design_path": str(args.design.resolve()),
                "design_sha256": sha256_file(args.design.resolve()),
                "source_bundle_manifest": str(
                    bundle_manifest
                ),
                "source_bundle_manifest_sha256": sha256_file(
                    bundle_manifest
                ),
                "eligibility_manifest": str(eligibility.resolve()),
                "records": records,
            },
        )
    result = {
        "schema_version": "formal-static-hindcasting-execution.v1",
        "status": "complete",
        "mode": "smoke" if args.smoke else "formal",
        "design_path": str(args.design.resolve()),
        "design_sha256": sha256_file(args.design.resolve()),
        "source_bundle_manifest": str(bundle_manifest),
        "source_bundle_manifest_sha256": sha256_file(
            bundle_manifest
        ),
        "eligibility_manifest": str(eligibility.resolve()),
        "records": records,
    }
    _atomic_json(output_root / "execution_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--source-bundle-manifest", type=Path, default=DEFAULT_BUNDLE
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-live-source", action="store_true")
    parser.add_argument(
        "--skip-reference-rehash",
        action="store_true",
        help="Diagnostic only; formal runs should rehash all referenced inputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
