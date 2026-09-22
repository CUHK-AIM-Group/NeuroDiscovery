"""Reapply a frozen cross-release policy bundle to closed-loop benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


SCHEMA = "cross-release-case-study-rerun.v1"
ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "core" / "scripts" / "case_study_closed_loop.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source_closure_path(source_run_manifest: Path) -> Path:
    task_dir = source_run_manifest.parent.parent
    standard = task_dir / "closure_manifest.json"
    if standard.is_file():
        return standard
    legacy = task_dir / "tables" / "biomarker_closure_manifest.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"source closure manifest is missing under {task_dir}")


def _compact_benchmark_manifest(
    benchmark: dict[str, Any], benchmark_path: Path
) -> dict[str, Any]:
    """Retain every audit field without duplicating large per-record payloads."""

    compact = json.loads(json.dumps(benchmark))
    delta = dict(compact.get("experimental_kg_delta") or {})
    overlays = list(delta.pop("overlays", []) or [])
    if overlays:
        delta["overlay_records_elided_from_closure_manifest"] = len(overlays)
        delta["full_overlay_bundle_in_run_manifest"] = str(benchmark_path)
    compact["experimental_kg_delta"] = delta
    compact["artifacts"] = {
        "full_run_manifest": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "bytes": benchmark_path.stat().st_size,
        }
    }
    compact["manifest_path"] = str(benchmark_path)
    compact["manifest_sha256"] = sha256_file(benchmark_path)
    return compact


def write_closure_manifest(
    *,
    source_run_manifest: Path,
    result_run_manifest: Path,
    policy_manifest_path: Path,
    policy_manifest: dict[str, Any],
    track: str,
    task: str,
    output_path: Path,
) -> None:
    source_closure_path = _source_closure_path(source_run_manifest)
    source_closure = load_json(source_closure_path)
    result_benchmark = load_json(result_run_manifest)
    payload = {
        "schema_version": "cross-release-case-study-closure.v1",
        "created_at": utc_now(),
        "status": "complete",
        "task": task,
        "track": track,
        "source_closure_manifest": {
            "path": str(source_closure_path),
            "sha256": sha256_file(source_closure_path),
        },
        "canonical_release": source_closure.get("canonical_release"),
        "table_manifest": source_closure.get("table_manifest"),
        "benchmark_manifest": _compact_benchmark_manifest(
            result_benchmark, result_run_manifest
        ),
        "protocol": source_closure.get("protocol"),
        "cross_release_policy": {
            "manifest_path": str(policy_manifest_path),
            "manifest_sha256": sha256_file(policy_manifest_path),
            "track": track,
            "profile": policy_manifest["tracks"][track]["profile"],
            "profile_id": policy_manifest["tracks"][track]["profile_id"],
            "selection_protocol": policy_manifest["selection_protocol"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def reuse_model_robustness_manifest(
    *, source_run_manifest: Path, target_task_dir: Path
) -> dict[str, Any] | None:
    source_task_dir = source_run_manifest.parent.parent
    source = source_task_dir / "model_robustness" / "manifest.json"
    if not source.is_file():
        return None
    target = target_task_dir / "model_robustness" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "path": str(target),
        "sha256": sha256_file(target),
        "reuse_reason": "ranking policy changed; fitted-model robustness evidence unchanged",
    }


def _input_path(run_manifest: dict[str, Any], name: str) -> Path | None:
    frozen = run_manifest.get("frozen_discovery") or {}
    inputs = frozen.get("inputs") or {}
    descriptor = inputs.get(name) or {}
    raw = descriptor.get("path")
    return Path(raw) if raw else None


def build_command(
    *,
    python: Path,
    task: str,
    run_manifest: dict[str, Any],
    policy_path: Path,
    output_dir: Path,
    seed: int,
) -> list[str]:
    public = _input_path(run_manifest, "public_candidates")
    internal = _input_path(run_manifest, "internal_outcomes")
    if public is None or internal is None:
        raise ValueError(f"{task} source run lacks frozen discovery inputs")
    command = [
        str(python),
        str(RUNNER),
        "--task",
        task,
        "--public-candidates",
        str(public),
        "--internal-outcomes",
        str(internal),
        "--neurodiscovery-config",
        str(policy_path),
        "--output-dir",
        str(output_dir),
        "--factor-fields",
        *map(str, run_manifest["factor_fields"]),
        "--methods",
        *map(str, run_manifest["methods"]),
        "--trials",
        str(int(run_manifest["trials"])),
        "--seed",
        str(int(seed)),
        "--budgets",
        *map(str, run_manifest["budgets"]),
        "--recall-targets",
        *map(str, run_manifest["recall_targets"]),
    ]
    external = (run_manifest.get("external") or {}).get("path")
    if external:
        command.extend(["--external-outcomes", str(external)])
    precomputed = (run_manifest.get("precomputed_baseline_rankings") or {}).get(
        "path"
    )
    search_policies = _input_path(run_manifest, "search_policies")
    if precomputed:
        command.extend(["--precomputed-rankings-manifest", str(precomputed)])
    elif search_policies is not None:
        command.extend(["--search-policies", str(search_policies)])
    else:
        raise ValueError(f"{task} has neither precomputed rankings nor search policies")
    return command


def _selected(values: Iterable[str] | None, available: Iterable[str]) -> list[str]:
    options = list(available)
    if not values:
        return options
    wanted = set(values)
    missing = sorted(wanted - set(options))
    if missing:
        raise ValueError(f"unknown selections: {missing}")
    return [value for value in options if value in wanted]


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    policy_manifest_path = args.policy_manifest.resolve()
    policy_manifest = load_json(policy_manifest_path)
    if policy_manifest.get("status") != "frozen":
        raise ValueError("policy bundle is not frozen")
    benchmark_root = Path(policy_manifest["target_release"]["benchmark_root"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _selected(args.tasks, policy_manifest["tasks"].keys())
    tracks = _selected(args.tracks, policy_manifest["tracks"].keys())
    python = args.python.resolve() if args.python else Path(sys.executable).resolve()

    jobs: list[dict[str, Any]] = []
    for track in tracks:
        for task in tasks:
            source_manifest_path = (
                benchmark_root / task / "benchmark" / "run_manifest.json"
            )
            run_manifest = load_json(source_manifest_path)
            policy_path = Path(
                policy_manifest["tasks"][task]["policies"][track]["path"]
            )
            output_dir = output_root / track / task / "benchmark"
            command = build_command(
                python=python,
                task=task,
                run_manifest=run_manifest,
                policy_path=policy_path,
                output_dir=output_dir,
                seed=args.seed,
            )
            jobs.append(
                {
                    "track": track,
                    "task": task,
                    "source_run_manifest": str(source_manifest_path),
                    "source_run_manifest_sha256": sha256_file(source_manifest_path),
                    "policy": str(policy_path),
                    "policy_sha256": sha256_file(policy_path),
                    "output_dir": str(output_dir),
                    "command": command,
                    "status": "planned",
                }
            )

    suite = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "planned" if args.dry_run else "running",
        "policy_manifest": {
            "path": str(policy_manifest_path),
            "sha256": sha256_file(policy_manifest_path),
        },
        "python": str(python),
        "runner": {"path": str(RUNNER), "sha256": sha256_file(RUNNER)},
        "seed": int(args.seed),
        "tasks": tasks,
        "tracks": tracks,
        "jobs": jobs,
    }
    suite_path = output_root / "rerun_suite_manifest.json"

    def save() -> None:
        suite["updated_at"] = utc_now()
        suite_path.write_text(
            json.dumps(suite, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    save()
    if args.dry_run:
        return {**suite, "manifest_path": str(suite_path)}

    for job in jobs:
        output_dir = Path(job["output_dir"])
        completed = output_dir / "run_manifest.json"
        if completed.is_file() and not args.force:
            job["status"] = "reused_complete"
            job["result_manifest_sha256"] = sha256_file(completed)
            closure_path = output_dir.parent / "closure_manifest.json"
            write_closure_manifest(
                source_run_manifest=Path(job["source_run_manifest"]),
                result_run_manifest=completed,
                policy_manifest_path=policy_manifest_path,
                policy_manifest=policy_manifest,
                track=str(job["track"]),
                task=str(job["task"]),
                output_path=closure_path,
            )
            robustness = reuse_model_robustness_manifest(
                source_run_manifest=Path(job["source_run_manifest"]),
                target_task_dir=output_dir.parent,
            )
            job["closure_manifest"] = str(closure_path)
            job["closure_manifest_sha256"] = sha256_file(closure_path)
            job["model_robustness"] = robustness
            save()
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "runner.stdout.log"
        stderr_path = output_dir / "runner.stderr.log"
        job["status"] = "running"
        job["started_at"] = utc_now()
        save()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                job["command"], cwd=ROOT, stdout=stdout, stderr=stderr
            )
        job["finished_at"] = utc_now()
        job["returncode"] = int(result.returncode)
        job["stdout"] = str(stdout_path)
        job["stderr"] = str(stderr_path)
        if result.returncode != 0:
            job["status"] = "failed"
            suite["status"] = "failed"
            save()
            raise RuntimeError(f"rerun failed for {job['track']} / {job['task']}")
        if not completed.is_file():
            raise RuntimeError(f"rerun produced no manifest: {completed}")
        job["status"] = "complete"
        job["result_manifest_sha256"] = sha256_file(completed)
        closure_path = output_dir.parent / "closure_manifest.json"
        write_closure_manifest(
            source_run_manifest=Path(job["source_run_manifest"]),
            result_run_manifest=completed,
            policy_manifest_path=policy_manifest_path,
            policy_manifest=policy_manifest,
            track=str(job["track"]),
            task=str(job["task"]),
            output_path=closure_path,
        )
        robustness = reuse_model_robustness_manifest(
            source_run_manifest=Path(job["source_run_manifest"]),
            target_task_dir=output_dir.parent,
        )
        job["closure_manifest"] = str(closure_path)
        job["closure_manifest_sha256"] = sha256_file(closure_path)
        job["model_robustness"] = robustness
        save()

    suite["status"] = "complete"
    save()
    seal_sha = sha256_file(suite_path)
    (output_root / "rerun_suite_manifest.sha256").write_text(
        f"{seal_sha}  {suite_path.name}\n", encoding="ascii"
    )
    return {**suite, "manifest_path": str(suite_path), "manifest_sha256": seal_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--tracks", nargs="+", choices=("static_ablation", "closed_loop")
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    result = run_suite(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
