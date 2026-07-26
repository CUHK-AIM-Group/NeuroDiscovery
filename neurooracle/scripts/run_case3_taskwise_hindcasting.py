from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from neurooracle.src.case3_subtasks import (
    Case3Subtask,
    select_case3_subtasks,
)

try:
    from .build_temporal_kg_snapshot import build_snapshot
    from .run_case3_general_hindcasting import DEFAULT_WINDOWS
except ImportError:  # Direct script execution.
    from build_temporal_kg_snapshot import build_snapshot
    from run_case3_general_hindcasting import DEFAULT_WINDOWS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "neurooracle/data/full_v2"
DEFAULT_SNAPSHOT_ROOT = ROOT / "neurooracle/data/experiments/case3/snapshots_full_v2_5year_2016_2020"
DEFAULT_OUTPUT_ROOT = ROOT / "neurooracle/data/experiments/case3/case3_all_tasks_5year_phase3_seed10_20260714"
TOP_KS = (10, 20, 50, 100, 200, 500, 1000)


def _window_args() -> list[str]:
    return [
        f"{window.freeze_year}:{window.future_start_year}:{window.future_end_year}"
        for window in DEFAULT_WINDOWS
    ]


def _command_for_subtask(
    *,
    subtask: Case3Subtask,
    input_dir: Path,
    snapshot_root: Path,
    output_root: Path,
    target_per_task: int,
    random_trials: int,
    seed: int,
    force: bool,
) -> list[str]:
    subtask_output = output_root / subtask.name
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_case3_general_hindcasting.py")),
        "--input-dir",
        str(input_dir),
        "--snapshot-root",
        str(snapshot_root),
        "--output-root",
        str(subtask_output),
        "--windows",
        *_window_args(),
        "--generator",
        "phase3-batch",
        "--tasks",
        subtask.name,
        "--chains",
        "",
        "--target-per-task",
        str(target_per_task),
        "--top-k",
        *(str(k) for k in TOP_KS),
        "--random-trials",
        str(random_trials),
        "--seed",
        str(seed),
    ]
    if force:
        command.append("--force")
    return command


def _run_subtask(
    *,
    subtask: Case3Subtask,
    input_dir: Path,
    snapshot_root: Path,
    output_root: Path,
    target_per_task: int,
    random_trials: int,
    seed: int,
    force: bool,
    max_retries: int,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    command = _command_for_subtask(
        subtask=subtask,
        input_dir=input_dir,
        snapshot_root=snapshot_root,
        output_root=output_root,
        target_per_task=target_per_task,
        random_trials=random_trials,
        seed=seed,
        force=force,
    )
    subtask_output = output_root / subtask.name
    log_path = output_root / "logs" / f"{subtask.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(max_retries + 1):
        with log_path.open("w" if attempt == 0 else "a", encoding="utf-8") as log:
            if attempt:
                log.write(f"\n[retry] attempt {attempt + 1}/{max_retries + 1}\n")
                log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode == 0:
            break
        if attempt < max_retries:
            time.sleep(retry_wait_seconds * (attempt + 1))
    assert completed is not None
    return {
        "subtask": subtask.name,
        "task": subtask.name,
        "kind": subtask.kind,
        "signature": subtask.signature,
        "returncode": completed.returncode,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
        "output": str(subtask_output),
        "command": command,
        "attempts": attempt + 1,
    }


def _select_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Case3Subtask, ...]:
    requested = list(args.tasks) if args.tasks is not None else None

    try:
        selected = select_case3_subtasks(requested, exclude=args.exclude_tasks)
    except KeyError as exc:
        parser.error(str(exc))
    if not selected:
        parser.error("no Case Study 3 subtasks selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 15 registered Case Study 3 tasks over fixed five-year hindcasting windows."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None, help="Case Study 3 task names. Default: all 15.")
    parser.add_argument("--exclude-tasks", nargs="*", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--target-per-task", type=int, default=1000)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    subtasks = _select_from_args(parser, args)
    if args.list_tasks:
        print(json.dumps([subtask.to_dict() for subtask in subtasks], indent=2, ensure_ascii=False))
        return
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.snapshot_root.mkdir(parents=True, exist_ok=True)

    snapshot_manifests: dict[str, Any] = {}
    if not args.dry_run:
        for window in DEFAULT_WINDOWS:
            snapshot_dir = args.snapshot_root / f"kg_{window.freeze_year}"
            manifest_path = snapshot_dir / "manifest.json"
            if args.force or not manifest_path.is_file():
                print(f"[snapshot] KG_{window.freeze_year} -> {snapshot_dir}", flush=True)
                manifest = build_snapshot(args.input_dir, snapshot_dir, window.freeze_year)
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                print(f"[snapshot] reuse KG_{window.freeze_year}", flush=True)
            snapshot_manifests[str(window.freeze_year)] = manifest

    run_manifest: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "snapshot_root": str(args.snapshot_root),
        "output_root": str(args.output_root),
        "windows": [window.__dict__ for window in DEFAULT_WINDOWS],
        "subtasks": [subtask.to_dict() for subtask in subtasks],
        "tasks": [subtask.name for subtask in subtasks if subtask.kind == "task"],
        "chains": [],
        "jobs": args.jobs,
        "target_per_task": args.target_per_task,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "snapshots": snapshot_manifests,
        "subtask_runs": [],
        "task_runs": [],
    }
    manifest_path = args.output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        planned = []
        for subtask in subtasks:
            planned.append({
                "subtask": subtask.name,
                "task": subtask.name,
                "kind": subtask.kind,
                "signature": subtask.signature,
                "status": "planned",
                "output": str(args.output_root / subtask.name),
                "log": str(args.output_root / "logs" / f"{subtask.name}.log"),
                "command": _command_for_subtask(
                    subtask=subtask,
                    input_dir=args.input_dir,
                    snapshot_root=args.snapshot_root,
                    output_root=args.output_root,
                    target_per_task=args.target_per_task,
                    random_trials=args.random_trials,
                    seed=args.seed,
                    force=args.force,
                ),
            })
        run_manifest["subtask_runs"] = planned
        run_manifest["task_runs"] = planned
        manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output_root": str(args.output_root), "planned_tasks": len(planned)}, indent=2))
        return

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _run_subtask,
                subtask=subtask,
                input_dir=args.input_dir,
                snapshot_root=args.snapshot_root,
                output_root=args.output_root,
                target_per_task=args.target_per_task,
                random_trials=args.random_trials,
                seed=args.seed,
                force=args.force,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
            ): subtask
            for subtask in subtasks
        }
        for future in as_completed(futures):
            subtask = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - subprocess boundary
                result = {
                    "subtask": subtask.name,
                    "task": subtask.name,
                    "kind": subtask.kind,
                    "signature": subtask.signature,
                    "returncode": -1,
                    "error": str(exc),
                    "finished": datetime.now().isoformat(timespec="seconds"),
                }
            result["status"] = "completed" if result["returncode"] == 0 else "failed"
            run_manifest["subtask_runs"].append(result)
            run_manifest["subtask_runs"].sort(key=lambda row: row["subtask"])
            run_manifest["task_runs"] = list(run_manifest["subtask_runs"])
            manifest_path.write_text(
                json.dumps(run_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[{result['kind']}] {result['subtask']}: returncode={result['returncode']}", flush=True)
            if result["returncode"] != 0:
                failures.append(result)
                if args.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    break

    aggregate_command = [
        sys.executable,
        str(Path(__file__).with_name("aggregate_case3_taskwise_hindcasting.py")),
        "--root",
        str(args.output_root),
    ]
    subprocess.run(aggregate_command, cwd=ROOT, check=True)
    run_manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    run_manifest["aggregate_command"] = aggregate_command
    run_manifest["completed_subtasks"] = sum(row["returncode"] == 0 for row in run_manifest["subtask_runs"])
    run_manifest["failed_subtasks"] = [row["subtask"] for row in failures]
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_root": str(args.output_root),
        "subtasks": len(subtasks),
        "completed": run_manifest["completed_subtasks"],
        "failed": run_manifest["failed_subtasks"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
