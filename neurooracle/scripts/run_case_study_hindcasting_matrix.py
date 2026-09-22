"""Run the generic hindcasting protocol across selected formal case studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
    write_release_manifest,
)
from neurooracle.scripts.build_temporal_kg_snapshot import (
    build_snapshot,
    snapshot_matches_input,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    DEFAULT_WINDOWS,
    Window,
    load_json,
    parse_window,
)
from neurooracle.src.case_studies import (
    CaseStudy,
    case_study_by_name,
    list_case_study_names,
)
from neurooracle.src.validation_protocols import HINDCASTING


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_SNAPSHOT_ROOT = (
    REPO / "neurooracle" / "data" / "experiments" / "hindcasting" / "snapshots_full_v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPO / "neurooracle" / "data" / "experiments" / "hindcasting" / "matrix_current"
)


def select_case_studies(
    names: list[str] | None,
    exclude: list[str],
) -> tuple[CaseStudy, ...]:
    requested = names or list(list_case_study_names())
    excluded = set(exclude)
    unknown = sorted((set(requested) | excluded) - set(list_case_study_names()))
    if unknown:
        raise KeyError(f"unknown case-study IDs: {', '.join(unknown)}")
    seen: set[str] = set()
    selected: list[CaseStudy] = []
    for name in requested:
        if name in seen or name in excluded:
            continue
        seen.add(name)
        case = case_study_by_name(name)
        if not HINDCASTING.supports(case.name):
            raise ValueError(f"hindcasting does not support {case.name!r}")
        selected.append(case)
    return tuple(selected)


def _ensure_snapshots(
    input_dir: Path,
    snapshot_root: Path,
    windows: list[Window],
    *,
    force: bool,
) -> None:
    for window in windows:
        snapshot_dir = snapshot_root / f"kg_{window.freeze_year}"
        manifest_path = snapshot_dir / "manifest.json"
        reusable = False
        if manifest_path.is_file() and not force:
            reusable = snapshot_matches_input(
                load_json(manifest_path), input_dir, window.freeze_year
            )
        if not reusable:
            build_snapshot(input_dir, snapshot_dir, window.freeze_year)


def command_for_case_study(
    *,
    case: CaseStudy,
    input_dir: Path,
    snapshot_root: Path,
    output_root: Path,
    windows: list[Window],
    target_per_case_study: int | None,
    random_trials: int,
    seed: int,
    generate_only: bool,
    force_generation: bool,
    force_evaluation: bool,
    generation_pool_size: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_case_study_hindcasting.py")),
        case.name,
        "--input-dir",
        str(input_dir),
        "--snapshot-root",
        str(snapshot_root),
        "--output-root",
        str(output_root / case.name),
        "--windows",
        *(f"{w.freeze_year}:{w.future_start_year}:{w.future_end_year}" for w in windows),
        "--random-trials",
        str(random_trials),
        "--seed",
        str(seed),
    ]
    if target_per_case_study is not None:
        command.extend(["--target-per-case-study", str(target_per_case_study)])
    if generation_pool_size is not None:
        command.extend(["--generation-pool-size", str(generation_pool_size)])
    if generate_only:
        command.append("--generate-only")
    if force_generation:
        command.append("--force-generation")
    if force_evaluation:
        command.append("--force-evaluation")
    return command


def _run(case: CaseStudy, command: list[str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case.name}.log"
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished = datetime.now(timezone.utc)
    return {
        "case_study_id": case.name,
        "validation_protocol": HINDCASTING.name,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "log": str(log_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-study-ids", nargs="*", default=None)
    parser.add_argument("--exclude-case-study-ids", nargs="*", default=[])
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--target-per-case-study", type=int, default=None)
    parser.add_argument(
        "--generation-pool-size",
        type=int,
        default=None,
        help="Generate a wider ranking pool than the Top-K evaluation budget.",
    )
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--force-snapshots", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.generation_pool_size is not None and args.generation_pool_size < 1:
        parser.error("--generation-pool-size must be positive")
    if (
        args.generation_pool_size is not None
        and args.target_per_case_study is not None
        and args.generation_pool_size <= args.target_per_case_study
    ):
        parser.error(
            "--generation-pool-size must exceed --target-per-case-study "
            "so same-pool random ranking remains identifiable"
        )

    input_dir = args.input_dir.resolve()
    canonical_release = validate_canonical_kg_release(
        kg_path=input_dir / "knowledge_graph.json",
        claims_path=input_dir / "extracted_claims.jsonl",
        state_path=input_dir / "CURRENT_STATE.json",
        expected_sha256=CURRENT_CANONICAL_SHA256,
    )

    selected = select_case_studies(
        args.case_study_ids,
        args.exclude_case_study_ids,
    )
    if not selected:
        parser.error("no case studies selected")
    commands = {
        case.name: command_for_case_study(
            case=case,
            input_dir=input_dir,
            snapshot_root=args.snapshot_root.resolve(),
            output_root=args.output_root.resolve(),
            windows=args.windows,
            target_per_case_study=args.target_per_case_study,
            random_trials=args.random_trials,
            seed=args.seed,
            generate_only=args.generate_only,
            force_generation=args.force_generation,
            force_evaluation=args.force_evaluation,
            generation_pool_size=args.generation_pool_size,
        )
        for case in selected
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.snapshot_root.mkdir(parents=True, exist_ok=True)
    write_release_manifest(
        args.output_root / "canonical_kg_release.json", canonical_release
    )
    write_release_manifest(
        args.snapshot_root / "canonical_kg_release.json", canonical_release
    )
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_protocol": HINDCASTING.name,
        "case_studies": [
            {
                "id": case.name,
                "chinese_name": case.chinese_name,
                "english_name": case.english_name,
            }
            for case in selected
        ],
        "windows": [
            {
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
            }
            for window in args.windows
        ],
        "commands": commands,
        "generation_pool_size": args.generation_pool_size,
        "runs": [],
        "dry_run": args.dry_run,
        "canonical_release": canonical_release,
    }
    manifest_path = args.output_root / "run_manifest.json"
    if args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    _ensure_snapshots(
        input_dir,
        args.snapshot_root.resolve(),
        args.windows,
        force=args.force_snapshots,
    )
    workers = max(1, min(args.max_workers, len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run,
                case,
                commands[case.name],
                args.output_root / "logs",
            ): case
            for case in selected
        }
        for future in as_completed(futures):
            result = future.result()
            manifest["runs"].append(result)
            manifest["runs"].sort(key=lambda row: row["case_study_id"])
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    failed = [row["case_study_id"] for row in manifest["runs"] if row["status"] != "completed"]
    manifest["status"] = "completed" if not failed else "partial_failure"
    manifest["failed_case_studies"] = failed
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
