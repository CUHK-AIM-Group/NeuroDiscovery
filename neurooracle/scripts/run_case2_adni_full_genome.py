"""Build references, impute, and validate all Case Study 2 ADNI autosomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_ancestry import ROOT
from neurooracle.scripts.run_case2_adni_imputation import DEFAULT_BREF3_ROOT


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND\n")
        handle.write(subprocess.list2cmdline(command))
        handle.write("\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chrom",
        type=int,
        choices=range(1, 23),
        action="append",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "imputation" / "full",
    )
    parser.add_argument(
        "--status-name",
        default="full_genome_status.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    chromosomes = args.chrom or list(range(1, 23))
    status_path = args.output_root / args.status_name
    status: dict[str, object] = {
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "chromosomes_requested": chromosomes,
        "completed_chromosomes": [],
        "current_chromosome": None,
        "current_stage": None,
        "state": "running",
    }
    if status_path.exists() and not args.force:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_at"] = previous.get("started_at", status["started_at"])
    args.output_root.mkdir(parents=True, exist_ok=True)

    def save_status() -> None:
        status["updated_at"] = timestamp()
        status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    save_status()
    completed: list[int] = []
    try:
        for chrom in chromosomes:
            chromosome_root = args.output_root / f"chr{chrom}"
            reference = (
                DEFAULT_BREF3_ROOT
                / f"chr{chrom}.1kg.phase3.v5a.b37.bref3"
            )
            status["current_chromosome"] = chrom
            if args.force or not reference.exists():
                status["current_stage"] = "reference"
                save_status()
                command = [
                    sys.executable,
                    "-m",
                    "neurooracle.scripts.prepare_case2_adni_bref3_reference",
                    "--chrom",
                    str(chrom),
                ]
                if args.force:
                    command.append("--force")
                run_command(
                    command,
                    args.output_root / "logs" / f"chr{chrom}.reference.log",
                )
            if args.force or not (chromosome_root / "summary.json").exists():
                status["current_stage"] = "imputation"
                save_status()
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "neurooracle.scripts.run_case2_adni_imputation",
                        "--chrom",
                        str(chrom),
                        "--threads",
                        str(args.threads),
                        "--memory-gb",
                        str(args.memory_gb),
                    ],
                    args.output_root / "logs" / f"chr{chrom}.imputation.log",
                )
            if args.force or not (chromosome_root / "validation.json").exists():
                status["current_stage"] = "validation"
                save_status()
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "neurooracle.scripts.validate_case2_adni_imputation",
                        "--root",
                        str(chromosome_root),
                        "--chrom",
                        str(chrom),
                    ],
                    args.output_root / "logs" / f"chr{chrom}.validation.log",
                )
            completed.append(chrom)
            status["completed_chromosomes"] = completed
            save_status()
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        save_status()
        raise
    status["current_chromosome"] = None
    status["current_stage"] = None
    status["state"] = "completed"
    status["completed_at"] = timestamp()
    save_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
