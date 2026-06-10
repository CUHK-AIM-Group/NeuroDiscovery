from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_temporal_kg_snapshot import build_snapshot
from case3_general_hindcasting_eval import evaluate
from case3_open_path_generator import generate as generate_open_paths


@dataclass(frozen=True)
class Window:
    freeze_year: int
    future_start_year: int
    future_end_year: int

    @property
    def label(self) -> str:
        return f"kg{self.freeze_year}_to_{self.future_start_year}_{self.future_end_year}"


DEFAULT_WINDOWS = (
    Window(2016, 2017, 2018),
    Window(2018, 2019, 2020),
    Window(2020, 2021, 2022),
    Window(2022, 2023, 2024),
)


def parse_window(raw: str) -> Window:
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("window must be freeze:start:end")
    freeze, start, end = (int(p) for p in parts)
    if not (freeze < start <= end):
        raise argparse.ArgumentTypeError("window must satisfy freeze < start <= end")
    return Window(freeze, start, end)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_hypotheses(
    *,
    kg_path: Path,
    output_path: Path,
    max_hops: int,
    min_hops: int,
    metapath_min_domains: int,
    max_paths: int,
    max_seeds: int,
    target_per_task: int | None,
    tasks: str | None,
    chains: str | None,
    generator: str,
    max_candidates: int,
    per_mediator_neighbor_limit: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.is_file() and not force:
        return load_json(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if generator == "open-path":
        return generate_open_paths(
            kg_path=kg_path,
            output_path=output_path,
            max_candidates=max_candidates,
            per_mediator_neighbor_limit=per_mediator_neighbor_limit,
            min_score=0.20,
        )
    cmd = [
        sys.executable,
        "-m",
        "neurooracle.src.hypothesis_cli",
        "--graph",
        str(kg_path),
        "batch",
        "--output",
        str(output_path),
        "--max-hops",
        str(max_hops),
        "--min-hops",
        str(min_hops),
        "--metapath-min-domains",
        str(metapath_min_domains),
        "--max-paths",
        str(max_paths),
        "--max-seeds",
        str(max_seeds),
    ]
    if target_per_task is not None:
        cmd.extend(["--target-per-task", str(target_per_task)])
    if tasks is not None:
        cmd.extend(["--tasks", tasks])
    if chains is not None:
        cmd.extend(["--chains", chains])
    subprocess.run(cmd, check=True)
    return load_json(output_path)


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Case Study 3 General Rolling Hindcasting",
        "",
        "This is the corrected Case Study 3 evaluation: it does not restrict hypotheses to gene-imaging-disease.",
        "",
        "| Freeze | Future | Hypotheses | Top-100 endpoint hits | Top-100 any hits | Top-100 random any | Top-1000 endpoint hits | Top-1000 any hits |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        top100 = row["topk"].get("100", {}).get("observed", {})
        rand100 = row["topk"].get("100", {}).get("random_same_hypothesis_pool", {})
        top1000 = row["topk"].get("1000", {}).get("observed", {})
        lines.append(
            "| {freeze} | {future} | {n} | {e100} | {a100} | {r100} | {e1000} | {a1000} |".format(
                freeze=row["freeze_year"],
                future=f"{row['future_start_year']}-{row['future_end_year']}",
                n=row["n_hypotheses"],
                e100=top100.get("endpoint_hits", "NA"),
                a100=top100.get("any_future_hits", "NA"),
                r100=_fmt(rand100.get("mean_any_future_hits")),
                e1000=top1000.get("endpoint_hits", "NA"),
                a1000=top1000.get("any_future_hits", "NA"),
            )
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected general Case Study 3 rolling hindcasting.")
    parser.add_argument("--input-dir", type=Path, default=Path("neurooracle/data/full_snapshot_v1"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("neurooracle/data/snapshots"))
    parser.add_argument("--output-root", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/general_rolling_full_snapshot_v1"))
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--metapath-min-domains", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--max-seeds", type=int, default=25)
    parser.add_argument("--target-per-task", type=int, default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--chains", default=None)
    parser.add_argument("--generator", choices=("open-path", "phase3-batch"), default="open-path")
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--per-mediator-neighbor-limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for window in args.windows:
        snapshot_dir = args.snapshot_root / f"kg_{window.freeze_year}_from_full_snapshot_v1"
        manifest_path = snapshot_dir / "manifest.json"
        if args.force or not manifest_path.is_file():
            print(f"[snapshot] building KG_{window.freeze_year}", flush=True)
            build_snapshot(args.input_dir, snapshot_dir, window.freeze_year)
        else:
            print(f"[snapshot] using KG_{window.freeze_year}", flush=True)

        run_dir = args.output_root / window.label
        hypotheses_path = run_dir / "hypotheses_general_raw.json"
        print(f"[generate] KG_{window.freeze_year} -> {hypotheses_path}", flush=True)
        generate_hypotheses(
            kg_path=snapshot_dir / "knowledge_graph.json",
            output_path=hypotheses_path,
            max_hops=args.max_hops,
            min_hops=args.min_hops,
            metapath_min_domains=args.metapath_min_domains,
            max_paths=args.max_paths,
            max_seeds=args.max_seeds,
            target_per_task=args.target_per_task,
            tasks=args.tasks,
            chains=args.chains,
            generator=args.generator,
            max_candidates=args.max_candidates,
            per_mediator_neighbor_limit=args.per_mediator_neighbor_limit,
            force=args.force,
        )
        print(f"[evaluate] {window.label}", flush=True)
        metrics = evaluate(
            kg_path=snapshot_dir / "knowledge_graph.json",
            hypotheses_path=hypotheses_path,
            future_claims_path=args.input_dir / "extracted_claims.jsonl",
            output_dir=run_dir / "general_hindcasting",
            freeze_year=window.freeze_year,
            future_start_year=window.future_start_year,
            future_end_year=window.future_end_year,
            top_ks=args.top_k,
            random_trials=args.random_trials,
            seed=31,
        )
        rows.append({
            "freeze_year": window.freeze_year,
            "future_start_year": window.future_start_year,
            "future_end_year": window.future_end_year,
            "run_dir": str(run_dir),
            "hypotheses_path": str(hypotheses_path),
            "n_hypotheses": metrics["n_hypotheses"],
            "future_stats": metrics["future_stats"],
            "topk": metrics["topk"],
            "by_type": metrics["by_type"],
        })
    write_summary(args.output_root, rows)
    print(json.dumps({"output_root": str(args.output_root), "n_windows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
