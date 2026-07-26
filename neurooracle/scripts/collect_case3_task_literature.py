"""Collect Case Study 3 task-conditioned literature candidates.

This script builds the literature pool for Hypothesis Hindcasting before claim
extraction. It runs Phase-2 task search in collect-only mode: abstracts
are cached and collection metadata is written, but no LLM extraction is called
and the KG is not modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.src.case3_subtasks import case3_subtasks
from neurooracle.src.chain_extract import run_chain_extraction


def _split_names(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return names


def _default_tasks() -> list[str]:
    return [subtask.name for subtask in case3_subtasks("task")]


def _ensure_graph(data_dir: Path, graph: Path, *, refresh_graph: bool) -> Path:
    target = data_dir / "knowledge_graph.json"
    if refresh_graph or not target.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(graph, target)
    return target


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CS3 task-conditioned literature candidates without extraction."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("neurooracle/data/full_v2/knowledge_graph.json"),
        help="KG used only to seed task/chain query terms.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/task_literature_20260616"),
    )
    parser.add_argument("--year-start", type=int, default=2000)
    parser.add_argument("--year-end", type=int, default=2026)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--terms-per-atom", type=int, default=12)
    parser.add_argument("--n-subqueries", type=int, default=3)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Default: all canonical tasks except CS1-specific ones.",
    )
    parser.add_argument(
        "--refresh-graph",
        action="store_true",
        help="Copy --graph into --data-dir even if a local KG already exists.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = args.data_dir
    graph_path = _ensure_graph(data_dir, args.graph, refresh_graph=args.refresh_graph)
    logging.info("using query KG: %s", graph_path)

    task_names = _split_names(args.tasks) or _default_tasks()
    unknown_tasks = sorted(set(task_names) - set(_default_tasks()))
    if unknown_tasks:
        parser.error(f"unknown Case Study 3 tasks: {', '.join(unknown_tasks)}")
    summary_path = data_dir / "case3_task_literature_collection_summary.json"

    started_at = datetime.now().isoformat()
    rows: list[dict[str, Any]] = []

    for name in task_names:
        logging.info("collecting task: %s", name)
        result = run_chain_extraction(
            name,
            is_chain=False,
            year_start=args.year_start,
            year_end=args.year_end,
            max_results_per_query=args.max_results,
            terms_per_atom=args.terms_per_atom,
            n_subqueries=args.n_subqueries,
            data_dir=data_dir,
            collect_only=True,
            sample_rate_seen=0.0,
        )
        rows.append({"kind": "task", "name": name, "result": result})
        _write_summary(summary_path, {
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "data_dir": str(data_dir),
            "graph": str(args.graph),
            "tasks": task_names,
            "chains": [],
            "rows": rows,
        })

    payload = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "graph": str(args.graph),
        "tasks": task_names,
        "chains": [],
        "rows": rows,
    }
    _write_summary(summary_path, payload)
    print(json.dumps({
        "data_dir": str(data_dir),
        "summary": str(summary_path),
        "n_tasks": len(task_names),
        "n_chains": 0,
        "total_collected": sum(int(r["result"].get("total_collected", 0)) for r in rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
