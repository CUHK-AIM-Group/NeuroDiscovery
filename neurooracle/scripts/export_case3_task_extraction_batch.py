"""Export balanced Case Study 3 extraction batches with abstracts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_DIR = (
    ROOT
    / "neurooracle"
    / "data"
    / "cs_runs"
    / "case3_hindcasting"
    / "merged_task_literature_20260616_v1"
)


def _load_abstracts(paths: list[Path]) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pmid = str(obj.get("pmid") or "").strip()
                abstract = str(obj.get("abstract") or "").strip()
                if pmid and abstract:
                    abstracts[pmid] = abstract
    return abstracts


def _load_done_ids(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                paper_id = str(obj.get("paper_id") or obj.get("id") or "").strip()
                if paper_id:
                    done.add(paper_id)
    return done


def export_batch(
    *,
    queue_csv: Path,
    cache_paths: list[Path],
    out_jsonl: Path,
    per_task: int,
    tier: str,
    done_paths: list[Path],
) -> dict[str, object]:
    abstracts = _load_abstracts(cache_paths)
    done_ids = _load_done_ids(done_paths)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    with queue_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if tier != "any" and row.get("priority_tier") != tier:
                continue
            paper_id = (row.get("id") or "").strip()
            if not paper_id or paper_id in done_ids or paper_id not in abstracts:
                continue
            by_task[row["task"]].append(row)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    task_counts: dict[str, int] = {}
    for task in sorted(by_task):
        rows = sorted(
            by_task[task],
            key=lambda r: (-float(r.get("priority_score") or 0), r.get("year") or "", r.get("title") or ""),
        )
        count = 0
        for row in rows:
            paper_id = row["id"]
            if paper_id in selected_ids:
                continue
            selected_ids.add(paper_id)
            selected.append({
                "paper_id": paper_id,
                "task": task,
                "priority_score": float(row.get("priority_score") or 0),
                "priority_tier": row.get("priority_tier") or "",
                "title": row.get("title") or "",
                "year": row.get("year") or "",
                "journal": row.get("journal") or "",
                "doi": row.get("doi") or "",
                "sources": row.get("sources") or "",
                "source_runs": row.get("source_runs") or "",
                "labels": row.get("labels") or "",
                "abstract": abstracts[paper_id],
            })
            count += 1
            if count >= per_task:
                break
        task_counts[task] = count

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "queue_csv": str(queue_csv),
        "out_jsonl": str(out_jsonl),
        "tier": tier,
        "per_task": per_task,
        "selected_papers": len(selected),
        "task_counts": task_counts,
        "done_ids": len(done_ids),
    }
    with out_jsonl.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", type=Path, default=DEFAULT_MERGED_DIR / "task_priority_extraction_queue.csv")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--tier", default="core", choices=["core", "standard", "low", "any"])
    parser.add_argument("--done", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = export_batch(
        queue_csv=args.queue_csv,
        cache_paths=args.cache,
        out_jsonl=args.out_jsonl,
        per_task=args.per_task,
        tier=args.tier,
        done_paths=args.done,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
