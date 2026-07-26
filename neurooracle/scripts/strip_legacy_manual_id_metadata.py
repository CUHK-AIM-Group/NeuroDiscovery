"""Remove legacy MANUAL ID provenance fields from the active v2 KG files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from neurooracle.scripts.canonicalize_manual_claim_ids import (
        DEFAULT_EXTRACTED,
        DEFAULT_GRAPH,
        atomic_swap_with_rollback,
        is_legacy_manual_claim_id,
    )
except ModuleNotFoundError:
    from canonicalize_manual_claim_ids import (  # type: ignore[no-redef]
        DEFAULT_EXTRACTED,
        DEFAULT_GRAPH,
        atomic_swap_with_rollback,
        is_legacy_manual_claim_id,
    )


MIGRATION_NAME = "strip_legacy_manual_id_metadata_20260722"
LEGACY_KEYS = {"legacy_claim_id", "legacy_concept_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_GRAPH.parent / "legacy_manual_metadata_cleanup_20260722.json",
    )
    return parser.parse_args()


def strip_legacy_manual_metadata(value: Any) -> int:
    removed = 0
    if isinstance(value, dict):
        for key in list(value):
            item = value[key]
            if key in LEGACY_KEYS and is_legacy_manual_claim_id(item):
                del value[key]
                removed += 1
            else:
                removed += strip_legacy_manual_metadata(item)
    elif isinstance(value, list):
        for item in value:
            removed += strip_legacy_manual_metadata(item)
    return removed


def count_legacy_manual_metadata(value: Any) -> int:
    if isinstance(value, dict):
        return sum(
            1
            if key in LEGACY_KEYS and is_legacy_manual_claim_id(item)
            else count_legacy_manual_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(count_legacy_manual_metadata(item) for item in value)
    return 0


def stage_extracted(source: Path, target: Path) -> tuple[int, int]:
    rows = 0
    removed = 0
    with source.open("r", encoding="utf-8") as src, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id"):
                raise ValueError(f"Missing claim id in {source} line {number}")
            rows += 1
            removed += strip_legacy_manual_metadata(row)
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows, removed


def main() -> None:
    args = parse_args()
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    graph_removed = strip_legacy_manual_metadata(graph)
    report = {
        "migration": MIGRATION_NAME,
        "mode": "apply" if args.apply else "dry_run",
        "graph": str(args.graph.resolve()),
        "extracted": str(args.extracted.resolve()),
        "graph_metadata_fields_selected": graph_removed,
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    graph_temp = args.graph.with_suffix(args.graph.suffix + f".{MIGRATION_NAME}.tmp")
    extracted_temp = args.extracted.with_suffix(
        args.extracted.suffix + f".{MIGRATION_NAME}.tmp"
    )
    graph_backup = args.graph.with_suffix(args.graph.suffix + f".{MIGRATION_NAME}.bak")
    extracted_backup = args.extracted.with_suffix(
        args.extracted.suffix + f".{MIGRATION_NAME}.bak"
    )
    for path in (graph_temp, extracted_temp):
        if path.exists():
            path.unlink()

    extracted_rows, extracted_removed = stage_extracted(args.extracted, extracted_temp)
    with graph_temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if count_legacy_manual_metadata(graph):
        raise ValueError("Legacy MANUAL metadata remains in staged graph")
    with extracted_temp.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip() and count_legacy_manual_metadata(json.loads(line)):
                raise ValueError(
                    f"Legacy MANUAL metadata remains in staged archive line {number}"
                )

    swapped_graph = False
    try:
        atomic_swap_with_rollback(graph_temp, args.graph, graph_backup)
        swapped_graph = True
        atomic_swap_with_rollback(extracted_temp, args.extracted, extracted_backup)
    except Exception:
        if swapped_graph and graph_backup.exists():
            if args.graph.exists():
                args.graph.unlink()
            os.replace(graph_backup, args.graph)
        raise

    report.update(
        {
            "graph_metadata_fields_removed": graph_removed,
            "extracted_rows": extracted_rows,
            "extracted_metadata_fields_removed": extracted_removed,
            "validation": {
                "legacy_manual_graph_metadata_remaining": 0,
                "legacy_manual_extracted_metadata_remaining": 0,
            },
            "rollback_files": [str(graph_backup), str(extracted_backup)],
        }
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
