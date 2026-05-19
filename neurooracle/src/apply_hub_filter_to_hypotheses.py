"""P1: apply PATH_IGNORE_NODE_IDS retrospectively to existing hypothesis files.

The hypothesis_engine and evolution_engine are now updated to skip vague
COGAT/MeSH umbrella hubs (memory, logic, loss, activation, risk, stress,
Brain, Neurons) as intermediates or endpoints. This script filters
ALREADY-GENERATED hypothesis files so we don't have to re-run Phase 3.

Each file is backed up to <name>.pre_hub_filter.json the first time.

Usage:
    python -m neurooracle.src.apply_hub_filter_to_hypotheses \
        --hyp-dir neurooracle/data/quick

    # dry-run
    python -m neurooracle.src.apply_hub_filter_to_hypotheses \
        --hyp-dir neurooracle/data/quick --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .hypothesis_engine import PATH_IGNORE_NODE_IDS

log = logging.getLogger("apply_hub_filter")


def _touches_ignored(hyp: dict) -> bool:
    if hyp.get("source_id") in PATH_IGNORE_NODE_IDS:
        return True
    if hyp.get("target_id") in PATH_IGNORE_NODE_IDS:
        return True
    for step in hyp.get("path") or []:
        if not isinstance(step, dict):
            continue
        if step.get("from_id") in PATH_IGNORE_NODE_IDS:
            return True
        if step.get("to_id") in PATH_IGNORE_NODE_IDS:
            return True
    return False


def filter_file(path: Path, dry_run: bool = False) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "hypotheses" in raw:
        items = raw["hypotheses"]
        wrapper = raw
    elif isinstance(raw, list):
        items = raw
        wrapper = None
    else:
        return {"file": str(path), "skipped": "unknown structure"}

    before = len(items)
    kept = [h for h in items if isinstance(h, dict) and not _touches_ignored(h)]
    removed = before - len(kept)

    info = {"file": str(path), "before": before, "after": len(kept),
            "removed": removed}

    if dry_run or removed == 0:
        return info

    backup = path.with_suffix(".pre_hub_filter.json")
    if not backup.exists():
        shutil.copy2(path, backup)
        info["backup"] = str(backup)

    if wrapper is not None:
        wrapper["hypotheses"] = kept
        wrapper["n_hypotheses"] = len(kept)
        out_obj = wrapper
    else:
        out_obj = kept
    path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    info["written"] = True
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", type=Path)
    ap.add_argument("--file", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    targets: list[Path] = []
    if args.file:
        targets.append(args.file)
    if args.hyp_dir:
        # Skip any .pre_* backups so we don't process pre-dedup or other snapshots.
        for f in sorted(args.hyp_dir.glob("hypotheses_*.json")):
            if ".pre_" in f.name:
                continue
            targets.append(f)

    if not targets:
        log.error("no hypothesis files found")
        return

    log.info(f"blacklist: {len(PATH_IGNORE_NODE_IDS)} ids")
    total_before = total_after = total_removed = 0
    for p in targets:
        try:
            info = filter_file(p, dry_run=args.dry_run)
        except Exception as e:
            log.error(f"{p}: {e}")
            continue
        if info.get("skipped"):
            log.info(f"{p.name}  SKIP ({info['skipped']})")
            continue
        log.info(f"{p.name}  {info['before']} -> {info['after']}  "
                 f"(removed {info['removed']})")
        total_before += info["before"]
        total_after += info["after"]
        total_removed += info["removed"]

    log.info("-" * 60)
    log.info(f"TOTAL  {total_before} -> {total_after} (removed {total_removed})")
    if args.dry_run:
        log.info("dry-run: no files written")


if __name__ == "__main__":
    main()
