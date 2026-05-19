"""Apply INTERMEDIATE_ONLY_IGNORE_IDS filter to existing hypothesis files.

Disease mega-hubs (Alzheimer, Schizophrenia, Epilepsy, etc.) are valid as
hypothesis endpoints but not as intermediate transit nodes. This script
removes hypotheses that route through these hubs without re-running Phase 3.

Each file is backed up to <name>.pre_intermediate_hub_filter.json the first time.

Usage:
    python -m neurooracle.src.apply_intermediate_hub_filter \
        --hyp-dir neurooracle/data/quick

    # dry-run
    python -m neurooracle.src.apply_intermediate_hub_filter \
        --hyp-dir neurooracle/data/quick --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .hypothesis_engine import INTERMEDIATE_ONLY_IGNORE_IDS

log = logging.getLogger("apply_intermediate_hub_filter")


def _transits_hub(hyp: dict) -> bool:
    """Check if any intermediate node in the path is a disease mega-hub."""
    path = hyp.get("path") or []
    if len(path) < 2:
        return False
    for i, step in enumerate(path):
        if not isinstance(step, dict):
            continue
        if i >= 1 and step.get("from_id") in INTERMEDIATE_ONLY_IGNORE_IDS:
            return True
        if i < len(path) - 1 and step.get("to_id") in INTERMEDIATE_ONLY_IGNORE_IDS:
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
    kept = [h for h in items if isinstance(h, dict) and not _transits_hub(h)]
    removed = before - len(kept)

    info = {
        "file": str(path),
        "before": before,
        "after": len(kept),
        "removed": removed,
    }

    if dry_run or removed == 0:
        return info

    backup = path.with_name(path.stem + ".pre_intermediate_hub_filter.json")
    if not backup.exists():
        shutil.copy2(path, backup)

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
    ap.add_argument("--hyp-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    targets = sorted(args.hyp_dir.glob("hypotheses_*.json"))
    targets = [t for t in targets if "pre_" not in t.name and "post_" not in t.name]

    if not targets:
        log.error("no hypothesis files found")
        return

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
        log.info(f"{p.name}  {info['before']} -> {info['after']} (removed {info['removed']})")
        total_before += info["before"]
        total_after += info["after"]
        total_removed += info["removed"]

    log.info("-" * 60)
    log.info(f"TOTAL  {total_before} -> {total_after} (removed {total_removed})")
    if args.dry_run:
        log.info("dry-run: no files written")


if __name__ == "__main__":
    main()
