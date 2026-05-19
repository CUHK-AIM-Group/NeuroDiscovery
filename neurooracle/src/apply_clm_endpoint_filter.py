"""Apply CLM_CONCEPT endpoint judgments to hypothesis files.

Removes hypotheses whose source or target was judged REMOVE by the
GPT voting process (judge_clm_endpoints.py).

Usage:
    python -m neurooracle.src.apply_clm_endpoint_filter \
        --hyp-dir neurooracle/data/quick \
        --judgments neurooracle/data/quick/clm_endpoint_judgments.json \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("apply_clm_endpoint_filter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    remove_names = set(n.lower().strip() for n in judgments.get("remove_names", []))
    log.info(f"loaded {len(remove_names)} names to remove")

    targets = sorted(
        f for f in args.hyp_dir.glob("hypotheses_*.json")
        if "pre_" not in f.name and "post_" not in f.name
    )

    total_before = total_after = total_removed = 0
    for fpath in targets:
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "hypotheses" in raw:
            items = raw["hypotheses"]
            wrapper = raw
        elif isinstance(raw, list):
            items = raw
            wrapper = None
        else:
            continue

        before = len(items)
        kept = []
        for h in items:
            if not isinstance(h, dict):
                kept.append(h)
                continue
            src_name = (h.get("source_name") or "").lower().strip()
            tgt_name = (h.get("target_name") or "").lower().strip()
            if src_name in remove_names or tgt_name in remove_names:
                continue
            kept.append(h)

        removed = before - len(kept)
        log.info(f"{fpath.name}  {before} -> {len(kept)} (removed {removed})")
        total_before += before
        total_after += len(kept)
        total_removed += removed

        if args.dry_run or removed == 0:
            continue

        backup = fpath.with_name(fpath.stem + ".pre_clm_endpoint_filter.json")
        if not backup.exists():
            shutil.copy2(fpath, backup)

        if wrapper is not None:
            wrapper["hypotheses"] = kept
            wrapper["n_hypotheses"] = len(kept)
            out_obj = wrapper
        else:
            out_obj = kept

        fpath.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    log.info("-" * 60)
    log.info(f"TOTAL  {total_before} -> {total_after} (removed {total_removed})")
    if args.dry_run:
        log.info("dry-run: no files written")


if __name__ == "__main__":
    main()
