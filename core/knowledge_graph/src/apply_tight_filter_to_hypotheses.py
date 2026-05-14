"""Apply the tightened post-process rules (C-1/2/3) to existing hypothesis
JSON files without re-running Phase 3.

Rules enforced:
  C-1  No generic-phrase intermediate nodes
       ("neural activity", "functional connectivity", "disease progression",
        "grey matter", "cognitive deficit", ...).
  C-2  Directional density: 3+ hop paths must have >= 50% directional edges.
  C-3  Target name is not an umbrella concept ("skill", "disease",
       "neurological disorder", "clinical features", ...).

Each file is backed up to <name>.pre_tight_filter.json the first time.

Usage:
    python -m core.knowledge_graph.src.apply_tight_filter_to_hypotheses \
        --hyp-dir core/knowledge_graph/data/quick

    # dry run
    python -m core.knowledge_graph.src.apply_tight_filter_to_hypotheses \
        --hyp-dir core/knowledge_graph/data/quick --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .hypothesis_engine import (
    DIRECTIONAL_RELATIONS,
    HypothesisEngine,
)

log = logging.getLogger("apply_tight_filter")


def _fails_c1(hyp: dict) -> bool:
    """Generic-phrase intermediate node."""
    path = hyp.get("path") or []
    if len(path) < 2:
        return False
    intermediate_names: list[str] = []
    for i, step in enumerate(path):
        if not isinstance(step, dict):
            continue
        if i >= 1:
            intermediate_names.append(step.get("from_name") or "")
        if i < len(path) - 1:
            intermediate_names.append(step.get("to_name") or "")
    return any(HypothesisEngine._is_generic_intermediate(n) for n in intermediate_names)


def _fails_c2(hyp: dict) -> bool:
    """Directional density too thin (3+ hops with < 50% directional)."""
    path = hyp.get("path") or []
    if len(path) < 3:
        return False
    directional = sum(
        1 for s in path
        if isinstance(s, dict)
        and s.get("relation_type") in DIRECTIONAL_RELATIONS
    )
    return directional * 2 < len(path)


def _fails_c3(hyp: dict) -> bool:
    """Target name is an umbrella concept."""
    return HypothesisEngine._is_too_broad_target(hyp.get("target_name") or "")


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
    kept: list[dict] = []
    c1 = c2 = c3 = 0
    for h in items:
        if not isinstance(h, dict):
            continue
        f1, f2, f3 = _fails_c1(h), _fails_c2(h), _fails_c3(h)
        if f1: c1 += 1
        if f2: c2 += 1
        if f3: c3 += 1
        if f1 or f2 or f3:
            continue
        kept.append(h)

    removed = before - len(kept)
    info = {
        "file": str(path), "before": before, "after": len(kept),
        "removed": removed, "c1_generic_intermediate": c1,
        "c2_thin_directional": c2, "c3_broad_target": c3,
    }

    if dry_run or removed == 0:
        return info

    backup = path.with_suffix(".pre_tight_filter.json")
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
        for f in sorted(args.hyp_dir.glob("hypotheses_*.json")):
            if ".pre_" in f.name:
                continue
            targets.append(f)
    if not targets:
        log.error("no hypothesis files found")
        return

    total_before = total_after = 0
    total_c1 = total_c2 = total_c3 = 0
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
                 f"(removed {info['removed']}: "
                 f"C1={info['c1_generic_intermediate']}, "
                 f"C2={info['c2_thin_directional']}, "
                 f"C3={info['c3_broad_target']})")
        total_before += info["before"]
        total_after += info["after"]
        total_c1 += info["c1_generic_intermediate"]
        total_c2 += info["c2_thin_directional"]
        total_c3 += info["c3_broad_target"]

    log.info("-" * 60)
    log.info(f"TOTAL  {total_before} -> {total_after}  "
             f"(C1={total_c1}, C2={total_c2}, C3={total_c3})")
    if args.dry_run:
        log.info("dry-run: no files written")


if __name__ == "__main__":
    main()
