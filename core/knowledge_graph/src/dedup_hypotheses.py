"""P0 step 2: dedup hypothesis files by full path id tuple.

The audit found 50-70% duplicate paths in hypotheses_evolved /
hypotheses_novel / imaging hypothesis files. Same node sequence with
identical edge ids appears multiple times because:
  * evolution_engine generates mutations without dedup
  * imaging dataset-constrained mode produces near-identical paths
    when the same path is valid in multiple datasets

This script:
  1. Loads each hypothesis JSON (schema: {n_hypotheses, hypotheses: [...]})
  2. For each hypothesis, builds the path identity key:
       (source_id, *[step.to_id for step in path])
     plus the relation_type sequence as a tiebreaker.
  3. Groups duplicates and keeps the BEST scoring one:
       priority: critic_score (desc) > composite_score > confidence_score
  4. Writes back the deduped file. Original file is backed up to
     <name>.pre_dedup.json the first time only.

Usage:
    # dedup all files in a directory
    python -m core.knowledge_graph.src.dedup_hypotheses \
        --hyp-dir core/knowledge_graph/data/quick

    # one file
    python -m core.knowledge_graph.src.dedup_hypotheses \
        --file core/knowledge_graph/data/quick/hypotheses_evolved.json

    # dry-run (count only)
    python -m core.knowledge_graph.src.dedup_hypotheses \
        --hyp-dir core/knowledge_graph/data/quick --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("dedup_hypotheses")


def _path_key(hyp: dict) -> tuple:
    """Identity for a hypothesis path. Includes node ids and relation types
    so that two paths with the same nodes but different predicates are
    NOT collapsed (e.g. A--causes-->B vs A--predicts-->B).
    """
    src = hyp.get("source_id") or ""
    path = hyp.get("path") or []
    nodes = [src]
    rels = []
    for step in path:
        if not isinstance(step, dict):
            continue
        nodes.append(step.get("to_id", ""))
        rels.append(step.get("relation_type", ""))
    target = hyp.get("target_id") or ""
    if target and (not nodes or nodes[-1] != target):
        nodes.append(target)
    return (tuple(nodes), tuple(rels))


def _score(hyp: dict) -> float:
    for key in ("critic_score", "composite_score", "evolve_score",
                "confidence_score"):
        v = hyp.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _claim_overlap_dedup(items: list[dict], threshold: float = 0.5) -> list[dict]:
    """Greedy dedup by claim overlap (Jaccard > threshold).

    Sorted by score descending; each hypothesis is kept only if it doesn't
    overlap too much with any already-kept hypothesis.
    """
    if not items:
        return items

    scored = sorted(enumerate(items), key=lambda x: _score(x[1]), reverse=True)
    kept_claims: list[set] = []
    kept_indices: list[int] = []

    for idx, h in scored:
        claims = set(h.get("supporting_claims") or [])
        if not claims:
            kept_claims.append(claims)
            kept_indices.append(idx)
            continue

        overlaps = False
        for existing in kept_claims:
            if not existing:
                continue
            intersection = len(claims & existing)
            union = len(claims | existing)
            if union > 0 and intersection / union > threshold:
                overlaps = True
                break

        if not overlaps:
            kept_claims.append(claims)
            kept_indices.append(idx)

    return [items[i] for i in sorted(kept_indices)]


def dedup_file(path: Path, dry_run: bool = False) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "hypotheses" in raw:
        items = raw["hypotheses"]
        wrapper = raw
    elif isinstance(raw, list):
        items = raw
        wrapper = None
    else:
        return {"file": str(path), "skipped": "unknown structure"}

    n_before = len(items)
    groups: dict[tuple, list[int]] = {}
    for idx, h in enumerate(items):
        if not isinstance(h, dict):
            continue
        groups.setdefault(_path_key(h), []).append(idx)

    keep_indices: set[int] = set()
    duplicates_removed = 0
    for key, idxs in groups.items():
        if len(idxs) == 1:
            keep_indices.add(idxs[0])
            continue
        # pick best by score
        best = max(idxs, key=lambda i: _score(items[i]))
        keep_indices.add(best)
        duplicates_removed += len(idxs) - 1

    deduped = [items[i] for i in sorted(keep_indices)]

    # Second pass: claim-overlap dedup (Jaccard > 0.5)
    before_overlap = len(deduped)
    deduped = _claim_overlap_dedup(deduped, threshold=0.5)
    overlap_removed = before_overlap - len(deduped)

    n_after = len(deduped)

    info = {
        "file": str(path),
        "before": n_before,
        "after": n_after,
        "removed_path_dup": duplicates_removed,
        "removed_claim_overlap": overlap_removed,
        "removed": duplicates_removed + overlap_removed,
        "unique_paths": len(groups),
    }

    if dry_run:
        return info

    backup = path.with_suffix(".pre_dedup.json")
    if not backup.exists():
        shutil.copy2(path, backup)

    if wrapper is not None:
        wrapper["hypotheses"] = deduped
        wrapper["n_hypotheses"] = n_after
        out_obj = wrapper
    else:
        out_obj = deduped

    path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    info["written"] = True
    info["backup"] = str(backup)
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
        targets.extend(sorted(args.hyp_dir.glob("hypotheses_*.json")))
    targets = [t for t in targets if not t.name.endswith(".pre_dedup.json")]

    if not targets:
        log.error("no hypothesis files found")
        return

    total_before = total_after = total_removed = 0
    for p in targets:
        try:
            info = dedup_file(p, dry_run=args.dry_run)
        except Exception as e:
            log.error(f"{p}: {e}")
            continue
        if info.get("skipped"):
            log.info(f"{p.name}  SKIP ({info['skipped']})")
            continue
        log.info(f"{p.name}  {info['before']} -> {info['after']} "
                 f"(removed {info['removed']}, unique {info['unique_paths']})")
        total_before += info["before"]
        total_after += info["after"]
        total_removed += info["removed"]

    log.info("-" * 60)
    log.info(f"TOTAL  {total_before} -> {total_after} (removed {total_removed})")
    if args.dry_run:
        log.info("dry-run: no files written")


if __name__ == "__main__":
    main()
