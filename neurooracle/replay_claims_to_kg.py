"""Fast replay of extracted_claims.jsonl onto a snapshot KG.

Bypasses ingest_claims (which re-runs entity resolution + salvage = O(KG)
per noise term, ~10 claims/s on 200K-node graph). Since the jsonl already
has final subject_id/object_id from the original ingest, we just:

    1. Load the snapshot KG.
    2. Stream the jsonl, skipping CLM ids already present.
    3. For each new claim: insert ConceptNode + simplified edge + 2 about edges.
       Skip if subject_id or object_id is missing from the KG.

Atomic save at the end (storage.save_graph already writes via .tmp + rename).

Usage::

    python -m neurooracle.replay_claims_to_kg \
        --snapshot neurooracle/data/full_snapshot_v1/knowledge_graph.json \
        --claims neurooracle/data/full_snapshot_v2/extracted_claims.jsonl \
        --out neurooracle/data/full_snapshot_v2/knowledge_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from neurooracle.src.schema import Claim, ConceptNode, Edge
from neurooracle.src.storage import load_graph, save_graph

logger = logging.getLogger(__name__)


def replay(
    snapshot: Path,
    claims_jsonl: Path,
    out_path: Path,
    save_every: int = 100_000,
    mint_missing_anchors: bool = True,
) -> None:
    logger.info(f"loading snapshot: {snapshot}")
    kg = load_graph(snapshot)
    snapshot_concepts = set(kg._index.keys())
    snapshot_clm = {nid for nid in snapshot_concepts if nid.startswith("CLM:")}
    logger.info(f"snapshot: {len(snapshot_concepts):,} concepts ({len(snapshot_clm):,} claim nodes)")

    t0 = time.time()
    n_total = 0
    n_already = 0
    n_dup = 0
    n_skipped_missing = 0
    n_anchors_minted = 0
    n_added = 0
    n_edges_added = 0
    last_save = 0

    seen_ids: set[str] = set()

    def _ensure_anchor(anchor_id: str, anchor_name: str, anchor_type: str) -> bool:
        """Mint a minimal ConceptNode for an anchor that's missing.

        Mirrors what ingest_claims would create when noise filter is OFF and
        the resolver falls through to step 6 (CLM_CONCEPT mint). Returns True
        if the anchor exists or was minted; False if we cannot mint (no name).
        """
        nonlocal n_anchors_minted
        if kg.has_concept(anchor_id):
            return True
        if not mint_missing_anchors:
            return False
        if not anchor_id or not anchor_name:
            return False
        domain = "claim_concept" if anchor_id.startswith("CLM_CONCEPT:") else "external"
        try:
            kg.add_concept(ConceptNode(
                id=anchor_id,
                preferred_name=anchor_name,
                domain_tags=[domain],
                source_vocab="replay_anchor_mint",
            ))
            n_anchors_minted += 1
            return True
        except Exception:
            return False

    with open(claims_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            cid = d.get("id", "")
            if cid in snapshot_clm:
                n_already += 1
                continue
            if cid in seen_ids:
                n_dup += 1
                continue
            seen_ids.add(cid)

            sid = d.get("subject_id", "")
            oid = d.get("object_id", "")
            pred = d.get("predicate", "")
            sname = d.get("subject_name", "")
            oname = d.get("object_name", "")
            if not (cid and sid and oid and pred):
                n_skipped_missing += 1
                continue

            stype = (d.get("metadata") or {}).get("subject_type", "")
            otype = (d.get("metadata") or {}).get("object_type", "")
            if not _ensure_anchor(sid, sname, stype) or not _ensure_anchor(oid, oname, otype):
                n_skipped_missing += 1
                continue

            try:
                claim = Claim.from_dict(d)
            except Exception as e:
                logger.debug(f"failed to parse claim {cid}: {e}")
                n_skipped_missing += 1
                continue

            kg.add_concept(ConceptNode(
                id=claim.id,
                preferred_name=f"{claim.subject_name} {claim.predicate} {claim.object_name}",
                domain_tags=["claim"],
                source_vocab="claim_extraction",
                definition=claim.raw_text,
                metadata=claim.to_dict(),
            ))
            n_added += 1

            try:
                kg.add_edge(claim.to_edge())
                n_edges_added += 1
            except Exception:
                pass

            for tgt in (sid, oid):
                try:
                    kg.add_edge(Edge(
                        source_id=cid,
                        target_id=tgt,
                        relation_type="about",
                        source="claim_extraction",
                        confidence=claim.confidence,
                    ))
                    n_edges_added += 1
                except Exception:
                    pass

            if n_added - last_save >= save_every:
                save_graph(kg, out_path)
                last_save = n_added

            if n_added % 50_000 == 0 and n_added > 0:
                elapsed = time.time() - t0
                rate = n_added / max(elapsed, 0.1)
                logger.info(
                    f"  added {n_added:,} claims  "
                    f"(minted {n_anchors_minted:,} anchors, "
                    f"skipped {n_skipped_missing:,} missing-anchor, "
                    f"{n_already:,} already-in-snapshot)  "
                    f"{rate:.0f}/s  elapsed {elapsed:.0f}s"
                )

    save_graph(kg, out_path)
    elapsed = time.time() - t0
    logger.info(
        f"replay complete in {elapsed/60:.1f}m: "
        f"jsonl total {n_total:,} | already {n_already:,} | dup {n_dup:,} | "
        f"missing-anchor (skipped) {n_skipped_missing:,} | "
        f"minted anchors {n_anchors_minted:,} | "
        f"added {n_added:,} claims, {n_edges_added:,} edges"
    )
    logger.info(
        f"final KG: {kg.stats()['n_concepts']:,} concepts, "
        f"{kg.stats()['n_edges']:,} edges"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--claims", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    replay(args.snapshot, args.claims, args.out, save_every=args.save_every)


if __name__ == "__main__":
    main()
