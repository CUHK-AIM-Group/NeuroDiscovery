"""PubMed abstract cache (JSONL on disk + in-memory pmid index).

Phase 2 hits PubMed efetch for every paper it processes. Re-runs (e.g.
new chain-aware query, prompt iteration, sparsity backfill) re-pay that
cost. This module persists fetched abstracts so subsequent runs can
either skip PubMed entirely (--rerun-cached) or only fetch the new pmids.

Layout:
    abstract_cache.jsonl   one record per line:
        {"pmid": "...", "abstract": "...", "paper": {<PaperRef dict>},
         "fetched_at": "ISO-8601"}

The file is append-only. The in-memory index is built on first access by
streaming the file once and recording byte offsets per pmid; subsequent
lookups seek+read a single line.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .schema import PaperRef

logger = logging.getLogger(__name__)


class AbstractCache:
    """JSONL-backed pmid → (abstract, PaperRef) store with in-memory index.

    Thread-safe for the typical Phase-2 access pattern: many concurrent
    readers, one writer at a time (writes serialised behind ``_write_lock``).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, int] = {}   # pmid → byte offset
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._build_index()

    def _build_index(self) -> None:
        if not self.path.exists():
            return
        offset = 0
        with open(self.path, "rb") as f:
            for line in f:
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                    pmid = obj.get("pmid")
                    if pmid:
                        # Last write wins (tail line for any duplicate pmid).
                        self._index[str(pmid)] = offset
                except Exception:
                    pass
                offset += len(line)
        logger.info(f"abstract cache: indexed {len(self._index):,} pmids from {self.path}")

    # ── lookups ─────────────────────────────────────────────────────────

    def __contains__(self, pmid: str) -> bool:
        return str(pmid) in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, pmid: str) -> Optional[tuple[str, PaperRef]]:
        pmid = str(pmid)
        offset = self._index.get(pmid)
        if offset is None:
            return None
        with self._read_lock:
            with open(self.path, "rb") as f:
                f.seek(offset)
                line = f.readline()
        try:
            obj = json.loads(line.decode("utf-8", errors="replace"))
        except Exception:
            return None
        abstract = obj.get("abstract") or ""
        if not abstract.strip():
            return None
        ref_dict = obj.get("paper") or {}
        try:
            paper = PaperRef.from_dict(ref_dict)
        except Exception:
            return None
        return abstract, paper

    def get_many(self, pmids: Iterable[str]) -> tuple[list[tuple[str, PaperRef]], list[str]]:
        """Bulk lookup. Returns (hits, misses)."""
        hits: list[tuple[str, PaperRef]] = []
        misses: list[str] = []
        for pmid in pmids:
            rec = self.get(pmid)
            if rec is None:
                misses.append(str(pmid))
            else:
                hits.append(rec)
        return hits, misses

    # ── writes ──────────────────────────────────────────────────────────

    def put(self, pmid: str, abstract: str, paper: PaperRef) -> None:
        if not abstract or not abstract.strip():
            return
        pmid = str(pmid)
        record = {
            "pmid": pmid,
            "abstract": abstract,
            "paper": _paper_to_dict(paper),
            "fetched_at": datetime.now().isoformat(),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        with self._write_lock:
            offset = self.path.stat().st_size if self.path.exists() else 0
            with open(self.path, "ab") as f:
                f.write(encoded)
            self._index[pmid] = offset

    def put_many(self, items: Iterable[tuple[str, str, PaperRef]]) -> int:
        """Bulk append. Items: (pmid, abstract, paper). Returns # written."""
        n = 0
        with self._write_lock:
            offset = self.path.stat().st_size if self.path.exists() else 0
            with open(self.path, "ab") as f:
                for pmid, abstract, paper in items:
                    if not abstract or not abstract.strip():
                        continue
                    pmid = str(pmid)
                    record = {
                        "pmid": pmid,
                        "abstract": abstract,
                        "paper": _paper_to_dict(paper),
                        "fetched_at": datetime.now().isoformat(),
                    }
                    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    f.write(encoded)
                    self._index[pmid] = offset
                    offset += len(encoded)
                    n += 1
        return n

    # ── iteration ───────────────────────────────────────────────────────

    def iter_records(self) -> Iterable[tuple[str, str, PaperRef]]:
        """Stream all (pmid, abstract, paper) triples in insertion order."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            for line in f:
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                    pmid = obj.get("pmid")
                    abstract = obj.get("abstract") or ""
                    if not pmid or not abstract.strip():
                        continue
                    ref_dict = obj.get("paper") or {}
                    try:
                        paper = PaperRef.from_dict(ref_dict)
                    except Exception:
                        continue
                    yield pmid, abstract, paper
                except Exception:
                    continue


def _paper_to_dict(paper: PaperRef) -> dict:
    return paper.to_dict()


def default_cache_path(data_dir: Optional[Path] = None) -> Path:
    """Default cache location. Honours NEUROCLAW_ABSTRACT_CACHE env var."""
    env = os.environ.get("NEUROCLAW_ABSTRACT_CACHE")
    if env:
        return Path(env)
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    return Path(data_dir) / "abstract_cache.jsonl"


__all__ = ["AbstractCache", "default_cache_path"]
