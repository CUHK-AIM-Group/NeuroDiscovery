"""UMLS-based disease ID canonicalization.

DisGeNET ships only UMLS CUIs (e.g. C0036341 for Schizophrenia), while
MeSH ingest gives us MSH:Dxxx ids (e.g. MSH:D012559 for the same
disease). They are the same disease in two ID spaces. Without a bridge,
ENIGMA edges land on MSH:D012559 and DisGeNET edges land on
DISGENET:C0036341, so a hypothesis walker can never traverse both.

This module builds a CUI -> MSH UI map by streaming the local UMLS
MRCONSO.RRF file (the all-vocabularies cross-reference). One CUI may
have multiple MSH UIs (cross-listings); we keep all and let the caller
pick the first that exists in the current KG.

The map is cached to JSON so subsequent ingests skip the ~60-90 s
MRCONSO scan. Cache invalidates by MRCONSO mtime.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DEFAULT_MRCONSO = DEFAULT_DATA_DIR / "MRCONSO.RRF"
DEFAULT_CACHE = DEFAULT_DATA_DIR / "_cui_to_mesh.json"


def build_cui_to_mesh(
    mrconso_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> dict[str, list[str]]:
    """Return CUI -> [MSH UI, ...] map. Cached.

    Streams MRCONSO.RRF and emits one entry per (CUI, MSH UI) row where
    LAT == 'ENG' and SAB == 'MSH'. A single CUI can map to several MSH
    UIs (synonyms); the caller picks one based on KG membership.
    """
    mrconso_path = Path(mrconso_path) if mrconso_path else DEFAULT_MRCONSO
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE
    if not mrconso_path.exists():
        logger.warning(
            f"MRCONSO.RRF not found at {mrconso_path}; "
            "disease canonicalization will be a no-op."
        )
        return {}

    if cache_path.exists() and not force_rebuild:
        try:
            cache_mtime = cache_path.stat().st_mtime
            mrconso_mtime = mrconso_path.stat().st_mtime
            if cache_mtime > mrconso_mtime:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.info("CUI->MSH cache unreadable; rebuilding from MRCONSO")

    logger.info(f"building CUI->MSH map from {mrconso_path.name} (one-off scan)...")
    cui_to_msh: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with open(mrconso_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 14:
                continue
            if parts[1] != "ENG":
                continue
            if parts[11] != "MSH":
                continue
            cui = parts[0]
            code = parts[13]
            key = (cui, code)
            if key in seen:
                continue
            seen.add(key)
            cui_to_msh.setdefault(cui, []).append(code)

    logger.info(
        f"CUI->MSH map built: {len(cui_to_msh)} CUIs with MSH crossref"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cui_to_msh, f)
    except OSError as e:
        logger.warning(f"could not cache CUI->MSH map: {e}")
    return cui_to_msh


def canonical_disease_id(
    cui: str,
    cui_to_msh: dict[str, list[str]],
    kg_msh_uis: set[str],
) -> Optional[str]:
    """Return the canonical KG node id for a UMLS CUI, or None.

    Resolution order:
      1. If any MSH UI in cui_to_msh[cui] is already an MSH node in the
         KG, return that as MSH:<UI>.
      2. Otherwise None (caller decides whether to keep DISGENET:<CUI>
         as a standalone node, subject to neuro-relevance checks).
    """
    candidates = cui_to_msh.get(cui, [])
    for ui in candidates:
        if ui in kg_msh_uis:
            return f"MSH:{ui}"
    return None
