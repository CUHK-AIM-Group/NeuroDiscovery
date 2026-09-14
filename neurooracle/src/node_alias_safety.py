"""Targeted safeguards for reviewed, non-equivalent imaging aliases.

These guards do not redirect a query to a different entity. In particular,
ambiguous CT requires context and fALFF must not fall back to the ALFF node.
"""

from __future__ import annotations

import re
import unicodedata


RETIRED_GLOBAL_ALIASES = {
    "IF:cortical_thickness": ("CT",),
    "IF:alff": ("fALFF",),
}


def allow_substring_fallback(query: str) -> bool:
    """An unqualified CT needs an exact name/alias, never an interior substring."""
    normalized = unicodedata.normalize("NFKC", str(query or "")).strip().casefold()
    return normalized != "ct"


def blocked_canonical_ids(query: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", str(query or "")).casefold().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    blocked = set()
    if re.search(r"\bct\b", normalized) and "cortical thickness" not in normalized:
        blocked.add("IF:cortical_thickness")
    if re.search(r"\bfalff\b|\bfractional\s+alff\b|\bfractional\s+amplitude\s+of\s+low", normalized):
        blocked.add("IF:alff")
    return frozenset(blocked)
