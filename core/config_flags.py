"""Small, single reader for the ``harness`` switches in ``core/config/features.json``.

The tool registry and the agent loop both need to ask whether a harness feature
is on. Reading the JSON in two places would let the answer drift, so the lookup
is centralized here. An explicit ``NEUROCLAW_HARNESS_*`` environment override wins
over the file, which is how a benchmark driver selects an arm without editing a
tracked file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


FEATURES_PATH = Path(__file__).resolve().parent / "config" / "features.json"


def _load() -> dict:
    try:
        return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def feature_enabled(name: str, *, default: bool = False) -> bool:
    """Return whether a feature is enabled.

    Resolution order: environment override, then the ``harness`` block of
    ``features.json``, then ``default``.
    """
    override = os.environ.get(name if name.startswith("NEUROCLAW_") else f"NEUROCLAW_HARNESS_{name.upper()}")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    value = (_load().get("harness") or {}).get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and "enabled" in value:
        return bool(value["enabled"])
    return default
