"""Credential-shape validation for the Case 2 Ollama-only continuation.

Only labelled key counts leave this module.  Secret values are returned solely
to the in-process gateway and must never be logged or persisted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


EXPECTED_KEY_COUNTS = {"opencode": 3, "ollama": 2, "deepseek": 1}
AUTOMATIC_ROUTE = (
    "ollama_cloud_key_1",
    "ollama_cloud_key_2",
)


def load_channel_keys(path: Path) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    channels: dict[str, list[str]] = {name: [] for name in EXPECTED_KEY_COUNTS}
    current: str | None = None
    for raw in lines:
        value = raw.strip()
        if not value:
            current = None
            continue
        lowered = value.casefold()
        if "opencode" in lowered and "api key" in lowered:
            current = "opencode"
            continue
        if "ollama" in lowered and "api key" in lowered:
            current = "ollama"
            continue
        if "deepseek" in lowered and "api key" in lowered:
            current = "deepseek"
            continue
        if current is not None:
            channels[current].append(value.strip('"').strip("'"))
    for channel, expected in EXPECTED_KEY_COUNTS.items():
        actual = len(channels[channel])
        if actual != expected:
            raise ValueError(
                f"expected exactly {expected} {channel} key(s), found {actual}"
            )
        if any(not secret for secret in channels[channel]):
            raise ValueError(f"empty secret in {channel} key section")
    return channels


def key_count_preflight(path: Path) -> dict[str, Any]:
    keys = load_channel_keys(path)
    return {
        "status": "ok",
        "loaded_key_counts": {name: len(values) for name, values in keys.items()},
        "key_values_persisted_or_printed": False,
    }


__all__ = [
    "AUTOMATIC_ROUTE",
    "EXPECTED_KEY_COUNTS",
    "key_count_preflight",
    "load_channel_keys",
]
