"""Keep NeuroSTORM external while exposing a stable NeuroClaw route."""

from __future__ import annotations

import os
from pathlib import Path


def locate_neurostorm(explicit_root: str | Path | None = None) -> Path:
    candidates = [
        Path(explicit_root) if explicit_root else None,
        Path(os.environ["NEUROSTORM_ROOT"]) if os.environ.get("NEUROSTORM_ROOT") else None,
        Path(__file__).resolve().parents[3].parent / "NeuroSTORM",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "NeuroSTORM was not found. Set NEUROSTORM_ROOT or pass an explicit root."
    )


def prepare_neurostorm_command(
    config: str | Path,
    root: str | Path | None = None,
    python_executable: str = "python",
) -> list[str]:
    project = locate_neurostorm(root)
    entrypoints = [
        project / "main.py",
        project / "train.py",
        project / "scripts" / "train.py",
    ]
    entry = next((path for path in entrypoints if path.exists()), None)
    if entry is None:
        raise FileNotFoundError("No supported NeuroSTORM training entrypoint was found")
    return [python_executable, str(entry), "--config", str(config)]
