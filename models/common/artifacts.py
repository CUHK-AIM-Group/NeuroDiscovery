"""Deterministic output artifacts shared by model trainers."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunArtifacts:
    """Write the common NeuroClaw model-run contract."""

    def __init__(self, output_dir: str | Path, config: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)

    def write_config(self) -> Path:
        return self.write_json("config.json", self.config)

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        return self.write_json("metrics.json", metrics)

    def write_predictions(self, rows: pd.DataFrame | dict[str, Any]) -> Path:
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        path = self.output_dir / "predictions.csv"
        frame.to_csv(path, index=False)
        return path

    def write_folds(self, rows: pd.DataFrame | dict[str, Any]) -> Path:
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        path = self.output_dir / "fold_assignments.csv"
        frame.to_csv(path, index=False)
        return path

    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        for key, name in (
            ("config_file", "config.json"),
            ("metrics_file", "metrics.json"),
            ("predictions_file", "predictions.csv"),
            ("fold_assignments_file", "fold_assignments.csv"),
            ("checkpoint_joblib", "checkpoint.joblib"),
            ("checkpoint_torch", "checkpoint.pt"),
        ):
            if (self.output_dir / name).exists():
                manifest[key] = name
        if extra:
            manifest.update(extra)
        return self.write_json("run_manifest.json", manifest)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / name
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
