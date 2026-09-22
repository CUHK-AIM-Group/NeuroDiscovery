"""Build discriminative, method-independent endpoints for saturated search tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "case-study-quality-endpoint.v1"
DEFAULT_COMPONENTS = (
    ("r_ci_low", "higher", 0.5),
    ("mae_improvement_ci_low", "higher", 0.5),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_bool(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{column} contains invalid booleans: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def derive_quality_outcomes(
    internal: pd.DataFrame,
    *,
    quantile: float = 0.75,
    components: Sequence[tuple[str, str, float]] = DEFAULT_COMPONENTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze a top-quantile robust utility endpoint without method information."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    required = {"candidate_id", "validated", *(item[0] for item in components)}
    missing = sorted(required - set(internal.columns))
    if missing:
        raise ValueError(f"internal outcomes lack quality columns: {missing}")
    if not components or any(float(item[2]) <= 0 for item in components):
        raise ValueError("quality components must have positive weights")

    out = internal.copy()
    if out["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id values must be unique")
    base_validated = _coerce_bool(out["validated"], column="validated")
    if not base_validated.any():
        raise ValueError("quality endpoint requires at least one base discovery")

    utility = pd.Series(0.0, index=out.index, dtype=float)
    total_weight = float(sum(float(item[2]) for item in components))
    component_audit: list[dict[str, Any]] = []
    for column, direction, raw_weight in components:
        values = pd.to_numeric(out[column], errors="coerce")
        if not np.isfinite(values[base_validated].to_numpy(float)).all():
            raise ValueError(f"{column} must be finite for every base discovery")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"invalid component direction: {direction!r}")
        ranked = values[base_validated].rank(method="average", pct=True)
        if direction == "lower":
            ranked = 1.0 - ranked + (1.0 / len(ranked))
        normalized_weight = float(raw_weight) / total_weight
        utility.loc[base_validated] += normalized_weight * ranked
        component_audit.append(
            {
                "column": column,
                "direction": direction,
                "weight": normalized_weight,
            }
        )

    valid_utility = utility[base_validated].to_numpy(float)
    cutoff = float(np.quantile(valid_utility, quantile, method="higher"))
    quality_validated = base_validated & utility.ge(cutoff)

    out["base_validated"] = base_validated
    if "strict_validated" in out:
        out["base_strict_validated"] = _coerce_bool(
            out["strict_validated"], column="strict_validated"
        )
    out["quality_score"] = utility
    out["feedback_utility"] = utility.clip(0.0, 1.0)
    out["quality_cutoff"] = cutoff
    out["quality_endpoint"] = quality_validated
    out["validated"] = quality_validated
    out["strict_validated"] = quality_validated
    errors = (
        out["error"].fillna("").astype(str).str.strip().ne("")
        if "error" in out
        else pd.Series(False, index=out.index)
    )
    out["feedback_status"] = np.select(
        [quality_validated, errors],
        ["supported", "execution_failed"],
        default="inconclusive",
    )

    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "selection_uses_method_identity": False,
        "selection_uses_generator_scores": False,
        "base_discoveries": int(base_validated.sum()),
        "quality_discoveries": int(quality_validated.sum()),
        "quantile": float(quantile),
        "cutoff": cutoff,
        "components": component_audit,
        "low_quality_base_discoveries_are": "inconclusive_search_results",
        "closed_loop_feedback": "continuous_method_independent_quality_utility",
        "semantic_contradictions_inferred": False,
    }
    return out, audit


def build_task_bundle(
    source_root: Path,
    output_root: Path,
    task: str,
    *,
    quantile: float,
) -> dict[str, Any]:
    source_dir = source_root / task / "tables"
    output_dir = output_root / task / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_public = source_dir / "public_candidates.csv"
    source_internal = source_dir / "internal_outcomes.csv"
    source_manifest = source_dir / "table_manifest.json"
    for path in (source_public, source_internal, source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    public_path = output_dir / "public_candidates.csv"
    shutil.copy2(source_public, public_path)
    internal, endpoint = derive_quality_outcomes(
        pd.read_csv(source_internal, low_memory=False), quantile=quantile
    )
    internal_path = output_dir / "internal_outcomes.csv"
    internal.to_csv(internal_path, index=False)

    table_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    table_manifest["quality_endpoint"] = {
        **endpoint,
        "source_internal_outcomes": {
            "path": str(source_internal.resolve()),
            "sha256": sha256_file(source_internal),
        },
        "source_public_candidates": {
            "path": str(source_public.resolve()),
            "sha256": sha256_file(source_public),
        },
    }
    table_manifest["internal_validated"] = endpoint["quality_discoveries"]
    table_manifest["files"] = {
        "public_candidates": {
            "path": str(public_path.resolve()),
            "sha256": sha256_file(public_path),
            "rows": int(len(internal)),
        },
        "internal_outcomes": {
            "path": str(internal_path.resolve()),
            "sha256": sha256_file(internal_path),
            "rows": int(len(internal)),
            "columns": list(internal.columns),
        },
    }
    manifest_path = output_dir / "table_manifest.json"
    manifest_path.write_text(
        json.dumps(table_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "public_candidates": str(public_path.resolve()),
        "internal_outcomes": str(internal_path.resolve()),
        "table_manifest": str(manifest_path.resolve()),
        "quality_endpoint": endpoint,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--quantile", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = [
        build_task_bundle(
            args.source_root.resolve(),
            args.output_root.resolve(),
            task,
            quantile=args.quantile,
        )
        for task in args.tasks
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tasks": records,
    }
    path = args.output_root / "quality_endpoint_manifest.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
