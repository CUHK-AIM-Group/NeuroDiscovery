"""Audit contract for frozen KGE, novelty, and critic candidate scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORE_COMPONENT_SCHEMA = "case-study-score-components.v1"
SCORE_COLUMNS = ("score_kge", "score_novelty", "score_critic")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_id_sha256(candidate_ids: pd.Series) -> str:
    payload = "\n".join(sorted(candidate_ids.astype(str).tolist())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embedded_score_component_audit(public: pd.DataFrame) -> dict[str, Any]:
    active = [column for column in SCORE_COLUMNS if column in public.columns]
    return {
        "schema_version": SCORE_COMPONENT_SCHEMA,
        "source": "public_candidates",
        "active": bool(active),
        "columns": active,
        "candidate_id_sha256": candidate_id_sha256(public["candidate_id"]),
        "outcome_blind": True,
        "frozen_before_experiment": True,
    }


def load_score_component_bundle(
    public: pd.DataFrame,
    *,
    table_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge a separately frozen score table after verifying its provenance."""

    table_path = table_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCORE_COMPONENT_SCHEMA:
        raise ValueError("unsupported score-component manifest schema")
    if manifest.get("outcome_blind") is not True:
        raise ValueError("score components must be outcome blind")
    if manifest.get("frozen_before_experiment") is not True:
        raise ValueError("score components must be frozen before experiments")
    expected_table_sha = str(manifest.get("score_table_sha256") or "")
    if expected_table_sha != _sha256_file(table_path):
        raise ValueError("score-component table SHA256 does not match its manifest")

    components = pd.read_csv(table_path, low_memory=False)
    if "candidate_id" not in components.columns:
        raise ValueError("score-component table requires candidate_id")
    if components["candidate_id"].astype(str).duplicated().any():
        raise ValueError("score-component candidate_id values must be unique")
    score_columns = [column for column in SCORE_COLUMNS if column in components.columns]
    unexpected = sorted(set(components.columns) - {"candidate_id", *score_columns})
    if unexpected:
        raise ValueError(
            "score-component table contains non-score columns: "
            + ", ".join(unexpected)
        )
    if not score_columns:
        raise ValueError("score-component table contains no supported score columns")

    public_ids = public["candidate_id"].astype(str)
    component_ids = components["candidate_id"].astype(str)
    if set(public_ids) != set(component_ids):
        raise ValueError("score components must cover the public registry exactly")
    expected_ids_sha = candidate_id_sha256(public_ids)
    if str(manifest.get("candidate_id_sha256") or "") != expected_ids_sha:
        raise ValueError("score-component candidate registry hash does not match")

    descriptors = manifest.get("components") or {}
    for column in score_columns:
        values = pd.to_numeric(components[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{column} values must be in [0, 1]")
        name = column.removeprefix("score_")
        descriptor = descriptors.get(name) or {}
        if descriptor.get("column") != column:
            raise ValueError(f"manifest does not describe {column}")
        if name == "kge" and not (
            descriptor.get("checkpoint_sha256")
            and descriptor.get("kg_snapshot_sha256")
        ):
            raise ValueError("KGE scores require checkpoint and KG snapshot hashes")
        if name == "kge" and descriptor.get("kg_snapshot_status") != "formal_release":
            raise ValueError("KGE scores require a formal-release KG snapshot")
        if name == "critic" and not (
            descriptor.get("model") and descriptor.get("prompt_sha256")
        ):
            raise ValueError("critic scores require model and prompt hashes")
        if name == "novelty" and not descriptor.get("definition"):
            raise ValueError("novelty scores require a frozen definition")

    conflicts = sorted(set(score_columns) & set(public.columns))
    if conflicts:
        raise ValueError(
            "score columns already exist in public candidates: " + ", ".join(conflicts)
        )
    ordered = components.assign(candidate_id=component_ids).set_index("candidate_id")
    merged = public.copy()
    for column in score_columns:
        merged[column] = public_ids.map(ordered[column])

    audit = {
        **manifest,
        "source": "frozen_bundle",
        "active": True,
        "columns": score_columns,
        "table_path": str(table_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }
    return merged, audit


__all__ = [
    "SCORE_COLUMNS",
    "SCORE_COMPONENT_SCHEMA",
    "candidate_id_sha256",
    "embedded_score_component_audit",
    "load_score_component_bundle",
]
