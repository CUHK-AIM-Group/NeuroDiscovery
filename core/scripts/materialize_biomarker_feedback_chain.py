"""Convert the authoritative CS1 batch audit into a compact KG delta chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import sha256_file


DEFAULT_CLOSURE = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v1"
    r"\biomarker_discovery\tables\biomarker_closure_manifest.json"
)
DEFAULT_AUDIT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260803_fullv2_membership_v2\internal_method_comparison"
    r"\case1_negative_feedback_ablation_audit.csv"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_neurodiscovery_orders(
    ranking: dict[str, Any], archive: np.lib.npyio.NpzFile
) -> set[int]:
    trials: set[int] = set()
    for record in (ranking.get("orders") or {}).get("records") or []:
        if str(record.get("method")) != "neurodiscovery":
            continue
        frozen_order = np.asarray(archive[str(record["array_key"])])
        expected = str(record.get("order_sha256") or "")
        actual = hashlib.sha256(frozen_order.tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen order hash mismatch for trial {record['trial']}")
        trials.add(int(record["trial"]))
    return trials


def materialize(
    closure_path: Path,
    audit_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    ranking = dict(closure.get("source_ranking_manifest") or {})
    candidates_path = Path(str((ranking.get("candidate_table") or {}).get("path") or ""))
    orders_path = Path(str((ranking.get("orders") or {}).get("path") or ""))
    source_manifest_path = audit_path.parent / "case1_method_comparison_manifest.json"
    for path in (candidates_path, orders_path, audit_path, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(candidates_path) != str(
        (ranking.get("candidate_table") or {}).get("sha256")
    ):
        raise ValueError("frozen candidate table SHA-256 mismatch")
    if sha256_file(orders_path) != str((ranking.get("orders") or {}).get("sha256")):
        raise ValueError("frozen order archive SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    ranking_kg_hash = str(
        ((ranking.get("inputs") or {}).get("kg") or {}).get("sha256") or ""
    )
    if str(source_manifest.get("kg_sha256") or "") != ranking_kg_hash:
        raise ValueError("internal feedback audit and frozen ranking use different KGs")
    expected_all_tests_hash = str(
        (closure.get("provenance") or {}).get("source_all_tests_sha256") or ""
    )
    if str(source_manifest.get("all_tests_sha256") or "") != expected_all_tests_hash:
        raise ValueError("internal feedback audit and closure use different exhaustive tables")
    audit = pd.read_csv(audit_path)
    audit = audit[
        (audit["strategy"].astype(str) == "positive_only")
        & (audit["method"].astype(str) == "neurodiscovery_positive_only")
    ].copy()
    audit["trial"] = pd.to_numeric(audit["trial"], errors="raise").astype(int)
    audit["selected_total"] = pd.to_numeric(
        audit["selected_total"], errors="raise"
    ).astype(int)
    audit["batch_n"] = pd.to_numeric(audit["batch_n"], errors="raise").astype(int)
    audit = audit.sort_values(["trial", "selected_total"], kind="mergesort")
    expected_trials = set(range(int(ranking.get("n_trials") or 0)))
    if set(audit["trial"].unique()) != expected_trials:
        raise ValueError("feedback audit does not contain every frozen-ranking trial")
    for trial, frame in audit.groupby("trial", sort=True):
        batch_sizes = frame["batch_n"].to_numpy(dtype=int)
        selected = frame["selected_total"].to_numpy(dtype=int)
        if not np.array_equal(selected, np.cumsum(batch_sizes)):
            raise ValueError(f"non-contiguous feedback batches in trial {trial}")
        accounted = (
            pd.to_numeric(frame["batch_supported_feedback"], errors="raise").to_numpy(int)
            + pd.to_numeric(frame["batch_contradicted_feedback"], errors="raise").to_numpy(int)
            + pd.to_numeric(frame["batch_inconclusive_feedback"], errors="raise").to_numpy(int)
        )
        if not np.array_equal(accounted, batch_sizes):
            raise ValueError(f"feedback counts do not sum to batch size in trial {trial}")
        cumulative_supported = np.cumsum(
            pd.to_numeric(frame["batch_supported_feedback"], errors="raise").to_numpy(int)
        )
        observed_hits = pd.to_numeric(frame["observed_hits"], errors="raise").to_numpy(int)
        if not np.array_equal(cumulative_supported, observed_hits):
            raise ValueError(f"cumulative feedback is inconsistent in trial {trial}")

    target_dir = output_dir or closure_path.parent.parent / "benchmark"
    target_dir.mkdir(parents=True, exist_ok=True)
    delta_path = target_dir / "experimental_kg_delta.jsonl"
    previous_hash = "0" * 64
    records = 0
    with np.load(orders_path) as archive:
        frozen_trials = _verify_neurodiscovery_orders(ranking, archive)
        if frozen_trials != expected_trials:
            raise ValueError(
                f"missing NeuroDiscovery frozen orders: expected {expected_trials}, got {frozen_trials}"
            )
    with delta_path.open("w", encoding="utf-8") as handle:
        for row in audit.itertuples(index=False):
            stop = int(row.selected_total)
            payload = {
                "schema_version": "experimental-kg-batch-delta.v1",
                "task": "biomarker_discovery",
                "method": "neurodiscovery",
                "trial": int(row.trial),
                "batch": stop // int(row.batch_n) - 1,
                "stage": str(row.stage),
                "selected_total": stop,
                "candidate_count": int(row.batch_n),
                "gt_hits": int(row.batch_gt_hits),
                "supported_feedback": int(row.batch_supported_feedback),
                "contradicted_feedback": int(row.batch_contradicted_feedback),
                "inconclusive_feedback": int(row.batch_inconclusive_feedback),
                "previous_hash": previous_hash,
            }
            record_hash = canonical_hash(payload)
            payload["record_hash"] = record_hash
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            previous_hash = record_hash
            records += 1

    delta = {
        "schema_version": "experimental-kg-batch-delta.v1",
        "created_at": utc_now(),
        "path": str(delta_path),
        "sha256": sha256_file(delta_path),
        "records": records,
        "final_chain_hash": previous_hash,
        "mutates_formal_kg": False,
        "source_rankings_verified": True,
        "batch_feedback_verified": True,
        "batch_granularity": 250,
        "feedback_horizon_per_trial": int(audit["selected_total"].max()),
        "trials": int(audit["trial"].nunique()),
        "source_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "source_internal_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "source_orders": {"path": str(orders_path), "sha256": sha256_file(orders_path)},
        "source_candidates": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
    }
    manifest_path = target_dir / "legacy_feedback_manifest.json"
    manifest_path.write_text(
        json.dumps(delta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    delta["manifest_path"] = str(manifest_path)
    delta["manifest_sha256"] = sha256_file(manifest_path)
    closure["experimental_kg_delta"] = delta
    closure["formal_kg_mutated"] = False
    temporary = closure_path.with_suffix(closure_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(closure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(closure_path)
    return delta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            materialize(args.closure, args.audit, output_dir=args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
