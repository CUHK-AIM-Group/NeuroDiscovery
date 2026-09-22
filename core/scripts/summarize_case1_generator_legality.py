"""Compare adapted and clean-upstream CS1 hypothesis-generator legality.

The primary matched comparison uses the first N requested ranks from every
adapted seed and the same N slots from each clean-upstream trial.  The adapted
full-batch rate is reported separately so truncating to N cannot hide failures
that occur near the end of a larger generation batch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)
METHODS = BASELINE_METHODS + ("neurodiscovery",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted-mapped", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-audit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matched-slots", type=int, default=10)
    parser.add_argument("--adapted-full-slots", type=int, default=80)
    parser.add_argument("--trials", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--neurodiscovery-overlays",
        type=Path,
        help="Directory containing seed_<seed>_trial_<trial>.jsonl overlays.",
    )
    parser.add_argument("--neurodiscovery-seed", type=int, default=260810)
    parser.add_argument(
        "--registry",
        type=Path,
        help="Exact public CS1 registry; required with NeuroDiscovery overlays.",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(BASELINE_METHODS)
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_probability(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def audit_adapted_group(frame: pd.DataFrame, requested: int) -> dict[str, Any]:
    """Apply the strict slot contract to one adapted method/seed group."""
    ranks = pd.to_numeric(frame["generated_rank"], errors="coerce")
    used_ids: set[str] = set()
    legal = 0
    errors: list[str] = []
    for rank in range(1, requested + 1):
        rows = frame.loc[ranks.eq(rank)]
        if len(rows) != 1:
            errors.append(f"rank_{rank}_count_{len(rows)}")
            continue
        row = rows.iloc[0]
        candidate_id = str(row.get("mapped_candidate_id") or "").strip()
        rationale = row.get("generated_rationale")
        if str(row.get("mapping_status") or "") != "mapped" or not candidate_id:
            errors.append(f"rank_{rank}_unmapped")
        elif candidate_id in used_ids:
            errors.append(f"rank_{rank}_duplicate_id")
        elif not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"rank_{rank}_missing_rationale")
        elif not _finite_probability(row.get("generated_confidence")):
            errors.append(f"rank_{rank}_invalid_confidence")
        else:
            legal += 1
            used_ids.add(candidate_id)
    return {
        "requested_slots": requested,
        "legal_slots": legal,
        "legal_rate": legal / requested if requested else None,
        "complete_legal_batch": legal == requested,
        "errors": ";".join(errors),
    }


def _registry_sha(manifest: dict[str, Any]) -> str:
    registry = manifest.get("registry")
    if isinstance(registry, dict):
        return str(registry.get("sha256") or "")
    return str(manifest.get("registry_sha256") or "")


def _adapted_rows(
    paths: list[Path],
    *,
    methods: list[str],
    trials: list[int],
    matched_slots: int,
    full_slots: int,
) -> list[dict[str, Any]]:
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    output: list[dict[str, Any]] = []
    for method in methods:
        method_frame = frame.loc[frame["method"].astype(str).eq(method)]
        for seed in trials:
            group = method_frame.loc[
                pd.to_numeric(method_frame["seed"], errors="coerce").eq(seed)
            ]
            for condition, requested in (
                ("adapted_matched", matched_slots),
                ("adapted_full", full_slots),
            ):
                result = audit_adapted_group(group, requested)
                output.append(
                    {
                        "method": method,
                        "condition": condition,
                        "trial": seed,
                        "measurement_status": "measured",
                        "process_status": "success",
                        **result,
                    }
                )
    return output


def _raw_rows(
    root: Path,
    *,
    methods: list[str],
    requested: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    combined_path = root / "unadapted_trial_audit.csv"
    combined = pd.read_csv(combined_path, low_memory=False) if combined_path.exists() else None
    for method in methods:
        if combined is not None:
            frame = combined.loc[combined["method"].astype(str).eq(method)]
        else:
            path = root / method / "unadapted_trial_audit.csv"
            frame = pd.read_csv(path, low_memory=False)
        for _, row in frame.iterrows():
            legal = int(row.get("schema_valid_slots") or 0)
            output.append(
                {
                    "method": method,
                    "condition": "raw_upstream",
                    "trial": int(row["trial"]),
                    "measurement_status": "measured",
                    "process_status": str(row.get("process_status") or ""),
                    "requested_slots": requested,
                    "legal_slots": legal,
                    "legal_rate": legal / requested,
                    "complete_legal_batch": legal == requested,
                    "errors": str(row.get("schema_errors") or row.get("error") or "")[:2000],
                }
            )
    return output


def audit_neurodiscovery_group(
    records: list[dict[str, Any]],
    registry_ids: set[str],
    requested: int,
) -> dict[str, Any]:
    """Audit native closed-loop proposals without reading their outcomes."""
    legal = 0
    used_ids: set[str] = set()
    errors: list[str] = []
    required_tuple_fields = ("disease", "feature_family", "map_group", "roi_key", "source")
    for rank in range(1, requested + 1):
        if rank > len(records):
            errors.append(f"rank_{rank}_count_0")
            continue
        record = records[rank - 1]
        candidate_id = str(record.get("candidate_id") or "").strip()
        candidate_tuple = record.get("candidate_tuple")
        if not candidate_id or candidate_id not in registry_ids:
            errors.append(f"rank_{rank}_unregistered")
        elif candidate_id in used_ids:
            errors.append(f"rank_{rank}_duplicate_id")
        elif str(record.get("hypothesis_id") or "") != candidate_id:
            errors.append(f"rank_{rank}_hypothesis_id_mismatch")
        elif str(record.get("case_study_id") or "") != "case1_transdiagnostic":
            errors.append(f"rank_{rank}_scope_mismatch")
        elif not isinstance(candidate_tuple, dict) or any(
            not str(candidate_tuple.get(field) or "").strip()
            for field in required_tuple_fields
        ):
            errors.append(f"rank_{rank}_invalid_candidate_tuple")
        else:
            legal += 1
            used_ids.add(candidate_id)
    return {
        "requested_slots": requested,
        "legal_slots": legal,
        "legal_rate": legal / requested if requested else None,
        "complete_legal_batch": legal == requested,
        "errors": ";".join(errors),
    }


def _load_registry_subset(path: Path, selected_ids: set[str]) -> set[str]:
    matched: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate_id = str(json.loads(line).get("candidate_id") or "")
            if candidate_id in selected_ids:
                matched.add(candidate_id)
                if len(matched) == len(selected_ids):
                    break
    return matched


def _neurodiscovery_rows(
    overlay_dir: Path,
    registry_path: Path,
    *,
    seed: int,
    trials: list[int],
    matched_slots: int,
    full_slots: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    read_slots = max(matched_slots, full_slots)
    records_by_trial: dict[int, list[dict[str, Any]]] = {}
    prefix_hashes: dict[str, str] = {}
    selected_ids: set[str] = set()
    for trial in trials:
        path = overlay_dir / f"seed_{seed}_trial_{trial:02d}.jsonl"
        records: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in itertools.islice(handle, read_slots):
                    digest.update(line.encode("utf-8"))
                    record = json.loads(line)
                    records.append(record)
                    candidate_id = str(record.get("candidate_id") or "").strip()
                    if candidate_id:
                        selected_ids.add(candidate_id)
        records_by_trial[trial] = records
        prefix_hashes[str(trial)] = digest.hexdigest()

    registry_ids = _load_registry_subset(registry_path, selected_ids)
    output: list[dict[str, Any]] = []
    for trial in trials:
        records = records_by_trial[trial]
        for condition, requested in (
            ("adapted_matched", matched_slots),
            ("adapted_full", full_slots),
        ):
            result = audit_neurodiscovery_group(records, registry_ids, requested)
            output.append(
                {
                    "method": "neurodiscovery",
                    "condition": condition,
                    "trial": trial,
                    "measurement_status": "measured",
                    "process_status": "success" if len(records) >= requested else "incomplete",
                    **result,
                }
            )
    provenance = {
        "overlay_dir": str(overlay_dir),
        "seed": seed,
        "trials": trials,
        "prefix_slots_per_trial": read_slots,
        "prefix_sha256_by_trial": prefix_hashes,
        "selected_unique_candidate_ids": len(selected_ids),
        "registered_selected_candidate_ids": len(registry_ids),
        "outcome_fields_used_for_legality": False,
    }
    return output, provenance


def _condition_summary(frame: pd.DataFrame, condition: str) -> dict[str, Any]:
    group = frame.loc[
        frame["condition"].eq(condition) & frame["measurement_status"].eq("measured")
    ].copy()
    if group.empty:
        return {
            "measurement_status": "not_applicable",
            "trials": 0,
            "legal_slots": None,
            "requested_slots": None,
            "legal_rate": None,
            "trial_legal_rate_mean": None,
            "trial_legal_rate_variance": None,
            "complete_trials": None,
        }
    legal = int(pd.to_numeric(group["legal_slots"], errors="coerce").sum())
    requested = int(pd.to_numeric(group["requested_slots"], errors="coerce").sum())
    complete = group["complete_legal_batch"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    trial_rates = pd.to_numeric(group["legal_rate"], errors="coerce").dropna()
    return {
        "measurement_status": "measured",
        "trials": len(group),
        "legal_slots": legal,
        "requested_slots": requested,
        "legal_rate": legal / requested if requested else None,
        "trial_legal_rate_mean": float(trial_rates.mean()) if len(trial_rates) else None,
        "trial_legal_rate_variance": (
            float(trial_rates.var(ddof=1)) if len(trial_rates) > 1 else 0.0
        ),
        "complete_trials": int(complete.sum()),
    }


def summarize_by_framework(trials: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_frame = trials.loc[trials["method"].eq(method)]
        row: dict[str, Any] = {"method": method}
        for condition in ("adapted_matched", "adapted_full", "raw_upstream"):
            summary = _condition_summary(method_frame, condition)
            for key, value in summary.items():
                row[f"{condition}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(args.methods)
    neurodiscovery_rows: list[dict[str, Any]] = []
    neurodiscovery_provenance: dict[str, Any] | None = None
    if args.neurodiscovery_overlays is not None:
        if args.registry is None:
            raise ValueError("--registry is required with --neurodiscovery-overlays")
        if "neurodiscovery" not in methods:
            methods.append("neurodiscovery")
        neurodiscovery_rows, neurodiscovery_provenance = _neurodiscovery_rows(
            args.neurodiscovery_overlays,
            args.registry,
            seed=args.neurodiscovery_seed,
            trials=list(args.trials),
            matched_slots=args.matched_slots,
            full_slots=args.adapted_full_slots,
        )
    adapted_methods = [method for method in methods if method != "neurodiscovery"]
    trials = pd.DataFrame(
        _adapted_rows(
            args.adapted_mapped,
            methods=adapted_methods,
            trials=list(args.trials),
            matched_slots=args.matched_slots,
            full_slots=args.adapted_full_slots,
        )
        + neurodiscovery_rows
        + _raw_rows(
            args.raw_audit_root,
            methods=methods,
            requested=args.matched_slots,
        )
    )
    trials.to_csv(args.out_dir / "generator_legality_by_trial.csv", index=False)
    summary = summarize_by_framework(trials, methods)
    summary.to_csv(args.out_dir / "generator_legality_by_framework.csv", index=False)

    raw_manifest_path = args.raw_audit_root / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    adapted_registry_hashes = []
    for path in args.adapted_mapped:
        registry_path = path.parent / "cs1_public_registry.jsonl"
        if registry_path.exists():
            adapted_registry_hashes.append(sha256_file(registry_path))
            continue
        manifest_path = path.parent / "official_adapter_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            registry_sha = _registry_sha(manifest)
            if registry_sha:
                adapted_registry_hashes.append(registry_sha)
    raw_registry_sha = _registry_sha(raw_manifest)
    neurodiscovery_registry_sha = sha256_file(args.registry) if args.registry else ""
    all_registry_hashes = (
        adapted_registry_hashes
        + ([raw_registry_sha] if raw_registry_sha else [])
        + ([neurodiscovery_registry_sha] if neurodiscovery_registry_sha else [])
    )
    provenance = {
        "schema_version": "case1-generator-legality-comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adapted_mapped": [str(path) for path in args.adapted_mapped],
        "adapted_mapped_sha256": [sha256_file(path) for path in args.adapted_mapped],
        "raw_audit_root": str(args.raw_audit_root),
        "matched_slots_per_trial": args.matched_slots,
        "adapted_full_slots_per_trial": args.adapted_full_slots,
        "adapted_registry_sha256": adapted_registry_hashes,
        "raw_registry_sha256": raw_registry_sha,
        "neurodiscovery_registry_sha256": neurodiscovery_registry_sha,
        "same_registry": bool(all_registry_hashes and len(set(all_registry_hashes)) == 1),
        "neurodiscovery": neurodiscovery_provenance,
        "denominator_policy": (
            "intention_to_generate: missing output, process failure, unsupported interface, "
            "invalid schema, duplicate, and unmapped slots all count as errors"
        ),
        "manual_repair": False,
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
