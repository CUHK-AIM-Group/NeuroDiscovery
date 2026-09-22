"""Freeze method-blind eligibility for dynamic closed-loop hindcasting.

Static hindcasting needs one evaluable future interval. Dynamic hindcasting
additionally needs an early interval that can supply feedback and a disjoint
terminal interval that can evaluate the resulting ranking. This audit splits
the already locked static-primary matrix without opening method outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
)
from neurooracle.scripts.audit_case_study_hindcasting_executability import (
    _future_contract_edges,
    _required_atoms,
    component_summary,
)
from neurooracle.scripts.case_study_hindcasting_eval import (
    MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK,
    _future_indexes,
    load_future_claim_records,
)
from neurooracle.scripts.generate_case_study_frozen_baselines import FrozenGraphIndex
from neurooracle.scripts.generate_case_study_frozen_baselines import (
    load_semantic_claim_adjacencies,
)
from neurooracle.scripts.run_case_study_hindcasting import Window
from neurooracle.scripts.summarize_hindcasting_replicates import (
    _load_primary_eligibility,
)
from neurooracle.src.case_studies import case_study_by_name


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_SNAPSHOT_ROOT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "snapshots_full_v2_endpoint_v3"
)
DEFAULT_STATIC_ELIGIBILITY = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "optimization_protocol_20260812"
    / "locked_hindcasting_eligibility"
    / "eligibility_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "optimization_protocol_20260812"
    / "locked_dynamic_hindcasting_eligibility"
)

OUTPUT_FIELDS = (
    "case_study_id",
    "freeze_year",
    "future_start_year",
    "future_end_year",
    "feedback_start_year",
    "feedback_end_year",
    "terminal_start_year",
    "terminal_end_year",
    "complete_path_required",
    "early_unique_pairs",
    "early_complete_components",
    "terminal_unique_pairs",
    "terminal_complete_components",
    "dynamic_status",
    "status_reason",
    "analysis_tier",
)
PERFORMANCE_COLUMN_TOKENS = frozenset(
    {
        "method",
        "score",
        "rank",
        "ranking",
        "hit",
        "hits",
        "recall",
        "precision",
        "accuracy",
        "auc",
        "auroc",
        "performance",
        "validation",
        "metric",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def classify_dynamic_window(
    *,
    has_terminal_interval: bool,
    complete_path_required: bool,
    early_unique_pairs: int,
    early_complete_components: int,
    terminal_unique_pairs: int,
    terminal_complete_components: int,
    min_early_support: int,
    min_terminal_pairs: int,
) -> tuple[str, str, str]:
    """Return structural status, reason, and analysis tier without outcomes."""

    if not has_terminal_interval:
        return "non_executable", "no disjoint terminal interval remains", "excluded"
    if early_unique_pairs < min_early_support:
        return (
            "non_executable",
            f"only {early_unique_pairs} early relations can supply feedback",
            "excluded",
        )
    if complete_path_required and early_complete_components <= 0:
        return (
            "non_executable",
            "early evidence cannot support a complete multi-input path",
            "excluded",
        )
    if terminal_unique_pairs <= 0:
        return "non_executable", "no terminal relation is evaluable", "excluded"
    if complete_path_required and terminal_complete_components <= 0:
        return (
            "non_executable",
            "terminal evidence cannot support a complete multi-input path",
            "excluded",
        )
    if terminal_unique_pairs < min_terminal_pairs:
        return (
            "sparse",
            f"only {terminal_unique_pairs} terminal relations are evaluable",
            "exploratory",
        )
    return "executable", "", "primary"


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    writer.writerows({field: row[field] for field in OUTPUT_FIELDS} for row in rows)
    return handle.getvalue().encode("utf-8")


def _release_hashes(release: Mapping[str, Any]) -> dict[str, str]:
    files = release.get("files") or {}
    return {
        key: str((files.get(key) or {}).get("sha256") or "").upper()
        for key in ("knowledge_graph", "extracted_claims", "current_state")
    }


def _reject_performance_columns(fieldnames: Sequence[str]) -> None:
    contaminated = []
    for field in fieldnames:
        tokens = set(str(field).lower().replace("-", "_").split("_"))
        if tokens & PERFORMANCE_COLUMN_TOKENS:
            contaminated.append(str(field))
    if contaminated:
        raise ValueError(
            "static eligibility matrix contains method-performance columns: "
            f"{sorted(contaminated)}"
        )


def _static_primary_rows(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], Path]:
    manifest, primary_keys, matrix_path = _load_primary_eligibility(manifest_path)
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _reject_performance_columns(tuple(reader.fieldnames or ()))
        rows = [
            dict(row)
            for row in reader
            if row.get("analysis_tier") == "primary"
        ]
    observed = {
        (
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
        for row in rows
    }
    if observed != primary_keys:
        raise ValueError("static primary eligibility rows differ from the locked matrix")
    return manifest, rows, matrix_path


def audit_dynamic_eligibility(
    *,
    input_dir: Path,
    snapshot_root: Path,
    static_eligibility_manifest: Path,
    output_dir: Path,
    feedback_years: int,
    min_early_support: int,
    min_terminal_pairs: int,
    allow_relocated_canonical_artifacts: bool = False,
) -> dict[str, Any]:
    if feedback_years < 1 or min_early_support < 1 or min_terminal_pairs < 1:
        raise ValueError("dynamic eligibility thresholds must be positive")

    canonical = validate_canonical_kg_release(
        kg_path=input_dir / "knowledge_graph.json",
        claims_path=input_dir / "extracted_claims.jsonl",
        state_path=input_dir / "CURRENT_STATE.json",
        expected_sha256=CURRENT_CANONICAL_SHA256,
        allow_relocated_artifacts=allow_relocated_canonical_artifacts,
    )
    static_manifest, static_rows, static_matrix = _static_primary_rows(
        static_eligibility_manifest
    )
    static_release = (
        (static_manifest.get("source") or {}).get("canonical_release") or {}
    )
    if _release_hashes(static_release) != _release_hashes(canonical):
        raise ValueError("static eligibility and dynamic audit use different KG releases")
    declared_snapshot = Path(
        str((static_manifest.get("source") or {}).get("snapshot_root") or "")
    )
    if declared_snapshot.resolve() != snapshot_root.resolve():
        raise ValueError("dynamic snapshot root differs from static eligibility")

    case_ids = tuple(static_manifest.get("primary_case_study_ids") or ())
    future_records = load_future_claim_records(
        input_dir / "extracted_claims.jsonl",
        min_year=min(int(row["future_start_year"]) for row in static_rows),
        max_year=max(int(row["future_end_year"]) for row in static_rows),
        case_study_ids=set(case_ids),
    )
    records_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in future_records:
        for case_id in record.get("case_study_ids") or ():
            if case_id in case_ids:
                records_by_case[case_id].append(record)

    rows: list[dict[str, Any]] = []
    for freeze_year in sorted({int(row["freeze_year"]) for row in static_rows}):
        graph_path = snapshot_root / f"kg_{freeze_year}" / "knowledge_graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        print(f"[load] dynamic eligibility KG_{freeze_year}", flush=True)
        index = FrozenGraphIndex.load(graph_path)
        claims_path = graph_path.parent / "extracted_claims.jsonl"
        load_semantic_claim_adjacencies(claims_path, index, case_ids)
        for static_row in (
            row for row in static_rows if int(row["freeze_year"]) == freeze_year
        ):
            case_id = str(static_row["case_study_id"])
            case = case_study_by_name(case_id)
            future_start = int(static_row["future_start_year"])
            future_end = int(static_row["future_end_year"])
            feedback_end = min(future_end, freeze_year + feedback_years)
            terminal_start = feedback_end + 1
            complete_path = str(static_row["complete_path_required"]).lower() == "true"
            case_records = records_by_case.get(case_id, [])

            early_window = Window(freeze_year, future_start, feedback_end)
            _, early_stats = _future_indexes(
                input_dir / "extracted_claims.jsonl",
                index.concepts,
                index.direct_pairs,
                early_window.future_start_year,
                early_window.future_end_year,
                case_study_id=case_id,
                future_records=case_records,
                historical_endpoint_atoms=index.atoms,
            )
            early_components = 0
            if complete_path:
                early_edges = _future_contract_edges(
                    case_records, case, index, early_window
                )
                early_components = component_summary(
                    index, early_edges, _required_atoms(case)
                )["complete_components"]

            terminal_pairs = 0
            terminal_components = 0
            has_terminal = terminal_start <= future_end
            if has_terminal:
                terminal_window = Window(freeze_year, terminal_start, future_end)
                _, terminal_stats = _future_indexes(
                    input_dir / "extracted_claims.jsonl",
                    index.concepts,
                    index.direct_pairs,
                    terminal_window.future_start_year,
                    terminal_window.future_end_year,
                    case_study_id=case_id,
                    future_records=case_records,
                    historical_endpoint_atoms=index.atoms,
                )
                terminal_pairs = int(terminal_stats.get("future_unique_pairs") or 0)
                if complete_path:
                    terminal_edges = _future_contract_edges(
                        case_records, case, index, terminal_window
                    )
                    terminal_components = component_summary(
                        index, terminal_edges, _required_atoms(case)
                    )["complete_components"]

            early_pairs = int(early_stats.get("future_unique_pairs") or 0)
            status, reason, tier = classify_dynamic_window(
                has_terminal_interval=has_terminal,
                complete_path_required=complete_path,
                early_unique_pairs=early_pairs,
                early_complete_components=early_components,
                terminal_unique_pairs=terminal_pairs,
                terminal_complete_components=terminal_components,
                min_early_support=min_early_support,
                min_terminal_pairs=min_terminal_pairs,
            )
            rows.append(
                {
                    "case_study_id": case_id,
                    "freeze_year": freeze_year,
                    "future_start_year": future_start,
                    "future_end_year": future_end,
                    "feedback_start_year": future_start,
                    "feedback_end_year": feedback_end,
                    "terminal_start_year": terminal_start,
                    "terminal_end_year": future_end,
                    "complete_path_required": complete_path,
                    "early_unique_pairs": early_pairs,
                    "early_complete_components": early_components,
                    "terminal_unique_pairs": terminal_pairs,
                    "terminal_complete_components": terminal_components,
                    "dynamic_status": status,
                    "status_reason": reason,
                    "analysis_tier": tier,
                }
            )
            print(
                f"  {case_id}: {status}; early={early_pairs} terminal={terminal_pairs}",
                flush=True,
            )
        del index

    expected_rows = int(static_manifest.get("primary_windows") or 0)
    if len(rows) != expected_rows:
        raise ValueError(
            f"dynamic audit coverage mismatch: expected={expected_rows} observed={len(rows)}"
        )
    rows.sort(
        key=lambda row: (
            case_ids.index(str(row["case_study_id"])),
            int(row["freeze_year"]),
        )
    )
    matrix_bytes = _csv_bytes(rows)
    matrix_hash = _sha256_bytes(matrix_bytes)
    status_counts = dict(Counter(str(row["dynamic_status"]) for row in rows))
    tier_counts = dict(Counter(str(row["analysis_tier"]) for row in rows))
    primary_case_ids = [
        case_id
        for case_id in case_ids
        if any(
            row["case_study_id"] == case_id and row["analysis_tier"] == "primary"
            for row in rows
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "dynamic_eligibility_matrix_locked.csv"
    manifest_path = output_dir / "dynamic_eligibility_manifest.json"
    manifest = {
        "schema_version": "neurodiscovery-dynamic-hindcasting-eligibility.v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_all_task_dynamic_evaluation",
        "selection_contract": {
            "performance_columns_consumed": False,
            "source_matrix": "static primary eligibility only",
            "feedback_rule": (
                f"at least {min_early_support} structurally evaluable early relations"
            ),
            "terminal_rule": (
                f"at least {min_terminal_pairs} structurally evaluable terminal relations"
            ),
            "complete_path_rule": (
                "complete-path Case Studies require at least one complete component "
                "in both early and terminal intervals"
            ),
            "task_or_window_selection_after_method_scoring_permitted": False,
        },
        "feedback_years": feedback_years,
        "min_early_support": min_early_support,
        "min_terminal_pairs": min_terminal_pairs,
        "canonical_release": canonical,
        "snapshot_root": str(snapshot_root.resolve()),
        "source_static_eligibility_manifest": str(
            static_eligibility_manifest.resolve()
        ),
        "source_static_eligibility_manifest_sha256": _sha256(
            static_eligibility_manifest
        ),
        "source_static_eligibility_matrix": str(static_matrix.resolve()),
        "source_static_eligibility_matrix_sha256": _sha256(static_matrix),
        "input_static_primary_windows": len(static_rows),
        "status_counts": status_counts,
        "tier_counts": tier_counts,
        "primary_case_study_ids": primary_case_ids,
        "primary_case_studies": len(primary_case_ids),
        "primary_windows": tier_counts.get("primary", 0),
        "exploratory_windows": tier_counts.get("exploratory", 0),
        "excluded_windows": tier_counts.get("excluded", 0),
        "locked_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": matrix_hash,
            "rows": len(rows),
        },
    }

    if manifest_path.is_file():
        if not matrix_path.is_file() or _sha256(matrix_path) != matrix_hash:
            raise ValueError("existing dynamic eligibility lock differs from audit")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["registered_at"] = existing.get("registered_at")
        if existing != manifest:
            raise ValueError("existing dynamic eligibility manifest differs from audit")
        return existing
    if matrix_path.exists():
        raise ValueError("dynamic eligibility matrix exists without its manifest")
    temporary_matrix = matrix_path.with_suffix(".csv.tmp")
    temporary_matrix.write_bytes(matrix_bytes)
    temporary_matrix.replace(matrix_path)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--static-eligibility-manifest",
        type=Path,
        default=DEFAULT_STATIC_ELIGIBILITY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feedback-years", type=int, default=2)
    parser.add_argument("--min-early-support", type=int, default=2)
    parser.add_argument(
        "--min-terminal-pairs",
        type=int,
        default=MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK,
    )
    parser.add_argument(
        "--allow-relocated-canonical-artifacts",
        action="store_true",
        help=(
            "Accept an immutable byte-identical canonical copy at paths different "
            "from CURRENT_STATE.json; canonical hashes and sizes remain required."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = audit_dynamic_eligibility(
        input_dir=args.input_dir.resolve(),
        snapshot_root=args.snapshot_root.resolve(),
        static_eligibility_manifest=args.static_eligibility_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        feedback_years=args.feedback_years,
        min_early_support=args.min_early_support,
        min_terminal_pairs=args.min_terminal_pairs,
        allow_relocated_canonical_artifacts=(
            args.allow_relocated_canonical_artifacts
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "primary_case_studies": manifest["primary_case_studies"],
                "primary_windows": manifest["primary_windows"],
                "exploratory_windows": manifest["exploratory_windows"],
                "excluded_windows": manifest["excluded_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


# Updated: 2026-08-13 06:27:05 HKT - evaluate dynamic feedback and terminal truth with frozen semantic endpoint atoms.
