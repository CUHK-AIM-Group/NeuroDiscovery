"""Evaluate all frozen-baseline replicates with shared temporal indexes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from neurooracle.scripts.case_study_hindcasting_eval import (
    SEMANTIC_PROJECTION_VERSION,
    _future_indexes,
    _historical_pairs,
    evaluate,
    load_future_claim_records,
)
from neurooracle.scripts.run_case_study_hindcasting import DEFAULT_WINDOWS, parse_window
from neurooracle.scripts.temporal_hindcasting_core import load_kg_index
from neurooracle.src.case_studies import case_study_by_name, list_case_study_names
from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_source_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("neurodiscovery", "sciagents", "openscholar_rag")
DEFAULT_SNAPSHOT_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/snapshots_full_v2_endpoint_v3"
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ".tmp"
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_source_bundle(
    manifest_path: Path | None, *, verify_references: bool = True
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest_path = manifest_path.resolve()
    archive_root = (manifest_path.parent / "files").resolve()
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    return verify_source_bundle(
        manifest_path,
        require_live_source=not executing_from_archive,
        verify_references=verify_references,
    )


def _validate_snapshot_root(
    generation_manifest: Mapping[str, Any],
    requested_snapshot_root: Path,
) -> str:
    """Prevent generation and evaluation from silently using different freezes."""

    declared = generation_manifest.get("snapshot_root")
    if not declared:
        return "legacy_manifest_missing_snapshot_root"
    declared_path = Path(str(declared)).resolve()
    requested_path = requested_snapshot_root.resolve()
    if declared_path != requested_path:
        raise ValueError(
            "generation/evaluation snapshot mismatch: "
            f"generation={declared_path}, evaluation={requested_path}"
        )
    return "matched_generation_manifest"


def _select_generation_runs(
    generation_manifest: dict[str, Any],
    *,
    methods: list[str],
    seeds: list[int],
    cases: list[str],
    windows: list[Any],
) -> list[dict[str, Any]]:
    """Select only manifest-declared executable task-year combinations."""

    requested_windows = {
        (window.freeze_year, window.future_start_year, window.future_end_year)
        for window in windows
    }
    selected = [
        dict(row)
        for row in generation_manifest.get("runs", [])
        if str(row.get("method")) in set(methods)
        and int(row.get("seed", -1)) in set(seeds)
        and str(row.get("case_study_id")) in set(cases)
        and (
            int(row.get("freeze_year", -1)),
            int(row.get("future_start_year", -1)),
            int(row.get("future_end_year", -1)),
        )
        in requested_windows
    ]
    return sorted(
        selected,
        key=lambda row: (
            int(row["freeze_year"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["case_study_id"]),
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_bundle = _runtime_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    generation_manifest_path = (
        args.generation_root / "generation_manifest.json"
    ).resolve()
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    if generation_manifest.get("source_bundle") != source_bundle:
        raise ValueError(
            "generation manifest and evaluation use different source bundles"
        )
    generation_manifest_hash = sha256_file(generation_manifest_path)
    snapshot_validation = _validate_snapshot_root(
        generation_manifest,
        args.snapshot_root,
    )
    cases = args.case_study_ids or list(generation_manifest["case_studies"])
    methods = args.methods or list(
        generation_manifest.get("methods")
        or [generation_manifest.get("method")]
    )
    declared_methods = set(
        generation_manifest.get("methods")
        or [generation_manifest.get("method")]
    )
    undeclared_methods = sorted(set(methods) - declared_methods)
    if undeclared_methods:
        raise ValueError(
            "requested method(s) are not declared by the generation manifest: "
            + ", ".join(undeclared_methods)
        )
    seeds = args.seeds or [int(seed) for seed in generation_manifest["seeds"]]
    for case_id in cases:
        case_study_by_name(case_id)
    selected_runs = _select_generation_runs(
        generation_manifest,
        methods=methods,
        seeds=seeds,
        cases=cases,
        windows=args.windows,
    )
    if not selected_runs:
        raise ValueError("No generation-manifest runs match the requested evaluation matrix")

    args.output_root.mkdir(parents=True, exist_ok=True)
    future_records = load_future_claim_records(
        args.future_claims,
        min_year=min(window.future_start_year for window in args.windows),
        max_year=max(window.future_end_year for window in args.windows),
    )
    rows: list[dict[str, Any]] = []
    for window in args.windows:
        window_runs = [
            row
            for row in selected_runs
            if int(row["freeze_year"]) == window.freeze_year
            and int(row["future_start_year"]) == window.future_start_year
            and int(row["future_end_year"]) == window.future_end_year
        ]
        if not window_runs:
            continue
        snapshot = args.snapshot_root / f"kg_{window.freeze_year}"
        kg_path = snapshot / "knowledge_graph.json"
        historical_claims_path = snapshot / "extracted_claims.jsonl"
        print(f"[index] evaluation KG_{window.freeze_year}", flush=True)
        kg_index = load_kg_index(kg_path)
        concepts, edges, _ = kg_index
        historical_counter: Counter[str] = Counter()
        historical_endpoint_atoms: dict[str, set[str]] = {}
        historical_pairs = _historical_pairs(
            edges,
            claims_path=historical_claims_path,
            concepts=concepts,
            stats=historical_counter,
            endpoint_atoms_out=historical_endpoint_atoms,
        )
        historical_stats = dict(historical_counter)
        future_by_case: dict[str, tuple[dict[str, Any], dict[str, int]]] = {}
        active_cases = sorted({str(row["case_study_id"]) for row in window_runs})
        for case_id in active_cases:
            future_by_case[case_id] = _future_indexes(
                args.future_claims,
                concepts,
                historical_pairs,
                window.future_start_year,
                window.future_end_year,
                case_study_id=case_id,
                future_records=future_records,
                historical_endpoint_atoms=historical_endpoint_atoms,
            )
        for generation_run in window_runs:
            method = str(generation_run["method"])
            seed = int(generation_run["seed"])
            case_id = str(generation_run["case_study_id"])
            source = Path(generation_run["hypotheses_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hash = sha256_file(source)
            declared_source_hash = str(
                generation_run.get("hypotheses_sha256") or ""
            ).upper()
            if declared_source_hash and source_hash != declared_source_hash:
                raise ValueError(f"generated hypotheses hash mismatch: {source}")
            output = (
                args.output_root / method / f"seed_{seed:02d}" / case_id
                / window.label / "hindcasting"
            )
            metrics_path = output / "metrics.json"
            execution_identity = {
                "source_bundle": source_bundle,
                "generation_manifest_sha256": generation_manifest_hash,
                "hypotheses_sha256": source_hash,
                "method": method,
                "seed": seed,
                "case_study_id": case_id,
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
                "top_k": list(args.top_k),
                "random_trials": args.random_trials,
            }
            if metrics_path.is_file() and not args.force:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("execution_identity") != execution_identity:
                    raise ValueError(
                        f"evaluation checkpoint identity mismatch at {metrics_path}; "
                        "rerun with --force"
                    )
            else:
                future, stats = future_by_case[case_id]
                metrics = evaluate(
                    kg_path=kg_path,
                    hypotheses_path=source,
                    future_claims_path=args.future_claims,
                    output_dir=output,
                    freeze_year=window.freeze_year,
                    future_start_year=window.future_start_year,
                    future_end_year=window.future_end_year,
                    top_ks=args.top_k,
                    random_trials=args.random_trials,
                    seed=seed,
                    case_study_id=case_id,
                    method=method,
                    kg_index=kg_index,
                    historical_pairs=historical_pairs,
                    historical_claims_path=historical_claims_path,
                    historical_stats=historical_stats,
                    historical_endpoint_atoms=historical_endpoint_atoms,
                    future_index=future,
                    future_stats=stats,
                )
                metrics["execution_identity"] = execution_identity
                _atomic_write_json(metrics_path, metrics)
            rows.append({
                "method": method,
                "seed": seed,
                "case_study_id": case_id,
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
                "metrics_path": str(metrics_path),
                "n_hypotheses": metrics["n_hypotheses"],
            })
            print(
                f"[evaluated] {method} seed={seed} {case_id} KG_{window.freeze_year}",
                flush=True,
            )
        del future_by_case, historical_pairs, historical_endpoint_atoms, kg_index, concepts, edges

    source_bundle_at_completion = _runtime_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    if source_bundle_at_completion != source_bundle:
        raise ValueError("source bundle changed during frozen-baseline evaluation")
    manifest = {
        "schema_version": "case-study-frozen-baseline-evaluation.v2",
        "semantic_projection": SEMANTIC_PROJECTION_VERSION,
        "generation_root": str(args.generation_root),
        "generation_manifest": {
            "path": str(generation_manifest_path),
            "sha256": generation_manifest_hash,
        },
        "snapshot_root": str(args.snapshot_root),
        "snapshot_validation": snapshot_validation,
        "future_claims": str(args.future_claims),
        "methods": sorted({str(row["method"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "case_studies": sorted({str(row["case_study_id"]) for row in rows}),
        "windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(
                {
                    (
                        int(row["freeze_year"]),
                        int(row["future_start_year"]),
                        int(row["future_end_year"]),
                    )
                    for row in rows
                }
            )
        ],
        "top_k": args.top_k,
        "random_trials": args.random_trials,
        "source_bundle": source_bundle_at_completion,
        "runs": rows,
    }
    _atomic_write_json(args.output_root / "evaluation_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--future-claims", type=Path, default=ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl")
    parser.add_argument("--source-bundle-manifest", type=Path, default=None)
    parser.add_argument("--skip-source-bundle-reference-rehash", action="store_true")
    parser.add_argument("--case-study-ids", nargs="*", choices=list_case_study_names(), default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Method labels declared by the generation manifest. Canonical "
            "baselines are neurodiscovery, sciagents, and openscholar_rag; "
            "explicit labels also support auditable development ablations."
        ),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps({"output_root": str(args.output_root), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 07:31 HKT
# Updated: 2026-08-13 06:26:23 HKT - share frozen endpoint atoms across baseline future-truth evaluation.
