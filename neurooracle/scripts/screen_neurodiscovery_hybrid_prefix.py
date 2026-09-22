"""Screen hybrid-prefix protection on fixed NeuroDiscovery candidate branches.

This development-only utility never reads future claims and never generates new
hypotheses. It re-merges the general and task-scoped branches already produced
by a matching frozen KG run so one shared prefix setting can be selected before
the held-out hindcasting windows are opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates import (
    _merge_hybrid_candidate_payloads,
)
from neurooracle.scripts.run_case_study_hindcasting import Window, parse_window
from neurooracle.src.case_studies import case_study_by_name


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOWS = (Window(2016, 2017, 2017), Window(2017, 2018, 2018))
DEFAULT_CASES = ("case1_transdiagnostic", "biomarker_discovery")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _method_label(protect_general_top_k: int) -> str:
    return f"neurodiscovery_prefix_k{protect_general_top_k:03d}"


def _resolve_path(raw: str, generation_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path.resolve()
    for candidate in (ROOT / path, generation_root / path, path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _select_source_runs(
    manifest: dict[str, Any],
    *,
    case_ids: tuple[str, ...],
    windows: tuple[Window, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    allowed = {
        (window.freeze_year, window.future_start_year, window.future_end_year)
        for window in windows
    }
    selected: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in manifest.get("runs") or ():
        window = (
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
        key = (window[0], str(row["case_study_id"]), int(row["seed"]))
        if (
            str(row.get("method")) != "neurodiscovery"
            or window not in allowed
            or key[1] not in case_ids
            or key[2] not in seeds
        ):
            continue
        if key in selected:
            raise ValueError(f"duplicate source generation run: {key}")
        selected[key] = dict(row)
    missing = [
        (window.freeze_year, case_id, seed)
        for window in windows
        for case_id in case_ids
        for seed in seeds
        if (window.freeze_year, case_id, seed) not in selected
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} source run(s): {missing[:5]}")
    return selected


def _branch_paths(source_hypotheses: Path) -> tuple[Path, Path]:
    branch_dir = source_hypotheses.parent / "candidate_branches"
    general = branch_dir / "general_graph.json"
    scoped = branch_dir / "task_scoped.json"
    if not general.is_file() or not scoped.is_file():
        raise FileNotFoundError(f"candidate branches missing under {branch_dir}")
    return general.resolve(), scoped.resolve()


def screen(args: argparse.Namespace) -> dict[str, Any]:
    case_ids = tuple(args.case_study_ids)
    windows = tuple(args.windows)
    seeds = tuple(sorted(set(map(int, args.seeds))))
    prefixes = tuple(sorted(set(map(int, args.protect_general_top_k))))
    if any(value < 0 or value > args.max_candidates for value in prefixes):
        raise ValueError("protect-general-top-k must be within [0, max-candidates]")
    for case_id in case_ids:
        case_study_by_name(case_id)

    manifest_path = args.generation_root / "generation_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_runs = _select_source_runs(
        source_manifest,
        case_ids=case_ids,
        windows=windows,
        seeds=seeds,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    branch_audit: dict[str, Any] = {}
    for window in windows:
        for case_id in case_ids:
            for seed in seeds:
                source_row = source_runs[(window.freeze_year, case_id, seed)]
                source_path = _resolve_path(
                    str(source_row["hypotheses_path"]), args.generation_root
                )
                general_path, scoped_path = _branch_paths(source_path)
                general_payload = json.loads(general_path.read_text(encoding="utf-8"))
                scoped_payload = json.loads(scoped_path.read_text(encoding="utf-8"))
                audit_key = f"{window.freeze_year}:{case_id}:seed{seed}"
                branch_audit[audit_key] = {
                    "general_path": str(general_path),
                    "general_sha256": _sha256(general_path),
                    "scoped_path": str(scoped_path),
                    "scoped_sha256": _sha256(scoped_path),
                }
                for prefix in prefixes:
                    method = _method_label(prefix)
                    merged = _merge_hybrid_candidate_payloads(
                        copy.deepcopy(general_payload),
                        copy.deepcopy(scoped_payload),
                        case_study_id=case_id,
                        max_candidates=args.max_candidates,
                        task_scope_fraction=args.task_scope_fraction,
                        protect_general_top_k=prefix,
                    )
                    metadata = dict(merged.get("metadata") or {})
                    pool = dict(metadata.get("candidate_pool") or {})
                    pool["scoped_fraction_actual"] = (
                        pool.get("scoped_included", 0) / len(merged["hypotheses"])
                        if merged["hypotheses"]
                        else 0.0
                    )
                    metadata["candidate_pool"] = pool
                    metadata["fixed_candidate_hybrid_prefix_screen"] = {
                        "schema_version": "neurodiscovery-hybrid-prefix-screen.v1",
                        "protect_general_top_k": prefix,
                        "source_hypotheses": str(source_path),
                        "source_hypotheses_sha256": _sha256(source_path),
                        "uses_future_outcomes": False,
                        "final_generation_benchmark": False,
                    }
                    merged["metadata"] = metadata
                    output = (
                        args.output_root
                        / method
                        / f"seed_{seed:02d}"
                        / case_id
                        / window.label
                        / "hypotheses_raw.json"
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    runs.append(
                        {
                            "method": method,
                            "seed": seed,
                            "case_study_id": case_id,
                            "freeze_year": window.freeze_year,
                            "future_start_year": window.future_start_year,
                            "future_end_year": window.future_end_year,
                            "kg_path": source_row["kg_path"],
                            "historical_claims_path": source_row["historical_claims_path"],
                            "hypotheses_path": str(output.resolve()),
                            "n_hypotheses": len(merged.get("hypotheses") or ()),
                            "protect_general_top_k": prefix,
                            "scoped_fraction_actual": pool["scoped_fraction_actual"],
                        }
                    )

    manifest = {
        "schema_version": "neurodiscovery-hybrid-prefix-screen.v1",
        "purpose": "fixed-candidate development screening; not final generation benchmark",
        "uses_future_outcomes": False,
        "source_generation_root": str(args.generation_root.resolve()),
        "source_generation_manifest": str(manifest_path.resolve()),
        "source_generation_manifest_sha256": _sha256(manifest_path),
        "snapshot_root": source_manifest.get("snapshot_root"),
        "case_studies": list(case_ids),
        "windows": [window.__dict__ for window in windows],
        "seeds": list(seeds),
        "protect_general_top_k": list(prefixes),
        "max_candidates": args.max_candidates,
        "task_scope_fraction": args.task_scope_fraction,
        "methods": [_method_label(value) for value in prefixes],
        "branch_audit": branch_audit,
        "runs": runs,
    }
    (args.output_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-study-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--windows", nargs="+", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--protect-general-top-k", nargs="+", type=int, default=[25, 50, 75, 100]
    )
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--task-scope-fraction", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    result = screen(parse_args())
    print(json.dumps({"runs": len(result["runs"]), "methods": result["methods"]}, indent=2))
