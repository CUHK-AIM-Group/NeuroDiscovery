"""Screen shared NeuroDiscovery pool and ranker ablations on fixed candidates.

This development-only utility never reads future claims and never generates new
hypotheses.  It reuses the general and task-scoped branches from a frozen-KG
generation run, then separates candidate-source coverage from static ranking:

* general-only candidates with their legacy order;
* task-scoped candidates with their legacy order;
* an unprotected hybrid interleave with its legacy order; and
* the identical hybrid candidates reordered by the frozen relation-aware policy.

The resulting manifest is compatible with the hindcasting evaluator, but is
explicitly marked as a fixed-candidate development screen rather than a final
generation benchmark.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Mapping

from neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates import (
    _merge_hybrid_candidate_payloads,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    Window,
    _enforce_fixed_budget,
    parse_window,
)
from neurooracle.scripts.screen_neurodiscovery_hybrid_prefix import (
    DEFAULT_CASES,
    DEFAULT_WINDOWS,
    _branch_paths,
    _resolve_path,
    _select_source_runs,
    _sha256,
)
from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.hindcasting_static_policy import annotate_relation_aware_payload
from neurooracle.src.hypothesis_cli import load_graph


METHOD_GENERAL_LEGACY = "neurodiscovery_pool_general_legacy"
METHOD_SCOPED_LEGACY = "neurodiscovery_pool_scoped_legacy"
METHOD_HYBRID_LEGACY = "neurodiscovery_pool_hybrid_legacy"
METHOD_HYBRID_RELATION = "neurodiscovery_pool_hybrid_relation_aware"
METHODS = (
    METHOD_GENERAL_LEGACY,
    METHOD_SCOPED_LEGACY,
    METHOD_HYBRID_LEGACY,
    METHOD_HYBRID_RELATION,
)


def _legacy_pool_variants(
    general_payload: Mapping[str, Any],
    scoped_payload: Mapping[str, Any],
    *,
    case_study_id: str,
    max_candidates: int,
    task_scope_fraction: float,
) -> dict[str, dict[str, Any]]:
    """Return independent general, scoped, and unprotected hybrid payloads."""

    general = copy.deepcopy(dict(general_payload))
    scoped = copy.deepcopy(dict(scoped_payload))
    hybrid = _merge_hybrid_candidate_payloads(
        copy.deepcopy(dict(general_payload)),
        copy.deepcopy(dict(scoped_payload)),
        case_study_id=case_study_id,
        max_candidates=max_candidates,
        task_scope_fraction=task_scope_fraction,
        protect_general_top_k=0,
    )
    return {
        METHOD_GENERAL_LEGACY: general,
        METHOD_SCOPED_LEGACY: scoped,
        METHOD_HYBRID_LEGACY: hybrid,
    }


def _candidate_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get("id") or "") for row in payload.get("hypotheses") or ())


def _tag_screen_payload(
    payload: Mapping[str, Any],
    *,
    method: str,
    pool_mode: str,
    score_family: str,
    source_hypotheses: Path,
    source_hypotheses_sha256: str,
    general_path: Path,
    general_sha256: str,
    scoped_path: Path,
    scoped_sha256: str,
    kg_path: Path,
    kg_sha256: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    metadata = dict(result.get("metadata") or {})
    metadata["fixed_candidate_pool_ranker_screen"] = {
        "schema_version": "neurodiscovery-fixed-candidate-ablation.v1",
        "method": method,
        "candidate_pool_mode": pool_mode,
        "static_score_family": score_family,
        "protect_general_top_k": 0 if pool_mode == "hybrid" else None,
        "source_hypotheses": str(source_hypotheses),
        "source_hypotheses_sha256": source_hypotheses_sha256,
        "general_branch": str(general_path),
        "general_branch_sha256": general_sha256,
        "task_scoped_branch": str(scoped_path),
        "task_scoped_branch_sha256": scoped_sha256,
        "frozen_kg": str(kg_path),
        "frozen_kg_sha256": kg_sha256,
        "uses_future_outcomes": False,
        "generates_new_hypotheses": False,
        "final_generation_benchmark": False,
    }
    result["metadata"] = metadata
    result["n_hypotheses"] = len(result.get("hypotheses") or ())
    return result


def screen(args: argparse.Namespace) -> dict[str, Any]:
    case_ids = tuple(args.case_study_ids)
    windows = tuple(args.windows)
    seeds = tuple(sorted(set(map(int, args.seeds))))
    if args.max_candidates < 1:
        raise ValueError("max-candidates must be positive")
    if not 0.0 <= args.task_scope_fraction <= 1.0:
        raise ValueError("task-scope-fraction must be in [0, 1]")
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
    fixed_budget = args.fixed_budget
    if fixed_budget is None:
        fixed_budget = int(source_manifest.get("target_per_case_study") or 0)
    if fixed_budget < 1:
        raise ValueError("fixed-budget must be positive or inferable from the source manifest")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    kg_hash_cache: dict[Path, str] = {}
    file_hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        cache = kg_hash_cache if path.name == "knowledge_graph.json" else file_hash_cache
        if path not in cache:
            cache[path] = _sha256(path)
        return cache[path]

    for window in windows:
        window_rows = [
            source_runs[(window.freeze_year, case_id, seed)]
            for case_id in case_ids
            for seed in seeds
        ]
        kg_paths = {
            _resolve_path(str(row["kg_path"]), args.generation_root)
            for row in window_rows
        }
        if len(kg_paths) != 1:
            raise ValueError(
                f"source runs for KG_{window.freeze_year} reference multiple snapshots: "
                f"{sorted(map(str, kg_paths))}"
            )
        kg_path = next(iter(kg_paths))
        print(f"[load] fixed-candidate relation scorer KG_{window.freeze_year}", flush=True)
        knowledge_graph = load_graph(kg_path)
        semantic_graph = knowledge_graph.semantic_view

        for case_id in case_ids:
            case = case_study_by_name(case_id)
            for seed in seeds:
                source_row = source_runs[(window.freeze_year, case_id, seed)]
                source_path = _resolve_path(
                    str(source_row["hypotheses_path"]), args.generation_root
                )
                general_path, scoped_path = _branch_paths(source_path)
                general_payload = json.loads(general_path.read_text(encoding="utf-8"))
                scoped_payload = json.loads(scoped_path.read_text(encoding="utf-8"))
                variants = _legacy_pool_variants(
                    general_payload,
                    scoped_payload,
                    case_study_id=case_id,
                    max_candidates=args.max_candidates,
                    task_scope_fraction=args.task_scope_fraction,
                )
                hybrid_legacy_ids = _candidate_ids(variants[METHOD_HYBRID_LEGACY])
                hybrid_relation = annotate_relation_aware_payload(
                    copy.deepcopy(variants[METHOD_HYBRID_LEGACY]),
                    graph=semantic_graph,
                    case_study_id=case_id,
                    use_path_pairs=case.chain is not None,
                )
                if set(_candidate_ids(hybrid_relation)) != set(hybrid_legacy_ids):
                    raise RuntimeError("relation-aware policy changed the fixed candidate set")
                variants[METHOD_HYBRID_RELATION] = hybrid_relation

                audit_key = f"{window.freeze_year}:{case_id}:seed{seed}"
                source_audit[audit_key] = {
                    "source_hypotheses": str(source_path),
                    "source_hypotheses_sha256": cached_hash(source_path),
                    "general_branch": str(general_path),
                    "general_branch_sha256": cached_hash(general_path),
                    "task_scoped_branch": str(scoped_path),
                    "task_scoped_branch_sha256": cached_hash(scoped_path),
                    "frozen_kg": str(kg_path),
                    "frozen_kg_sha256": cached_hash(kg_path),
                    "candidate_counts": {
                        method: len(payload.get("hypotheses") or ())
                        for method, payload in variants.items()
                    },
                    "hybrid_relation_candidate_set_unchanged": True,
                }
                source_sha256 = cached_hash(source_path)
                general_sha256 = cached_hash(general_path)
                scoped_sha256 = cached_hash(scoped_path)
                kg_sha256 = cached_hash(kg_path)

                for method, payload in variants.items():
                    pool_mode = (
                        "general" if method == METHOD_GENERAL_LEGACY
                        else "task_scoped" if method == METHOD_SCOPED_LEGACY
                        else "hybrid"
                    )
                    score_family = (
                        "relation_aware"
                        if method == METHOD_HYBRID_RELATION
                        else "legacy"
                    )
                    tagged = _tag_screen_payload(
                        payload,
                        method=method,
                        pool_mode=pool_mode,
                        score_family=score_family,
                        source_hypotheses=source_path,
                        source_hypotheses_sha256=source_sha256,
                        general_path=general_path,
                        general_sha256=general_sha256,
                        scoped_path=scoped_path,
                        scoped_sha256=scoped_sha256,
                        kg_path=kg_path,
                        kg_sha256=kg_sha256,
                    )
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
                        json.dumps(tagged, ensure_ascii=False), encoding="utf-8"
                    )
                    fixed = _enforce_fixed_budget(
                        tagged,
                        output_path=output,
                        target_per_case_study=fixed_budget,
                        case_study_id=case_id,
                    )
                    fixed_meta = (fixed.get("metadata") or {}).get("fixed_budget") or {}
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "case_study_id": case_id,
                            "freeze_year": window.freeze_year,
                            "future_start_year": window.future_start_year,
                            "future_end_year": window.future_end_year,
                            "kg_path": str(kg_path),
                            "historical_claims_path": source_row["historical_claims_path"],
                            "hypotheses_path": str(output.resolve()),
                            "n_hypotheses": len(fixed.get("hypotheses") or ()),
                            "valid_before_padding": fixed_meta.get("valid_before_padding"),
                            "candidate_pool_mode": pool_mode,
                            "static_score_family": score_family,
                        }
                    )
        del semantic_graph
        del knowledge_graph
        gc.collect()

    manifest = {
        "schema_version": "neurodiscovery-fixed-candidate-ablation.v1",
        "purpose": "fixed-candidate development screening; not final generation benchmark",
        "uses_future_outcomes": False,
        "generates_new_hypotheses": False,
        "source_generation_root": str(args.generation_root.resolve()),
        "source_generation_manifest": str(manifest_path.resolve()),
        "source_generation_manifest_sha256": _sha256(manifest_path),
        "snapshot_root": source_manifest.get("snapshot_root"),
        "case_studies": list(case_ids),
        "windows": [window.__dict__ for window in windows],
        "seeds": list(seeds),
        "fixed_budget": fixed_budget,
        "max_candidates": args.max_candidates,
        "task_scope_fraction": args.task_scope_fraction,
        "protect_general_top_k": 0,
        "methods": list(METHODS),
        "source_audit": source_audit,
        "runs": rows,
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
    parser.add_argument(
        "--windows", nargs="+", type=parse_window, default=list(DEFAULT_WINDOWS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--fixed-budget", type=int)
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--task-scope-fraction", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    result = screen(parse_args())
    print(
        json.dumps(
            {"runs": len(result["runs"]), "methods": result["methods"]},
            indent=2,
        )
    )
