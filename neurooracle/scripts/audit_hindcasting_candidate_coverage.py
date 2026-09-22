"""Audit why a frozen hindcasting candidate pool misses future-supported pairs.

This is a post-hoc development diagnostic.  It reads future claims and therefore
must never be imported by hypothesis generation, ranking, or final benchmark
selection.  The audit distinguishes exact endpoint coverage from weaker path-edge,
same-hypothesis, and single-node coverage so generation failures are not confused
with ranking failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from neurooracle.scripts.case_study_hindcasting_eval import (
    _edge_pair,
    _future_indexes,
    _historical_pairs,
    _hyp_endpoint_pair,
    _hyp_path_edges,
    load_future_claim_records,
)
from neurooracle.scripts.run_case_study_hindcasting import Window, parse_window
from neurooracle.scripts.temporal_hindcasting_core import load_kg_index
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_hypotheses_path(raw: str, generation_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path.resolve()
    for candidate in (ROOT / path, generation_root / path, path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _select_runs(
    manifest: dict[str, Any],
    *,
    case_ids: tuple[str, ...],
    windows: tuple[Window, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    allowed_freezes = {window.freeze_year for window in windows}
    selected: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in manifest.get("runs") or ():
        if str(row.get("method") or "") != "neurodiscovery":
            continue
        key = (
            int(row["freeze_year"]),
            str(row["case_study_id"]),
            int(row["seed"]),
        )
        if key[0] not in allowed_freezes or key[1] not in case_ids or key[2] not in seeds:
            continue
        if key in selected:
            raise ValueError(f"duplicate generation run: {key}")
        selected[key] = dict(row)
    missing = [
        (window.freeze_year, case_id, seed)
        for window in windows
        for case_id in case_ids
        for seed in seeds
        if (window.freeze_year, case_id, seed) not in selected
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} generation run(s): {missing[:5]}")
    return selected


def _hypothesis_nodes(hypothesis: dict[str, Any]) -> set[str]:
    metadata = hypothesis.get("metadata") or {}
    values: list[Any] = [hypothesis.get("source_id"), hypothesis.get("target_id")]
    for link in hypothesis.get("path") or ():
        values.extend((link.get("from_id"), link.get("to_id")))
    for key in ("input_entity_ids", "path_node_ids", "mediator_ids"):
        values.extend(metadata.get(key) or ())
    return {str(value) for value in values if str(value or "")}


def _first_rank(index: dict[Any, int], key: Any) -> int | None:
    value = index.get(key)
    return int(value) if value is not None else None


def _candidate_indexes(
    hypotheses: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    endpoint_rank: dict[tuple[str, str], int] = {}
    path_edge_rank: dict[tuple[str, str], int] = {}
    node_rank: dict[str, int] = {}
    node_hypotheses: dict[str, set[int]] = defaultdict(set)
    valid = 0
    for rank, hypothesis in enumerate(hypotheses, start=1):
        if hypothesis.get("hypothesis_type") == "generation_failure":
            continue
        valid += 1
        endpoint = _hyp_endpoint_pair(hypothesis)
        if endpoint:
            endpoint_rank.setdefault(_edge_pair(*endpoint), rank)
        for source, target in _hyp_path_edges(hypothesis):
            path_edge_rank.setdefault(_edge_pair(source, target), rank)
        for node_id in _hypothesis_nodes(hypothesis):
            node_rank.setdefault(node_id, rank)
            node_hypotheses[node_id].add(rank)
    return {
        "endpoint_rank": endpoint_rank,
        "path_edge_rank": path_edge_rank,
        "node_rank": node_rank,
        "node_hypotheses": node_hypotheses,
        "valid_hypotheses": valid,
    }


def _pair_claim_summary(future: dict[str, Any], pair: tuple[str, str]) -> dict[str, Any]:
    claims = sorted(
        future["pair_claims"].get(pair) or (),
        key=lambda row: (int(row.get("year") or 9999), str(row.get("claim_id") or "")),
    )
    first = claims[0] if claims else {}
    return {
        "future_claim_count": len(claims),
        "first_claim_id": first.get("claim_id"),
        "first_pmid": first.get("pmid"),
        "first_doi": first.get("doi"),
        "first_title": first.get("title"),
        "first_predicate": first.get("predicate"),
        "first_raw_text": first.get("raw_text"),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    case_ids = tuple(args.case_study_ids)
    windows = tuple(args.windows)
    seeds = tuple(sorted(set(map(int, args.seeds))))
    for case_id in case_ids:
        case_study_by_name(case_id)
    if len({window.freeze_year for window in windows}) != len(windows):
        raise ValueError("coverage windows must use distinct freeze years")

    generation_manifest_path = args.generation_root / "generation_manifest.json"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    selected = _select_runs(
        generation_manifest,
        case_ids=case_ids,
        windows=windows,
        seeds=seeds,
    )
    future_records = load_future_claim_records(
        args.future_claims,
        min_year=min(window.future_start_year for window in windows),
        max_year=max(window.future_end_year for window in windows),
        case_study_ids=set(case_ids),
    )

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for window in windows:
        graph_path = args.snapshot_root / f"kg_{window.freeze_year}" / "knowledge_graph.json"
        historical_claims_path = graph_path.parent / "extracted_claims.jsonl"
        if not graph_path.is_file() or not historical_claims_path.is_file():
            raise FileNotFoundError(graph_path)
        print(f"[load] KG_{window.freeze_year}", flush=True)
        concepts, edges, names = load_kg_index(graph_path)
        historical_endpoint_atoms: dict[str, set[str]] = {}
        historical_pairs = _historical_pairs(
            edges,
            claims_path=historical_claims_path,
            concepts=concepts,
            endpoint_atoms_out=historical_endpoint_atoms,
        )
        for case_id in case_ids:
            future, future_stats = _future_indexes(
                args.future_claims,
                concepts,
                historical_pairs,
                window.future_start_year,
                window.future_end_year,
                case_study_id=case_id,
                future_records=future_records,
                historical_endpoint_atoms=historical_endpoint_atoms,
            )
            gt_pairs = sorted(future["novel_pair_year"])
            for seed in seeds:
                source_row = selected[(window.freeze_year, case_id, seed)]
                hypotheses_path = _resolve_hypotheses_path(
                    str(source_row["hypotheses_path"]), args.generation_root
                )
                payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
                indexes = _candidate_indexes(payload.get("hypotheses") or ())
                exact_count = 0
                path_count = 0
                cooccurrence_count = 0
                both_nodes_count = 0
                for pair in gt_pairs:
                    left, right = pair
                    endpoint_rank = _first_rank(indexes["endpoint_rank"], pair)
                    path_edge_rank = _first_rank(indexes["path_edge_rank"], pair)
                    shared_ranks = (
                        indexes["node_hypotheses"].get(left, set())
                        & indexes["node_hypotheses"].get(right, set())
                    )
                    cooccurrence_rank = min(shared_ranks) if shared_ranks else None
                    left_rank = _first_rank(indexes["node_rank"], left)
                    right_rank = _first_rank(indexes["node_rank"], right)
                    exact_count += endpoint_rank is not None
                    path_count += path_edge_rank is not None
                    cooccurrence_count += cooccurrence_rank is not None
                    both_nodes_count += left_rank is not None and right_rank is not None
                    pair_rows.append(
                        {
                            "case_study_id": case_id,
                            "freeze_year": window.freeze_year,
                            "future_start_year": window.future_start_year,
                            "future_end_year": window.future_end_year,
                            "seed": seed,
                            "left_id": left,
                            "left_name": names.get(left, left),
                            "right_id": right,
                            "right_name": names.get(right, right),
                            "future_year": future["novel_pair_year"][pair],
                            "exact_endpoint_rank": endpoint_rank,
                            "path_edge_rank": path_edge_rank,
                            "same_hypothesis_rank": cooccurrence_rank,
                            "left_node_rank": left_rank,
                            "right_node_rank": right_rank,
                            "both_nodes_present": left_rank is not None and right_rank is not None,
                            **_pair_claim_summary(future, pair),
                        }
                    )
                total = len(gt_pairs)
                summary_rows.append(
                    {
                        "case_study_id": case_id,
                        "freeze_year": window.freeze_year,
                        "future_start_year": window.future_start_year,
                        "future_end_year": window.future_end_year,
                        "seed": seed,
                        "future_unique_pairs": total,
                        "valid_hypotheses": indexes["valid_hypotheses"],
                        "exact_endpoint_pairs": exact_count,
                        "exact_endpoint_coverage": exact_count / total if total else 0.0,
                        "path_edge_pairs": path_count,
                        "path_edge_coverage": path_count / total if total else 0.0,
                        "same_hypothesis_pairs": cooccurrence_count,
                        "same_hypothesis_coverage": cooccurrence_count / total if total else 0.0,
                        "both_nodes_pairs": both_nodes_count,
                        "both_nodes_coverage": both_nodes_count / total if total else 0.0,
                        "future_evaluable_claims": future_stats.get("future_evaluable_claims", 0),
                    }
                )
            print(
                f"[audit] {case_id} KG_{window.freeze_year}: "
                f"{len(gt_pairs)} future pair(s), {len(seeds)} seed(s)",
                flush=True,
            )
        del concepts, edges, names, historical_pairs, historical_endpoint_atoms

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "coverage_by_pair.csv", pair_rows)
    _write_csv(args.output_root / "coverage_summary.csv", summary_rows)
    manifest = {
        "schema_version": "hindcasting-candidate-coverage-audit.v1",
        "status": "development_diagnostic_only",
        "uses_future_outcomes": True,
        "permitted_for_candidate_generation": False,
        "permitted_for_final_benchmark_selection": False,
        "generation_root": str(args.generation_root.resolve()),
        "generation_manifest": str(generation_manifest_path.resolve()),
        "generation_manifest_sha256": _sha256(generation_manifest_path),
        "snapshot_root": str(args.snapshot_root.resolve()),
        "future_claims": str(args.future_claims.resolve()),
        "case_studies": list(case_ids),
        "windows": [window.__dict__ for window in windows],
        "seeds": list(seeds),
        "runs": len(summary_rows),
        "future_pair_rows": len(pair_rows),
        "outputs": {
            "coverage_by_pair": "coverage_by_pair.csv",
            "coverage_summary": "coverage_summary.csv",
        },
    }
    (args.output_root / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--future-claims", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-study-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--windows", nargs="+", type=parse_window, default=list(DEFAULT_WINDOWS)
    )
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())


# Updated: 2026-08-13 06:26:23 HKT - audit candidate coverage against freeze-valid endpoint atom roles.
