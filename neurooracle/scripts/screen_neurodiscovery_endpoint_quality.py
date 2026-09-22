"""Screen an outcome-blind endpoint-quality prior on fixed candidate pools.

This utility does not generate new hypotheses and never reads future claims. It
re-ranks an existing NeuroDiscovery candidate pool using concept metadata from
the matching frozen KG, producing a compact ablation manifest for the standard
hindcasting evaluator. Results are a development-screening artifact, not a
replacement for final end-to-end generation runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import ijson

from neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates import (
    _apply_candidate_canonical_quality,
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


def _method_label(weight: float) -> str:
    return f"neurodiscovery_quality_w{int(round(weight * 100)):03d}"


def _resolve_hypotheses_path(raw: str, generation_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (ROOT / path, generation_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _candidate_node_ids(payload: dict[str, Any]) -> set[str]:
    node_ids: set[str] = set()
    for row in payload.get("hypotheses") or ():
        metadata = row.get("metadata") or {}
        values: Iterable[Any] = (
            row.get("source_id"),
            *(metadata.get("input_entity_ids") or ()),
            *(metadata.get("path_node_ids") or ()),
            *(metadata.get("mediator_ids") or ()),
            row.get("target_id"),
        )
        node_ids.update(str(value) for value in values if str(value or ""))
    return node_ids


def _load_required_concepts(graph_path: Path, required: set[str]) -> dict[str, Any]:
    concepts: dict[str, Any] = {}
    with graph_path.open("rb") as handle:
        for node_id, node in ijson.kvitems(handle, "concepts"):
            node_id = str(node_id)
            if node_id in required:
                concepts[node_id] = node
                if len(concepts) == len(required):
                    break
    return concepts


def _select_source_runs(
    manifest: dict[str, Any],
    *,
    case_ids: tuple[str, ...],
    windows: tuple[Window, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    selected: dict[tuple[int, str, int], dict[str, Any]] = {}
    allowed_freezes = {window.freeze_year for window in windows}
    for row in manifest.get("runs") or ():
        if str(row.get("method") or "") != "neurodiscovery":
            continue
        seed = int(row["seed"])
        case_id = str(row["case_study_id"])
        freeze_year = int(row["freeze_year"])
        if (
            seed not in seeds
            or case_id not in case_ids
            or freeze_year not in allowed_freezes
        ):
            continue
        key = (freeze_year, case_id, seed)
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
        raise ValueError(f"missing {len(missing)} source generation run(s): {missing[:5]}")
    return selected


def screen(args: argparse.Namespace) -> dict[str, Any]:
    for case_id in args.case_study_ids:
        case_study_by_name(case_id)
    if len({window.freeze_year for window in args.windows}) != len(args.windows):
        raise ValueError("screening windows must use distinct freeze years")
    if any(not 0.0 <= weight <= 1.0 for weight in args.weights):
        raise ValueError("all endpoint-quality weights must be in [0, 1]")

    generation_manifest_path = args.generation_root / "generation_manifest.json"
    source_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    windows = tuple(args.windows)
    case_ids = tuple(args.case_study_ids)
    seeds = tuple(sorted(set(map(int, args.seeds))))
    weights = tuple(sorted(set(map(float, args.weights))))
    source_runs = _select_source_runs(
        source_manifest,
        case_ids=case_ids,
        windows=windows,
        seeds=seeds,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_runs: list[dict[str, Any]] = []
    concept_audit: dict[str, dict[str, int]] = {}
    for window in windows:
        graph_path = (
            args.snapshot_root / f"kg_{window.freeze_year}" / "knowledge_graph.json"
        )
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)

        payloads: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
        required_ids: set[str] = set()
        for case_id in case_ids:
            for seed in seeds:
                source_row = source_runs[(window.freeze_year, case_id, seed)]
                source_path = _resolve_hypotheses_path(
                    str(source_row["hypotheses_path"]),
                    args.generation_root,
                )
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                payloads[(case_id, seed)] = (payload, source_path)
                required_ids.update(_candidate_node_ids(payload))

        print(
            f"[concepts] KG_{window.freeze_year}: "
            f"loading {len(required_ids)} candidate node(s)",
            flush=True,
        )
        concepts = _load_required_concepts(graph_path, required_ids)
        concept_audit[str(window.freeze_year)] = {
            "candidate_node_ids": len(required_ids),
            "frozen_concepts_found": len(concepts),
            "claim_local_or_missing_ids": len(required_ids - set(concepts)),
        }

        for weight in weights:
            method = _method_label(weight)
            for case_id in case_ids:
                for seed in seeds:
                    payload, source_path = payloads[(case_id, seed)]
                    adjusted = _apply_candidate_canonical_quality(
                        concepts,
                        copy.deepcopy(payload),
                        weight=weight,
                    )
                    metadata = dict(adjusted.get("metadata") or {})
                    metadata["fixed_candidate_quality_screen"] = {
                        "schema_version": "neurodiscovery-endpoint-quality-screen.v1",
                        "source_hypotheses": str(source_path),
                        "source_hypotheses_sha256": _sha256(source_path),
                        "freeze_year": window.freeze_year,
                        "weight": weight,
                        "uses_future_outcomes": False,
                        "final_generation_benchmark": False,
                    }
                    adjusted["metadata"] = metadata
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
                        json.dumps(adjusted, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    output_runs.append(
                        {
                            "method": method,
                            "seed": seed,
                            "case_study_id": case_id,
                            "freeze_year": window.freeze_year,
                            "future_start_year": window.future_start_year,
                            "future_end_year": window.future_end_year,
                            "kg_path": str(graph_path.resolve()),
                            "historical_claims_path": str(
                                (graph_path.parent / "extracted_claims.jsonl").resolve()
                            ),
                            "hypotheses_path": str(output.resolve()),
                            "n_hypotheses": len(adjusted.get("hypotheses") or ()),
                            "endpoint_canonical_quality_weight": weight,
                        }
                    )
        del concepts, payloads

    manifest = {
        "schema_version": "neurodiscovery-endpoint-quality-screen.v1",
        "purpose": "fixed-candidate development screening; not final generation benchmark",
        "uses_future_outcomes": False,
        "source_generation_root": str(args.generation_root.resolve()),
        "source_generation_manifest": str(generation_manifest_path.resolve()),
        "source_generation_manifest_sha256": _sha256(generation_manifest_path),
        "snapshot_root": str(args.snapshot_root.resolve()),
        "case_studies": list(case_ids),
        "windows": [window.__dict__ for window in windows],
        "seeds": list(seeds),
        "weights": list(weights),
        "methods": [_method_label(weight) for weight in weights],
        "concept_audit_by_freeze_year": concept_audit,
        "runs": output_runs,
    }
    (args.output_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-study-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--windows", nargs="+", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.10, 0.20])
    return parser.parse_args()


def main() -> None:
    manifest = screen(parse_args())
    print(
        json.dumps(
            {
                "output_root": str(Path(manifest["runs"][0]["hypotheses_path"]).parents[5]),
                "runs": len(manifest["runs"]),
                "methods": manifest["methods"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
