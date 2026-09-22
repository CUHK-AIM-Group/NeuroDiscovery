"""Run temporal hindcasting for one formal Case Study ID."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurooracle.scripts.build_temporal_kg_snapshot import (
    build_snapshot,
    snapshot_matches_input,
)
from neurooracle.scripts.case_study_hindcasting_eval import evaluate
from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.validation_protocols import HINDCASTING


@dataclass(frozen=True)
class Window:
    freeze_year: int
    future_start_year: int
    future_end_year: int

    @property
    def label(self) -> str:
        return f"kg{self.freeze_year}_to_{self.future_start_year}_{self.future_end_year}"


DEFAULT_WINDOWS = (
    Window(2016, 2017, 2021),
    Window(2017, 2018, 2022),
    Window(2018, 2019, 2023),
    Window(2019, 2020, 2024),
    Window(2020, 2021, 2025),
)


def parse_window(raw: str) -> Window:
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("window must be freeze:start:end")
    freeze, start, end = (int(part) for part in parts)
    if not (freeze < start <= end):
        raise argparse.ArgumentTypeError("window must satisfy freeze < start <= end")
    return Window(freeze, start, end)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _hypothesis_payload_semantic_key(hypothesis: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the serialized equivalent of hypothesis_cli's semantic path key."""

    nodes = [str(hypothesis.get("source_id") or "")]
    for link in hypothesis.get("path") or []:
        from_id = str(link.get("from_id") or "")
        to_id = str(link.get("to_id") or "")
        if from_id and nodes[-1] != from_id:
            nodes.append(from_id)
        if to_id:
            nodes.append(to_id)
    if len(nodes) == 1 and hypothesis.get("target_id"):
        nodes.append(str(hypothesis["target_id"]))
    if not any(nodes):
        nodes = [str(hypothesis.get("id") or "")]
    return str(hypothesis.get("hypothesis_type") or ""), tuple(nodes)


def _enforce_fixed_budget(
    payload: dict[str, Any],
    *,
    output_path: Path,
    target_per_case_study: int | None,
    case_study_id: str,
) -> dict[str, Any]:
    """Keep the ranking pool and pad only when it cannot fill the target K."""

    if target_per_case_study is None:
        return payload
    valid_hypotheses = [
        hypothesis
        for hypothesis in (payload.get("hypotheses") or [])
        if not (
            hypothesis.get("hypothesis_type") == "generation_failure"
            or (hypothesis.get("metadata") or {}).get("generation_failure") is True
        )
    ]
    hypotheses: list[dict[str, Any]] = []
    seen_semantic_keys: set[tuple[str, tuple[str, ...]]] = set()
    duplicate_count = 0
    for hypothesis in valid_hypotheses:
        key = _hypothesis_payload_semantic_key(hypothesis)
        if key in seen_semantic_keys:
            duplicate_count += 1
            continue
        seen_semantic_keys.add(key)
        hypotheses.append(hypothesis)
    valid_count = len(hypotheses)
    for hypothesis in hypotheses:
        metadata = dict(hypothesis.get("metadata") or {})
        metadata["case_study_id"] = case_study_id
        hypothesis["metadata"] = metadata
    for rank in range(valid_count + 1, target_per_case_study + 1):
        hypotheses.append(
            {
                "id": f"NEURODISCOVERY:{case_study_id}:INVALID:{rank:04d}",
                "hypothesis_type": "generation_failure",
                "source_id": "",
                "source_name": "",
                "target_id": "",
                "target_name": "",
                "path": [],
                "confidence_score": 0.0,
                "novelty_score": 0.0,
                "evidence_score": 0.0,
                "testability_score": 0.0,
                "composite_score": 0.0,
                "supporting_claims": [],
                "explanation": "No valid candidate was available for this fixed-budget slot.",
                "metadata": {
                    "case_study_id": case_study_id,
                    "generator_method": "neurodiscovery",
                    "generation_failure": True,
                },
            }
        )
    result = dict(payload)
    result["hypotheses"] = hypotheses
    metadata = dict(result.get("metadata") or {})
    metadata["fixed_budget"] = {
        "case_study_id": case_study_id,
        "requested": target_per_case_study,
        "requested_evaluation_prefix": target_per_case_study,
        "valid_before_semantic_dedup": len(valid_hypotheses),
        "semantic_duplicates_removed": duplicate_count,
        "valid_before_padding": valid_count,
        "valid_in_evaluation_prefix": min(valid_count, target_per_case_study),
        "failure_slots_in_evaluation_prefix": max(
            0, target_per_case_study - valid_count
        ),
        "candidate_tail_retained_beyond_budget": max(
            0, valid_count - target_per_case_study
        ),
        "ranked_pool_after_padding": len(hypotheses),
        "policy": (
            "preserve the complete ranked candidate pool; execute only the ranked "
            "prefix at each requested K; append zero-credit failures only when valid "
            "candidates are fewer than the requested evaluation prefix"
        ),
    }
    result["metadata"] = metadata
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _is_complete_fixed_budget_payload(
    payload: Any,
    *,
    case_study_id: str,
    target_per_case_study: int | None,
) -> bool:
    """Return whether an existing output is safe to reuse after interruption."""

    if target_per_case_study is None or not isinstance(payload, dict):
        return False
    hypotheses = payload.get("hypotheses")
    metadata = payload.get("metadata")
    if not isinstance(hypotheses, list) or not isinstance(metadata, dict):
        return False
    fixed = metadata.get("fixed_budget")
    if not isinstance(fixed, dict):
        return False
    try:
        requested = int(fixed["requested"])
        requested_prefix = int(fixed["requested_evaluation_prefix"])
        failure_slots = int(fixed["failure_slots_in_evaluation_prefix"])
        ranked_pool = int(fixed["ranked_pool_after_padding"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        str(fixed.get("case_study_id") or "") == case_study_id
        and requested == target_per_case_study
        and requested_prefix == target_per_case_study
        and len(hypotheses) >= target_per_case_study
        and ranked_pool == len(hypotheses)
        and 0 <= failure_slots <= target_per_case_study
        and metadata.get("uses_future_outcomes") is not True
    )


def _generation_command(
    *,
    case_study_id: str,
    kg_path: Path,
    output_dir: Path,
    target_per_case_study: int | None,
    generation_pool_size: int | None = None,
    seed: int,
) -> list[str]:
    """Build a literature-compatible generation command for hindcasting."""

    case = case_study_by_name(case_study_id)
    base = [
        sys.executable,
        "-m",
        "neurooracle.src.hypothesis_cli",
        "--graph",
        str(kg_path),
        "--generation-seed",
        str(seed),
    ]
    if case_study_id != "case1_transdiagnostic":
        command = [
            *base,
            "case-study",
            case_study_id,
            "--output-dir",
            str(output_dir),
            "--stages",
            "batch",
        ]
        if generation_pool_size is not None:
            command.extend(["--target-per-task", str(generation_pool_size)])
        return command

    # The CS1 primary experiment ranks synthetic disease x ROI x feature tests.
    # Those synthetic IDs cannot occur in future literature claims. Hindcasting
    # therefore uses the same registered task over frozen KG entities, while the
    # primary dataset experiment keeps its exhaustive candidate-space generator.
    batch = case.stage_params.batch
    target = generation_pool_size or target_per_case_study or batch.target_per_task
    command = [
        *base,
        "batch",
        "--output",
        str(output_dir / "hypotheses_raw.json"),
        "--max-hops",
        str(batch.max_hops),
        "--min-hops",
        str(batch.min_hops),
        "--metapath-min-domains",
        str(batch.metapath_min_domains),
        "--max-paths",
        str(batch.max_paths),
        "--max-seeds",
        str(batch.max_seeds),
        "--tasks",
        str(case.task.name),
        "--chains",
        "",
        "--claim-scope",
        case_study_id,
        "--target-per-task",
        str(target),
        "--max-retries",
        str(batch.max_retries),
        "--retry-scale",
        str(batch.retry_scale),
    ]
    if not batch.prefer_longer_paths:
        command.append("--no-prefer-longer-paths")
    return command


def generate_hypotheses(
    *,
    case_study_id: str,
    kg_path: Path,
    output_dir: Path,
    target_per_case_study: int | None,
    generation_pool_size: int | None,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    output_path = output_dir / "hypotheses_raw.json"
    if output_path.is_file() and not force:
        try:
            existing = load_json(output_path)
        except (OSError, json.JSONDecodeError):
            existing = None
        if _is_complete_fixed_budget_payload(
            existing,
            case_study_id=case_study_id,
            target_per_case_study=target_per_case_study,
        ):
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _generation_command(
        case_study_id=case_study_id,
        kg_path=kg_path,
        output_dir=output_dir,
        target_per_case_study=target_per_case_study,
        generation_pool_size=generation_pool_size,
        seed=seed,
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    subprocess.run(command, check=True, env=environment)
    return _enforce_fixed_budget(
        load_json(output_path),
        output_path=output_path,
        target_per_case_study=target_per_case_study,
        case_study_id=case_study_id,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_summary(
    output_root: Path,
    case_study_id: str,
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Hindcasting: {case_study_id}",
        "",
        "Future evidence is selected by each claim's formal Case Study membership.",
        "",
        "| Freeze | Future | Hypotheses | Top-100 primary hits | Top-100 random primary | Top-100 endpoint hits | Top-1000 primary hits | Top-1000 endpoint hits |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        top100 = row["topk"].get("100", {}).get("observed", {})
        rand100 = row["topk"].get("100", {}).get("random_same_hypothesis_pool", {})
        top1000 = row["topk"].get("1000", {}).get("observed", {})
        lines.append(
            "| {freeze} | {future} | {n} | {p100} | {r100} | {e100} | {p1000} | {e1000} |".format(
                freeze=row["freeze_year"],
                future=f"{row['future_start_year']}-{row['future_end_year']}",
                n=row["n_hypotheses"],
                p100=top100.get("primary_hits", "NA"),
                r100=_fmt(rand100.get("mean_primary_hits")),
                e100=top100.get("endpoint_hits", "NA"),
                p1000=top1000.get("primary_hits", "NA"),
                e1000=top1000.get("endpoint_hits", "NA"),
            )
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "case_study_id": case_study_id,
                "validation_protocol": HINDCASTING.name,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_study_id", help="One of the 17 formal Case Study IDs.")
    parser.add_argument("--input-dir", type=Path, default=Path("neurooracle/data/full_v2"))
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("neurooracle/data/experiments/hindcasting/snapshots_full_v2"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--target-per-case-study", type=int, default=None)
    parser.add_argument(
        "--generation-pool-size",
        type=int,
        default=None,
        help="Generate at least this many ranked candidates before evaluating Top-K.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--method", default="neurodiscovery")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-snapshot", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    args = parser.parse_args()

    if args.generation_pool_size is not None and args.generation_pool_size < 1:
        parser.error("--generation-pool-size must be positive")
    if (
        args.generation_pool_size is not None
        and args.target_per_case_study is not None
        and args.generation_pool_size <= args.target_per_case_study
    ):
        parser.error(
            "--generation-pool-size must exceed --target-per-case-study "
            "so same-pool random ranking remains identifiable"
        )

    case = case_study_by_name(args.case_study_id)
    if not HINDCASTING.supports(case.name):
        parser.error(f"hindcasting does not support {case.name!r}")
    output_root = args.output_root or Path(
        "neurooracle/data/experiments/hindcasting"
    ) / case.name
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for window in args.windows:
        snapshot_dir = args.snapshot_root / f"kg_{window.freeze_year}"
        manifest_path = snapshot_dir / "manifest.json"
        reusable = False
        if manifest_path.is_file() and not (args.force or args.force_snapshot):
            reusable = snapshot_matches_input(
                load_json(manifest_path),
                args.input_dir,
                window.freeze_year,
            )
        if not reusable:
            print(f"[snapshot] building KG_{window.freeze_year}", flush=True)
            build_snapshot(args.input_dir, snapshot_dir, window.freeze_year)
        else:
            print(f"[snapshot] using KG_{window.freeze_year}", flush=True)

        run_dir = output_root / window.label
        print(f"[generate] {case.name} KG_{window.freeze_year}", flush=True)
        generate_hypotheses(
            case_study_id=case.name,
            kg_path=snapshot_dir / "knowledge_graph.json",
            output_dir=run_dir,
            target_per_case_study=args.target_per_case_study,
            generation_pool_size=args.generation_pool_size,
            seed=args.seed,
            force=args.force or args.force_generation,
        )
        hypotheses_path = run_dir / "hypotheses_raw.json"
        if args.generate_only:
            rows.append(
                {
                    "case_study_id": case.name,
                    "validation_protocol": HINDCASTING.name,
                    "freeze_year": window.freeze_year,
                    "future_start_year": window.future_start_year,
                    "future_end_year": window.future_end_year,
                    "hypotheses_path": str(hypotheses_path),
                    "generation_only": True,
                }
            )
            continue

        metrics_dir = run_dir / "hindcasting"
        metrics_path = metrics_dir / "metrics.json"
        if metrics_path.is_file() and not (args.force or args.force_evaluation):
            metrics = load_json(metrics_path)
        else:
            metrics = evaluate(
                kg_path=snapshot_dir / "knowledge_graph.json",
                hypotheses_path=hypotheses_path,
                future_claims_path=args.input_dir / "extracted_claims.jsonl",
                output_dir=metrics_dir,
                freeze_year=window.freeze_year,
                future_start_year=window.future_start_year,
                future_end_year=window.future_end_year,
                top_ks=args.top_k,
                random_trials=args.random_trials,
                seed=args.seed,
                case_study_id=case.name,
                method=args.method,
            )
        rows.append(
            {
                "case_study_id": case.name,
                "validation_protocol": HINDCASTING.name,
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
                "run_dir": str(run_dir),
                "hypotheses_path": str(hypotheses_path),
                "n_hypotheses": metrics["n_hypotheses"],
                "future_stats": metrics["future_stats"],
                "topk": metrics["topk"],
                "by_type": metrics["by_type"],
            }
        )

    if args.generate_only:
        (output_root / "generation_summary.json").write_text(
            json.dumps(
                {
                    "case_study_id": case.name,
                    "validation_protocol": HINDCASTING.name,
                    "rows": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        write_summary(output_root, case.name, rows)
    print(
        json.dumps(
            {
                "case_study_id": case.name,
                "validation_protocol": HINDCASTING.name,
                "output_root": str(output_root),
                "n_windows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 17:56 HKT
