"""Rebuild KG priors and rankings without repeating imaging experiments.

This is the fast path after a KG or semantic-grounding update. Hidden internal
and external outcome values are reused as completed experiment results;
only the public KG-derived scores, frozen rankings, benchmark summaries, and
append-only feedback trace are regenerated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.scripts.case_study_candidate_tables import (
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
)
from core.scripts.case_study_closed_loop import run_benchmark
from core.scripts.case_study_closed_loop_specs import TASK_PROTOCOLS, protocol_for
from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    sha256_file as canonical_sha256_file,
    validate_canonical_kg_release,
    write_release_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG = ROOT / "neurooracle/data/full_v2/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
DEFAULT_STATE = ROOT / "neurooracle/data/full_v2/CURRENT_STATE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def closure_matches_release(
    path: Path,
    canonical_release: dict[str, Any],
    *,
    task: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        closure = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if closure.get("status") != "complete" or closure.get("task") != task:
        return False
    recorded = (closure.get("canonical_release") or {}).get("files") or {}
    expected = canonical_release.get("files") or {}
    for name in ("knowledge_graph", "extracted_claims", "current_state"):
        if (recorded.get(name) or {}).get("sha256") != (
            expected.get(name) or {}
        ).get("sha256"):
            return False
    return True


def load_policy_inventory(path: Path) -> tuple[list[str], int]:
    """Return stable method order and require a complete common trial set."""

    methods: list[str] = []
    trials_by_method: dict[str, set[int]] = defaultdict(set)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        method = str(payload.get("method") or "").strip()
        if not method:
            raise ValueError(f"policy line {line_number} lacks method: {path}")
        trial = int(payload.get("trial"))
        if method not in trials_by_method:
            methods.append(method)
        if trial in trials_by_method[method]:
            raise ValueError(f"duplicate policy for {method} trial {trial}: {path}")
        trials_by_method[method].add(trial)
    if not methods:
        raise ValueError(f"policy file is empty: {path}")
    expected = set(range(max(max(values) for values in trials_by_method.values()) + 1))
    incomplete = {
        method: sorted(expected - trials)
        for method, trials in trials_by_method.items()
        if trials != expected
    }
    if incomplete:
        raise ValueError(f"policy trials are incomplete: {incomplete}")
    return methods, len(expected)


def load_frozen_ranking_inventory(manifest: dict[str, Any]) -> tuple[list[str], int]:
    """Inventory non-KG baselines in an older frozen-ranking bundle."""

    methods: list[str] = []
    trials_by_method: dict[str, set[int]] = defaultdict(set)
    for item in (manifest.get("orders") or {}).get("records") or []:
        method = str(item.get("method") or "").strip()
        if method in {"", "random_walk", "neurodiscovery"}:
            continue
        trial = int(item.get("trial", -1))
        if trial < 0:
            raise ValueError(f"invalid frozen trial for {method}: {trial}")
        if method not in trials_by_method:
            methods.append(method)
        if trial in trials_by_method[method]:
            raise ValueError(f"duplicate frozen ranking for {method} trial {trial}")
        trials_by_method[method].add(trial)
    if not methods:
        raise ValueError("frozen ranking manifest contains no reusable baselines")
    common_trials = min(len(values) for values in trials_by_method.values())
    expected = set(range(common_trials))
    incomplete = {
        method: sorted(expected - trials)
        for method, trials in trials_by_method.items()
        if not expected.issubset(trials)
    }
    if incomplete:
        raise ValueError(f"frozen baseline trials are incomplete: {incomplete}")
    return methods, common_trials


def rescore_task(
    source_root: Path,
    task: str,
    kg: Path,
    *,
    output_root: Path | None = None,
    canonical_release: dict[str, Any] | None = None,
    case_study_statistics: dict[str, Any] | None = None,
    policy_run_name: str | None = None,
    seed: int = 20260806,
) -> Path:
    output_root = source_root if output_root is None else output_root
    canonical_release = canonical_release or {
        "taxonomy_version": "unrecorded",
        "files": {
            "knowledge_graph": {
                "path": str(kg.resolve()),
                "sha256": canonical_sha256_file(kg),
            }
        },
        "general_statistics": None,
        "quality": None,
        "reaudit": None,
    }
    case_study_statistics = case_study_statistics or {}
    source_task_dir = source_root / task
    output_task_dir = output_root / task
    source_closure_path = source_task_dir / "closure_manifest.json"
    if not source_closure_path.is_file() and task == "biomarker_discovery":
        source_closure_path = (
            source_task_dir / "tables" / "biomarker_closure_manifest.json"
        )
    source_closure = load_json(source_closure_path)
    closure = dict(source_closure)
    if closure.get("schema_version") == "case-study-table-bundle.v1":
        table = dict(closure)
    else:
        table = dict(closure.get("table_manifest") or {})
    benchmark = dict(closure.get("benchmark_manifest") or {})
    source_ranking_manifest = dict(
        closure.get("source_ranking_manifest")
        or table.get("source_ranking_manifest")
        or {}
    )
    if table.get("status") != "complete" or (
        benchmark.get("status") != "complete" and not source_ranking_manifest
    ):
        raise ValueError(f"task is not a completed generic closure: {task}")

    files = table.get("files") or {}
    public_old = pd.read_csv(files["public_candidates"]["path"], low_memory=False)
    internal = pd.read_csv(files["internal_outcomes"]["path"], low_memory=False)
    external = pd.read_csv(files["external_outcomes"]["path"], low_memory=False)
    factor_fields = tuple(str(item) for item in table.get("factor_fields") or [])
    if task == "biomarker_discovery":
        factor_fields = tuple(protocol_for(task).factor_fields)
    semantic_fields = tuple(
        str(item) for item in (table.get("kg_scoring") or {}).get("semantic_fields") or []
    )
    if not semantic_fields and task == "biomarker_discovery":
        semantic_fields = tuple(field for field in factor_fields if field != "roi_index")
    if not factor_fields or not semantic_fields:
        raise ValueError(f"task lacks factor or semantic fields: {task}")
    if task == "biomarker_discovery":
        metadata_fields = (
            "candidate_id",
            "modality",
            "source",
            "roi_id",
            "roi_name",
            "anatomy_full",
            "network",
        )
        columns = list(
            dict.fromkeys(
                [*factor_fields, *[field for field in metadata_fields if field in public_old]]
            )
        )
        factors = public_old.loc[:, columns].copy()
        factors["candidate_id"] = factors["candidate_id"].astype(str)
        if factors["candidate_id"].duplicated().any():
            raise ValueError("biomarker candidate IDs are not unique")
        if factors.duplicated(list(factor_fields)).any():
            raise ValueError("biomarker ROI identity remains ambiguous")
    else:
        factors = attach_candidate_ids(
            public_old.loc[:, list(factor_fields)].copy(),
            task=task,
            identity_fields=factor_fields,
        )
        if "candidate_id" in public_old and not factors["candidate_id"].equals(
            public_old["candidate_id"].astype(str)
        ):
            raise ValueError(f"candidate IDs changed during rescoring: {task}")
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=semantic_fields,
        case_study_id=task,
        kg_path=kg.resolve(),
        seed=seed,
    )

    provenance = dict(table.get("provenance") or {})
    provenance["rescoring"] = {
        "created_at": utc_now(),
        "reason": "canonical KG v2 membership and evidence update",
        "reused_hidden_outcomes": True,
        "models_refit": False,
        "source_closure_manifest": str(source_closure_path.resolve()),
        "source_closure_manifest_sha256": (
            hashlib.sha256(source_closure_path.read_bytes()).hexdigest()
        ),
        "baseline_policies_reused_only_when_outcome_blind_and_kg_blind": bool(
            policy_run_name
        ),
    }
    kg_audit["canonical_release"] = {
        "taxonomy_version": canonical_release["taxonomy_version"],
        "files": canonical_release["files"],
        "general_statistics": canonical_release["general_statistics"],
        "case_study_statistics": case_study_statistics,
        "quality": canonical_release["quality"],
        "reaudit": canonical_release["reaudit"],
    }
    tables_dir = output_task_dir / "tables"
    table_manifest = export_table_bundle(
        task=task,
        public=public,
        internal=internal,
        external=external,
        factor_fields=factor_fields,
        output_dir=tables_dir,
        provenance=provenance,
        kg_audit=kg_audit,
    )

    protocol = protocol_for(task)
    registered_budgets = sorted(
        value for value in protocol.budgets if value <= len(public)
    )
    budgets = sorted(set(registered_budgets) | {len(public)})
    policy_path: Path | None = None
    precomputed_manifest_path: Path | None = None
    if policy_run_name:
        policy_path = source_task_dir / policy_run_name / "search_policies.jsonl"
        if policy_path.is_file():
            policy_methods, trials = load_policy_inventory(policy_path)
            methods = ["random_walk", *policy_methods, "neurodiscovery"]
        elif source_ranking_manifest:
            policy_path = None
            policy_methods, trials = load_frozen_ranking_inventory(
                source_ranking_manifest
            )
            methods = ["random_walk", *policy_methods, "neurodiscovery"]
            precomputed_manifest_path = (
                output_task_dir / "source_frozen_baseline_rankings.json"
            )
            precomputed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            precomputed_manifest_path.write_text(
                json.dumps(source_ranking_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            raise FileNotFoundError(policy_path)
    else:
        methods = [str(item) for item in benchmark.get("methods") or []]
        trial_counts = benchmark.get("method_trial_counts") or {}
        trials = min(
            (int(value) for value in trial_counts.values()),
            default=protocol.minimum_trials,
        )
    benchmark_args = argparse.Namespace(
        task=task,
        public_candidates=tables_dir / "public_candidates.csv",
        internal_outcomes=tables_dir / "internal_outcomes.csv",
        external_outcomes=tables_dir / "external_outcomes.csv",
        score_components=None,
        score_components_manifest=None,
        search_policies=policy_path,
        precomputed_rankings_manifest=precomputed_manifest_path,
        policy_schema="case-study-search-policy.v1",
        output_dir=output_task_dir / "benchmark",
        factor_fields=list(factor_fields),
        methods=methods,
        trials=max(protocol.minimum_trials, trials),
        seed=seed,
        batch_size=min(64, max(8, len(public) // 10)),
        warmup_batches=1,
        sampling_temperature=0.01,
        max_feedback_rounds=32 if len(public) >= 100_000 else 128,
        feedback_horizon=(
            max(registered_budgets) if registered_budgets else len(public)
        ),
        budgets=budgets,
        recall_targets=list(protocol.recall_targets),
    )
    benchmark_manifest = run_benchmark(benchmark_args)
    closure = {
        "schema_version": "case-study-kg-refresh-closure.v1",
        "created_at": utc_now(),
        "status": "complete",
        "task": task,
        "source_closure_manifest": str(source_closure_path.resolve()),
        "canonical_release": kg_audit["canonical_release"],
        "table_manifest": table_manifest,
        "benchmark_manifest": benchmark_manifest,
        "protocol": asdict(protocol),
        "reused_search_policies": (
            {
                "path": str(policy_path.resolve()),
                "methods": methods[1:-1],
                "trials": trials,
            }
            if policy_path is not None
            else None
        ),
        "reused_frozen_baseline_rankings": (
            {
                "path": str(precomputed_manifest_path.resolve()),
                "methods": methods[1:-1],
                "trials": trials,
                "neurodiscovery_ranking_reused": False,
            }
            if precomputed_manifest_path is not None
            else None
        ),
    }
    closure_path = output_task_dir / "closure_manifest.json"
    temporary = closure_path.with_suffix(".json.tmp")
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(closure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(closure_path)
    return closure_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--allow-relocated-canonical-artifacts",
        action="store_true",
        help=(
            "Accept an immutable byte-identical KG/claims copy at paths different "
            "from CURRENT_STATE.json; expected hashes and recorded sizes remain required."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_PROTOCOLS),
        required=True,
    )
    parser.add_argument(
        "--policy-run-name",
        default=None,
        help=(
            "Source task subdirectory containing outcome-blind search_policies.jsonl. "
            "When set, those frozen non-KG baselines are evaluated beside the refreshed "
            "random-walk and NeuroDiscovery rankings."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse only complete task closures sealed to the same canonical release.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("--output-root must differ from --source-root")
    output_root.mkdir(parents=True, exist_ok=True)
    canonical_release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        expected_sha256=CURRENT_CANONICAL_SHA256,
        allow_relocated_artifacts=args.allow_relocated_canonical_artifacts,
    )
    write_release_manifest(output_root / "canonical_kg_release.json", canonical_release)
    state = load_json(args.current_state)
    case_study_stats = (
        (state.get("formal_kg_statistics") or {}).get("case_studies") or {}
    )

    written: dict[str, str] = {}
    failures: dict[str, str] = {}
    resumed: list[str] = []
    for task in args.tasks:
        existing = output_root / task / "closure_manifest.json"
        if args.resume_existing and closure_matches_release(
            existing, canonical_release, task=task
        ):
            written[task] = str(existing)
            resumed.append(task)
            print(f"[{task}] resumed -> {existing}", flush=True)
            continue
        try:
            path = rescore_task(
                source_root,
                task,
                args.kg,
                output_root=output_root,
                canonical_release=canonical_release,
                case_study_statistics=dict(case_study_stats.get(task) or {}),
                policy_run_name=args.policy_run_name,
                seed=args.seed,
            )
        except Exception as exc:
            failures[task] = f"{type(exc).__name__}: {exc}"
        else:
            written[task] = str(path)
            print(f"[{task}] complete -> {path}", flush=True)

    manifest = {
        "schema_version": "case-study-kg-refresh-matrix.v1",
        "created_at": utc_now(),
        "status": "complete" if not failures else "incomplete",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tasks": list(args.tasks),
        "seed": args.seed,
        "policy_run_name": args.policy_run_name,
        "canonical_release": canonical_release,
        "closures": written,
        "failures": failures,
        "resumed_existing": resumed,
    }
    manifest_path = output_root / "rerun_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
