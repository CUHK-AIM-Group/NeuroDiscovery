"""Generate task-wise hypotheses with the official BrainPilot and Biomni agents.

The agents receive the atom-level task definition but no NeuroOracle graph,
future labels, exhaustive results, or NeuroDiscovery scores. Invalid output
still occupies its requested rank and is retained for zero-credit scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE_SCRIPTS = ROOT / "core" / "scripts"
if str(CORE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORE_SCRIPTS))

from case1_native_baseline_experiment import (  # noqa: E402
    BASELINES_ROOT,
    extract_json,
    run_native_batch,
)
from neurooracle.src.atoms import task_by_name  # noqa: E402
from neurooracle.src.case3_subtasks import select_case3_subtasks  # noqa: E402


METHODS = ("brainpilot_native", "biomni_native")
METHOD_LABELS = {"brainpilot_native": "BrainPilot", "biomni_native": "Biomni"}
DEFAULT_OUT_DIR = (
    ROOT / "neurooracle/data/experiments/case3/native_baselines_gpt56sol_high_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[10])
    parser.add_argument("--n-hypotheses", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument(
        "--brainpilot-urls",
        nargs="+",
        default=["http://127.0.0.1:9460/api"],
        help="BrainPilot backends. This task-wise runner uses the first endpoint serially.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def _normalise_entity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def task_prompt(
    method: str,
    task_name: str,
    seed: int,
    start_rank: int,
    batch_size: int,
    previous: list[dict[str, Any]],
) -> str:
    task = task_by_name(task_name)
    end_rank = start_rank + batch_size - 1
    source_atoms = [atom.value for atom in sorted(task.inputs, key=lambda atom: atom.value)]
    if method == "brainpilot_native":
        previous_lines = (
            "- The evaluator withholds prior batches to keep agent delegation compact. "
            "Do not ask for the list; generate a fresh diverse batch. Any cross-batch "
            "duplicate will consume its rank and receive zero credit."
        )
    else:
        previous_lines = "\n".join(
            f"- {item.get('source_atom')}:{item.get('source_entity')} -> "
            f"{item.get('target_atom')}:{item.get('target_entity')}"
            for item in previous[-300:]
            if isinstance(item, dict)
        ) or "- none"
    delivery = (
        "Deliver the final JSON through BrainPilot's result_deliver tool."
        if method == "brainpilot_native"
        else "Put the final JSON inside one <solution>...</solution> tag."
    )
    return f"""Blinded neuroscience autoresearch hypothesis-generation benchmark.

Framework: {METHOD_LABELS[method]}
Seed: {seed}
Task: {task.name}
Task signature: {task.signature}
Task definition: {task.description}
Illustrative example only: {task.example}
Requested ranks: {start_rank}-{end_rank}

Generate exactly {batch_size} ranked, directional and scientifically testable
hypotheses for this task. Each hypothesis is one source entity to one target
entity relation. For a multi-input task, choose the input atom that carries the
most informative proposed relation; this matches the benchmark's union-over-
input-atoms representation.

Allowed source_atom values: {json.dumps(source_atoms)}
Required target_atom value: {json.dumps(task.output.value)}

Rules:
- Use established, specific biomedical entity names, not generic placeholders.
- An imaging_marker must be a concrete measurement or anatomical phenotype,
  such as hippocampal volume, amyloid PET SUVR, or DMN connectivity. Pure
  modality names such as MRI, fMRI, PET, SPECT, or CT are invalid entities.
- A disease, drug, gene target, cognitive task, individual variable, or outcome
  must be specific enough to be tested against a literature claim.
- Do not inspect NeuroClaw files, the NeuroOracle graph, experiment labels,
  future papers, or another method's outputs. Do not retrieve external papers.
- Return each rank exactly once. Avoid duplicate source-target pairs within the
  seed. Invalid, duplicate, missing, and unmappable outputs consume their rank
  and receive zero credit; there is no manual repair.
- relation is a short directional predicate. rationale is one sentence.
- confidence is a number from 0 to 1.

Already generated in earlier batches for this task and seed; do not repeat:
{previous_lines}

Return this JSON object and no explanatory prose:
{{
  "method": "{method}",
  "task": "{task.name}",
  "hypotheses": [
    {{
      "rank": {start_rank},
      "source_atom": "one allowed source_atom",
      "source_entity": "specific source entity",
      "target_atom": "{task.output.value}",
      "target_entity": "specific target entity",
      "relation": "directional predicate",
      "rationale": "one concise sentence",
      "confidence": 0.75
    }}
  ]
}}

{delivery}
"""


def validate_batches(
    method: str,
    task_name: str,
    seed: int,
    batches: list[tuple[int, int, dict[str, Any] | None, str | None]],
) -> pd.DataFrame:
    task = task_by_name(task_name)
    allowed_sources = {atom.value for atom in task.inputs}
    target_atom = task.output.value
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for start_rank, end_rank, payload, batch_error in batches:
        items = payload.get("hypotheses") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []
        by_rank: dict[int, list[Any]] = {}
        for item in items:
            try:
                rank = int(item.get("rank")) if isinstance(item, dict) else -1
            except (TypeError, ValueError):
                rank = -1
            by_rank.setdefault(rank, []).append(item)

        for rank in range(start_rank, end_rank + 1):
            candidates = by_rank.get(rank, [])
            item = candidates[0] if len(candidates) == 1 and isinstance(candidates[0], dict) else None
            errors: list[str] = []
            if batch_error:
                errors.append(batch_error)
            if not candidates:
                errors.append("missing_rank")
            elif len(candidates) > 1:
                errors.append("duplicate_rank")
            elif item is None:
                errors.append("non_object_hypothesis")
            item = item or {}

            source_atom = str(item.get("source_atom") or "").strip()
            source_entity = str(item.get("source_entity") or "").strip()
            observed_target_atom = str(item.get("target_atom") or "").strip()
            target_entity = str(item.get("target_entity") or "").strip()
            relation = str(item.get("relation") or "").strip()
            rationale = str(item.get("rationale") or "").strip()
            if source_atom not in allowed_sources:
                errors.append("invalid_source_atom")
            if observed_target_atom != target_atom:
                errors.append("invalid_target_atom")
            if len(_normalise_entity(source_entity)) < 3:
                errors.append("invalid_source_entity")
            if len(_normalise_entity(target_entity)) < 3:
                errors.append("invalid_target_entity")
            if not relation:
                errors.append("missing_relation")
            if not rationale:
                errors.append("missing_rationale")
            try:
                confidence = float(item.get("confidence"))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                confidence = math.nan
                errors.append("invalid_confidence")

            key = (
                source_atom,
                _normalise_entity(source_entity),
                observed_target_atom,
                _normalise_entity(target_entity),
            )
            if not errors and key in seen:
                errors.append("duplicate_hypothesis")
            if not errors:
                seen.add(key)
            rows.append(
                {
                    "method": method,
                    "task": task_name,
                    "signature": task.signature,
                    "seed": seed,
                    "generated_rank": rank,
                    "source_atom": source_atom,
                    "source_entity": source_entity,
                    "target_atom": observed_target_atom,
                    "target_entity": target_entity,
                    "relation": relation,
                    "rationale": rationale,
                    "confidence": confidence,
                    "native_validation_status": "valid" if not errors else ";".join(dict.fromkeys(errors)),
                    "native_schema_valid": not errors,
                }
            )
    return pd.DataFrame(rows).sort_values("generated_rank", kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.n_hypotheses < 1 or args.batch_size < 1:
        raise ValueError("n_hypotheses and batch_size must be positive")
    if args.n_hypotheses % args.batch_size:
        raise ValueError("n_hypotheses must be divisible by batch_size")
    selected = select_case3_subtasks(args.tasks)
    if args.list_tasks:
        print(json.dumps([task.to_dict() for task in selected], indent=2, ensure_ascii=False))
        return

    secret = os.environ.get("AUTORESEARCH_LOCAL_API_KEY")
    if not args.prepare_only and not secret:
        raise RuntimeError(
            "AUTORESEARCH_LOCAL_API_KEY is required and is passed only to child-process environments"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for method in args.methods:
        for task_entry in selected:
            for seed in args.seeds:
                seed_dir = args.out_dir / method / task_entry.name / f"seed_{seed:02d}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                previous: list[dict[str, Any]] = []
                batch_payloads: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
                for start_rank in range(1, args.n_hypotheses + 1, args.batch_size):
                    end_rank = start_rank + args.batch_size - 1
                    batch_dir = seed_dir / f"batch_{start_rank:04d}_{end_rank:04d}"
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    prompt_path = batch_dir / "prompt.txt"
                    prompt_path.write_text(
                        task_prompt(
                            method,
                            task_entry.name,
                            seed,
                            start_rank,
                            args.batch_size,
                            previous,
                        ),
                        encoding="utf-8",
                    )
                    payload: dict[str, Any] | None = None
                    error: str | None = None
                    if not args.prepare_only:
                        for attempt in range(args.max_retries + 1):
                            try:
                                final_text = run_native_batch(
                                    method,
                                    prompt_path,
                                    batch_dir,
                                    args,
                                    secret or "",
                                )
                                payload = extract_json(final_text)
                                (batch_dir / "parsed.json").write_text(
                                    json.dumps(payload, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                                (batch_dir / "error.txt").unlink(missing_ok=True)
                                items = payload.get("hypotheses") or []
                                if isinstance(items, list):
                                    previous.extend(item for item in items if isinstance(item, dict))
                                error = None
                                break
                            except Exception as exc:
                                error = f"batch_failed:{type(exc).__name__}"
                                (batch_dir / "error.txt").write_text(
                                    f"attempt={attempt + 1}/{args.max_retries + 1}\n{exc}",
                                    encoding="utf-8",
                                )
                                if attempt < args.max_retries:
                                    time.sleep(args.retry_wait_seconds * (attempt + 1))
                    batch_payloads.append((start_rank, end_rank, payload, error))
                    print(
                        f"{method} task={task_entry.name} seed={seed} "
                        f"batch={start_rank}-{end_rank} status={error or 'ok'}",
                        flush=True,
                    )
                if args.prepare_only:
                    continue
                validated = validate_batches(
                    method,
                    task_entry.name,
                    seed,
                    batch_payloads,
                )
                validated.to_csv(seed_dir / "native_hypotheses.csv", index=False)
                all_frames.append(validated)
                run_rows.append(
                    {
                        "method": method,
                        "task": task_entry.name,
                        "seed": seed,
                        "requested": args.n_hypotheses,
                        "schema_valid": int(validated["native_schema_valid"].sum()),
                        "schema_valid_rate": float(validated["native_schema_valid"].mean()),
                    }
                )

    manifest = {
        "out_dir": str(args.out_dir),
        "tasks": [task.name for task in selected],
        "methods": args.methods,
        "seeds": args.seeds,
        "n_hypotheses_per_task_seed": args.n_hypotheses,
        "batch_size": args.batch_size,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "max_retries": args.max_retries,
        "official_code": {
            "brainpilot": str(BASELINES_ROOT / "BrainPilot"),
            "biomni": str(BASELINES_ROOT / "Biomni"),
        },
        "failure_policy": (
            "Invalid, duplicate, missing, and unmappable outputs consume their experiment slot "
            "and receive zero credit; no manual repair."
        ),
        "credential_policy": "API key is supplied only through child-process environments.",
        "temporal_caveat": (
            "Native agents use a current pretrained model and therefore are workflow baselines, "
            "not historically frozen knowledge controls."
        ),
    }
    (args.out_dir / "native_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.prepare_only:
        print(args.out_dir)
        return
    persisted_frames = [
        pd.read_csv(path, keep_default_na=False)
        for path in args.out_dir.glob("*/*/seed_*/native_hypotheses.csv")
    ]
    combined = pd.concat(persisted_frames, ignore_index=True) if persisted_frames else pd.DataFrame()
    combined.to_csv(args.out_dir / "native_hypotheses_all.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    if not combined.empty:
        for (method, task_name, seed), sub in combined.groupby(
            ["method", "task", "seed"], sort=False
        ):
            summary_rows.append(
                {
                    "method": method,
                    "task": task_name,
                    "seed": int(seed),
                    "requested": len(sub),
                    "schema_valid": int(sub["native_schema_valid"].astype(str).str.casefold().eq("true").sum()),
                    "schema_valid_rate": float(
                        sub["native_schema_valid"].astype(str).str.casefold().eq("true").mean()
                    ),
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "native_generation_summary.csv", index=False)
    print(args.out_dir)


if __name__ == "__main__":
    main()
