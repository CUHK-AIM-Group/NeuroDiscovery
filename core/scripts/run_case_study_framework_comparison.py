"""Run the six native autoresearch baselines through DeepSeek V4 Pro."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from core.scripts.case_study_closed_loop import run_benchmark
from core.scripts.case_study_closed_loop_specs import (
    EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES,
    EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES,
    TASK_PROTOCOLS,
    protocol_for,
)
from core.scripts.case_study_framework_runtime import (
    BASELINES_ROOT,
    METHODS,
    NATIVE_METHODS,
    OFFICIAL_METHODS,
    REPOSITORIES,
    _ensure_directory_with_retry,
    _write_text_with_retry,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v1")
DEFAULT_TASKS = tuple(task for task in TASK_PROTOCOLS if task != "biomarker_discovery")
POLICY_SCHEMA = "case-study-search-policy.v1"
OUTPUT_NAME = "framework_comparison_deepseek_v4_pro_high"
HANDLER = ROOT / "skills/autoresearch-framework-benchmark/handler.py"
DEFAULT_N_ANCHORS = 80
DEFAULT_NATIVE_BATCH_SIZE = 20
DEFAULT_MAX_WORKERS = 4
DEFAULT_PROCESS_RETRIES = 1
OPEN_COSCIENTIST_COMPUTE_CONTROLS = {
    "debate_turns": 3,
    "max_iterations": 0,
    "max_output_tokens": 3000,
    "max_concurrent_llm_calls_per_job": 1,
    "overgeneration_factor": 1.0,
    "tool_generation_enabled": False,
    "debate_cohorts_enabled": True,
}

IMPORT_PROBES = {
    "ai_scientist_v2": (
        "from ai_scientist.llm import create_client; "
        "from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea"
    ),
    "open_coscientist": (
        "import sys; sys.path.insert(0, 'src'); "
        "from open_coscientist import HypothesisGenerator"
    ),
    "sciagents": "import autogen",
    "virtual_lab": (
        "import sys; sys.path.insert(0, 'src'); "
        "from virtual_lab import Agent, run_meeting"
    ),
}

TASK_OBJECTIVES = {
    "biomarker_discovery": (
        "Prioritize disease-specific atlas ROI and imaging-feature combinations "
        "that are plausible, measurable biomarkers and likely to reproduce in "
        "held-out TCP subjects."
    ),
    "differential_diagnosis": (
        "Prioritize imaging representations and classifiers that can distinguish "
        "registered psychiatric diagnostic contrasts and transfer across cohorts."
    ),
    "disease_subtyping": (
        "Prioritize reproducible imaging-based subtype solutions that are stable "
        "under resampling and clinically distinct in held-out data."
    ),
    "connectome_behavior": (
        "Prioritize connectome representations and predictive models for robust "
        "out-of-sample cognitive association."
    ),
    "brain_age": (
        "Prioritize atlas, feature, and model configurations that predict age and "
        "generalize to independent healthy-control cohorts."
    ),
    "progression_prediction": (
        "Prioritize baseline marker sets and models for predicting registered "
        "clinical progression outcomes at fixed horizons."
    ),
    "prognosis": (
        "Prioritize baseline marker sets and survival models for registered "
        "time-to-event clinical endpoints."
    ),
    "imaging_genetics": (
        "Prioritize gene or pathway scores, imaging phenotypes, and association "
        "models likely to replicate in held-out genetic data."
    ),
}

RULE_FIELDS_BY_TASK = {
    "biomarker_discovery": (
        "disease",
        "atlas",
        "anatomy",
        "feature_family",
        "feature",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secret(base_url: str | None = None) -> str:
    for name in (
        "SUB2API_OPENAI_API_KEY",
        "CASE_STUDY_LOCAL_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if base_url and any(
        marker in base_url.casefold()
        for marker in ("127.0.0.1", "localhost", "[::1]")
    ):
        return "neuroclaw-local-router"
    raise RuntimeError("SUB2API_OPENAI_API_KEY is required")


def responses_base_url(value: str) -> str:
    return value.rstrip("/")


def chat_base_url(value: str) -> str:
    base = responses_base_url(value)
    return base if base.endswith("/v1") else f"{base}/v1"


def probe_sub2api(base_url: str, model: str, reasoning_effort: str) -> dict[str, Any]:
    secret = _secret(base_url)
    base = responses_base_url(base_url)
    endpoint = f"{base}/responses"
    body = json.dumps(
        {
            "model": model,
            "input": "Reply with exactly OK",
            "reasoning": {"effort": reasoning_effort},
            "store": False,
            "max_output_tokens": 16,
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        },
    )
    started = time.time()
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "status": payload.get("status"),
        "model": payload.get("model"),
        "duration_seconds": time.time() - started,
        "wire_api": "responses",
        "store": False,
        "endpoint": endpoint,
    }


def _complete_policy_matrix_present(args: argparse.Namespace) -> bool:
    """Return true only when every requested framework trial has a policy."""

    return all(
        (
            args.root
            / task
            / args.output_name
            / method
            / f"trial_{trial:02d}"
            / "search_policy.json"
        ).is_file()
        for task in args.tasks
        for method in args.methods
        for trial in range(args.trials)
    )


def _repository_python(repo: Path) -> Path:
    for candidate in (
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no Python runtime below {repo}")


def preflight(methods: list[str], brainpilot_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method in OFFICIAL_METHODS:
            repo = BASELINES_ROOT / REPOSITORIES[method]
            runtime = _repository_python(repo)
            result = subprocess.run(
                [str(runtime), "-c", IMPORT_PROBES[method]],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            rows.append(
                {
                    "method": method,
                    "status": "ok" if result.returncode == 0 else "failed",
                    "runtime": str(runtime),
                    "repository": str(repo),
                    "error": result.stderr[-1000:] if result.returncode else "",
                }
            )
        elif method == "biomni_native":
            runtime = BASELINES_ROOT / "Biomni/.venv/Scripts/python.exe"
            rows.append(
                {
                    "method": method,
                    "status": "ok" if runtime.is_file() else "failed",
                    "runtime": str(runtime),
                    "retrieval_enabled": False,
                }
            )
        else:
            try:
                with urlopen(brainpilot_url.rstrip("/") + "/health", timeout=10) as response:
                    ok = response.status < 400
            except Exception as exc:
                rows.append(
                    {"method": method, "status": "failed", "error": str(exc)}
                )
            else:
                rows.append(
                    {
                        "method": method,
                        "status": "ok" if ok else "failed",
                        "url": brainpilot_url,
                    }
                )
    return rows


def _registry_text(registry: pd.DataFrame, factor_fields: tuple[str, ...]) -> str:
    lines = []
    for row in registry.sort_values("candidate_id", kind="stable").itertuples(index=False):
        fields = "; ".join(
            f"{field}={getattr(row, field)}" for field in factor_fields
        )
        lines.append(f"- {row.candidate_id} :: {fields}")
    return "\n".join(lines)


def _compact_rule_registry(
    registry: pd.DataFrame,
    *,
    rule_fields: tuple[str, ...],
    trial: int,
    max_anatomies: int = 768,
) -> str:
    """Describe a large registry without exposing outcomes or KG scores."""

    blocks: list[str] = []
    for field in rule_fields:
        values = sorted(registry[field].fillna("").astype(str).unique())
        if field == "anatomy" and len(values) > max_anatomies:
            ranked = sorted(
                values,
                key=lambda value: hashlib.sha256(
                    f"{trial}|{value}".encode("utf-8")
                ).hexdigest(),
            )
            values = sorted(ranked[:max_anatomies])
            suffix = (
                f" (deterministic outcome-blind trial sample; "
                f"{max_anatomies} of {registry[field].nunique()} values)"
            )
        else:
            suffix = ""
        blocks.append(
            f"{field}{suffix}:\n"
            + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(blocks)


def build_task(
    *,
    task: str,
    method: str,
    trial: int,
    n_anchors: int,
    registry: pd.DataFrame,
    registry_path: Path,
) -> dict[str, Any]:
    protocol = protocol_for(task)
    rule_fields = RULE_FIELDS_BY_TASK.get(task)
    if rule_fields:
        registry_lines = _compact_rule_registry(
            registry,
            rule_fields=rule_fields,
            trial=trial,
        )
        delivery_contract = f"""Return exactly {n_anchors} ranked, distinct search
rules. Each rule must have a numeric weight in [-1, 1], a concise rationale, and
a non-empty `when` object using only these fields: {', '.join(rule_fields)}.
Every value must be copied exactly from the compact registry below. A rule may
use one or several fields and applies to every executable candidate matching all
of them. Invalid or duplicate rules consume their slots and are not repaired.

Final machine-readable shape:
{{"rules":[{{"rank":1,"weight":0.9,"when":{{"disease":"exact value","feature_family":"exact value"}},"rationale":"..."}}]}}"""
        policy_mode = "factor_rules"
    else:
        registry_lines = _registry_text(registry, protocol.factor_fields)
        delivery_contract = f"""Rank exactly {n_anchors} distinct executable candidates.
Only verbatim candidate_id values are accepted. Invalid, duplicate, or missing
proposals consume their slots and are not repaired manually."""
        policy_mode = "exact_candidate_ids"
    goal = f"""Blinded executable {task} hypothesis prioritization.

Scientific objective: {TASK_OBJECTIVES[task]}

Candidate unit: {protocol.candidate_unit}
Internal validation contract: {protocol.internal_validation}
External validation contract: {protocol.external_validation}

You may use the native {method} planning, reflection, debate, and critic workflow,
but literature retrieval and all external tools are disabled for the primary
comparison. You are not given and must not infer coefficients, P values, FDR
values, validation labels, NeuroDiscovery scores, or KG support scores.

{delivery_contract}

Only the following public factor values are executable:

{registry_lines}

Prioritize scientific plausibility, statistical appropriateness, testability,
and portfolio diversity."""
    return {
        "case_study_id": task,
        "policy_schema_version": POLICY_SCHEMA,
        "candidate_id_template": f"{task}:<24 hexadecimal characters>",
        "candidate_id_fields": list(protocol.factor_fields),
        "coordinate_constraint": (
            "Use only the factor values and candidate IDs in the public registry."
        ),
        "ontology_path_fields": list(protocol.factor_fields),
        "lead_expertise": "neuroimaging and blinded research prioritization",
        "imaging_expertise": "multimodal neuroimaging representations and validation",
        "biological_expertise": "clinical neuroscience and biological plausibility",
        "method": method,
        "trial": trial,
        "n_anchors": n_anchors,
        "policy_mode": policy_mode,
        "policy_rule_fields": list(rule_fields or ()),
        "public_registry_path": str(registry_path.resolve()),
        "research_goal": goal,
        "native_retrieval_enabled": False,
        "hidden_outcomes_available": False,
        # Keep the per-trial task payload identical to the registered compute
        # controls written to manifest.json.  The official adapter deliberately
        # defaults to one refinement iteration for general use, so the formal
        # supplemental benchmark must carry its zero-iteration budget explicitly.
        "open_coscientist_max_iterations": OPEN_COSCIENTIST_COMPUTE_CONTROLS[
            "max_iterations"
        ],
        "open_coscientist_overgeneration_factor": OPEN_COSCIENTIST_COMPUTE_CONTROLS[
            "overgeneration_factor"
        ],
        "open_coscientist_tool_generation_enabled": OPEN_COSCIENTIST_COMPUTE_CONTROLS[
            "tool_generation_enabled"
        ],
        "open_coscientist_debate_cohorts_enabled": OPEN_COSCIENTIST_COMPUTE_CONTROLS[
            "debate_cohorts_enabled"
        ],
    }


def _load_tool_runtime():
    path = ROOT / "core/tool-runtime/runtime.py"
    spec = importlib.util.spec_from_file_location("neuroclaw_tool_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load NeuroRuntime from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ToolRuntime


def _run_runtime_job(payload: dict[str, Any]) -> dict[str, Any]:
    ToolRuntime = _load_tool_runtime()
    runtime = ToolRuntime(timeout=int(payload["timeout_seconds"]) + 120)
    result = runtime.run(HANDLER, payload)
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "NeuroRuntime job failed"))
    output = result.get("output")
    if not isinstance(output, dict):
        raise RuntimeError(f"invalid NeuroRuntime output: {type(output).__name__}")
    return output


def _append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        for attempt in range(1, 9):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                return
            except OSError:
                if attempt == 8:
                    raise
                time.sleep(min(4.0, 0.25 * (2 ** (attempt - 1))))


def _verify_formal_code_lock(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    code_lock = getattr(args, "code_lock", None)
    if code_lock is None:
        return {
            "created_at": utc_now(),
            "stage": stage,
            "enabled": False,
            "ok": True,
            "errors": [],
        }
    from core.scripts.run_supplemental_case_studies_formal import (
        load_and_verify_lock,
    )

    _payload, errors = load_and_verify_lock(
        Path(code_lock),
        str(getattr(args, "lock_phase", "development")),
        verify_frameworks=True,
    )
    return {
        "created_at": utc_now(),
        "stage": stage,
        "enabled": True,
        "ok": not errors,
        "errors": errors,
    }


def _combine_policies(
    task_dir: Path,
    *,
    methods: list[str],
    trials: int,
) -> Path:
    path = task_dir / "search_policies.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for method in methods:
            for trial in range(trials):
                policy_path = task_dir / method / f"trial_{trial:02d}" / "search_policy.json"
                if not policy_path.is_file():
                    raise FileNotFoundError(policy_path)
                payload = json.loads(policy_path.read_text(encoding="utf-8"))
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _benchmark_external_outcomes(task: str, tables: Path) -> Path | None:
    """Resolve external outcomes from the registered validation assignment."""

    if task in EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES:
        return None
    path = tables / "external_outcomes.csv"
    if task in EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES and not path.is_file():
        raise FileNotFoundError(
            f"registered external-validation outcomes are missing for {task}: {path}"
        )
    return path if path.is_file() else None


def _external_validation_assignment(task: str) -> str:
    """Return the registered external-validation role for a framework task."""

    if task in EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES:
        return "required"
    if task in EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES:
        return "registered_exempt"
    return "outside_seven_case_scope"


def _benchmark_task(
    *,
    task: str,
    source_root: Path,
    comparison_dir: Path,
    methods: list[str],
    trials: int,
    seed: int,
    neurodiscovery_config: Path | None = None,
) -> dict[str, Any]:
    protocol = protocol_for(task)
    tables = source_root / task / "tables"
    public = pd.read_csv(tables / "public_candidates.csv", low_memory=False)
    budgets = sorted({value for value in protocol.budgets if value <= len(public)})
    if len(public) not in budgets:
        budgets.append(len(public))
    policies = _combine_policies(comparison_dir, methods=methods, trials=trials)
    score_components = tables / "score_components.csv"
    score_components_manifest = tables / "score_components.manifest.json"
    if score_components.is_file() != score_components_manifest.is_file():
        raise FileNotFoundError(
            "score_components.csv and score_components.manifest.json must coexist"
        )
    benchmark_args = argparse.Namespace(
        task=task,
        public_candidates=tables / "public_candidates.csv",
        internal_outcomes=tables / "internal_outcomes.csv",
        external_outcomes=_benchmark_external_outcomes(task, tables),
        score_components=score_components if score_components.is_file() else None,
        score_components_manifest=(
            score_components_manifest if score_components_manifest.is_file() else None
        ),
        search_policies=policies,
        neurodiscovery_config=neurodiscovery_config,
        policy_schema=POLICY_SCHEMA,
        output_dir=comparison_dir / "benchmark",
        factor_fields=list(protocol.factor_fields),
        methods=["random_walk", *methods, "neurodiscovery"],
        trials=trials,
        seed=seed,
        batch_size=min(64, max(8, len(public) // 10)),
        warmup_batches=1,
        max_feedback_rounds=128,
        feedback_horizon=max(budgets),
        budgets=budgets,
        recall_targets=list(protocol.recall_targets),
    )
    return run_benchmark(benchmark_args)


def run_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    protocol = protocol_for(task)
    tables = args.root / task / "tables"
    public_path = tables / "public_candidates.csv"
    public = pd.read_csv(public_path, low_memory=False)
    public_registry = public[[*protocol.factor_fields, "candidate_id"]].copy()
    public_registry = public_registry.sort_values("candidate_id", kind="stable")
    comparison_dir = args.root / task / args.output_name
    _ensure_directory_with_retry(comparison_dir)
    registry_path = comparison_dir / "public_registry.jsonl"
    _write_text_with_retry(
        registry_path,
        public_registry.to_json(orient="records", lines=True, force_ascii=False),
    )
    n_anchors = min(args.n_anchors, len(public_registry))
    status_path = comparison_dir / "job_status.jsonl"
    status_lock = threading.Lock()
    lock_status_path = comparison_dir / "code_lock_status.jsonl"
    lock_drift = threading.Event()
    method_locks = {method: threading.Lock() for method in NATIVE_METHODS}

    jobs: list[tuple[str, int, dict[str, Any]]] = []
    # Interleave frameworks by trial. Native implementations are serialized per
    # repository, but different repositories are independent and may run in
    # parallel. Method-major ordering otherwise leaves worker threads blocked on
    # the same lock and silently reduces the whole matrix to serial execution.
    for trial in range(args.trials):
        for method in args.methods:
            trial_dir = comparison_dir / method / f"trial_{trial:02d}"
            _ensure_directory_with_retry(trial_dir)
            task_payload = build_task(
                task=task,
                method=method,
                trial=trial,
                n_anchors=n_anchors,
                registry=public_registry,
                registry_path=registry_path,
            )
            task_path = trial_dir / "task.json"
            _write_text_with_retry(
                task_path,
                json.dumps(task_payload, indent=2, ensure_ascii=False),
            )
            runtime_payload = {
                "method": method,
                "task_path": str(task_path),
                "output_dir": str(trial_dir),
                "model": args.model,
                "responses_base_url": responses_base_url(args.base_url),
                "chat_base_url": chat_base_url(args.base_url),
                "reasoning_effort": args.reasoning_effort,
                "timeout_seconds": args.timeout_seconds,
                "request_timeout_seconds": args.request_timeout_seconds,
                "max_retries": args.max_retries,
                "batch_size": args.batch_size,
                "brainpilot_url": args.brainpilot_url,
                "force": args.force,
            }
            jobs.append((method, trial, runtime_payload))

    def execute(method: str, trial: int, payload: dict[str, Any]) -> dict[str, Any]:
        def guarded_run() -> dict[str, Any]:
            if lock_drift.is_set():
                raise RuntimeError("formal code lock drift was detected by another job")
            before = _verify_formal_code_lock(
                args,
                stage=f"before:{task}:{method}:trial_{trial:02d}",
            )
            _append_jsonl(lock_status_path, before, status_lock)
            if not before["ok"]:
                lock_drift.set()
                raise RuntimeError("formal code lock changed before framework job")
            guarded_result = _run_runtime_job(payload)
            after = _verify_formal_code_lock(
                args,
                stage=f"after:{task}:{method}:trial_{trial:02d}",
            )
            _append_jsonl(lock_status_path, after, status_lock)
            if not after["ok"]:
                lock_drift.set()
                raise RuntimeError("formal code lock changed during framework job")
            return guarded_result

        lock = method_locks.get(method)
        if lock is None:
            result = guarded_run()
        else:
            with lock:
                result = guarded_run()
        _append_jsonl(
            status_path,
            {"created_at": utc_now(), "task": task, **result},
            status_lock,
        )
        return result

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as pool:
        futures = {
            pool.submit(execute, method, trial, payload): (method, trial)
            for method, trial, payload in jobs
        }
        completed = 0
        for future in as_completed(futures):
            method, trial = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                failure = {
                    "created_at": utc_now(),
                    "task": task,
                    "method": method,
                    "trial": trial,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                _append_jsonl(status_path, failure, status_lock)
                print(
                    f"[{task}] {completed}/{len(jobs)} {method} trial={trial} FAILED",
                    flush=True,
                )
            else:
                results.append(result)
                print(
                    f"[{task}] {completed}/{len(jobs)} {method} trial={trial} "
                    f"anchors={result.get('valid_anchors', 0)}",
                    flush=True,
                )

    _write_text_with_retry(
        comparison_dir / "failures.json",
        json.dumps(failures, indent=2, ensure_ascii=False),
    )
    benchmark_manifest = None
    neurodiscovery_config = getattr(args, "neurodiscovery_config", None)
    if neurodiscovery_config is not None and neurodiscovery_config.is_dir():
        neurodiscovery_config = (
            neurodiscovery_config / task / "selected_policy.json"
        )
    if neurodiscovery_config is not None and not neurodiscovery_config.is_file():
        raise FileNotFoundError(neurodiscovery_config)
    if not failures and args.trials >= 2 and not args.skip_benchmark:
        benchmark_manifest = _benchmark_task(
            task=task,
            source_root=args.root,
            comparison_dir=comparison_dir,
            methods=args.methods,
            trials=args.trials,
            seed=args.seed,
            neurodiscovery_config=neurodiscovery_config,
        )
    manifest = {
        "schema_version": "case-study-framework-comparison.v1",
        "created_at": utc_now(),
        "task": task,
        "status": "complete" if not failures else "incomplete",
        "source_public_candidates": str(public_path),
        "source_public_candidates_sha256": sha256_file(public_path),
        "sanitized_registry": str(registry_path),
        "sanitized_columns": [*protocol.factor_fields, "candidate_id"],
        "candidate_count": len(public_registry),
        "n_anchors": n_anchors,
        "requested_n_anchors": args.n_anchors,
        "native_batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "process_max_retries": args.max_retries,
        "benchmark_seed": args.seed,
        "external_validation_assignment": _external_validation_assignment(task),
        "methods": args.methods,
        "trials": args.trials,
        "completed_jobs": len(results),
        "failed_jobs": len(failures),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout_seconds": args.request_timeout_seconds,
        "job_timeout_seconds": args.timeout_seconds,
        "responses_base_url": responses_base_url(args.base_url),
        "chat_compatibility_base_url": chat_base_url(args.base_url),
        "disable_response_storage": True,
        "native_retrieval_enabled": False,
        "neurodiscovery_config": (
            {
                "path": str(neurodiscovery_config.resolve()),
                "sha256": sha256_file(neurodiscovery_config),
            }
            if neurodiscovery_config is not None
            else None
        ),
        "framework_compute_controls": {
            "open_coscientist": dict(OPEN_COSCIENTIST_COMPUTE_CONTROLS)
        },
        "code_lock": (
            str(Path(args.code_lock).resolve())
            if getattr(args, "code_lock", None) is not None
            else None
        ),
        "code_lock_phase": getattr(args, "lock_phase", None),
        "code_lock_drift_detected": lock_drift.is_set(),
        "benchmark_manifest": benchmark_manifest,
    }
    _write_text_with_retry(
        comparison_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
    )
    if failures and not args.allow_incomplete:
        raise RuntimeError(f"{task}: {len(failures)} framework jobs failed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[*TASK_PROTOCOLS, "all"],
        default=list(DEFAULT_TASKS),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-anchors", type=int, default=DEFAULT_N_ANCHORS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_NATIVE_BATCH_SIZE)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=1800,
        help="Per-request wait; the finite router still owns provider retries.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_PROCESS_RETRIES,
        help="Process retries; provider retries are owned by the finite router.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--brainpilot-url", default="http://127.0.0.1:9600/api")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--code-lock",
        type=Path,
        help="Formal supplemental code lock re-verified around every framework job.",
    )
    parser.add_argument(
        "--lock-phase",
        choices=("development", "final"),
        default="development",
    )
    parser.add_argument(
        "--neurodiscovery-config",
        type=Path,
        help=(
            "Frozen selected_policy.json, or a tuning root containing "
            "<task>/selected_policy.json."
        ),
    )
    parser.add_argument("--skip-api-probe", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "all" in args.tasks:
        args.tasks = list(DEFAULT_TASKS)
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.n_anchors < 1 or args.batch_size < 1:
        raise ValueError("--n-anchors and --batch-size must be positive")
    if args.timeout_seconds < 1 or args.request_timeout_seconds < 1:
        raise ValueError("--timeout-seconds and --request-timeout-seconds must be positive")
    _secret(args.base_url)
    complete_policy_resume = _complete_policy_matrix_present(args)
    probe_skip_reason = (
        "explicit_skip_api_probe"
        if args.skip_api_probe
        else "complete_policy_matrix_resume"
        if complete_policy_resume
        else None
    )
    provider_probe = (
        None
        if probe_skip_reason is not None
        else probe_sub2api(args.base_url, args.model, args.reasoning_effort)
    )
    preflight_rows = preflight(args.methods, args.brainpilot_url)
    failed = [row for row in preflight_rows if row["status"] != "ok"]
    global_dir = args.root / args.output_name
    _ensure_directory_with_retry(global_dir)
    _write_text_with_retry(
        global_dir / "preflight.json",
        json.dumps(
            {
                "provider": provider_probe,
                "provider_probe_skipped_reason": probe_skip_reason,
                "frameworks": preflight_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    if failed:
        names = ", ".join(str(row["method"]) for row in failed)
        raise RuntimeError(f"framework preflight failed: {names}")

    manifests: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for task in args.tasks:
        try:
            manifests[task] = run_task(task, args)
        except Exception as exc:
            failures[task] = f"{type(exc).__name__}: {exc}"
            if not args.allow_incomplete:
                break
    summary = {
        "schema_version": "case-study-framework-matrix.v1",
        "created_at": utc_now(),
        "tasks": args.tasks,
        "methods": args.methods,
        "trials": args.trials,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout_seconds": args.request_timeout_seconds,
        "job_timeout_seconds": args.timeout_seconds,
        "requested_n_anchors": args.n_anchors,
        "native_batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "process_max_retries": args.max_retries,
        "benchmark_seed": args.seed,
        "external_validation_required_tasks": list(
            EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES
        ),
        "external_validation_exempt_tasks": list(
            EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES
        ),
        "provider_probe": provider_probe,
        "provider_probe_skipped_reason": probe_skip_reason,
        "framework_preflight": preflight_rows,
        "task_manifests": manifests,
        "failures": failures,
        "complete": not failures and len(manifests) == len(args.tasks),
    }
    summary_path = global_dir / "matrix_manifest.json"
    _write_text_with_retry(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
    )
    print(summary_path)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
