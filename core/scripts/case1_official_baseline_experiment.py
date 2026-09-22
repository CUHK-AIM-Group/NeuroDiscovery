"""Run official autoresearch adapters under the blinded CS1 SearchPolicy contract.

This runner does not emulate systems with role prompts. Each primary method is
invoked through its official source tree, and its native artifact must contain
exact public candidate identifiers. A deterministic compiler validates those
identifiers without a second LLM interpretation. data-to-paper and OpenScholar
are supplementary adapters and require their native artifact to be supplied
when their full upstream workflow is run separately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd

try:
    from core.scripts.case1_method_comparison import DEFAULT_ALL_TESTS, load_results
    from core.scripts.case1_policy_audit import write_policy_independence_audit
    from core.scripts.case1_search_policy import (
        COMPILER_VERSION,
        SearchPolicy,
        build_public_registry,
        policy_from_payload,
        policy_to_payload,
        validate_policy,
    )
except ModuleNotFoundError:
    from case1_method_comparison import DEFAULT_ALL_TESTS, load_results
    from case1_policy_audit import write_policy_independence_audit
    from case1_search_policy import (
        COMPILER_VERSION,
        SearchPolicy,
        build_public_registry,
        policy_from_payload,
        policy_to_payload,
        validate_policy,
    )


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
DEFAULT_OUT_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260707_fullv2_kg_rerun"
    r"\official_baselines_gpt55_high"
)

PRIMARY_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
SUPPLEMENTARY_METHODS = ("data_to_paper", "openscholar_rag")
METHODS = (*PRIMARY_METHODS, *SUPPLEMENTARY_METHODS)
REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
    "data_to_paper": "data-to-paper",
    "openscholar_rag": "OpenScholar",
}
RUNTIME_ENV_VARS = {
    "ai_scientist_v2": "CS1_AI_SCIENTIST_V2_PYTHON",
    "open_coscientist": "CS1_OPEN_COSCIENTIST_PYTHON",
    "sciagents": "CS1_SCIAGENTS_PYTHON",
    "virtual_lab": "CS1_VIRTUAL_LAB_PYTHON",
    "data_to_paper": "CS1_DATA_TO_PAPER_PYTHON",
    "openscholar_rag": "CS1_OPENSCHOLAR_PYTHON",
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


class InfrastructureFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(PRIMARY_METHODS),
    )
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-anchors", type=int, default=80)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--native-artifacts",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <method>/trial_XX/native_result.json. "
            "Required to collect separately-run supplementary workflows."
        ),
    )
    parser.add_argument(
        "--merge-policy-files",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Additional case1_search_policies.jsonl files, for example the "
            "BrainPilot/Biomni native-adapter output."
        ),
    )
    parser.add_argument(
        "--enable-native-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow each framework's own retrieval/tool layer during native generation "
            "(enabled by default; use --no-enable-native-retrieval for an ablation)."
        ),
    )
    return parser.parse_args()


def repository_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def repository_python(method: str, path: Path) -> Path:
    configured = os.environ.get(RUNTIME_ENV_VARS[method])
    candidates = [
        Path(configured) if configured else None,
        path / ".venv" / "Scripts" / "python.exe",
        path / "venv" / "Scripts" / "python.exe",
        Path(os.environ["CONDA_PYTHON"]) if os.environ.get("CONDA_PYTHON") else None,
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise InfrastructureFailure(
        f"no Python runtime found for {method}; set {RUNTIME_ENV_VARS[method]}"
    )


def repository_preflight(method: str) -> dict[str, Any]:
    repo = BASELINES_ROOT / REPOSITORIES[method]
    if method not in PRIMARY_METHODS:
        return {
            "method": method,
            "required_for_runner": False,
            "status": "native_artifact_required",
            "repository": str(repo),
        }
    if not repo.exists():
        return {
            "method": method,
            "required_for_runner": True,
            "status": "failed",
            "repository": str(repo),
            "runtime_env_var": RUNTIME_ENV_VARS[method],
            "error": "official repository is missing",
        }
    try:
        runtime = repository_python(method, repo)
    except InfrastructureFailure as exc:
        return {
            "method": method,
            "required_for_runner": True,
            "status": "failed",
            "repository": str(repo),
            "runtime_env_var": RUNTIME_ENV_VARS[method],
            "error": str(exc),
        }
    result = subprocess.run(
        [str(runtime), "-c", IMPORT_PROBES[method]],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    return {
        "method": method,
        "required_for_runner": True,
        "status": "ok" if result.returncode == 0 else "failed",
        "repository": str(repo),
        "runtime": str(runtime),
        "runtime_env_var": RUNTIME_ENV_VARS[method],
        "returncode": result.returncode,
        "error": result.stderr[-4000:] if result.returncode else "",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_prompt(
    method: str,
    trial: int,
    n_anchors: int,
    registry: pd.DataFrame,
) -> str:
    diseases = "\n".join(f"- {value}" for value in sorted(registry["disease"].unique()))
    features = "\n".join(f"- {value}" for value in sorted(registry["feature"].unique()))
    anatomy = (
        registry[
            [
                "anatomy_id",
                "anatomy_full",
                "hemisphere",
                "map_group",
                "network",
                "structure_class",
            ]
        ]
        .drop_duplicates("anatomy_id")
        .sort_values("anatomy_id", kind="mergesort")
    )
    anatomy_lines = "\n".join(
        f"- {row.anatomy_id} = {row.anatomy_full}"
        for row in anatomy.itertuples(index=False)
    )
    return f"""Blinded Case Study 1 transdiagnostic neuroimaging search.

Official adapter: {method}
Independent trial ID: {trial}

Scientific objective:
Prioritize executable disease x imaging-feature x atlas-ROI hypotheses that are
most likely to reveal reproducible transdiagnostic neuroimaging effects. Apply
the official framework's native planning, criticism, ranking, graph reasoning,
or team process. Do not inspect any CS1 experimental result file.

Allowed diseases:
{diseases}

Allowed imaging features:
{features}

Allowed anatomy coordinates:
{anatomy_lines}

Executable candidate IDs have this exact form:
modality|source|disease|feature|roi_index

Produce scientific proposals for {n_anchors} distinct executable candidates.
Native output may use the framework's own schema, but every final proposal must
contain its exact candidate_id verbatim. Build it by combining the selected
anatomy_id fields with disease and feature:
  anatomy_id = modality|source|roi_index
  candidate_id = modality|source|disease|feature|roi_index
There is no downstream LLM projection or fuzzy repair. Proposals without an
exact registered candidate_id consume their slot and receive zero credit.
Neither generation nor deterministic validation may use effect sizes, p-values,
FDR, GT labels, NeuroDiscovery scores, or closed-loop feedback.
"""


def prepare_task(
    method: str,
    trial: int,
    args: argparse.Namespace,
    registry: pd.DataFrame,
    registry_path: Path,
) -> Path:
    trial_dir = args.out_dir / method / f"trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    task_path = trial_dir / "task.json"
    payload = {
        "method": method,
        "trial": trial,
        "n_anchors": args.n_anchors,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "public_registry_path": str(registry_path),
        "research_goal": task_prompt(method, trial, args.n_anchors, registry),
        "blinding": {
            "outcomes_exposed": False,
            "effect_sizes_exposed": False,
            "p_values_exposed": False,
            "fdr_exposed": False,
            "gt_labels_exposed": False,
            "neurodiscovery_scores_exposed": False,
        },
    }
    task_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_path


def run_adapter(
    method: str,
    trial: int,
    task_path: Path,
    args: argparse.Namespace,
    secret: str,
) -> SearchPolicy:
    trial_dir = task_path.parent
    policy_path = trial_dir / "search_policy.json"
    if policy_path.exists() and not args.force:
        return policy_from_payload(json.loads(policy_path.read_text(encoding="utf-8")))

    supplied_artifact = None
    if args.native_artifacts is not None:
        candidate = (
            args.native_artifacts / method / f"trial_{trial:02d}" / "native_result.json"
        )
        if candidate.exists():
            supplied_artifact = candidate

    repo = BASELINES_ROOT / REPOSITORIES[method]
    if not repo.exists():
        raise InfrastructureFailure(f"official repository is missing: {repo}")

    command = [
        str(repository_python(method, repo)),
        str(ROOT / "core" / "scripts" / "case_study_official_adapter_client.py"),
        "--method",
        method,
        "--task",
        str(task_path),
        "--out",
        str(trial_dir),
        "--repo",
        str(repo),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if args.enable_native_retrieval:
        command.append("--enable-native-retrieval")
    if supplied_artifact is not None:
        command.extend(["--native-artifact", str(supplied_artifact)])

    env = os.environ.copy()
    env["CS1_LOCAL_API_KEY"] = secret
    env["OPENAI_API_KEY"] = secret
    env["OPENAI_BASE_URL"] = args.base_url
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        started = time.time()
        try:
            result = subprocess.run(
                command,
                cwd=repo,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout={args.timeout_seconds}s"
            (trial_dir / f"adapter_attempt_{attempt:02d}.stderr.log").write_text(
                str(exc).replace(secret, "<redacted>"),
                encoding="utf-8",
            )
            if attempt < args.max_retries:
                time.sleep((2, 8, 30)[min(attempt - 1, 2)])
            continue
        (trial_dir / f"adapter_attempt_{attempt:02d}.stdout.log").write_text(
            result.stdout.replace(secret, "<redacted>"),
            encoding="utf-8",
        )
        (trial_dir / f"adapter_attempt_{attempt:02d}.stderr.log").write_text(
            result.stderr.replace(secret, "<redacted>"),
            encoding="utf-8",
        )
        (trial_dir / f"adapter_attempt_{attempt:02d}.duration.txt").write_text(
            f"{time.time() - started:.3f}\n",
            encoding="utf-8",
        )
        if result.returncode == 0 and policy_path.exists():
            return policy_from_payload(
                json.loads(policy_path.read_text(encoding="utf-8"))
            )
        last_error = f"exit={result.returncode}"
        if attempt < args.max_retries:
            time.sleep((2, 8, 30)[min(attempt - 1, 2)])
    raise InfrastructureFailure(
        f"{method} trial {trial} failed after {args.max_retries} attempts ({last_error})"
    )


def policy_rows(policy: SearchPolicy) -> list[dict[str, Any]]:
    return [
        {
            "method": policy.method,
            "seed": policy.trial,
            "generated_rank": rank,
            "mapping_status": "mapped",
            "mapping_score": 1.0,
            "mapped_candidate_id": anchor.candidate_id,
            "generated_confidence": anchor.score,
            "generated_rationale": anchor.rationale,
        }
        for rank, anchor in enumerate(policy.anchors, start=1)
    ]


def validate_trial_policy(
    policy: SearchPolicy,
    *,
    method: str,
    trial: int,
    max_anchors: int,
    registry: pd.DataFrame,
) -> SearchPolicy:
    validate_policy(policy, registry)
    if policy.method != method:
        raise ValueError(
            f"adapter returned method={policy.method!r}, expected {method!r}"
        )
    if policy.trial != trial:
        raise ValueError(f"adapter returned trial={policy.trial}, expected {trial}")
    if len(policy.anchors) > max_anchors:
        raise ValueError(
            f"adapter returned {len(policy.anchors)} anchors, maximum is {max_anchors}"
        )
    metadata = {
        **dict(policy.metadata),
        "requested_anchor_count": max_anchors,
        "valid_anchor_count": len(policy.anchors),
        "missing_anchor_slots": max_anchors - len(policy.anchors),
    }
    return SearchPolicy(
        method=policy.method,
        trial=policy.trial,
        anchors=policy.anchors,
        rules=policy.rules,
        quotas=policy.quotas,
        metadata=metadata,
        schema_version=policy.schema_version,
    )


def write_proposal_summary(
    out_dir: Path,
    *,
    methods: list[str],
    trials: list[int],
    requested_anchors: int,
    policies: list[SearchPolicy],
    failures: list[dict[str, Any]],
) -> None:
    policy_lookup = {(policy.method, policy.trial): policy for policy in policies}
    failure_lookup = {(str(row["method"]), int(row["trial"])): row for row in failures}
    rows = []
    for method in methods:
        for trial in trials:
            policy = policy_lookup.get((method, trial))
            failure = failure_lookup.get((method, trial))
            valid_count = len(policy.anchors) if policy is not None else 0
            rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "requested_anchors": requested_anchors,
                    "valid_anchors": valid_count,
                    "missing_or_invalid_slots": requested_anchors - valid_count,
                    "completion_status": (
                        "complete"
                        if valid_count == requested_anchors
                        else "partial"
                        if policy is not None
                        else "failed"
                    ),
                    "failure_type": failure.get("error_type", "") if failure else "",
                    "failure_message": failure.get("error", "") if failure else "",
                }
            )
    pd.DataFrame(rows).to_csv(
        out_dir / "official_native_proposal_summary.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    secret = os.environ.get("CS1_LOCAL_API_KEY")
    if not args.prepare_only and not secret:
        raise RuntimeError("CS1_LOCAL_API_KEY is required and is never serialized")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight = {method: repository_preflight(method) for method in args.methods}
    (args.out_dir / "official_adapter_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failed_preflight = [
        method
        for method, status in preflight.items()
        if status.get("required_for_runner") and status.get("status") != "ok"
    ]
    if failed_preflight and not args.prepare_only:
        details = ", ".join(
            f"{method} ({RUNTIME_ENV_VARS[method]})" for method in failed_preflight
        )
        raise InfrastructureFailure(
            "official adapter runtime preflight failed: "
            f"{details}; inspect official_adapter_preflight.json"
        )

    scored = load_results(args.all_tests, args.gt_top_frac)
    registry = build_public_registry(scored)
    registry_path = args.out_dir / "cs1_public_registry.jsonl"
    registry.to_json(
        registry_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    task_paths = {
        (method, trial): prepare_task(method, trial, args, registry, registry_path)
        for method in args.methods
        for trial in args.trials
    }
    repo_manifest = {
        method: {
            "path": str(BASELINES_ROOT / REPOSITORIES[method]),
            "commit": repository_commit(BASELINES_ROOT / REPOSITORIES[method]),
            "tier": "primary" if method in PRIMARY_METHODS else "supplementary",
        }
        for method in args.methods
    }
    manifest = {
        "schema_version": "case1-official-adapters-v1",
        "all_tests": str(args.all_tests),
        "all_tests_sha256": file_sha256(args.all_tests),
        "registry_path": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "candidate_count": len(registry),
        "methods": list(args.methods),
        "independent_trial_ids": list(args.trials),
        "n_anchors": args.n_anchors,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "runtime_preflight": preflight,
        "repositories": repo_manifest,
        "credential_policy": "CS1_LOCAL_API_KEY is passed only through child environments",
        "randomness_note": (
            "Trial IDs index repeated independent agent runs; upstream APIs do not "
            "guarantee a deterministic random seed."
        ),
        "native_mapping_mode": "deterministic_exact_native_only",
        "native_retrieval_enabled": bool(args.enable_native_retrieval),
        "policy_compiler_version": COMPILER_VERSION,
        "llm_brainstorm_included": False,
    }
    (args.out_dir / "official_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.prepare_only:
        print(args.out_dir)
        return

    policies: list[SearchPolicy] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.max_workers, len(task_paths)))
    ) as pool:
        futures = {
            pool.submit(
                run_adapter,
                method,
                trial,
                task_path,
                args,
                secret or "",
            ): (method, trial)
            for (method, trial), task_path in task_paths.items()
        }
        for future in as_completed(futures):
            method, trial = futures[future]
            try:
                policy = validate_trial_policy(
                    future.result(),
                    method=method,
                    trial=trial,
                    max_anchors=args.n_anchors,
                    registry=registry,
                )
                policies.append(policy)
                print(
                    f"{method} trial={trial} anchors={len(policy.anchors)}", flush=True
                )
            except Exception as exc:
                failures.append(
                    {
                        "method": method,
                        "trial": trial,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"{method} trial={trial} excluded={type(exc).__name__}", flush=True
                )

    write_proposal_summary(
        args.out_dir,
        methods=list(args.methods),
        trials=list(args.trials),
        requested_anchors=args.n_anchors,
        policies=policies,
        failures=failures,
    )
    if failures:
        (args.out_dir / "excluded_trials.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not args.allow_incomplete:
            raise InfrastructureFailure(
                f"{len(failures)} official-adapter trials did not complete"
            )

    for policy_file in args.merge_policy_files:
        with policy_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                policy = policy_from_payload(json.loads(line))
                validate_policy(policy, registry)
                policies.append(policy)

    policies.sort(key=lambda policy: (policy.method, policy.trial))
    seen_policy_keys: set[tuple[str, int]] = set()
    for policy in policies:
        key = (policy.method, policy.trial)
        if key in seen_policy_keys:
            raise ValueError(f"duplicate merged SearchPolicy: {key}")
        seen_policy_keys.add(key)
    independence = write_policy_independence_audit(
        args.out_dir,
        registry,
        policies,
    )
    manifest["policy_independence_audit"] = independence
    (args.out_dir / "official_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out_dir / "case1_search_policies.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for policy in policies:
            handle.write(
                json.dumps(
                    policy_to_payload(policy),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    mapped = pd.DataFrame([row for policy in policies for row in policy_rows(policy)])
    mapped.to_csv(
        args.out_dir / "generation_first_mapped_hypotheses.csv",
        index=False,
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()


# Last Updated At: 2026-08-01 10:20 HKT
