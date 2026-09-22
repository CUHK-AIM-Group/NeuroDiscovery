"""Run the frozen single-seed Case Study 2 closed-loop benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.scripts import case2_official_baseline_experiment as baseline_experiment
from core.scripts.case2_official_baseline_experiment import (
    InfrastructureFailure,
    _run_native,
    build_task as build_static_task,
)
from core.scripts.case2_search_policy import build_public_registry
from core.scripts.case_study_closed_loop import ClosedLoopConfig
from core.scripts.case_study_closed_loop_engine import run_closed_loop_order
from core.scripts.case_study_feedback_adapters import adapter_for
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import load_hypotheses
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni_longitudinal import build_ranked_candidates
from neurooracle.scripts.materialize_case2_closed_loop_subgraph import materialize as materialize_case2_subgraph
from neurooracle.scripts.prepare_case2_adni_closed_loop_benchmark import (
    EXPECTED_METHODS,
    load_json,
    sha256_file,
    verify_freeze,
    write_json,
)


BENCHMARK_ID = "case2_adni_closed_loop_benchmark_v3_luna_seed1"
OFFICIAL_METHODS = ("ai_scientist_v2", "open_coscientist", "sciagents", "virtual_lab")
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
INVALID_PREFIX = "__INVALID_CLOSED_LOOP_ACTION_"
LOCAL_TOKEN = "neuroclaw-local-router"
FEEDBACK_COLUMNS = (
    "candidate_id", "complete_case_n", "a_path_std", "a_path_hc3_p", "b_path_std",
    "b_path_hc3_p", "indirect_effect_std", "indirect_bootstrap_ci_lower",
    "indirect_bootstrap_ci_upper", "indirect_bootstrap_p", "feedback_status",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]+", "<redacted>", text)
    return text[:2000]


def gateway_preflight(base_url: str, config: Mapping[str, Any]) -> dict[str, Any]:
    health = base_url.rstrip("/")
    if health.endswith("/v1"):
        health = health[:-3]
    with urlopen(health + "/health", timeout=10) as response:
        payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
    backend = config["model_backend"]
    _require(payload.get("status") == "ok", "Gateway is not healthy")
    _require(payload.get("benchmark_id") == config["benchmark_id"] == BENCHMARK_ID, "Wrong gateway benchmark")
    _require(payload.get("model") == backend["logical_model"], "Gateway model changed")
    _require(payload.get("forced_reasoning_effort") == backend["reasoning_effort"], "Reasoning effort changed")
    _require(int(payload.get("max_concurrent_upstream_requests")) == 1, "Concurrency changed")
    _require(payload.get("execution_channel") == backend["execution_channel"], "Wrong execution channel")
    _require(payload.get("provider_api_calls") is False, "Gateway reports provider API calls")
    _require(payload.get("provider_keys_required") is False, "Gateway requires provider keys")
    _require(payload.get("temperature_control") == backend["provider_temperature_control"], "Temperature disclosure changed")
    _require(payload.get("thread_id") == backend["codex_thread_id"], "Provider thread changed")
    _require(payload.get("host_id") == backend["codex_host_id"], "Provider host changed")
    _require(payload.get("request_hash_lock") is True, "Request hash lock disabled")
    _require(payload.get("response_hash_lock") is True, "Response hash lock disabled")
    _require(
        payload.get("dispatch_delivery_mode") == backend["dispatch_delivery_mode"],
        "Dispatch delivery mode changed",
    )
    _require(payload.get("secrets_exposed") is False, "Gateway reports exposed secrets")
    return {key: payload.get(key) for key in (
        "benchmark_id", "model", "forced_reasoning_effort",
        "max_concurrent_upstream_requests", "execution_channel",
        "provider_api_calls", "provider_keys_required", "temperature_control",
        "thread_id", "host_id", "request_hash_lock", "response_hash_lock", "pending_count",
        "dispatch_delivery_mode",
    )}


def brainpilot_preflight(url: str, manifest_path: Path | None, config: Mapping[str, Any]) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        raise InfrastructureFailure("BrainPilot runtime manifest is required")
    manifest = load_json(manifest_path)
    backend = config["model_backend"]
    _require(manifest.get("model") == backend["logical_model"], "BrainPilot model changed")
    _require(manifest.get("reasoning_effort") == backend["reasoning_effort"], "BrainPilot reasoning changed")
    _require(manifest.get("native_retrieval_enabled") is False, "BrainPilot retrieval enabled")
    _require(manifest.get("credentials_persisted") is False, "BrainPilot persisted credentials")
    _require(manifest.get("base_urls") == [url.rstrip("/")], "BrainPilot URL changed")
    with urlopen(url.rstrip("/") + "/health", timeout=10) as response:
        _require(response.status < 400, "BrainPilot health failed")
    return {"status": "passed", "runtime_manifest": str(manifest_path), "retrieval": False}


def load_public(benchmark_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        benchmark_root / "generator_inputs/PUBLIC_CANDIDATE_REGISTRY.csv",
        dtype=str, keep_default_na=False,
    )
    registry = build_public_registry(raw)
    _require(len(registry) == 168, "Candidate count changed")
    _require(registry["candidate_id"].tolist() == raw["candidate_id"].tolist(), "Registry order changed")
    return registry


def load_oracle(benchmark_root: Path) -> pd.DataFrame:
    oracle = pd.read_csv(benchmark_root / "evaluator_only/EXPERIMENT_ORACLE.csv")
    _require(len(oracle) == 168 and oracle["candidate_id"].is_unique, "Oracle changed")
    return oracle.set_index("candidate_id", drop=False)


def _feedback_text(feedback: Sequence[Mapping[str, Any]]) -> str:
    if not feedback:
        return "No experiment has been executed yet."
    lines = [
        "Only the following previously committed candidates have been executed. "
        "Use these observations to update the next selection. No multiplicity-adjusted q value is available."
    ]
    for row in feedback:
        lines.append(
            "- {candidate_id}: n={complete_case_n}; a={a_path_std:.6g} "
            "(P={a_path_hc3_p:.6g}); b={b_path_std:.6g} (P={b_path_hc3_p:.6g}); "
            "indirect={indirect_effect_std:.6g}, CI=[{indirect_bootstrap_ci_lower:.6g}, "
            "{indirect_bootstrap_ci_upper:.6g}], bootstrap P={indirect_bootstrap_p:.6g}; "
            "status={feedback_status}.".format(**row)
        )
    return "\n".join(lines)


def build_round_task(
    remaining: pd.DataFrame,
    *,
    method: str,
    round_index: int,
    requested: int,
    registry_path: Path,
    feedback: Sequence[Mapping[str, Any]],
    phase: str,
    native_coordinate_compiler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = build_static_task(
        remaining,
        method=method,
        trial=0,
        n_anchors=requested,
        registry_path=registry_path,
    )
    feedback_text = _feedback_text(feedback)
    task["case_study_id"] = "case2_closed_loop"
    task["round_index"] = round_index
    task["round_phase"] = phase
    task["research_goal"] = (
        task["research_goal"]
        + "\n\nCLOSED-LOOP ROUND RULES\n"
        + f"This is {phase} round {round_index}. Select exactly {requested} next actions from the remaining menu. "
        "Do not repeat any previously executed candidate. Update priorities from the revealed observations, "
        "but do not infer unobserved candidate results. Return the candidates in intended execution order.\n\n"
        + feedback_text
    )
    task["previous_round_summaries"] = [feedback_text]
    task["round_contexts"] = [feedback_text]
    task["closed_loop_feedback_fields"] = [
        "complete_case_n", "a_path_std", "a_path_hc3_p", "b_path_std", "b_path_hc3_p",
        "indirect_effect_std", "indirect_bootstrap_ci_lower", "indirect_bootstrap_ci_upper",
        "indirect_bootstrap_p", "feedback_status",
    ]
    task["multiplicity_results_available"] = False
    if native_coordinate_compiler is not None:
        task["native_coordinate_compiler"] = json.loads(
            json.dumps(native_coordinate_compiler, sort_keys=True)
        )
    return task


def _runner_namespace(args: argparse.Namespace, config: Mapping[str, Any], requested: int) -> argparse.Namespace:
    backend = config["model_backend"]
    return argparse.Namespace(
        model=backend["logical_model"], base_url=backend["gateway_base_url"],
        reasoning_effort=backend["reasoning_effort"],
        timeout_seconds=float(
            backend.get("framework_round_timeout_seconds", args.timeout_seconds)
        ),
        max_retries=args.max_retries, n_anchors=requested, batch_size=requested,
        request_timeout_seconds=float(backend["response_timeout_seconds"]),
        enable_native_retrieval=False, prepare_only=False, force=False,
        brainpilot_url=args.brainpilot_url,
        brainpilot_runtime_manifest=args.brainpilot_runtime_manifest,
        methods=list(args.methods),
    )


def _run_frozen_official(
    benchmark_root: Path,
    method: str,
    task_path: Path,
    round_dir: Path,
    runner_args: argparse.Namespace,
) -> Any:
    adapter = benchmark_root / "generator_inputs/official_adapter.py"
    audit = load_json(benchmark_root / "generator_inputs/FROZEN_RUNTIME_SOURCE_AUDIT.json")
    _require(adapter.is_file(), "Frozen official adapter is missing")
    _require(
        sha256_file(adapter) == audit["frozen_copy_sha256"],
        "Frozen official adapter hash changed",
    )
    policy_path = round_dir / "search_policy.json"
    if policy_path.exists() and not runner_args.force:
        return baseline_experiment.policy_from_payload(load_json(policy_path))
    repo = baseline_experiment.BASELINES_ROOT / baseline_experiment.REPOSITORIES[method]
    command = [
        str(baseline_experiment._repository_python(method, repo)),
        str(adapter),
        "--method", method,
        "--task", str(task_path),
        "--out", str(round_dir),
        "--repo", str(repo),
        "--model", runner_args.model,
        "--base-url", runner_args.base_url,
        "--reasoning-effort", runner_args.reasoning_effort,
    ]
    if runner_args.enable_native_retrieval:
        command.append("--enable-native-retrieval")
    environment = os.environ.copy()
    environment["CASE_STUDY_LOCAL_API_KEY"] = LOCAL_TOKEN
    environment["CS1_LOCAL_API_KEY"] = LOCAL_TOKEN
    environment["OPENAI_API_KEY"] = LOCAL_TOKEN
    environment["OPENAI_BASE_URL"] = runner_args.base_url
    environment["CASE_STUDY_LLM_REQUEST_TIMEOUT"] = str(
        runner_args.request_timeout_seconds
    )
    environment["PYTHONUTF8"] = "1"
    _require(runner_args.max_retries == 1, "Frozen finite route permits one framework process only")
    baseline_experiment._run_process(
        command,
        cwd=repo,
        env=environment,
        out_dir=round_dir,
        timeout=runner_args.timeout_seconds,
        secret=LOCAL_TOKEN,
    )
    return baseline_experiment.policy_from_payload(load_json(policy_path))


def feedback_for_ids(oracle: pd.DataFrame, candidate_ids: Sequence[str]) -> list[dict[str, Any]]:
    allowed = FEEDBACK_COLUMNS
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        row = oracle.loc[candidate_id]
        record = {column: row[column] for column in allowed}
        record["candidate_id"] = str(record["candidate_id"])
        record["complete_case_n"] = int(record["complete_case_n"])
        for column in allowed[2:-1]:
            record[column] = float(record[column])
        record["feedback_status"] = str(record["feedback_status"])
        records.append(record)
    return records


def _selection_paths(round_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        round_dir / "selection.csv",
        round_dir / "SELECTION.lock.json",
        round_dir / "feedback.csv",
        round_dir / "ROUND.lock.json",
    )


def _load_locked_round(round_dir: Path, oracle: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    selection_path, selection_lock_path, feedback_path, round_lock_path = _selection_paths(round_dir)
    selection_lock = load_json(selection_lock_path)
    _require(sha256_file(selection_path) == selection_lock["selection_sha256"], "Selection drift")
    selection = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    actions = selection["candidate_id"].astype(str).tolist()
    valid_ids = selection.loc[selection["valid_selected"].str.casefold().eq("true"), "candidate_id"].tolist()
    expected_feedback = feedback_for_ids(oracle, valid_ids)
    if not feedback_path.exists():
        pd.DataFrame(expected_feedback, columns=FEEDBACK_COLUMNS).to_csv(feedback_path, index=False)
    feedback = pd.read_csv(feedback_path).to_dict("records")
    _require([str(row["candidate_id"]) for row in feedback] == valid_ids, "Feedback candidates changed")
    if round_lock_path.exists():
        round_lock = load_json(round_lock_path)
        _require(sha256_file(feedback_path) == round_lock["feedback_sha256"], "Feedback drift")
    else:
        write_json(
            round_lock_path,
            {
                "schema_version": "neurooracle.case2_closed_loop_round_lock.v3",
                "status": "selection_committed_then_feedback_revealed",
                "selection_lock_sha256": sha256_file(selection_lock_path),
                "feedback_sha256": sha256_file(feedback_path),
                "selected_candidate_count": len(valid_ids),
                "q_values_revealed": False,
                "unselected_results_revealed": False,
            },
        )
    return actions, expected_feedback


def run_baseline_round(
    benchmark_root: Path,
    *,
    method: str,
    round_index: int,
    phase: str,
    requested: int,
    remaining: pd.DataFrame,
    prior_feedback: Sequence[Mapping[str, Any]],
    oracle: pd.DataFrame,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    native_lock: threading.Lock,
) -> tuple[list[str], list[dict[str, Any]]]:
    round_dir = benchmark_root / "runs" / method / "trial_00" / f"round_{round_index:02d}_{phase}"
    round_dir.mkdir(parents=True, exist_ok=True)
    selection_path, selection_lock_path, _, round_lock_path = _selection_paths(round_dir)
    if round_lock_path.exists() or selection_lock_path.exists():
        return _load_locked_round(round_dir, oracle)

    registry_path = round_dir / "remaining_public_registry.csv"
    remaining.to_csv(registry_path, index=False)
    task = build_round_task(
        remaining, method=method, round_index=round_index, requested=requested,
        registry_path=registry_path, feedback=prior_feedback, phase=phase,
        native_coordinate_compiler=config.get("native_coordinate_compiler"),
    )
    task_path = round_dir / "task.json"
    write_json(task_path, task)
    runner_args = _runner_namespace(args, config, requested)
    if method in OFFICIAL_METHODS:
        policy = _run_frozen_official(
            benchmark_root, method, task_path, round_dir, runner_args,
        )
    elif method in NATIVE_METHODS:
        policy = _run_native(
            method, task, round_dir, remaining, runner_args, LOCAL_TOKEN, native_lock,
        )
    else:
        raise ValueError(f"Not a baseline method: {method}")

    remaining_ids = set(remaining["candidate_id"].astype(str))
    valid_ids: list[str] = []
    seen: set[str] = set()
    for anchor in policy.anchors:
        candidate_id = str(anchor.candidate_id)
        if candidate_id in remaining_ids and candidate_id not in seen and len(valid_ids) < requested:
            valid_ids.append(candidate_id)
            seen.add(candidate_id)
    failed = requested - len(valid_ids)
    sentinels = [
        f"{INVALID_PREFIX}{method}_r{round_index:02d}_{index:03d}__"
        for index in range(1, failed + 1)
    ]
    actions = [*valid_ids, *sentinels]
    selection = pd.DataFrame(
        {
            "within_round_action": range(1, requested + 1),
            "candidate_id": actions,
            "valid_selected": [True] * len(valid_ids) + [False] * failed,
        }
    )
    selection.to_csv(selection_path, index=False)
    write_json(
        selection_lock_path,
        {
            "schema_version": "neurooracle.case2_closed_loop_selection_lock.v3",
            "benchmark_id": BENCHMARK_ID,
            "method": method,
            "trial": 0,
            "round": round_index,
            "phase": phase,
            "requested_actions": requested,
            "valid_selected": len(valid_ids),
            "failed_actions": failed,
            "selection_sha256": sha256_file(selection_path),
            "task_sha256": sha256_file(task_path),
            "policy_sha256": sha256_file(round_dir / "search_policy.json"),
            "outcomes_accessed_before_commit": False,
            "automatic_completion": False,
        },
    )
    return _load_locked_round(round_dir, oracle)


def _trajectory_paths(benchmark_root: Path, method: str) -> tuple[Path, Path]:
    base = benchmark_root / "trajectories" / method / "trial_00"
    return base / "trajectory.csv", base / "TRAJECTORY.lock.json"


def verify_completed_trajectory(benchmark_root: Path, method: str) -> bool:
    trajectory_path, lock_path = _trajectory_paths(benchmark_root, method)
    if not lock_path.exists():
        return False
    lock = load_json(lock_path)
    _require(lock["status"] == "locked_complete_method_selected_trajectory", "Bad trajectory lock")
    _require(sha256_file(trajectory_path) == lock["trajectory_sha256"], "Trajectory drift")
    return True


def run_baseline_method(
    benchmark_root: Path,
    registry: pd.DataFrame,
    oracle: pd.DataFrame,
    *, method: str, config: Mapping[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    if verify_completed_trajectory(benchmark_root, method):
        return load_json(_trajectory_paths(benchmark_root, method)[1])
    selected: set[str] = set()
    feedback: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    native_lock = threading.Lock()
    round_index = 0
    for requested in map(int, config["sequential_design"]["main_round_batch_sizes"]):
        remaining = registry.loc[~registry["candidate_id"].isin(selected)].reset_index(drop=True)
        actions, new_feedback = run_baseline_round(
            benchmark_root, method=method, round_index=round_index, phase="main",
            requested=requested, remaining=remaining, prior_feedback=feedback,
            oracle=oracle, config=config, args=args, native_lock=native_lock,
        )
        valid = {str(row["candidate_id"]) for row in new_feedback}
        _require(not (valid & selected), "A method repeated a selected candidate")
        for within, candidate_id in enumerate(actions, start=1):
            action_rows.append({
                "action_slot": len(action_rows) + 1, "round": round_index,
                "phase": "main", "within_round_action": within,
                "candidate_id": candidate_id,
                "valid_selected": not candidate_id.startswith(INVALID_PREFIX),
            })
        selected.update(valid)
        feedback.extend(new_feedback)
        print(f"{method} round {round_index + 1}/8: {len(valid)}/{requested} valid; cumulative {len(selected)}/168", flush=True)
        round_index += 1

    recovery_rounds = 0
    max_recovery = int(config["sequential_design"]["max_recovery_rounds"])
    recovery_batch = int(config["sequential_design"]["recovery_round_batch_size"])
    while len(selected) < len(registry) and recovery_rounds < max_recovery:
        remaining = registry.loc[~registry["candidate_id"].isin(selected)].reset_index(drop=True)
        requested = min(recovery_batch, len(remaining))
        actions, new_feedback = run_baseline_round(
            benchmark_root, method=method, round_index=round_index, phase="recovery",
            requested=requested, remaining=remaining, prior_feedback=feedback,
            oracle=oracle, config=config, args=args, native_lock=native_lock,
        )
        valid = {str(row["candidate_id"]) for row in new_feedback}
        _require(not (valid & selected), "Recovery repeated a selected candidate")
        for within, candidate_id in enumerate(actions, start=1):
            action_rows.append({
                "action_slot": len(action_rows) + 1, "round": round_index,
                "phase": "recovery", "within_round_action": within,
                "candidate_id": candidate_id,
                "valid_selected": not candidate_id.startswith(INVALID_PREFIX),
            })
        selected.update(valid)
        feedback.extend(new_feedback)
        recovery_rounds += 1
        round_index += 1
        print(f"{method} recovery {recovery_rounds}: cumulative {len(selected)}/168", flush=True)
    _require(len(selected) == len(registry), f"{method} did not select all candidates after recovery")

    trajectory_path, lock_path = _trajectory_paths(benchmark_root, method)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(action_rows).to_csv(trajectory_path, index=False)
    lock = {
        "schema_version": "neurooracle.case2_closed_loop_trajectory_lock.v3",
        "benchmark_id": BENCHMARK_ID, "method": method, "trial": 0,
        "status": "locked_complete_method_selected_trajectory",
        "trajectory_sha256": sha256_file(trajectory_path),
        "main_action_slots": 168,
        "total_action_slots": len(action_rows),
        "invalid_main_actions": sum(
            row["phase"] == "main" and not row["valid_selected"] for row in action_rows
        ),
        "recovery_action_slots": sum(row["phase"] == "recovery" for row in action_rows),
        "real_candidate_count": 168,
        "all_real_candidates_method_selected": True,
        "automatic_completion": False,
        "q_values_revealed_during_generation": False,
    }
    write_json(lock_path, lock)
    return lock


def _generate_neurodiscovery_initial(
    benchmark_root: Path, registry: pd.DataFrame, config: Mapping[str, Any], seed: int,
) -> pd.DataFrame:
    run_dir = benchmark_root / "runs/neurodiscovery/trial_00"
    hypotheses_dir = run_dir / "hypothesis_generation"
    raw_path = hypotheses_dir / "hypotheses_raw.json"
    scoped_graph = run_dir / "case2_scoped_knowledge_graph.json"
    source_v1_lock = load_json(
        Path(config["paths"]["source_static_benchmark_root"]) / "BENCHMARK_FREEZE.lock.json"
    )
    claims_pin = source_v1_lock["lock_material"]["external_input_pins"]["extracted_claims"]
    scoped_manifest = materialize_case2_subgraph(
        Path(config["paths"]["extracted_claims"]), scoped_graph, source_pin=claims_pin,
    )
    _require(scoped_manifest["case_study_id"] == "case2_pathway_mediation", "Wrong scoped graph")
    _require(int(scoped_manifest["claim_count"]) > 0, "Scoped graph has no Case 2 claims")
    if not raw_path.exists():
        command = [
            sys.executable, "-m", "neurooracle.src.hypothesis_cli",
            "--graph", str(scoped_graph),
            "--generation-seed", str(seed), "case-study", "case2_pathway_mediation",
            "--output-dir", str(hypotheses_dir), "--stages", "batch",
            "--target-per-task", str(config["methods"]["neurodiscovery"]["generation_target"]),
        ]
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=4 * 3600, check=False,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "generator.stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_dir / "generator.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise InfrastructureFailure(f"NeuroDiscovery generation exited {result.returncode}")
    _require(raw_path.is_file(), "NeuroDiscovery produced no hypotheses")
    hypotheses = load_hypotheses(raw_path)
    universe = registry.copy()
    universe["gene_count"] = pd.to_numeric(universe["gene_count"], errors="raise").astype(int)
    exposures = (
        universe[["exposure", "pathway_id", "pathway_source", "pathway_name", "threshold_label", "gene_count"]]
        .drop_duplicates("exposure").rename(columns={"exposure": "score_name"})
    )
    pathway_catalog = pd.read_csv(config["paths"]["pathway_catalog"])
    ranked, support, audit = build_ranked_candidates(hypotheses, exposures, pathway_catalog, universe)
    ranked["candidate_id"] = (
        ranked["exposure"].astype(str) + "|" + ranked["modality"].astype(str)
        + "|" + ranked["marker"].astype(str) + "|" + ranked["outcome"].astype(str)
    )
    _require(len(ranked) == 168 and set(ranked["candidate_id"]) == set(registry["candidate_id"]), "ND mapping changed universe")
    ranked.to_csv(run_dir / "result_blind_mapping.csv", index=False)
    support.to_csv(run_dir / "result_blind_support_audit.csv", index=False)
    audit.to_csv(run_dir / "hypothesis_mapping_audit.csv", index=False)
    static = ranked[["candidate_id"]].copy()
    static.insert(0, "static_rank", range(1, 169))
    static.to_csv(run_dir / "static_initial_ranking.csv", index=False)
    write_json(
        run_dir / "scoped_graph_use.json",
        {
            "manifest": scoped_manifest,
            "canonical_claim_store_pin": claims_pin,
            "experimental_result_fields_used": [],
        },
    )
    return static


def run_neurodiscovery_method(
    benchmark_root: Path, registry: pd.DataFrame, oracle: pd.DataFrame,
    *, config: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    method = "neurodiscovery"
    if verify_completed_trajectory(benchmark_root, method):
        return load_json(_trajectory_paths(benchmark_root, method)[1])
    static = _generate_neurodiscovery_initial(benchmark_root, registry, config, seed)
    rank_lookup = dict(zip(static["candidate_id"], static["static_rank"], strict=True))
    public = registry.copy()
    public["score_neurodiscovery"] = public["candidate_id"].map(
        lambda value: 169.0 - float(rank_lookup[str(value)])
    )
    feedback_columns = [
        "candidate_id", "complete_case_n", "a_path_std", "a_path_hc3_p", "b_path_std",
        "b_path_hc3_p", "indirect_effect_std", "indirect_bootstrap_ci_lower",
        "indirect_bootstrap_ci_upper", "indirect_bootstrap_p", "feedback_status",
    ]
    outcomes = public.merge(
        oracle.reset_index(drop=True)[feedback_columns], on="candidate_id", how="left", validate="one_to_one"
    )
    outcomes["validated"] = outcomes["feedback_status"].eq("supported")
    outcomes["execution_succeeded"] = True
    run_dir = benchmark_root / "runs/neurodiscovery/trial_00"

    def commit(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        round_index = int(payload["batch"])
        round_dir = run_dir / f"round_{round_index:02d}_main"
        round_dir.mkdir(parents=True, exist_ok=True)
        selection_path, lock_path, _, _ = _selection_paths(round_dir)
        ids = list(map(str, payload["candidate_ids"]))
        pd.DataFrame({
            "within_round_action": range(1, len(ids) + 1),
            "candidate_id": ids, "valid_selected": [True] * len(ids),
        }).to_csv(selection_path, index=False)
        lock = {
            "schema_version": "neurooracle.case2_closed_loop_selection_lock.v3",
            "benchmark_id": BENCHMARK_ID, "method": method, "trial": 0,
            "round": round_index, "phase": "main", "requested_actions": len(ids),
            "valid_selected": len(ids), "failed_actions": 0,
            "selection_sha256": sha256_file(selection_path),
            "outcomes_accessed_before_commit": False, "automatic_completion": False,
        }
        write_json(lock_path, lock)
        return {"selection_lock_sha256": sha256_file(lock_path)}

    loop_config = ClosedLoopConfig(
        batch_size=21, warmup_batches=1, sampling_temperature=0.0,
        max_feedback_rounds=8, feedback_horizon=168,
        feedback_projection=config["methods"]["neurodiscovery"]["feedback_projection"],
        feedback_model=config["methods"]["neurodiscovery"]["feedback_model"],
        preserve_static_until_informative_feedback=True,
    )
    order, trace, overlay = run_closed_loop_order(
        public, outcomes, adapter=adapter_for("case2_pathway_mediation"),
        factor_fields=("exposure", "modality", "marker", "outcome"),
        rng=np.random.default_rng(seed), config=loop_config, seed=seed, trial=0,
        overlay_path=run_dir / "experimental_overlay.jsonl",
        batch_commit_callback=commit,
    )
    candidate_order = public.iloc[order]["candidate_id"].astype(str).tolist()
    _require(len(candidate_order) == len(set(candidate_order)) == 168, "ND trajectory is not complete")
    action_rows: list[dict[str, Any]] = []
    for row in trace:
        round_index = int(row["batch"])
        round_dir = run_dir / f"round_{round_index:02d}_main"
        ids = list(map(str, row["candidate_ids"]))
        round_feedback = feedback_for_ids(oracle, ids)
        feedback_path = round_dir / "feedback.csv"
        pd.DataFrame(round_feedback, columns=FEEDBACK_COLUMNS).to_csv(feedback_path, index=False)
        _, selection_lock_path, _, round_lock_path = _selection_paths(round_dir)
        write_json(
            round_lock_path,
            {
                "schema_version": "neurooracle.case2_closed_loop_round_lock.v3",
                "status": "selection_committed_then_feedback_revealed",
                "selection_lock_sha256": sha256_file(selection_lock_path),
                "feedback_sha256": sha256_file(feedback_path),
                "selected_candidate_count": len(ids),
                "q_values_revealed": False, "unselected_results_revealed": False,
                "selection_changed_by_overlay": bool(row["selection_changed_by_overlay"]),
            },
        )
        for within, candidate_id in enumerate(ids, start=1):
            action_rows.append({
                "action_slot": len(action_rows) + 1, "round": round_index,
                "phase": "main", "within_round_action": within,
                "candidate_id": candidate_id, "valid_selected": True,
            })
        print(
            f"neurodiscovery round {round_index + 1}/8: {len(ids)}/{len(ids)} valid; "
            f"overlay_changed={bool(row['selection_changed_by_overlay'])}", flush=True,
        )
    write_json(run_dir / "closed_loop_trace.json", {"trace": trace, "overlay": overlay})
    trajectory_path, lock_path = _trajectory_paths(benchmark_root, method)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(action_rows).to_csv(trajectory_path, index=False)
    lock = {
        "schema_version": "neurooracle.case2_closed_loop_trajectory_lock.v3",
        "benchmark_id": BENCHMARK_ID, "method": method, "trial": 0,
        "status": "locked_complete_method_selected_trajectory",
        "trajectory_sha256": sha256_file(trajectory_path),
        "main_action_slots": 168, "total_action_slots": 168,
        "invalid_main_actions": 0, "recovery_action_slots": 0,
        "real_candidate_count": 168, "all_real_candidates_method_selected": True,
        "automatic_completion": False, "q_values_revealed_during_generation": False,
        "static_initial_ranking_sha256": sha256_file(run_dir / "static_initial_ranking.csv"),
        "feedback_consumed_during_ranking": bool(overlay.get("feedback_consumed_during_ranking")),
        "selection_changed_batches": int(overlay.get("selection_changed_batches", 0)),
    }
    write_json(lock_path, lock)
    return lock


def finalize(benchmark_root: Path, methods: Sequence[str]) -> dict[str, Any]:
    config = load_json(benchmark_root / "PROTOCOL.json")
    migration = config.get("provider_migration") or {}
    old_thread_id = migration.get("old_thread_id")
    new_thread_id = config["model_backend"]["codex_thread_id"]
    records = []
    for method in methods:
        _require(verify_completed_trajectory(benchmark_root, method), f"Missing trajectory: {method}")
        path, lock_path = _trajectory_paths(benchmark_root, method)
        records.append({
            "method": method, "trajectory_relative_path": path.relative_to(benchmark_root).as_posix(),
            "trajectory_sha256": sha256_file(path),
            "trajectory_lock_sha256": sha256_file(lock_path),
        })
    payload = {
        "schema_version": "neurooracle.case2_closed_loop_trajectories_lock.v3",
        "benchmark_id": BENCHMARK_ID, "status": "locked_complete_single_seed_trajectories",
        "method_count": len(records), "trial_count": 1, "records": records,
        "automatic_completion": False, "single_seed_descriptive_only": True,
        "hybrid_resume": True,
        "reused_methods": ["neurodiscovery", "ai_scientist_v2"],
        "model_backed_methods_provider": "gpt-5.6-luna/max via audited mixed-thread Codex CLI replay",
        "provider_provenance_by_method": {
            "neurodiscovery": "API-independent hash-verified v2 import",
            "ai_scientist_v2": f"completed under source provider thread {old_thread_id} and hash-verified imported",
            "open_coscientist": f"exact sealed-response cache from {old_thread_id} plus new responses from {new_thread_id}",
            "sciagents": f"destination provider thread {new_thread_id}; any exact request-hash cache hit retains its recorded source thread",
            "virtual_lab": f"destination provider thread {new_thread_id}; any exact request-hash cache hit retains its recorded source thread",
            "brainpilot_native": f"destination provider thread {new_thread_id}; any exact request-hash cache hit retains its recorded source thread",
            "biomni_native": f"destination provider thread {new_thread_id}; any exact request-hash cache hit retains its recorded source thread",
        },
    }
    write_json(benchmark_root / "TRAJECTORIES.lock.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=EXPECTED_METHODS, default=list(EXPECTED_METHODS))
    parser.add_argument("--brainpilot-url", default="http://127.0.0.1:18080/api")
    parser.add_argument("--brainpilot-runtime-manifest", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser.parse_args(argv)


def _configure_frozen_baselines_root(config: Mapping[str, Any]) -> Path:
    configured = Path(config["paths"]["baseline_repository_root"]).resolve()
    _require(configured.is_dir(), "Frozen baseline repository root is missing")
    # The upstream helper defaults to a mutable sibling checkout. Formal runs
    # must instead execute the exact repository snapshot verified by PROTOCOL.
    baseline_experiment.BASELINES_ROOT = configured
    return configured


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    frozen = verify_freeze(benchmark_root)
    config = load_json(benchmark_root / "PROTOCOL.json")
    configured_baselines_root = _configure_frozen_baselines_root(config)
    _require(
        int(args.max_retries) == int(config["model_backend"]["framework_process_retries"]),
        "Framework process retries differ from the frozen finite-route protocol",
    )
    runtime = {
        "freeze": frozen,
        "gateway": gateway_preflight(config["model_backend"]["gateway_base_url"], config),
        "baseline_repository_root": str(configured_baselines_root),
        "baseline_repository_root_source": "frozen PROTOCOL paths.baseline_repository_root",
    }
    if "brainpilot_native" in args.methods:
        runtime["brainpilot"] = brainpilot_preflight(
            args.brainpilot_url, args.brainpilot_runtime_manifest, config,
        )
    write_json(benchmark_root / "runtime/RUN_PREFLIGHT.json", runtime)
    registry = load_public(benchmark_root)
    oracle = load_oracle(benchmark_root)
    seed = int(config["sequential_design"]["seed"])
    failures: list[dict[str, Any]] = []
    for method in args.methods:
        try:
            print(f"START {method}", flush=True)
            if method == "neurodiscovery":
                run_neurodiscovery_method(
                    benchmark_root, registry, oracle, config=config, seed=seed,
                )
            else:
                run_baseline_method(
                    benchmark_root, registry, oracle, method=method, config=config, args=args,
                )
            print(f"COMPLETE {method}", flush=True)
        except Exception as exc:
            failure = {"method": method, "error": _safe_error(exc)}
            failures.append(failure)
            write_json(benchmark_root / "runtime/LAST_FAILURE.json", failure)
            print(f"FAILED {method}: {failure['error']}", flush=True)
            break
    if failures:
        raise InfrastructureFailure(f"Closed-loop run stopped: {failures[-1]}")
    if all(verify_completed_trajectory(benchmark_root, method) for method in EXPECTED_METHODS):
        finalize(benchmark_root, EXPECTED_METHODS)
    end_verify = verify_freeze(benchmark_root)
    write_json(
        benchmark_root / "runtime/RUN_COMPLETE.json",
        {
            "status": "single_seed_execution_complete",
            "methods_executed_this_invocation": list(args.methods),
            "hybrid_resume": bool(config.get("hybrid_resume", {}).get("enabled")),
            "reused_methods": [
                config["hybrid_resume"]["reused_method"],
                *(config["hybrid_resume"].get("additional_reused_methods") or []),
            ],
            "provider_migration": config.get("provider_migration"),
            "freeze": end_verify,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
