"""Run the frozen seven-method Case Study 2 generator benchmark.

The command verifies every frozen input before execution.  API-based methods
talk only to the loopback DeepSeek V4 Pro gateway; this runner never reads or
passes an upstream provider key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.scripts.case2_official_baseline_experiment import (
    InfrastructureFailure,
    _run_native,
    _run_official,
)
from core.scripts.case2_search_policy import (
    PolicyAnchor,
    SCHEMA_VERSION,
    SearchPolicy,
    build_public_registry,
    compile_policy_order,
    policy_to_payload,
    validate_policy,
)
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import load_hypotheses
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni_longitudinal import (
    build_ranked_candidates,
)
from neurooracle.scripts.prepare_case2_adni_formal_benchmark import (
    EXPECTED_METHODS,
    load_json,
    sha256_file,
    verify_freeze,
    write_json,
)


OFFICIAL_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
INVALID_SLOT_PREFIX = "__INVALID_GENERATION_SLOT_"
LOCAL_GATEWAY_COMPATIBILITY_TOKEN = "neuroclaw-local-router"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]+", "<redacted>", text)
    text = re.sub(r"\bwrk_[A-Za-z0-9]+\b", "wrk_<redacted>", text)
    return text[:2000]


def gateway_preflight(base_url: str) -> dict[str, Any]:
    health_url = base_url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url += "/health"
    with urlopen(health_url, timeout=10) as response:
        _require(response.status == 200, f"Gateway health HTTP {response.status}")
        payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
    _require(payload.get("status") == "ok", "DeepSeek gateway is not healthy")
    _require(payload.get("model") == "deepseek-v4-pro", "Gateway model changed")
    _require(payload.get("benchmark_id") == "case2_adni_formal_benchmark_v1", "Generic or wrong benchmark gateway is running")
    _require(payload.get("forced_reasoning_effort") == "high", "Gateway reasoning setting changed")
    _require(float(payload.get("forced_temperature")) == 0.0, "Gateway temperature setting changed")
    _require(int(payload.get("forced_max_output_tokens")) == 8192, "Gateway output-token cap changed")
    _require(int(payload.get("max_concurrent_upstream_requests")) == 1, "Gateway concurrency changed")
    _require(payload.get("official_deepseek_enabled") is False, "Official DeepSeek is enabled without user opt-in")
    _require(payload.get("secrets_exposed") is False, "Gateway reports exposed secrets")
    return {
        "status": "passed",
        "model": payload["model"],
        "official_deepseek_enabled": False,
        "forced_reasoning_effort": payload["forced_reasoning_effort"],
        "forced_temperature": payload["forced_temperature"],
        "forced_max_output_tokens": payload["forced_max_output_tokens"],
        "max_concurrent_upstream_requests": payload["max_concurrent_upstream_requests"],
        "automatic_route": payload.get("automatic_route"),
    }


def brainpilot_preflight(url: str, manifest_path: Path | None, config: Mapping[str, Any]) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        raise InfrastructureFailure("BrainPilot requires a benchmark-specific runtime manifest")
    manifest = load_json(manifest_path)
    _require(manifest.get("model") == config["model_backend"]["logical_model"], "BrainPilot model differs from protocol")
    _require(manifest.get("reasoning_effort") == config["model_backend"]["reasoning_effort"], "BrainPilot reasoning differs from protocol")
    _require(manifest.get("native_retrieval_enabled") is False, "BrainPilot live retrieval must be disabled")
    _require(manifest.get("credentials_persisted") is False, "BrainPilot manifest reports persisted credentials")
    with urlopen(url.rstrip("/") + "/health", timeout=10) as response:
        _require(response.status < 400, f"BrainPilot health HTTP {response.status}")
    return {"status": "passed", "runtime_manifest": str(manifest_path), "native_retrieval_enabled": False}


def load_public_registry(benchmark_root: Path) -> pd.DataFrame:
    path = benchmark_root / "generator_inputs" / "PUBLIC_CANDIDATE_REGISTRY.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_columns = (
        "candidate_id", "exposure", "pathway_id", "pathway_name",
        "pathway_source", "threshold_label", "gene_count", "modality",
        "marker", "outcome",
    )
    _require(tuple(raw.columns) == expected_columns, "Public registry schema changed")
    registry = build_public_registry(raw)
    _require(registry["candidate_id"].tolist() == raw["candidate_id"].tolist(), "Public registry order changed")
    _require(len(registry) == 168, "Public registry count changed")
    return registry


def normalize_policy(policy: SearchPolicy, registry: pd.DataFrame, *, n_anchors: int) -> SearchPolicy:
    validate_policy(policy, registry)
    anchors = tuple(
        PolicyAnchor(
            candidate_id=anchor.candidate_id,
            score=1.0,
            rationale=anchor.rationale,
            evidence_ids=anchor.evidence_ids,
        )
        for anchor in policy.anchors[:n_anchors]
    )
    return SearchPolicy(
        method=policy.method,
        trial=policy.trial,
        schema_version=SCHEMA_VERSION,
        anchors=anchors,
        metadata={
            **dict(policy.metadata),
            "anchor_score_normalization": "rank_only_all_valid_anchor_scores_set_to_one",
            "requested_slots": n_anchors,
            "valid_unique_anchors": len(anchors),
            "failed_slots": n_anchors - len(anchors),
        },
    )


def compile_experiment_stream(
    policy: SearchPolicy,
    registry: pd.DataFrame,
    *,
    n_anchors: int,
) -> tuple[list[str], int]:
    normalized = normalize_policy(policy, registry, n_anchors=n_anchors)
    valid_ids = [anchor.candidate_id for anchor in normalized.anchors]
    failed_slots = n_anchors - len(valid_ids)
    sentinels = [
        f"{INVALID_SLOT_PREFIX}{normalized.method}_{normalized.trial:02d}_{slot:03d}__"
        for slot in range(len(valid_ids) + 1, n_anchors + 1)
    ]
    compiled_indices = compile_policy_order(registry, normalized)
    compiled = registry.iloc[compiled_indices]["candidate_id"].astype(str).tolist()
    seen = set(valid_ids)
    remainder = [candidate_id for candidate_id in compiled if candidate_id not in seen]
    stream = [*valid_ids, *sentinels, *remainder]
    real_ids = [value for value in stream if not value.startswith(INVALID_SLOT_PREFIX)]
    _require(len(real_ids) == 168 and len(set(real_ids)) == 168, "Compiled stream is not a complete real-candidate permutation")
    _require(len(stream) == 168 + failed_slots, "Compiled stream length does not include failed slots")
    return stream, failed_slots


def _runner_namespace(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    backend = config["model_backend"]
    return argparse.Namespace(
        model=backend["logical_model"],
        base_url=backend["gateway_base_url"],
        reasoning_effort=backend["reasoning_effort"],
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        n_anchors=config["methods"]["shared_anchor_slots"],
        batch_size=args.batch_size,
        enable_native_retrieval=False,
        prepare_only=False,
        force=False,
        brainpilot_url=args.brainpilot_url,
        brainpilot_runtime_manifest=args.brainpilot_runtime_manifest,
        methods=list(args.methods),
    )


def _trial_paths(benchmark_root: Path, method: str, trial: int) -> tuple[Path, Path, Path]:
    run_dir = benchmark_root / "runs" / method / f"trial_{trial:02d}"
    ranking_dir = benchmark_root / "rankings" / method / f"trial_{trial:02d}"
    return run_dir, ranking_dir / "ranking.csv", ranking_dir / "RANKING.lock.json"


def verify_completed_trial(benchmark_root: Path, method: str, trial: int) -> bool:
    _, ranking_path, lock_path = _trial_paths(benchmark_root, method, trial)
    if not lock_path.exists():
        return False
    lock = load_json(lock_path)
    _require(lock.get("status") == "locked_complete_ranking", f"Invalid trial lock: {lock_path}")
    _require(lock.get("method") == method and int(lock.get("trial")) == trial, f"Trial lock identity changed: {lock_path}")
    _require(sha256_file(ranking_path) == lock["ranking_sha256"], f"Completed ranking changed: {ranking_path}")
    return True


def _write_completed_trial(
    benchmark_root: Path,
    *,
    method: str,
    trial: int,
    task_path: Path,
    policy_path: Path,
    stream: Sequence[str],
    failed_slots: int,
    generation_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    _, ranking_path, lock_path = _trial_paths(benchmark_root, method, trial)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rank": range(1, len(stream) + 1), "candidate_id": list(stream)}).to_csv(ranking_path, index=False)
    artifact_records = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in generation_artifacts.items()
        if path.is_file()
    }
    lock = {
        "schema_version": "neurooracle.case2_formal_benchmark_trial_ranking_lock.v1",
        "benchmark_id": "case2_adni_formal_benchmark_v1",
        "method": method,
        "trial": trial,
        "status": "locked_complete_ranking",
        "task_sha256": sha256_file(task_path),
        "policy_sha256": sha256_file(policy_path),
        "ranking_sha256": sha256_file(ranking_path),
        "candidate_count": 168,
        "failed_generation_slots": failed_slots,
        "experiment_stream_length": len(stream),
        "association_results_accessed_by_generator": False,
        "manual_output_repair": False,
        "generation_artifacts": artifact_records,
    }
    write_json(lock_path, lock)
    return lock


def run_baseline_trial(
    benchmark_root: Path,
    registry: pd.DataFrame,
    *,
    method: str,
    trial: int,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    native_lock: threading.Lock,
) -> dict[str, Any]:
    run_dir, _, _ = _trial_paths(benchmark_root, method, trial)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_path = benchmark_root / "generator_inputs" / "tasks" / method / f"trial_{trial:02d}" / "task.json"
    task = load_json(task_path)
    local_task_path = run_dir / "task.json"
    if not local_task_path.exists():
        local_task_path.write_bytes(task_path.read_bytes())
    runner_args = _runner_namespace(args, config)
    if method in OFFICIAL_METHODS:
        policy = _run_official(
            method,
            local_task_path,
            run_dir,
            runner_args,
            LOCAL_GATEWAY_COMPATIBILITY_TOKEN,
        )
    elif method in NATIVE_METHODS:
        policy = _run_native(
            method,
            task,
            run_dir,
            registry,
            runner_args,
            LOCAL_GATEWAY_COMPATIBILITY_TOKEN,
            native_lock,
        )
    else:
        raise ValueError(f"Not a baseline method: {method}")
    policy = normalize_policy(policy, registry, n_anchors=config["methods"]["shared_anchor_slots"])
    policy_path = run_dir / "normalized_search_policy.json"
    write_json(policy_path, policy_to_payload(policy))
    stream, failed_slots = compile_experiment_stream(
        policy, registry, n_anchors=config["methods"]["shared_anchor_slots"]
    )
    artifacts = {
        "native_result": run_dir / "native_result.json",
        "search_policy": run_dir / "search_policy.json",
        "normalized_search_policy": policy_path,
        "native_compile_audit": run_dir / "native_compile_audit.json",
        "native_validation": run_dir / "native_validation.csv",
    }
    return _write_completed_trial(
        benchmark_root,
        method=method,
        trial=trial,
        task_path=task_path,
        policy_path=policy_path,
        stream=stream,
        failed_slots=failed_slots,
        generation_artifacts=artifacts,
    )


def run_neurodiscovery_trial(
    benchmark_root: Path,
    registry: pd.DataFrame,
    *,
    trial: int,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method = "neurodiscovery"
    run_dir, _, _ = _trial_paths(benchmark_root, method, trial)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_path = benchmark_root / "generator_inputs" / "tasks" / method / f"trial_{trial:02d}" / "task.json"
    hypotheses_dir = run_dir / "hypothesis_generation"
    raw_path = hypotheses_dir / "hypotheses_raw.json"
    if not raw_path.exists():
        command = [
            sys.executable,
            "-m",
            "neurooracle.src.hypothesis_cli",
            "--graph",
            config["paths"]["knowledge_graph"],
            "--generation-seed",
            str(seed),
            "case-study",
            "case2_pathway_mediation",
            "--output-dir",
            str(hypotheses_dir),
            "--stages",
            "batch",
            "--target-per-task",
            str(config["methods"]["neurodiscovery"]["generation_target_per_task"]),
        ]
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4 * 3600,
            check=False,
        )
        hypotheses_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "generator.stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_dir / "generator.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise InfrastructureFailure(f"NeuroDiscovery generator exited {result.returncode}")
    _require(raw_path.is_file(), "NeuroDiscovery produced no hypotheses_raw.json")
    hypotheses = load_hypotheses(raw_path)
    universe = registry.copy()
    universe["gene_count"] = pd.to_numeric(universe["gene_count"], errors="raise").astype(int)
    exposures = (
        universe.loc[:, ["exposure", "pathway_id", "pathway_source", "pathway_name", "threshold_label", "gene_count"]]
        .drop_duplicates("exposure")
        .rename(columns={"exposure": "score_name"})
    )
    pathway_catalog = pd.read_csv(config["paths"]["pathway_catalog"])
    ranked, support, mapping_audit = build_ranked_candidates(
        hypotheses,
        exposures,
        pathway_catalog,
        universe,
    )
    ranked["candidate_id"] = (
        ranked["exposure"].astype(str)
        + "|" + ranked["modality"].astype(str)
        + "|" + ranked["marker"].astype(str)
        + "|" + ranked["outcome"].astype(str)
    )
    _require(len(ranked) == 168 and set(ranked["candidate_id"]) == set(registry["candidate_id"]), "NeuroDiscovery mapper changed the universe")
    ranked.to_csv(run_dir / "result_blind_mapping.csv", index=False)
    support.to_csv(run_dir / "result_blind_support_audit.csv", index=False)
    mapping_audit.to_csv(run_dir / "hypothesis_mapping_audit.csv", index=False)
    anchor_ids = ranked.head(config["methods"]["shared_anchor_slots"])["candidate_id"].tolist()
    policy = SearchPolicy(
        method=method,
        trial=trial,
        schema_version=SCHEMA_VERSION,
        anchors=tuple(
            PolicyAnchor(
                candidate_id=candidate_id,
                score=1.0,
                rationale="canonical result-blind Case 2 KG mapping order",
            )
            for candidate_id in anchor_ids
        ),
        metadata={
            "generation_seed": seed,
            "kg_sha256": load_json(benchmark_root / "BENCHMARK_FREEZE.lock.json")["lock_material"]["external_input_pins"]["knowledge_graph"]["sha256"],
            "ranking_profile": config["methods"]["neurodiscovery"]["ranking_profile"],
            "feedback_enabled": False,
            "formal_results_accessed": False,
        },
    )
    policy = normalize_policy(policy, registry, n_anchors=config["methods"]["shared_anchor_slots"])
    policy_path = run_dir / "normalized_search_policy.json"
    write_json(policy_path, policy_to_payload(policy))
    stream, failed_slots = compile_experiment_stream(
        policy, registry, n_anchors=config["methods"]["shared_anchor_slots"]
    )
    _require(failed_slots == 0, "NeuroDiscovery must map 80 valid unique anchors")
    return _write_completed_trial(
        benchmark_root,
        method=method,
        trial=trial,
        task_path=task_path,
        policy_path=policy_path,
        stream=stream,
        failed_slots=failed_slots,
        generation_artifacts={
            "hypotheses_raw": raw_path,
            "result_blind_mapping": run_dir / "result_blind_mapping.csv",
            "result_blind_support_audit": run_dir / "result_blind_support_audit.csv",
            "hypothesis_mapping_audit": run_dir / "hypothesis_mapping_audit.csv",
            "normalized_search_policy": policy_path,
        },
    )


def finalize_rankings(benchmark_root: Path) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        for trial in range(10):
            if not verify_completed_trial(benchmark_root, method, trial):
                return None
            _, ranking_path, trial_lock_path = _trial_paths(benchmark_root, method, trial)
            trial_lock = load_json(trial_lock_path)
            records.append(
                {
                    "method": method,
                    "trial": trial,
                    "relative_path": ranking_path.relative_to(benchmark_root).as_posix(),
                    "bytes": ranking_path.stat().st_size,
                    "sha256": sha256_file(ranking_path),
                    "trial_lock_relative_path": trial_lock_path.relative_to(benchmark_root).as_posix(),
                    "trial_lock_sha256": sha256_file(trial_lock_path),
                    "failed_generation_slots": trial_lock["failed_generation_slots"],
                    "experiment_stream_length": trial_lock["experiment_stream_length"],
                }
            )
    payload = {
        "schema_version": "neurooracle.case2_formal_benchmark_rankings_lock.v1",
        "benchmark_id": "case2_adni_formal_benchmark_v1",
        "status": "locked_complete_70_rankings",
        "ranking_count": 70,
        "methods": list(EXPECTED_METHODS),
        "trials": list(range(10)),
        "manual_output_repair": False,
        "rankings": records,
        "ranking_set_id": hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    lock_path = benchmark_root / "rankings" / "RANKINGS.lock.json"
    if lock_path.exists():
        _require(load_json(lock_path) == payload, "Existing complete ranking lock differs")
    else:
        write_json(lock_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=EXPECTED_METHODS, default=list(EXPECTED_METHODS))
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--fast-large-files", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--brainpilot-url", default="http://127.0.0.1:18080/api")
    parser.add_argument("--brainpilot-runtime-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require(set(args.trials) <= set(range(10)), "Trial indices must be 0-9")
    benchmark_root = args.benchmark_root.resolve()
    verification = verify_freeze(benchmark_root, fast_large_files=args.fast_large_files)
    config = load_json(benchmark_root / "PROTOCOL.json")
    selected_baselines = [method for method in args.methods if method != "neurodiscovery"]
    runtime_preflight: dict[str, Any] = {"freeze": verification["status"]}
    if selected_baselines:
        runtime_preflight["deepseek_gateway"] = gateway_preflight(config["model_backend"]["gateway_base_url"])
    if "brainpilot_native" in args.methods:
        runtime_preflight["brainpilot"] = brainpilot_preflight(
            args.brainpilot_url, args.brainpilot_runtime_manifest, config
        )
    write_json(benchmark_root / "runs" / "EXECUTION_PREFLIGHT.json", runtime_preflight)
    registry = load_public_registry(benchmark_root)
    native_lock = threading.Lock()
    failures: list[dict[str, Any]] = []
    completed = 0
    for method in args.methods:
        for trial in args.trials:
            if verify_completed_trial(benchmark_root, method, trial):
                completed += 1
                continue
            try:
                if method == "neurodiscovery":
                    run_neurodiscovery_trial(
                        benchmark_root,
                        registry,
                        trial=trial,
                        seed=config["trials"]["seeds"][trial],
                        config=config,
                    )
                else:
                    run_baseline_trial(
                        benchmark_root,
                        registry,
                        method=method,
                        trial=trial,
                        config=config,
                        args=args,
                        native_lock=native_lock,
                    )
                completed += 1
            except Exception as exc:
                failures.append({"method": method, "trial": trial, "error": _safe_error(exc)})
                write_json(benchmark_root / "runs" / "EXECUTION_FAILURES.json", {"failures": failures})
                raise
    write_json(benchmark_root / "runs" / "EXECUTION_FAILURES.json", {"failures": failures})
    ranking_lock = finalize_rankings(benchmark_root)
    result = {
        "benchmark_id": config["benchmark_id"],
        "selected_rankings_completed_or_verified": completed,
        "all_70_rankings_locked": ranking_lock is not None,
        "api_backend": config["model_backend"]["logical_model"],
        "official_deepseek_automatic_fallback": False,
        "failures": len(failures),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
