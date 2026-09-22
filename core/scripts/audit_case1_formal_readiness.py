"""Audit the resumable Case Study 1 formal experiment without calling an LLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
DEFAULT_RUN_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison"
    r"\20260826_kg2c0273_deepseekv4pro_true_closed_loop_3seeds_v1"
)
EXPECTED_KG_SHA256 = "2C02732582DA9907C68300D791C3560B8E66CC268D987D453E7C971BD6CDEFF5"
EXPECTED_CLAIMS_SHA256 = "705B079989FDEC3D7756CF737F12B41756EB8058F8ED66A009D6F832489F5BA4"
EXPECTED_STATE_SHA256 = "744B75718B2BEEBFDAF9055595CB2161ED841643ACD58F365F55717BC7B5360E"
EXPECTED_REGISTRY_SHA256 = "5E7E422FC767E088FAD46501048C60F2AA043BB6AA00904D3F2DC1E21CCD6502"
EXPECTED_CANDIDATES = 426_555
EXPECTED_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
    "neurodiscovery",
)
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
SEEDS = (0, 1, 2)
NATIVE_BATCHES = ((1, 20), (21, 40), (41, 60), (61, 80))
PROMPT_PATTERN = re.compile(
    r"^(brainpilot_native|biomni_native)[\\/]seed_(\d{2})[\\/]"
    r"batch_(\d{3})_(\d{3})[\\/]prompt\.txt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-runtime-preflight", action="store_true")
    parser.add_argument("--verified-test-count", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_audit(run_root: Path) -> dict[str, Any]:
    path = run_root / "canonical_kg_release.json"
    payload = load_json(path)
    files = payload.get("files") or {}
    observed = {
        "knowledge_graph": str((files.get("knowledge_graph") or {}).get("sha256") or "").upper(),
        "extracted_claims": str((files.get("extracted_claims") or {}).get("sha256") or "").upper(),
        "current_state": str((files.get("current_state") or {}).get("sha256") or "").upper(),
    }
    expected = {
        "knowledge_graph": EXPECTED_KG_SHA256,
        "extracted_claims": EXPECTED_CLAIMS_SHA256,
        "current_state": EXPECTED_STATE_SHA256,
    }
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(path),
        "release_id": payload.get("release_id"),
        "status": payload.get("status"),
        "observed_sha256": observed,
        "expected_sha256": expected,
        "matches_expected": observed == expected and payload.get("status") == "canonical_current",
    }


def registry_audit(run_root: Path) -> dict[str, Any]:
    paths = {
        "formal_official": run_root
        / "baseline_generation"
        / "formal_official"
        / "cs1_public_registry.jsonl",
        "native": run_root
        / "baseline_generation"
        / "native"
        / "cs1_public_registry.jsonl",
    }
    rows: dict[str, Any] = {}
    for label, path in paths.items():
        with path.open("r", encoding="utf-8") as handle:
            first = json.loads(next(handle))
            count = 1 + sum(1 for line in handle if line.strip())
        keys = sorted(first)
        forbidden = sorted(
            set(keys)
            & {
                "is_gt_top",
                "is_strict_fdr",
                "abs_adjusted_residual_d",
                "effect_size",
                "p_value",
                "fdr",
                "neurodiscovery_score",
            }
        )
        rows[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "candidate_count": count,
            "first_row_fields": keys,
            "forbidden_outcome_fields": forbidden,
        }
    hashes = {row["sha256"] for row in rows.values()}
    return {
        "registries": rows,
        "same_sha256": len(hashes) == 1,
        "matches_expected_sha256": hashes == {EXPECTED_REGISTRY_SHA256},
        "candidate_counts_match": all(
            row["candidate_count"] == EXPECTED_CANDIDATES for row in rows.values()
        ),
        "outcome_fields_absent": all(
            not row["forbidden_outcome_fields"] for row in rows.values()
        ),
    }


def native_prompt_audit(run_root: Path) -> dict[str, Any]:
    native_root = run_root / "baseline_generation" / "native"
    records: list[dict[str, Any]] = []
    observed: set[tuple[str, int, int, int]] = set()
    for path in sorted(native_root.rglob("prompt.txt")):
        relative = path.relative_to(native_root).as_posix()
        match = PROMPT_PATTERN.match(relative)
        if not match:
            records.append({"relative_path": relative, "valid_path": False})
            continue
        method, seed_text, start_text, end_text = match.groups()
        seed, start, end = int(seed_text), int(start_text), int(end_text)
        text = path.read_text(encoding="utf-8")
        checks = {
            "seed_bound": f"Seed: {seed}" in text,
            "batch_bound": f"Batch ranks: {start}-{end}" in text,
            "outcome_blinding_declared": (
                "You are not given GT, effect sizes, p-values, FDR values" in text
                and "NeuroDiscovery scores, or closed-loop feedback" in text
            ),
            "exact_candidate_contract": (
                "candidate_id must be an exact executable ID" in text
            ),
            "no_repair_contract": (
                "Invalid, duplicate, missing, or unmappable outputs are failures" in text
            ),
        }
        observed.add((method, seed, start, end))
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "valid_path": True,
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            }
        )
    expected = {
        (method, seed, start, end)
        for method in NATIVE_METHODS
        for seed in SEEDS
        for start, end in NATIVE_BATCHES
    }
    unexpected_outputs = sorted(
        str(path.relative_to(native_root))
        for pattern in ("final.txt", "mapped_hypotheses.csv", "native_result.json")
        for path in native_root.rglob(pattern)
    )
    return {
        "expected_prompt_count": len(expected),
        "observed_prompt_count": len(records),
        "coverage_complete": observed == expected,
        "missing_batches": sorted(
            f"{method}/seed_{seed:02d}/batch_{start:03d}_{end:03d}"
            for method, seed, start, end in expected - observed
        ),
        "unexpected_batches": sorted(
            f"{method}/seed_{seed:02d}/batch_{start:03d}_{end:03d}"
            for method, seed, start, end in observed - expected
        ),
        "all_prompt_contracts_pass": all(
            row.get("valid_path") and row.get("all_checks_pass") for row in records
        ),
        "unexpected_result_artifacts": unexpected_outputs,
        "records": records,
    }


def official_runtime_preflight(skip: bool) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "all_ok": None, "methods": {}}
    from core.scripts.case1_official_baseline_experiment import (
        PRIMARY_METHODS,
        repository_preflight,
    )

    rows = {method: repository_preflight(method) for method in PRIMARY_METHODS}
    return {
        "skipped": False,
        "all_ok": all(row.get("status") == "ok" for row in rows.values()),
        "methods": rows,
    }


def local_client_preflight(run_root: Path) -> dict[str, Any]:
    biomni_root = BASELINES_ROOT / "Biomni"
    biomni_python = biomni_root / ".venv" / "Scripts" / "python.exe"
    biomni = subprocess.run(
        [str(biomni_python), "-c", "from biomni.agent import A1"],
        cwd=biomni_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    brainpilot_dist = (
        BASELINES_ROOT / "BrainPilot" / "packages" / "client-cli" / "dist" / "index.js"
    )
    node = subprocess.run(
        ["node", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    runtime_manifest_path = run_root / "runtime" / "brainpilot_seed0" / "runtime_manifest.json"
    runtime_manifest = load_json(runtime_manifest_path)
    health: list[dict[str, Any]] = []
    for base_url in runtime_manifest.get("base_urls") or []:
        health_url = str(base_url).rstrip("/") + "/health"
        try:
            with urlopen(health_url, timeout=2) as response:
                status = int(response.status)
            health.append({"url": health_url, "ok": status < 400, "status": status})
        except Exception as exc:
            health.append(
                {
                    "url": health_url,
                    "ok": False,
                    "error_type": type(exc).__name__,
                }
            )
    return {
        "biomni": {
            "python": str(biomni_python),
            "import_ok": biomni.returncode == 0,
            "returncode": biomni.returncode,
        },
        "brainpilot": {
            "node_version": node.stdout.strip(),
            "node_ok": node.returncode == 0,
            "client_dist": str(brainpilot_dist),
            "client_dist_exists": brainpilot_dist.is_file(),
            "runtime_manifest": str(runtime_manifest_path),
            "model_matches": runtime_manifest.get("model") == "deepseek-v4-pro",
            "reasoning_effort_matches": runtime_manifest.get("reasoning_effort") == "high",
            "health": health,
            "service_live": bool(health) and all(row["ok"] for row in health),
            "restart_required_before_brainpilot_run": not health
            or not all(row["ok"] for row in health),
        },
    }


def score_components_audit(run_root: Path) -> dict[str, Any]:
    path = run_root / "score_components" / "score_components.manifest.json"
    payload = load_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "candidate_count": payload.get("candidate_count"),
        "outcome_blind": payload.get("outcome_blind"),
        "frozen_before_experiment": payload.get("frozen_before_experiment"),
        "canonical_kg_release": payload.get("canonical_kg_release"),
        "ready": (
            payload.get("candidate_count") == EXPECTED_CANDIDATES
            and payload.get("outcome_blind") is True
            and payload.get("frozen_before_experiment") is True
        ),
    }


def formal_progress_audit(run_root: Path) -> dict[str, Any]:
    formal_root = run_root / "baseline_generation" / "formal_official"
    ai_path = formal_root / "ai_scientist_v2" / "trial_00" / "search_policy.json"
    ai = load_json(ai_path)
    anchors = [str(row.get("candidate_id") or "") for row in ai.get("anchors") or []]
    cache_path = (
        formal_root
        / "open_coscientist"
        / "trial_00"
        / "open_coscientist_cache_manifest.json"
    )
    cache = load_json(cache_path)
    registry_sha = sha256_file(formal_root / "cs1_public_registry.jsonl")
    cache_inputs = (cache.get("identity") or {}).get("sealed_inputs") or {}
    cache_kg = cache_inputs.get("canonical_kg_release") or {}
    cache_binding_ok = (
        str(cache_inputs.get("public_registry_sha256") or "").upper() == registry_sha
        and str(cache_kg.get("knowledge_graph_sha256") or "").upper()
        == EXPECTED_KG_SHA256
        and ((cache.get("identity") or {}).get("cache_policy") or {}).get(
            "node_cache_enabled"
        )
        is False
    )
    scoreable_paths = [
        formal_root / "ai_scientist_v2" / "trial_00" / "search_policy.json",
        *(
            formal_root / method / f"trial_{seed:02d}" / "search_policy.json"
            for method in ("ai_scientist_v2", "open_coscientist", "sciagents", "virtual_lab")
            for seed in SEEDS
            if not (method == "ai_scientist_v2" and seed == 0)
        ),
        *(
            run_root
            / "baseline_generation"
            / "native"
            / method
            / f"seed_{seed:02d}"
            / "mapped_hypotheses.csv"
            for method in NATIVE_METHODS
            for seed in SEEDS
        ),
    ]
    existing = [str(path) for path in scoreable_paths if path.is_file()]
    return {
        "ai_scientist_v2_seed0": {
            "path": str(ai_path),
            "method": ai.get("method"),
            "trial": ai.get("trial"),
            "anchor_count": len(anchors),
            "unique_anchor_count": len(set(anchors)),
            "scoreable": len(anchors) == 80 and len(set(anchors)) == 80,
        },
        "open_coscientist_seed0_cache": {
            "path": str(cache_path),
            "status": cache.get("status"),
            "open_count": cache.get("open_count"),
            "cache_files": (cache.get("stats_at_close") or {}).get("cache_files"),
            "identity_sha256": cache.get("identity_sha256"),
            "binding_ok": cache_binding_ok,
        },
        "scoreable_artifacts": existing,
        "scoreable_method_seeds": len(existing),
        "required_method_seeds": len(EXPECTED_METHODS) * len(SEEDS),
        "progress_fraction": len(existing) / (len(EXPECTED_METHODS) * len(SEEDS)),
    }


def latest_gateway_checkpoint(run_root: Path) -> dict[str, Any] | None:
    checkpoint_dirs = (
        run_root / "runtime" / "deepseek_v4_pro_gateway" / "checkpoints",
        run_root / "runtime" / "gateway_checkpoints",
    )
    paths = sorted(
        (
            path
            for checkpoint_dir in checkpoint_dirs
            for path in checkpoint_dir.glob("*.json")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not paths:
        return None
    path = paths[0]
    payload = load_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "attempt_count": len(payload.get("attempts") or []),
        "secrets_persisted": payload.get("secrets_persisted"),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_root / "runtime" / "formal_baseline_readiness_audit.json"
    )
    canonical = canonical_audit(run_root)
    registry = registry_audit(run_root)
    prompts = native_prompt_audit(run_root)
    official = official_runtime_preflight(args.skip_runtime_preflight)
    clients = local_client_preflight(run_root)
    scores = score_components_audit(run_root)
    progress = formal_progress_audit(run_root)
    gateway = latest_gateway_checkpoint(run_root)
    offline_checks = {
        "canonical_kg_matches": canonical["matches_expected"],
        "registries_match": (
            registry["same_sha256"]
            and registry["matches_expected_sha256"]
            and registry["candidate_counts_match"]
            and registry["outcome_fields_absent"]
        ),
        "native_prompts_complete_and_blinded": (
            prompts["coverage_complete"]
            and prompts["all_prompt_contracts_pass"]
            and not prompts["unexpected_result_artifacts"]
        ),
        "official_runtimes_ok": official["all_ok"] is True,
        "biomni_runtime_ok": clients["biomni"]["import_ok"],
        "brainpilot_client_ok": (
            clients["brainpilot"]["node_ok"]
            and clients["brainpilot"]["client_dist_exists"]
            and clients["brainpilot"]["model_matches"]
            and clients["brainpilot"]["reasoning_effort_matches"]
        ),
        "score_components_frozen_and_blinded": scores["ready"],
        "ai_scientist_seed0_scoreable": progress["ai_scientist_v2_seed0"][
            "scoreable"
        ],
        "open_coscientist_cache_binding_ok": progress[
            "open_coscientist_seed0_cache"
        ]["binding_ok"],
    }
    quota_ready = bool(gateway) and gateway.get("status") == "complete"
    payload = {
        "schema_version": "case1.formal-baseline-readiness-audit.v1",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "experiment": {
            "methods": list(EXPECTED_METHODS),
            "seeds": list(SEEDS),
            "model": "deepseek-v4-pro",
            "reasoning_effort": "high",
            "response_storage": False,
            "official_deepseek_automatic_fallback": False,
        },
        "canonical_kg": canonical,
        "public_registry": registry,
        "native_prompts": prompts,
        "official_runtime_preflight": official,
        "local_client_preflight": clients,
        "score_components": scores,
        "formal_progress": progress,
        "latest_gateway_checkpoint": gateway,
        "offline_checks": offline_checks,
        "offline_ready": all(offline_checks.values()),
        "brainpilot_service_live": clients["brainpilot"]["service_live"],
        "api_quota_ready": quota_ready,
        "ready_to_resume_after_account_switch": all(offline_checks.values()),
        "verified_offline_test_count": max(0, int(args.verified_test_count)),
        "secrets_read": False,
        "secrets_persisted": False,
    }
    write_json_atomic(output, payload)
    print(output)


if __name__ == "__main__":
    main()
