"""Freeze every non-generative input for the current formal Case Study 2 benchmark.

This command performs no model API call.  It creates a result-isolated generator
bundle and a separate evaluator-only reference bundle, pins the canonical KG and
all source/runtime state, and fails closed if the target already exists.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = (
    REPO_ROOT / "neurooracle" / "configs" / "case2_adni_formal_benchmark_v1.json"
)
PUBLIC_COLUMNS = (
    "candidate_id",
    "exposure",
    "pathway_id",
    "pathway_name",
    "pathway_source",
    "threshold_label",
    "gene_count",
    "modality",
    "marker",
    "outcome",
)
EXPECTED_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
    "neurodiscovery",
)
OFFICIAL_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
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
SENSITIVE_GENERATOR_KEYS = frozenset(
    {
        "complete_case_n",
        "n",
        "a_path_std",
        "a_path_hc3_p",
        "b_path_std",
        "b_path_hc3_p",
        "indirect_effect_std",
        "indirect_bootstrap_p",
        "indirect_bootstrap_q_global_168",
        "indirect_bootstrap_q_family_8",
        "result_tier",
    }
)
LARGE_FILE_BYTES = 1_000_000_000
HKT = timezone(timedelta(hours=8))


def hkt_now() -> str:
    return datetime.now(HKT).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, *, announce_large: bool = False) -> str:
    size = path.stat().st_size
    if announce_large and size >= LARGE_FILE_BYTES:
        print(f"Hashing frozen large input: {path.name} ({size} bytes)", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_pin(path: Path, *, announce_large: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path, announce_large=announce_large),
    }


def derive_trial_seed(benchmark_id: str, trial: int) -> int:
    digest = hashlib.sha256(f"{benchmark_id}:trial:{trial}".encode("utf-8")).digest()
    return int(struct.unpack("<I", digest[:4])[0])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "neurooracle.case2_formal_generator_benchmark.v1",
        "Unexpected benchmark schema",
    )
    _require(config.get("source_protocol_id") == "case2_adni_formal_supplementary_v2", "Wrong source protocol")
    _require(config["data"]["source_subject_count"] == 691, "Source cohort must remain 691")
    _require(config["data"]["outcome_blind_eligible_subject_count"] == 683, "Eligible cohort must remain 683")
    _require(config["data"]["candidate_count"] == 168, "Candidate count must remain 168")
    methods = tuple(config["methods"]["baselines"]) + (config["methods"]["target"],)
    _require(methods == EXPECTED_METHODS, f"Method order differs from {EXPECTED_METHODS}")
    _require(config["methods"]["shared_anchor_slots"] == 80, "Anchor budget must be 80")
    _require(config["methods"]["compiled_ranking_size"] == 168, "Ranking size must be 168")
    _require(config["methods"]["native_retrieval_enabled"] is False, "Live native retrieval must remain disabled")
    _require(config["model_backend"]["logical_model"] == "deepseek-v4-pro", "Model must be DeepSeek V4 Pro")
    _require(config["model_backend"]["reasoning_effort"] == "high", "Reasoning must be high")
    _require(float(config["model_backend"]["temperature"]) == 0.0, "Temperature must be zero")
    _require(config["model_backend"]["deepseek_official_automatic_fallback"] is False, "Official DeepSeek cannot be automatic fallback")
    _require(config["model_backend"]["max_concurrent_api_requests"] == 1, "API concurrency must remain one")
    _require(config["trials"]["count"] == 10, "Exactly ten trials are required")
    expected_seeds = [derive_trial_seed(config["benchmark_id"], trial) for trial in range(10)]
    _require(config["trials"]["seeds"] == expected_seeds, "Trial seed derivation or values changed")
    _require(config["evaluation"]["budgets"] == [5, 10, 20, 50, 100, 168], "Unexpected budgets")
    _require(
        config["evaluation"]["primary_metric"]["name"]
        == "bootstrap_weakest_link_evidence_ndcg",
        "Unexpected primary metric",
    )
    _require(config["data"]["external_validation_required"] is False, "No external validation queue is allowed")
    _require(
        config["design_status"]["formal_results_were_accessed_before_this_benchmark_was_designed"] is True,
        "Post-result design status must be explicit",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_public_registry(master_path: Path, score_manifest_path: Path) -> list[dict[str, str]]:
    master_columns, master = _read_csv(master_path)
    required_master = {
        "candidate_id",
        "score_name",
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "modality",
        "marker",
        "outcome",
    }
    _require(required_master <= set(master_columns), "Master registry schema changed")
    score_columns, scores = _read_csv(score_manifest_path)
    _require(
        {"score_name", "pathway_source", "gene_count"} <= set(score_columns),
        "Score manifest schema changed",
    )
    score_by_name = {row["score_name"].strip(): row for row in scores}
    public: list[dict[str, str]] = []
    for row in master:
        exposure = row["score_name"].strip()
        score = score_by_name.get(exposure)
        if score is None:
            raise ValueError(f"Missing score metadata for {exposure}")
        candidate_id = "|".join(
            (exposure, row["modality"].strip(), row["marker"].strip(), row["outcome"].strip())
        )
        _require(candidate_id == row["candidate_id"].strip(), f"Candidate ID mismatch: {candidate_id}")
        public.append(
            {
                "candidate_id": candidate_id,
                "exposure": exposure,
                "pathway_id": row["pathway_id"].strip(),
                "pathway_name": row["pathway_name"].strip(),
                "pathway_source": score["pathway_source"].strip(),
                "threshold_label": row["threshold_label"].strip(),
                "gene_count": score["gene_count"].strip(),
                "modality": row["modality"].strip(),
                "marker": row["marker"].strip(),
                "outcome": row["outcome"].strip(),
            }
        )
    public.sort(key=lambda row: row["candidate_id"])
    ids = [row["candidate_id"] for row in public]
    _require(len(public) == 168, f"Expected 168 candidates, got {len(public)}")
    _require(len(set(ids)) == len(ids), "Public candidate IDs are not unique")
    _require(tuple(public[0]) == PUBLIC_COLUMNS, "Public registry column order changed")
    return public


def _finite_probability(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"Invalid probability {field}={row[field]!r}")
    return value


def build_evaluator_reference(
    formal_results_path: Path,
    candidate_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns, rows = _read_csv(formal_results_path)
    required = {
        "candidate_id",
        "a_path_hc3_p",
        "b_path_hc3_p",
        "indirect_bootstrap_p",
        "indirect_bootstrap_q_family_8",
        "indirect_bootstrap_q_global_168",
        "analysis_status",
    }
    _require(required <= set(columns), "Formal result schema changed")
    _require(len(rows) == 168, f"Formal result count changed: {len(rows)}")
    by_id = {row["candidate_id"].strip(): row for row in rows}
    _require(set(by_id) == set(candidate_ids), "Formal result candidates differ from public registry")
    reference: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        row = by_id[candidate_id]
        _require(row["analysis_status"].strip() == "estimated", f"Non-estimated candidate: {candidate_id}")
        a_p = _finite_probability(row, "a_path_hc3_p")
        b_p = _finite_probability(row, "b_path_hc3_p")
        indirect_p = _finite_probability(row, "indirect_bootstrap_p")
        family_q = _finite_probability(row, "indirect_bootstrap_q_family_8")
        global_q = _finite_probability(row, "indirect_bootstrap_q_global_168")
        weakest = max(a_p, b_p, indirect_p)
        relevance = min(50.0, -math.log10(max(weakest, 1e-300)))
        reference.append(
            {
                "candidate_id": candidate_id,
                "bootstrap_weakest_link_evidence": format(relevance, ".17g"),
                "supplemental_family_fdr_hit": str(family_q < 0.05).lower(),
                "global_fdr_hit": str(global_q < 0.05).lower(),
                "nominal_bootstrap_hit": str(indirect_p < 0.05).lower(),
            }
        )
    counts = {
        "nominal_bootstrap_hits": sum(row["nominal_bootstrap_hit"] == "true" for row in reference),
        "supplemental_family_fdr_hits": sum(row["supplemental_family_fdr_hit"] == "true" for row in reference),
        "global_fdr_hits": sum(row["global_fdr_hit"] == "true" for row in reference),
    }
    _require(counts == {"nominal_bootstrap_hits": 15, "supplemental_family_fdr_hits": 6, "global_fdr_hits": 0}, f"Locked formal label counts changed: {counts}")
    return reference, counts


def _registry_lines(public: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(
        f"- {row['candidate_id']} :: {row['pathway_name']}; {row['modality']} {row['marker']}; {row['outcome']}"
        for row in public
    )


def build_task(
    public: Sequence[Mapping[str, str]],
    *,
    method: str,
    trial: int,
    seed: int,
    n_anchors: int,
    final_registry_path: Path,
) -> dict[str, Any]:
    goal = f"""Blinded Case Study 2 hypothesis prioritization.

Across this trial, prioritize exactly {n_anchors} executable pathway-PRS -> imaging-marker -> future-clinical-status hypotheses. The scientific question is which inherited pathway burdens may be associated with later clinical status through a measured imaging marker. This is an association screen, not a causal claim.

Use trial seed {seed} only as the registered replicate identifier. You are not given and must not infer local experimental coefficients, probability values, multiplicity-adjusted labels, complete-case sizes, NeuroDiscovery scores, or evaluator labels. Do not access files outside the supplied generator bundle. Live retrieval is disabled.

Only the following exact candidate_id values are executable. Copy IDs verbatim; an invalid, duplicate, or missing proposal consumes its slot and receives zero anchor credit.
{_registry_lines(public)}

Rank candidates by biological plausibility, specificity of the pathway-marker-outcome chain, likely longitudinal testability, and portfolio diversity. Every final item must contain one exact candidate_id and a concise rationale."""
    return {
        "case_study_id": "case2",
        "benchmark_id": "case2_adni_formal_benchmark_v1",
        "policy_schema_version": "case2-search-policy-v1",
        "candidate_id_template": "exposure|modality|marker|outcome",
        "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
        "coordinate_constraint": "Use only registered pathway PRS, imaging marker, and future outcome coordinates.",
        "ontology_path_fields": ["pathway_name", "modality", "marker", "outcome"],
        "lead_expertise": "imaging genetics and longitudinal research prioritization",
        "imaging_expertise": "amyloid, tau, FDG PET and structural MRI biomarkers",
        "biological_expertise": "Alzheimer disease genetics and clinical progression",
        "sciagents_planner_prompt": "Plan a blinded graph search over pathway PRS, imaging marker, and future clinical outcome coordinates. Do not propose final answers.",
        "sciagents_ontologist_prompt": "Organize pathway, molecular process, imaging biomarker, and longitudinal clinical outcome relations without hidden experiment results.",
        "method": method,
        "trial": trial,
        "generation_seed": seed,
        "n_anchors": n_anchors,
        "public_registry_path": str(final_registry_path),
        "native_retrieval_enabled": False,
        "research_goal": goal,
    }


def _json_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_keys(child)


def audit_generator_bundle(generator_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sensitive_paths = {
        str(Path(config["paths"]["formal_results"])).casefold(),
        str(Path(config["paths"]["formal_results_lock"])).casefold(),
    }
    findings: list[str] = []
    files = sorted(path for path in generator_root.rglob("*") if path.is_file())
    for path in files:
        if path.suffix.lower() == ".csv":
            columns, _ = _read_csv(path)
            overlap = SENSITIVE_GENERATOR_KEYS & {column.casefold() for column in columns}
            if overlap:
                findings.append(f"sensitive_csv_columns:{path.name}:{sorted(overlap)}")
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            overlap = SENSITIVE_GENERATOR_KEYS & {key.casefold() for key in _json_keys(payload)}
            if overlap:
                findings.append(f"sensitive_json_keys:{path.name}:{sorted(overlap)}")
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        for sensitive in sensitive_paths:
            if sensitive and sensitive in text:
                findings.append(f"evaluator_path_leak:{path.name}")
    return {
        "status": "passed" if not findings else "failed",
        "files_scanned": len(files),
        "sensitive_key_count": len(SENSITIVE_GENERATOR_KEYS),
        "findings": findings,
    }


def _run(command: Sequence[str], *, cwd: Path, timeout: int = 120, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=not binary,
        timeout=timeout, check=False,
    )


def _repository_python(repo: Path) -> Path | None:
    candidates = (
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "python.exe",
    )
    return next((path for path in candidates if path.is_file()), None)


def _excluded_repo_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    parts = {part.casefold() for part in normalized.split("/")}
    if parts & {".git", ".venv", "venv", "node_modules", ".cache", "__pycache__", "runs", "outputs"}:
        return True
    name = Path(normalized).name.casefold()
    return name.startswith(".env") or name in {"keys.txt", "credentials.json", "secrets.json"}


def pin_repository(repo: Path) -> dict[str, Any]:
    _require((repo / ".git").exists(), f"Baseline repository is not a Git worktree: {repo}")
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    _require(head.returncode == 0, f"Cannot read baseline HEAD: {repo}")
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"], cwd=repo, binary=True)
    diff = _run(["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], cwd=repo, timeout=300, binary=True)
    dirty_names = _run(["git", "diff", "--name-only", "-z", "HEAD", "--"], cwd=repo, binary=True)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo, binary=True)
    for result, label in ((status, "status"), (diff, "diff"), (dirty_names, "dirty names"), (untracked, "untracked names")):
        _require(result.returncode == 0, f"Cannot read baseline {label}: {repo}")
    current_paths: list[str] = []
    excluded_count = 0
    raw_paths = [
        value.decode("utf-8", "surrogateescape")
        for payload in (dirty_names.stdout, untracked.stdout)
        for value in payload.split(b"\0")
        if value
    ]
    for relative in sorted(set(raw_paths)):
        if _excluded_repo_path(relative):
            excluded_count += 1
        else:
            current_paths.append(relative)
    current_files: list[dict[str, Any]] = []
    for relative in current_paths:
        path = repo / relative
        if path.is_file():
            current_files.append({"relative_path": relative, **{key: value for key, value in file_pin(path).items() if key != "path"}})
        elif not path.exists():
            current_files.append({"relative_path": relative, "state": "deleted"})
    return {
        "repository": str(repo),
        "head": head.stdout.strip(),
        "git_status_sha256": hashlib.sha256(status.stdout).hexdigest(),
        "git_status_bytes": len(status.stdout),
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "tracked_diff_bytes": len(diff.stdout),
        "dirty_or_untracked_source_files": current_files,
        "excluded_untracked_or_generated_path_count": excluded_count,
        "exclusion_rule": "Caches, virtual environments, node_modules, runs, outputs, and secret-shaped local files are not execution inputs.",
    }


def runtime_preflight(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline_root = Path(config["paths"]["baseline_repository_root"])
    repo_map = config["methods"]["repository_map"]
    rows: list[dict[str, Any]] = []
    for method in OFFICIAL_METHODS:
        repo = baseline_root / repo_map[method]
        runtime = _repository_python(repo)
        if runtime is None:
            rows.append({"method": method, "status": "failed", "reason": "python_runtime_missing"})
            continue
        probe = _run([str(runtime), "-c", IMPORT_PROBES[method]], cwd=repo, timeout=180)
        version = _run([str(runtime), "--version"], cwd=repo)
        rows.append(
            {
                "method": method,
                "status": "ok" if probe.returncode == 0 else "failed",
                "runtime": str(runtime),
                "runtime_version": (version.stdout or version.stderr).strip(),
                "import_probe_returncode": probe.returncode,
            }
        )
    biomni_repo = baseline_root / repo_map["biomni_native"]
    biomni_runtime = _repository_python(biomni_repo)
    biomni_probe = (
        _run([str(biomni_runtime), "-c", "from biomni.agent import A1"], cwd=biomni_repo, timeout=180)
        if biomni_runtime is not None
        else None
    )
    rows.append(
        {
            "method": "biomni_native",
            "status": "ok" if biomni_probe is not None and biomni_probe.returncode == 0 else "failed",
            "runtime": str(biomni_runtime) if biomni_runtime else "",
            "import_probe_returncode": biomni_probe.returncode if biomni_probe is not None else None,
        }
    )
    brainpilot_repo = baseline_root / repo_map["brainpilot_native"]
    client_dist = brainpilot_repo / "packages" / "client-cli" / "dist" / "index.js"
    root_client = REPO_ROOT / "core" / "scripts" / "brainpilot_case_study_batch_client.mjs"
    node = shutil.which("node")
    node_version = _run([node, "--version"], cwd=brainpilot_repo) if node else None
    rows.append(
        {
            "method": "brainpilot_native",
            "status": "ok" if node and client_dist.is_file() and root_client.is_file() and node_version and node_version.returncode == 0 else "failed",
            "node": node or "",
            "node_version": node_version.stdout.strip() if node_version else "",
            "client_dist_present": client_dist.is_file(),
            "batch_client_present": root_client.is_file(),
            "execution_time_requirement": "BrainPilot local service must pass /health immediately before its trials run.",
        }
    )
    rows.append(
        {
            "method": "neurodiscovery",
            "status": "ok",
            "runtime": sys.executable,
            "execution_time_requirement": "Use the benchmark environment and the hash-pinned canonical KG.",
        }
    )
    return rows


def key_count_preflight(keys_path: Path) -> dict[str, Any]:
    from neurooracle.scripts.probe_deepseek_v4_pro_channels import load_channel_keys

    keys = load_channel_keys(keys_path)
    counts = {name: len(values) for name, values in keys.items()}
    _require(counts == {"opencode": 3, "ollama": 1, "deepseek": 1}, f"Unexpected channel key counts: {counts}")
    return {
        "status": "passed",
        "loaded_key_counts": counts,
        "secret_values_persisted": False,
        "secret_fingerprints_persisted": False,
        "network_requests_made": False,
    }


def benchmark_environment_preflight() -> dict[str, Any]:
    expected = {
        "numpy": "2.3.5",
        "pandas": "2.3.3",
        "scipy": "1.16.3",
        "statsmodels": "0.14.5",
        "pyarrow": "21.0.0",
        "requests": "2.32.5",
        "threadpoolctl": "3.5.0",
        "pytest": "8.4.2",
        "openai": "2.51.0",
    }
    observed = {name: importlib.metadata.version(name) for name in expected}
    python_version = ".".join(map(str, sys.version_info[:3]))
    _require(python_version == "3.13.9", f"Benchmark Python changed: {python_version}")
    _require(observed == expected, f"Benchmark package versions changed: {observed}")
    return {
        "status": "passed",
        "python": python_version,
        "python_executable": sys.executable,
        "packages": observed,
    }


def validate_source_locks(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    readiness = load_json(Path(paths["readiness_lock"]))
    summary = load_json(Path(paths["readiness_summary"]))
    result_lock = load_json(Path(paths["formal_results_lock"]))
    _require(readiness.get("status") == "locked_ready_for_formal_execution", "Readiness lock is not ready")
    _require(summary.get("status") == "locked_ready_for_formal_execution", "Readiness summary is not ready")
    _require(readiness.get("protocol_id") == config["source_protocol_id"], "Readiness protocol changed")
    _require(summary["cohort"]["expected_source_subjects"] == 691, "Readiness source N changed")
    _require(summary["cohort"]["combined_pca_eligible_subjects"] == 683, "Readiness eligible N changed")
    _require(summary["registry"]["master_candidates"] == 168, "Readiness candidate count changed")
    _require(summary["registry"]["executable_candidates"] == 168, "Not all candidates are executable")
    registry_pin = readiness["lock_material"]["artifact_pins"]["master_candidate_registry"]
    _require(sha256_file(Path(paths["master_candidate_registry"])) == registry_pin["sha256"], "Master registry differs from readiness lock")
    _require(result_lock.get("protocol_id") == config["source_protocol_id"], "Formal result protocol changed")
    actual_result_sha = sha256_file(Path(paths["formal_results"]))
    _require(actual_result_sha == result_lock["formal_results_sha256"], "Formal result table differs from result lock")
    state = load_json(Path(paths["kg_current_state"]))
    _require(state.get("status") == "canonical_current", "KG state is not canonical_current")
    for key in ("knowledge_graph", "extracted_claims"):
        state_file = state["canonical_files"][key]
        configured = Path(paths[key])
        _require(Path(state_file["path"]).resolve() == configured.resolve(), f"CURRENT_STATE {key} path changed")
        _require(configured.stat().st_size == int(state_file["bytes"]), f"CURRENT_STATE {key} byte count changed")
    return {
        "readiness_freeze_id": readiness["freeze_id"],
        "formal_run_id": result_lock["run_id"],
        "formal_results_sha256": actual_result_sha,
        "kg_generated_at": state["generated_at"],
        "kg_case2_papers": state["formal_kg_statistics"]["case_studies"]["case2_pathway_mediation"]["papers"],
        "kg_case2_claims": state["formal_kg_statistics"]["case_studies"]["case2_pathway_mediation"]["claims"],
        "temporal_hindcasting_freeze": False,
    }


def source_code_paths(config_path: Path) -> list[Path]:
    relative = (
        "neurooracle/configs/case2_adni_formal_benchmark_v1.json",
        "neurooracle/configs/case2_adni_formal_benchmark_requirements.txt",
        "neurooracle/configs/case2_adni_formal_benchmark_requirements.lock.txt",
        "neurooracle/scripts/prepare_case2_adni_formal_benchmark.py",
        "neurooracle/scripts/run_case2_adni_formal_benchmark.py",
        "neurooracle/scripts/evaluate_case2_adni_formal_benchmark.py",
        "neurooracle/tests/test_case2_adni_formal_benchmark.py",
        "core/scripts/case2_search_policy.py",
        "core/scripts/case2_official_baseline_experiment.py",
        "core/scripts/case_study_official_adapter_client.py",
        "core/scripts/case_study_native_output.py",
        "core/scripts/biomni_case_study_batch_client.py",
        "core/scripts/brainpilot_case_study_batch_client.mjs",
        "neurooracle/scripts/deepseek_v4_pro_router.py",
        "neurooracle/scripts/deepseek_v4_pro_gateway.py",
        "neurooracle/scripts/case2_formal_deepseek_gateway.py",
        "neurooracle/scripts/probe_deepseek_v4_pro_channels.py",
        "neurooracle/scripts/start_case2_formal_deepseek_gateway.ps1",
        "neurooracle/scripts/start_case2_formal_brainpilot_service.ps1",
        "neurooracle/src/hypothesis_cli.py",
        "neurooracle/src/case_studies.py",
        "neurooracle/scripts/map_case2_kg_hypotheses_to_adni.py",
        "neurooracle/scripts/map_case2_kg_hypotheses_to_adni_longitudinal.py",
    )
    paths = [REPO_ROOT / value for value in relative]
    if config_path.resolve() not in {path.resolve() for path in paths}:
        paths.append(config_path)
    return paths


def artifact_pins(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    excluded = set(exclude)
    pins: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        pin = file_pin(path)
        pins[relative] = {key: value for key, value in pin.items() if key != "path" and key != "mtime_ns"}
    return pins


def _write_runbook(path: Path, config_path: Path, output_root: Path) -> None:
    text = f"""# Case Study 2 formal generator benchmark runbook

Status: prepared only. No model API request is made by the preparation command.

This is a supplemental post-result-design-freeze benchmark. It is not a prospective, preregistered, external-validation, causal-mediation, or automatically clear-SOTA analysis.

## 1. Verify the freeze

```powershell
.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\prepare_case2_adni_formal_benchmark.py --config \"{config_path}\" --verify-only
```

## 2. Start the finite DeepSeek V4 Pro gateway

```powershell
powershell -ExecutionPolicy Bypass -File neurooracle\\scripts\\start_case2_formal_deepseek_gateway.ps1 -DataDir \"{output_root}\\runtime\\deepseek_gateway\" -Python \"{REPO_ROOT}\\.venv-case2-benchmark\\Scripts\\python.exe\"
```

The gateway reads keys only from the configured Downloads/keys.txt at runtime. Do not pass keys on the command line. The automatic route is Go key 1, 2, 3, Ollama Cloud, then one final Go pass. DeepSeek official remains disabled unless the user explicitly opts in.

Start the frozen-input BrainPilot service as well:

```powershell
powershell -ExecutionPolicy Bypass -File neurooracle\\scripts\\start_case2_formal_brainpilot_service.ps1 -DataDir \"{output_root}\\runtime\\brainpilot\"
```

## 3. Execute the seven methods

```powershell
.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\run_case2_adni_formal_benchmark.py --benchmark-root \"{output_root}\" --brainpilot-runtime-manifest \"{output_root}\\runtime\\brainpilot\\runtime_manifest.json\"
```

BrainPilot additionally requires its local service to be healthy immediately before the BrainPilot trials. Runs are resumable but an existing completed native artifact is never silently regenerated. Live retrieval stays disabled.

## 4. Evaluate only after all 70 rankings are frozen

```powershell
.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\evaluate_case2_adni_formal_benchmark.py --benchmark-root \"{output_root}\"
```

The evaluator reads evaluator_only/REFERENCE_LABELS.csv. Generator processes receive only generator_inputs and never receive this file or the formal result table.
"""
    path.write_text(text, encoding="utf-8")


def prepare(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    output_root = Path(config["paths"]["output_root"])
    if output_root.exists():
        raise FileExistsError(f"Benchmark target already exists; use --verify-only: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging target already exists: {staging}")
    staging.mkdir(parents=False)

    source_lock_summary = validate_source_locks(config)
    paths = config["paths"]
    public = build_public_registry(Path(paths["master_candidate_registry"]), Path(paths["score_manifest"]))
    generator_root = staging / "generator_inputs"
    registry_path = generator_root / "PUBLIC_CANDIDATE_REGISTRY.csv"
    final_registry_path = output_root / "generator_inputs" / registry_path.name
    _write_csv(registry_path, PUBLIC_COLUMNS, public)

    shared_contract = {
        "schema_version": "neurooracle.case2_generator_input_contract.v1",
        "benchmark_id": config["benchmark_id"],
        "candidate_count": len(public),
        "anchor_slots_per_method_trial": config["methods"]["shared_anchor_slots"],
        "candidate_universe_size": config["methods"]["compiled_ranking_size"],
        "experiment_stream_size_rule": "168 + failed generation slots",
        "candidate_id_template": "exposure|modality|marker|outcome",
        "native_retrieval_enabled": False,
        "result_isolation": "No empirical association value, sample-size field, multiplicity label, or evaluator reference is present in this bundle.",
        "invalid_output_rule": config["candidate_input_contract"]["invalid_output_rule"],
        "methods": list(EXPECTED_METHODS),
        "trials": [{"trial": index, "seed": seed} for index, seed in enumerate(config["trials"]["seeds"])],
    }
    write_json(generator_root / "SHARED_TASK_CONTRACT.json", shared_contract)
    for method in EXPECTED_METHODS:
        for trial, seed in enumerate(config["trials"]["seeds"]):
            task = build_task(
                public,
                method=method,
                trial=trial,
                seed=seed,
                n_anchors=config["methods"]["shared_anchor_slots"],
                final_registry_path=final_registry_path,
            )
            write_json(generator_root / "tasks" / method / f"trial_{trial:02d}" / "task.json", task)

    generator_pre_manifest_pins = artifact_pins(generator_root)
    input_manifest = {
        "schema_version": "neurooracle.case2_generator_input_manifest.v1",
        "benchmark_id": config["benchmark_id"],
        "status": "frozen_result_isolated_generator_inputs",
        "candidate_count": 168,
        "method_count": 7,
        "trials_per_method": 10,
        "task_count": 70,
        "files": generator_pre_manifest_pins,
        "api_calls_made": False,
        "evaluator_bundle_included": False,
    }
    write_json(generator_root / "INPUT_MANIFEST.json", input_manifest)
    leakage_audit = audit_generator_bundle(generator_root, config)
    _require(leakage_audit["status"] == "passed", f"Generator leakage audit failed: {leakage_audit['findings']}")
    write_json(generator_root / "GENERATOR_LEAKAGE_AUDIT.json", leakage_audit)

    evaluator_root = staging / "evaluator_only"
    reference, label_counts = build_evaluator_reference(
        Path(paths["formal_results"]), [row["candidate_id"] for row in public]
    )
    reference_path = evaluator_root / "REFERENCE_LABELS.csv"
    reference_columns = (
        "candidate_id",
        "bootstrap_weakest_link_evidence",
        "supplemental_family_fdr_hit",
        "global_fdr_hit",
        "nominal_bootstrap_hit",
    )
    _write_csv(reference_path, reference_columns, reference)
    evaluator_contract = {
        "schema_version": "neurooracle.case2_formal_benchmark_evaluator_contract.v1",
        "benchmark_id": config["benchmark_id"],
        "visibility": "evaluator_only_never_generator_input",
        "source_formal_run_id": source_lock_summary["formal_run_id"],
        "source_formal_results_sha256": source_lock_summary["formal_results_sha256"],
        "reference_labels_file": "REFERENCE_LABELS.csv",
        "reference_labels_sha256": sha256_file(reference_path),
        "label_counts": label_counts,
        "primary_metric": config["evaluation"]["primary_metric"],
        "secondary_metrics": config["evaluation"]["secondary_metrics"],
        "structurally_non_identifiable_metrics": config["evaluation"]["structurally_non_identifiable_metrics"],
        "budgets": config["evaluation"]["budgets"],
        "comparison": config["evaluation"]["comparison"],
        "post_result_design_freeze": True,
    }
    write_json(evaluator_root / "EVALUATOR_CONTRACT.json", evaluator_contract)

    write_json(staging / "PROTOCOL.json", config)
    key_preflight = key_count_preflight(Path(paths["keys_file"]))
    runtime_rows = runtime_preflight(config)
    _require(all(row["status"] == "ok" for row in runtime_rows), f"Runtime preflight failed: {runtime_rows}")
    baseline_root = Path(paths["baseline_repository_root"])
    repository_pins = {
        method: pin_repository(baseline_root / config["methods"]["repository_map"][method])
        for method in config["methods"]["repository_map"]
    }
    runtime_lock = {
        "schema_version": "neurooracle.case2_formal_benchmark_runtime_lock.v1",
        "benchmark_id": config["benchmark_id"],
        "model_backend": config["model_backend"],
        "key_preflight": key_preflight,
        "benchmark_environment_preflight": benchmark_environment_preflight(),
        "method_preflight": runtime_rows,
        "repository_pins": repository_pins,
        "api_calls_made": False,
    }
    write_json(staging / "METHOD_RUNTIME_LOCK.json", runtime_lock)
    _write_runbook(staging / "RUNBOOK.md", config_path, output_root)

    source_pins = {
        str(path.relative_to(REPO_ROOT).as_posix()): file_pin(path)
        for path in source_code_paths(config_path)
    }
    external_paths = {
        "readiness_lock": Path(paths["readiness_lock"]),
        "readiness_summary": Path(paths["readiness_summary"]),
        "master_candidate_registry": Path(paths["master_candidate_registry"]),
        "score_manifest": Path(paths["score_manifest"]),
        "pathway_catalog": Path(paths["pathway_catalog"]),
        "formal_results": Path(paths["formal_results"]),
        "formal_results_lock": Path(paths["formal_results_lock"]),
        "kg_current_state": Path(paths["kg_current_state"]),
        "knowledge_graph": Path(paths["knowledge_graph"]),
        "extracted_claims": Path(paths["extracted_claims"]),
    }
    external_pins = {
        name: file_pin(path, announce_large=True) for name, path in external_paths.items()
    }
    artifacts = artifact_pins(staging)
    freeze_material = {
        "benchmark_id": config["benchmark_id"],
        "protocol_sha256": sha256_file(config_path),
        "source_lock_summary": source_lock_summary,
        "source_code_pins": source_pins,
        "external_input_pins": external_pins,
        "repository_state_commit": canonical_sha(repository_pins),
        "artifact_pins": artifacts,
        "trial_seeds": config["trials"]["seeds"],
        "generator_leakage_audit": leakage_audit,
        "api_calls_made": False,
    }
    freeze_id = canonical_sha(freeze_material)
    lock = {
        "schema_version": "neurooracle.case2_formal_benchmark_freeze_lock.v1",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": freeze_id,
        "created_at_hkt": hkt_now(),
        "status": "locked_ready_for_generator_execution",
        "analysis_role": config["analysis_role"],
        "post_result_design_freeze": True,
        "generator_result_isolation": True,
        "api_calls_made": False,
        "no_external_validation_queue": True,
        "lock_material": freeze_material,
    }
    lock_path = staging / "BENCHMARK_FREEZE.lock.json"
    write_json(lock_path, lock)
    ready = {
        "schema_version": "neurooracle.case2_formal_benchmark_ready.v1",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": freeze_id,
        "status": "ready_for_generator_execution",
        "prepared_at_hkt": hkt_now(),
        "benchmark_lock_sha256": sha256_file(lock_path),
        "candidate_count": 168,
        "method_count": 7,
        "trials_per_method": 10,
        "expected_rankings": 70,
        "api_calls_made": False,
        "formal_biological_results_modified": False,
        "external_validation_queue_created": False,
        "next_action": "Start the local finite DeepSeek V4 Pro gateway, then run the generator runner exactly as documented in RUNBOOK.md.",
    }
    write_json(staging / "READY_TO_RUN.json", ready)
    os.replace(staging, output_root)
    return ready


def _same_pin(path: Path, pin: Mapping[str, Any], *, fast_large_files: bool) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != int(pin["bytes"]):
        return False
    if fast_large_files and stat.st_size >= LARGE_FILE_BYTES:
        return stat.st_mtime_ns == int(pin["mtime_ns"])
    return sha256_file(path, announce_large=not fast_large_files) == pin["sha256"]


def verify_freeze(
    output_root: Path,
    *,
    fast_large_files: bool = False,
    verify_repositories: bool = True,
) -> dict[str, Any]:
    ready_path = output_root / "READY_TO_RUN.json"
    lock_path = output_root / "BENCHMARK_FREEZE.lock.json"
    ready = load_json(ready_path)
    lock = load_json(lock_path)
    _require(ready.get("status") == "ready_for_generator_execution", "Benchmark is not ready")
    _require(lock.get("status") == "locked_ready_for_generator_execution", "Benchmark lock is not ready")
    _require(sha256_file(lock_path) == ready["benchmark_lock_sha256"], "Benchmark lock changed")
    _require(lock["freeze_id"] == ready["freeze_id"], "Freeze IDs differ")
    for relative, pin in lock["lock_material"]["artifact_pins"].items():
        path = output_root / Path(relative)
        _require(path.is_file(), f"Frozen artifact missing: {relative}")
        _require(path.stat().st_size == int(pin["bytes"]), f"Frozen artifact size changed: {relative}")
        _require(sha256_file(path) == pin["sha256"], f"Frozen artifact changed: {relative}")
    for name, pin in lock["lock_material"]["external_input_pins"].items():
        _require(_same_pin(Path(pin["path"]), pin, fast_large_files=fast_large_files), f"External input changed: {name}")
    for relative, pin in lock["lock_material"]["source_code_pins"].items():
        _require(_same_pin(Path(pin["path"]), pin, fast_large_files=False), f"Source code changed: {relative}")
    config = load_json(output_root / "PROTOCOL.json")
    leakage = audit_generator_bundle(output_root / "generator_inputs", config)
    _require(leakage["status"] == "passed", f"Generator leakage audit failed: {leakage['findings']}")
    key_preflight = key_count_preflight(Path(config["paths"]["keys_file"]))
    if verify_repositories:
        baseline_root = Path(config["paths"]["baseline_repository_root"])
        current_repositories = {
            method: pin_repository(baseline_root / config["methods"]["repository_map"][method])
            for method in config["methods"]["repository_map"]
        }
        runtime_lock = load_json(output_root / "METHOD_RUNTIME_LOCK.json")
        _require(current_repositories == runtime_lock["repository_pins"], "Baseline repository state changed")
    return {
        "benchmark_id": ready["benchmark_id"],
        "freeze_id": ready["freeze_id"],
        "status": "verified_ready_for_generator_execution",
        "artifact_count": len(lock["lock_material"]["artifact_pins"]),
        "external_input_count": len(lock["lock_material"]["external_input_pins"]),
        "source_code_pin_count": len(lock["lock_material"]["source_code_pins"]),
        "large_files_rehashed": not fast_large_files,
        "repository_state_verified": verify_repositories,
        "key_preflight": key_preflight,
        "api_calls_made": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--fast-large-files",
        action="store_true",
        help="During verification only, validate >=1 GB files by frozen bytes and mtime instead of rehashing them.",
    )
    parser.add_argument("--skip-repository-verification", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config.resolve())
    validate_config(config)
    output_root = Path(config["paths"]["output_root"])
    if args.verify_only:
        result = verify_freeze(
            output_root,
            fast_large_files=args.fast_large_files,
            verify_repositories=not args.skip_repository_verification,
        )
    else:
        if args.fast_large_files or args.skip_repository_verification:
            raise ValueError("Verification shortcuts are valid only with --verify-only")
        result = prepare(args.config)
    safe = {
        key: value
        for key, value in result.items()
        if key not in {"key_preflight"}
    }
    print(json.dumps(safe, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
