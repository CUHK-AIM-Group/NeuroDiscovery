"""Freeze, preflight, and run the seven supplemental Case Studies.

Formal execution is impossible without a cryptographic code lock bound to the
current canonical KG. Development uses five seeds and five outer folds; final
uses ten seeds and five outer folds. The same seed-specific splits are reused
across model families. This runner never updates the KG or runs hindcasting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import urlopen

import pandas as pd

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_RELEASE_ID,
    CURRENT_CANONICAL_SHA256,
    sha256_file,
    validate_canonical_kg_release,
)
from core.scripts.case_study_closed_loop_specs import (
    EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES,
    EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES,
    SUPPLEMENTAL_CASE_STUDIES,
    evaluation_profile_for,
    protocol_for,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\supplemental_case_studies_formal"
)
DEFAULT_KG = ROOT / "neurooracle/data/full_v2/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
DEFAULT_STATE = ROOT / "neurooracle/data/full_v2/CURRENT_STATE.json"
DEFAULT_GATEWAY = "http://127.0.0.1:18082/v1"
DEFAULT_BRAINPILOT = "http://127.0.0.1:18080/api"

PREDICTIVE_TASKS = frozenset(
    {
        "differential_diagnosis",
        "connectome_behavior",
        "brain_age",
        "progression_prediction",
        "prognosis",
    }
)

TASK_MODULES = {
    "differential_diagnosis": "core.scripts.run_psychiatric_closed_loop_experiments",
    "disease_subtyping": "core.scripts.run_psychiatric_closed_loop_experiments",
    "connectome_behavior": "core.scripts.run_hcp_closed_loop_experiments",
    "brain_age": "core.scripts.run_hcp_closed_loop_experiments",
    "progression_prediction": "core.scripts.run_adni_closed_loop_experiments",
    "prognosis": "core.scripts.run_adni_closed_loop_experiments",
    "imaging_genetics": "core.scripts.run_adni_closed_loop_experiments",
}

CODE_FILES = (
    "core/scripts/run_supplemental_case_studies_formal.py",
    "core/scripts/canonical_kg_release.py",
    "core/scripts/case1_exhaustive_full.py",
    "core/scripts/case1_exhaustive_v1.py",
    "core/scripts/case1_exhaustive_v2.py",
    "core/scripts/case1_kg_stream.py",
    "core/scripts/case1_multimodel_pilot.py",
    "core/scripts/case_study_closed_loop_specs.py",
    "core/scripts/case_study_candidate_tables.py",
    "core/scripts/case_study_closed_loop.py",
    "core/scripts/case_study_closed_loop_engine.py",
    "core/scripts/case_study_feedback_adapters.py",
    "core/scripts/case_study_framework_runtime.py",
    "core/scripts/case_study_native_output.py",
    "core/scripts/case_study_neurodiscovery_policy.py",
    "core/scripts/case_study_official_adapter_client.py",
    "core/scripts/case_study_score_components.py",
    "core/scripts/case_study_search_policy.py",
    "core/scripts/run_case_study_framework_comparison.py",
    "core/scripts/run_psychiatric_closed_loop_experiments.py",
    "core/scripts/run_hcp_closed_loop_experiments.py",
    "core/scripts/run_adni_closed_loop_experiments.py",
    "core/scripts/run_lifespan_multimodel_expansion.py",
    "core/scripts/run_supplemental_differential_deep_models.py",
    "core/scripts/biomni_case_study_batch_client.py",
    "core/scripts/brainpilot_case_study_batch_client.mjs",
    "core/scripts/start_brainpilot_local_service.ps1",
    "core/scripts/start_deepseek_v4_pro_gateway.ps1",
    "neurooracle/scripts/probe_deepseek_v4_pro_channels.py",
    "neurooracle/scripts/deepseek_v4_pro_router.py",
    "neurooracle/scripts/deepseek_v4_pro_gateway.py",
    "core/tool-runtime/runtime.py",
    "skills/autoresearch-framework-benchmark/handler.py",
)
FRAMEWORK_ONLY_CODE_FILES = frozenset(
    {
        "core/scripts/case_study_framework_runtime.py",
        "core/scripts/case_study_native_output.py",
        "core/scripts/case_study_official_adapter_client.py",
        "core/scripts/run_case_study_framework_comparison.py",
        "core/scripts/biomni_case_study_batch_client.py",
        "core/scripts/brainpilot_case_study_batch_client.mjs",
        "core/scripts/start_brainpilot_local_service.ps1",
        "core/scripts/start_deepseek_v4_pro_gateway.ps1",
        "neurooracle/scripts/probe_deepseek_v4_pro_channels.py",
        "neurooracle/scripts/deepseek_v4_pro_router.py",
        "neurooracle/scripts/deepseek_v4_pro_gateway.py",
        "core/tool-runtime/runtime.py",
        "skills/autoresearch-framework-benchmark/handler.py",
    }
)
CODE_DIRECTORIES = (
    "models/statistical_ml",
    "models/subtyping",
    "models/survival_models",
    "models/imaging_genetics",
    "models/temporal_models",
    "models/bnt",
    "models/braingnn",
    "models/brainnetcnn",
    "models/lggnn",
    "models/ibgnn",
    "models/combraintf",
    "models/common",
    "skills/brain_gnn/scripts",
)
CODE_SUFFIXES = frozenset({".py", ".mjs", ".js", ".ps1", ".json"})
FRAMEWORK_REPOSITORIES = {
    "ai_scientist_v2": ROOT.parent / "autoresearch_baselines/AI-Scientist-v2",
    "open_coscientist": ROOT.parent / "autoresearch_baselines/open-coscientist",
    "sciagents": ROOT.parent / "autoresearch_baselines/SciAgentsDiscovery",
    "virtual_lab": ROOT.parent / "autoresearch_baselines/virtual-lab",
    "brainpilot_native": ROOT.parent / "autoresearch_baselines/BrainPilot",
    "biomni_native": ROOT.parent / "autoresearch_baselines/Biomni",
}
FRAMEWORK_UNTRACKED_CODE_SUFFIXES = frozenset(
    {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".json", ".yaml", ".yml"}
)

REQUIRED_DATA = {
    "ucla": Path(r"\\192.168.3.61\data\Public Dataset\ucla_preprocessed"),
    "hcpep": Path(r"\\192.168.3.61\data\Public Dataset\hcpep_preprocessed"),
    "cobre": Path(r"\\192.168.3.61\data\Public Dataset\cobre_preprocessed"),
    "adhd200": Path(r"\\192.168.3.61\data\Public Dataset\adhd200_preprocessed"),
    "hcpya_metadata": Path(r"\\192.168.3.61\data\Public Dataset\csv\hcp.csv"),
    "adni": Path(
        r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
        r"\case2_adni_genetics_v1\experiment_tables\case2_adni_multimodal_v1"
    ),
    "fc_cache": ROOT / "data/braingnn_input",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _source_files() -> tuple[Path, ...]:
    files: set[Path] = set()
    for relative in CODE_FILES:
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing formal source file: {path}")
        files.add(path)
    for relative in CODE_DIRECTORIES:
        directory = (ROOT / relative).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"missing formal source directory: {directory}")
        files.update(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in CODE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return tuple(sorted(files, key=lambda path: str(path).casefold()))


def _git_metadata(directory: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "root": str(directory.resolve()),
        "head": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_lines": len(status.splitlines()) if status else 0,
    }


def _git_output(directory: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=directory,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"git {' '.join(arguments)} failed below {directory}: {detail}")
    return result.stdout


def framework_repository_snapshot(directory: Path) -> dict[str, Any]:
    """Fingerprint a baseline checkout without storing its source or diff."""

    directory = directory.resolve()
    if not directory.is_dir():
        return {"available": False, "root": str(directory)}
    head = _git_output(directory, "rev-parse", "HEAD").decode().strip()
    diff = _git_output(directory, "diff", "HEAD", "--binary", "--no-ext-diff")
    untracked_raw = _git_output(
        directory,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked: dict[str, dict[str, Any]] = {}
    for encoded in untracked_raw.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = directory / relative
        if (
            path.is_file()
            and path.suffix.casefold() in FRAMEWORK_UNTRACKED_CODE_SUFFIXES
            and ".cache" not in {part.casefold() for part in path.parts}
        ):
            untracked[Path(relative).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "available": True,
        "root": str(directory),
        "head": head,
        "tracked_diff_bytes": len(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_code": untracked,
    }


def framework_snapshot() -> dict[str, dict[str, Any]]:
    return {
        name: framework_repository_snapshot(path)
        for name, path in FRAMEWORK_REPOSITORIES.items()
    }


def verify_framework_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, directory in FRAMEWORK_REPOSITORIES.items():
        expected = snapshot.get(name)
        if not isinstance(expected, Mapping) or not expected.get("available"):
            errors.append(f"framework repository was not locked: {name}")
            continue
        try:
            current = framework_repository_snapshot(directory)
        except Exception as exc:
            errors.append(f"cannot verify framework repository {name}: {exc}")
            continue
        for field in ("head", "tracked_diff_bytes", "tracked_diff_sha256"):
            if current.get(field) != expected.get(field):
                errors.append(f"framework repository changed after code freeze: {name}")
                break
        else:
            if current.get("untracked_code") != expected.get("untracked_code"):
                errors.append(f"framework untracked code changed after code freeze: {name}")
    return errors


def code_snapshot() -> dict[str, Any]:
    files = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _source_files()
    }
    return {
        "repository": _git_metadata(ROOT),
        "files": files,
        "file_count": len(files),
        "snapshot_sha256": _hash_payload(files),
    }


def _scoped_source_records(
    snapshot: Mapping[str, Any],
    *,
    include_framework_code: bool,
) -> dict[str, Any]:
    records = snapshot.get("files") or {}
    if not isinstance(records, Mapping):
        return {}
    return {
        str(relative): record
        for relative, record in records.items()
        if include_framework_code or str(relative) not in FRAMEWORK_ONLY_CODE_FILES
    }


def scoped_source_snapshot_sha256(
    snapshot: Mapping[str, Any],
    *,
    include_framework_code: bool,
) -> str:
    return _hash_payload(
        _scoped_source_records(
            snapshot,
            include_framework_code=include_framework_code,
        )
    )


def verify_code_snapshot(
    snapshot: Mapping[str, Any],
    *,
    include_framework_code: bool = True,
) -> list[str]:
    errors: list[str] = []
    raw_expected = snapshot.get("files") or {}
    if not isinstance(raw_expected, Mapping):
        return ["code lock has no file map"]
    expected = _scoped_source_records(
        snapshot,
        include_framework_code=include_framework_code,
    )
    current_names = {
        path.relative_to(ROOT).as_posix()
        for path in _source_files()
        if include_framework_code
        or path.relative_to(ROOT).as_posix() not in FRAMEWORK_ONLY_CODE_FILES
    }
    expected_names = set(map(str, expected))
    if current_names != expected_names:
        errors.append("formal source file set changed after code freeze")
    for relative, record in expected.items():
        path = ROOT / str(relative)
        if not path.is_file():
            errors.append(f"missing locked source: {relative}")
            continue
        actual = sha256_file(path)
        expected_hash = str((record or {}).get("sha256") or "")
        if actual.casefold() != expected_hash.casefold():
            errors.append(f"locked source changed: {relative}")
    return errors


def fast_kg_preflight() -> dict[str, Any]:
    state_hash = sha256_file(DEFAULT_STATE)
    state = json.loads(DEFAULT_STATE.read_text(encoding="utf-8"))
    errors: list[str] = []
    if state_hash.casefold() != CURRENT_CANONICAL_SHA256["current_state"].casefold():
        errors.append("CURRENT_STATE.json hash differs from the locked latest KG")
    if state.get("status") != "canonical_current":
        errors.append("KG status is not canonical_current")
    for name, path in (
        ("knowledge_graph", DEFAULT_KG),
        ("extracted_claims", DEFAULT_CLAIMS),
    ):
        record = (state.get("canonical_files") or {}).get(name) or {}
        if not path.is_file():
            errors.append(f"missing canonical {name}")
        elif path.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"canonical {name} size differs from CURRENT_STATE.json")
    return {
        "release_id": CURRENT_CANONICAL_RELEASE_ID,
        "state_sha256": state_hash,
        "expected_sha256": dict(CURRENT_CANONICAL_SHA256),
        "errors": errors,
        "ok": not errors,
    }


def freeze_code_lock(path: Path, phase: str, *, verify_full_kg: bool) -> dict[str, Any]:
    profile = evaluation_profile_for(phase)
    fast = fast_kg_preflight()
    if not fast["ok"]:
        raise RuntimeError("; ".join(fast["errors"]))
    release: dict[str, Any] | None = None
    if verify_full_kg:
        release = validate_canonical_kg_release(
            kg_path=DEFAULT_KG,
            claims_path=DEFAULT_CLAIMS,
            state_path=DEFAULT_STATE,
            expected_sha256=CURRENT_CANONICAL_SHA256,
        )
    payload: dict[str, Any] = {
        "schema_version": "supplemental-case-studies-code-lock.v1",
        "created_at": utc_now(),
        "phase": phase,
        "evaluation_profile": asdict(profile),
        "tasks": list(SUPPLEMENTAL_CASE_STUDIES),
        "external_validation": {
            "required": list(EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES),
            "exempt": list(EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES),
        },
        "brain_age_discovery": "HCP-YA",
        "code": code_snapshot(),
        "framework_repositories": framework_snapshot(),
        "kg": {
            **fast,
            "full_hashes_verified": bool(release),
            "release_manifest": release,
        },
        "deepseek_v4_pro": {
            "automatic_route": [
                "opencode_go_key_1",
                "opencode_go_key_2",
                "opencode_go_key_3",
                "ollama_cloud",
                "opencode_go_final_pass_once",
            ],
            "official_api_automatic_fallback": False,
            "transient_retry_seconds": [2, 8, 30],
            "secrets_persisted": False,
        },
        "scope_exclusions": ["hindcasting", "knowledge_graph_updates"],
    }
    payload["lock_id"] = _hash_payload(payload)
    _write_json_atomic(path.resolve(), payload)
    return payload


def load_and_verify_lock(
    path: Path,
    phase: str,
    *,
    verify_frameworks: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != "supplemental-case-studies-code-lock.v1":
        errors.append("unsupported code lock schema")
    if payload.get("phase") != phase:
        errors.append("code lock phase does not match requested phase")
    expected_id = payload.pop("lock_id", None)
    actual_id = _hash_payload(payload)
    payload["lock_id"] = expected_id
    if expected_id != actual_id:
        errors.append("code lock payload hash is invalid")
    errors.extend(
        verify_code_snapshot(
            payload.get("code") or {},
            include_framework_code=verify_frameworks,
        )
    )
    if verify_frameworks:
        errors.extend(
            verify_framework_snapshot(payload.get("framework_repositories") or {})
        )
    kg = fast_kg_preflight()
    errors.extend(kg["errors"])
    locked_kg = payload.get("kg") or {}
    if not bool(locked_kg.get("full_hashes_verified")):
        errors.append("formal code lock was not created with full KG hash verification")
    if (locked_kg.get("expected_sha256") or {}) != dict(CURRENT_CANONICAL_SHA256):
        errors.append("code lock is bound to a different KG hash set")
    return payload, errors


@dataclass(frozen=True)
class Job:
    job_id: str
    task: str
    seed: int | None
    kind: str
    output_dir: Path
    command: tuple[str, ...]


def _closure_command(
    *,
    task: str,
    seed: int,
    seed_root: Path,
    benchmark_trials: int,
    quick: bool,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        TASK_MODULES[task],
        "--task",
        task,
        "--output-root",
        str(seed_root),
        "--kg",
        str(DEFAULT_KG),
        "--seed",
        str(seed),
        "--cv-repeats",
        "1",
        "--benchmark-trials",
        str(benchmark_trials),
    ]
    if task == "prognosis":
        command.extend(["--prognosis-mode", "marker_association"])
    if quick:
        command.append("--quick")
    return tuple(command)


def build_jobs(
    *,
    phase: str,
    run_dir: Path,
    cases: Sequence[str],
    quick: bool,
    include_robustness: bool,
) -> list[Job]:
    profile = evaluation_profile_for(phase)
    seeds = profile.seeds[:1] if quick else profile.seeds
    jobs: list[Job] = []
    for seed in seeds:
        seed_root = run_dir / "seeds" / f"seed_{seed}"
        for task in cases:
            jobs.append(
                Job(
                    job_id=f"closure__{task}__seed_{seed}",
                    task=task,
                    seed=seed,
                    kind="closure",
                    output_dir=seed_root / task,
                    command=_closure_command(
                        task=task,
                        seed=seed,
                        seed_root=seed_root,
                        benchmark_trials=profile.benchmark_trials,
                        quick=quick,
                    ),
                )
            )
    if include_robustness and "differential_diagnosis" in cases:
        selected_seeds = seeds[:1] if quick else seeds
        out = run_dir / "model_robustness" / "differential_diagnosis"
        command = [
            sys.executable,
            "-m",
            "core.scripts.run_supplemental_differential_deep_models",
            "--out-root",
            str(out.parent),
            "--run-name",
            out.name,
            "--seeds",
            ",".join(map(str, selected_seeds)),
            "--folds",
            str(profile.outer_folds),
            "--resume",
        ]
        if quick:
            command.append("--smoke")
        jobs.append(
            Job(
                job_id="robustness__differential_diagnosis",
                task="differential_diagnosis",
                seed=None,
                kind="model_robustness",
                output_dir=out,
                command=tuple(command),
            )
        )
    if include_robustness and "brain_age" in cases:
        selected_seeds = seeds[:1] if quick else seeds
        out = run_dir / "model_robustness" / "brain_age"
        command = [
            sys.executable,
            "-m",
            "core.scripts.run_lifespan_multimodel_expansion",
            "--out-root",
            str(out.parent),
            "--run-name",
            out.name,
            "--datasets",
            "hcpya",
            "--cohort-name",
            "HCP-YA",
            "--seeds",
            ",".join(map(str, selected_seeds)),
            "--folds",
            str(profile.outer_folds),
            "--resume",
        ]
        if quick:
            command.append("--smoke")
        jobs.append(
            Job(
                job_id="robustness__brain_age",
                task="brain_age",
                seed=None,
                kind="model_robustness",
                output_dir=out,
                command=tuple(command),
            )
        )
    if include_robustness and "prognosis" in cases:
        for seed in seeds:
            robustness_root = (
                run_dir
                / "model_robustness"
                / "prognosis_risk_models"
                / f"seed_{seed}"
            )
            command = [
                sys.executable,
                "-m",
                "core.scripts.run_adni_closed_loop_experiments",
                "--task",
                "prognosis",
                "--prognosis-mode",
                "risk_model",
                "--output-root",
                str(robustness_root),
                "--kg",
                str(DEFAULT_KG),
                "--seed",
                str(seed),
                "--cv-repeats",
                "1",
                "--benchmark-trials",
                str(profile.benchmark_trials),
            ]
            if quick:
                command.append("--quick")
            jobs.append(
                Job(
                    job_id=f"robustness__prognosis__seed_{seed}",
                    task="prognosis",
                    seed=seed,
                    kind="closure_robustness",
                    output_dir=robustness_root / "prognosis",
                    command=tuple(command),
                )
            )
    return jobs


def _gateway_health(base_url: str) -> dict[str, Any]:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]
    endpoint += "/health"
    try:
        with urlopen(endpoint, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ok = response.status == 200 and payload.get("model") == "deepseek-v4-pro"
        return {"ok": ok, "endpoint": endpoint, "payload": payload}
    except Exception as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "error": f"{type(exc).__name__}: {exc}",
        }


def preflight(
    *,
    phase: str,
    lock_path: Path | None,
    framework_lock_path: Path | None = None,
    with_frameworks: bool,
    gateway_url: str,
    brainpilot_url: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    profile = evaluation_profile_for(phase)
    core_lock: dict[str, Any] | None = None
    framework_lock: dict[str, Any] | None = None
    if lock_path is None or not lock_path.is_file():
        errors.append("a full-hash core code lock is required before formal execution")
    else:
        core_lock, lock_errors = load_and_verify_lock(
            lock_path,
            phase,
            verify_frameworks=False,
        )
        errors.extend(lock_errors)
    if with_frameworks:
        selected_framework_lock = framework_lock_path or lock_path
        if selected_framework_lock is None or not selected_framework_lock.is_file():
            errors.append(
                "a full-hash framework/API code lock is required when frameworks are enabled"
            )
        else:
            framework_lock, framework_lock_errors = load_and_verify_lock(
                selected_framework_lock,
                phase,
                verify_frameworks=True,
            )
            errors.extend(framework_lock_errors)
    data = {name: path.exists() for name, path in REQUIRED_DATA.items()}
    errors.extend(f"missing required data: {name}" for name, ok in data.items() if not ok)
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "statsmodels",
            "torch",
            "sksurv",
            "xgboost",
        )
    }
    errors.extend(f"missing Python package: {name}" for name, ok in packages.items() if not ok)
    if Path(sys.executable).name.casefold() == "python.exe" and "windowsapps" in str(
        Path(sys.executable)
    ).casefold():
        errors.append("Windows Store Python is not allowed")
    gateway = None
    brainpilot = None
    if with_frameworks:
        gateway = _gateway_health(gateway_url)
        if not gateway["ok"]:
            errors.append("DeepSeek V4 Pro gateway is not healthy")
        try:
            with urlopen(brainpilot_url.rstrip("/") + "/health", timeout=5) as response:
                brainpilot = {"ok": response.status < 400}
        except Exception as exc:
            brainpilot = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            errors.append("BrainPilot local service is not healthy")
    if not DEFAULT_OUTPUT_ROOT.parent.exists():
        warnings.append("default NAS output parent is not currently reachable")
    return {
        "schema_version": "supplemental-case-studies-preflight.v1",
        "created_at": utc_now(),
        "phase": phase,
        "profile": asdict(profile),
        # Keep lock_id for compatibility; it is the stable scientific/core lock.
        "lock_id": core_lock.get("lock_id") if core_lock else None,
        "core_lock_id": core_lock.get("lock_id") if core_lock else None,
        "framework_lock_id": (
            framework_lock.get("lock_id") if framework_lock else None
        ),
        "kg": fast_kg_preflight(),
        "data": data,
        "packages": packages,
        "python": sys.executable,
        "external_validation_required": list(
            EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES
        ),
        "external_validation_exempt": list(
            EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES
        ),
        "brain_age_discovery": "HCP-YA",
        "gateway": gateway,
        "brainpilot": brainpilot,
        "errors": errors,
        "warnings": warnings,
        "ready": not errors,
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def _job_complete(job: Job) -> bool:
    if job.kind.startswith("closure"):
        path = job.output_dir / "closure_manifest.json"
    else:
        path = job.output_dir / "manifest.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if job.kind.startswith("closure"):
        return payload.get("status") == "complete"
    return int(payload.get("n_jobs_failed", payload.get("n_failed_model_folds", 0))) == 0


def run_job(job: Job, *, run_dir: Path, resume: bool) -> dict[str, Any]:
    if resume and _job_complete(job):
        return {
            "created_at": utc_now(),
            "job_id": job.job_id,
            "task": job.task,
            "seed": job.seed,
            "kind": job.kind,
            "status": "completed",
            "resumed": True,
            "returncode": 0,
        }
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job.job_id}.stdout.log"
    stderr_path = log_dir / f"{job.job_id}.stderr.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    process = subprocess.run(
        list(job.command),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    complete = process.returncode == 0 and _job_complete(job)
    return {
        "created_at": utc_now(),
        "job_id": job.job_id,
        "task": job.task,
        "seed": job.seed,
        "kind": job.kind,
        "status": "completed" if complete else "failed",
        "resumed": False,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def runtime_lock_check(
    *,
    lock_path: Path,
    phase: str,
    stage: str,
    job_id: str | None = None,
    verify_frameworks: bool = False,
) -> dict[str, Any]:
    """Re-verify the frozen sources at every subprocess boundary."""

    lock, errors = load_and_verify_lock(
        lock_path,
        phase,
        verify_frameworks=verify_frameworks,
    )
    return {
        "created_at": utc_now(),
        "stage": stage,
        "job_id": job_id,
        "lock_id": lock.get("lock_id"),
        "lock_path": str(lock_path.resolve()),
        "source_scope": "framework_and_core" if verify_frameworks else "core",
        "source_snapshot_sha256": scoped_source_snapshot_sha256(
            lock.get("code") or {},
            include_framework_code=verify_frameworks,
        ),
        "verify_frameworks": bool(verify_frameworks),
        "ok": not errors,
        "errors": errors,
    }


def audit_closure(job: Job, *, expected_folds: int) -> dict[str, Any]:
    tables = job.output_dir / "tables"
    manifest_path = tables / "table_manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"job_id": job.job_id, "ok": False, "errors": ["missing table manifest"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kg_hash = str((manifest.get("kg_scoring") or {}).get("kg_sha256") or "")
    if kg_hash.casefold() != CURRENT_CANONICAL_SHA256["knowledge_graph"].casefold():
        errors.append("candidate scores do not use the locked latest KG")
    public_path = tables / "public_candidates.csv"
    internal_path = tables / "internal_outcomes.csv"
    external_path = tables / "external_outcomes.csv"
    for path in (public_path, internal_path):
        if not path.is_file():
            errors.append(f"missing {path.name}")
    protocol = protocol_for(job.task)
    if protocol.external_required:
        if not external_path.is_file():
            errors.append("required external outcomes are missing")
        elif int(manifest.get("external_executable") or 0) < 1:
            errors.append("required external validation has no executable candidate")
    if job.task in PREDICTIVE_TASKS and internal_path.is_file():
        internal = pd.read_csv(internal_path, low_memory=False)
        if "n_folds" not in internal:
            errors.append("predictive outcomes do not record n_folds")
        else:
            observed = set(
                pd.to_numeric(internal["n_folds"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            if observed != {expected_folds}:
                errors.append(f"expected exactly {expected_folds} folds, observed {sorted(observed)}")
    return {"job_id": job.job_id, "ok": not errors, "errors": errors}


def _framework_command(
    *,
    source_root: Path,
    cases: Sequence[str],
    phase: str,
    gateway_url: str,
    brainpilot_url: str,
    code_lock: Path,
) -> list[str]:
    profile = evaluation_profile_for(phase)
    return [
        sys.executable,
        "-m",
        "core.scripts.run_case_study_framework_comparison",
        "--root",
        str(source_root),
        "--tasks",
        *cases,
        "--trials",
        str(profile.benchmark_trials),
        "--model",
        "deepseek-v4-pro",
        "--base-url",
        gateway_url,
        "--brainpilot-url",
        brainpilot_url,
        "--code-lock",
        str(code_lock.resolve()),
        "--lock-phase",
        phase,
        "--max-retries",
        "1",
        "--seed",
        str(profile.seeds[0]),
    ]


def execute_run(args: argparse.Namespace, cases: Sequence[str]) -> int:
    if not args.run_name:
        raise ValueError("--run-name is required for a reproducible run")
    framework_lock_path = (args.framework_lock or args.lock) if args.with_frameworks else None
    report = preflight(
        phase=args.phase,
        lock_path=args.lock,
        framework_lock_path=framework_lock_path,
        with_frameworks=args.with_frameworks,
        gateway_url=args.gateway_url,
        brainpilot_url=args.brainpilot_url,
    )
    if not report["ready"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["errors"]))
    profile = evaluation_profile_for(args.phase)
    run_dir = args.output_root.resolve() / args.run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(run_dir / "preflight.json", report)
    jobs = build_jobs(
        phase=args.phase,
        run_dir=run_dir,
        cases=cases,
        quick=args.smoke,
        include_robustness=not args.skip_robustness,
    )
    manifest = {
        "schema_version": "supplemental-case-studies-run.v1",
        "created_at": utc_now(),
        "status": "running",
        "formal": not args.smoke,
        "phase": args.phase,
        "profile": asdict(profile),
        "cases": list(cases),
        # Legacy aliases continue to identify the scientific/core lock.
        "code_lock": str(args.lock.resolve()),
        "code_lock_sha256": sha256_file(args.lock.resolve()),
        "core_code_lock": str(args.lock.resolve()),
        "core_code_lock_sha256": sha256_file(args.lock.resolve()),
        "framework_code_lock": (
            str(framework_lock_path.resolve()) if framework_lock_path else None
        ),
        "framework_code_lock_sha256": (
            sha256_file(framework_lock_path.resolve()) if framework_lock_path else None
        ),
        "kg_release_id": CURRENT_CANONICAL_RELEASE_ID,
        "kg_sha256": dict(CURRENT_CANONICAL_SHA256),
        "brain_age_discovery": "HCP-YA",
        "jobs": [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "command": list(job.command),
            }
            for job in jobs
        ],
        "scope_exclusions": ["hindcasting", "knowledge_graph_updates"],
    }
    _write_json_atomic(run_dir / "run_manifest.json", manifest)
    status_path = run_dir / "job_status.jsonl"
    lock_status_path = run_dir / "lock_status.jsonl"
    records: list[dict[str, Any]] = []
    lock_checks: list[dict[str, Any]] = []
    lock_drift_detected = False
    for index, job in enumerate(jobs, start=1):
        before = runtime_lock_check(
            lock_path=args.lock,
            phase=args.phase,
            stage="before_job",
            job_id=job.job_id,
        )
        lock_checks.append(before)
        _append_jsonl(lock_status_path, before)
        if not before["ok"]:
            record = {
                "created_at": utc_now(),
                "job_id": job.job_id,
                "task": job.task,
                "seed": job.seed,
                "kind": job.kind,
                "status": "failed",
                "resumed": False,
                "returncode": None,
                "error_type": "code_lock_drift_before_job",
                "errors": before["errors"],
            }
            records.append(record)
            _append_jsonl(status_path, record)
            lock_drift_detected = True
            break
        print(f"[{index}/{len(jobs)}] {job.job_id}", flush=True)
        record = run_job(job, run_dir=run_dir, resume=args.resume)
        records.append(record)
        _append_jsonl(status_path, record)
        after = runtime_lock_check(
            lock_path=args.lock,
            phase=args.phase,
            stage="after_job",
            job_id=job.job_id,
        )
        lock_checks.append(after)
        _append_jsonl(lock_status_path, after)
        if not after["ok"]:
            # The child imported its modules only after the matching before-job
            # check. Preserve a completed checkpoint, but stop before any new
            # subprocess can import the changed working tree.
            record["lock_drift_detected_after_job"] = True
            record["lock_drift_errors"] = after["errors"]
            lock_drift_detected = True
            break
        if record["status"] != "completed" and args.fail_fast:
            break
    closure_audits = [
        audit_closure(job, expected_folds=profile.outer_folds)
        for job in jobs
        if job.kind.startswith("closure") and _job_complete(job)
    ]
    framework_returncode: int | None = None
    framework_lock_check: dict[str, Any] | None = None
    if (
        args.with_frameworks
        and not lock_drift_detected
        and all(record["status"] == "completed" for record in records)
        and all(item["ok"] for item in closure_audits)
    ):
        framework_lock_check = runtime_lock_check(
            lock_path=framework_lock_path,
            phase=args.phase,
            stage="before_frameworks",
            verify_frameworks=True,
        )
        lock_checks.append(framework_lock_check)
        _append_jsonl(lock_status_path, framework_lock_check)
        if framework_lock_check["ok"]:
            source_root = run_dir / "seeds" / f"seed_{profile.seeds[0]}"
            command = _framework_command(
                source_root=source_root,
                cases=cases,
                phase=args.phase,
                gateway_url=args.gateway_url,
                brainpilot_url=args.brainpilot_url,
                code_lock=framework_lock_path,
            )
            env = os.environ.copy()
            env["CASE_STUDY_LOCAL_API_KEY"] = "neuroclaw-local-router"
            process = subprocess.run(command, cwd=ROOT, env=env, check=False)
            framework_returncode = process.returncode
        else:
            lock_drift_detected = True
    final_core_lock_check = runtime_lock_check(
        lock_path=args.lock,
        phase=args.phase,
        stage="final_core",
        verify_frameworks=False,
    )
    lock_checks.append(final_core_lock_check)
    _append_jsonl(lock_status_path, final_core_lock_check)
    if not final_core_lock_check["ok"]:
        lock_drift_detected = True
    final_framework_lock_check: dict[str, Any] | None = None
    if args.with_frameworks:
        final_framework_lock_check = runtime_lock_check(
            lock_path=framework_lock_path,
            phase=args.phase,
            stage="final_frameworks",
            verify_frameworks=True,
        )
        lock_checks.append(final_framework_lock_check)
        _append_jsonl(lock_status_path, final_framework_lock_check)
        if not final_framework_lock_check["ok"]:
            lock_drift_detected = True
    complete = (
        not lock_drift_detected
        and len(records) == len(jobs)
        and all(record["status"] == "completed" for record in records)
        and len(closure_audits)
        == sum(job.kind.startswith("closure") for job in jobs)
        and all(item["ok"] for item in closure_audits)
        and (not args.with_frameworks or framework_returncode == 0)
    )
    manifest.update(
        {
            "completed_at": utc_now(),
            "status": "complete" if complete else "incomplete",
            "completed_jobs": sum(row["status"] == "completed" for row in records),
            "failed_jobs": sum(row["status"] != "completed" for row in records),
            "closure_audits": closure_audits,
            "framework_returncode": framework_returncode,
            "framework_lock_check": framework_lock_check,
            # Keep the legacy name bound to the core result.
            "final_lock_check": final_core_lock_check,
            "final_core_lock_check": final_core_lock_check,
            "final_framework_lock_check": final_framework_lock_check,
            "lock_drift_detected": lock_drift_detected,
            "lock_checks_completed": len(lock_checks),
        }
    )
    _write_json_atomic(run_dir / "run_manifest.json", manifest)
    print(json.dumps({"complete": complete, "run_dir": str(run_dir)}, indent=2))
    return 0 if complete else 1


def parse_cases(values: Sequence[str]) -> tuple[str, ...]:
    if not values or "all" in values:
        return SUPPLEMENTAL_CASE_STUDIES
    unknown = set(values) - set(SUPPLEMENTAL_CASE_STUDIES)
    if unknown:
        raise ValueError(f"unknown supplemental cases: {sorted(unknown)}")
    return tuple(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "freeze", "preflight", "run"))
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--cases", nargs="+", default=["all"])
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--framework-lock",
        type=Path,
        help=(
            "separate full-source/framework lock; defaults to --lock for "
            "backward compatibility"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY)
    parser.add_argument("--brainpilot-url", default=DEFAULT_BRAINPILOT)
    parser.add_argument("--with-frameworks", action="store_true")
    parser.add_argument("--verify-full-kg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = parse_cases(args.cases)
    profile = evaluation_profile_for(args.phase)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "profile": asdict(profile),
                    "cases": list(cases),
                    "external_required": list(EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES),
                    "external_exempt": list(EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES),
                    "brain_age_discovery": "HCP-YA",
                    "kg_release_id": CURRENT_CANONICAL_RELEASE_ID,
                    "formal_run_requires_code_lock": True,
                    "scope_exclusions": ["hindcasting", "knowledge_graph_updates"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.action == "freeze":
        if args.lock is None:
            raise ValueError("freeze requires --lock")
        payload = freeze_code_lock(
            args.lock,
            args.phase,
            verify_full_kg=args.verify_full_kg,
        )
        print(json.dumps({"lock": str(args.lock.resolve()), "lock_id": payload["lock_id"]}, indent=2))
        return 0
    if args.action == "preflight":
        report = preflight(
            phase=args.phase,
            lock_path=args.lock,
            framework_lock_path=args.framework_lock,
            with_frameworks=args.with_frameworks,
            gateway_url=args.gateway_url,
            brainpilot_url=args.brainpilot_url,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready"] else 1
    if args.lock is None:
        raise ValueError("run requires --lock")
    return execute_run(args, cases)


if __name__ == "__main__":
    raise SystemExit(main())
