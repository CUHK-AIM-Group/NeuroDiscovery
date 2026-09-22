"""Execute one blinded autoresearch-framework job for an executable case study."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from core.scripts.case_study_native_output import (
    anchors_from_validated,
    extract_json,
    validate_ranked_hypotheses,
)
from core.scripts.case_study_search_policy import (
    PolicyRule,
    SearchPolicy,
    policy_to_payload,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
OFFICIAL_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
METHODS = (*OFFICIAL_METHODS, *NATIVE_METHODS)
REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
}


def _repository_python(method: str, repo: Path) -> Path:
    env_names = (
        f"CASE_STUDY_{method.upper()}_PYTHON",
        f"CS1_{method.upper()}_PYTHON",
    )
    candidates = [
        *(Path(os.environ[name]) for name in env_names if os.environ.get(name)),
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"no Python runtime found for {method}")


def _secret(local_base_url: str | None = None) -> str:
    for name in (
        "SUB2API_OPENAI_API_KEY",
        "CASE_STUDY_LOCAL_API_KEY",
        "CS1_LOCAL_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if local_base_url and any(
        marker in local_base_url.casefold()
        for marker in ("127.0.0.1", "localhost", "[::1]")
    ):
        return "neuroclaw-local-router"
    raise RuntimeError("SUB2API_OPENAI_API_KEY is required")


def _scrub(text: str, secret: str) -> str:
    cleaned = str(text).replace(secret, "<redacted>")
    if len(secret) >= 8:
        cleaned = cleaned.replace(secret[-8:], "<redacted-suffix>")
    return cleaned


def _ensure_directory_with_retry(path: Path, *, attempts: int = 8) -> None:
    """Create an output directory despite short-lived SMB disconnects."""

    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(min(4.0, 0.25 * (2 ** (attempt - 1))))


def _write_text_with_retry(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    attempts: int = 8,
) -> int:
    """Write a text artifact with bounded retries for transient SMB failures."""

    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.write_text(text, encoding=encoding)
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(min(4.0, 0.25 * (2 ** (attempt - 1))))
    raise AssertionError("unreachable")


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    out_dir: Path,
    timeout_seconds: int,
    max_retries: int,
    secret: str,
) -> None:
    _ensure_directory_with_retry(out_dir)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        started = time.time()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout after {timeout_seconds}s: {_scrub(str(exc), secret)}"
            _write_text_with_retry(
                out_dir / f"launcher_attempt_{attempt:02d}.stderr.log",
                last_error,
            )
        else:
            _write_text_with_retry(
                out_dir / f"launcher_attempt_{attempt:02d}.stdout.log",
                _scrub(result.stdout, secret),
            )
            _write_text_with_retry(
                out_dir / f"launcher_attempt_{attempt:02d}.stderr.log",
                _scrub(result.stderr, secret),
            )
            _write_text_with_retry(
                out_dir / f"launcher_attempt_{attempt:02d}.duration.txt",
                f"{time.time() - started:.3f}\n",
            )
            if result.returncode == 0:
                return
            last_error = f"child process exited {result.returncode}"
        if attempt < max_retries:
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(last_error or "framework child process failed")


def _without_selected_candidates(goal: str, previous_ids: list[str]) -> str:
    """Drop prior exact-ID rows so later native batches stay context bounded."""

    selected = set(previous_ids)
    if not selected:
        return goal
    remaining: list[str] = []
    removed: set[str] = set()
    for line in goal.splitlines():
        candidate_id = ""
        if line.startswith("- ") and " :: " in line:
            candidate_id = line[2:].split(" :: ", 1)[0].strip()
        if candidate_id in selected:
            removed.add(candidate_id)
            continue
        remaining.append(line)
    missing = selected - removed
    if missing:
        raise ValueError(
            "previous native candidate IDs were not present in the public registry: "
            + ", ".join(sorted(missing))
        )
    return "\n".join(remaining)


def _native_prompt(
    task: dict[str, Any],
    *,
    method: str,
    start_rank: int,
    end_rank: int,
    previous_ids: list[str],
) -> str:
    delivery = (
        "Deliver the final JSON through BrainPilot's result_deliver tool."
        if method == "brainpilot_native"
        else "Put the final JSON inside one <solution>...</solution> tag."
    )
    previous = "\n".join(f"- {value}" for value in previous_ids) or "- none"
    if str(task.get("policy_mode") or "") == "factor_rules":
        rule_fields = ", ".join(task.get("policy_rule_fields") or [])
        return f"""{task['research_goal']}

Generate exactly rule ranks {start_rank}-{end_rank}, each once. Do not repeat
these earlier valid rule signatures:
{previous}

Return only this object:
{{
  "method": "{method}",
  "rules": [
    {{"rank": {start_rank}, "weight": 0.75, "when": {{"disease": "exact registered value"}}, "rationale": "one sentence"}}
  ]
}}
Allowed rule fields are: {rule_fields}. Every value must be copied exactly from
the compact registry in the task. {delivery}
"""
    remaining_goal = _without_selected_candidates(
        str(task["research_goal"]), previous_ids
    )
    return f"""{remaining_goal}

Generate exactly ranks {start_rank}-{end_rank}, each once. Candidates selected
in earlier batches have been removed from the registry; select only verbatim IDs
that remain visible there.

Return only this object:
{{
  "method": "{method}",
  "hypotheses": [
    {{"rank": {start_rank}, "candidate_id": "exact registered ID", "rationale": "one sentence", "confidence": 0.75}}
  ]
}}
{delivery}
"""


def _iter_rule_objects(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("when"), dict):
            yield value
        for nested in value.values():
            yield from _iter_rule_objects(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_rule_objects(nested)


def _compile_native_rules(
    *,
    method: str,
    task: dict[str, Any],
    registry: pd.DataFrame,
    payloads: list[tuple[int, int, dict[str, Any] | None, str | None]],
) -> tuple[SearchPolicy, pd.DataFrame]:
    fields = tuple(str(field) for field in task.get("policy_rule_fields") or ())
    allowed = {
        field: set(registry[field].fillna("").astype(str)) for field in fields
    }
    rows: list[dict[str, Any]] = []
    rules: list[PolicyRule] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for start_rank, end_rank, parsed, parse_error in payloads:
        if parsed is None:
            for rank in range(start_rank, end_rank + 1):
                rows.append(
                    {
                        "rank": rank,
                        "valid": False,
                        "reason": parse_error or "missing_payload",
                        "signature": "",
                    }
                )
            continue
        native_rules = list(_iter_rule_objects(parsed))
        for offset, raw in enumerate(native_rules):
            try:
                rank = int(raw.get("rank", start_rank + offset))
                weight = float(raw.get("weight"))
            except (TypeError, ValueError):
                rank = start_rank + offset
                weight = float("nan")
            when = {
                str(key): str(value).strip()
                for key, value in (raw.get("when") or {}).items()
            }
            errors = []
            if rank < start_rank or rank > end_rank:
                errors.append("rank_out_of_batch")
            if not np.isfinite(weight) or not -1.0 <= weight <= 1.0:
                errors.append("invalid_weight")
            if not when:
                errors.append("empty_when")
            for field, value in when.items():
                if field not in allowed:
                    errors.append(f"unknown_field:{field}")
                elif value not in allowed[field]:
                    errors.append(f"unknown_value:{field}={value}")
            signature = tuple(sorted(when.items()))
            if signature in seen:
                errors.append("duplicate")
            valid = not errors and len(rules) < int(task["n_anchors"])
            if valid:
                seen.add(signature)
                rules.append(
                    PolicyRule(
                        weight=weight,
                        when=when,
                        rationale=str(raw.get("rationale") or "")[:1000],
                    )
                )
            rows.append(
                {
                    "rank": rank,
                    "valid": valid,
                    "reason": "|".join(errors),
                    "signature": json.dumps(when, sort_keys=True, ensure_ascii=False),
                    "weight": weight,
                }
            )
    policy = SearchPolicy(
        method=method,
        trial=int(task["trial"]),
        schema_version=str(task["policy_schema_version"]),
        rules=tuple(rules),
        metadata={
            "adapter": "neuroruntime_native_factor_rules",
            "rule_matching": "hierarchical_partial_plus_joint",
            "requested_slots": int(task["n_anchors"]),
            "valid_rule_count": len(rules),
            "failed_slots": max(0, int(task["n_anchors"]) - len(rules)),
            "native_retrieval_enabled": False,
        },
    )
    return policy, pd.DataFrame(rows)


def _run_official(payload: dict[str, Any], secret: str) -> None:
    method = str(payload["method"])
    out_dir = Path(payload["output_dir"])
    repo = BASELINES_ROOT / REPOSITORIES[method]
    command = [
        str(_repository_python(method, repo)),
        str(ROOT / "core/scripts/case_study_official_adapter_client.py"),
        "--method",
        method,
        "--task",
        str(Path(payload["task_path"])),
        "--out",
        str(out_dir),
        "--repo",
        str(repo),
        "--model",
        str(payload["model"]),
        "--base-url",
        str(payload["chat_base_url"]),
        "--reasoning-effort",
        str(payload["reasoning_effort"]),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CASE_STUDY_LOCAL_API_KEY": secret,
            "CS1_LOCAL_API_KEY": secret,
            "OPENAI_API_KEY": secret,
            "OPENAI_BASE_URL": str(payload["chat_base_url"]),
            "OPENAI_API_BASE": str(payload["chat_base_url"]),
            "OPENAI_REASONING_EFFORT": str(payload["reasoning_effort"]),
            "OPENAI_DISABLE_RESPONSE_STORAGE": "true",
            "CASE_STUDY_LLM_REQUEST_TIMEOUT": str(
                payload.get("request_timeout_seconds", 1800)
            ),
            "OPEN_COSCIENTIST_MAX_CONCURRENT_LLM_CALLS": "1",
            "OPEN_COSCIENTIST_TRANSIENT_RETRIES": "2",
            "OPEN_COSCIENTIST_MAX_OUTPUT_TOKENS": "3000",
            "OPEN_COSCIENTIST_DEBATE_TURNS": "3",
            "SCIAGENTS_LLM_MAX_RETRIES": "0",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    _run_process(
        command,
        cwd=repo,
        env=env,
        out_dir=out_dir,
        timeout_seconds=int(payload["timeout_seconds"]),
        max_retries=int(payload["max_retries"]),
        secret=secret,
    )


def _run_native_batch(
    payload: dict[str, Any],
    *,
    prompt_path: Path,
    batch_dir: Path,
    secret: str,
) -> str:
    method = str(payload["method"])
    final_path = batch_dir / "final.txt"
    if final_path.is_file() and not bool(payload.get("force")):
        return final_path.read_text(encoding="utf-8")

    env = os.environ.copy()
    if method == "brainpilot_native":
        env.update(
            {
                "ANTHROPIC_API_KEY": secret,
                "ANTHROPIC_MODEL": str(payload["model"]),
                "BP_THINKING_LEVEL": str(payload["reasoning_effort"]),
            }
        )
        command = [
            "node",
            str(ROOT / "core/scripts/brainpilot_case_study_batch_client.mjs"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(batch_dir),
            "--client-dist",
            str(BASELINES_ROOT / "BrainPilot/packages/client-cli/dist/index.js"),
            "--base-url",
            str(payload["brainpilot_url"]),
            "--max-events",
            "1000",
        ]
        cwd = BASELINES_ROOT / "BrainPilot"
    else:
        env.update(
            {
                "BIOMNI_API_KEY": secret,
                "OPENAI_DISABLE_RESPONSE_STORAGE": "true",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        command = [
            str(BASELINES_ROOT / "Biomni/.venv/Scripts/python.exe"),
            str(ROOT / "core/scripts/biomni_case_study_batch_client.py"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(batch_dir),
            "--workspace",
            str(batch_dir / "workspace"),
            "--biomni-root",
            str(BASELINES_ROOT / "Biomni"),
            "--model",
            str(payload["model"]),
            "--base-url",
            str(payload["responses_base_url"]),
            "--reasoning-effort",
            str(payload["reasoning_effort"]),
        ]
        cwd = BASELINES_ROOT / "Biomni"

    _run_process(
        command,
        cwd=cwd,
        env=env,
        out_dir=batch_dir,
        timeout_seconds=int(payload["timeout_seconds"]),
        max_retries=int(payload["max_retries"]),
        secret=secret,
    )
    if not final_path.is_file():
        raise RuntimeError(f"{method} produced no final.txt")
    return final_path.read_text(encoding="utf-8")


def _run_native(payload: dict[str, Any], task: dict[str, Any], secret: str) -> None:
    method = str(payload["method"])
    out_dir = Path(payload["output_dir"])
    registry = pd.read_json(task["public_registry_path"], lines=True)
    candidate_ids = set(registry["candidate_id"].astype(str))
    factor_rule_mode = str(task.get("policy_mode") or "") == "factor_rules"
    n_anchors = int(task["n_anchors"])
    batch_size = int(payload["batch_size"])
    previous_ids: list[str] = []
    payloads: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
    for start_rank in range(1, n_anchors + 1, batch_size):
        end_rank = min(n_anchors, start_rank + batch_size - 1)
        batch_dir = out_dir / f"batch_{start_rank:03d}_{end_rank:03d}"
        _ensure_directory_with_retry(batch_dir)
        prompt_path = batch_dir / "prompt.txt"
        _write_text_with_retry(
            prompt_path,
            _native_prompt(
                task,
                method=method,
                start_rank=start_rank,
                end_rank=end_rank,
                previous_ids=previous_ids,
            ),
        )
        final_text = _run_native_batch(
            payload, prompt_path=prompt_path, batch_dir=batch_dir, secret=secret
        )
        try:
            parsed = extract_json(final_text)
            error = None
        except Exception as exc:
            parsed = None
            error = f"invalid_native_output:{type(exc).__name__}"
            _write_text_with_retry(
                batch_dir / "parse_error.txt",
                _scrub(str(exc), secret),
            )
        if parsed:
            if factor_rule_mode:
                for item in _iter_rule_objects(parsed):
                    when = item.get("when") or {}
                    signature = json.dumps(
                        when, sort_keys=True, ensure_ascii=False
                    )
                    if signature not in previous_ids:
                        previous_ids.append(signature)
            else:
                for item in parsed.get("hypotheses") or []:
                    if not isinstance(item, dict):
                        continue
                    candidate_id = str(item.get("candidate_id") or "")
                    if candidate_id in candidate_ids and candidate_id not in previous_ids:
                        previous_ids.append(candidate_id)
        payloads.append((start_rank, end_rank, parsed, error))

    if factor_rule_mode:
        policy, validated = _compile_native_rules(
            method=method,
            task=task,
            registry=registry,
            payloads=payloads,
        )
    else:
        validated = validate_ranked_hypotheses(
            method=method,
            trial=int(task["trial"]),
            payloads=payloads,
            candidate_ids=candidate_ids,
        )
        anchors = anchors_from_validated(validated)
        policy = SearchPolicy(
            method=method,
            trial=int(task["trial"]),
            schema_version=str(task["policy_schema_version"]),
            anchors=anchors,
            metadata={
                "adapter": "neuroruntime_native",
                "requested_slots": n_anchors,
                "valid_anchor_count": len(anchors),
                "failed_slots": int((~validated["valid"]).sum()),
                "native_retrieval_enabled": False,
            },
        )
    _write_text_with_retry(
        out_dir / "native_validation.csv",
        validated.to_csv(index=False),
    )
    _write_text_with_retry(
        out_dir / "search_policy.json",
        json.dumps(policy_to_payload(policy), indent=2, ensure_ascii=False),
    )


def execute_framework_job(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("method") or "")
    if method not in METHODS:
        raise ValueError(f"unsupported framework method: {method}")
    task_path = Path(payload["task_path"])
    out_dir = Path(payload["output_dir"])
    _ensure_directory_with_retry(out_dir)
    policy_path = out_dir / "search_policy.json"
    if policy_path.is_file() and not bool(payload.get("force")):
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        return {
            "method": method,
            "trial": int(policy["trial"]),
            "status": "reused",
            "search_policy": str(policy_path),
            "valid_anchors": len(policy.get("anchors") or [])
            + len(policy.get("rules") or []),
        }

    local_base_url = str(
        payload.get("responses_base_url")
        or payload.get("chat_base_url")
        or ""
    )
    secret = _secret(local_base_url)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    started = time.time()
    if method in OFFICIAL_METHODS:
        _run_official(payload, secret)
    else:
        _run_native(payload, task, secret)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    result = {
        "method": method,
        "trial": int(task["trial"]),
        "task": str(task["case_study_id"]),
        "status": "complete",
        "duration_seconds": time.time() - started,
        "search_policy": str(policy_path),
        "valid_anchors": len(policy.get("anchors") or [])
        + len(policy.get("rules") or []),
        "requested_anchors": int(task["n_anchors"]),
    }
    _write_text_with_retry(
        out_dir / "neuroruntime_result.json",
        json.dumps(result, indent=2, ensure_ascii=False),
    )
    return result
