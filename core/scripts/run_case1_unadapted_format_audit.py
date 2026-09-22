"""Measure strict CS1 format compliance from clean upstream frameworks.

The audit deliberately avoids the NeuroClaw task adapters.  A clean detached
checkout of each upstream repository receives the same one-shot prompt.  Raw
native artifacts are preserved and evaluated without retries, fuzzy mapping,
LLM repair, slot completion, or manual correction.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)

REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}

CANDIDATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_/-])"
    r"([A-Za-z0-9_./-]+\|[A-Za-z0-9_./-]+\|[A-Za-z0-9_./-]+\|"
    r"[A-Za-z0-9_./-]+\|[0-9]+)"
    r"(?![A-Za-z0-9_/-])"
)


class UnsupportedOfficialInterface(RuntimeError):
    """The clean upstream repository has no generic task entry point."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-hypotheses", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--brainpilot-url")
    parser.add_argument("--child-method", choices=METHODS)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--prompt", type=Path)
    return parser.parse_args()


def load_registry(path: Path) -> tuple[set[str], list[str], list[str], list[dict[str, str]]]:
    candidate_ids: set[str] = set()
    diseases: set[str] = set()
    features: set[str] = set()
    anatomy: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = str(row.get("candidate_id") or "")
            anatomy_id = str(row.get("anatomy_id") or "")
            if candidate_id:
                candidate_ids.add(candidate_id)
            if row.get("disease"):
                diseases.add(str(row["disease"]))
            if row.get("feature"):
                features.add(str(row["feature"]))
            if anatomy_id and anatomy_id not in anatomy:
                anatomy[anatomy_id] = {
                    "anatomy_id": anatomy_id,
                    "anatomy_full": str(row.get("anatomy_full") or ""),
                }
    return candidate_ids, sorted(diseases), sorted(features), [anatomy[key] for key in sorted(anatomy)]


def build_prompt(
    *,
    trial: int,
    n_hypotheses: int,
    diseases: list[str],
    features: list[str],
    anatomy: list[dict[str, str]],
) -> str:
    disease_lines = "\n".join(f"- {value}" for value in diseases)
    feature_lines = "\n".join(f"- {value}" for value in features)
    anatomy_lines = "\n".join(
        f"- {row['anatomy_id']} = {row['anatomy_full']}" for row in anatomy
    )
    return f"""Blinded Case Study 1 transdiagnostic neuroimaging task.

Independent trial: {trial}

Generate exactly {n_hypotheses} ranked, distinct and testable hypotheses. Each
hypothesis must combine one allowed disease, one allowed imaging feature and one
allowed anatomy coordinate. Do not inspect experimental outcomes, effect sizes,
p-values, FDR labels, ground-truth labels or NeuroDiscovery scores.

Allowed disease codes:
{disease_lines}

Allowed imaging feature codes:
{feature_lines}

Allowed anatomy coordinates:
{anatomy_lines}

Construct candidate_id exactly as:
  modality|source|disease|feature|roi_index
where anatomy_id supplies modality|source|roi_index.

Return exactly one JSON object and no prose:
{{
  "hypotheses": [
    {{
      "rank": 1,
      "candidate_id": "one exact registered candidate_id",
      "rationale": "one concise scientific rationale",
      "confidence": 0.75
    }}
  ]
}}

Ranks must be exactly 1 through {n_hypotheses}. Do not repeat candidate IDs.
"""


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def run_ai_scientist(repo: Path, prompt: str, out_dir: Path, model: str, **_: Any) -> Any:
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from ai_scientist.llm import create_client
    from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

    ideas_path = out_dir / "official_ideas.json"
    client, client_model = create_client(model)
    return generate_temp_free_idea(
        idea_fname=str(ideas_path),
        client=client,
        model=client_model,
        workshop_description=prompt,
        max_num_generations=1,
        num_reflections=3,
        reload_ideas=False,
    )


async def _run_open_coscientist(repo: Path, prompt: str, model: str) -> Any:
    sys.path.insert(0, str(repo / "src"))
    from open_coscientist import HypothesisGenerator

    model_name = model if "/" in model else f"openai/{model}"
    generator = HypothesisGenerator(
        model_name=model_name,
        max_iterations=2,
        initial_hypotheses_count=7,
        evolution_max_count=4,
        enable_cache=False,
    )
    return await generator.generate_hypotheses(
        research_goal=prompt,
        opts={
            "enable_literature_review_node": False,
            "enable_tool_calling_generation": False,
        },
        stream=False,
        run_id=f"raw-format-{int(time.time())}",
    )


def run_open_coscientist(repo: Path, prompt: str, model: str, **_: Any) -> Any:
    os.chdir(repo)
    return asyncio.run(_run_open_coscientist(repo, prompt, model))


def run_virtual_lab(repo: Path, prompt: str, out_dir: Path, model: str, **_: Any) -> Any:
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)
    from virtual_lab import Agent, run_meeting

    lead = Agent(
        title="principal investigator",
        expertise="neuroimaging",
        goal="complete the supplied scientific task",
        role="deliver the final answer",
        model=model,
    )
    member = Agent(
        title="neuroimaging scientist",
        expertise="functional and structural neuroimaging",
        goal="complete the supplied scientific task",
        role="propose and check hypotheses",
        model=model,
    )
    summary = run_meeting(
        meeting_type="team",
        agenda=prompt,
        save_dir=out_dir,
        save_name="official_discussion",
        team_lead=lead,
        team_members=(member,),
        num_rounds=0,
        pubmed_search=False,
        return_summary=True,
    )
    return {"summary": summary}


def run_biomni(
    repo: Path,
    prompt: str,
    out_dir: Path,
    model: str,
    base_url: str,
    **_: Any,
) -> Any:
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from biomni.agent import A1

    agent = A1(
        path=str(out_dir / "workspace"),
        llm=model,
        source="Custom",
        base_url=base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        use_tool_retriever=False,
        expected_data_lake_files=[],
    )
    log, final = agent.go(prompt)
    (out_dir / "official_agent_log.json").write_text(
        json.dumps(jsonable(log), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"final": final}


def run_sciagents(repo: Path, **_: Any) -> Any:
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    try:
        import ScienceDiscovery.agents  # noqa: F401
    except Exception as exc:
        raise UnsupportedOfficialInterface(
            "clean SciAgents is a materials-specific notebook pipeline and its import "
            f"cannot expose a generic task entry point: {type(exc).__name__}: {exc}"
        ) from exc
    raise UnsupportedOfficialInterface(
        "clean SciAgents exports a fixed materials graph workflow, not a generic "
        "research-goal entry point"
    )


def child_main(args: argparse.Namespace) -> int:
    if not args.repo or not args.prompt:
        raise ValueError("--repo and --prompt are required in child mode")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8")
    runners = {
        "ai_scientist_v2": run_ai_scientist,
        "open_coscientist": run_open_coscientist,
        "sciagents": run_sciagents,
        "virtual_lab": run_virtual_lab,
        "biomni_native": run_biomni,
    }
    if args.child_method == "brainpilot_native":
        raise UnsupportedOfficialInterface("BrainPilot is launched through its official CLI")
    runner = runners[str(args.child_method)]
    try:
        result = runner(
            repo=args.repo,
            prompt=prompt,
            out_dir=args.out_dir,
            model=args.model,
            base_url=args.base_url,
        )
    except UnsupportedOfficialInterface as exc:
        (args.out_dir / "unsupported_official_interface.txt").write_text(
            str(exc), encoding="utf-8"
        )
        print(str(exc), file=sys.stderr)
        return 86
    (args.out_dir / "native_result.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def balanced_json_objects(text: str) -> list[Any]:
    values: list[Any] = []
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            values.append(value)
    return values


def nested_values(value: Any, depth: int = 0) -> list[Any]:
    if depth > 10:
        return []
    values = [value]
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(nested_values(nested, depth + 1))
    elif isinstance(value, list):
        for nested in value:
            values.extend(nested_values(nested, depth + 1))
    elif isinstance(value, str):
        for parsed in balanced_json_objects(value):
            values.extend(nested_values(parsed, depth + 1))
    return values


def evaluate_artifact(
    artifact: Any,
    *,
    known_ids: set[str],
    requested: int,
) -> dict[str, Any]:
    text = json.dumps(artifact, ensure_ascii=False, default=str)
    exact_ids = []
    seen_exact: set[str] = set()
    invalid_mentions: set[str] = set()
    for candidate_id in CANDIDATE_PATTERN.findall(text.replace(r"\|", "|")):
        if candidate_id in known_ids:
            if candidate_id not in seen_exact:
                exact_ids.append(candidate_id)
                seen_exact.add(candidate_id)
        else:
            invalid_mentions.add(candidate_id)

    best_valid = 0
    best_total = 0
    strict_payloads = 0
    best_errors: list[str] = []
    for value in nested_values(artifact):
        if not isinstance(value, dict) or not isinstance(value.get("hypotheses"), list):
            continue
        strict_payloads += 1
        items = value["hypotheses"]
        best_total = max(best_total, len(items))
        ranks: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                rank = int(item.get("rank"))
            except (TypeError, ValueError):
                continue
            ranks.setdefault(rank, []).append(item)
        valid = 0
        errors: list[str] = []
        used_ids: set[str] = set()
        for rank in range(1, requested + 1):
            candidates = ranks.get(rank, [])
            if len(candidates) != 1:
                errors.append(f"rank_{rank}_count_{len(candidates)}")
                continue
            item = candidates[0]
            candidate_id = str(item.get("candidate_id") or "").strip()
            rationale = item.get("rationale")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                confidence = float("nan")
            if candidate_id not in known_ids:
                errors.append(f"rank_{rank}_invalid_id")
            elif candidate_id in used_ids:
                errors.append(f"rank_{rank}_duplicate_id")
            elif not isinstance(rationale, str) or not rationale.strip():
                errors.append(f"rank_{rank}_missing_rationale")
            elif not 0.0 <= confidence <= 1.0:
                errors.append(f"rank_{rank}_invalid_confidence")
            else:
                valid += 1
                used_ids.add(candidate_id)
        if valid > best_valid:
            best_valid = valid
            best_errors = errors

    return {
        "native_artifact_produced": True,
        "strict_payloads_found": strict_payloads,
        "best_payload_item_count": best_total,
        "schema_valid_slots": best_valid,
        "schema_slot_success_rate": best_valid / requested,
        "complete_schema_success": best_valid == requested,
        "exact_unique_registered_ids_anywhere": len(exact_ids),
        "exact_id_coverage_rate": min(requested, len(exact_ids)) / requested,
        "invalid_candidate_mentions": len(invalid_mentions),
        "schema_errors": ";".join(best_errors[:40]),
    }


def runtime_python(runtime_root: Path, method: str) -> Path:
    repo = runtime_root / REPOSITORIES[method]
    candidate = repo / ".venv" / "Scripts" / "python.exe"
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def run_trial(
    *,
    args: argparse.Namespace,
    method: str,
    trial: int,
    prompt_path: Path,
    known_ids: set[str],
) -> dict[str, Any]:
    trial_dir = args.out_dir / method / f"trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    repo = args.clean_root / REPOSITORIES[method]
    started = time.time()
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = args.base_url
    env["OPENAI_API_BASE"] = args.base_url
    env["OPENAI_REASONING_EFFORT"] = "high"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if method == "brainpilot_native":
        if not args.brainpilot_url:
            return {
                "method": method,
                "trial": trial,
                "process_status": "unsupported_provider_or_service",
                "process_returncode": None,
                "duration_seconds": time.time() - started,
                "native_artifact_produced": False,
                "error_type": "MissingBrainPilotService",
                "error": "clean BrainPilot service URL was not supplied",
            }
        command = [
            "node",
            str(Path(__file__).with_name("run_brainpilot_unadapted_client.mjs")),
            str(repo),
            args.brainpilot_url,
            str(prompt_path),
        ]
    else:
        command = [
            str(runtime_python(args.runtime_root, method)),
            str(Path(__file__).resolve()),
            "--child-method",
            method,
            "--repo",
            str(repo),
            "--prompt",
            str(prompt_path),
            "--out-dir",
            str(trial_dir),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
        ]
    try:
        completed = subprocess.run(
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
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        if returncode == 0:
            process_status = "success"
            error_type = ""
        elif returncode == 86:
            process_status = "unsupported_official_interface"
            error_type = "UnsupportedOfficialInterface"
        else:
            process_status = "failed"
            error_type = "OfficialProcessFailure"
        error = "" if returncode == 0 else stderr[-2000:]
    except subprocess.TimeoutExpired as exc:
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        returncode = None
        process_status = "timeout"
        error_type = "TimeoutExpired"
        error = str(exc)

    (trial_dir / "official.stdout.log").write_text(stdout, encoding="utf-8")
    (trial_dir / "official.stderr.log").write_text(stderr, encoding="utf-8")
    native_path = trial_dir / "native_result.json"
    if method == "brainpilot_native" and stdout.strip():
        native_path.write_text(
            json.dumps({"official_cli_stdout": stdout}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    row: dict[str, Any] = {
        "method": method,
        "trial": trial,
        "process_status": process_status,
        "process_returncode": returncode,
        "duration_seconds": time.time() - started,
        "native_artifact_produced": native_path.is_file(),
        "error_type": error_type,
        "error": error.replace("\n", " ")[:2000],
    }
    stdout_metrics = evaluate_artifact(
        {"official_stdout": stdout},
        known_ids=known_ids,
        requested=args.n_hypotheses,
    )
    row.update(
        {
            "stdout_strict_payloads_found": stdout_metrics["strict_payloads_found"],
            "stdout_schema_valid_slots": stdout_metrics["schema_valid_slots"],
            "stdout_schema_slot_success_rate": stdout_metrics[
                "schema_slot_success_rate"
            ],
            "stdout_complete_schema_success": stdout_metrics[
                "complete_schema_success"
            ],
            "stdout_exact_unique_registered_ids_anywhere": stdout_metrics[
                "exact_unique_registered_ids_anywhere"
            ],
            "stdout_exact_id_coverage_rate": stdout_metrics["exact_id_coverage_rate"],
        }
    )
    if native_path.is_file():
        artifact = json.loads(native_path.read_text(encoding="utf-8"))
        row.update(
            evaluate_artifact(
                artifact,
                known_ids=known_ids,
                requested=args.n_hypotheses,
            )
        )
    else:
        row.update(
            {
                "strict_payloads_found": 0,
                "best_payload_item_count": 0,
                "schema_valid_slots": 0,
                "schema_slot_success_rate": 0.0,
                "complete_schema_success": False,
                "exact_unique_registered_ids_anywhere": 0,
                "exact_id_coverage_rate": 0.0,
                "invalid_candidate_mentions": 0,
                "schema_errors": "",
            }
        )
    (trial_dir / "audit.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        process_success = sum(row["process_status"] == "success" for row in group)
        artifacts = sum(bool(row.get("native_artifact_produced")) for row in group)
        valid_slots = sum(int(row.get("schema_valid_slots", 0)) for row in group)
        exact_ids = sum(
            min(requested, int(row.get("exact_unique_registered_ids_anywhere", 0)))
            for row in group
        )
        denominator = requested * len(group)
        stdout_valid_slots = sum(
            int(row.get("stdout_schema_valid_slots", 0)) for row in group
        )
        output.append(
            {
                "method": method,
                "trials": len(group),
                "official_process_success_trials": process_success,
                "official_process_success_rate": process_success / len(group),
                "native_artifact_trials": artifacts,
                "native_artifact_rate": artifacts / len(group),
                "requested_slots": denominator,
                "strict_schema_valid_slots": valid_slots,
                "strict_schema_slot_success_rate": valid_slots / denominator,
                "exact_registered_ids_anywhere": exact_ids,
                "exact_registered_id_coverage_rate": exact_ids / denominator,
                "complete_schema_trials": sum(
                    bool(row.get("complete_schema_success")) for row in group
                ),
                "stdout_strict_schema_valid_slots": stdout_valid_slots,
                "stdout_strict_schema_slot_success_rate": stdout_valid_slots
                / denominator,
                "unsupported_official_interface_trials": sum(
                    row["process_status"] == "unsupported_official_interface"
                    for row in group
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if args.child_method:
        return child_main(args)
    if not args.registry or not args.clean_root or not args.runtime_root:
        raise ValueError("--registry, --clean-root and --runtime-root are required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be provided through the environment")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    known_ids, diseases, features, anatomy = load_registry(args.registry)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    jobs: list[tuple[str, int, Path]] = []
    for trial in args.trials:
        for method in args.methods:
            trial_dir = args.out_dir / method / f"trial_{trial:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = trial_dir / "prompt.txt"
            prompt_path.write_text(
                build_prompt(
                    trial=trial,
                    n_hypotheses=args.n_hypotheses,
                    diseases=diseases,
                    features=features,
                    anatomy=anatomy,
                ),
                encoding="utf-8",
            )
            jobs.append((method, trial, prompt_path))

    method_locks = {"brainpilot_native": threading.Lock()}

    def execute(method: str, trial: int, prompt_path: Path) -> dict[str, Any]:
        lock = method_locks.get(method)
        if lock is None:
            return run_trial(
                args=args,
                method=method,
                trial=trial,
                prompt_path=prompt_path,
                known_ids=known_ids,
            )
        with lock:
            return run_trial(
                args=args,
                method=method,
                trial=trial,
                prompt_path=prompt_path,
                known_ids=known_ids,
            )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as pool:
        futures = {
            pool.submit(execute, method, trial, prompt_path): (method, trial)
            for method, trial, prompt_path in jobs
        }
        for future in as_completed(futures):
            method, trial = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"{method} trial={trial} process={row['process_status']} "
                f"schema={row.get('schema_valid_slots', 0)}/{args.n_hypotheses} "
                f"exact={row.get('exact_unique_registered_ids_anywhere', 0)}",
                flush=True,
            )
    rows.sort(key=lambda row: (METHODS.index(str(row["method"])), int(row["trial"])))
    write_csv(args.out_dir / "unadapted_trial_audit.csv", rows)
    summary = aggregate(rows, args.n_hypotheses)
    write_csv(args.out_dir / "unadapted_format_summary.csv", summary)
    manifest = {
        "schema_version": "case1-unadapted-format-audit.v1",
        "created_at": utc_now(),
        "protocol": {
            "clean_detached_upstream_commits": True,
            "identical_prompt_across_frameworks": True,
            "one_shot_per_trial": True,
            "scientific_output_retries": 0,
            "fuzzy_mapping": False,
            "llm_repair": False,
            "manual_repair": False,
            "requested_hypotheses_per_trial": args.n_hypotheses,
            "trials": args.trials,
            "max_workers": args.max_workers,
            "brainpilot_trials_serialized": True,
        },
        "model": args.model,
        "base_url": args.base_url,
        "registry": {
            "path": str(args.registry),
            "sha256": sha256_file(args.registry),
            "candidate_count": len(known_ids),
            "disease_count": len(diseases),
            "feature_count": len(features),
            "anatomy_count": len(anatomy),
        },
        "clean_repositories": {
            method: str(args.clean_root / REPOSITORIES[method]) for method in args.methods
        },
        "outputs": {
            "trials": "unadapted_trial_audit.csv",
            "summary": "unadapted_format_summary.csv",
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
