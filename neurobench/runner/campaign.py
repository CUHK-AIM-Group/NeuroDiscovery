"""Unified, bounded campaign runner with immutable inputs and attempt receipts."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurobench.runner import conditions as conditions_mod
from neurobench.runner import ledger as ledger_mod
from neurobench.runner import prompts as prompts_mod
from neurobench.runner.artifacts import file_sha256, freeze_bytes, freeze_text, json_text, sha256_text
from neurobench.runner.cli_drivers import run_claude_code, run_codex_cli, write_claude_mcp_config, write_codex_home
from neurobench.runner.drivers import AttemptOutcome, SKILL_TOOLS, classify_retry, resolve_api_key, run_http_chat

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dev/outputs/neurobench_harness_10000_20261002"
DEFAULT_TIMEOUT = 1500
atomic_write_text = freeze_text  # Existing callers now receive immutable publication semantics.


def provider_of(condition):
    if condition.channel.rstrip("/") in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
        return "deepseek"
    return "nvidia" if "nvidia" in condition.channel else "ollama"


def _condition_slug(condition):
    return "".join(c if c.isalnum() or c in "-." else "_" for c in condition.key)


def _receipt_path(root, condition, task, attempt):
    return root / "attempts" / _condition_slug(condition) / task.case_id / ("attempt%d.json" % attempt)


def render_report(task, condition, outcome, *, prompt_sha256):
    thinking, code = prompts_mod.split_answer(outcome.response)
    tool_lines = ["%d. [tool] %s (success=%s)" % (i, e["name"], e["ok"])
                  for i, e in enumerate(outcome.tool_events, 1)] or ["1. No tool command was used."]
    header = prompts_mod.report_metadata_header(
        task, report_model=condition.report_model, actual_model=condition.model,
        harness_enabled=condition.harness_enabled, driver=condition.driver,
        session_id=outcome.session_id or "unavailable", prompt_sha256=prompt_sha256,
        response_sha256=sha256_text(outcome.response), elapsed_seconds=outcome.elapsed_seconds,
        input_tokens=outcome.input_tokens, cached_input_tokens=outcome.cached_input_tokens,
        output_tokens=outcome.output_tokens, skill_names=outcome.skills_loaded,
        tool_lines=tool_lines, reasoning_effort=outcome.reasoning_effort)
    extra = [
        "- Uncached input tokens: %d" % outcome.uncached_input_tokens,
        "- Cache creation input tokens: %d" % outcome.cache_creation_input_tokens,
        "- Usage complete: %s" % outcome.usage_complete,
        "- Returned model ids: " + json.dumps(outcome.model_ids),
        "- Answer normalization: " + str(outcome.runtime_settings.get("answer_normalization", "none")),
        "- Effective settings: " + json.dumps(outcome.runtime_settings, ensure_ascii=False, sort_keys=True),
        "- Gateway token caveat: " + ("Grok gateway prefix is included; report this model separately."
                                     if "grok" in condition.model else "none recorded"),
    ]
    return "\n".join([header, *extra, "", "## Solution Thinking", thinking, "",
                      "## Commands Or Code", code, "", "## Tool Events", *tool_lines, ""]), sha256_text(outcome.response)


def validate_outcome(outcome, condition):
    """Validate a terminal answer, preserving raw provider text separately.

    The only normalization is removal of a leading prose preamble before the
    required first heading.  It is structural contract repair, never content
    editing; ``run_one`` writes the original response beside the scored text.
    """
    if not outcome.ok:
        return
    if not outcome.raw_response:
        outcome.raw_response = outcome.response
    normalized, normalization = prompts_mod.normalize_answer_for_contract(outcome.response)
    if normalization != "none":
        outcome.response = normalized
        outcome.runtime_settings["answer_normalization"] = normalization
    allowed = set()
    if condition.harness_enabled:
        allowed = set(SKILL_TOOLS) | {"neuro_runtime/" + t for t in SKILL_TOOLS} | {
            "mcp__neuro_runtime__" + t for t in SKILL_TOOLS}
    forbidden = [e["name"] for e in outcome.tool_events if e["name"] not in allowed]
    error = prompts_mod.validate_answer_contract(outcome.response)
    if forbidden:
        outcome.ok, outcome.error, outcome.retry_class = False, "Forbidden tool use: " + ", ".join(forbidden), "policy"
    elif error:
        outcome.ok, outcome.error, outcome.retry_class = False, error, "contract"
    elif not outcome.usage_complete:
        outcome.ok, outcome.error, outcome.retry_class = False, "Provider did not supply complete usage", "accounting"


def run_one(*, output_root, condition, task, prompt, attempt, timeout_seconds, max_attempts=2,
            max_output_tokens=12000):
    started = time.monotonic()
    slug = _condition_slug(condition)
    receipt_path = _receipt_path(output_root, condition, task, attempt)
    control = output_root / "control" / slug / task.case_id / ("attempt%d" % attempt)
    workspace = control / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    prompt_sha = sha256_text(prompt)
    freeze_text(output_root / "prompts" / condition.key / (task.case_id + ".txt"), prompt)
    driver_name = {"http_chat": "http", "codex_cli": "codex", "claude_code": "claude"}[condition.driver]
    stem = "%s.%s.attempt%d" % (task.case_id, driver_name, attempt)
    events = output_root / "events" / slug / (stem + ".jsonl")
    stderr = output_root / "stderr" / slug / (stem + ".txt")
    try:
        verify_frozen_inputs(output_root)
        skills_dir = output_root / "skill_snapshot"
        if condition.driver == "http_chat":
            outcome = run_http_chat(prompt=prompt, model=condition.model, base_url=condition.channel,
                provider=provider_of(condition), harness_enabled=condition.harness_enabled, workspace=workspace,
                timeout_seconds=timeout_seconds, events_path=events, max_output_tokens=max_output_tokens,
                request_overrides={**(condition.request_overrides or {}),
                                   "reasoning_effort": condition.reasoning_effort}, skills_dir=skills_dir)
        elif condition.driver == "codex_cli":
            home = write_codex_home(control / "codex_home", base_url=condition.channel,
                                    harness_enabled=condition.harness_enabled, skills_dir=skills_dir)
            outcome = run_codex_cli(prompt=prompt, model=condition.model, base_url=condition.channel,
                harness_enabled=condition.harness_enabled, workspace=workspace, codex_home=home,
                output_root=output_root, case_id=task.case_id, attempt=attempt, timeout_seconds=timeout_seconds,
                reasoning_effort=condition.reasoning_effort, artifact_slug=slug)
        elif condition.driver == "claude_code":
            mcp = write_claude_mcp_config(control / "mcp.json", workspace=REPO_ROOT,
                                          harness_enabled=condition.harness_enabled, skills_dir=skills_dir)
            outcome = run_claude_code(prompt=prompt, model=condition.model, base_url=condition.channel,
                harness_enabled=condition.harness_enabled, workspace=workspace, output_root=output_root,
                case_id=task.case_id, attempt=attempt, timeout_seconds=timeout_seconds, mcp_config_path=mcp,
                artifact_slug=slug, claude_config_dir=control / "claude_config",
                reasoning_effort=condition.reasoning_effort)
        else:
            raise ValueError("Unknown driver: " + condition.driver)
    except Exception as exc:
        text = "%s: %s" % (type(exc).__name__, exc)
        retry_class = classify_retry(text)
        outcome = AttemptOutcome(False, text, retry_class, "", "", 0, 0, 0, time.monotonic()-started,
                                 usage_complete=False, remote_outcome_unknown=events.exists())
    validate_outcome(outcome, condition)
    try:
        verify_frozen_inputs(output_root)
    except (OSError, ValueError, RuntimeError) as exc:
        outcome.ok, outcome.error, outcome.retry_class = False, str(exc), "input_drift"
    response_path = receipt_path.with_suffix(".response.txt")
    freeze_text(response_path, outcome.response)
    raw_response = outcome.raw_response or outcome.response
    raw_response_path = receipt_path.with_suffix(".raw_response.txt")
    freeze_text(raw_response_path, raw_response)
    report = {"response_path": str(response_path), "response_sha256": file_sha256(response_path),
              "raw_response_path": str(raw_response_path), "raw_response_sha256": file_sha256(raw_response_path),
              "prompt_sha256": prompt_sha, "report_path": "", "report_sha256": "",
              "events_path": str(events) if events.exists() else "",
              "stderr_path": str(stderr) if stderr.exists() else ""}
    for prefix, path in (("events", events), ("stderr", stderr)):
        report[prefix + "_sha256"] = file_sha256(path) if path.exists() else ""
    if outcome.ok:
        markdown, _ = render_report(task, condition, outcome, prompt_sha256=prompt_sha)
        path = output_root / "reports" / condition.report_model / ("%d_%s_run1.md" % (task.number, condition.report_model))
        freeze_text(path, markdown)
        report.update(report_path=str(path), report_sha256=file_sha256(path))
    freeze_text(receipt_path, json_text({
        "condition": condition.key, "case_id": task.case_id, "attempt": attempt,
        "source_sha256": task.source_sha256, "prompt_sha256": prompt_sha,
        "outcome": ledger_mod.outcome_record(outcome) | {"response": ""},
        "report": report, "max_attempts": max_attempts}))
    return outcome, report


@contextlib.contextmanager
def campaign_lock(root):
    """OS lock is automatically released if the runner dies; no stale PID guesses."""
    path = root / ".runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another runner owns this output directory") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def freeze_manifest(root, conditions, tasks, args):
    # Bind runtime code that can affect a provider attempt.  Tests are an
    # offline gate and do not participate in the campaign process, so adding
    # a regression test must not invalidate an otherwise unchanged run.
    source_paths = [REPO_ROOT / "neurobench/__init__.py"] + sorted(p for p in (REPO_ROOT / "neurobench" / "runner").rglob("*.py")
                          if not p.name.startswith("test_")) + [
        REPO_ROOT / p for p in ("core/mcp_stdio.py", "core/tool_registry.py", "core/tools_impl.py",
          "core/tool_pipeline.py", "core/tool_outcomes.py", "core/permissions.py", "core/shell_safety.py",
          "core/research_reading.py", "core/config_flags.py", "core/config/features.json",
          "core/harness_skills.py", "core/skill_loader/loader.py", "core/agent/main.py")]
    sources = {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p) for p in source_paths}
    # SKILL.md contents and catalogs are the only model-accessible skill material.
    skills = {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p)
              for p in sorted((REPO_ROOT / "skills").glob("*/SKILL.md"))}
    import importlib.metadata
    runtime = {"python": sys.version, "packages": {
        name: importlib.metadata.version(name) for name in ("openai", "httpx")}}
    manifest = {"version": 2, "prompt_version": prompts_mod.PROMPT_VERSION,
                "conditions": [dataclasses.asdict(c) for c in conditions],
                "tasks": [{"case_id": t.case_id, "number": t.number, "sha256": t.source_sha256} for t in tasks],
                "source_sha256": sources, "skill_sha256": skills, "runtime": runtime,
                "budgets": {k: getattr(args, k) for k in ("workers", "timeout", "max_attempts",
                    "max_output_tokens", "max_dispatches", "max_total_tokens", "wall_seconds")},
                "exposure": "Previously published tasks; engineering acceptance, not blind validation"}
    freeze_text(root / "manifest.json", json_text(manifest))
    for path in source_paths:
        freeze_bytes(root / "source_snapshot" / path.relative_to(REPO_ROOT), path.read_bytes())
    for relative in skills:
        freeze_bytes(root / "skill_snapshot" / Path(relative).relative_to("skills"),
                     (REPO_ROOT / relative).read_bytes())
    for task in tasks:
        freeze_text(root / "task_snapshot" / (task.case_id + ".md"), task.text)
    verify_frozen_inputs(root, manifest)
    return manifest


def verify_frozen_inputs(root, manifest=None):
    """Check both running code and the exact skill snapshot actually served."""
    manifest = manifest or json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        for path in (REPO_ROOT / relative, root / "source_snapshot" / relative):
            if file_sha256(path) != expected:
                raise RuntimeError("Frozen source changed: " + relative)
    actual_skills = {"skills/" + p.relative_to(root / "skill_snapshot").as_posix(): file_sha256(p)
                     for p in (root / "skill_snapshot").glob("*/SKILL.md")}
    if actual_skills != manifest["skill_sha256"]:
        raise RuntimeError("Frozen skill catalog changed")
    for task in manifest["tasks"]:
        if file_sha256(root / "task_snapshot" / (task["case_id"] + ".md")) != task["sha256"]:
            raise RuntimeError("Frozen task changed: " + task["case_id"])


def run_condition(*, connection, output_root, condition, tasks, workers, timeout_seconds,
                  max_attempts, limit=None, retry_failed=False, max_output_tokens=12000,
                  budget_allows=lambda: True):
    by_id = {task.case_id: task for task in tasks}
    rows = connection.execute("SELECT * FROM runs WHERE condition=? ORDER BY task_number", (condition.key,)).fetchall()
    queue = [by_id[r["case_id"]] for r in rows if r["case_id"] in by_id and r["attempts"] < max_attempts
             and (r["status"] == "pending" or
                  (retry_failed and r["status"] == "failed" and r["retry_class"] in ("transient", "rate_limit")))]
    if limit is not None:
        # Limit applies to a frozen task selection in main, never to "next N unfinished".
        queue = [task for task in queue if task.case_id in {t.case_id for t in tasks[:limit]}]
    in_flight = {}
    print("[%s] queued=%d" % (condition.key, len(queue)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while queue or in_flight:
            while queue and len(in_flight) < workers and budget_allows():
                task = queue.pop(0)
                n = connection.execute("SELECT attempts FROM runs WHERE condition=? AND case_id=?",
                                       (condition.key, task.case_id)).fetchone()[0] + 1
                attempt = ledger_mod.mark_started(connection, condition, task,
                              receipt_path=_receipt_path(output_root, condition, task, n))
                future = executor.submit(run_one, output_root=output_root, condition=condition, task=task,
                    prompt=prompts_mod.render_prompt(task, harness_enabled=condition.harness_enabled), attempt=attempt,
                    timeout_seconds=timeout_seconds, max_attempts=max_attempts, max_output_tokens=max_output_tokens)
                in_flight[future] = task
            if not in_flight:
                break
            done, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = in_flight.pop(future)
                try:
                    outcome, report = future.result()
                except Exception as exc:
                    # Receipt publication failure may follow a completed remote response.
                    outcome = AttemptOutcome(False, "runner error: %s" % exc, "runner", "", "", 0, 0, 0, 0,
                                             usage_complete=False, remote_outcome_unknown=True)
                    report = {}
                status = ledger_mod.mark_finished(connection, condition, task, outcome,
                                                   report=report, max_attempts=max_attempts)
                print("  %s %s -> %s (%s)" % (condition.key, task.case_id, status,
                                             (outcome.error or "ok")[:160]), flush=True)
                if status == "pending":
                    queue.append(task)
    return ledger_mod.counts(connection, condition.key)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NeuroBench harness campaign")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-root", default=str(REPO_ROOT / "neurobench" / "tasks"))
    parser.add_argument("--conditions", default="")
    parser.add_argument("--group", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--max-dispatches", type=int, default=0, help="0 = condition/task matrix x max-attempts")
    parser.add_argument("--max-total-tokens", type=int, default=0, help="Stop dispatch at observed cumulative usage; in-flight calls may exceed it")
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Freeze inputs and seed ledger without model calls")
    args = parser.parse_args(argv)
    for k in ("workers", "timeout", "max_attempts", "max_output_tokens", "wall_seconds"):
        if getattr(args, k) <= 0:
            parser.error(k + " must be positive")
    if min(args.limit, args.max_dispatches, args.max_total_tokens) < 0:
        parser.error("limits must be nonnegative")
    return args


def select_conditions(args):
    selected = [c for c in conditions_mod.CONDITIONS.values() if not args.group or c.group == args.group]
    if args.conditions.strip():
        wanted = set(args.conditions.split(","))
        if wanted - set(conditions_mod.CONDITIONS):
            raise SystemExit("Unknown conditions: %s" % sorted(wanted - set(conditions_mod.CONDITIONS)))
        selected = [c for c in selected if c.key in wanted]
    if not selected:
        raise SystemExit("No conditions selected")
    return selected


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    if args.status:
        db = root / "state.sqlite3"
        if not db.exists():
            print("Ledger does not exist: " + str(db))
            return 0
        with contextlib.closing(ledger_mod.connect(db, readonly=True)) as con:
            print(json.dumps({c.key: ledger_mod.counts(con, c.key)
                              for c in conditions_mod.CONDITIONS.values()}, indent=2))
        return 0
    conditions = select_conditions(args)
    tasks = prompts_mod.discover_tasks(Path(args.benchmark_root).resolve())
    if args.task_ids:
        wanted = set(args.task_ids.split(","))
        if wanted - {t.case_id for t in tasks}:
            raise SystemExit("Unknown task ids")
        tasks = [t for t in tasks if t.case_id in wanted]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("No tasks selected")
    with campaign_lock(root), contextlib.closing(ledger_mod.connect(root / "state.sqlite3")) as con:
        freeze_manifest(root, conditions, tasks, args)
        for condition in conditions:
            prompts = {t.case_id: prompts_mod.render_prompt(t, harness_enabled=condition.harness_enabled) for t in tasks}
            for t in tasks:
                freeze_text(root / "prompts" / condition.key / (t.case_id + ".txt"), prompts[t.case_id])
            ledger_mod.seed_condition(con, condition, tasks, prompts=prompts)
        ledger_mod.recover_interrupted(con)
        if args.prepare:
            print("Prepared %d tasks x %d conditions; zero model calls" % (len(tasks), len(conditions)))
            return 0
        # Preflight all selected credentials before the first paid dispatch.
        for provider in {provider_of(c) for c in conditions}:
            resolve_api_key(provider)
        row = con.execute("SELECT timestamp FROM campaign_events WHERE event_type='execution_window_started' ORDER BY id LIMIT 1").fetchone()
        if row:
            began = dt.datetime.fromisoformat(row[0])
        else:
            began = dt.datetime.now(dt.timezone.utc)
            with con:
                ledger_mod.record_event(con, "execution_window_started")
        deadline = began.timestamp() + args.wall_seconds
        dispatch_cap = args.max_dispatches or len(tasks) * len(conditions) * args.max_attempts

        def budget_allows():
            totals = ledger_mod.totals(con)
            return (time.time() < deadline and totals["attempts"] < dispatch_cap
                    and (not args.max_total_tokens or totals["input_tokens"] + totals["output_tokens"] < args.max_total_tokens))

        summary = {}
        for condition in conditions:
            summary[condition.key] = run_condition(connection=con, output_root=root, condition=condition,
                tasks=tasks, workers=args.workers, timeout_seconds=args.timeout, max_attempts=args.max_attempts,
                retry_failed=args.retry_failed, max_output_tokens=args.max_output_tokens, budget_allows=budget_allows)
        print(json.dumps({"conditions": summary, "all_attempt_usage": ledger_mod.totals(con)}, indent=2))
        return 0 if all(v["success"] == len(tasks) for v in summary.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
