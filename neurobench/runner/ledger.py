"""One campaign ledger; every attempt and its usage remain independently auditable."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from neurobench.runner.artifacts import file_sha256, json_text, sha256_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 condition TEXT NOT NULL, run_group INTEGER NOT NULL, case_id TEXT NOT NULL,
 task_number INTEGER NOT NULL, task_path TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 condition_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL DEFAULT '',
 requested_model TEXT NOT NULL, driver TEXT NOT NULL, harness_enabled INTEGER NOT NULL,
 report_model TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
 started_at TEXT NOT NULL DEFAULT '', finished_at TEXT NOT NULL DEFAULT '',
 session_id TEXT NOT NULL DEFAULT '', response_sha256 TEXT NOT NULL DEFAULT '',
 report_path TEXT NOT NULL DEFAULT '', report_sha256 TEXT NOT NULL DEFAULT '',
 elapsed_seconds REAL NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
 cached_input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
 tool_event_count INTEGER NOT NULL DEFAULT 0, skills_loaded_json TEXT NOT NULL DEFAULT '[]',
 events_path TEXT NOT NULL DEFAULT '', stderr_path TEXT NOT NULL DEFAULT '',
 last_error TEXT NOT NULL DEFAULT '', retry_class TEXT NOT NULL DEFAULT '',
 PRIMARY KEY (condition,case_id)
);
CREATE TABLE IF NOT EXISTS attempts (
 condition TEXT NOT NULL, case_id TEXT NOT NULL, attempt INTEGER NOT NULL,
 status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT '',
 prompt_sha256 TEXT NOT NULL, receipt_path TEXT NOT NULL DEFAULT '',
 input_tokens INTEGER NOT NULL DEFAULT 0, cached_input_tokens INTEGER NOT NULL DEFAULT 0,
 cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0, uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
 output_tokens INTEGER NOT NULL DEFAULT 0, usage_complete INTEGER NOT NULL DEFAULT 0,
 remote_outcome_unknown INTEGER NOT NULL DEFAULT 0, retry_class TEXT NOT NULL DEFAULT '',
 outcome_json TEXT NOT NULL DEFAULT '{}', report_json TEXT NOT NULL DEFAULT '{}',
 PRIMARY KEY(condition,case_id,attempt)
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(condition,status,task_number);
CREATE TABLE IF NOT EXISTS campaign_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
 condition TEXT NOT NULL DEFAULT '', case_id TEXT NOT NULL DEFAULT '', payload_json TEXT NOT NULL DEFAULT '{}'
);
PRAGMA user_version=2;
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path, *, readonly=False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path), timeout=60)
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
        if exists and connection.execute("PRAGMA user_version").fetchone()[0] != 2:
            connection.close()
            raise RuntimeError("Legacy ledger is historical evidence; use a new output directory")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
    connection.row_factory = sqlite3.Row
    return connection


def record_event(connection, event_type, *, condition="", case_id="", payload=None):
    connection.execute("INSERT INTO campaign_events(timestamp,event_type,condition,case_id,payload_json) VALUES(?,?,?,?,?)",
                       (utc_now(), event_type, condition, case_id, json.dumps(payload or {}, ensure_ascii=False)))


def seed_condition(connection, condition, tasks, *, prompts=None):
    condition_sha = sha256_text(json_text(dataclasses.asdict(condition)))
    inserted = 0
    with connection:
        for task in tasks:
            prompt_sha = sha256_text(prompts[task.case_id]) if prompts is not None else ""
            row = connection.execute("SELECT * FROM runs WHERE condition=? AND case_id=?",
                                     (condition.key, task.case_id)).fetchone()
            if row:
                if (row["source_sha256"], row["condition_sha256"], row["prompt_sha256"]) != (
                        task.source_sha256, condition_sha, prompt_sha):
                    raise RuntimeError("Frozen task/condition/prompt changed: %s/%s" % (condition.key, task.case_id))
                continue
            connection.execute(
                """INSERT INTO runs(condition,run_group,case_id,task_number,task_path,source_sha256,
                   condition_sha256,prompt_sha256,requested_model,driver,harness_enabled,report_model)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (condition.key, condition.group, task.case_id, task.number, str(task.path), task.source_sha256,
                 condition_sha, prompt_sha, condition.model, condition.driver, int(condition.harness_enabled),
                 condition.report_model))
            inserted += 1
    return inserted


def counts(connection, condition_key):
    result = dict.fromkeys(("pending", "running", "success", "failed", "held_unknown"), 0)
    result.update({r[0]: r[1] for r in connection.execute(
        "SELECT status,COUNT(*) FROM runs WHERE condition=? GROUP BY status", (condition_key,))})
    return result


def mark_started(connection, condition, task, *, receipt_path=""):
    with connection:
        row = connection.execute("SELECT * FROM runs WHERE condition=? AND case_id=?",
                                 (condition.key, task.case_id)).fetchone()
        if row is None or row["status"] not in ("pending", "failed"):
            raise RuntimeError("Row is not dispatchable")
        if row["status"] == "failed" and row["retry_class"] not in ("transient", "rate_limit"):
            raise RuntimeError("A completed answer or unknown outcome cannot be automatically resent")
        attempt, started = row["attempts"] + 1, utc_now()
        connection.execute("""INSERT INTO attempts(condition,case_id,attempt,status,started_at,prompt_sha256,receipt_path)
                              VALUES(?,?,?,'running',?,?,?)""",
                           (condition.key, task.case_id, attempt, started, row["prompt_sha256"], str(receipt_path)))
        connection.execute("""UPDATE runs SET status='running',attempts=?,started_at=?,finished_at='',
                              last_error='',retry_class='' WHERE condition=? AND case_id=?""",
                           (attempt, started, condition.key, task.case_id))
        record_event(connection, "attempt_started", condition=condition.key, case_id=task.case_id,
                     payload={"attempt": attempt, "prompt_sha256": row["prompt_sha256"]})
    return attempt


def outcome_record(outcome):
    data = dataclasses.asdict(outcome)
    # Full response and native CLI events are separate immutable artifacts.
    for key in ("raw_stdout", "raw_stderr", "response", "raw_response"):
        data.pop(key, None)
    return data


def mark_finished(connection, condition, task, outcome, *, report, max_attempts):
    row = connection.execute("SELECT attempts,status FROM runs WHERE condition=? AND case_id=?",
                             (condition.key, task.case_id)).fetchone()
    if row is None or row["status"] != "running":
        raise RuntimeError("Attempt has already settled or was never started")
    attempt = row["attempts"]
    if outcome.remote_outcome_unknown:
        status = "held_unknown"
    elif outcome.ok:
        status = "success"
    elif outcome.retry_class in ("transient", "rate_limit") and attempt < max_attempts:
        status = "pending"
    else:
        status = "failed"
    finished = utc_now()
    data = outcome_record(outcome)
    with connection:
        connection.execute(
            """UPDATE attempts SET status=?,finished_at=?,input_tokens=?,cached_input_tokens=?,
               cache_creation_input_tokens=?,uncached_input_tokens=?,output_tokens=?,usage_complete=?,
               remote_outcome_unknown=?,retry_class=?,outcome_json=?,report_json=?
               WHERE condition=? AND case_id=? AND attempt=? AND status='running'""",
            ("held_unknown" if outcome.remote_outcome_unknown else ("success" if outcome.ok else "failed"), finished, outcome.input_tokens, outcome.cached_input_tokens,
             outcome.cache_creation_input_tokens, outcome.uncached_input_tokens, outcome.output_tokens,
             int(outcome.usage_complete), int(outcome.remote_outcome_unknown), outcome.retry_class,
             json.dumps(data, ensure_ascii=False), json.dumps(report, ensure_ascii=False),
             condition.key, task.case_id, attempt))
        connection.execute(
            """UPDATE runs SET status=?,finished_at=?,session_id=?,response_sha256=?,report_path=?,
               report_sha256=?,elapsed_seconds=?,input_tokens=?,cached_input_tokens=?,output_tokens=?,
               tool_event_count=?,skills_loaded_json=?,events_path=?,stderr_path=?,last_error=?,retry_class=?
               WHERE condition=? AND case_id=?""",
            (status, finished, outcome.session_id, report.get("response_sha256", ""), report.get("report_path", ""),
             report.get("report_sha256", ""), outcome.elapsed_seconds, outcome.input_tokens,
             outcome.cached_input_tokens, outcome.output_tokens, outcome.tool_event_count,
             json.dumps(outcome.skills_loaded, ensure_ascii=False), report.get("events_path", ""),
             report.get("stderr_path", ""), outcome.error, outcome.retry_class, condition.key, task.case_id))
        record_event(connection, "attempt_succeeded" if outcome.ok else "attempt_failed",
                     condition=condition.key, case_id=task.case_id,
                     payload={"status": status, "attempt": attempt, "retry_class": outcome.retry_class,
                              "error": outcome.error[:1000], "usage_complete": outcome.usage_complete})
    return status


def recover_interrupted(connection):
    """Recover a verified receipt without dispatch; otherwise preserve UNKNOWN."""
    from neurobench.runner.drivers import AttemptOutcome
    rows = connection.execute("""SELECT a.*,r.source_sha256 FROM attempts a JOIN runs r
                  ON a.condition=r.condition AND a.case_id=r.case_id
                  WHERE a.status='running' AND r.status='running'""").fetchall()
    for row in rows:
        error = "Interrupted attempt has no verified terminal receipt"
        try:
            receipt = json.loads(Path(row["receipt_path"]).read_text(encoding="utf-8"))
            for name in ("condition", "case_id", "attempt", "prompt_sha256", "source_sha256"):
                if receipt[name] != row[name]:
                    raise ValueError("Receipt binding mismatch: " + name)
            report = receipt["report"]
            for prefix in ("response", "raw_response", "events", "stderr", "report"):
                path = report.get(prefix + "_path")
                if path and file_sha256(path) != report[prefix + "_sha256"]:
                    raise ValueError("Artifact hash mismatch: " + prefix)
            outcome = AttemptOutcome(**receipt["outcome"])
            outcome.response = Path(report["response_path"]).read_text(encoding="utf-8")
            raw_path = report.get("raw_response_path")
            if raw_path:
                outcome.raw_response = Path(raw_path).read_text(encoding="utf-8")
            mark_finished(connection, SimpleNamespace(key=row["condition"]), SimpleNamespace(case_id=row["case_id"]),
                          outcome, report=report, max_attempts=receipt["max_attempts"])
            continue
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error += ": " + type(exc).__name__
        with connection:
            connection.execute("""UPDATE attempts SET status='held_unknown',finished_at=?,remote_outcome_unknown=1,
                                  retry_class='unknown' WHERE condition=? AND case_id=? AND attempt=?""",
                               (utc_now(), row["condition"], row["case_id"], row["attempt"]))
            connection.execute("""UPDATE runs SET status='held_unknown',retry_class='unknown',last_error=?
                                  WHERE condition=? AND case_id=?""", (error, row["condition"], row["case_id"]))
            record_event(connection, "interrupted_attempt_held", condition=row["condition"], case_id=row["case_id"],
                         payload={"attempt": row["attempt"], "error": error})


def totals(connection):
    row = connection.execute("""SELECT COUNT(*) AS attempts,
             COALESCE(SUM(input_tokens),0) AS input_tokens, COALESCE(SUM(output_tokens),0) AS output_tokens,
             COALESCE(SUM(cached_input_tokens),0) AS cached_input_tokens,
             COALESCE(SUM(cache_creation_input_tokens),0) AS cache_creation_input_tokens,
             COALESCE(SUM(usage_complete=0),0) AS usage_incomplete_attempts,
             COALESCE(SUM(remote_outcome_unknown),0) AS unknown_remote_attempts FROM attempts""").fetchone()
    return dict(row)
