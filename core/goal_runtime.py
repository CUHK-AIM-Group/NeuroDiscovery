"""Durable, bounded Goal mode state for the NeuroRuntime web queue.

Goals are deliberately a small controller around the existing queue.  The
controller records intent, turn budget and explicit lifecycle transitions; it
does not create a second agent loop or silently resume work after a restart.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path
from typing import Any

from core.runtime_store import RuntimeStore


GOAL_STATES = frozenset({"active", "paused", "blocked", "verifying", "complete", "cancelled"})
NON_TERMINAL_GOAL_STATES = frozenset({"active", "paused", "blocked", "verifying"})
GOAL_MARKERS = frozenset({"complete", "blocked"})
_MARKER_RE = re.compile(r"^\s*<goal:(complete|blocked)>\s*$", re.IGNORECASE)


def goal_marker(text: str) -> str | None:
    """Return an explicit lifecycle marker outside fenced code blocks.

    A marker must occupy the final nonblank line. This prevents an example in
    a code block, quoted material, or ordinary prose from changing durable state.
    """

    lines = str(text or "").splitlines()
    final = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()), -1)
    fenced = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced or index != final:
            continue
        if len(raw_line) - len(raw_line.lstrip()) >= 4:
            continue
        match = _MARKER_RE.match(raw_line)
        if match:
            return match.group(1).lower()
    return None


def _validate_text(value: Any, field: str, limit: int, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{field} is required")
    if len(text) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    return text


class GoalStore:
    """SQLite-backed goal lifecycle shared with :class:`RuntimeStore`."""

    def __init__(self, path: Path):
        self.store = RuntimeStore(Path(path))
        with self.store.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_goals (
                  goal_id TEXT PRIMARY KEY,
                  chat_id TEXT NOT NULL,
                  workspace_path TEXT NOT NULL,
                  objective TEXT NOT NULL,
                  criterion TEXT NOT NULL,
                  status TEXT NOT NULL,
                  max_turns INTEGER NOT NULL,
                  turns_started INTEGER NOT NULL,
                  active_request_id TEXT,
                  last_request_id TEXT,
                  last_report TEXT NOT NULL,
                  verification TEXT NOT NULL,
                  created REAL NOT NULL,
                  updated REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS runtime_one_unfinished_goal
                  ON runtime_goals(chat_id)
                  WHERE status NOT IN ('complete', 'cancelled');
                CREATE INDEX IF NOT EXISTS runtime_goals_chat_updated
                  ON runtime_goals(chat_id, updated DESC);
                """
            )

    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        for key in ("verification",):
            try:
                result[key] = json.loads(result[key]) if result[key] else None
            except (TypeError, ValueError):
                result[key] = {"raw": str(result[key])}
        return result

    def create(
        self,
        chat_id: str,
        workspace: Path,
        objective: str,
        criterion: str = "",
        max_turns: int = 12,
        goal_id: str | None = None,
    ) -> dict[str, Any]:
        chat_id = _validate_text(chat_id, "Conversation", 200, required=True)
        objective = _validate_text(objective, "Objective", 8000, required=True)
        criterion = _validate_text(criterion, "Acceptance criterion", 4000)
        if type(max_turns) is not int or not 1 <= max_turns <= 40:
            raise ValueError("Goal turn budget must be an integer from 1 to 40")
        workspace_path = Path(workspace).resolve()
        identifier = str(goal_id or ("goal_" + secrets.token_urlsafe(16))).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", identifier):
            raise ValueError("Invalid goal_id")
        now = time.time()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO runtime_goals
                    (goal_id,chat_id,workspace_path,objective,criterion,status,max_turns,
                     turns_started,active_request_id,last_request_id,last_report,verification,created,updated)
                    VALUES(?,?,?,?,?,'active',?,0,NULL,NULL,'','',?,?)""",
                    (identifier, chat_id, str(workspace_path), objective, criterion, max_turns, now, now),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise ValueError("This conversation already has an unfinished goal") from exc
                raise
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (identifier,)).fetchone()
        return self._public(dict(row))

    def get(self, goal_id: str, chat_id: str | None = None) -> dict[str, Any] | None:
        with self.store.connect() as connection:
            if chat_id is None:
                row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
            else:
                row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=? AND chat_id=?", (goal_id, chat_id)).fetchone()
        return self._public(dict(row)) if row else None

    def list(self, chat_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM runtime_goals WHERE chat_id=? ORDER BY updated DESC, created DESC", (chat_id,)
            )]
        return [self._public(row) for row in rows]

    def pause_on_restart(self) -> None:
        """Stop live goals at process boundaries; never resume them implicitly."""
        now = time.time()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE runtime_goals SET status='paused', active_request_id=NULL,
                   last_report=?, updated=? WHERE status IN ('active', 'verifying')""",
                ("Backend restarted; inspect the saved run and explicitly resume.", now),
            )
            # A queued Goal turn was never dispatched.  Retain its durable row,
            # but cancel it so an explicit resume creates a fresh request rather
            # than silently replaying stale work.
            connection.execute(
                """UPDATE runtime_queue SET status='cancelled'
                   WHERE status='queued' AND json_extract(payload, '$.goal_id') IN
                     (SELECT goal_id FROM runtime_goals WHERE status='paused')"""
            )

    def bind_request(self, goal_id: str, request_id: str) -> dict[str, Any]:
        request_id = _validate_text(request_id, "Request ID", 128, required=True)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
            if not row:
                raise ValueError("Goal not found")
            if row["active_request_id"] and row["active_request_id"] != request_id:
                raise ValueError("Goal already has an active turn")
            if row["status"] != "active":
                raise ValueError("Goal is not ready for another turn")
            if row["turns_started"] >= row["max_turns"]:
                connection.execute(
                    "UPDATE runtime_goals SET status='paused',last_report=?,updated=? WHERE goal_id=?",
                    ("Goal turn budget exhausted; explicitly resume is not available until the budget is increased.", time.time(), goal_id),
                )
                raise ValueError("Goal turn budget exhausted")
            now = time.time()
            connection.execute(
                """UPDATE runtime_goals SET status='active', turns_started=turns_started+1,
                   active_request_id=?, updated=? WHERE goal_id=?""",
                (request_id, now, goal_id),
            )
            updated = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._public(dict(updated))

    def abort_unqueued_request(self, goal_id: str, request_id: str, reason: str) -> dict[str, Any]:
        """Return a turn reservation that never reached the durable queue."""
        reason = _validate_text(reason, "Queue failure", 8000, required=True)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
            if not row or row["active_request_id"] != request_id or row["status"] != "active":
                raise ValueError("Goal request ownership changed")
            if connection.execute("SELECT 1 FROM runtime_queue WHERE request_id=?", (request_id,)).fetchone():
                raise ValueError("Queued goal requests cannot be released")
            connection.execute(
                """UPDATE runtime_goals SET status='paused', active_request_id=NULL,
                   turns_started=turns_started-1, last_report=?, updated=? WHERE goal_id=?""",
                (reason, time.time(), goal_id),
            )
            updated = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._public(dict(updated))

    def record_turn(self, goal_id: str, request_id: str, result: dict[str, Any], execution_status: str) -> dict[str, Any]:
        """Record one queue turn and return the resulting lifecycle state."""

        content = str(result.get("content") or result.get("message") or "") if isinstance(result, dict) else ""
        marker = goal_marker(content)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
            if not row:
                raise ValueError("Goal not found")
            if row["status"] in {"paused", "cancelled"} and not row["active_request_id"]:
                return self._public(dict(row))
            if row["active_request_id"] != request_id:
                raise ValueError("Goal request ownership changed")
            now = time.time()
            current_status = str(row["status"])
            if current_status == "cancelled":
                status = "cancelled"
            elif marker == "complete" and execution_status == "completed":
                status = "verifying"
            elif marker == "blocked" and execution_status == "completed":
                status = "blocked"
            elif execution_status == "completed" and row["turns_started"] < row["max_turns"]:
                status = "active"
            elif execution_status == "completed":
                status = "paused"
            else:
                status = "paused"
            report = content[-12000:]
            if execution_status != "completed" and not report:
                report = f"Turn ended with status {execution_status}."
            connection.execute(
                """UPDATE runtime_goals SET status=?,active_request_id=NULL,last_request_id=?,
                   last_report=?,updated=? WHERE goal_id=?""",
                (status, request_id, report, now, goal_id),
            )
            updated = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._public(dict(updated))

    def mark_verification(
        self, goal_id: str, accepted: bool | None, reason: str, *,
        request_id: str | None = None, review_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reason = _validate_text(reason, "Verification reason", 8000)
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
            if not row:
                raise ValueError("Goal not found")
            if row["status"] != "verifying":
                raise ValueError("Goal is not awaiting verification")
            if request_id and row["last_request_id"] != request_id:
                raise ValueError("Verification request does not match the latest goal turn")
            status = (
                "complete" if accepted is True
                else ("active" if accepted is False and row["turns_started"] < row["max_turns"] else "paused")
            )
            now = time.time()
            verification = json.dumps(
                {"accepted": accepted, "reason": reason, "review": review_record},
                ensure_ascii=False,
            )
            connection.execute(
                "UPDATE runtime_goals SET status=?,verification=?,last_report=?,updated=? WHERE goal_id=?",
                (status, verification, reason, now, goal_id),
            )
            updated = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._public(dict(updated))

    def control(self, goal_id: str, chat_id: str, action: str) -> dict[str, Any]:
        if action not in {"pause", "resume", "cancel"}:
            raise ValueError("Unsupported goal action")
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=? AND chat_id=?", (goal_id, chat_id)).fetchone()
            if not row:
                raise ValueError("Goal not found")
            status = str(row["status"])
            if action == "pause":
                if status in {"complete", "cancelled"}:
                    raise ValueError("Goal is terminal")
                next_status = "paused"
                report = "Paused by explicit user action."
            elif action == "resume":
                if status in {"complete", "cancelled"}:
                    raise ValueError("Goal is terminal")
                if status not in {"paused", "blocked"}:
                    raise ValueError("Only a paused or blocked goal can resume")
                if row["turns_started"] >= row["max_turns"]:
                    raise ValueError("Goal turn budget is exhausted")
                next_status = "active"
                report = "Resumed by explicit user action."
            else:
                if status in {"complete", "cancelled"}:
                    raise ValueError("Goal is terminal")
                next_status = "cancelled"
                report = "Cancelled by explicit user action."
            now = time.time()
            connection.execute(
                "UPDATE runtime_goals SET status=?,active_request_id=?,last_report=?,updated=? WHERE goal_id=?",
                (next_status, None, report, now, goal_id),
            )
            updated = connection.execute("SELECT * FROM runtime_goals WHERE goal_id=?", (goal_id,)).fetchone()
        return self._public(dict(updated))
