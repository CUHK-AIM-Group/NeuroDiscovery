"""Evidence-bounded AutoResearch turn lifecycle; no model or experiment dispatch here."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from core.autoresearch import build_autoresearch_scope_prompt, normalize_autoresearch_mode


FINISH_TOOL_NAME = "finish_autoresearch"
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL_NAME,
        "description": (
            "Finish AutoResearch only after ALL requested deliverables are checked, or report a genuine hard "
            "blocker. Ordinary text without this tool is treated as progress, not completion. Call this alone, "
            "after execution/validation tools return. Never mark an intermediate artifact as the entire task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["completed", "blocked"]},
                "summary": {"type": "string", "description": "User-facing final report in the user's language."},
                "artifacts": {"type": "array", "items": {"type": "string"},
                              "description": "Actual nonempty deliverable files inside the workspace; not plans/placeholders."},
                "validation": {"type": "string", "description": "Checks performed, outcomes and limitations; not a promise to check."},
                "evidence_ids": {"type": "array", "items": {"type": "integer"},
                                 "description": "autoresearch_evidence_id values from relevant actual tool results."},
                "blocker_kind": {"type": "string", "enum": ["missing_input", "access", "authorization", "budget", "execution"]},
                "missing_requirement": {"type": "string", "description": "Exactly what must change before execution can resume."},
                "attempted_recovery": {"type": "string", "description": "Checks and safe alternatives actually tried; why none can proceed."},
            },
            "required": ["status", "summary", "artifacts", "validation", "evidence_ids"],
        },
    },
}


def iteration_limit() -> int | None:
    """No ordinary-chat 8-round cutoff; an explicitly configured cap still binds."""
    raw = os.environ.get("NEUROCLAW_MAX_TOOL_ITERATIONS", "").strip()
    if not raw:
        return None
    # A malformed explicit budget is not permission for unlimited execution.
    value = int(raw)
    if value < 1:
        raise ValueError("NEUROCLAW_MAX_TOOL_ITERATIONS must be a positive integer")
    return value


class AutoResearchRun:
    def __init__(self, workspace: Path, mode: object):
        self.workspace = workspace.resolve()
        self.mode = normalize_autoresearch_mode(mode)
        self.limit = iteration_limit()
        self.path = self.workspace / ".neurodiscovery" / "autoresearch" / uuid.uuid4().hex / "run.json"
        self.state = {"mode": self.mode, "status": "running", "iterations": 0,
                      "iteration_limit": self.limit, "artifacts": [], "evidence": [],
                      "started_at": self._now(), "state_path": str(self.path)}
        self._text_only = 0
        self._failed_attempts: deque[str] = deque(maxlen=12)
        self.save()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        self.state["updated_at"] = self._now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def prompt(self) -> str:
        return (
            build_autoresearch_scope_prompt(self.mode)
            + "\n[AutoResearch runtime contract]\n"
            + "Continue calling tools in this same turn until finish_autoresearch accepts a completed or blocked report. "
            + "Text-only progress or a request for routine confirmation does not end this turn. "
            + "Use evidence IDs from returned tool results, not invented IDs. The runtime verifies file existence "
            + "and recorded tool execution, not scientific truth: you must still check every requested deliverable. "
            + "Put delivery manifests/reports in the workspace, including references to authorized external outputs. "
            + "Do not report a running background job as complete; do not use fire-and-forget delegation. "
            + f"Progress receipt: {self.path}. This is runtime bookkeeping, NOT a research deliverable. "
            + (f"Explicit model/tool iteration budget: {self.limit}. Do not increase it."
               if self.limit else "There is no default total round cutoff in AutoResearch; existing user budgets still apply.")
        )

    def begin_iteration(self) -> None:
        self.state["iterations"] += 1
        self.save()

    def needs_continuation(self) -> bool:
        self._text_only += 1
        return self._text_only < 4

    def observe(self, name: str, args: dict, result: dict) -> bool:
        """Record metadata, not credentials/prompts/tool stdout; detect repeated failures.

        Unchanged SUCCESSFUL job polling never counts as a failure/stall.
        """
        self._text_only = 0
        evidence_id = len(self.state["evidence"]) + 1
        result["autoresearch_evidence_id"] = evidence_id
        self.state["evidence"].append({
            "id": evidence_id, "tool": name, "success": bool(result.get("success")),
            "executed": result.get("executed", True) is not False,
        })
        if result.get("success"):
            self._failed_attempts.clear()
        else:
            # In-memory equality only. Never persist or hash command arguments (which may be sensitive).
            self._failed_attempts.append(json.dumps([name, args, result.get("error_type"), result.get("error")], sort_keys=True))
        stalled = bool(self._failed_attempts and Counter(self._failed_attempts).most_common(1)[0][1] >= 6)
        self.save()
        return stalled

    def finish(self, args: dict) -> dict:
        def reject(reason: str) -> dict:
            return {"success": False, "executed": False, "error_type": "incomplete_delivery", "error": reason,
                    "recovery_hint": "Continue the unfinished work or report a genuine evidenced blocker. Do not ask for routine confirmation."}

        status = args.get("status")
        if not isinstance(status, str) or status not in {"completed", "blocked"}:
            return reject("status must be completed or blocked")
        summary = args.get("summary")
        validation = args.get("validation")
        if not isinstance(summary, str) or not summary.strip() or not isinstance(validation, str) or not validation.strip():
            return reject("Provide a concrete final summary and actual validation/inspection outcomes.")
        ids = args.get("evidence_ids")
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            return reject("evidence_ids must contain actual integer tool evidence IDs.")
        evidence = {event["id"]: event for event in self.state["evidence"] if event["tool"] != FINISH_TOOL_NAME}
        if any(i not in evidence for i in ids):
            return reject("Unknown or invalid tool evidence ID.")
        paths = args.get("artifacts")
        if not isinstance(paths, list) or any(not isinstance(p, str) or not p.strip() for p in paths):
            return reject("artifacts must be a list of actual file paths.")
        artifacts = []
        for raw in paths:
            try:
                path = (self.workspace / raw).resolve()
                path.relative_to(self.workspace)
                # Bookkeeping files cannot be submitted to fake delivery.
                if path.is_relative_to(self.workspace / ".neurodiscovery" / "autoresearch"):
                    return reject("Runtime receipts are not research deliverables.")
                if not path.is_file() or path.stat().st_size == 0:
                    return reject(f"Deliverable is missing, empty or not a file: {raw}")
            except (ValueError, OSError, RuntimeError):
                return reject("Use an accessible workspace deliverable or a checked workspace manifest for external outputs.")
            artifacts.append(str(path))
        if status == "completed":
            if not artifacts or not any(evidence[i]["success"] and evidence[i]["executed"] for i in ids):
                return reject("Completion requires nonempty deliverable files and successful execution/validation evidence.")
        else:
            kind = args.get("blocker_kind")
            if not isinstance(kind, str) or kind not in {"missing_input", "access", "authorization", "budget", "execution"}:
                return reject("Report a concrete hard blocker, not ordinary uncertainty or a plan awaiting confirmation.")
            for key in ("missing_requirement", "attempted_recovery"):
                if not isinstance(args.get(key), str) or not args[key].strip():
                    return reject(f"A blocked report requires {key}.")
            if not ids and kind not in {"authorization", "budget"}:
                return reject("Inspect the actual inputs/tools and safe alternatives before declaring an execution/input blocker.")
            self.state.update({key: args[key] for key in ("blocker_kind", "missing_requirement", "attempted_recovery")})
        self.state.update(status=status, summary=summary.strip(), validation=validation.strip(),
                          artifacts=artifacts, evidence_ids=ids)
        self.save()
        return {"success": True, "status": status}

    def halt(self, status: str, reason: str) -> str:
        """Operational interruption is explicitly incomplete, never a fabricated scientific blocker."""
        self.state.update(status=status, summary=reason)
        self.save()
        return self.response()

    def response(self) -> str:
        lines = [self.state.get("summary", ""), f"[AutoResearch: {self.state['status']}]"]
        if self.state.get("validation"):
            lines.append(self.state["validation"])
        lines.extend(f"- [{Path(path).name}]({Path(path).as_posix()})" for path in self.state["artifacts"])
        if self.state.get("missing_requirement"):
            lines.append(self.state["missing_requirement"])
        lines.append(f"[AutoResearch run record]({self.path.as_posix()})")
        return "\n\n".join(lines)
