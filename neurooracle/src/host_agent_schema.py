"""Schemas for host-agent-driven NeuroOracle autoresearch.

The host-agent mode lets Codex, Claude Code, Cursor, or another installed host
agent use its own reasoning model while NeuroClaw owns the run state and
auditable file protocol.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "host-agent-autoresearch/v1"

VALID_RESULT_STATUS = {"continue", "complete", "blocked"}
VALID_RUN_STATUS = {"waiting_for_host_agent", "complete", "blocked"}


def task_result_template(task_id: str, round_index: int) -> dict[str, Any]:
    """Return the JSON shape a host agent should write for one round."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "round": round_index,
        "status": "continue",
        "hypotheses": [
            {
                "id": "H1",
                "title": "",
                "claim": "",
                "rationale": "",
                "evidence_refs": [],
                "graph_nodes": [],
                "novelty_rationale": "",
                "testability": "",
                "risks": [],
                "next_experiment": "",
            }
        ],
        "critic_review": [
            {
                "hypothesis_id": "H1",
                "verdict": "pass|revise|reject",
                "score": 0.0,
                "issues": [],
                "required_checks": [],
            }
        ],
        "experiment_plan": [
            {
                "name": "",
                "objective": "",
                "commands": [],
                "expected_outputs": [],
                "stop_conditions": [],
            }
        ],
        "result_supervision": {
            "reviewed_artifacts": [],
            "interpretation": "",
            "failure_modes": [],
        },
        "next_round": {
            "decision": "continue|stop|blocked",
            "reason": "",
            "focus": "",
        },
    }


def validate_task_result(data: dict[str, Any], task_id: str, round_index: int) -> list[str]:
    """Validate the host agent's JSON output and return human-readable errors."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["result must be a JSON object"]

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if data.get("task_id") != task_id:
        errors.append(f"task_id must be {task_id!r}")
    if data.get("round") != round_index:
        errors.append(f"round must be {round_index}")
    if data.get("status") not in VALID_RESULT_STATUS:
        errors.append("status must be one of continue, complete, blocked")

    for key in ("hypotheses", "critic_review", "experiment_plan"):
        if key not in data:
            errors.append(f"missing required key: {key}")
        elif not isinstance(data[key], list):
            errors.append(f"{key} must be a list")

    if "result_supervision" not in data:
        errors.append("missing required key: result_supervision")
    elif not isinstance(data["result_supervision"], dict):
        errors.append("result_supervision must be an object")

    next_round = data.get("next_round")
    if not isinstance(next_round, dict):
        errors.append("next_round must be an object")
    else:
        decision = next_round.get("decision")
        if decision not in {"continue", "stop", "blocked"}:
            errors.append("next_round.decision must be continue, stop, or blocked")

    return errors
