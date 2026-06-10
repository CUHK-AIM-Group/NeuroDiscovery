"""File-based host-agent autoresearch loop for NeuroOracle.

This module does not call an LLM API. It creates auditable task files that a
host agent such as Codex or Claude Code can read, solve with its own built-in
model, and write back as JSON. NeuroClaw then validates the result and advances
the run state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .host_agent_schema import (
    SCHEMA_VERSION,
    task_result_template,
    validate_task_result,
)


STATE_FILE = "run_state.json"
TASK_DIR = "tasks"
OUTPUT_DIR = "host_outputs"


def init_run(
    *,
    case_study: str,
    output_dir: str | Path,
    graph_path: str | Path,
    kge_path: str | Path | None = None,
    max_rounds: int = 5,
    deterministic_stages: str = "batch,novelty",
) -> dict[str, Any]:
    """Create a host-agent autoresearch run and its first task."""
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / TASK_DIR).mkdir(exist_ok=True)
    (run_dir / OUTPUT_DIR).mkdir(exist_ok=True)

    state = {
        "schema_version": SCHEMA_VERSION,
        "mode": "host_agent_autoresearch",
        "status": "waiting_for_host_agent",
        "case_study": case_study,
        "graph_path": str(graph_path),
        "kge_path": str(kge_path) if kge_path else "",
        "max_rounds": max_rounds,
        "current_round": 1,
        "deterministic_stages": deterministic_stages,
        "created_at": _now(),
        "updated_at": _now(),
        "rounds": [],
    }
    task = _create_round_task(state, run_dir, 1)
    state["rounds"].append(task["round_record"])
    _write_json(run_dir / STATE_FILE, state)
    return {"state": state, "task": task["payload"]}


def advance_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate current host-agent output and create the next round if needed."""
    run_path = Path(run_dir)
    state = _read_json(run_path / STATE_FILE)
    if state.get("status") != "waiting_for_host_agent":
        return {"state": state, "message": f"run is {state.get('status')}"}

    current = _current_round_record(state)
    result_path = run_path / current["expected_result_path"]
    if not result_path.exists():
        raise FileNotFoundError(
            f"host-agent result is missing: {result_path}. "
            "Write the JSON result before running host-agent-next."
        )

    result = _read_json(result_path)
    errors = validate_task_result(result, current["task_id"], current["round"])
    if errors:
        raise ValueError("invalid host-agent result:\n- " + "\n- ".join(errors))

    current["status"] = "accepted"
    current["accepted_at"] = _now()
    current["result_status"] = result.get("status")
    current["next_decision"] = result.get("next_round", {}).get("decision")

    should_stop = (
        result.get("status") in {"complete", "blocked"}
        or current["next_decision"] in {"stop", "blocked"}
        or int(state["current_round"]) >= int(state["max_rounds"])
    )
    if should_stop:
        state["status"] = "blocked" if result.get("status") == "blocked" or current["next_decision"] == "blocked" else "complete"
        state["completed_at"] = _now()
        state["updated_at"] = _now()
        _write_json(run_path / STATE_FILE, state)
        return {"state": state, "message": f"run marked {state['status']}"}

    next_round = int(state["current_round"]) + 1
    state["current_round"] = next_round
    task = _create_round_task(state, run_path, next_round, previous_result=result)
    state["rounds"].append(task["round_record"])
    state["updated_at"] = _now()
    _write_json(run_path / STATE_FILE, state)
    return {"state": state, "task": task["payload"]}


def load_status(run_dir: str | Path) -> dict[str, Any]:
    """Load the current host-agent autoresearch run state."""
    return _read_json(Path(run_dir) / STATE_FILE)


def _create_round_task(
    state: dict[str, Any],
    run_dir: Path,
    round_index: int,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = f"{state['case_study']}-round-{round_index:03d}"
    task_rel = Path(TASK_DIR) / f"round_{round_index:03d}_task.json"
    result_rel = Path(OUTPUT_DIR) / f"round_{round_index:03d}_result.json"
    neurooracle_case_dir = run_dir / "neurooracle_case"
    deterministic_command = (
        "python -m neurooracle.src.hypothesis_cli "
        f"--graph {state['graph_path']} case-study {state['case_study']} "
        f"--output-dir {neurooracle_case_dir} "
        f"--stages {state['deterministic_stages']}"
    )
    if state.get("kge_path"):
        deterministic_command += f" --kge {state['kge_path']}"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "round": round_index,
        "case_study": state["case_study"],
        "graph_path": state["graph_path"],
        "run_dir": str(run_dir),
        "expected_result_path": str(run_dir / result_rel),
        "role_contract": [
            "Use the host agent's built-in reasoning model as NeuroClaw's generator, critic, and experiment supervisor.",
            "Do not call NeuroClaw's configured LLM API for this task unless the user explicitly requests API mode.",
            "Ground claims in graph evidence, generated artifacts, cited literature, or inspected command outputs.",
            "Run local commands when useful, then record exact commands and checked outputs.",
        ],
        "deterministic_neurooracle_command": deterministic_command,
        "instructions": [
            "Inspect the deterministic NeuroOracle artifacts if they exist; otherwise run the command above or explain why it cannot be run.",
            "Generate or refine hypotheses for this case study.",
            "Critique each hypothesis from statistical, clinical, and methodological perspectives.",
            "Plan executable experiments or NeuroBench checks for the next round.",
            "Write only JSON to the expected_result_path using expected_result_template.",
            "Run host-agent-next after writing the result to advance the loop.",
        ],
        "previous_result": previous_result or {},
        "expected_result_template": task_result_template(task_id, round_index),
    }
    _write_json(run_dir / task_rel, payload)
    round_record = {
        "round": round_index,
        "task_id": task_id,
        "task_path": str(task_rel),
        "expected_result_path": str(result_rel),
        "status": "waiting_for_host_agent",
        "created_at": _now(),
    }
    return {"payload": payload, "round_record": round_record}


def _current_round_record(state: dict[str, Any]) -> dict[str, Any]:
    current_round = int(state["current_round"])
    for record in state.get("rounds", []):
        if int(record.get("round", 0)) == current_round:
            return record
    raise ValueError(f"run_state has no record for round {current_round}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
