import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.permissions import tool_permission
from core.research_progress import PROGRESS_TOOL_NAME, inspect_material
from core.tool_outcomes import normalize_shell_outcome


def record(run, path, kind):
    result = inspect_material(run.workspace, path, kind)
    stalled = run.observe(PROGRESS_TOOL_NAME, {"path": path, "kind": kind}, result)
    return result, stalled


@pytest.mark.parametrize("output", [
    "Error: biopython library not installed\nInstall with: pip install biopython\n",
    "Traceback (most recent call last):\n  module\nModuleNotFoundError: No module named 'Bio'",
])
def test_missing_dependency_even_with_zero_exit(output):
    result = normalize_shell_outcome('python research.py pubmed query 2>&1 | more',
                                     {"success": True, "returncode": 0, "stdout": output})
    assert not result["success"]
    assert result["error_type"] == "missing_dependency"
    assert "task-local" in result["recovery_hint"]
    assert "approval" in result["recovery_hint"]


@pytest.mark.parametrize("output", ['{"success": false}', '{"error": "quota"}',
                                  'Error searching PubMed: transport failed',
                                  'Traceback (most recent call last):\nRuntimeError: failed'])
def test_business_failure(output):
    result = normalize_shell_outcome("python research.py", {"success": True, "stdout": output})
    assert not result["success"]
    assert result["error_type"] == "tool_business_error"


@pytest.mark.parametrize("output", ['[]', '{"papers": []}', '{"error": null}',
                                  '0 errors', '{"results":[{"error":"quoted source"}]}'])
def test_legitimate_empty_search_or_quoted_data(output):
    assert normalize_shell_outcome("python research.py", {"success": True, "stdout": output})["success"]


@pytest.mark.parametrize("command", ['type error.log | more', 'Get-Content errors.txt', 'cat errors.txt'])
def test_reading_error_log_is_not_failed_execution(command):
    result = normalize_shell_outcome(command, {"success": True, "stdout": "Error: biopython library not installed"})
    assert result["success"]


def test_actual_pipeline_masks_upstream_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "AGENT_SHELL_STATUS_FILE", tmp_path / "status.json")
    script = tmp_path / "failure.py"
    script.write_text("import sys\nprint('Error: biopython library not installed')\nsys.exit(1)\n")
    result = main._run_shell_command('python failure.py | ' + ('more' if os.name == 'nt' else 'cat'), tmp_path)
    assert result["returncode"] == 0
    assert not result["success"]
    assert result["error_type"] == "missing_dependency"


def test_failed_tool_ui_and_evidence_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_run_shell_command", lambda **kwargs: {
        "success": True, "returncode": 0, "stdout": "Error: biopython library not installed"})
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "1")
    agent, calls = session(tmp_path, [reply(calls=[tool("run_shell_command", command="python search.py | more")])])
    events = []
    monkeypatch.setattr(agent, "_emit_execution_event", events.append)
    agent._chat()
    assert not agent.autoresearch_state["evidence"][0]["success"]
    assert not agent._tool_events[0]["success"]
    assert next(event for event in events if event["type"] == "tool_end")["status"] == "failed"


def test_material_progress_and_duplicate_content(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    paper = {"pmid": "123", "title": "Synthetic paper", "abstract": "Actual retrieved abstract"}
    path = tmp_path / "papers.json"
    path.write_text(json.dumps([paper]))
    result, stalled = record(run, path.name, "literature")
    assert result["new_material_count"] == 1 and not stalled
    path.write_text(json.dumps([paper, paper], indent=4))
    result, _ = record(run, path.name, "literature")
    assert result["new_material_count"] == 0
    (tmp_path / "copy.json").write_bytes(path.read_bytes())
    assert record(run, "copy.json", "literature")[0]["new_material_count"] == 0
    paper["summary"] = "Source-bound reading summary"
    path.write_text(json.dumps([paper]))
    assert record(run, path.name, "literature")[0]["new_material_count"] == 1
    assert run.state["no_progress_count"] == 0
    assert not result["scientifically_validated"]


def test_candidates_and_delivery_changes(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    path = tmp_path / "ideas.json"
    candidate = {"hypothesis": "Testable prediction", "rationale": "Evidence rationale", "source_ids": ["PMID:123"]}
    path.write_text(json.dumps({"hypotheses": [candidate]}))
    assert record(run, path.name, "hypotheses")[0]["new_material_count"] == 1
    candidate["timestamp"] = "changing metadata"
    path.write_text(json.dumps({"hypotheses": [candidate]}))
    assert record(run, path.name, "hypotheses")[0]["new_material_count"] == 0
    (tmp_path / "IDEA.md").write_text("# Candidate report\nEvidence, limitations and test plan.")
    assert record(run, "IDEA.md", "deliverable")[0]["new_material_count"] == 1
    (tmp_path / "IDEA.md").write_text("# Candidate report\nEvidence, revised limitations and test plan.")
    assert record(run, "IDEA.md", "deliverable")[0]["new_material_count"] == 1


@pytest.mark.parametrize("name,kind,content", [
    ("papers.json", "literature", "[]"),
    ("papers.json", "literature", '{"claims": [{"title":"graph claim"}]}'),
    ("ideas.json", "hypotheses", '[{"hypothesis":"unsupported"}]'),
    ("report.json", "deliverable", '{"error":"failed"}'),
    ("report.json", "deliverable", '{}'),
    ("AGENTS.md", "deliverable", "instructions"),
    ("SOUL.md", "deliverable", "instructions"),
    ("report.md", "deliverable", "  "),
    ("code.py", "deliverable", "print('not research')"),
])
def test_invalid_material_never_counts(tmp_path, name, kind, content):
    (tmp_path / name).write_text(content)
    run = AutoResearchRun(tmp_path, "idea")
    result, _ = record(run, name, kind)
    assert not result["success"]
    assert not run.state["research_progress"]
    assert run.state["no_progress_count"] == 1


def test_boundaries_and_permissions(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    assert not inspect_material(tmp_path, str(run.path), "deliverable")["success"]
    assert not inspect_material(tmp_path, "../outside.md", "deliverable")["success"]
    (tmp_path / "large.md").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    assert not inspect_material(tmp_path, "large.md", "deliverable")["success"]
    for mode in ("ask", "risk", "never", "read_only"):
        assert tool_permission(mode, PROGRESS_TOOL_NAME, {"path": "papers.json"}, tmp_path) == "allow"
        assert tool_permission(mode, PROGRESS_TOOL_NAME, {"path": "../outside.md"}, tmp_path) == "deny"
    assert tool_permission("read_only", "run_shell_command", {"command": "pip install biopython"}, tmp_path) == "deny"
    assert tool_permission("ask", "run_shell_command", {"command": "pip install biopython"}, tmp_path) == "ask"


def test_resume_keeps_material_fingerprints(tmp_path):
    (tmp_path / "report.md").write_text("Synthetic material")
    previous = AutoResearchRun(tmp_path, "idea")
    record(previous, "report.md", "deliverable")
    previous.halt("interrupted", "Test pause")
    run = AutoResearchRun(tmp_path, "idea")
    run.inherit_checkpoint(previous.path.parent.name, "continue")
    assert record(run, "report.md", "deliverable")[0]["new_material_count"] == 0


def test_real_tool_dispatch_resets_material_counter(tmp_path):
    (tmp_path / "report.md").write_text("Synthetic actual deliverable")
    agent, calls = session(tmp_path, [reply(calls=[tool(PROGRESS_TOOL_NAME, path="report.md", kind="deliverable")])])
    agent._chat()
    assert agent.autoresearch_state["research_progress"]["deliverable"]["files"]["report.md"]
    assert any(entry["function"]["name"] == PROGRESS_TOOL_NAME for entry in calls[0]["tools"])


@pytest.mark.parametrize("network_failure", [True, False])
def test_search_transport_failure_is_not_empty_success(monkeypatch, network_failure):
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace())
    path = Path(__file__).resolve().parents[2] / "skills" / "academic-research-hub" / "scripts" / "research.py"
    spec = importlib.util.spec_from_file_location("research_test", path.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fail(**kwargs):
        raise OSError("synthetic network failure")
    if network_failure:
        monkeypatch.setattr(module, "Entrez", SimpleNamespace(esearch=fail))
        with pytest.raises(SystemExit) as failure:
            module.search_pubmed("synthetic")
        assert failure.value.code == 1
    else:
        handle = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(module, "Entrez", SimpleNamespace(esearch=lambda **kwargs: handle,
                                                              read=lambda handle: {"IdList": []}))
        assert module.search_pubmed("synthetic") == []


def test_crash_recovery_does_not_credit_old_material(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "1")
    (tmp_path / "report.md").write_text("Synthetic actual deliverable")
    previous = AutoResearchRun(tmp_path, "idea")
    record(previous, "report.md", "deliverable")
    agent, calls = session(tmp_path, [reply(calls=[tool(PROGRESS_TOOL_NAME, path="report.md", kind="deliverable")])])
    agent._recovery_state = json.loads(previous.path.read_text())
    agent._chat()
    assert agent.autoresearch_state["no_progress_count"] == 1
    assert len(agent.autoresearch_state["research_progress"]["deliverable"]["fingerprints"]) == 1
