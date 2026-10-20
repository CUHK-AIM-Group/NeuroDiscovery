"""Step 5 verification: one MCP surface mounts into either external CLI.

The server is exercised the way a CLI would drive it: line-delimited JSON-RPC
over stdin/stdout, protocol negotiation, tool listing and a real tool call
against the repository's own skill library.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from core import mcp_stdio


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rpc(*requests: dict) -> list[dict]:
    out = io.StringIO()
    lines = [json.dumps(request) + "\n" for request in requests]
    mcp_stdio.serve(lines, out, workspace=REPO_ROOT)
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def test_initialize_negotiates_a_supported_protocol_version():
    [response] = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-03-26"}})
    result = response["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["serverInfo"]["name"] == "neuroruntime-harness"
    assert "tools" in result["capabilities"]


def test_initialize_falls_back_for_an_unknown_protocol_version():
    [response] = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "1999-01-01"}})
    assert response["result"]["protocolVersion"] == mcp_stdio.DEFAULT_PROTOCOL_VERSION


def test_default_surface_exposes_only_the_skill_seam():
    [response] = _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [tool["name"] for tool in response["result"]["tools"]]
    assert names == ["search_skills", "read_skill", "compile_task", "read_skill_section"]


def test_tool_definitions_carry_a_json_schema_each():
    [response] = _rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    for tool in response["result"]["tools"]:
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"


def test_tools_call_loads_a_real_skill():
    [response] = _rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "read_skill", "arguments": {"name": "adni-skill"}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["success"] is True
    assert payload["name"] == "adni-skill"
    assert response["result"]["isError"] is False


def test_tools_call_search_returns_matches():
    [response] = _rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                       "params": {"name": "search_skills", "arguments": {"query": "eeg"}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["success"] is True
    assert payload["matches"]


def test_tools_call_refuses_a_tool_outside_the_exposed_surface():
    [response] = _rpc({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                       "params": {"name": "run_shell_command", "arguments": {"command": "dir"}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["success"] is False
    assert response["result"]["isError"] is True


def test_unknown_method_is_a_jsonrpc_error_not_a_crash():
    [response] = _rpc({"jsonrpc": "2.0", "id": 7, "method": "tools/nonexistent"})
    assert response["error"]["code"] == -32601


def test_notifications_get_no_response():
    responses = _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert responses == []


def test_malformed_line_does_not_abort_the_session():
    out = io.StringIO()
    mcp_stdio.serve(["{not json\n", json.dumps({"jsonrpc": "2.0", "id": 8, "method": "ping"}) + "\n"],
                    out, workspace=REPO_ROOT)
    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["id"] == 8


def test_oversized_request_is_refused():
    out = io.StringIO()
    mcp_stdio.serve(["x" * (mcp_stdio.MAX_REQUEST_BYTES + 10) + "\n"], out, workspace=REPO_ROOT)
    [response] = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert response["error"]["code"] == -32600


def test_mount_config_uses_a_module_entry_point():
    config = mcp_stdio.mount_config(workspace=str(REPO_ROOT))
    assert config["args"] == [str(REPO_ROOT / "core" / "mcp_stdio.py")]
    assert config["env"]["NEURORUNTIME_MCP_WORKSPACE"] == str(REPO_ROOT)
    assert config["command"]


def test_server_runs_from_an_unrelated_working_directory(tmp_path):
    """The CLI picks the cwd, so the server must not depend on it.

    Regression: mounting as ``-m core.mcp_stdio`` failed with
    ``No module named 'core'`` whenever the CLI ran outside the repository.
    """
    import subprocess
    import sys

    requests = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "search_skills", "arguments": {"query": "fmri"}}}) + "\n"
    )
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "core" / "mcp_stdio.py")],
        input=requests, capture_output=True, text=True, encoding="utf-8", cwd=str(tmp_path), timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert responses[0]["result"]["serverInfo"]["name"] == "neuroruntime-harness"
    payload = json.loads(responses[1]["result"]["content"][0]["text"])
    assert payload["skill_count"] > 50


def test_default_workspace_is_the_repository_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURORUNTIME_MCP_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mcp_stdio._workspace() == REPO_ROOT


@pytest.mark.parametrize("inherited_encoding", ["gbk", "cp1252"])
def test_stdio_uses_utf8_for_unicode_requests_and_skill_contents(tmp_path, inherited_encoding):
    """A real pipe must preserve Unicode even under a legacy Windows encoding."""
    import os
    import subprocess
    import sys

    skills_dir = tmp_path / "skills"
    skill = skills_dir / "unicode-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    body = "---\nname: unicode-skill\ndescription: 中文方法\n---\n脑科学 résumé — 🧠\n"
    skill.write_text(body, encoding="utf-8")
    request = {"jsonrpc": "2.0", "id": "读取-🧠", "method": "tools/call",
               "params": {"name": "read_skill", "arguments": {"name": "unicode-skill"}}}
    env = os.environ.copy()
    env.update(PYTHONIOENCODING=inherited_encoding, PYTHONUTF8="0",
               NEURORUNTIME_MCP_SKILLS_DIR=str(skills_dir),
               NEURORUNTIME_MCP_TOOLS="search_skills,read_skill")
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "core" / "mcp_stdio.py")],
        input=(json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"),
        capture_output=True, cwd=str(tmp_path), env=env, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    [response] = [json.loads(line) for line in completed.stdout.decode("utf-8").splitlines()]
    assert response["id"] == request["id"]
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["success"] is True
    assert payload["content"] == body


def test_interactive_pipe_responds_without_closing_stdin(tmp_path):
    """A client that keeps stdin open must still get an answer per request.

    Regression: iterating ``sys.stdin`` used a read-ahead buffer, so a short
    request written while the pipe stayed open produced no line and the server
    appeared to hang until the client sent 8 KiB or closed stdin. Real MCP
    clients do neither, so the earlier non-interactive test passed while the
    live Codex mount timed out after 300 seconds.
    """
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "core" / "mcp_stdio.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, cwd=str(tmp_path),
    )
    try:
        def rpc(payload: dict) -> dict:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            assert line.strip(), "server produced no response while stdin stayed open"
            return json.loads(line)

        ready = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2025-06-18"}})
        assert ready["result"]["serverInfo"]["name"] == "neuroruntime-harness"
        called = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "search_skills",
                                 "arguments": {"query": "fmri connectivity", "limit": 3}}})
        payload = json.loads(called["result"]["content"][0]["text"])
        assert payload["success"] is True
        assert payload["matches"]
    finally:
        process.kill()


def test_exposed_surface_can_be_widened_by_environment(monkeypatch):
    monkeypatch.setenv("NEURORUNTIME_MCP_TOOLS", "search_skills,read_skill,read_workspace_file")
    names = [tool["name"] for tool in mcp_stdio.tool_definitions()]
    assert "read_workspace_file" in names
