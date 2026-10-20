"""NeuroRuntime MCP server: the harness surface for external agent CLIs.

This is the mount point for Group 2 of the benchmark. Codex CLI and Claude Code
both speak Model Context Protocol, so one stdio server exposes the NeuroRuntime
harness tool surface to either CLI without a per-CLI integration:

* Codex mounts it as ``-c mcp_servers.neuro_runtime=...``
  (Camellia does the same with ``mcp_servers.camellia_goals`` in
  ``src/engines/codex-session.js``).
* Claude Code mounts it with ``--mcp-config``
  (``src/engines/claude-session.js``).

The wire shape follows Camellia's ``src/engines/goal-mcp-stdio.js``, which is
itself a minimal but correct MCP stdio server: line-delimited JSON-RPC 2.0,
``initialize`` protocol-version negotiation, ``ping``, ``tools/list`` and
``tools/call``, with a bounded request size.

Only the harness seam tools are exposed. The external CLI already owns a shell
and a file reader; what it cannot have on its own is NeuroRuntime's skill
library, so that is what this server provides.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


# The server is launched as a subprocess by a CLI whose working directory is
# arbitrary, so `import core.*` must work without relying on cwd. Prepending the
# repository root here is what makes the mount descriptor independent of where
# the CLI happens to run.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "neuroruntime-harness"
SERVER_VERSION = "1.1.0"

# Request and response framing limits, matching Camellia's server.
MAX_REQUEST_BYTES = 65536
MAX_RESPONSE_BYTES = 131072

# The tools exposed over MCP. Overridable so a driver can widen the surface.
DEFAULT_EXPOSED_TOOLS = ("search_skills", "read_skill", "compile_task", "read_skill_section")


def exposed_tool_names() -> tuple[str, ...]:
    raw = (os.environ.get("NEURORUNTIME_MCP_TOOLS") or "").strip()
    if not raw:
        return DEFAULT_EXPOSED_TOOLS
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _instructions() -> str:
    names = set(exposed_tool_names())
    lines = ['NeuroRuntime harness tools.']
    if 'compile_task' in names:
        lines.append('Use compile_task to organize public requirements and input metadata.')
    if 'search_skills' in names:
        lines.append('Use search_skills to find relevant guidance.')
    if 'read_skill_section' in names:
        lines.append('Use read_skill_section for relevant paragraphs with a session cache.')
    if 'read_skill' in names:
        lines.append('Use read_skill for full paginated instructions.')
    return ' '.join(lines + ['Skills are reusable guidance.'])


class HarnessHost:
    """The minimal session the harness tools need, without an agent loop.

    ``search_skills`` and ``read_skill`` read only ``workspace`` and append to
    ``_tool_events``, so no LLM client, checkpoint manager or cancel event is
    required. Keeping the host this small is the point: the seam must not drag
    the whole loop behind it.
    """

    def __init__(self, workspace: Path, *, skills_dir: Path | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self._tool_events: list[dict[str, Any]] = []
        if skills_dir is not None:
            from core.harness_skills import default_capability
            self._skill_capability = default_capability(Path(skills_dir).resolve(), include_cwd=False)

    def _tool_guard(self, call) -> dict | None:
        # An exposed name is not authorization to execute shell/subagent tools.
        # The standalone host only provides cooperative read-only capabilities.
        from core.permissions import tool_permission

        if tool_permission("read_only", call.name, call.arguments, self.workspace) != "allow":
            return {"success": False, "executed": False, "error_type": "permission_denied",
                    "error": "The standalone harness host permits read-only tools only."}
        return None


def _workspace() -> Path:
    raw = (os.environ.get("NEURORUNTIME_MCP_WORKSPACE") or "").strip()
    # The harness skill library lives at the repository root, which is also the
    # only directory guaranteed to be correct no matter where the CLI launches
    # the server from.
    return Path(raw).resolve() if raw else _REPO_ROOT


def tool_definitions() -> list[dict[str, Any]]:
    """MCP tool definitions for the exposed harness tools."""
    from core import tool_registry
    from core.tools_impl import register_core_tools

    register_core_tools()
    if set(exposed_tool_names()) & {'describe_task_contract', 'validate_proposal'}:
        from core.harness_validation import register_validation_tools
        register_validation_tools()
    definitions: list[dict[str, Any]] = []
    for name in exposed_tool_names():
        spec = tool_registry.get(name)
        if spec is None:
            continue
        definitions.append({
            "name": spec.name,
            "description": spec.description,
            "inputSchema": dict(spec.parameters),
        })
    return definitions


def _make_host(workspace: Path) -> HarnessHost:
    frozen_skills = (os.environ.get("NEURORUNTIME_MCP_SKILLS_DIR") or "").strip()
    host = HarnessHost(workspace, skills_dir=Path(frozen_skills) if frozen_skills else None)
    task_file = (os.environ.get('NEURORUNTIME_MCP_TASK_FILE') or '').strip()
    if task_file:
        host._evaluation_task_file = Path(task_file)
        host._evaluation_task_sha256 = os.environ.get('NEURORUNTIME_MCP_TASK_SHA256', '')
    return host


def call_tool(name: str, arguments: dict[str, Any], workspace: Path | None = None,
              *, host: HarnessHost | None = None) -> dict[str, Any]:
    """Dispatch one MCP tool call and wrap the result as MCP content."""
    from core import tool_registry
    from core.tools_impl import register_core_tools

    register_core_tools()
    if name in {'describe_task_contract', 'validate_proposal'}:
        from core.harness_validation import register_validation_tools
        register_validation_tools()
    if name not in exposed_tool_names():
        return _content({"success": False, "error": f"tool '{name}' is not exposed by this server"}, is_error=True)
    if not isinstance(arguments, dict):
        return _content({"success": False, "error": "arguments must be an object"}, is_error=True)
    host = host if host is not None else _make_host(workspace or _workspace())
    result = tool_registry.dispatch(host, name, arguments, host.workspace)
    return _content(result, is_error=result.get("success") is False)


def _content(payload: dict[str, Any], *, is_error: bool) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        payload = {
            "success": False,
            "error": "response exceeded the server size limit; narrow the request",
        }
        text = json.dumps(payload, ensure_ascii=False)
        is_error = True
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def dispatch(request: dict[str, Any], *, host: HarnessHost | None = None) -> dict[str, Any]:
    """Handle one JSON-RPC request, mirroring the standard MCP methods."""
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        negotiated = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": _instructions(),
        }
    if method in {"ping", "notifications/initialized", "notifications/cancelled"}:
        return {}
    if method == "tools/list":
        return {"tools": tool_definitions()}
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        return call_tool(name, arguments, host=host)
    raise McpError(-32601, f"Unsupported MCP method: {method}")


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def serve(input_stream: Iterable[str], output, *, workspace: Path | None = None) -> None:
    """Line-delimited JSON-RPC loop. Never raises on a bad line.

    The loop deliberately uses ``readline()`` rather than ``for line in
    input_stream``. Iterating a pipe uses a read-ahead buffer, so a short request
    written while the client keeps the pipe open never yields a line; the server
    appears to hang until the client sends 8 KiB or closes stdin. An MCP client
    does neither, which is why an interactive session must read one line at a
    time.
    """
    host = _make_host(workspace or _workspace())
    readline = getattr(input_stream, "readline", None)
    lines = iter(input_stream) if readline is None else iter(readline, "")
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        _debug("<- " + line[:400])
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            _write(output, {"jsonrpc": "2.0", "id": None,
                            "error": {"code": -32600, "message": "Request too large"}})
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _write(output, {"jsonrpc": "2.0", "id": None,
                            "error": {"code": -32700, "message": "Parse error"}})
            continue
        request_id = request.get("id")
        if request_id is None:
            # A notification carries no id and expects no response.
            continue
        try:
            result = dispatch(request, host=host)
            _write(output, {"jsonrpc": "2.0", "id": request_id, "result": result})
        except McpError as error:
            _write(output, {"jsonrpc": "2.0", "id": request_id,
                            "error": {"code": error.code, "message": error.message}})
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            _write(output, {"jsonrpc": "2.0", "id": request_id,
                            "error": {"code": -32603, "message": f"{type(error).__name__}: {error}"}})


def _write(output, payload: dict[str, Any]) -> None:
    _debug("-> " + json.dumps(payload, ensure_ascii=False)[:400])
    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    output.flush()


def _debug(message: str) -> None:
    """Append one line to the debug log when NEURORUNTIME_MCP_DEBUG is set.

    A stdio MCP server's stderr is swallowed by the client, so a mount failure is
    otherwise invisible. This writes to the path in the variable instead.
    """
    path = (os.environ.get("NEURORUNTIME_MCP_DEBUG") or "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def mount_config(*, python_executable: str | None = None, workspace: str | None = None) -> dict[str, Any]:
    """The mount descriptor a CLI needs, in the shared shape both CLIs accept."""
    env = {}
    if workspace:
        env["NEURORUNTIME_MCP_WORKSPACE"] = str(workspace)
    return {
        "command": python_executable or sys.executable,
        # The script path, not ``-m core.mcp_stdio``: the CLI chooses the working
        # directory, and ``-m`` resolves the module through cwd, so it fails when
        # the CLI runs elsewhere. Running the file directly is cwd-independent.
        "args": [str(Path(__file__).resolve())],
        "env": env,
    }


def main() -> int:
    # MCP is UTF-8 on the wire, independent of the Windows console code page
    # or an inherited PYTHONIOENCODING (the client may not pass that variable).
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict", newline="\n")
    serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
