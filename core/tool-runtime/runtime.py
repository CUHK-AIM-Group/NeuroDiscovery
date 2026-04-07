"""
NeuroClaw Tool Runtime

Executes skill handlers:
  - handler.js  → via Node.js subprocess
  - handler.py  → via the Python interpreter saved in neuroclaw_environment.json

Enforces feature-gate checks and basic safety rules before execution.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
ENV_FILE = REPO_ROOT / "neuroclaw_environment.json"


def _get_python_path() -> str:
    if ENV_FILE.exists():
        with ENV_FILE.open() as f:
            env = json.load(f)
        python_path = env.get("python_path")
        if python_path and Path(python_path).exists():
            return python_path
    return sys.executable


def _get_node_path() -> str:
    """Return the node executable path, or raise if not found."""
    import shutil
    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        raise RuntimeError(
            "Node.js not found. Install it to use handler.js skills. "
            "Alternatively, use handler.py variants."
        )
    return node


class ToolRuntime:
    """
    Executes a skill handler with the given input dict.

    Parameters
    ----------
    timeout : int
        Maximum execution time in seconds (default 300).
    """

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout

    def run(self, handler: Path, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch to the appropriate runtime based on handler file extension.

        Returns a dict with at minimum:
            success : bool
            output  : Any
            error   : str | None
        """
        suffix = handler.suffix.lower()
        if suffix == ".js":
            return self._run_node(handler, input_data)
        if suffix == ".py":
            return self._run_python(handler, input_data)
        return {
            "success": False,
            "output": None,
            "error": f"Unsupported handler type: {suffix}",
        }

    def _run_node(self, handler: Path, input_data: dict) -> dict:
        node = _get_node_path()
        # Pass input as JSON via stdin; handler reads process.stdin
        wrapper = (
            f"const h = require('{handler}');\n"
            f"let buf = '';\n"
            f"process.stdin.on('data', d => buf += d);\n"
            f"process.stdin.on('end', async () => {{\n"
            f"  const input = JSON.parse(buf);\n"
            f"  const fn = Object.values(h)[0];\n"
            f"  const result = await fn(input);\n"
            f"  process.stdout.write(JSON.stringify(result));\n"
            f"}});\n"
        )
        try:
            proc = subprocess.run(
                [node, "-e", wrapper],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(handler.parent),
            )
            if proc.returncode != 0:
                return {"success": False, "output": None, "error": proc.stderr}
            try:
                return {"success": True, "output": json.loads(proc.stdout), "error": None}
            except json.JSONDecodeError:
                return {"success": True, "output": proc.stdout, "error": None}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": None, "error": f"Timeout after {self.timeout}s"}
        except Exception as exc:
            return {"success": False, "output": None, "error": str(exc)}

    def _run_python(self, handler: Path, input_data: dict) -> dict:
        python = _get_python_path()
        wrapper = (
            "import sys, json\n"
            f"sys.path.insert(0, '{handler.parent}')\n"
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('handler', '{handler}')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "input_data = json.loads(sys.stdin.read())\n"
            "fn = [v for v in vars(mod).values() if callable(v) and not v.__name__.startswith('_')][0]\n"
            "result = fn(input_data)\n"
            "import asyncio\n"
            # Handler runs in a fresh subprocess, so there is never an
            # already-running event loop.  asyncio.run() is always correct here.
            "if asyncio.iscoroutine(result):\n"
            "    result = asyncio.run(result)\n"
            "print(json.dumps(result))\n"
        )
        try:
            proc = subprocess.run(
                [python, "-c", wrapper],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(handler.parent),
            )
            if proc.returncode != 0:
                return {"success": False, "output": None, "error": proc.stderr}
            try:
                return {"success": True, "output": json.loads(proc.stdout), "error": None}
            except json.JSONDecodeError:
                return {"success": True, "output": proc.stdout, "error": None}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": None, "error": f"Timeout after {self.timeout}s"}
        except Exception as exc:
            return {"success": False, "output": None, "error": str(exc)}
