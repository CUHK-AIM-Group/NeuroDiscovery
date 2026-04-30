"""
NeuroClaw Core Agent — LLM conversation loop and tool-call dispatcher.

------------
  AgentSession
    │
    ├─ SkillLoader        loads skills/*/SKILL.md, registers tools
    ├─ ToolRuntime        executes handler.js / Python handlers
    ├─ SessionManager     context window, persistence, compression
    └─ LLMBackend         OpenAI / Anthropic / local model adapter

Messaging connectors (WhatsApp, Slack, Telegram) are intentionally excluded;
see core/config/features.json to audit disabled features.

Usage
-----
    # Interactive REPL
    python core/agent/main.py

    # Interactive REPL in benchmark mode
    python core/agent/main.py --benchmark

    # Browser-based Web UI (served on http://localhost:7080 by default)
    python core/agent/main.py --web [--port 7080] [--host 127.0.0.1]

    # Browser-based Web UI in benchmark mode
    python core/agent/main.py --web --benchmark
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
ENV_FILE = REPO_ROOT / "neuroclaw_environment.json"
FEATURES_FILE = REPO_ROOT / "core" / "config" / "features.json"
AGENT_SHELL_STATUS_FILE = Path("/tmp/neuroclaw_agent_shell_status.json")
BENCHMARK_ENV_FLAG = "NEUROCLAW_BENCHMARK"
BENCHMARK_SCORER_MODEL = "gpt-5.4"
BENCHMARK_SCORE_WEIGHTS = {
    "planning_completeness": 0.30,
    "tool_reasonableness": 0.40,
    "code_command_correctness": 0.30,
}
BENCHMARK_WITH_SKILLS_SUFFIX = "_withskills"
BENCHMARK_NO_SKILLS_SUFFIX = "_noskills"
_BENCHMARK_SKILLS_CACHE: list[dict[str, Any]] | None = None
_BENCHMARK_SCORER_CLIENT_CACHE: dict[str, Any] = {}
_BENCHMARK_SCORER_CLIENT_CACHE_LOCK = threading.Lock()

# Ensure the repo root is on sys.path so that `from core.X import Y` works
# regardless of the working directory when main.py is invoked directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Environment bootstrap ──────────────────────────────────────────────────────

def load_environment() -> dict:
    """
    Load neuroclaw_environment.json and export toolchain environment variables.

    Called automatically at session start (as specified in SOUL.md).
    If the file does not exist, the agent will prompt the user to run the
    installer before proceeding.
    """
    if not ENV_FILE.exists():
        return {}

    with ENV_FILE.open() as f:
        env = json.load(f)

    _normalize_llm_backend(env)
    _export_toolchain_env(env)
    return env


def _normalize_llm_backend(env: dict) -> None:
    """
    Normalize llm_backend config and support OpenAI-compatible provider profiles.

    Supported compatibility shape (top-level profile):
      {
        "api-proxy-gpt": {
          "baseUrl": "https://example.com/v1",
          "api": "openai-completions",
          "models": [{"id": "gpt-5.2"}, ...]
        }
      }
    """

    llm_cfg = env.get("llm_backend")
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}
        env["llm_backend"] = llm_cfg

    profile_name = llm_cfg.get("profile")
    profile: dict | None = None

    if isinstance(profile_name, str):
        candidate = env.get(profile_name)
        if isinstance(candidate, dict):
            profile = candidate

    if profile is None:
        for name, value in env.items():
            if not isinstance(value, dict):
                continue
            if value.get("api") == "openai-completions" and (
                value.get("baseUrl") or value.get("base_url")
            ):
                profile_name = name
                profile = value

    if profile is not None:
        llm_cfg.setdefault("provider", "openai")

        if not llm_cfg.get("base_url"):
            llm_cfg["base_url"] = profile.get("base_url") or profile.get("baseUrl")

        if not llm_cfg.get("model"):
            explicit_model = profile.get("model") or profile.get("defaultModel")
            if isinstance(explicit_model, str) and explicit_model.strip():
                llm_cfg["model"] = explicit_model.strip()
            else:
                models = profile.get("models", [])
                if isinstance(models, list) and models:
                    first = models[0]
                    if isinstance(first, dict):
                        llm_cfg["model"] = first.get("id") or first.get("name")
                    elif isinstance(first, str):
                        llm_cfg["model"] = first

        if not llm_cfg.get("api_key_env"):
            llm_cfg["api_key_env"] = (
                profile.get("api_key_env") or profile.get("apiKeyEnv")
            )

        if not llm_cfg.get("api_key"):
            llm_cfg["api_key"] = profile.get("api_key") or profile.get("apiKey")

        if profile_name:
            llm_cfg["profile_name"] = profile_name

    llm_cfg["provider"] = llm_cfg.get("provider", "openai")
    llm_cfg["base_url"] = llm_cfg.get("base_url") or llm_cfg.get("baseUrl")
    llm_cfg["model"] = llm_cfg.get("model") or "gpt-4o"
    llm_cfg["available_models"] = _normalize_available_models(llm_cfg, profile)


def _normalize_available_models(llm_cfg: dict, profile: dict | None) -> list[dict[str, Any]]:
    """Normalize configured model options into a provider/model catalog."""
    configured = llm_cfg.get("available_models")
    if isinstance(configured, list) and configured:
        normalized = _coerce_model_catalog(configured)
        if normalized:
            return normalized

    if profile is not None:
        profile_models = profile.get("models")
        if isinstance(profile_models, list) and profile_models:
            normalized = _coerce_model_catalog(profile_models, default_provider="openai")
            if normalized:
                return normalized

    provider = str(llm_cfg.get("provider") or "openai")
    model = str(llm_cfg.get("model") or "gpt-4o")
    return [{"provider": provider, "model": model, "label": f"{provider} / {model}"}]


def _coerce_model_catalog(
    items: list[Any], default_provider: str | None = None
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if isinstance(item, str):
            provider = default_provider or "openai"
            model = item.strip()
            label = f"{provider} / {model}"
        elif isinstance(item, dict):
            provider = str(item.get("provider") or default_provider or "openai").strip()
            model = str(item.get("model") or item.get("id") or item.get("name") or "").strip()
            label = str(item.get("label") or f"{provider} / {model}").strip()
        else:
            continue

        if not provider or not model:
            continue
        key = (provider, model)
        if key in seen:
            continue
        seen.add(key)
        catalog.append({"provider": provider, "model": model, "label": label})
    return catalog


def save_environment(env: dict) -> None:
    """Persist environment config back to neuroclaw_environment.json."""
    with ENV_FILE.open("w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _resolve_openai_api_key(cfg: dict) -> tuple[str, str | None]:
    """Resolve OpenAI-compatible API key from config/env and return (key, source_env)."""
    direct = cfg.get("api_key") or cfg.get("apiKey")
    if isinstance(direct, str) and direct.strip():
        return direct.strip(), None

    # Fallback: if the top-level config doesn't have a key, try to reuse a key
    # from model_overrides (common in setups that only store keys per route).
    try:
        overrides = cfg.get("model_overrides") or {}
        if isinstance(overrides, dict) and overrides:
            base_url = str(cfg.get("base_url") or cfg.get("baseUrl") or "").strip()

            def _pick_key(match_base_url: bool) -> str:
                for ov in overrides.values():
                    if not isinstance(ov, dict):
                        continue
                    if str(ov.get("provider") or "openai").strip().lower() != "openai":
                        continue
                    if match_base_url:
                        ov_base = str(ov.get("base_url") or ov.get("baseUrl") or "").strip()
                        if base_url and ov_base and ov_base != base_url:
                            continue
                    k = ov.get("api_key") or ov.get("apiKey")
                    if isinstance(k, str) and k.strip():
                        return k.strip()
                return ""

            picked = _pick_key(match_base_url=True) or _pick_key(match_base_url=False)
            if picked:
                return picked, None
    except Exception:
        pass

    env_candidates: list[str] = []
    explicit_env = cfg.get("api_key_env")
    if isinstance(explicit_env, str) and explicit_env.strip():
        env_candidates.append(explicit_env.strip())

    profile_name = cfg.get("profile_name")
    if isinstance(profile_name, str) and profile_name.strip():
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", profile_name).strip("_").upper()
        if normalized:
            env_candidates.append(f"{normalized}_API_KEY")

    env_candidates.append("OPENAI_API_KEY")

    seen: set[str] = set()
    for env_name in env_candidates:
        if env_name in seen:
            continue
        seen.add(env_name)
        val = os.environ.get(env_name, "").strip()
        if val:
            return val, env_name

    return "", env_candidates[0] if env_candidates else None


def _apply_runtime_api_key(env: dict, api_key: str) -> None:
    """Apply a runtime-only API key override for the configured LLM backend."""
    if not api_key:
        return

    llm_cfg = env.setdefault("llm_backend", {})
    provider = str(llm_cfg.get("provider", "openai") or "openai").strip().lower()
    env_var_name = str(llm_cfg.get("api_key_env") or "").strip()
    if not env_var_name:
        env_var_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        llm_cfg["api_key_env"] = env_var_name
    os.environ[env_var_name] = api_key


def _export_toolchain_env(env: dict) -> None:
    """
    Export FSLDIR, FREESURFER_HOME, CUDA_VISIBLE_DEVICES, etc.
    into the current process environment so all child processes inherit them.
    """
    toolchain = env.get("toolchain", {})
    if toolchain.get("fsl_home"):
        os.environ.setdefault("FSLDIR", toolchain["fsl_home"])
        fsl_bin = str(Path(toolchain["fsl_home"]) / "bin")
        _prepend_path(fsl_bin)

    if toolchain.get("freesurfer_home"):
        os.environ.setdefault("FREESURFER_HOME", toolchain["freesurfer_home"])
        fs_bin = str(Path(toolchain["freesurfer_home"]) / "bin")
        _prepend_path(fs_bin)

    cuda = env.get("cuda", {})
    device = cuda.get("device", "")
    if device and device.startswith("cuda:"):
        gpu_index = device.split(":")[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_index)


def _prepend_path(directory: str) -> None:
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current


# ── Feature gate ───────────────────────────────────────────────────────────────

def is_feature_enabled(category: str, name: str) -> bool:
    """Return True if features.json marks category.name as enabled."""
    if not FEATURES_FILE.exists():
        return True  # permissive default if features file is missing
    with FEATURES_FILE.open() as f:
        features = json.load(f)
    return features.get(category, {}).get(name, {}).get("enabled", True)


# ── LLM backend factory ────────────────────────────────────────────────────────

def build_llm_client(env: dict) -> Any:
    """
    Return a thin LLM client object based on env['llm_backend'].
    Raises RuntimeError if the required library is not installed or
    the provider is not enabled in features.json.
    """
    llm_cfg = env.get("llm_backend", {})
    provider = llm_cfg.get("provider", "openai")

    if not is_feature_enabled("llm_backends", provider):
        raise RuntimeError(
            f"LLM provider '{provider}' is disabled in features.json."
        )

    if provider == "openai":
        return _build_openai_client(llm_cfg)
    if provider == "anthropic":
        return _build_anthropic_client(llm_cfg)
    if provider == "local":
        return _build_local_client(llm_cfg)

    raise RuntimeError(f"Unknown LLM provider: {provider}")


def _build_openai_client(cfg: dict):
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc
    api_key, key_source = _resolve_openai_api_key(cfg)
    if not api_key:
        profile = cfg.get("profile_name")
        profile_hint = f" (profile: {profile})" if profile else ""
        raise RuntimeError(
            "OpenAI-compatible API key is missing"
            f"{profile_hint}. Set env var {key_source or 'OPENAI_API_KEY'} "
            "or provide llm_backend.api_key in neuroclaw_environment.json."
        )

    base_url = cfg.get("base_url") or cfg.get("baseUrl") or None
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


def _run_openai_tool_probe(env: dict, model: str, workspace: Path) -> int:
    """Run a minimal one-shot tool-call probe against the configured OpenAI-compatible route."""
    client = build_llm_client(env)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_workspace_file",
                "description": "Read a UTF-8 text file inside the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to a text file inside the workspace.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return.",
                        },
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are testing tool-calling. Before answering, call the provided "
                "read_workspace_file tool exactly once on skills/nilearn-tool/SKILL.md. "
                "After the tool call result is available, summarize the first sentence."
            ),
        },
        {
            "role": "user",
            "content": "Read skills/nilearn-tool/SKILL.md with the tool first, then answer.",
        },
    ]

    print(f"[probe] model={model}")
    print(f"[probe] workspace={workspace}")
    resp = _retry_api_call(
        "OpenAI tool probe",
        lambda: client.chat.completions.create(
            **_get_openai_chat_create_kwargs(
                env,
                model,
                messages,
                tools=tools,
                tool_choice="auto",
            )
        ),
        retries=3,
    )

    choice = resp.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)
    tool_calls = list(getattr(message, "tool_calls", []) or [])
    content = getattr(message, "content", "") or ""

    print(f"[probe] finish_reason={finish_reason}")
    print(f"[probe] tool_call_count={len(tool_calls)}")
    if tool_calls:
        for idx, tc in enumerate(tool_calls, start=1):
            print(f"[probe] tool_call_{idx}.name={tc.function.name}")
            print(f"[probe] tool_call_{idx}.arguments={tc.function.arguments}")
        return 0


def _run_openai_tool_loop_probe(env: dict, model: str, workspace: Path) -> int:
    session = AgentSession(workspace=workspace, benchmark_mode=False)
    session.env = env
    session._llm = _build_openai_client(env.get("llm_backend") or {})  # type: ignore[attr-defined]

    # Keep this probe minimal: only the read_workspace_file tool.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_workspace_file",
                "description": "Read a UTF-8 text file inside the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to a text file inside the workspace.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return.",
                        },
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You MUST call read_workspace_file, then wait for the tool result, then answer OK.",
        },
        {
            "role": "user",
            "content": "Call read_workspace_file on skills/ukb-skill/SKILL.md.",
        },
    ]

    print(f"[probe-loop] model={model}")

    resp1 = session._llm.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    msg1 = resp1.choices[0].message
    tool_calls = getattr(msg1, "tool_calls", None) or []
    print(f"[probe-loop] turn1.finish_reason={resp1.choices[0].finish_reason}")
    print(f"[probe-loop] turn1.tool_call_count={len(tool_calls)}")
    if not tool_calls:
        print("[probe-loop] No tool_calls returned")
        return 2

    # Mirror the exact codepath used in the main tool loop to build assistant tool message.
    assistant_tool_msg = {
        "role": "assistant",
        "content": msg1.content or "",
        "tool_calls": [
            (
                (tc.model_dump() if hasattr(tc, "model_dump") else tc.to_dict() if hasattr(tc, "to_dict") else dict(getattr(tc, "__dict__", {}) or {}))
                if _model_uses_gemini_compat(model)
                else {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
            for tc in tool_calls
        ],
    }
    if _model_uses_gemini_compat(model) and os.environ.get("NEUROCLAW_DEBUG_GEMINI_TOOLCALLS") == "1":
        try:
            debug_path = (REPO_ROOT / "output" / "debug_gemini_toolcalls.jsonl")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "model": model,
                            "assistant_tool_msg": assistant_tool_msg,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
    messages.append(assistant_tool_msg)

    # Execute local tool(s) and append tool result messages.
    for tc in tool_calls:
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)
        if tool_name == "read_workspace_file":
            result = _read_workspace_file(
                str(tool_args.get("path") or ""),
                workspace,
                max_chars=int(tool_args.get("max_chars") or 12000),
            )
        else:
            result = {"success": False, "error": f"Unsupported tool in probe: {tool_name}", "output": None}

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    resp2 = session._llm.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    msg2 = resp2.choices[0].message
    print(f"[probe-loop] turn2.finish_reason={resp2.choices[0].finish_reason}")
    print(f"[probe-loop] turn2.text={str(msg2.content or '')[:120]}")
    return 0

    preview = content if len(content) <= 800 else content[:800] + "..."
    print("[probe] no tool call returned")
    print(f"[probe] assistant_content={preview}")
    return 2


def _build_anthropic_client(cfg: dict):
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc
    key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(key_env, "")
    return anthropic.Anthropic(api_key=api_key)


def _build_local_client(cfg: dict):
    """Return a minimal dict config; actual HTTP calls are handled by ToolRuntime."""
    return {
        "provider": "local",
        "endpoint": cfg.get("local_endpoint", "http://localhost:11434"),
        "model": cfg.get("model", "llama3:8b"),
    }


def _model_uses_proxy_forwarded_claude_openai_compat(env: dict) -> bool:
    """Detect proxy-forwarded Claude over OpenAI compat routing with weak tool-call parity."""
    llm_cfg = env.get("llm_backend", {}) if isinstance(env.get("llm_backend", {}), dict) else {}
    provider = str(llm_cfg.get("provider", "openai") or "openai").strip().lower()
    if provider != "openai":
        return False

    model = str(llm_cfg.get("model", "") or "").strip().lower()
    if "claude" not in model:
        return False

    base_url = str(llm_cfg.get("base_url") or llm_cfg.get("baseUrl") or "").strip().lower()
    if not base_url:
        return False

    return "api.openai.com" not in base_url


def _model_uses_deepseek_compat(env: dict, model_name: str = "") -> bool:
    """Detect DeepSeek models routed through an OpenAI-compatible endpoint."""
    llm_cfg = env.get("llm_backend", {}) if isinstance(env.get("llm_backend", {}), dict) else {}
    provider = str(llm_cfg.get("provider", "openai") or "openai").strip().lower()
    if provider != "openai":
        return False

    model = str(model_name or llm_cfg.get("model") or "").strip().lower()
    base_url = str(llm_cfg.get("base_url") or llm_cfg.get("baseUrl") or "").strip().lower()
    return "deepseek" in model or "deepseek" in base_url or "yunwu.ai" in base_url


def _get_openai_chat_create_kwargs(
    env: dict,
    model: str,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """Build provider-compatible kwargs for OpenAI chat.completions.create."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    kwargs.update(extra)

    if _model_uses_deepseek_compat(env, model):
        thinking = env.get("llm_backend", {}).get("thinking")
        if isinstance(thinking, dict):
            thinking_type = str(thinking.get("type") or "").strip()
            if thinking_type:
                kwargs["thinking"] = {"type": thinking_type}

    return kwargs


def _looks_dangerous_shell_command(command: str) -> bool:
    """Return True for obviously destructive shell commands."""
    lower = f" {str(command or '').lower()} "
    blocked = [
        " rm -rf ",
        " mkfs ",
        " shutdown ",
        " reboot ",
        " poweroff ",
        " dd if=",
        " :(){",
    ]
    return any(token in lower for token in blocked)


def _write_agent_shell_status(command: str, cwd: Path, pid: int) -> None:
    payload = {
        "source": "agent_shell",
        "command": command,
        "started_at": int(time.time() * 1000),
        "pid": int(pid),
        "cwd": str(cwd),
    }
    try:
        AGENT_SHELL_STATUS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _clear_agent_shell_status() -> None:
    try:
        if AGENT_SHELL_STATUS_FILE.exists():
            AGENT_SHELL_STATUS_FILE.unlink()
    except Exception:
        pass


def _run_shell_command(
    command: str,
    cwd: Path,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """
    Run command in user's default shell while inheriting process environment.

    The command runs with the same env vars exported by load_environment(),
    including FSLDIR/FREESURFER_HOME/CUDA_VISIBLE_DEVICES.
    """
    cmd = str(command or "").strip()
    if not cmd:
        return {"success": False, "error": "empty command"}

    if _looks_dangerous_shell_command(cmd):
        return {
            "success": False,
            "error": "blocked_dangerous_command",
            "message": "Command blocked by safety policy. Ask user for explicit confirmation.",
        }

    shell_path = os.environ.get("SHELL") or "/bin/bash"
    shell_name = Path(shell_path).name.lower()
    if shell_name in {"bash", "zsh", "sh", "dash", "ksh", "fish"}:
        argv = [shell_path, "-lc", cmd]
    else:
        argv = [shell_path, "-c", cmd]

    env = os.environ.copy()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _write_agent_shell_status(cmd, cwd, proc.pid)
        stdout, stderr = proc.communicate(timeout=max(1, int(timeout_sec)))
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "shell": shell_path,
            "cwd": str(cwd),
        }
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                _stdout, _stderr = proc.communicate(timeout=5)
            except Exception:
                _stdout, _stderr = "", ""
        else:
            _stdout, _stderr = "", ""
        return {
            "success": False,
            "error": f"timeout_after_{int(timeout_sec)}s",
            "stdout": _stdout or exc.stdout or "",
            "stderr": _stderr or exc.stderr or "",
            "shell": shell_path,
            "cwd": str(cwd),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "error": str(exc),
            "shell": shell_path,
            "cwd": str(cwd),
        }
    finally:
        _clear_agent_shell_status()


def _read_workspace_file(path_text: str, workspace: Path, max_chars: int = 12000) -> dict[str, Any]:
    """Read a text file under the workspace for benchmark/tool-assisted reasoning."""
    raw_path = str(path_text or "").strip()
    if not raw_path:
        return {"success": False, "error": "empty_path"}

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        workspace_resolved = workspace

    try:
        candidate.relative_to(workspace_resolved)
    except Exception:
        return {"success": False, "error": "path_outside_workspace", "path": str(candidate)}

    if not candidate.exists() or not candidate.is_file():
        return {"success": False, "error": "file_not_found", "path": str(candidate)}

    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"success": False, "error": "not_utf8_text", "path": str(candidate)}
    except Exception as exc:
        return {"success": False, "error": str(exc), "path": str(candidate)}

    content = text[: max(1, int(max_chars))]
    truncated = len(text) > len(content)
    return {
        "success": True,
        "path": str(candidate),
        "content": content,
        "truncated": truncated,
        "total_chars": len(text),
    }


def _is_benchmark_enabled_from_env() -> bool:
    raw = str(os.environ.get(BENCHMARK_ENV_FLAG, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _looks_file_io_shell_command(command: str) -> bool:
    """Heuristic: True when command is likely to read/write/check files or datasets."""
    cmd = str(command or "")
    if not cmd.strip():
        return False

    # File redirection / piping to files.
    if re.search(r">\s*\S|>>\s*\S|<\s*\S", cmd):
        return True

    fileish_patterns = [
        r"\b(cat|less|more|head|tail|sed|awk|grep|find|ls|stat|du|wc|cut|sort|uniq|realpath|readlink)\b",
        r"\b(cp|mv|rm|mkdir|rmdir|touch|ln|chmod|chown|tar|zip|unzip|gzip|gunzip|7z)\b",
        r"\b(open|xdg-open)\b",
        r"\b(dcm2niix|fsl|freesurfer|recon-all|bet|flirt|fnirt|eddy|topup|fmriprep|qsiprep)\b",
        r"\b(nii|nii\.gz|dcm|dicom|bids|nifti|csv|tsv|vcf|bam|fastq|fasta|h5|hdf5|parquet|pdf|docx|xlsx|pptx)\b",
        r"\bpython\b[\s\S]*\b(open\(|read_csv\(|to_csv\(|save\(|load\(|Path\()",
    ]
    return any(re.search(p, cmd, flags=re.IGNORECASE) for p in fileish_patterns)


def _sanitize_task_filename(task_name: str, max_len: int = 120) -> str:
    raw = str(task_name or "").strip()
    if not raw:
        raw = "task"
    raw = raw.splitlines()[0].strip()
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = re.sub(r"[\x00-\x1f\x7f]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .")
    if not raw:
        raw = "task"
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip()
    return raw


def _slugify_filename_part(text: str, fallback: str, max_len: int = 80) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        raw = fallback
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._-")
    if not raw:
        raw = fallback
    return raw[:max_len].rstrip("._-") or fallback


def _extract_case_id(task_name: str, fallback_source: str = "") -> str:
    text = str(task_name or "").strip()
    fallback_text = str(fallback_source or "").strip()
    if not text and fallback_text:
        text = fallback_text
    if not text:
        return "unknown"

    patterns = [
        r"\bcase[_\s-]*id\s*[:=]\s*([a-zA-Z0-9._-]+)",
        r"\bbenchmark\s*test\s*case\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\btest\s*case\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\bcase\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\bT(\d{1,3})(?=\b|[_-])",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            token = m.group(1)
            if token:
                return _slugify_filename_part(token, "unknown")

    if fallback_text and fallback_text != text:
        for p in patterns:
            m = re.search(p, fallback_text, flags=re.IGNORECASE)
            if m:
                token = m.group(1)
                if token:
                    return _slugify_filename_part(token, "unknown")

    return "unknown"


def _benchmark_report_stem(task_name: str, model_name: str, fallback_source: str = "") -> str:
    case_id = _extract_case_id(task_name, fallback_source=fallback_source)
    model_part = _slugify_filename_part(model_name, "model_unknown")
    return f"{case_id}_{model_part}"


def _benchmark_report_filename(
    task_name: str,
    model_name: str,
    run_index: int | None = None,
    fallback_source: str = "",
) -> str:
    stem = _benchmark_report_stem(task_name, model_name, fallback_source=fallback_source)
    if isinstance(run_index, int) and run_index > 0:
        return f"{stem}_run{run_index}"
    return stem


def _extract_task_summary_for_benchmark(task_markdown: str) -> str:
    """Keep benchmark task details while removing only non-execution scaffolding."""
    text = str(task_markdown or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    kept: list[str] = []
    dropping = False
    drop_headings = {
        "## input requirement",
        "## inputs",
        "## evaluation",
    }

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if stripped.startswith("# ") or lower in {"## objective", "## task description"}:
            dropping = False
            kept.append(line)
            continue

        if lower in drop_headings:
            dropping = True
            continue

        if not dropping:
            kept.append(line)

    summary = "\n".join(kept).strip()
    return summary or text


def _load_system_prompt_text(benchmark_mode: bool, workspace: Path, no_skill_mode: bool = False) -> str:
    candidate_files = []
    if benchmark_mode:
        if no_skill_mode:
            candidate_files.append(workspace / "SOUL_BENCHMARK_NO_SKILL.md")
        candidate_files.append(workspace / "SOUL_BENCHMARK.md")
    candidate_files.append(workspace / "SOUL.md")

    for path in candidate_files:
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                continue
    return ""


def _extract_script_argument_hints(script_path: Path, max_args: int = 10) -> dict[str, list[str]]:
    """Parse simple argparse-style hints from a Python script."""
    if not script_path.exists() or script_path.suffix.lower() != ".py":
        return {"positionals": [], "options": []}

    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return {"positionals": [], "options": []}

    positionals: list[str] = []
    options: list[str] = []
    for m in re.finditer(r"parser\.add_argument\(\s*([\s\S]*?)\)", text):
        arg_blob = m.group(1)
        quoted = re.findall(r'"([^"]+)"', arg_blob)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", arg_blob)
        if not quoted:
            continue

        first = quoted[0].strip()
        if first.startswith("-"):
            long_opt = ""
            for token in quoted:
                if token.startswith("--"):
                    long_opt = token
                    break
            options.append(long_opt or first)
        else:
            positionals.append(first)

        if len(positionals) + len(options) >= max_args:
            break

    # de-duplicate while preserving order
    dedup_pos = list(dict.fromkeys(positionals))
    dedup_opt = list(dict.fromkeys(options))
    return {"positionals": dedup_pos, "options": dedup_opt}


def _build_skill_exec_summary(skills: list[dict[str, Any]], workspace: Path, max_skills: int = 40) -> str:
    """Build a compact executable-entry summary injected into benchmark system prompt."""
    if not skills:
        return ""

    lines: list[str] = [
        "[Executable Skill Entry Summary]",
        "For benchmark with-skills tasks, prefer concrete executable entry points below.",
    ]

    for idx, skill in enumerate(skills[:max_skills], start=1):
        name = str(skill.get("name", "")).strip() or f"skill_{idx}"
        skill_path_raw = skill.get("path")
        skill_path = Path(str(skill_path_raw)) if skill_path_raw else (workspace / "skills" / name)

        scripts_dir = skill_path / "scripts"
        script_candidates: list[Path] = []
        if scripts_dir.exists() and scripts_dir.is_dir():
            for ext in ("*.py", "*.sh", "*.js"):
                script_candidates.extend(sorted(scripts_dir.glob(ext)))

        handler = skill.get("handler")
        handler_path = Path(str(handler)) if handler else None

        primary_entry = script_candidates[0] if script_candidates else handler_path
        if primary_entry is None:
            rel_skill = skill_path
            try:
                rel_skill = skill_path.relative_to(workspace)
            except Exception:
                rel_skill = skill_path
            lines.append(f"- {name}: path={rel_skill}; command_template=<no explicit script/handler found>; params=<unknown>")
            continue

        try:
            rel_entry = primary_entry.relative_to(workspace)
        except Exception:
            rel_entry = primary_entry

        arg_hints = _extract_script_argument_hints(primary_entry)
        positionals = arg_hints.get("positionals", [])
        options = arg_hints.get("options", [])

        if primary_entry.suffix.lower() == ".py":
            base_cmd = f"python {rel_entry}"
        elif primary_entry.suffix.lower() == ".js":
            base_cmd = f"node {rel_entry}"
        else:
            base_cmd = str(rel_entry)

        positional_template = " ".join(f"<{p}>" for p in positionals[:3])
        if positional_template:
            command_template = f"{base_cmd} {positional_template} [options]"
        else:
            command_template = f"{base_cmd} [args]"

        params_parts: list[str] = []
        if positionals:
            params_parts.append(f"positionals={positionals[:4]}")
        if options:
            params_parts.append(f"options={options[:8]}")
        params_text = "; ".join(params_parts) if params_parts else "params=<not detected>"

        lines.append(
            f"- {name}: script_path={rel_entry}; command_template={command_template}; {params_text}"
        )

    return "\n".join(lines)


def _build_skill_hint_summary(skills: list[dict[str, Any]], max_skills: int = 24) -> str:
    """Build short skill introduction phrases for benchmark model and scorer prompts."""
    if not skills:
        return ""

    lines = [
        "[Skill Hints]",
        "Use the following short descriptions as valid capability hints for this benchmark run.",
        "Only count a skill as used when the task solution actually relies on it.",
    ]

    for skill in skills[:max_skills]:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
        summary = re.sub(r"\s+", " ", summary).strip()
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        if not summary:
            summary = f"{name} handles a specialized NeuroClaw workflow."
        lines.append(f"- {name}: {summary}")

    return "\n".join(lines)


def _build_compact_skill_hint_summary(skills: list[dict[str, Any]]) -> str:
    """Build a single-line skill hint for routes that need minimal prompt overhead."""
    if not skills:
        return ""
    skill = skills[0]
    name = str(skill.get("name", "")).strip()
    if not name:
        return ""
    summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > 90:
        summary = summary[:87].rstrip() + "..."
    if not summary:
        summary = "Use only if it clearly improves the task solution."
    return f"[Primary Skill Hint] {name}: {summary}"


def _benchmark_scoring_expectation_for_model(model_name: str) -> str:
    """Return variant-aware scoring expectation text."""
    normalized = str(model_name or "").strip().lower()
    if normalized.endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        return (
            "with-skills expectation: treat cited skills as valid when they match the provided skill hints and the task intent. "
            "Do not require a literal runnable skill entry point just to credit skill usage. "
            "Judge whether the chosen skill is reasonable for the task, whether the plan appears grounded in the skill's described capability, "
            "and whether the final code/commands are adapted to the benchmark task's required inputs and outputs rather than copied verbatim from generic skill text. "
            "The model may internally compare task-required inputs/outputs against a skill's default example inputs/outputs before deciding whether and how to adapt the skill, but that intermediate deliberation does not need to appear in the final report. "
            "Give positive credit when the report shows evidence that the model inspected local skill content and then rewrote the solution around the benchmark task contract. "
            "Also give positive credit when the final solution is clearly skill-informed, meaning it borrows methodology, sequencing, validation ideas, parameter choices, file conventions, or implementation lessons from a relevant skill even if it does not present itself as a full adopted skill workflow. "
            "If no local skill file was actually inspected, skill hints should only refine the task's canonical solution family rather than replacing it with a different task framing, different starting data representation, different main toolchain, or a broader generic workflow. "
            "Do not penalize a with-skills answer merely because it customizes or replaces the skill's default example I/O with task-specific filenames, paths, inputs, or outputs. "
            "Do not penalize a with-skills answer merely because the model likely reasoned about whether to use a skill before choosing the final workflow; judge the final task-directed answer rather than requiring the intermediate comparison to be shown. "
            "Give extra credit when the report explicitly identifies missing task-critical inputs such as atlas/parcellation files, label tables, registration/alignment prerequisites, or other required artifacts, and then asks the user to choose the key analysis options or hyperparameters that materially affect the result. "
            "Penalize hallucinated capabilities only when the claimed skill use clearly contradicts the provided skill hints or task contract. "
            "Penalize reports when the final commands still follow the skill's default interface even though the benchmark task contract is more specific or clearly different. "
            "Also penalize reports when, without inspecting local skill content, the final answer changes the task into a different canonical family than the benchmark asked for, such as switching from tensor-derived metrics to raw-DWI tensor fitting, from HCP Pipelines to a generic fMRIPrep substitute, or from reuse of existing preprocessing transforms to de novo registration without task evidence. "
            "If required inputs are missing, reward answers that say 'Missing required input' but still provide the most task-directed executable fallback they can; penalize answers that retreat into generic orchestration placeholders, wrapper commands, or high-level skill chaining that are less concrete than a no-skills baseline. "
            "Also penalize answers that create an extra orchestrator layer around a skill, such as subprocess wrappers, temporary intermediate artifacts, or shelling out to a skill CLI, when the task can be solved more directly with task-level code or commands."
        )
    if normalized.endswith(BENCHMARK_NO_SKILLS_SUFFIX):
        return (
            "no-skills baseline expectation: do not penalize for not using skills. "
            "Judge only planning quality, tool/code reasonableness, and execution correctness under no-skill constraints."
        )
    return (
        "standard expectation: judge planning quality, tool/code reasonableness, and execution correctness based on the task contract."
    )


def _prompt_with_default(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        value = ""
    return value if value else default


def _load_available_model_catalog() -> tuple[list[dict[str, Any]], str, str]:
    env = load_environment()
    llm_cfg = env.get("llm_backend", {}) if isinstance(env.get("llm_backend", {}), dict) else {}
    available_models = llm_cfg.get("available_models") if isinstance(llm_cfg, dict) else []
    default_model = str(llm_cfg.get("model", "gpt-4o")) if isinstance(llm_cfg, dict) else "gpt-4o"
    default_provider = str(llm_cfg.get("provider", "openai")) if isinstance(llm_cfg, dict) else "openai"

    catalog: list[dict[str, Any]] = []
    if isinstance(available_models, list):
        for item in available_models[:10]:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or "openai").strip()
            model = str(item.get("model") or item.get("id") or item.get("name") or "").strip()
            label = str(item.get("label") or f"{provider} / {model}").strip()
            if provider and model:
                catalog.append({"provider": provider, "model": model, "label": label})

    return catalog, default_provider, default_model


def _prompt_model_selection(
    prompt_label: str,
    default_model: str | None = None,
) -> dict[str, str]:
    catalog, default_provider, env_default_model = _load_available_model_catalog()
    resolved_default_model = str(default_model or "").strip() or env_default_model

    if not catalog:
        model_name = _prompt_with_default(f"{prompt_label} model name", resolved_default_model)
        return {"provider": default_provider, "model": model_name}

    default_index = 1
    for idx, item in enumerate(catalog, start=1):
        if item.get("model") == resolved_default_model:
            default_index = idx
            break

    print(f"Available {prompt_label.lower()} models:")
    for idx, item in enumerate(catalog, start=1):
        marker = " (default)" if idx == default_index else ""
        print(f"  {idx}. {item['label']}{marker}")

    while True:
        choice = _prompt_with_default(f"Select {prompt_label.lower()} model [1-10]", str(default_index))
        try:
            selected_index = int(choice)
        except ValueError:
            print("Please enter a number from 1 to 10.")
            continue
        if 1 <= selected_index <= len(catalog):
            selected = catalog[selected_index - 1]
            return {
                "provider": str(selected.get("provider") or default_provider),
                "model": str(selected.get("model") or resolved_default_model),
            }
        print(f"Please enter a number from 1 to {len(catalog)}.")


def _is_retryable_api_exception(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 425, 429, 500, 502, 503, 504, 529}:
        return True

    status = getattr(exc, "status", None)
    if isinstance(status, int) and status in {408, 425, 429, 500, 502, 503, 504, 529}:
        return True

    retryable_tokens = (
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "rate limit",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(token in text for token in retryable_tokens)


def _retry_api_call(operation_name: str, action, retries: int = 10):
    last_exc: Exception | None = None
    for attempt in range(1, max(1, int(retries or 1)) + 1):
        try:
            return action()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_api_exception(exc) or attempt >= max(1, int(retries or 1)):
                raise
            wait_seconds = min(8.0, 1.0 * (2 ** (attempt - 1)))
            print(
                f"{operation_name} failed ({attempt}/{retries}): {exc}. Retrying in {wait_seconds:.1f}s...",
                flush=True,
            )
            time.sleep(wait_seconds)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation_name} failed unexpectedly")


def _prompt_benchmark_model_name() -> dict[str, str]:
    return _prompt_model_selection("Benchmark")


def _resolve_benchmark_root(raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def _discover_benchmark_task_files(benchmark_root: Path) -> list[Path]:
    task_files = [p for p in benchmark_root.rglob("task.md") if p.is_file()]
    return sorted(task_files, key=lambda p: str(p.parent.relative_to(benchmark_root)).lower())


def _load_skill_loader_class():
    import importlib.util as _ilu

    loader_mod = _ilu.spec_from_file_location(
        "neuroclaw_skill_loader",
        REPO_ROOT / "core" / "skill-loader" / "loader.py",
    )
    if loader_mod is None or loader_mod.loader is None:
        raise RuntimeError("Cannot find core/skill-loader/loader.py")
    mod = __import__("importlib").util.module_from_spec(loader_mod)
    loader_mod.loader.exec_module(mod)
    return mod.SkillLoader


def _discover_task_contracts(benchmark_root: Path) -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    for task_file in _discover_benchmark_task_files(benchmark_root):
        task_text = ""
        try:
            task_text = task_file.read_text(encoding="utf-8")
        except Exception:
            task_text = ""
        case_id = _extract_case_id(task_text, fallback_source=task_file.parent.name)
        tasks[case_id] = task_file
    return tasks


def _parse_benchmark_report_filename(path: Path) -> tuple[str, str, int] | None:
    stem = path.stem
    if not re.match(r"^\d+_.+_run\d+$", stem, flags=re.IGNORECASE):
        return None
    if "_" not in stem:
        return None
    run_index = 1
    run_match = re.search(r"_run(\d+)$", stem, flags=re.IGNORECASE)
    if run_match:
        try:
            run_index = int(run_match.group(1))
        except Exception:
            run_index = 1
        stem = stem[: run_match.start()]
    case_id, model_name = stem.split("_", 1)
    case_id = _slugify_filename_part(case_id, "unknown")
    model_name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    if not case_id or not model_name:
        return None
    return case_id, model_name, max(1, run_index)


def _discover_benchmark_reports(output_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    reports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for file_path in sorted(output_dir.rglob("*.md")):
        stem_lower = file_path.stem.lower()
        if stem_lower.startswith("benchmark_leaderboard_") or stem_lower.startswith("benchmark_scores_"):
            continue
        parsed = _parse_benchmark_report_filename(file_path)
        if not parsed:
            continue
        case_id, model_name, run_index = parsed
        reports.setdefault(model_name, {}).setdefault(case_id, []).append(
            {
                "run_index": run_index,
                "report_file": file_path,
            }
        )

    for case_map in reports.values():
        for run_entries in case_map.values():
            run_entries.sort(key=lambda item: (int(item.get("run_index", 1)), str(item.get("report_file", ""))))
    return reports


def _mean_and_variance(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return round(mean, 4), round(variance, 4)


def _model_uses_gemini_compat(model_name: str) -> bool:
    """Detect Gemini models for provider-specific compatibility handling."""
    name = str(model_name or "").strip().lower()
    return "gemini" in name

def _model_uses_grok_compat(model_name: str) -> bool:
    """Detect Grok models for benchmark-specific prompt shaping."""
    name = str(model_name or "").strip().lower()
    return "grok" in name


def _model_is_withskills_variant(model_name: str) -> bool:
    """Return True when the benchmark model name is the with-skills variant."""
    name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    return name.endswith(BENCHMARK_WITH_SKILLS_SUFFIX)


def _split_skill_condition(model_name: str) -> tuple[str, str]:
    name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    if name.endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        return name[: -len(BENCHMARK_WITH_SKILLS_SUFFIX)], "with_skills"
    if name.endswith(BENCHMARK_NO_SKILLS_SUFFIX):
        return name[: -len(BENCHMARK_NO_SKILLS_SUFFIX)], "no_skills"
    return name, "unknown"


def _compute_normalized_gain(with_skills_score: float, no_skills_score: float) -> float:
    delta = float(with_skills_score) - float(no_skills_score)
    if delta >= 0:
        denom = max(1e-6, 100.0 - float(no_skills_score))
    else:
        denom = max(1e-6, float(no_skills_score))
    gain = delta / denom
    if gain > 1.0:
        gain = 1.0
    if gain < -1.0:
        gain = -1.0
    return round(gain, 4)


def _interpret_normalized_gain(avg_abs_improvement: float, avg_gain: float) -> str:
    if avg_gain >= 0.5 and avg_abs_improvement < 5.0:
        return (
            "High normalized gain with low absolute improvement suggests ceiling effects; "
            "proportional benefit exists but raw improvement is limited."
        )
    if avg_gain >= 0.5 and avg_abs_improvement >= 5.0:
        return (
            "High normalized gain with high absolute improvement suggests substantial "
            "scaffolding benefit."
        )
    if avg_gain > 0:
        return (
            "Positive normalized gain indicates proportional benefit from skills; "
            "consistency refers to similar proportion, not identical absolute improvement."
        )
    if avg_gain == 0:
        return "No measurable proportional gain between with-skills and no-skills conditions."
    return "Negative gain indicates skills condition underperformed the no-skills baseline."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _clamp_score_1_10(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 1.0
    if num < 1:
        return 1.0
    if num > 10:
        return 10.0
    return round(num, 2)


def _extract_token_usage_from_response(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _read_int(field: str) -> int:
        value = 0
        try:
            if isinstance(usage, dict):
                value = int(usage.get(field, 0) or 0)
            else:
                value = int(getattr(usage, field, 0) or 0)
        except Exception:
            value = 0
        return max(0, value)

    return {
        "prompt_tokens": _read_int("prompt_tokens"),
        "completion_tokens": _read_int("completion_tokens"),
        "total_tokens": _read_int("total_tokens"),
    }


def _classify_shell_command_tool_kinds(command: str) -> set[str]:
    cmd = str(command or "")
    kinds: set[str] = {"shell"}
    if re.search(r"\bpython(\d+(\.\d+)*)?\b", cmd, flags=re.IGNORECASE):
        kinds.add("python")
    if re.search(r"\b(docker|podman)\b", cmd, flags=re.IGNORECASE):
        kinds.add("docker")
    if re.search(
        r"\b(fsl|fslmaths|fslstats|bet|flirt|fnirt|eddy|topup|fast|melodic|feat)\b",
        cmd,
        flags=re.IGNORECASE,
    ):
        kinds.add("fsl")
    if re.search(r"\b(freesurfer|recon-all|mri_\w+)\b", cmd, flags=re.IGNORECASE):
        kinds.add("freesurfer")
    if re.search(r"\b(dcm2niix|dcm2nii)\b", cmd, flags=re.IGNORECASE):
        kinds.add("dcm2niix")
    return kinds


def _normalize_skill_token(text: str) -> str:
    """Normalize skill tokens for robust command/text matching."""
    token = str(text or "").lower()
    return re.sub(r"[^a-z0-9]+", "", token)


def _infer_skills_from_text(text: str, skills: list[dict[str, Any]]) -> list[str]:
    """Infer referenced skills from command or answer text using skill and folder names."""
    norm_text = _normalize_skill_token(text)
    if not norm_text or not skills:
        return []

    inferred: list[str] = []
    for skill in skills:
        skill_name = str(skill.get("name", "")).strip()
        if not skill_name:
            continue

        dir_name = ""
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""

        for candidate in (skill_name, dir_name):
            norm_candidate = _normalize_skill_token(candidate)
            if norm_candidate and norm_candidate in norm_text:
                inferred.append(skill_name)
                break

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in inferred:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    return unique_names


def _extract_skill_contract_candidates(skill_md_path: Any) -> list[str]:
    """Read static contract-related skill references from SKILL.md.

    This is only used as a candidate source for expansion. Actual counting still
    requires relevance support from the current task plan/commands.
    """
    try:
        skill_md = Path(skill_md_path)
    except Exception:
        return []
    if not skill_md.exists():
        return []

    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", text):
        token = str(match.group(1) or "").strip()
        key = _normalize_skill_token(token)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(token)
    return candidates


def _build_skill_context_text(skill: dict[str, Any]) -> str:
    parts = [
        str(skill.get("name", "")).strip(),
        str(skill.get("summary_en", "")).strip(),
        str(skill.get("description", "")).strip(),
    ]
    return " ".join(part for part in parts if part).strip()


def _extract_specialized_skill_signals(skill: dict[str, Any]) -> list[str]:
    """Extract high-specificity execution signals for a skill from its metadata/docs."""
    context = _build_skill_context_text(skill)
    signals: list[str] = []
    seen: set[str] = set()

    for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]{3,}", context):
        raw = str(token or "").strip()
        norm = _normalize_skill_token(raw)
        if len(norm) < 5 or norm in seen:
            continue
        if not any(ch.isdigit() for ch in raw) and raw.lower() == raw and "-" not in raw and "_" not in raw and "/" not in raw and "." not in raw:
            # Skip broad plain words like 'install', 'workflow', 'execution'.
            continue
        seen.add(norm)
        signals.append(raw)

    return signals


def _expand_skill_contract_usage(
    used_skills: list[str],
    skills: list[dict[str, Any]],
    assistant_response: str,
    tool_events: list[dict[str, Any]],
) -> list[str]:
    """Expand counted skills using skill contracts, filtered by task-plan relevance.

    A contract candidate is counted only when:
    - it is referenced by a top-level used skill's SKILL.md, and
    - the current plan/commands mention evidence aligned with that candidate's
      capability hint or explicit skill name.
    """
    if not used_skills or not skills:
        return []

    skill_by_key: dict[str, dict[str, Any]] = {}
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        key = _normalize_skill_token(name)
        if key:
            skill_by_key[key] = skill
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""
        dir_key = _normalize_skill_token(dir_name)
        if dir_key and dir_key not in skill_by_key:
            skill_by_key[dir_key] = skill

    plan_parts: list[str] = []
    thinking_body, commands_body = _split_benchmark_answer_sections(assistant_response)
    if thinking_body:
        plan_parts.append(thinking_body)
    if commands_body:
        plan_parts.append(commands_body)
    for event in tool_events:
        command = str(event.get("command", "")).strip()
        if command:
            plan_parts.append(command)
    plan_text = "\n".join(part for part in plan_parts if part).strip()
    norm_plan_text = _normalize_skill_token(plan_text)
    if not norm_plan_text:
        return []

    expanded: list[str] = []
    expanded_seen: set[str] = set()
    already_used = {_normalize_skill_token(name) for name in used_skills}

    for used_name in used_skills:
        used_key = _normalize_skill_token(used_name)
        used_skill = skill_by_key.get(used_key)
        if not used_skill:
            continue

        for candidate_name in _extract_skill_contract_candidates(used_skill.get("skill_md")):
            candidate_key = _normalize_skill_token(candidate_name)
            candidate_skill = skill_by_key.get(candidate_key)
            if not candidate_key or not candidate_skill:
                continue
            canonical_name = str(candidate_skill.get("name", "")).strip()
            canonical_key = _normalize_skill_token(canonical_name)
            if not canonical_key or canonical_key in already_used or canonical_key in expanded_seen:
                continue

            candidate_signal = False
            if candidate_key in norm_plan_text:
                candidate_signal = True
            else:
                for signal in _extract_specialized_skill_signals(candidate_skill):
                    norm_signal = _normalize_skill_token(signal)
                    if norm_signal and norm_signal in norm_plan_text:
                        candidate_signal = True
                        break

            if candidate_signal:
                expanded_seen.add(canonical_key)
                expanded.append(canonical_name)

    return expanded


def _summarize_skill_usage(
    tool_events: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    assistant_response: str = "",
) -> dict[str, Any]:
    """Summarize unique skill usage from explicit runtime signals only.

    Benchmark reports should distinguish between skills that actually influenced
    execution and skills that were only mentioned during contract audit or plan
    narration. Counting raw answer-text mentions as usage creates false positives
    in with-skills runs where the model discusses a skill but ultimately solves
    the task directly without adopting it.
    """
    seen: set[str] = set()
    used_skills: list[str] = []
    read_seen: set[str] = set()
    read_skills: list[str] = []
    catalog: dict[str, str] = {}

    for skill in skills:
        skill_name = str(skill.get("name", "")).strip()
        if not skill_name:
            continue
        catalog[_normalize_skill_token(skill_name)] = skill_name
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""
        if dir_name:
            catalog.setdefault(_normalize_skill_token(dir_name), skill_name)

    def record(names: list[str]) -> None:
        for name in names:
            key = _normalize_skill_token(str(name).strip())
            canonical = catalog.get(key)
            if not key or not canonical or key in seen:
                continue
            seen.add(key)
            used_skills.append(canonical)

    def record_reads(names: list[str]) -> None:
        for name in names:
            key = _normalize_skill_token(str(name).strip())
            canonical = catalog.get(key)
            if not key or not canonical or key in read_seen:
                continue
            read_seen.add(key)
            read_skills.append(canonical)

    for event in tool_events:
        tool = str(event.get("tool", "")).strip()
        explicit = event.get("skills_used")
        if isinstance(explicit, list):
            record([str(name).strip() for name in explicit if str(name).strip()])

        command = str(event.get("command", "")).strip()
        if command:
            record(_infer_skills_from_text(command, skills))
            if tool == "read_workspace_file":
                record_reads(_infer_skills_from_text(command, skills))

        payload_skills = _extract_skills_from_result_payload(event.get("result"))
        if payload_skills:
            record(payload_skills)
            if tool == "read_workspace_file":
                record_reads(payload_skills)

    # Assistant-text mentions are not runtime evidence. In benchmark reports we
    # only count explicit tool/runtime signals as skill usage, otherwise direct
    # answers that merely cite or imitate a skill inflate usage statistics.
    response_text = str(assistant_response or "")
    normalized_response = response_text.lower()
    contract_not_adopted = (
        "contract decision: not adopted" in normalized_response
        or "contract decision: skill not adopted" in normalized_response
        or "skill not adopted" in normalized_response
    )

    # Only expand from already-confirmed runtime usage. This preserves canonical
    # aliases without treating answer-text narration as evidence of actual use.
    record(_expand_skill_contract_usage(used_skills, skills, "", tool_events))

    return {
        "total_unique_skills": len(used_skills),
        "skills": used_skills,
        "read_unique_skills": len(read_skills),
        "read_skills": read_skills,
    }


def _extract_skills_from_result_payload(payload: Any) -> list[str]:
    """Collect explicitly reported skill usage from nested handler results.

    Skills may invoke other lower-level skills for dependency installation or other
    atomic work. Handlers must surface that chain through explicit skill-oriented
    fields such as skills_used / used_skills / invoked_skills; this extractor
    flattens those runtime traces only.
    """
    seen: set[str] = set()
    collected: list[str] = []
    candidate_keys = {
        "skill",
        "skill_name",
        "skills",
        "skills_used",
        "used_skills",
        "invoked_skills",
        "called_skills",
        "dependent_skills",
    }

    def record(value: Any) -> None:
        name = str(value or "").strip()
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        collected.append(name)

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key or "").strip().lower()
                if key_text in candidate_keys:
                    if isinstance(item, list):
                        for entry in item:
                            if isinstance(entry, (dict, list)):
                                walk(entry, key_text)
                            else:
                                record(entry)
                        continue
                    if isinstance(item, dict):
                        nested_name = item.get("name") or item.get("skill") or item.get("skill_name")
                        if nested_name:
                            record(nested_name)
                        walk(item, key_text)
                        continue
                    record(item)
                    continue
                walk(item, key_text)
            return

        if isinstance(value, list):
            for item in value:
                walk(item, parent_key)
            return

        if parent_key in candidate_keys:
            record(value)

    walk(payload)
    return collected


def _extract_usage_count_from_report(report_text: str) -> int | None:
    text = str(report_text or "")
    m = re.search(r"-\s*Skills used \(unique\):\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            return max(0, int(m.group(1)))
        except Exception:
            return None

    count = len(re.findall(r"^\d+\.\s+\[[^\]]+\]", text, flags=re.MULTILINE))
    return count if count > 0 else None


def _extract_elapsed_seconds_from_report(report_text: str) -> float | None:
    text = str(report_text or "")
    m = re.search(r"-\s*Elapsed seconds:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return max(0.0, float(m.group(1)))
    except Exception:
        return None


def _extract_token_usage_from_report_text(report_text: str) -> dict[str, int]:
    text = str(report_text or "")
    m = re.search(
        r"-\s*Token usage:\s*prompt=(\d+),\s*completion=(\d+),\s*total=(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        return {
            "prompt_tokens": max(0, int(m.group(1))),
            "completion_tokens": max(0, int(m.group(2))),
            "total_tokens": max(0, int(m.group(3))),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _build_skill_catalog_summary(skills: list[dict[str, Any]], max_skills: int = 80) -> str:
    """Build a concise skill catalog for benchmark evaluator/model context."""
    if not skills:
        return ""

    lines = [
        "[Available Skill Library]",
        "The following NeuroClaw skills are available in this benchmark run.",
        "Treat these skills as valid in-repo capabilities when judging feasibility or completeness.",
    ]
    for skill in skills[:max_skills]:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
        if not summary:
            summary = f"{name} provides specialized workflow support."
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines)

def _select_benchmark_skill_hints(skills: list[dict[str, Any]], model_name: str = "") -> list[dict[str, Any]]:
    """Choose a smaller skill-hint set for models with weak tool-call reliability."""
    if not skills:
        return []
    if _model_uses_grok_compat(model_name):
        return skills[:1]
    return skills


def _format_duration(seconds: float | int | None) -> str:
    """Format duration as H:MM:SS (or M:SS for short spans)."""
    if seconds is None:
        return "-"
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return "-"

    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _split_benchmark_answer_sections(answer_text: str) -> tuple[str, str]:
    """Extract Solution Thinking / Commands Or Code bodies from benchmark answer text.

    This prevents duplicated section headers in saved benchmark reports when the model
    already follows the required two-section output format.
    """
    text = str(answer_text or "").strip()
    if not text:
        return "(empty)", ""

    m_thinking = re.search(r"^##\s+Solution\s+Thinking\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    m_commands = re.search(r"^##\s+Commands\s+Or\s+Code\s*$", text, flags=re.IGNORECASE | re.MULTILINE)

    if not m_thinking or not m_commands:
        return text, ""

    if m_thinking.start() > m_commands.start():
        # Unexpected order; keep raw text as thinking and leave commands empty.
        return text, ""

    thinking = text[m_thinking.end(): m_commands.start()].strip()
    commands = text[m_commands.end():].strip()
    return (thinking or "(empty)"), commands


def _is_incomplete_benchmark_response(answer_text: str) -> bool:
    """Return True when a benchmark answer stops at tool traces or inspection intent."""
    text = str(answer_text or "").strip()
    if not text:
        return True

    lower = text.lower()
    if "<tool_call>" in lower or "<tool_result>" in lower:
        return True

    thinking_body, commands_body = _split_benchmark_answer_sections(text)
    commands_lower = str(commands_body or "").strip().lower()

    if not commands_lower or commands_lower == "1. no tool command was used.":
        return True

    inspect_markers = (
        "let me inspect",
        "i'll inspect",
        "i will inspect",
        "need to inspect",
        "inspect the ",
    )
    if any(marker in lower for marker in inspect_markers):
        meaningful_commands = any(
            token in commands_lower
            for token in ("```", "conda ", "python ", "bash", "#!/", "docker ", "export ")
        )
        if not meaningful_commands:
            return True

    return thinking_body.strip() == "(empty)" and not commands_lower


def _tool_events_show_inspection_only(tool_events: list[dict[str, Any]]) -> bool:
    """Return True when the trace only inspects skill files without task execution."""
    if not tool_events:
        return False

    saw_skill_read = False
    for event in tool_events:
        tool = str(event.get("tool") or "").strip()
        command = str(event.get("command") or "").strip().lower()
        executed = bool(event.get("executed", False))

        if tool == "read_workspace_file":
            if "skills/" in command or command.endswith("skill.md"):
                saw_skill_read = True
            continue

        if tool == "run_shell_command":
            # In benchmark mode, suggested-only shell commands are inspection intent, not execution.
            if not executed:
                continue
            return False

        if executed:
            return False

    return saw_skill_read


def _score_single_benchmark_case(
    llm_client: Any,
    task_text: str,
    report_text: str,
    model_name: str,
    scorer_model: str,
    reference_examples: list[dict[str, Any]] | None = None,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scoring_system = (
        "You are a strict benchmark evaluator for neuroscience-agent outputs. "
        "Score only from the provided task and report. "
        "Return JSON only."
    )
    reference_examples = reference_examples or []
    skills = skills or []
    reference_section = ""
    skill_hint_section = ""
    if str(model_name or "").strip().lower().endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        skill_hint_section = _build_skill_hint_summary(skills)
    if reference_examples:
        lines: list[str] = [
            "Reference scored examples for calibration (use them to keep scoring criteria consistent):",
        ]
        for idx, ref in enumerate(reference_examples, start=1):
            ref_model = str(ref.get("model_name", ""))
            ref_run = max(1, int(ref.get("run_index", 1) or 1))
            ref_report = str(ref.get("report_text", ""))
            ref_report_preview = ref_report[:1200] if len(ref_report) > 1200 else ref_report
            lines.extend(
                [
                    f"[{idx}] model={ref_model} run={ref_run}",
                    "Scored dimensions:",
                    f"- planning_completeness: {ref.get('planning_completeness')}",
                    f"- action_reasonableness: {ref.get('tool_reasonableness')}",
                    f"- code_command_correctness: {ref.get('code_command_correctness')}",
                    f"- brief_justification: {ref.get('brief_justification', '')}",
                    "Report log (truncated if long):",
                    ref_report_preview,
                    "",
                ]
            )
        reference_section = "\n".join(lines)

    expectation_hint = _benchmark_scoring_expectation_for_model(model_name)
    scoring_user = (
        "Evaluate the benchmark report on a 1-10 scale for three dimensions:\n"
        "1) planning_completeness\n"
        "2) action_reasonableness\n"
        "3) code_command_correctness\n"
        "Do not score usage_efficiency here; it is derived separately from skill usage counts across models for the same case.\n"
        "Important: action selection quality is still very important.\n"
        "For with-skills reports, do not require the final answer to expose the model's internal skill-vs-task comparison or any explicit contract-audit lines.\n"
        "For with-skills reports, reward solutions that inspect or rely on local skill content and then adapt that material to the benchmark task's required inputs, outputs, filenames, and validation rules.\n"
        "For with-skills reports, also reward skill-informed answers that clearly borrow relevant methodology, sequencing, validation logic, parameterization, or implementation lessons from a hinted skill even when the final answer does not present a full explicit skill workflow.\n"
        "Do NOT reduce scores merely because a with-skills report reads or inspects a relevant local skill before adapting it to the task; that behavior is positive evidence when the final workflow remains task-directed.\n"
        "Do NOT reduce scores merely because the model appears to have thought about whether to use a skill, compared task-required I/O against skill-default I/O, or revised away an intermediate skill-based plan before presenting the final answer.\n"
        "If no local skill file was inspected, do NOT reward skill hints being used to replace the benchmark's canonical task family, starting representation, or main toolchain; in that situation, skill hints should only sharpen the direct task solution.\n"
        "Do not treat task-specific rewrites of a skill's default example I/O as a flaw by themselves.\n"
        "When required artifacts or decisions are missing, give higher scores to reports that explicitly ask the user for task-critical inputs such as atlas/parcellation files, label tables, alignment/registration choices, or other result-shaping hyperparameters, especially when those choices materially change the downstream analysis.\n"
        "Also give higher scores when the report makes the reuse-versus-recompute decision explicit: if required derived maps or intermediates already exist and satisfy the task contract, reusing them should be rewarded; if they are missing, the report should justify deriving them from the nearest skill-compatible or pipeline-compatible path instead of silently assuming either branch.\n"
        "For tasks involving atlas/parcellation summaries or derived imaging maps, reward reports that explicitly validate spatial consistency, state atlas/label/LUT provenance, and distinguish outputs summarized from pre-existing maps versus outputs that depend on newly generated maps in the current workflow.\n"
        "For with-skills reports, reduce action_reasonableness and code_command_correctness only when the final commands or code still keep the skill's default interface instead of the task-required interface.\n"
        "For with-skills reports, also reduce action_reasonableness and code_command_correctness when the answer drifts into a different canonical workflow than the task requested, especially when no local skill file was inspected and the drift appears to come from over-generalizing the skill hint.\n"
        "If the report says 'Missing required input', do not reward generic skill orchestration alone; prefer the answer that still gives the most task-specific executable fallback, concrete file expectations, and validation logic under the missing-input constraint.\n"
        "Do NOT penalize solutions purely for being longer when the added content improves coverage, safety, validation, or clarifies key assumptions/choices.\n"
        "Prefer solutions that comprehensively consider the problem (inputs, outputs, edge cases, validation, reproducibility, failure modes, and clinically/analytically meaningful QC) even if the plan/code is not the shortest possible.\n"
        "Do NOT reduce scores merely because the report includes environment setup or dependency installation steps. Only reduce scores if those steps are clearly irrelevant, replace required task execution, or crowd out task-specific commands/validation.\n"
        "Do NOT reduce scores merely because the report includes a small extra validation script or post-run verification block. Reward it when it concretely checks the benchmark contract, output artifacts, schema, numerical integrity, or reproducibility assumptions.\n"
        "Do NOT reduce scores merely because the report includes a thin task-specific adapter layer that rewrites paths, normalizes fields, enforces the benchmark export schema, or preserves degraded partial results around a skill or native tool; reward that when it improves contract compliance.\n"
        "If the report adds wrappers/orchestration, only reduce action_reasonableness when those wrappers are clearly redundant, create a second workflow or broad generic scaffolding, AND do not provide additional validation, traceability, contract compliance, or necessary interface adaptation.\n"
        "Output strict JSON with keys exactly:\n"
        "planning_completeness, tool_reasonableness, code_command_correctness, "
        "brief_justification.\n"
        "Variant-aware expectation:\n"
        f"- {expectation_hint}\n"
        "\n"
        + (f"{skill_hint_section}\n\n" if skill_hint_section else "")
        + f"Task contract:\n{task_text}\n\n"
        + (f"{reference_section}\n\n" if reference_section else "")
        + f"Target model: {model_name}\n"
        + f"Model report:\n{report_text}\n"
    )
    resp = _retry_api_call(
        "Benchmark scoring request",
        lambda: llm_client.chat.completions.create(
            model=scorer_model,
            messages=[
                {"role": "system", "content": scoring_system},
                {"role": "user", "content": scoring_user},
            ],
        ),
        retries=10,
    )
    content = ""
    try:
        content = str(resp.choices[0].message.content or "")
    except Exception:
        content = ""

    parsed = _extract_json_object(content) or {}
    planning = _clamp_score_1_10(parsed.get("planning_completeness", 1))
    tool = _clamp_score_1_10(parsed.get("tool_reasonableness", 1))
    code = _clamp_score_1_10(parsed.get("code_command_correctness", 1))
    return {
        "planning_completeness": planning,
        "tool_reasonableness": tool,
        "code_command_correctness": code,
        "usage_count": _extract_usage_count_from_report(report_text),
        "brief_justification": str(parsed.get("brief_justification", "")).strip(),
        "raw_model_output": content,
    }


def _get_benchmark_scorer_client(scorer_model: str = BENCHMARK_SCORER_MODEL) -> Any:
    global _BENCHMARK_SCORER_CLIENT_CACHE
    resolved_scorer_model = str(scorer_model or "").strip() or BENCHMARK_SCORER_MODEL
    with _BENCHMARK_SCORER_CLIENT_CACHE_LOCK:
        cached = _BENCHMARK_SCORER_CLIENT_CACHE.get(resolved_scorer_model)
        if cached is not None:
            return cached

        scorer_env = load_environment()
        llm_cfg = scorer_env.setdefault("llm_backend", {})
        llm_cfg["provider"] = "openai"
        llm_cfg["model"] = resolved_scorer_model
        _BENCHMARK_SCORER_CLIENT_CACHE[resolved_scorer_model] = build_llm_client(scorer_env)
        return _BENCHMARK_SCORER_CLIENT_CACHE[resolved_scorer_model]


def _score_batch_benchmark_case(
    llm_client: Any,
    task_text: str,
    runs_by_model: dict[str, list[dict[str, Any]]],
    scorer_model: str,
    case_id: str = "",
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Score multiple model-runs for the same task in a single LLM call."""
    skills = skills or []
    run_blocks: list[str] = []
    expectation_lines: list[str] = []
    seen_models: set[str] = set()
    include_skill_hints = False
    for model_name in sorted(runs_by_model.keys()):
        if model_name not in seen_models:
            seen_models.add(model_name)
            expectation_lines.append(
                f"- {model_name}: {_benchmark_scoring_expectation_for_model(model_name)}"
            )
            if str(model_name or "").strip().lower().endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
                include_skill_hints = True
        for run_data in runs_by_model[model_name]:
            run_index = int(run_data.get("run_index", 1) or 1)
            report_text = str(run_data.get("report_text", ""))
            run_blocks.append(
                f"=== MODEL: {model_name} | RUN: {run_index} ===\n"
                f"{report_text}\n"
                "=== END REPORT ==="
            )
    skill_hint_section = _build_skill_hint_summary(skills) if include_skill_hints else ""

    scoring_system = (
        "You are a strict benchmark evaluator for neuroscience-agent outputs. "
        "You evaluate multiple model-runs for the SAME task to keep scoring criteria consistent. "
        "Return JSON only."
    )
    scoring_user = (
        "Evaluate EACH report on a 1-10 scale for three dimensions:\n"
        "1) planning_completeness\n"
        "2) action_reasonableness\n"
        "3) code_command_correctness\n"
        "Do not score usage_efficiency here; it is computed separately from skill usage counts.\n"
        "For with-skills reports, do not require the final answer to expose the model's internal skill-vs-task comparison or any explicit contract-audit lines.\n"
        "For with-skills reports, reward solutions that inspect or rely on local skill content and then adapt that material to the benchmark task's required inputs, outputs, filenames, and validation rules.\n"
        "For with-skills reports, also reward skill-informed answers that clearly borrow relevant methodology, sequencing, validation logic, parameterization, or implementation lessons from a hinted skill even when the final answer does not present a full explicit skill workflow.\n"
        "Do NOT reduce scores merely because a with-skills report reads or inspects a relevant local skill before adapting it to the task; that behavior is positive evidence when the final workflow remains task-directed.\n"
        "Do NOT reduce scores merely because the model appears to have thought about whether to use a skill, compared task-required I/O against skill-default I/O, or revised away an intermediate skill-based plan before presenting the final answer.\n"
        "If no local skill file was inspected, do NOT reward skill hints being used to replace the benchmark's canonical task family, starting representation, or main toolchain; in that situation, skill hints should only sharpen the direct task solution.\n"
        "Do not treat task-specific rewrites of a skill's default example I/O as a flaw by themselves.\n"
        "When required artifacts or decisions are missing, give higher scores to reports that explicitly ask the user for task-critical inputs such as atlas/parcellation files, label tables, alignment/registration choices, or other result-shaping hyperparameters, especially when those choices materially change the downstream analysis.\n"
        "Also give higher scores when the report makes the reuse-versus-recompute decision explicit: if required derived maps or intermediates already exist and satisfy the task contract, reusing them should be rewarded; if they are missing, the report should justify deriving them from the nearest skill-compatible or pipeline-compatible path instead of silently assuming either branch.\n"
        "For tasks involving atlas/parcellation summaries or derived imaging maps, reward reports that explicitly validate spatial consistency, state atlas/label/LUT provenance, and distinguish outputs summarized from pre-existing maps versus outputs that depend on newly generated maps in the current workflow.\n"
        "For with-skills reports, reduce action_reasonableness and code_command_correctness only when the final commands or code still keep the skill's default interface instead of the task-required interface.\n"
        "For with-skills reports, also reduce action_reasonableness and code_command_correctness when the answer drifts into a different canonical workflow than the task requested, especially when no local skill file was inspected and the drift appears to come from over-generalizing the skill hint.\n"
        "If a report says 'Missing required input', do not reward generic skill orchestration alone; prefer the report that still gives the most task-specific executable fallback, concrete file expectations, and validation logic under the missing-input constraint.\n"
        "Do NOT penalize solutions purely for being longer when the added content improves coverage, safety, validation, or clarifies key assumptions/choices.\n"
        "Prefer solutions that comprehensively consider the problem (inputs, outputs, edge cases, validation, reproducibility, failure modes, and clinically/analytically meaningful QC) even if the plan/code is not the shortest possible.\n"
        "Do NOT reduce scores merely because the report includes environment setup or dependency installation steps. Only reduce scores if those steps are clearly irrelevant, replace required task execution, or crowd out task-specific commands/validation.\n"
        "Do NOT reduce scores merely because the report includes a small extra validation script or post-run verification block. Reward it when it concretely checks the benchmark contract, output artifacts, schema, numerical integrity, or reproducibility assumptions.\n"
        "Do NOT reduce scores merely because the report includes a thin task-specific adapter layer that rewrites paths, normalizes fields, enforces the benchmark export schema, or preserves degraded partial results around a skill or native tool; reward that when it improves contract compliance.\n"
        "For with-skills reports, reduce action_reasonableness only when wrappers/orchestration around the skill are clearly redundant, create a second workflow or broad generic scaffolding, AND do not provide additional validation, traceability, contract compliance, or necessary interface adaptation.\n"
        "\n"
        "Return a strict JSON ARRAY. Each item must contain keys exactly:\n"
        "model, run_index, planning_completeness, tool_reasonableness, code_command_correctness, brief_justification\n"
        "\n"
        "Variant-aware expectations (apply per model):\n"
        + ("\n".join(expectation_lines) if expectation_lines else "- (none)")
        + "\n\n"
        + (f"{skill_hint_section}\n\n" if skill_hint_section else "")
        + f"Task contract:\n{task_text}\n\n"
        "Reports:\n"
        + "\n\n".join(run_blocks)
    )

    resp = _retry_api_call(
        "Benchmark batch scoring request",
        lambda: llm_client.chat.completions.create(
            model=scorer_model,
            messages=[
                {"role": "system", "content": scoring_system},
                {"role": "user", "content": scoring_user},
            ],
        ),
        retries=10,
    )
    content = ""
    try:
        content = str(resp.choices[0].message.content or "")
    except Exception:
        content = ""

    parsed_list: list[Any] = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            parsed_list = parsed
        elif isinstance(parsed, dict):
            parsed_list = [parsed]
    except Exception:
        parsed_list = []

    def _normalize_model_name_key(name: str) -> str:
        # Normalize model names to tolerate case/punctuation/style variations
        # from the scorer model output (e.g., "MiniMax-M2.7" vs "minimax-m2.7").
        return re.sub(r"[^a-z0-9]", "", str(name or "").strip().lower())

    results_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in runs_by_model.keys()}
    normalized_model_lookup: dict[str, str] = {}
    for canonical_model in runs_by_model.keys():
        normalized_model_lookup.setdefault(
            _normalize_model_name_key(canonical_model),
            canonical_model,
        )
    expected_pairs: set[tuple[str, int]] = set()
    for model_name, run_entries in runs_by_model.items():
        for run_data in run_entries:
            expected_pairs.add((model_name, max(1, int(run_data.get("run_index", 1) or 1))))

    for item in parsed_list:
        if not isinstance(item, dict):
            continue
        raw_model = str(item.get("model", "")).strip()
        model = raw_model
        if model not in runs_by_model:
            model = normalized_model_lookup.get(_normalize_model_name_key(raw_model), "")
        if model not in runs_by_model:
            continue
        run_index = max(1, int(item.get("run_index", 1) or 1))

        report_text = ""
        for run_data in runs_by_model[model]:
            if int(run_data.get("run_index", 1) or 1) == run_index:
                report_text = str(run_data.get("report_text", ""))
                break

        results_by_model[model].append(
            {
                "run_index": run_index,
                "planning_completeness": _clamp_score_1_10(item.get("planning_completeness", 1)),
                "tool_reasonableness": _clamp_score_1_10(item.get("tool_reasonableness", 1)),
                "code_command_correctness": _clamp_score_1_10(item.get("code_command_correctness", 1)),
                "usage_count": _extract_usage_count_from_report(report_text),
                "brief_justification": str(item.get("brief_justification", "")).strip(),
                "raw_model_output": content,
            }
        )

    scored_pairs: set[tuple[str, int]] = set()
    for model_name, run_entries in results_by_model.items():
        for run in run_entries:
            scored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

    missing_pairs = sorted(expected_pairs - scored_pairs)

    # If all model-runs failed in this case, retry batch scoring first.
    if missing_pairs and not scored_pairs:
        print(
            f"[batch scorer] all model-runs missing for case={case_id}; retrying batch scoring",
            flush=True,
        )
        retry_limit = 2
        for retry_idx in range(1, retry_limit + 1):
            rescored = _score_batch_benchmark_case(
                llm_client=llm_client,
                task_text=task_text,
                runs_by_model=runs_by_model,
                scorer_model=scorer_model,
                case_id=case_id,
            )
            rescored_pairs: set[tuple[str, int]] = set()
            for model_name, runs in rescored.items():
                for run in runs:
                    rescored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

            rescored_missing = sorted(expected_pairs - rescored_pairs)
            if rescored_pairs:
                scored = rescored
                scored_pairs = rescored_pairs
                missing_pairs = rescored_missing
                print(
                    f"[batch scorer] retry {retry_idx}/{retry_limit} recovered "
                    f"{len(scored_pairs)}/{len(expected_pairs)} model-runs for case={case_id}",
                    flush=True,
                )
                break
    if missing_pairs:
        preview = ", ".join(f"{name}#run{idx}" for name, idx in missing_pairs[:6])
        if len(missing_pairs) > 6:
            preview = f"{preview}, ..."
        print(
            "[batch scorer] warning: incomplete batch output "
            f"(missing {len(missing_pairs)}/{len(expected_pairs)}): {preview}",
            flush=True,
        )

    return results_by_model


def _score_benchmark_batch_job(job: dict[str, Any]) -> dict[str, Any]:
    """Score all model runs for one case in one batch call."""
    task_file = Path(str(job.get("task_file", "")))
    case_id = str(job.get("case_id", ""))
    runs_by_model_raw = job.get("runs_by_model", {})
    scorer_model = str(job.get("scorer_model", BENCHMARK_SCORER_MODEL))

    task_text = task_file.read_text(encoding="utf-8")
    scorer_client = _get_benchmark_scorer_client(scorer_model)
    skills = _get_benchmark_skills()

    runs_by_model: dict[str, list[dict[str, Any]]] = {}
    run_meta: dict[tuple[str, int], dict[str, Any]] = {}
    for model_name, run_entries in runs_by_model_raw.items():
        runs_by_model[model_name] = []
        for entry in run_entries:
            report_file = Path(str(entry.get("report_file", "")))
            run_index = max(1, int(entry.get("run_index", 1) or 1))
            report_text = report_file.read_text(encoding="utf-8")

            runs_by_model[model_name].append({"run_index": run_index, "report_text": report_text})
            token_usage = _extract_token_usage_from_report_text(report_text)
            run_meta[(model_name, run_index)] = {
                "report_file": str(report_file),
                "elapsed_seconds": _extract_elapsed_seconds_from_report(report_text),
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
            }

    scored = _score_batch_benchmark_case(
        llm_client=scorer_client,
        task_text=task_text,
        runs_by_model=runs_by_model,
        scorer_model=scorer_model,
        case_id=case_id,
        skills=skills,
    )

    # Fallback: if batch JSON omits some model-run pairs, score them one-by-one.
    expected_pairs: set[tuple[str, int]] = set(run_meta.keys())
    scored_pairs: set[tuple[str, int]] = set()
    for model_name, runs in scored.items():
        for run in runs:
            scored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

    missing_pairs = sorted(expected_pairs - scored_pairs)
    if missing_pairs:
        print(
            f"[batch scorer] fallback to single-run scoring for case={case_id}, "
            f"missing={len(missing_pairs)}/{len(expected_pairs)}",
            flush=True,
        )

        runs_by_model_lookup: dict[tuple[str, int], str] = {}
        for model_name, run_entries in runs_by_model.items():
            for run_data in run_entries:
                run_index = max(1, int(run_data.get("run_index", 1) or 1))
                runs_by_model_lookup[(model_name, run_index)] = str(run_data.get("report_text", ""))

        # Build calibration references from successfully scored model-runs.
        reference_examples: list[dict[str, Any]] = []
        for ref_model, ref_runs in scored.items():
            for ref_run in ref_runs:
                ref_run_index = max(1, int(ref_run.get("run_index", 1) or 1))
                reference_examples.append(
                    {
                        "model_name": ref_model,
                        "run_index": ref_run_index,
                        "report_text": runs_by_model_lookup.get((ref_model, ref_run_index), ""),
                        "planning_completeness": ref_run.get("planning_completeness"),
                        "tool_reasonableness": ref_run.get("tool_reasonableness"),
                        "code_command_correctness": ref_run.get("code_command_correctness"),
                        "brief_justification": ref_run.get("brief_justification", ""),
                    }
                )

        for model_name, run_index in missing_pairs:
            report_text = runs_by_model_lookup.get((model_name, run_index), "")
            single = _score_single_benchmark_case(
                llm_client=scorer_client,
                task_text=task_text,
                report_text=report_text,
                model_name=model_name,
                scorer_model=scorer_model,
                reference_examples=reference_examples,
                skills=skills,
            )
            scored.setdefault(model_name, []).append(
                {
                    "run_index": run_index,
                    **single,
                }
            )
            reference_examples.append(
                {
                    "model_name": model_name,
                    "run_index": run_index,
                    "report_text": report_text,
                    "planning_completeness": single.get("planning_completeness"),
                    "tool_reasonableness": single.get("tool_reasonableness"),
                    "code_command_correctness": single.get("code_command_correctness"),
                    "brief_justification": single.get("brief_justification", ""),
                }
            )

    packed_results: list[dict[str, Any]] = []
    for model_name, runs in scored.items():
        for run in runs:
            run_index = max(1, int(run.get("run_index", 1) or 1))
            packed_results.append(
                {
                    "model_name": model_name,
                    "case_id": case_id,
                    "run_index": run_index,
                    "task_file": str(task_file),
                    **run_meta.get((model_name, run_index), {}),
                    **run,
                }
            )

    return {"case_id": case_id, "results": packed_results}


def _score_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    task_file = Path(str(job.get("task_file", "")))
    report_file = Path(str(job.get("report_file", "")))
    model_name = str(job.get("model_name", ""))
    case_id = str(job.get("case_id", ""))
    run_index = max(1, int(job.get("run_index", 1) or 1))
    scorer_model = str(job.get("scorer_model", BENCHMARK_SCORER_MODEL))

    task_text = task_file.read_text(encoding="utf-8")
    report_text = report_file.read_text(encoding="utf-8")
    elapsed_seconds = _extract_elapsed_seconds_from_report(report_text)
    token_usage = _extract_token_usage_from_report_text(report_text)
    scorer_client = _get_benchmark_scorer_client(scorer_model)
    skills = _get_benchmark_skills()
    case_score = _score_single_benchmark_case(
        llm_client=scorer_client,
        task_text=task_text,
        report_text=report_text,
        model_name=model_name,
        scorer_model=scorer_model,
        skills=skills,
    )
    return {
        "model_name": model_name,
        "case_id": case_id,
        "run_index": run_index,
        "task_file": str(task_file),
        "report_file": str(report_file),
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        **case_score,
    }


def _render_benchmark_leaderboard_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {}) if isinstance(results, dict) else {}
    ranking = results.get("ranking", []) if isinstance(results, dict) else []

    lines: list[str] = [
        "# Benchmark Leaderboard",
        "",
        f"- Timestamp: {meta.get('timestamp', '')}",
        f"- Scorer model: {meta.get('scorer_model', '')}",
        f"- Benchmark root: {meta.get('benchmark_root', '')}",
        f"- Output dir: {meta.get('output_dir', '')}",
        f"- Scored case count: {meta.get('scored_case_count', '')}",
        "",
        "## Ranking",
        "",
        "| Rank | Model | Average Score (%) | Avg Skill Usage | Avg Tokens | Avg Time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for idx, item in enumerate(ranking, start=1):
        model = str(item.get("model", ""))
        avg = item.get("average_weighted_score", "")
        avg_calls = item.get("average_usage_count", "-")
        avg_tokens = item.get("average_total_tokens", "-")
        avg_time = item.get("average_elapsed_seconds", "-")
        lines.append(f"| {idx} | {model} | {avg} | {avg_calls} | {avg_tokens} | {avg_time} |")

    if not ranking:
        lines.append("| - | (no complete models) | - | - | - | - |")

    gain = results.get("skill_gain_analysis") if isinstance(results, dict) else None
    comparisons = gain.get("comparisons", []) if isinstance(gain, dict) else []
    if comparisons:
        lines.extend([
            "",
            "## Skill Gain (With Skills vs No Skills)",
            "",
            "| Base Model | With Skills (%) | No Skills (%) | A_abs (%) | g | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for item in comparisons:
            lines.append(
                "| "
                f"{item.get('base_model', '')} | "
                f"{item.get('with_skills_average', '')} | "
                f"{item.get('no_skills_average', '')} | "
                f"{item.get('absolute_improvement', '')} | "
                f"{item.get('normalized_gain', '')} | "
                f"{item.get('interpretation', '')} |"
            )

    return "\n".join(lines) + "\n"


def _score_benchmark_reports(
    benchmark_root: Path,
    output_dir: Path,
    score_workers: int = 8,
    scorer_model: str = BENCHMARK_SCORER_MODEL,
    compare_base_model: str | None = None,
) -> tuple[Path, Path]:
    if not benchmark_root.exists() or not benchmark_root.is_dir():
        raise RuntimeError(f"Benchmark directory not found: {benchmark_root}")
    if not output_dir.exists() or not output_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_dir}")

    task_contracts = _discover_task_contracts(benchmark_root)
    expected_case_ids = sorted(task_contracts.keys())
    if not expected_case_ids:
        raise RuntimeError("No benchmark task.md files discovered.")

    report_index = _discover_benchmark_reports(output_dir)
    if not report_index:
        raise RuntimeError("No benchmark report files found in output directory.")

    print(f"Expected benchmark cases: {len(expected_case_ids)}")
    print(f"Discovered benchmark report models in output: {len(report_index)}")

    complete_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, case_map in sorted(report_index.items()):
        if case_map:
            complete_models[model_name] = case_map

    if not complete_models:
        raise RuntimeError("No models with benchmark reports found.")

    case_sets = [set(case_map.keys()) for case_map in complete_models.values()]
    common_case_ids = sorted(set.intersection(*case_sets) if case_sets else set())
    common_case_ids = [case_id for case_id in common_case_ids if case_id in task_contracts]
    if not common_case_ids:
        raise RuntimeError("No shared case ids found across benchmark reports.")

    dropped_models: list[dict[str, Any]] = []
    kept_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    comparable_runs_per_model: dict[str, int] = {}
    for model_name, case_map in sorted(complete_models.items()):
        case_ids = set(case_map.keys())
        missing_shared = sorted(set(common_case_ids) - case_ids)
        extra = sorted(case_ids - set(common_case_ids))
        run_counts = {case_id: len(case_map.get(case_id, [])) for case_id in common_case_ids}
        comparable_run_count = min(run_counts.values()) if run_counts else 0
        if missing_shared or comparable_run_count <= 0:
            dropped_models.append(
                {
                    "model": model_name,
                    "reason": "missing_shared_case_ids" if missing_shared else "no_runs_for_shared_cases",
                    "found_count": sum(len(case_map.get(case_id, [])) for case_id in common_case_ids),
                    "expected_count": len(common_case_ids),
                    "missing_case_ids": missing_shared,
                    "run_counts": run_counts,
                    "comparable_runs_per_case": comparable_run_count,
                    "extra_case_ids": extra,
                }
            )
            continue
        kept_models[model_name] = case_map
        comparable_runs_per_model[model_name] = comparable_run_count

    complete_models = kept_models
    if not complete_models:
        raise RuntimeError("No models with benchmark reports covering the shared cases found.")

    required_runs_per_case = min(comparable_runs_per_model.values()) if comparable_runs_per_model else 0
    if required_runs_per_case <= 0:
        raise RuntimeError("Unable to infer a comparable benchmark run count from the discovered reports.")

    # If caller requested comparison for a specific base model, restrict to
    # that base model's with_skills and no_skills variants only.
    if compare_base_model:
        want_with = f"{compare_base_model}{BENCHMARK_WITH_SKILLS_SUFFIX}"
        want_no = f"{compare_base_model}{BENCHMARK_NO_SKILLS_SUFFIX}"
        filtered = {k: v for k, v in complete_models.items() if k in {want_with, want_no}}
        if not filtered:
            raise RuntimeError(f"No reports found for specified base model variants: {compare_base_model}")
        complete_models = filtered

    print(f"Comparable models kept for scoring: {len(complete_models)}")
    print(f"Shared case ids selected for scoring: {len(common_case_ids)}")
    print(f"Comparable runs per case used for scoring: {required_runs_per_case}")

    workers = max(1, int(score_workers or 1))
    print(f"Score workers: {workers}")
    resolved_scorer_model = str(scorer_model or "").strip() or BENCHMARK_SCORER_MODEL
    print(f"Scorer model: {resolved_scorer_model}")

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "scorer_model": resolved_scorer_model,
            "scoring_mode": "per_task_joint_scoring_all_models",
            "fairness_policy": (
                "For each task case, all participating models are scored in a single batch call "
                "to minimize rubric drift across models."
            ),
            "score_scale": "0-100",
            "llm_score_components": [
                "planning_completeness",
                "tool_reasonableness",
                "code_command_correctness",
            ],
            "usage_efficiency_note": "Computed separately from skill usage counts within the same case across models; not included in the LLM weighted score.",
            "weights": BENCHMARK_SCORE_WEIGHTS,
            "benchmark_root": str(benchmark_root),
            "output_dir": str(output_dir),
            "runs_per_case": required_runs_per_case,
            "comparable_models": sorted(complete_models.keys()),
            "comparable_model_count": len(complete_models),
            "expected_case_ids": expected_case_ids,
            "scored_case_ids": common_case_ids,
            "scored_case_count": len(common_case_ids),
        },
        "dropped_models": dropped_models,
        "models": {},
        "ranking": [],
        "skill_gain_analysis": {
            "note": "Interpreting Normalized Gain: report both absolute improvement (A_abs) and normalized gain (g). Similar g means similar proportional benefit, not identical absolute gains.",
            "comparisons": [],
        },
    }

    jobs: list[dict[str, Any]] = []
    for case_id in common_case_ids:
        runs_by_model: dict[str, list[dict[str, Any]]] = {}
        for model_name in sorted(complete_models.keys()):
            case_map = complete_models[model_name]
            run_entries = case_map[case_id][:required_runs_per_case]
            runs_by_model[model_name] = [
                {
                    "run_index": int(entry.get("run_index", 1) or 1),
                    "report_file": str(entry["report_file"]),
                }
                for entry in run_entries
            ]
        jobs.append(
            {
                "task_file": str(task_contracts[case_id]),
                "case_id": case_id,
                "runs_by_model": runs_by_model,
                "scorer_model": resolved_scorer_model,
            }
        )

    total_jobs = len(jobs)
    print(f"Total scoring jobs: {total_jobs}", flush=True)

    print(
        "Note: Fair scoring mode enabled - all comparable models are jointly scored per task case",
        flush=True,
    )

    score_start_time = time.time()
    per_case_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_case_id: dict[concurrent.futures.Future[Any], str] = {}
        for job in jobs:
            case_id = str(job.get("case_id", ""))
            future = executor.submit(
                _retry_api_call,
                f"Benchmark scoring job {case_id}",
                lambda job=job: _score_benchmark_batch_job(job),
                10,
            )
            future_to_case_id[future] = case_id

        for done_jobs, future in enumerate(concurrent.futures.as_completed(future_to_case_id), start=1):
            elapsed = time.time() - score_start_time
            avg_per_job = elapsed / done_jobs if done_jobs > 0 else 0.0
            remaining_jobs = max(0, total_jobs - done_jobs)
            eta_seconds = avg_per_job * remaining_jobs

            case_id = future_to_case_id.get(future, f"case-{done_jobs}")
            try:
                batch_result = future.result()
            except Exception as exc:
                print(
                    f"[score job {done_jobs}/{total_jobs}] case={case_id} ERROR: batch scoring failed: {exc} "
                    f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                    flush=True,
                )
                continue

            case_id = str(batch_result.get("case_id", case_id))
            results_list = batch_result.get("results", [])
            unique_result_pairs = {
                (
                    str(item.get("model_name", "")),
                    max(1, int(item.get("run_index", 1) or 1)),
                )
                for item in results_list
                if isinstance(item, dict)
            }
            unique_count = len(unique_result_pairs)
            for result in results_list:
                model_name = str(result.get("model_name", ""))
                run_index = max(1, int(result.get("run_index", 1) or 1))
                result["run_index"] = run_index
                per_case_results.setdefault((model_name, case_id), []).append(result)

            print(
                f"Scoring [{done_jobs}/{total_jobs}] case={case_id} "
                f"(batch scored unique_count={unique_count} model-runs, "
                f"elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                flush=True,
            )

    for model_name, case_map in sorted(complete_models.items()):
        per_case: dict[str, Any] = {}
        for case_id in common_case_ids:
            run_results = sorted(
                per_case_results.get((model_name, case_id), []),
                key=lambda item: int(item.get("run_index", 1) or 1),
            )
            if len(run_results) < required_runs_per_case:
                raise RuntimeError(f"Missing scoring results for model={model_name} case={case_id}")
            run_results = run_results[:required_runs_per_case]
            weighted_distribution: list[float] = []
            usage_distribution: list[float] = []
            elapsed_distribution: list[float] = []
            prompt_distribution: list[float] = []
            completion_distribution: list[float] = []
            total_distribution: list[float] = []
            run_records: list[dict[str, Any]] = []
            for result in run_results:
                planning = _clamp_score_1_10(result.get("planning_completeness", 1))
                tool = _clamp_score_1_10(result.get("tool_reasonableness", 1))
                code = _clamp_score_1_10(result.get("code_command_correctness", 1))
                weighted_10 = round(
                    planning * BENCHMARK_SCORE_WEIGHTS["planning_completeness"]
                    + tool * BENCHMARK_SCORE_WEIGHTS["tool_reasonableness"]
                    + code * BENCHMARK_SCORE_WEIGHTS["code_command_correctness"],
                    3,
                )
                weighted_score = round(weighted_10 * 10.0, 2)
                weighted_distribution.append(weighted_score)
                usage_count = result.get("usage_count")
                if isinstance(usage_count, int):
                    usage_distribution.append(float(usage_count))
                if isinstance(result.get("elapsed_seconds"), (int, float)):
                    elapsed_distribution.append(float(result["elapsed_seconds"]))
                if isinstance(result.get("prompt_tokens"), int):
                    prompt_distribution.append(float(result["prompt_tokens"]))
                if isinstance(result.get("completion_tokens"), int):
                    completion_distribution.append(float(result["completion_tokens"]))
                if isinstance(result.get("total_tokens"), int):
                    total_distribution.append(float(result["total_tokens"]))
                run_records.append(
                    {
                        "run_index": int(result.get("run_index", 1) or 1),
                        "task_file": result["task_file"],
                        "report_file": result["report_file"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["total_tokens"],
                        "planning_completeness": result["planning_completeness"],
                        "tool_reasonableness": result["tool_reasonableness"],
                        "code_command_correctness": result["code_command_correctness"],
                        "usage_count": result.get("usage_count"),
                        "brief_justification": result["brief_justification"],
                        "raw_model_output": result["raw_model_output"],
                        "weighted_score_10": weighted_10,
                        "weighted_score": weighted_score,
                    }
                )

            weighted_mean, weighted_variance = _mean_and_variance(weighted_distribution)
            usage_mean, usage_variance = _mean_and_variance(usage_distribution)
            elapsed_mean, elapsed_variance = _mean_and_variance(elapsed_distribution)
            prompt_mean, prompt_variance = _mean_and_variance(prompt_distribution)
            completion_mean, completion_variance = _mean_and_variance(completion_distribution)
            total_mean, total_variance = _mean_and_variance(total_distribution)
            per_case[case_id] = {
                "task_file": run_records[0]["task_file"],
                "runs": run_records,
                "run_count": len(run_records),
                "score_distribution": weighted_distribution,
                "average_weighted_score": weighted_mean,
                "variance_weighted_score": weighted_variance,
                "average_usage_count": usage_mean,
                "variance_usage_count": usage_variance,
                "average_elapsed_seconds": elapsed_mean,
                "variance_elapsed_seconds": elapsed_variance,
                "average_prompt_tokens": prompt_mean,
                "variance_prompt_tokens": prompt_variance,
                "average_completion_tokens": completion_mean,
                "variance_completion_tokens": completion_variance,
                "average_total_tokens": total_mean,
                "variance_total_tokens": total_variance,
            }
        results["models"][model_name] = {
            "case_count": len(common_case_ids),
            "average_weighted_score": 0.0,
            "variance_weighted_score": None,
            "average_usage_count": None,
            "variance_usage_count": None,
            "average_elapsed_seconds": None,
            "variance_elapsed_seconds": None,
            "average_prompt_tokens": None,
            "variance_prompt_tokens": None,
            "average_completion_tokens": None,
            "variance_completion_tokens": None,
            "average_total_tokens": None,
            "variance_total_tokens": None,
            "cases": per_case,
        }

    for data in results["models"].values():
        score_values: list[float] = []
        usage_values: list[float] = []
        elapsed_values: list[float] = []
        prompt_values: list[float] = []
        completion_values: list[float] = []
        total_values: list[float] = []
        for case_id in common_case_ids:
            case_data = data.get("cases", {}).get(case_id, {})
            if isinstance(case_data.get("average_weighted_score"), (int, float)):
                score_values.append(float(case_data["average_weighted_score"]))
            usage_value = case_data.get("average_usage_count")
            if isinstance(usage_value, (int, float)):
                usage_values.append(float(usage_value))
            if isinstance(case_data.get("average_elapsed_seconds"), (int, float)):
                elapsed_values.append(float(case_data["average_elapsed_seconds"]))
            if isinstance(case_data.get("average_prompt_tokens"), (int, float)):
                prompt_values.append(float(case_data["average_prompt_tokens"]))
            if isinstance(case_data.get("average_completion_tokens"), (int, float)):
                completion_values.append(float(case_data["average_completion_tokens"]))
            if isinstance(case_data.get("average_total_tokens"), (int, float)):
                total_values.append(float(case_data["average_total_tokens"]))

        score_mean, score_variance = _mean_and_variance(score_values)
        usage_mean, usage_variance = _mean_and_variance(usage_values)
        elapsed_mean, elapsed_variance = _mean_and_variance(elapsed_values)
        prompt_mean, prompt_variance = _mean_and_variance(prompt_values)
        completion_mean, completion_variance = _mean_and_variance(completion_values)
        total_mean, total_variance = _mean_and_variance(total_values)
        data["average_weighted_score"] = round(score_mean, 2) if score_mean is not None else None
        data["variance_weighted_score"] = score_variance
        data["average_usage_count"] = round(usage_mean, 2) if usage_mean is not None else None
        data["variance_usage_count"] = usage_variance
        data["average_elapsed_seconds"] = round(elapsed_mean, 3) if elapsed_mean is not None else None
        data["variance_elapsed_seconds"] = elapsed_variance
        data["average_prompt_tokens"] = round(prompt_mean, 2) if prompt_mean is not None else None
        data["variance_prompt_tokens"] = prompt_variance
        data["average_completion_tokens"] = round(completion_mean, 2) if completion_mean is not None else None
        data["variance_completion_tokens"] = completion_variance
        data["average_total_tokens"] = round(total_mean, 2) if total_mean is not None else None
        data["variance_total_tokens"] = total_variance

    ranking = sorted(
        (
            {
                "model": model,
                "average_weighted_score": data["average_weighted_score"],
                "average_usage_count": data.get("average_usage_count"),
                "average_total_tokens": data.get("average_total_tokens"),
                "average_elapsed_seconds": data.get("average_elapsed_seconds"),
            }
            for model, data in results["models"].items()
        ),
        key=lambda x: (
            -float(x.get("average_weighted_score", 0.0) or 0.0),
            float(x.get("average_usage_count", 1e9) or 1e9),
            float(x.get("average_total_tokens", 1e18) or 1e18),
            float(x.get("average_elapsed_seconds", 1e18) or 1e18),
        ),
    )
    results["ranking"] = ranking

    condition_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, data in results["models"].items():
        base_model, cond = _split_skill_condition(model_name)
        condition_groups.setdefault(base_model, {})[cond] = data

    comparisons: list[dict[str, Any]] = []
    for base_model, cond_map in sorted(condition_groups.items()):
        with_data = cond_map.get("with_skills")
        no_data = cond_map.get("no_skills")
        if not with_data or not no_data:
            continue

        with_avg = float(with_data.get("average_weighted_score", 0.0) or 0.0)
        no_avg = float(no_data.get("average_weighted_score", 0.0) or 0.0)
        abs_improvement = round(with_avg - no_avg, 2)
        norm_gain = _compute_normalized_gain(with_avg, no_avg)

        per_case: list[dict[str, Any]] = []
        for case_id in common_case_ids:
            with_case = with_data.get("cases", {}).get(case_id, {})
            no_case = no_data.get("cases", {}).get(case_id, {})
            with_case_score = float(with_case.get("average_weighted_score", 0.0) or 0.0)
            no_case_score = float(no_case.get("average_weighted_score", 0.0) or 0.0)
            case_abs = round(with_case_score - no_case_score, 2)
            case_g = _compute_normalized_gain(with_case_score, no_case_score)
            per_case.append(
                {
                    "case_id": case_id,
                    "with_skills_score": with_case_score,
                    "no_skills_score": no_case_score,
                    "with_skills_variance": with_case.get("variance_weighted_score"),
                    "no_skills_variance": no_case.get("variance_weighted_score"),
                    "absolute_improvement": case_abs,
                    "normalized_gain": case_g,
                }
            )

        comparisons.append(
            {
                "base_model": base_model,
                "with_skills_model": f"{base_model}{BENCHMARK_WITH_SKILLS_SUFFIX}",
                "no_skills_model": f"{base_model}{BENCHMARK_NO_SKILLS_SUFFIX}",
                "with_skills_average": round(with_avg, 2),
                "no_skills_average": round(no_avg, 2),
                "absolute_improvement": abs_improvement,
                "normalized_gain": norm_gain,
                "interpretation": _interpret_normalized_gain(abs_improvement, norm_gain),
                "per_case": per_case,
            }
        )

    results["skill_gain_analysis"]["comparisons"] = comparisons

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    scorer_slug = _slugify_filename_part(resolved_scorer_model, "scorer", max_len=80)
    out_file = output_dir / f"benchmark_scores_{scorer_slug}_{timestamp}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    leaderboard_file = output_dir / f"benchmark_leaderboard_{scorer_slug}_{timestamp}.md"
    leaderboard_file.write_text(_render_benchmark_leaderboard_markdown(results), encoding="utf-8")
    return out_file, leaderboard_file


def _get_benchmark_skills() -> list[dict[str, Any]]:
    global _BENCHMARK_SKILLS_CACHE
    if _BENCHMARK_SKILLS_CACHE is not None:
        return _BENCHMARK_SKILLS_CACHE

    SkillLoader = _load_skill_loader_class()
    loader = SkillLoader(REPO_ROOT / "skills")
    _BENCHMARK_SKILLS_CACHE = loader.load_all()
    return _BENCHMARK_SKILLS_CACHE


def _write_benchmark_failure_report(
    *,
    task_text: str,
    task_prompt: str,
    task_label: str,
    variant_label: str,
    run_model_name: str,
    run_index: int = 1,
    fallback_source: str,
    failure_message: str,
) -> Path:
    model_dir = _slugify_filename_part(run_model_name, "model_unknown", max_len=120)
    output_dir = REPO_ROOT / "output" / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = _benchmark_report_filename(
        task_text,
        run_model_name,
        run_index=max(1, int(run_index or 1)),
        fallback_source=fallback_source,
    )
    report_path = output_dir / f"{report_stem}.md"
    report_path.write_text(
        "\n".join([
            f"# Benchmark Report: {report_stem}",
            "",
            f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
            "- Mode: benchmark",
            f"- Benchmark variant: {variant_label}",
            f"- Case ID: {_extract_case_id(task_text, fallback_source=fallback_source)}",
            f"- Model: {run_model_name}",
            f"- Task: {task_prompt.splitlines()[0].strip() if task_prompt.splitlines() else task_label}",
            "- Elapsed seconds: 0.0",
            "- Token usage: prompt=0, completion=0, total=0",
            "- Skills used (unique): 0",
            "- Skill names used: []",
            "- Skill files read via tool (unique): 0",
            "- Skill files read via tool: []",
            "",
            "## Solution Thinking",
            failure_message,
            "",
            "## Commands Or Code",
            "1. No tool command was used.",
            "",
        ]),
        encoding="utf-8",
    )
    return report_path


def _run_single_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    idx = int(job.get("idx", 0))
    total = int(job.get("total", 0))
    task_file = Path(str(job.get("task_file", "")))
    task_label = str(job.get("task_label", task_file.parent.name))
    variant_label = str(job.get("variant_label", "standard"))
    no_skill_mode = bool(job.get("no_skill_mode", False))
    provider = str(job.get("provider", "openai"))
    model_name = str(job.get("model_name", ""))
    run_model_name = str(job.get("run_model_name", model_name))
    run_index = max(1, int(job.get("run_index", 1) or 1))

    try:
        task_text = task_file.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "ok": False,
            "idx": idx,
            "total": total,
            "task_label": task_label,
            "variant_label": variant_label,
            "error": f"ERROR reading {task_file.name}: {exc}",
            "report_path": None,
        }

    task_prompt = _extract_task_summary_for_benchmark(task_text)
    session = AgentSession(benchmark_mode=True, no_skill_mode=no_skill_mode)
    session.env.setdefault("llm_backend", {})["provider"] = provider
    session.env.setdefault("llm_backend", {})["model"] = model_name
    session._benchmark_report_model = run_model_name
    session._benchmark_report_run_index = run_index

    try:
        session.set_llm_client(build_llm_client(session.env))
    except Exception as exc:
        report_path = _write_benchmark_failure_report(
            task_text=task_text,
            task_prompt=task_prompt,
            task_label=task_label,
            variant_label=variant_label,
            run_model_name=run_model_name,
            run_index=run_index,
            fallback_source=task_file.parent.name,
            failure_message=f"Model initialization failed: {exc}",
        )
        return {
            "ok": False,
            "idx": idx,
            "total": total,
            "task_label": task_label,
            "variant_label": variant_label,
            "error": f"ERROR initializing model [{variant_label}]: {exc}",
            "report_path": str(report_path),
        }

    try:
        skills = _get_benchmark_skills()
        session.skills = skills
        system_prompt = session._build_system_prompt(skills)
        session.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        _ = session._chat()
        report_stem = _benchmark_report_filename(
            task_text,
            run_model_name,
            run_index=run_index,
            fallback_source=task_file.parent.name,
        )
        model_dir = _slugify_filename_part(run_model_name, "model_unknown", max_len=120)
        report_path = REPO_ROOT / "output" / model_dir / f"{report_stem}.md"
        return {
            "ok": True,
            "idx": idx,
            "total": total,
            "task_label": task_label,
            "variant_label": variant_label,
            "run_index": run_index,
            "error": "",
            "report_path": str(report_path),
        }
    except Exception as exc:
        report_path = _write_benchmark_failure_report(
            task_text=task_text,
            task_prompt=task_prompt,
            task_label=task_label,
            variant_label=variant_label,
            run_model_name=run_model_name,
            run_index=run_index,
            fallback_source=task_file.parent.name,
            failure_message=f"Benchmark task execution failed: {exc}",
        )
        return {
            "ok": False,
            "idx": idx,
            "total": total,
            "task_label": task_label,
            "variant_label": variant_label,
            "run_index": run_index,
            "error": f"ERROR [{variant_label}]: {exc}",
            "report_path": str(report_path),
        }


def _run_benchmark_suite(
    benchmark_root: Path,
    benchmark_provider: str,
    model_name: str,
    compare_skills: bool = False,
    benchmark_workers: int = 8,
    benchmark_repeats: int = 3,
) -> None:
    if not benchmark_root.exists() or not benchmark_root.is_dir():
        raise RuntimeError(f"Benchmark directory not found: {benchmark_root}")

    task_files = _discover_benchmark_task_files(benchmark_root)
    if not task_files:
        raise RuntimeError(f"No task.md files found under {benchmark_root}")

    total = len(task_files)
    workers = max(1, int(benchmark_workers or 1))
    repeats = max(1, int(benchmark_repeats or 1))
    print(f"Benchmark root: {benchmark_root}")
    print(f"Benchmark provider: {benchmark_provider}")
    print(f"Benchmark model: {model_name}")
    if compare_skills:
        print("Benchmark comparison mode: with-skills vs no-skills")
    print(f"Benchmark workers: {workers}")
    print(f"Benchmark repeats per case: {repeats}")
    print(f"Tasks discovered: {total}")
    print("Starting benchmark run...\n")

    variants: list[tuple[str, bool, str]] = [("standard", False, "")]
    if compare_skills:
        variants = [
            ("with-skills", False, BENCHMARK_WITH_SKILLS_SUFFIX),
            ("no-skills", True, BENCHMARK_NO_SKILLS_SUFFIX),
        ]

    jobs: list[dict[str, Any]] = []
    for idx, task_file in enumerate(task_files, start=1):
        task_label = task_file.parent.name
        for variant_label, no_skill_mode, variant_suffix in variants:
            run_model_name = f"{model_name}{variant_suffix}"
            for run_index in range(1, repeats + 1):
                jobs.append(
                    {
                        "idx": idx,
                        "total": total,
                        "task_file": str(task_file),
                        "task_label": task_label,
                        "variant_label": variant_label,
                        "no_skill_mode": no_skill_mode,
                        "provider": benchmark_provider,
                        "model_name": model_name,
                        "run_model_name": run_model_name,
                        "run_index": run_index,
                    }
                )

    print(f"Total benchmark jobs: {len(jobs)}", flush=True)

    benchmark_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_single_benchmark_job, job) for job in jobs]
        for done_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            elapsed = time.time() - benchmark_start_time
            avg_per_job = elapsed / done_idx if done_idx > 0 else 0.0
            remaining_jobs = max(0, len(jobs) - done_idx)
            eta_seconds = avg_per_job * remaining_jobs
            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[job {done_idx}/{len(jobs)}] ERROR: worker crashed: {exc} "
                    f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                    flush=True,
                )
                continue

            prefix = f"[{result.get('idx', '?')}/{result.get('total', '?')}] [{result.get('variant_label', 'standard')}]"
            report_path_raw = str(result.get("report_path") or "").strip()
            report_rel = ""
            if report_path_raw:
                try:
                    report_rel = str(Path(report_path_raw).resolve().relative_to(REPO_ROOT))
                except Exception:
                    report_rel = report_path_raw

            if bool(result.get("ok", False)):
                print(
                    f"{prefix} Saved {report_rel} "
                    f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                    flush=True,
                )
            else:
                error_text = str(result.get("error", "unknown error"))
                if report_rel:
                    print(
                        f"{prefix} {error_text} (saved {report_rel}) "
                        f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                        flush=True,
                    )
                else:
                    print(
                        f"{prefix} {error_text} "
                        f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                        flush=True,
                    )

    print(f"\nBenchmark run complete. Reports saved under {REPO_ROOT / 'output'}")


# ── Agent session ──────────────────────────────────────────────────────────────

class AgentSession:
    """
    Minimal NeuroClaw agent session.

    Responsibilities:
    - Bootstrap environment (load_environment)
    - Load skills via SkillLoader
    - Maintain conversation history
    - Route tool calls to ToolRuntime
    - Stream responses from the LLM backend
    """

    def __init__(
        self,
        workspace: Path | None = None,
        benchmark_mode: bool | None = None,
        no_skill_mode: bool = False,
    ) -> None:
        self.workspace = workspace or REPO_ROOT
        self.env = load_environment()
        self.history: list[dict] = []
        self._llm: Any = None
        self.benchmark_mode = (
            _is_benchmark_enabled_from_env()
            if benchmark_mode is None
            else bool(benchmark_mode)
        )
        self._tool_events: list[dict[str, Any]] = []
        self.skills: list[dict[str, Any]] = []
        self.no_skill_mode = bool(no_skill_mode)
        self._benchmark_report_model: str | None = None
        self._last_chat_elapsed_sec: float = 0.0
        self._last_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ── Public API for external callers (e.g. the web server) ──────────────────

    def set_llm_client(self, client: Any) -> None:
        """
        Attach an already-constructed LLM client to this session.

        Prefer calling this over accessing ``self._llm`` directly so that
        the internal field can be renamed without breaking callers.
        """
        self._llm = client

    def start(self) -> None:
        """Interactive REPL — called by main.py."""
        if not ENV_FILE.exists():
            self._prompt_setup()
            return

        import importlib.util as _ilu

        # SkillLoader lives in core/skill-loader/ (hyphen) so we must use
        # importlib rather than a regular package import.
        _loader_mod = _ilu.spec_from_file_location(
            "neuroclaw_skill_loader",
            REPO_ROOT / "core" / "skill-loader" / "loader.py",
        )
        if _loader_mod is None or _loader_mod.loader is None:
            raise RuntimeError("Cannot find core/skill-loader/loader.py")
        _m = __import__("importlib").util.module_from_spec(_loader_mod)
        _loader_mod.loader.exec_module(_m)
        SkillLoader = _m.SkillLoader

        from core.session.manager import SessionManager  # type: ignore

        loader = SkillLoader(self.workspace / "skills")
        skills = loader.load_all()
        self.skills = skills

        manager = SessionManager(env=self.env)
        self._llm = build_llm_client(self.env)

        system_prompt = self._build_system_prompt(skills)
        self.history = [{"role": "system", "content": system_prompt}]

        print("NeuroClaw ready. Type your message (Ctrl-C to exit).\n")
        if self.benchmark_mode:
            print("Benchmark mode is ON: file input/output tasks are simulated; fast no-file tasks can run.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
            if not user_input:
                continue

            self.history.append({"role": "user", "content": user_input})
            response = self._chat()
            print(f"\nNeuroClaw: {response}\n")
            self.history.append({"role": "assistant", "content": response})
            manager.maybe_compress(self.history)

    def _chat(self) -> str:
        """Send history to LLM and return response text (simplified, no streaming)."""
        provider = self.env.get("llm_backend", {}).get("provider", "openai")
        model = self.env.get("llm_backend", {}).get("model", "gpt-4o")
        self._tool_events = []
        self._last_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        chat_started = time.perf_counter()

        if provider in {"openai", "anthropic"} and self._llm is None:
            return "[Agent: LLM backend not configured]"

        response = "[Agent: LLM backend not configured]"
        if provider == "openai":
            response = self._chat_openai_with_tools(model)
        elif provider == "anthropic":
            system_msg = next(
                (m["content"] for m in self.history if m["role"] == "system"), ""
            )
            user_msgs = [m for m in self.history if m["role"] != "system"]
            resp = _retry_api_call(
                "Anthropic chat request",
                lambda: self._llm.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_msg,
                    messages=user_msgs,
                ),
                retries=10,
            )
            response = resp.content[0].text if resp.content else ""
        elif provider == "local":
            import urllib.request  # stdlib only

            endpoint = self._llm["endpoint"]
            payload = json.dumps(
                {"model": self._llm["model"], "messages": self.history, "stream": False}
            ).encode()
            req = urllib.request.Request(
                f"{endpoint}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            response = data.get("message", {}).get("content", "")

        self._last_chat_elapsed_sec = round(max(0.0, time.perf_counter() - chat_started), 3)

        if self.benchmark_mode and self._should_write_benchmark_report():
            try:
                self._write_benchmark_report(response)
            except Exception as exc:
                print(f"[benchmark] failed to write report: {exc}", flush=True)

        return response

    def _chat_openai_with_tools(self, model: str) -> str:
        """OpenAI chat with a minimal native shell tool-call loop."""
        if self.no_skill_mode:
            def _request_once(req_messages: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
                resp = _retry_api_call(
                    "OpenAI chat request",
                    lambda: self._llm.chat.completions.create(
                        **_get_openai_chat_create_kwargs(
                            self.env,
                            model,
                            req_messages,
                        )
                    ),
                    retries=10,
                )
                usage = _extract_token_usage_from_response(resp)
                try:
                    content = resp.choices[0].message.content or ""
                except Exception:
                    content = ""
                return content, usage

            response_text, usage_totals = _request_once(list(self.history))
            if self.benchmark_mode and _is_incomplete_benchmark_response(response_text):
                retry_text, retry_usage = _request_once(
                    list(self.history)
                    + [{"role": "assistant", "content": response_text}]
                    + [{
                        "role": "user",
                        "content": (
                            "Your previous benchmark answer was incomplete. Do not emit tool-call markup, tool results, inspection intent, or placeholders. "
                            "Finish the task now in the required final format with exactly two top-level sections: ## Solution Thinking and ## Commands Or Code. "
                            "If this is a with-skills run, include the required contract-audit lines and then provide task-complete executable commands or code."
                        ),
                    }]
                )
                usage_totals = {
                    "prompt_tokens": int(usage_totals.get("prompt_tokens", 0)) + int(retry_usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage_totals.get("completion_tokens", 0)) + int(retry_usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage_totals.get("total_tokens", 0)) + int(retry_usage.get("total_tokens", 0)),
                }
                if retry_text.strip():
                    response_text = retry_text

            self._last_token_usage = usage_totals
            return response_text

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": (
                        "Run a shell command in the local workspace using default shell "
                        "and inherited environment variables."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Exact shell command to execute.",
                            },
                            "timeout_sec": {
                                "type": "integer",
                                "description": "Timeout in seconds (default 180).",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_workspace_file",
                    "description": (
                        "Read a UTF-8 text file inside the current workspace, such as a SKILL.md, script, or config file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative or absolute path to a text file inside the workspace.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return (default 12000).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

        messages: list[dict[str, Any]] = list(self.history)
        token_usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for _ in range(4):
            resp = _retry_api_call(
                "OpenAI tool-call request",
                lambda: self._llm.chat.completions.create(
                    **_get_openai_chat_create_kwargs(
                        self.env,
                        model,
                        messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                ),
                retries=10,
            )
            usage = _extract_token_usage_from_response(resp)
            token_usage_totals["prompt_tokens"] += usage["prompt_tokens"]
            token_usage_totals["completion_tokens"] += usage["completion_tokens"]
            token_usage_totals["total_tokens"] += usage["total_tokens"]
            self._last_token_usage = dict(token_usage_totals)
            message = resp.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", []) or [])

            if not tool_calls:
                return message.content or ""

            def _tool_call_to_message_dict(tc: Any) -> dict[str, Any]:
                """Best-effort tool_call serialization.

                Some OpenAI-compatible proxy routes (notably Gemini) attach extra
                metadata to tool calls (e.g., thought signatures). Rebuilding
                tool_calls from only id/name/arguments can drop required fields.
                """
                # Prefer model-provided serialization if available.
                for attr in ("model_dump", "to_dict", "dict"):
                    fn = getattr(tc, attr, None)
                    if callable(fn):
                        try:
                            data = fn()
                            if isinstance(data, dict):
                                return data
                        except Exception:
                            pass

                # Fallback to the minimal OpenAI-compatible structure.
                return {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function") or "function",
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                    },
                }

            assistant_tool_msg = {
                "role": "assistant",
                "content": message.content or "",
                # Preserve extra provider metadata for Gemini routes.
                "tool_calls": [
                    _tool_call_to_message_dict(tc)
                    if _model_uses_gemini_compat(model)
                    else {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }

            if _model_uses_gemini_compat(model) and os.environ.get("NEUROCLAW_DEBUG_GEMINI_TOOLCALLS") == "1":
                try:
                    debug_path = (REPO_ROOT / "output" / "debug_gemini_toolcalls.jsonl")
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    with debug_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "model": model,
                            "assistant_tool_msg": assistant_tool_msg,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            messages.append(assistant_tool_msg)

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                if name == "run_shell_command":
                    shell_cmd = str(args.get("command", ""))
                    timeout_sec = int(args.get("timeout_sec", 180))

                    if self.benchmark_mode and _looks_file_io_shell_command(shell_cmd):
                        result = {
                            "success": True,
                            "benchmark_mode": True,
                            "executed": False,
                            "message": (
                                "Benchmark mode skipped real execution for file/dataset I/O task. "
                                "Return command/code only."
                            ),
                            "suggested_command": shell_cmd,
                        }
                    else:
                        result = _run_shell_command(
                            command=shell_cmd,
                            cwd=self.workspace,
                            timeout_sec=timeout_sec,
                        )
                        if self.benchmark_mode:
                            result["benchmark_mode"] = True
                            result["executed"] = True

                    self._tool_events.append(
                        {
                            "tool": "run_shell_command",
                            "command": shell_cmd,
                            "executed": bool(result.get("executed", result.get("success", False))),
                            "success": bool(result.get("success", False)),
                            "skills_used": _extract_skills_from_result_payload(result),
                            "result": result,
                        }
                    )
                elif name == "read_workspace_file":
                    read_path = str(args.get("path", ""))
                    max_chars = int(args.get("max_chars", 12000))
                    result = _read_workspace_file(read_path, self.workspace, max_chars=max_chars)
                    self._tool_events.append(
                        {
                            "tool": "read_workspace_file",
                            "command": read_path,
                            "executed": bool(result.get("success", False)),
                            "success": bool(result.get("success", False)),
                            "skills_used": _extract_skills_from_result_payload(result),
                            "result": result,
                        }
                    )
                else:
                    result = {"success": False, "error": f"unknown tool: {name}"}
                    self._tool_events.append(
                        {
                            "tool": name,
                            "executed": False,
                            "success": False,
                            "skills_used": _extract_skills_from_result_payload(result),
                            "result": result,
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        force_benchmark_finalization = (
            self.benchmark_mode
            and _tool_events_show_inspection_only(self._tool_events)
            and (
                _model_uses_deepseek_compat(self.env, model)
                or (
                    _model_uses_gemini_compat(model)
                    and _model_is_withskills_variant(model)
                )
            )
        )
        if force_benchmark_finalization:
            forced_messages = list(messages) + [{
                "role": "user",
                "content": (
                    "You have already inspected the relevant skill files. Do not inspect any more skills, directories, or scripts. "
                    "Do not emit tool calls. Finish the benchmark task now in the required final format with exactly two top-level sections: "
                    "## Solution Thinking and ## Commands Or Code. If this is a with-skills run, keep the contract-audit lines, then provide one primary executable workflow. "
                    "Keep the task's canonical workflow family stable. Do not switch to a broader alternate skill workflow, and do not include parallel options, alternate toolchains, or a second full script."
                ),
            }]
            resp = _retry_api_call(
                (
                    "OpenAI Gemini benchmark finalization request"
                    if _model_uses_gemini_compat(model)
                    else "OpenAI DeepSeek finalization request"
                ),
                lambda: self._llm.chat.completions.create(
                    **_get_openai_chat_create_kwargs(
                        self.env,
                        model,
                        forced_messages,
                    )
                ),
                retries=10,
            )
            usage = _extract_token_usage_from_response(resp)
            token_usage_totals["prompt_tokens"] += usage["prompt_tokens"]
            token_usage_totals["completion_tokens"] += usage["completion_tokens"]
            token_usage_totals["total_tokens"] += usage["total_tokens"]
            self._last_token_usage = dict(token_usage_totals)
            try:
                content = resp.choices[0].message.content or ""
            except Exception:
                content = ""
            if content.strip():
                return content

        return "[Agent: tool-call loop reached max iterations]"

    def _latest_user_task(self) -> str:
        for msg in reversed(self.history):
            if str(msg.get("role", "")).strip().lower() == "user":
                return str(msg.get("content", "")).strip()
        return ""

    def _should_write_benchmark_report(self) -> bool:
        if not self.benchmark_mode:
            return False
        return bool(self._latest_user_task())

    def _write_benchmark_report(self, assistant_response: str) -> None:
        task = self._latest_user_task()
        if not task:
            return

        llm_model = str(
            self._benchmark_report_model
            or self.env.get("llm_backend", {}).get("model", "model_unknown")
        )
        model_dir = _slugify_filename_part(llm_model, "model_unknown", max_len=120)
        output_dir = self.workspace / "output" / model_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        variant_label = "standard"
        if self.no_skill_mode:
            variant_label = "no-skills"
        elif llm_model.endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
            variant_label = "with-skills"
        run_index = max(1, int(getattr(self, "_benchmark_report_run_index", 1) or 1))
        report_stem = _benchmark_report_filename(task, llm_model, run_index=run_index)
        report_path = output_dir / f"{report_stem}.md"

        if self.no_skill_mode:
            skill_summary = {"total_unique_skills": 0, "skills": []}
        else:
            skill_summary = _summarize_skill_usage(self._tool_events, self.skills, assistant_response)

        command_lines: list[str] = []
        for idx, event in enumerate(self._tool_events, start=1):
            tool = str(event.get("tool", "unknown"))
            cmd = str(event.get("command", "")).strip()
            executed = bool(event.get("executed", False))
            status = "executed" if executed else "suggested-only"
            if cmd:
                command_lines.append(f"{idx}. [{tool}] ({status}) `{cmd}`")
            else:
                command_lines.append(f"{idx}. [{tool}] ({status})")

        if not command_lines:
            command_lines.append("1. No tool command was used.")

        thinking_body, commands_body = _split_benchmark_answer_sections(assistant_response)
        commands_lines_from_answer = [line for line in commands_body.splitlines()] if commands_body else []
        final_command_lines = commands_lines_from_answer if commands_lines_from_answer else command_lines
        tool_event_lines = list(command_lines)
        if self.no_skill_mode:
            skill_summary = {"total_unique_skills": 0, "skills": []}

        report = [
            f"# Benchmark Report: {report_stem}",
            "",
            f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
            "- Mode: benchmark",
            f"- Benchmark variant: {variant_label}",
            f"- Case ID: {_extract_case_id(task)}",
            f"- Model: {llm_model}",
            f"- Task: {task}",
            f"- Elapsed seconds: {self._last_chat_elapsed_sec}",
            (
                "- Token usage: "
                f"prompt={int(self._last_token_usage.get('prompt_tokens', 0))}, "
                f"completion={int(self._last_token_usage.get('completion_tokens', 0))}, "
                f"total={int(self._last_token_usage.get('total_tokens', 0))}"
            ),
            f"- Skills used (unique): {int(skill_summary.get('total_unique_skills', 0))}",
            f"- Skill names used: {json.dumps(skill_summary.get('skills', []), ensure_ascii=False)}",
            f"- Skill files read via tool (unique): {int(skill_summary.get('read_unique_skills', 0))}",
            f"- Skill files read via tool: {json.dumps(skill_summary.get('read_skills', []), ensure_ascii=False)}",
            "",
            "## Solution Thinking",
            thinking_body,
            "",
            "## Commands Or Code",
            *final_command_lines,
            "",
            "## Tool Events",
            *tool_event_lines,
            "",
        ]
        report_path.write_text("\n".join(report), encoding="utf-8")

    def _build_system_prompt(self, skills: list[dict]) -> str:
        soul = _load_system_prompt_text(
            self.benchmark_mode,
            self.workspace,
            no_skill_mode=self.no_skill_mode,
        )
        loaded_skills_line = ""
        skill_hint_summary = ""
        skill_exec_summary = ""
        skill_catalog_summary = ""
        model_name = str(self.env.get("model") or "")
        if self.benchmark_mode and not self.no_skill_mode:
            selected_skills = _select_benchmark_skill_hints(skills, model_name)
            skill_names = ", ".join(s.get("name", "") for s in selected_skills)
            loaded_skills_line = f"\n\nLoaded skills: {skill_names}"
            if _model_uses_grok_compat(model_name):
                skill_hint_summary = _build_compact_skill_hint_summary(selected_skills)
            else:
                skill_hint_summary = _build_skill_hint_summary(selected_skills)
            if not _model_uses_grok_compat(model_name):
                # Limit catalog and exec summaries to the smaller selected set
                # to avoid prompt bloat; prefer a compact/top-1 executable hint.
                skill_catalog_summary = _build_skill_catalog_summary(selected_skills, max_skills=8)
                skill_exec_summary = _build_skill_exec_summary(selected_skills, self.workspace, max_skills=1)
        benchmark_policy = ""
        if self.benchmark_mode:
            benchmark_policy = (
                "\n\n[Benchmark Task Contract]\n"
                "- Treat the benchmark instructions below as task-scoped output and execution constraints for this response only.\n"
                "- The user will provide only one task description and expects direct completion.\n"
                "- Do NOT ask the user for intermediate confirmations, approvals, or step-by-step permission.\n"
                "- Do NOT repeat or save the task scaffolding sections such as Input Requirement, Constraints, or Evaluation.\n"
                "- If the task requires data input files (e.g., imaging/genomics) or output files, do not perform real file I/O; provide executable commands/code snippets only.\n"
                "- If the task is a quick no-file task, execute it autonomously end-to-end.\n"
                "- Do not ask the user to read files first in benchmark mode; reason from the task description and give the commands/code you would use.\n"
                "- In with-skills benchmark runs, treat local SKILL.md content as task guidance rather than a rigid workflow template. When relevant skill hints are provided and the current route exposes tools, make one relevant SKILL.md inspection your default first action before finalizing the solution, unless this is a clearly self-contained quick no-file task that obviously does not benefit from skill guidance. Prefer reading the single most relevant hinted SKILL.md first rather than skipping retrieval entirely. Differences between the task's required inputs/outputs and the skill's default example inputs/outputs are normal; adapt the skill's methodology, sequencing, validation ideas, naming conventions, and implementation experience to the task-specific contract instead of rejecting the skill just because the example I/O differs. If local SKILL.md inspection does not happen in this run, do not claim file-inspected adoption; instead state whether the skill was actually adopted from local inspection, adopted only from injected hints, or not adopted. Do not write that the route lacks support unless a tool call actually fails.\n"
                "- Skill inspection is an intermediate step only, never the final answer. After any skill inspection, continue in the same turn to produce the full task-complete Solution Thinking and Commands Or Code. Do not stop at statements like 'I will inspect the skill' or 'Let me inspect the skill'.\n"
                "- In benchmark with-skills mode, do not treat skill inspection as optional merely because you can already draft a direct answer from prior knowledge. If a relevant skill hint is present and tools are available, inspect one relevant local SKILL.md first, then either use it directly, use it as skill-informed guidance, or explicitly decide not to adopt it.\n"
                "- When using a skill, adapt its examples and guidance to the task's required inputs, outputs, filenames, and paths; do not copy the skill's default I/O assumptions unchanged. I/O mismatch alone is not a reason to reject a skill if its method, parameters, sequencing, validation, or implementation pattern still helps solve the task.\n"
                "- Do not stop at a skill's default shortcut command or minimal example if the benchmark task requires more explicit deliverables, parameter choices, file conventions, or checks. Reading a skill is not enough; the final workflow must still satisfy the benchmark task's concrete output contract.\n"
                "- In with-skills benchmark runs, internally compare the task's required inputs/outputs against the skill's default example inputs/outputs before producing the final solution, but do not include that intermediate comparison in the final answer unless it is strictly needed to justify a task-critical choice.\n"
                "- If the skill's default I/O conflicts with the benchmark task contract, the benchmark task contract always wins, and you must rewrite the solution around the task contract instead of the skill default while preserving any useful skill-derived method choices that still fit the task.\n"
                "- Distinguish between 'skill-informed' and 'skill-adopted' reasoning. If the final plan borrows useful methodology, sequencing, validation steps, parameter choices, file conventions, or implementation lessons from a relevant skill, that counts as skill-informed use even when the final answer does not preserve a full explicit skill workflow. Reserve stronger 'skill adopted' wording for cases where the final plan clearly keeps a substantial task-specific portion of the skill's workflow or interface decisions.\n"
                "- Do not say a skill was irrelevant merely because you did not fully adopt its default workflow. If the final solution benefits from the skill's experience, let that influence the solution directly instead of forcing an all-or-nothing adoption decision.\n"
                "- If no local SKILL.md was actually inspected in this run, do not let hinted skills change the benchmark's canonical task family. They may sharpen defaults, parameter choices, validation, sequencing, or implementation details, but they must not switch the task to a different starting representation, different primary software stack, or a broader substitute workflow unless the benchmark text itself explicitly requires that branch.\n"
                "- When the task already implies a canonical family, keep that family stable unless local inspected skill content or explicit task text justifies a change. For example, do not rewrite tensor-derived-metric tasks into raw-DWI fitting tasks, HCP Pipeline tasks into generic fMRIPrep tasks, or transform-reuse alignment tasks into de novo registration tasks just because a hinted skill makes that alternative familiar.\n"
                "- When a skill is adopted, include task-specific validation for the benchmark contract, such as checking the exact required output artifacts, verifying path/interface rewrites, or naming the QC/consistency checks that prove the task was completed. Do not rely on generic 'check the output directory' language when the task expects more specific deliverables.\n"
                "- If you inspect a skill and then decide the direct task-level solution is better, silently drop the intermediate skill-comparison narrative and present the final task-level solution directly. Do not carry the rejected skill path forward into the remaining plan, commands, or validation narrative.\n"
                "- In with-skills benchmark runs, when the final implementation does not adopt any skill, the answer should read like a direct task solution rather than a skill-mediated workflow.\n"
                "- In with-skills benchmark runs, if the final implementation does not adopt any skill, keep the reasoning at the same level of directness and brevity you would use in a no-skills run. Do not add extra explanation, extra setup branches, or extra validation narrative merely because a skill was inspected or hinted earlier.\n"
                "- Once you choose the final implementation path, keep that decision binding for the remainder of the answer. Do not keep parallel candidate paths, side-by-side implementations, or 'Option 1 / Option 2' answers unless the benchmark task explicitly requires multiple deliverables or a prerequisite is genuinely missing.\n"
                "- If you mention a fallback, it must be brief and subordinate to the chosen primary path. Do not give the fallback equal weight, equal code volume, or a second full command block when the main path is already task-complete.\n"
                "- If the benchmark task requires prerequisite inputs or prior-step artifacts that are missing, say `Missing required input` rather than silently switching back to a different default skill interface.\n"
                "- If you say `Missing required input`, you must still provide the most task-specific executable fallback you can from the stated task contract; do not retreat into generic wrapper orchestration, download/setup boilerplate, or placeholder skill chaining when a more direct task-level command/code path can be written.\n"
                "- For canonical pipeline tasks with a strong standard solution path, prefer the narrowest task-complete mainline solution first. Do not widen the answer with extra branches, side tools, or optional workflows unless the task explicitly requires them or a missing prerequisite forces a fallback.\n"
                "- If a skill's default interface is broader, more generic, or less concrete than the benchmark task contract, do not keep that interface in the final commands just because it came from the skill.\n"
                "- Do not cite or display unrelated skills just to show ecosystem coverage. Irrelevant modality skills, optional helper tools, or side-car setup branches that do not materially improve the benchmark task should be omitted from the plan and commands.\n"
                "- Do not add an extra orchestrator layer just to invoke a skill. Avoid subprocess wrappers, temporary staging files, and shelling out to a skill CLI when the same task can be solved more directly in task-level code or commands. Use the skill for implementation guidance, not as an excuse to lengthen the path.\n"
                "- Even when a skill is adopted, keep environment setup, helper wrappers, and validation scoped to the single chosen mainline. Do not turn setup, reproducibility notes, or QC checks into a second implementation path or a second full command/code block unless the benchmark task explicitly requires that additional artifact.\n"
                "- It is acceptable to cite a skill by capability even when you do not expose a literal skill entry-point command, as long as the chosen skill matches the provided skill hints and the resulting plan/code are task-correct.\n"
                "- Prefer at most one adopted primary skill in the final implementation unless a second skill is strictly necessary to satisfy a distinct required subtask. Do not accumulate multiple candidate skills in the final answer if only one or none is actually needed.\n"
                "- The final Commands Or Code section should contain one primary executable workflow. Do not include a second full script, second full toolchain, or second full environment path merely to hedge uncertainty.\n"
                "- Final answer format must contain exactly two top-level sections: ## Solution Thinking and ## Commands Or Code.\n"
                "- The final answer must not include sections named Input Requirement, Constraints, Evaluation, or any repeated task-header boilerplate.\n"
                "- A response that only reports skill inspection intent or only lists inspected skills is incomplete. The final answer must still include a task-complete plan and executable commands/code even when a skill was inspected successfully.\n"
                "- In ## Solution Thinking, provide a concrete and detailed step-by-step plan that covers the full task flow, key assumptions, validation points, expected outputs, and any required fallback handling.\n"
                "- In ## Commands Or Code, provide accurate, executable commands or code snippets rather than placeholders; include specific paths, filenames, arguments, environment setup, and output locations whenever the task implies them.\n"
                "- Prefer task-specific, implementation-ready instructions over generic advice; avoid vague templates, pseudo-code, or placeholder values unless the task explicitly leaves a value unknown.\n"
                "- Continue until a complete final answer is produced in the same turn.\n"
            )
        if self.no_skill_mode:
            benchmark_policy += (
                "\n[No-Skill Baseline Contract]\n"
                "- This run is baseline without skills.\n"
                "- Do not call tools or external skills; reason and answer directly.\n"
                "- Keep output format unchanged (Solution Thinking + Commands Or Code).\n"
            )
        extra_parts = [part for part in (skill_hint_summary, skill_catalog_summary, skill_exec_summary) if part]
        extra = "\n\n" + "\n\n".join(extra_parts) if extra_parts else ""
        return f"{soul}{loaded_skills_line}{extra}{benchmark_policy}"

    @staticmethod
    def _prompt_setup() -> None:
        print(
            "neuroclaw_environment.json not found.\n"
            "Please run the installer first:\n\n"
            "    python installer/setup.py\n"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroClaw — neuroscience AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python core/agent/main.py                # interactive REPL\n"
            "  python core/agent/main.py --benchmark    # benchmark batch runner\n"
            "  python core/agent/main.py --benchmark --benchmark-workers 8\n"
            "  python core/agent/main.py --benchmark --benchmark-repeats 3\n"
            "  python core/agent/main.py --benchmark --benchmark-compare-skills  # run with-skills/no-skills pair\n"
            "  python core/agent/main.py --score-benchmark --score-workers 8\n"
            "  python core/agent/main.py --score-benchmark --score-model gpt-5.4\n"
            "  python core/agent/main.py --web          # browser GUI on :7080\n"
            "  python core/agent/main.py --web --port 8080\n"
            "  python core/agent/main.py --web --benchmark"
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the browser-based Web UI instead of the interactive REPL.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7080,
        metavar="PORT",
        help="Port for the Web UI (default: 7080). Ignored unless --web is set.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="Bind host for the Web UI (default: 127.0.0.1). Ignored unless --web is set.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        metavar="KEY",
        help=(
            "Runtime-only LLM API key override. This is not written to setup files; "
            "it is applied only to the current process."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Enable benchmark mode: file/dataset input-output tasks are simulated "
            "(command/code only), quick no-file tasks can still execute, and a report "
            "is saved to output/<task-name>.md."
        ),
    )
    parser.add_argument(
        "--benchmark-compare-skills",
        action="store_true",
        help=(
            "When used with --benchmark, run two variants per task: with-skills and no-skills "
            "for gain analysis (A_abs and normalized gain g)."
        ),
    )
    parser.add_argument(
        "--benchmark-workers",
        type=int,
        default=8,
        metavar="N",
        help="Worker process count for benchmark execution (default: 8).",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        metavar="N",
        help="Repeat count per benchmark case and model variant (default: 3).",
    )
    parser.add_argument(
        "--score-benchmark",
        action="store_true",
        help=(
            "Score benchmark reports in output/ using gpt-5.4 with weighted rubric: "
            "for each task, all comparable models are scored jointly in one batch to reduce standard drift; "
            "LLM score = planning completeness 30%%, action reasonableness 40%%, "
            "code/command correctness 30%%; rubric rewards comprehensive problem coverage and validation, not just minimal executable plans; usage efficiency is computed separately "
            "from skill usage counts, plus A_abs and normalized gain g for with-skills vs no-skills pairs."
        ),
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=8,
        metavar="N",
        help="Worker process count for benchmark scoring (default: 8).",
    )
    parser.add_argument(
        "--score-model",
        default="",
        metavar="MODEL",
        help=(
            "Scorer model for --score-benchmark. If omitted, an interactive 1-10 model picker is shown. "
            f"Default picker selection: {BENCHMARK_SCORER_MODEL}. Example: --score-model gpt-5.4-mini"
        ),
    )
    parser.add_argument(
        "--score-compare-model",
        default="",
        metavar="MODEL",
        help=(
            "When used with --score-benchmark, compare a specific base model's with-skills vs no-skills pair. "
            "Example: --score-compare-model gpt-5.4"
        ),
    )
    parser.add_argument(
        "--probe-tool-call",
        default="",
        metavar="MODEL",
        help=(
            "Run a minimal one-shot tool-call probe against the configured OpenAI-compatible route "
            "using the provided model name, then exit. Example: --probe-tool-call grok-4-20-non-reasoning"
        ),
    )
    parser.add_argument(
        "--probe-tool-loop",
        default="",
        metavar="MODEL",
        help=(
            "Run a minimal 2-turn tool loop probe (tool call + tool result) against the configured "
            "OpenAI-compatible route using the provided model name, then exit. Use this to debug "
            "provider-specific tool-call metadata such as thought_signature."
        ),
    )
    args = parser.parse_args()
    env = load_environment()
    if args.api_key:
        _apply_runtime_api_key(env, args.api_key)

    if args.probe_tool_call:
        raise SystemExit(
            _run_openai_tool_probe(env, str(args.probe_tool_call).strip(), REPO_ROOT)
        )

    if args.probe_tool_loop:
        raise SystemExit(
            _run_openai_tool_loop_probe(env, str(args.probe_tool_loop).strip(), REPO_ROOT)
        )

    if args.benchmark:
        os.environ[BENCHMARK_ENV_FLAG] = "1"

    if args.score_benchmark:
        default_benchmark_root = str(REPO_ROOT / "neuro_bench")
        default_output_root = str(REPO_ROOT / "output")

        while True:
            benchmark_root_input = _prompt_with_default(
                "Benchmark directory",
                default_benchmark_root,
            )
            benchmark_root = _resolve_benchmark_root(benchmark_root_input)
            if benchmark_root.exists() and benchmark_root.is_dir():
                break
            print(f"Benchmark directory not found: {benchmark_root}")

        while True:
            output_root_input = _prompt_with_default(
                "Benchmark report output directory",
                default_output_root,
            )
            output_root = _resolve_benchmark_root(output_root_input)
            if output_root.exists() and output_root.is_dir():
                break
            print(f"Output directory not found: {output_root}")

        scorer_selection = None
        explicit_scorer_model = str(args.score_model or "").strip()
        if explicit_scorer_model:
            scorer_selection = {"provider": "openai", "model": explicit_scorer_model}
        else:
            scorer_selection = _prompt_model_selection("Scorer", BENCHMARK_SCORER_MODEL)

        # If user provided --score-compare-model, use it; otherwise prompt interactively.
        explicit_compare = str(args.score_compare_model or "").strip()
        compare_base_model: str | None = explicit_compare or None
        if not compare_base_model:
            try:
                report_index = _discover_benchmark_reports(output_root)
                # Build base model candidates
                conds_by_base: dict[str, set[str]] = {}
                for model_name in sorted(report_index.keys()):
                    base, cond = _split_skill_condition(model_name)
                    conds_by_base.setdefault(base, set()).add(cond)

                candidates = sorted(conds_by_base.keys())
                if candidates:
                    print("\nAvailable base models in output (select 0 to score all):")
                    for idx, base in enumerate(candidates, start=1):
                        flags = ",".join(sorted(conds_by_base.get(base, set())))
                        print(f"  {idx}. {base} [{flags}]")
                    while True:
                        choice = _prompt_with_default(f"Select base model to compare [0-{len(candidates)}] (0=all)", "0")
                        try:
                            sel = int(choice)
                        except Exception:
                            print("Please enter a valid number.")
                            continue
                        if sel == 0:
                            compare_base_model = None
                            break
                        if 1 <= sel <= len(candidates):
                            compare_base_model = candidates[sel - 1]
                            break
                        print(f"Please enter a number from 0 to {len(candidates)}.")
            except Exception:
                compare_base_model = None

        score_file, leaderboard_file = _score_benchmark_reports(
            benchmark_root,
            output_root,
            score_workers=max(1, int(args.score_workers or 1)),
            scorer_model=str(scorer_selection.get("model") or "").strip() or BENCHMARK_SCORER_MODEL,
            compare_base_model=compare_base_model,
        )
        print(f"Benchmark scoring completed: {score_file}")
        print(f"Leaderboard generated: {leaderboard_file}")
        return

    if args.benchmark and not args.web:
        default_benchmark_root = str(REPO_ROOT / "neuro_bench")
        while True:
            benchmark_root_input = _prompt_with_default(
                "Benchmark directory",
                default_benchmark_root,
            )
            benchmark_root = _resolve_benchmark_root(benchmark_root_input)
            if benchmark_root.exists() and benchmark_root.is_dir():
                break
            print(f"Benchmark directory not found: {benchmark_root}")
            print("Please enter a valid benchmark directory path.\n")

        benchmark_selection = _prompt_benchmark_model_name()
        _run_benchmark_suite(
            benchmark_root,
            str(benchmark_selection.get("provider", "openai")),
            str(benchmark_selection.get("model", "gpt-4o")),
            compare_skills=bool(args.benchmark_compare_skills),
            benchmark_workers=max(1, int(args.benchmark_workers or 1)),
            benchmark_repeats=max(1, int(args.benchmark_repeats or 1)),
        )
        return

    if args.web:
        # Import lazily so FastAPI/uvicorn are only required when --web is used
        import importlib.util as _ilu

        _srv_path = Path(__file__).parent.parent / "web" / "server.py"
        _spec = _ilu.spec_from_file_location("neuroclaw_web_server", _srv_path)
        if _spec is None or _spec.loader is None:
            print(f"ERROR: Cannot find web server at {_srv_path}", file=sys.stderr)
            sys.exit(1)
        _srv_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_srv_mod)  # type: ignore[union-attr]
        _srv_mod.run_server(host=args.host, port=args.port)
    else:
        session = AgentSession(env=env, benchmark_mode=args.benchmark)
        session.start()


if __name__ == "__main__":
    main()
