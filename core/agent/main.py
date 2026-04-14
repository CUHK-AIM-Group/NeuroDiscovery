"""
NeuroClaw Core Agent — LLM conversation loop and tool-call dispatcher.

Architecture
------------
  AgentSession
    │
    ├─ SkillLoader        loads skills/*/SKILL.md, registers tools
    ├─ ToolRuntime        executes handler.js / Python handlers
    ├─ SessionManager     context window, persistence, compression
    └─ LLMBackend         OpenAI / Anthropic / local model adapter

This module is the self-contained replacement for OpenClaw's agent loop.
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
_BENCHMARK_SCORER_CLIENT_CACHE: Any | None = None

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
    """Keep only the task description and output target, dropping evaluation scaffolding."""
    text = str(task_markdown or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    kept: list[str] = []
    dropping = False
    drop_headings = {
        "## input requirement",
        "## inputs",
        "## constraints",
        "## evaluation",
        "## success criteria",
        "## implementation details",
        "## notes",
        "## expected output",
        "## expected outputs",
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

        if stripped.startswith("## ") and lower not in {"## objective", "## task description"}:
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


def _prompt_with_default(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        value = ""
    return value if value else default


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
    name = str(model_name or "").strip().lower()
    return "gemini" in name


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


def _summarize_tool_events(tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    total_calls = 0
    executed_calls = 0
    by_type: dict[str, int] = {}

    for event in tool_events:
        total_calls += 1
        if bool(event.get("executed", False)):
            executed_calls += 1

        tool = str(event.get("tool", "")).strip()
        command = str(event.get("command", "")).strip()

        if tool == "run_shell_command":
            kinds = _classify_shell_command_tool_kinds(command)
        elif tool:
            kinds = {tool}
        else:
            kinds = {"unknown"}

        for kind in kinds:
            by_type[kind] = int(by_type.get(kind, 0)) + 1

    return {
        "total_calls": total_calls,
        "executed_calls": executed_calls,
        "suggested_only_calls": max(0, total_calls - executed_calls),
        "by_type": dict(sorted(by_type.items())),
    }


def _extract_tool_call_count_from_report(report_text: str) -> int | None:
    text = str(report_text or "")
    m = re.search(r"-\s*Tool calls \(total\):\s*(\d+)", text, flags=re.IGNORECASE)
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


def _compute_tool_efficiency_score(tool_count: int | None, peer_counts: list[int]) -> float:
    if tool_count is None:
        return 5.0
    if not peer_counts:
        return 10.0

    min_count = min(peer_counts)
    max_count = max(peer_counts)
    if max_count <= min_count:
        return 10.0

    # Fewer calls are better for the same case; map [min, max] -> [10, 1].
    score = 10.0 - 9.0 * ((float(tool_count) - float(min_count)) / float(max_count - min_count))
    if score < 1.0:
        score = 1.0
    if score > 10.0:
        score = 10.0
    return round(score, 2)


def _score_single_benchmark_case(
    llm_client: Any,
    task_text: str,
    report_text: str,
    model_name: str,
) -> dict[str, Any]:
    scoring_system = (
        "You are a strict benchmark evaluator for neuroscience-agent outputs. "
        "Score only from the provided task and report. "
        "Return JSON only."
    )
    scoring_user = (
        "Evaluate the benchmark report on a 1-10 scale for three dimensions:\n"
        "1) planning_completeness\n"
        "2) tool_reasonableness\n"
        "3) code_command_correctness\n"
        "Do not score tool_efficiency here; it will be calculated from tool-call counts across models for the same case.\n"
        "Important: tool selection quality is still very important.\n"
        "Output strict JSON with keys exactly:\n"
        "planning_completeness, tool_reasonableness, code_command_correctness, "
        "brief_justification.\n"
        "\n"
        f"Task contract:\n{task_text}\n\n"
        f"Model report:\n{report_text}\n"
    )
    resp = llm_client.chat.completions.create(
        model=BENCHMARK_SCORER_MODEL,
        messages=[
            {"role": "system", "content": scoring_system},
            {"role": "user", "content": scoring_user},
        ],
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
        "tool_call_count": _extract_tool_call_count_from_report(report_text),
        "brief_justification": str(parsed.get("brief_justification", "")).strip(),
        "raw_model_output": content,
    }


def _get_benchmark_scorer_client() -> Any:
    global _BENCHMARK_SCORER_CLIENT_CACHE
    if _BENCHMARK_SCORER_CLIENT_CACHE is not None:
        return _BENCHMARK_SCORER_CLIENT_CACHE

    scorer_env = load_environment()
    llm_cfg = scorer_env.setdefault("llm_backend", {})
    llm_cfg["provider"] = "openai"
    llm_cfg["model"] = BENCHMARK_SCORER_MODEL
    _BENCHMARK_SCORER_CLIENT_CACHE = build_llm_client(scorer_env)
    return _BENCHMARK_SCORER_CLIENT_CACHE


def _score_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    task_file = Path(str(job.get("task_file", "")))
    report_file = Path(str(job.get("report_file", "")))
    model_name = str(job.get("model_name", ""))
    case_id = str(job.get("case_id", ""))
    run_index = max(1, int(job.get("run_index", 1) or 1))

    task_text = task_file.read_text(encoding="utf-8")
    report_text = report_file.read_text(encoding="utf-8")
    elapsed_seconds = _extract_elapsed_seconds_from_report(report_text)
    token_usage = _extract_token_usage_from_report_text(report_text)
    scorer_client = _get_benchmark_scorer_client()
    case_score = _score_single_benchmark_case(
        llm_client=scorer_client,
        task_text=task_text,
        report_text=report_text,
        model_name=model_name,
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
        "| Rank | Model | Average Score (%) | Avg Tool Calls | Avg Tokens | Avg Time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for idx, item in enumerate(ranking, start=1):
        model = str(item.get("model", ""))
        avg = item.get("average_weighted_score", "")
        avg_calls = item.get("average_tool_calls", "-")
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
    print(f"Discovered models in output: {len(report_index)}")

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

    print(f"Comparable models kept for scoring: {len(complete_models)}")
    print(f"Shared case ids selected for scoring: {len(common_case_ids)}")
    print(f"Comparable runs per case used for scoring: {required_runs_per_case}")

    workers = max(1, int(score_workers or 1))
    print(f"Score workers: {workers}")

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "scorer_model": BENCHMARK_SCORER_MODEL,
            "score_scale": "0-100",
            "llm_score_components": [
                "planning_completeness",
                "tool_reasonableness",
                "code_command_correctness",
            ],
            "tool_efficiency_note": "Computed separately from tool_call_count within same case across models; not included in LLM weighted score.",
            "weights": BENCHMARK_SCORE_WEIGHTS,
            "benchmark_root": str(benchmark_root),
            "output_dir": str(output_dir),
            "runs_per_case": required_runs_per_case,
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
    for model_name, case_map in sorted(complete_models.items()):
        for case_id in common_case_ids:
            run_entries = case_map[case_id][:required_runs_per_case]
            for entry in run_entries:
                jobs.append(
                    {
                        "model_name": model_name,
                        "case_id": case_id,
                        "run_index": int(entry.get("run_index", 1) or 1),
                        "task_file": str(task_contracts[case_id]),
                        "report_file": str(entry["report_file"]),
                    }
                )

    total_jobs = len(jobs)
    print(f"Total scoring jobs: {total_jobs}", flush=True)

    per_case_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_score_benchmark_job, job) for job in jobs]
        for done_jobs, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                print(f"[score job {done_jobs}/{total_jobs}] ERROR: worker crashed: {exc}", flush=True)
                continue

            model_name = str(result.get("model_name", ""))
            case_id = str(result.get("case_id", ""))
            run_index = max(1, int(result.get("run_index", 1) or 1))
            result["run_index"] = run_index
            per_case_results.setdefault((model_name, case_id), []).append(result)
            print(f"Scoring [{done_jobs}/{total_jobs}] model={model_name} case={case_id} run={run_index}", flush=True)

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
            tool_distribution: list[float] = []
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
                if isinstance(result.get("tool_call_count"), int):
                    tool_distribution.append(float(result["tool_call_count"]))
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
                        "tool_call_count": result["tool_call_count"],
                        "brief_justification": result["brief_justification"],
                        "raw_model_output": result["raw_model_output"],
                        "weighted_score_10": weighted_10,
                        "weighted_score": weighted_score,
                    }
                )

            weighted_mean, weighted_variance = _mean_and_variance(weighted_distribution)
            tool_mean, tool_variance = _mean_and_variance(tool_distribution)
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
                "average_tool_calls": tool_mean,
                "variance_tool_calls": tool_variance,
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
            "average_tool_calls": None,
            "variance_tool_calls": None,
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
        tool_values: list[float] = []
        elapsed_values: list[float] = []
        prompt_values: list[float] = []
        completion_values: list[float] = []
        total_values: list[float] = []
        for case_id in common_case_ids:
            case_data = data.get("cases", {}).get(case_id, {})
            if isinstance(case_data.get("average_weighted_score"), (int, float)):
                score_values.append(float(case_data["average_weighted_score"]))
            if isinstance(case_data.get("average_tool_calls"), (int, float)):
                tool_values.append(float(case_data["average_tool_calls"]))
            if isinstance(case_data.get("average_elapsed_seconds"), (int, float)):
                elapsed_values.append(float(case_data["average_elapsed_seconds"]))
            if isinstance(case_data.get("average_prompt_tokens"), (int, float)):
                prompt_values.append(float(case_data["average_prompt_tokens"]))
            if isinstance(case_data.get("average_completion_tokens"), (int, float)):
                completion_values.append(float(case_data["average_completion_tokens"]))
            if isinstance(case_data.get("average_total_tokens"), (int, float)):
                total_values.append(float(case_data["average_total_tokens"]))

        score_mean, score_variance = _mean_and_variance(score_values)
        tool_mean, tool_variance = _mean_and_variance(tool_values)
        elapsed_mean, elapsed_variance = _mean_and_variance(elapsed_values)
        prompt_mean, prompt_variance = _mean_and_variance(prompt_values)
        completion_mean, completion_variance = _mean_and_variance(completion_values)
        total_mean, total_variance = _mean_and_variance(total_values)
        data["average_weighted_score"] = round(score_mean, 2) if score_mean is not None else None
        data["variance_weighted_score"] = score_variance
        data["average_tool_calls"] = round(tool_mean, 2) if tool_mean is not None else None
        data["variance_tool_calls"] = tool_variance
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
                "average_tool_calls": data.get("average_tool_calls"),
                "average_total_tokens": data.get("average_total_tokens"),
                "average_elapsed_seconds": data.get("average_elapsed_seconds"),
            }
            for model, data in results["models"].items()
        ),
        key=lambda x: (
            -float(x.get("average_weighted_score", 0.0) or 0.0),
            float(x.get("average_tool_calls", 1e9) or 1e9),
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
    out_file = output_dir / f"benchmark_scores_{BENCHMARK_SCORER_MODEL}_{timestamp}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    leaderboard_file = output_dir / f"benchmark_leaderboard_{BENCHMARK_SCORER_MODEL}_{timestamp}.md"
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
            "- Tools used: none",
            "- Elapsed seconds: 0.0",
            "- Token usage: prompt=0, completion=0, total=0",
            "- Tool calls (total): 0",
            "- Tool calls (executed): 0",
            "- Tool calls (suggested-only): 0",
            "- Tool calls by type: {}",
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
    model_name: str,
    compare_skills: bool = False,
    benchmark_workers: int = 8,
    benchmark_repeats: int = 5,
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
                        "model_name": model_name,
                        "run_model_name": run_model_name,
                        "run_index": run_index,
                    }
                )

    print(f"Total benchmark jobs: {len(jobs)}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_single_benchmark_job, job) for job in jobs]
        for done_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                print(f"[job {done_idx}/{len(jobs)}] ERROR: worker crashed: {exc}", flush=True)
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
                print(f"{prefix} Saved {report_rel}", flush=True)
            else:
                error_text = str(result.get("error", "unknown error"))
                if report_rel:
                    print(f"{prefix} {error_text} (saved {report_rel})", flush=True)
                else:
                    print(f"{prefix} {error_text}", flush=True)

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
            resp = self._llm.messages.create(
                model=model,
                max_tokens=4096,
                system=system_msg,
                messages=user_msgs,
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
            except Exception:
                pass

        return response

    def _chat_openai_with_tools(self, model: str) -> str:
        """OpenAI chat with a minimal native shell tool-call loop."""
        if self.no_skill_mode or (self.benchmark_mode and _model_uses_gemini_compat(model)):
            resp = self._llm.chat.completions.create(
                model=model,
                messages=list(self.history),
            )
            self._last_token_usage = _extract_token_usage_from_response(resp)
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return ""

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
            }
        ]

        messages: list[dict[str, Any]] = list(self.history)
        token_usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for _ in range(4):
            resp = self._llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
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

            assistant_tool_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
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

        return "[Agent: tool-call loop reached max iterations]"

    def _latest_user_task(self) -> str:
        for msg in reversed(self.history):
            if str(msg.get("role", "")).strip().lower() == "user":
                return str(msg.get("content", "")).strip()
        return ""

    def _should_write_benchmark_report(self) -> bool:
        # Skip internal utility calls (title generation, skill-summary prompts, etc.)
        for msg in self.history:
            if str(msg.get("role", "")).strip().lower() != "system":
                continue
            if "Loaded skills:" in str(msg.get("content", "")):
                return True
        return False

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

        tool_names = sorted({str(e.get("tool", "")).strip() for e in self._tool_events if str(e.get("tool", "")).strip()})
        if not tool_names:
            tool_names = ["none"]
        tool_summary = _summarize_tool_events(self._tool_events)

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

        report = [
            f"# Benchmark Report: {report_stem}",
            "",
            f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
            "- Mode: benchmark",
            f"- Benchmark variant: {variant_label}",
            f"- Case ID: {_extract_case_id(task)}",
            f"- Model: {llm_model}",
            f"- Task: {task}",
            f"- Tools used: {', '.join(tool_names)}",
            f"- Elapsed seconds: {self._last_chat_elapsed_sec}",
            (
                "- Token usage: "
                f"prompt={int(self._last_token_usage.get('prompt_tokens', 0))}, "
                f"completion={int(self._last_token_usage.get('completion_tokens', 0))}, "
                f"total={int(self._last_token_usage.get('total_tokens', 0))}"
            ),
            f"- Tool calls (total): {int(tool_summary.get('total_calls', 0))}",
            f"- Tool calls (executed): {int(tool_summary.get('executed_calls', 0))}",
            f"- Tool calls (suggested-only): {int(tool_summary.get('suggested_only_calls', 0))}",
            f"- Tool calls by type: {json.dumps(tool_summary.get('by_type', {}), ensure_ascii=False)}",
            "",
            "## Solution Thinking",
            assistant_response or "(empty)",
            "",
            "## Commands Or Code",
            *command_lines,
            "",
        ]
        report_path.write_text("\n".join(report), encoding="utf-8")

    def _build_system_prompt(self, skills: list[dict]) -> str:
        soul = _load_system_prompt_text(
            self.benchmark_mode,
            self.workspace,
            no_skill_mode=self.no_skill_mode,
        )
        skill_names = ", ".join(s.get("name", "") for s in skills)
        benchmark_policy = ""
        if self.benchmark_mode:
            benchmark_policy = (
                "\n\n[Benchmark Mode Override]\n"
                "- The user will provide only one task description and expects direct completion.\n"
                "- Do NOT ask the user for intermediate confirmations, approvals, or step-by-step permission.\n"
                "- Do NOT repeat or save the task scaffolding sections such as Input Requirement, Constraints, or Evaluation.\n"
                "- If the task requires data input files (e.g., imaging/genomics) or output files, do not perform real file I/O; provide executable commands/code snippets only.\n"
                "- If the task is a quick no-file task, execute it autonomously end-to-end.\n"
                "- Do not ask the user to read files first in benchmark mode; reason from the task description and give the commands/code you would use.\n"
                "- Final answer format must contain exactly two top-level sections: ## Solution Thinking and ## Commands Or Code.\n"
                "- The final answer must not include sections named Input Requirement, Constraints, Evaluation, or any repeated task-header boilerplate.\n"
                "- In ## Solution Thinking, provide a concrete and detailed step-by-step plan that covers the full task flow, key assumptions, validation points, expected outputs, and any required fallback handling.\n"
                "- In ## Commands Or Code, provide accurate, executable commands or code snippets rather than placeholders; include specific paths, filenames, arguments, environment setup, and output locations whenever the task implies them.\n"
                "- Prefer task-specific, implementation-ready instructions over generic advice; avoid vague templates, pseudo-code, or placeholder values unless the task explicitly leaves a value unknown.\n"
                "- Continue until a complete final answer is produced in the same turn.\n"
            )
        if self.no_skill_mode:
            benchmark_policy += (
                "\n[No-Skill Baseline Override]\n"
                "- This run is baseline without skills.\n"
                "- Do not call tools or external skills; reason and answer directly.\n"
                "- Keep output format unchanged (Solution Thinking + Commands Or Code).\n"
            )
        return f"{soul}\n\nLoaded skills: {skill_names}{benchmark_policy}"

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
            "  python core/agent/main.py --benchmark --benchmark-repeats 5\n"
            "  python core/agent/main.py --benchmark --benchmark-compare-skills  # run with-skills/no-skills pair\n"
            "  python core/agent/main.py --score-benchmark --score-workers 8\n"
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
        default=5,
        metavar="N",
        help="Repeat count per benchmark case and model variant (default: 5).",
    )
    parser.add_argument(
        "--score-benchmark",
        action="store_true",
        help=(
            "Score benchmark reports in output/ using gpt-5.4 with weighted rubric: "
            "LLM score = planning completeness 30%, tool reasonableness 40%, "
            "code/command correctness 30%; tool efficiency is computed separately "
            "from tool-call counts, plus A_abs and normalized gain g for with-skills vs no-skills pairs."
        ),
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=8,
        metavar="N",
        help="Worker process count for benchmark scoring (default: 8).",
    )
    args = parser.parse_args()

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

        score_file, leaderboard_file = _score_benchmark_reports(
            benchmark_root,
            output_root,
            score_workers=max(1, int(args.score_workers or 1)),
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

        default_model_name = load_environment().get("llm_backend", {}).get("model", "gpt-4o")
        benchmark_model = _prompt_with_default("Benchmark model name", str(default_model_name))
        _run_benchmark_suite(
            benchmark_root,
            benchmark_model,
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
        session = AgentSession(benchmark_mode=args.benchmark)
        session.start()


if __name__ == "__main__":
    main()
