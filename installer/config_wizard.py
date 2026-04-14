"""
NeuroClaw interactive configuration wizard.

Run once after cloning the repository:

    python installer/config_wizard.py

The wizard:
  1. Detects system info (OS, Python, CUDA, existing toolchain)
  2. Prompts for Python environment (system / conda / Docker)
  3. Prompts for CUDA / GPU settings and optional PyTorch install
  4. Prompts for neuroscience toolchain paths
  5. Prompts for LLM backend credentials
  6. Writes neuroclaw_environment.json and core/config/features.json
  7. Saves an install log to installer/install_log.txt
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
ENV_FILE = REPO_ROOT / "neuroclaw_environment.json"
FEATURES_FILE = REPO_ROOT / "core" / "config" / "features.json"
DEFAULTS_FILE = Path(__file__).parent / "neuro_defaults.json"
LOG_FILE = Path(__file__).parent / "install_log.txt"

# ── Helpers ────────────────────────────────────────────────────────────────────
_log_lines: list[str] = []


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    _log_lines.append(line)
    print(line)


def _ask(prompt: str, default: str = "") -> str:
    default_hint = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{default_hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""
    return answer if answer else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{prompt} [{hint}]", "").lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _detect_cuda() -> str | None:
    """Return CUDA version string (e.g. '12.1') or None."""
    for bin_name in ("nvcc", "nvidia-smi"):
        result = _run([bin_name, "--version"])
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower() or "cuda version" in line.lower():
                    for token in line.replace(",", " ").split():
                        if token.replace(".", "").isdigit() and "." in token:
                            return token
    return None


def _detect_python(path: str) -> str | None:
    result = _run([path, "--version"])
    if result.returncode == 0:
        return result.stdout.strip() or result.stderr.strip()
    return None


def _detect_tool(name: str, env_var: str | None = None) -> str | None:
    """Check env var first, then PATH."""
    if env_var:
        val = os.environ.get(env_var, "")
        if val and Path(val).exists():
            return val
    found = shutil.which(name)
    return found


# ── Step 1: System snapshot ────────────────────────────────────────────────────
def _system_snapshot() -> dict:
    snap: dict = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_current": sys.version.split()[0],
        "cuda_detected": _detect_cuda(),
    }
    _log(f"OS: {snap['os']} {snap['os_release']} ({snap['machine']})")
    _log(f"Current Python: {snap['python_current']}")
    if snap["cuda_detected"]:
        _log(f"CUDA detected: {snap['cuda_detected']}")
    else:
        _log("CUDA: not detected (will configure cpu-only by default)")
    return snap


# ── Step 2: Python environment ─────────────────────────────────────────────────
def _setup_python(snap: dict) -> dict:
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 1 — Python Environment")
    print("=" * 60)
    print("  1. System Python  → auto-detect or enter path")
    print("  2. Conda env      → enter env name (created if missing)")
    print("  3. Docker         → enter image + python path")
    choice = _ask("Choose [1/2/3]", "2")

    result: dict = {"setup_type": None, "python_path": None,
                    "conda_env": None, "docker_config": None}

    if choice == "1":
        result["setup_type"] = "system"
        candidates = [sys.executable, "/usr/bin/python3", "/usr/local/bin/python3"]
        default_path = next((c for c in candidates if shutil.which(c)), sys.executable)
        path = _ask("Full path to Python executable", default_path)
        if not _detect_python(path):
            _log(f"WARNING: '{path}' does not appear to be a valid Python executable.")
        result["python_path"] = path
        _log(f"System Python selected: {path}")

    elif choice == "3":
        result["setup_type"] = "docker"
        image = _ask("Docker image name (e.g. neuroclaw/base:latest)")
        py_path = _ask("Python path inside container", "/usr/bin/python3")
        run_prefix = _ask("Docker run/exec prefix",
                           "docker run --rm -v $(pwd):/workspace")
        result["docker_config"] = {
            "image": image,
            "python_path": py_path,
            "run_prefix": run_prefix,
        }
        result["python_path"] = py_path
        _log(f"Docker selected: image={image}, python={py_path}")

    else:  # default: conda
        result["setup_type"] = "conda"
        env_name = _ask("Conda environment name", "neuroclaw")
        result["conda_env"] = env_name

        # Try to resolve python path inside the env
        conda_bin = shutil.which("conda") or shutil.which("mamba")
        if conda_bin:
            info = _run(["conda", "info", "--json"])
            if info.returncode == 0:
                try:
                    data = json.loads(info.stdout)
                    envs_dirs = data.get("envs_dirs", [])
                    for d in envs_dirs:
                        candidate = Path(d) / env_name / "bin" / "python"
                        if candidate.exists():
                            result["python_path"] = str(candidate)
                            break
                except (json.JSONDecodeError, KeyError):
                    pass

        if not result["python_path"]:
            guessed = f"/opt/conda/envs/{env_name}/bin/python"
            result["python_path"] = _ask(
                "Full path to python inside conda env", guessed
            )

        _log(f"Conda env selected: {env_name}, python={result['python_path']}")

        # Offer to create the env if not present
        if not Path(result["python_path"]).exists():
            if _ask_yn(f"Env '{env_name}' not found. Create it now?"):
                py_ver = _ask("Python version", "3.11")
                _log(f"Creating conda env '{env_name}' with Python {py_ver} …")
                proc = subprocess.run(
                    ["conda", "create", "-n", env_name, f"python={py_ver}", "-y"],
                    check=False,
                )
                if proc.returncode == 0:
                    _log(f"Conda env '{env_name}' created successfully.")
                else:
                    _log("WARNING: conda env creation failed. Check conda installation.")

    return result


# ── Step 3: CUDA / GPU ─────────────────────────────────────────────────────────
def _setup_cuda(snap: dict, python_path: str) -> dict:
    """
    Parameters
    ----------
    snap : dict
        System snapshot from _system_snapshot().
    python_path : str
        Full path to the Python executable chosen in _setup_python(); used
        for any automatic PyTorch installation.
    """
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 2 — CUDA / GPU Settings")
    print("=" * 60)

    detected = snap["cuda_detected"]
    if detected:
        print(f"  Detected CUDA: {detected}")
    else:
        print("  CUDA not detected on this machine.")

    cuda_version = _ask("CUDA version to use (leave blank for cpu-only)",
                        detected or "")
    if not cuda_version:
        _log("GPU: cpu-only mode selected.")
        return {"version": None, "torch_build": "cpu-only", "device": "cpu"}

    # Map CUDA version to PyTorch build tag
    _cuda_map = {
        "12.4": "cu124", "12.3": "cu123", "12.1": "cu121",
        "11.8": "cu118", "11.7": "cu117",
    }
    major_minor = ".".join(cuda_version.split(".")[:2])
    if major_minor in _cuda_map:
        default_build = _cuda_map[major_minor]
    else:
        # Version not in the known map — construct a best-guess tag and warn
        default_build = "cu" + major_minor.replace(".", "")
        _log(
            f"WARNING: CUDA {major_minor} is not in the known PyTorch build map "
            f"({', '.join(_cuda_map.keys())}). Using best-guess tag '{default_build}'. "
            "Verify at https://pytorch.org/get-started/locally/ and adjust if needed."
        )
    torch_build = _ask("PyTorch CUDA build tag", default_build)

    # Detect GPU device
    gpu_count_result = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if gpu_count_result.returncode == 0:
        gpus = [g.strip() for g in gpu_count_result.stdout.strip().splitlines() if g.strip()]
        if gpus:
            _log(f"Detected GPU(s): {', '.join(gpus)}")
    device = _ask("Default CUDA device", "cuda:0")

    cuda_cfg: dict = {"version": cuda_version, "torch_build": torch_build, "device": device}

    # Offer to install PyTorch using the user-selected Python environment
    if _ask_yn("Install torch + torchvision automatically?"):
        cmd = [
            python_path, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{torch_build}",
        ]
        _log(f"Installing PyTorch ({torch_build}) …")
        _log("Command: " + " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            _log("PyTorch installed successfully.")
        else:
            _log("WARNING: PyTorch installation failed. You can retry manually with:")
            _log("  " + " ".join(cmd))

    return cuda_cfg

# ── Step 4: Neuroscience toolchain ─────────────────────────────────────────────
def _setup_toolchain(snap: dict, python_path: str) -> tuple[dict, dict]:
    """
    Parameters
    ----------
    snap : dict
        System snapshot (reserved for future use).
    python_path : str
        Full path to the Python executable chosen in _setup_python(); used
        for pip install commands (MNE, nibabel, etc.) so packages land in
        the correct environment rather than a random system python3.

    Returns
    -------
    toolchain : dict
        Paths for FSL, FreeSurfer, dcm2niix, MATLAB.
    features : dict
        Boolean feature-enable flags for core/config/features.json.
    """
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 3 — Neuroscience Toolchain")
    print("=" * 60)

    toolchain: dict = {
        "fsl_home": None,
        "freesurfer_home": None,
        "dcm2niix": None,
        "matlab_path": None,
    }
    features: dict = {}  # will update features.json

    # FSL
    fsl = os.environ.get("FSLDIR") or _detect_tool("fsl")
    if fsl or _ask_yn("Configure FSL?"):
        path = _ask("FSLDIR path", fsl or "/usr/local/fsl")
        toolchain["fsl_home"] = path
        features["fsl"] = True
        _log(f"FSL: {path}")
    else:
        features["fsl"] = False

    # FreeSurfer
    fs = os.environ.get("FREESURFER_HOME") or _detect_tool("recon-all")
    if fs or _ask_yn("Configure FreeSurfer?"):
        path = _ask("FREESURFER_HOME path", fs or "/usr/local/freesurfer")
        toolchain["freesurfer_home"] = path
        features["freesurfer"] = True
        _log(f"FreeSurfer: {path}")
    else:
        features["freesurfer"] = False

    # MNE-Python — install into the user-selected Python environment
    if _ask_yn("Install MNE-Python (pip install mne)?"):
        subprocess.run([python_path, "-m", "pip", "install", "mne"], check=False)
        features["mne"] = True
        _log("MNE-Python: installed")
    else:
        features["mne"] = False

    # nibabel / nilearn / dipy — install into the user-selected Python environment
    if _ask_yn("Install nibabel, nilearn, dipy?"):
        subprocess.run(
            [python_path, "-m", "pip", "install", "nibabel", "nilearn", "dipy"],
            check=False,
        )
        _log("nibabel / nilearn / dipy: installed")

    # dcm2niix
    dcm = shutil.which("dcm2niix")
    if dcm:
        toolchain["dcm2niix"] = dcm
        _log(f"dcm2niix: found at {dcm}")
    elif _ask_yn("dcm2niix not found. Attempt to install via conda-forge?"):
        subprocess.run(
            ["conda", "install", "-c", "conda-forge", "dcm2niix", "-y"],
            check=False,
        )
        dcm = shutil.which("dcm2niix")
        toolchain["dcm2niix"] = dcm
        _log(f"dcm2niix: {'installed at ' + dcm if dcm else 'install failed — install manually'}")

    # Docker (for fMRIPrep / qsiPrep)
    if _ask_yn("Enable Docker containers (fMRIPrep, qsiPrep)?"):
        features["docker_containers"] = True
        _log("Docker containers: enabled")
    else:
        features["docker_containers"] = False

    # MATLAB / SPM
    if _ask_yn("Configure MATLAB / SPM? (requires MATLAB license)"):
        matlab = shutil.which("matlab") or ""
        path = _ask("matlab executable path", matlab)
        toolchain["matlab_path"] = path or None
        features["matlab_spm"] = True
        _log(f"MATLAB: {path}")
    else:
        features["matlab_spm"] = False

    return toolchain, features


# ── Step 5: LLM backend ────────────────────────────────────────────────────────
def _setup_llm() -> dict:
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 4 — LLM Backend")
    print("=" * 60)
    print("  1. OpenAI API               → enter API key")
    print("  2. OpenAI-compatible API    → custom baseUrl (Azure, Groq, DeepSeek, …)")
    print("  3. Anthropic                → enter API key")
    print("  4. Local model              → Ollama endpoint or llama.cpp path")
    choice = _ask("Choose [1/2/3/4]", "1")

    llm: dict = {
        "provider": None,
        "model": None,
        "api_key_env": None,
        "base_url": None,
        "local_endpoint": None,
    }

    if choice == "3":
        llm["provider"] = "anthropic"
        llm["model"] = _ask("Model name", "claude-3-5-sonnet-20241022")
        llm["api_key_env"] = "ANTHROPIC_API_KEY"
        key = _ask("Paste API key to set it now (or press Enter to set it later via env var)")
        # Anthropic keys begin with "sk-ant-"; accept any non-empty input that looks like a key
        if key and key.startswith("sk-ant-"):
            os.environ["ANTHROPIC_API_KEY"] = key
            _log("Anthropic API key stored in environment (not persisted to disk).")
        elif key:
            _log(
                "Input does not look like an Anthropic API key (expected 'sk-ant-...'). "
                "Set ANTHROPIC_API_KEY manually before starting NeuroClaw."
            )
        _log(f"LLM: Anthropic {llm['model']}")

    elif choice == "4":
        llm["provider"] = "local"
        llm["model"] = _ask("Model name / tag (e.g. llama3:8b)", "llama3:8b")
        endpoint = _ask("Ollama / llama.cpp HTTP endpoint", "http://localhost:11434")
        llm["local_endpoint"] = endpoint
        _log(f"LLM: local model {llm['model']} at {endpoint}")

    elif choice == "2":
        # OpenAI-compatible provider (Azure, Groq, DeepSeek, Together, OpenRouter, …)
        llm["provider"] = "openai"
        llm["model"] = _ask("Model name (as required by the provider)", "")
        base_url = _ask("Base URL (e.g. https://api.deepseek.com/v1)", "")
        llm["base_url"] = base_url or None
        env_var = _ask("Environment variable name for the API key", "OPENAI_API_KEY")
        llm["api_key_env"] = env_var
        key = _ask("Paste API key to set it now (or press Enter to set it later via env var)", "")
        if key:
            os.environ[env_var] = key
            _log(f"API key stored in environment variable {env_var} (not persisted to disk).")
        _log(f"LLM: OpenAI-compatible {llm['model']} at {base_url}")

    else:  # default: OpenAI
        llm["provider"] = "openai"
        llm["model"] = _ask("Model name", "gpt-4o")
        llm["api_key_env"] = "OPENAI_API_KEY"
        key = _ask("Paste API key to set it now (or press Enter to set it later via env var)")
        # OpenAI keys begin with "sk-"
        if key and key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = key
            _log("OpenAI API key stored in environment (not persisted to disk).")
        elif key:
            _log(
                "Input does not look like an OpenAI API key (expected 'sk-...'). "
                "Set OPENAI_API_KEY manually before starting NeuroClaw."
            )
        _log(f"LLM: OpenAI {llm['model']}")

    return llm


# ── Step 6: Neuro defaults ─────────────────────────────────────────────────────
def _setup_neuro_defaults() -> dict:
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 5 — Neuroscience Defaults")
    print("=" * 60)
    bids_root = _ask("Default BIDS data root", "~/data/bids")
    output_root = _ask("Default output root", "~/data/outputs")
    n_jobs_raw = _ask("Default parallel jobs (n_jobs)", "4")
    try:
        n_jobs = int(n_jobs_raw)
    except ValueError:
        n_jobs = 4
    _log(f"Neuro defaults: bids={bids_root}, output={output_root}, n_jobs={n_jobs}")
    return {"bids_root": bids_root, "output_root": output_root, "n_jobs": n_jobs}


# ── Step 7: Web UI dependencies ────────────────────────────────────────────────
def _setup_webui(python_path: str) -> None:
    """
    Optionally install Web UI + attachment parser dependencies.

    Parameters
    ----------
    python_path : str
        Full path to the Python executable to install packages into.
    """
    print("\n" + "=" * 60)
    print("[NeuroClaw Setup] Step 6 — Browser Web UI (optional)")
    print("=" * 60)
    print(
        "  NeuroClaw can serve a browser-based chat interface at\n"
        "  http://localhost:7080  (start with: python core/agent/main.py --web)\n"
    )

    required_modules = ["fastapi", "uvicorn", "pypdf", "docx", "openpyxl", "pptx"]
    if all(_check_importable(python_path, module) for module in required_modules):
        _log(
            "Web UI dependencies (fastapi, uvicorn, pypdf, python-docx, openpyxl, python-pptx) "
            "already installed — skipping."
        )
        return

    if not _ask_yn(
        "Install Web UI dependencies (fastapi + uvicorn + attachment parsers)?",
        default=True,
    ):
        _log(
            "Skipped Web UI dependencies. Run  pip install 'fastapi[standard]' uvicorn "
            "pypdf python-docx openpyxl python-pptx  to install later."
        )
        return

    cmd = [
        python_path,
        "-m",
        "pip",
        "install",
        "fastapi[standard]",
        "uvicorn",
        "pypdf",
        "python-docx",
        "openpyxl",
        "python-pptx",
    ]
    _log("Installing: " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode == 0:
        _log("Web UI dependencies installed successfully (including attachment parsers).")
    else:
        _log(
            "WARNING: Web UI installation failed. Install manually with:\n"
            "    pip install 'fastapi[standard]' uvicorn pypdf python-docx openpyxl python-pptx"
        )


def _check_importable(python_path: str, module: str) -> bool:
    """Return True if *module* can be imported by *python_path*."""
    result = subprocess.run(
        [python_path, "-c", f"import {module}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


# ── Write outputs ──────────────────────────────────────────────────────────────
def _write_env_file(config: dict) -> None:
    ENV_FILE.write_text(json.dumps(config, indent=2) + "\n")
    _log(f"Written: {ENV_FILE}")


def _update_features(toolchain_features: dict) -> None:
    if not FEATURES_FILE.exists():
        _log(f"WARNING: {FEATURES_FILE} not found — skipping features update.")
        return
    with FEATURES_FILE.open() as f:
        features = json.load(f)
    for key, enabled in toolchain_features.items():
        if key in features.get("neuroscience", {}):
            features["neuroscience"][key]["enabled"] = enabled
        elif key in features.get("connectors", {}):
            features["connectors"][key]["enabled"] = enabled
    FEATURES_FILE.write_text(json.dumps(features, indent=2) + "\n")
    _log(f"Updated: {FEATURES_FILE}")


def _write_log() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("w") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\nInstall log saved to: {LOG_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════╗
    ║         NeuroClaw Setup — Self-Contained Installer       ║
    ║  Neuroscience-first AI assistant, no OpenClaw required   ║
    ╚══════════════════════════════════════════════════════════╝
    This wizard configures your Python environment, CUDA, neuroimaging
    toolchain, and LLM backend, then writes neuroclaw_environment.json
    so every NeuroClaw session picks up your settings automatically.
    """))

    snap = _system_snapshot()

    python_cfg = _setup_python(snap)
    # Resolve the selected Python executable; used by later steps for pip installs
    selected_python = python_cfg.get("python_path") or sys.executable

    cuda_cfg = _setup_cuda(snap, python_path=selected_python)
    toolchain_cfg, toolchain_features = _setup_toolchain(snap, python_path=selected_python)
    llm_cfg = _setup_llm()
    neuro_cfg = _setup_neuro_defaults()
    _setup_webui(python_path=selected_python)

    config = {
        "setup_type": python_cfg["setup_type"],
        "python_path": selected_python,
        "conda_env": python_cfg.get("conda_env"),
        "docker_config": python_cfg.get("docker_config"),
        "cuda": cuda_cfg,
        "toolchain": toolchain_cfg,
        "llm_backend": llm_cfg,
        "neuro_defaults": neuro_cfg,
    }

    _write_env_file(config)
    _update_features(toolchain_features)
    _write_log()

    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    Setup Complete!                       ║
    ╚══════════════════════════════════════════════════════════╝
    Environment saved to:  {ENV_FILE}
    Features config at:    {FEATURES_FILE}
    Install log at:        {LOG_FILE}

    To start NeuroClaw (interactive REPL):
        python core/agent/main.py

    To open the browser-based Web UI:
        python core/agent/main.py --web
    Then visit  http://localhost:7080  in your browser.
    """))


if __name__ == "__main__":
    main()
