"""Native framework controllers for the direct closed-loop CS1 protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from core.scripts.case_study_native_output import extract_json


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
OFFICIAL_PYTHON_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
NATIVE_SERVICE_METHODS = ("brainpilot_native", "biomni_native")
FRAMEWORK_METHODS = (*OFFICIAL_PYTHON_METHODS, *NATIVE_SERVICE_METHODS)
REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}


@dataclass(frozen=True)
class ControllerSelection:
    items: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]
    native_summary: str = ""


def _scrub(text: str, secret: str) -> str:
    cleaned = str(text).replace(secret, "<redacted>")
    if len(secret) >= 8:
        cleaned = cleaned.replace(secret[-8:], "<redacted-suffix>")
    return cleaned


def _repo_python(method: str, repo: Path) -> Path:
    env_names = (
        f"CASE_STUDY_{method.upper()}_PYTHON",
        f"CS1_{method.upper()}_PYTHON",
    )
    candidates = [
        *(Path(os.environ[name]) for name in env_names if os.environ.get(name)),
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"no Python runtime found for {method}")


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _selection_items_from_policy(
    policy: Mapping[str, Any], *, start_rank: int
) -> tuple[dict[str, Any], ...]:
    items = []
    for offset, anchor in enumerate(policy.get("anchors") or []):
        if not isinstance(anchor, Mapping):
            continue
        items.append(
            {
                "rank": start_rank + offset,
                "candidate_id": str(anchor.get("candidate_id") or ""),
                "rationale": str(anchor.get("rationale") or "native exact selection"),
                "confidence": float(anchor.get("score", 1.0)),
            }
        )
    return tuple(items)


def _selection_items_from_text(text: str) -> tuple[dict[str, Any], ...]:
    payload = extract_json(text)
    hypotheses = payload.get("hypotheses") or []
    return tuple(dict(item) for item in hypotheses if isinstance(item, Mapping))


def _brainpilot_tool_audit(events_path: Path) -> dict[str, Any]:
    if not events_path.is_file():
        return {"tool_calls": 0, "tool_errors": 0, "retrieval_tools": {}}
    events = json.loads(events_path.read_text(encoding="utf-8"))
    names_by_call: dict[str, str] = {}
    counts: dict[str, int] = {}
    errors: dict[str, int] = {}
    for event in events if isinstance(events, list) else ():
        if not isinstance(event, Mapping):
            continue
        call_id = str(event.get("tool_call_id") or "")
        if event.get("type") == "TOOL_CALL_START":
            name = str(event.get("tool_call_name") or "")
            if call_id and name:
                names_by_call[call_id] = name
                counts[name] = counts.get(name, 0) + 1
        elif event.get("type") == "TOOL_CALL_RESULT" and event.get("is_error"):
            name = names_by_call.get(call_id, "unknown")
            errors[name] = errors.get(name, 0) + 1
    retrieval = {
        name: count
        for name, count in counts.items()
        if name.startswith("mcp__")
        or name in {"get_domain_knowledge_local", "search_papers_local"}
    }
    return {
        "tool_calls": int(sum(counts.values())),
        "tool_errors": int(sum(errors.values())),
        "tool_calls_by_name": counts,
        "tool_errors_by_name": errors,
        "retrieval_tools": retrieval,
    }


def _biomni_tool_audit(execution_audit_path: Path) -> dict[str, Any]:
    if not execution_audit_path.is_file():
        return {
            "audit_source": "missing",
            "executed_blocks": 0,
            "pubmed_query_calls": 0,
            "pubmed_result_titles": 0,
            "execution_errors": 0,
        }
    payload = json.loads(execution_audit_path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {"audit_source": "invalid"}


class OfficialPythonController:
    """Run the upstream Python workflow once per feedback round."""

    def __init__(
        self,
        *,
        method: str,
        secret: str,
        model: str,
        base_url: str,
        reasoning_effort: str,
        workflow_timeout_seconds: int,
        request_timeout_seconds: int,
        max_retries: int,
        native_retrieval_enabled: bool,
    ) -> None:
        if method not in OFFICIAL_PYTHON_METHODS:
            raise ValueError(method)
        self.method = method
        self.secret = secret
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.workflow_timeout_seconds = int(workflow_timeout_seconds)
        self.request_timeout_seconds = int(request_timeout_seconds)
        self.max_retries = int(max_retries)
        self.native_retrieval_enabled = bool(native_retrieval_enabled)
        self.repo = (BASELINES_ROOT / REPOSITORIES[method]).resolve()
        self.python = _repo_python(method, self.repo)
        self.commit = _git_commit(self.repo)

    def select(
        self,
        *,
        task: dict[str, Any],
        round_dir: Path,
        prompt: str,
        start_rank: int,
    ) -> ControllerSelection:
        del prompt
        round_dir.mkdir(parents=True, exist_ok=True)
        task_path = round_dir / "task.json"
        task_path.write_text(
            json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        policy_path = round_dir / "search_policy.json"
        if not policy_path.is_file():
            command = [
                str(self.python),
                str(ROOT / "core/scripts/case_study_official_adapter_client.py"),
                "--method",
                self.method,
                "--task",
                str(task_path),
                "--out",
                str(round_dir),
                "--repo",
                str(self.repo),
                "--model",
                self.model,
                "--base-url",
                self.base_url,
                "--reasoning-effort",
                self.reasoning_effort,
            ]
            if self.native_retrieval_enabled:
                command.append("--enable-native-retrieval")
            env = os.environ.copy()
            env.update(
                {
                    "CASE_STUDY_LOCAL_API_KEY": self.secret,
                    "CS1_LOCAL_API_KEY": self.secret,
                    "OPENAI_API_KEY": self.secret,
                    "OPENAI_BASE_URL": self.base_url,
                    "OPENAI_API_BASE": self.base_url,
                    "OPENAI_REASONING_EFFORT": self.reasoning_effort,
                    "OPENAI_DISABLE_RESPONSE_STORAGE": "true",
                    "CASE_STUDY_LLM_REQUEST_TIMEOUT": str(
                        self.request_timeout_seconds
                    ),
                    "OPEN_COSCIENTIST_MAX_CONCURRENT_LLM_CALLS": "4",
                    "OPEN_COSCIENTIST_TRANSIENT_RETRIES": "4",
                    "OPEN_COSCIENTIST_MAX_OUTPUT_TOKENS": "6000",
                    "OPEN_COSCIENTIST_DEBATE_TURNS": "3",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            last_error = ""
            for attempt in range(1, self.max_retries + 1):
                started = time.time()
                returncode: int | None = None
                try:
                    result = subprocess.run(
                        command,
                        cwd=self.repo,
                        env=env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.workflow_timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    last_error = f"timeout: {exc}"
                    stdout = ""
                    stderr = last_error
                else:
                    stdout = result.stdout
                    stderr = result.stderr
                    returncode = result.returncode
                (round_dir / f"attempt_{attempt:02d}.stdout.log").write_text(
                    _scrub(stdout, self.secret), encoding="utf-8"
                )
                (round_dir / f"attempt_{attempt:02d}.stderr.log").write_text(
                    _scrub(stderr, self.secret), encoding="utf-8"
                )
                (round_dir / f"attempt_{attempt:02d}.duration.txt").write_text(
                    f"{time.time() - started:.3f}\n", encoding="utf-8"
                )
                if returncode == 0 and policy_path.is_file():
                    break
                if returncode is not None:
                    last_error = f"exit_code={returncode}"
                if attempt < self.max_retries:
                    time.sleep(min(20.0, 2.0**attempt))
            if not policy_path.is_file():
                raise RuntimeError(f"{self.method} failed: {last_error}")

        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        native_path = round_dir / "native_result.json"
        native_summary = (
            native_path.read_text(encoding="utf-8")[-12000:]
            if native_path.is_file()
            else ""
        )
        return ControllerSelection(
            items=_selection_items_from_policy(policy, start_rank=start_rank),
            metadata={
                "controller": "official_python_round_workflow",
                "repository": str(self.repo),
                "repository_commit": self.commit,
                "python": str(self.python),
                "native_retrieval_enabled": self.native_retrieval_enabled,
                "policy_metadata": dict(policy.get("metadata") or {}),
            },
            native_summary=native_summary,
        )

    def close(self) -> None:
        return None


class BrainPilotController:
    """Use one official BrainPilot session for every round of a seed."""

    def __init__(
        self,
        *,
        secret: str,
        model: str,
        reasoning_effort: str,
        service_url: str,
        timeout_seconds: int,
    ) -> None:
        self.method = "brainpilot_native"
        self.secret = secret
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.repo = (BASELINES_ROOT / REPOSITORIES[self.method]).resolve()
        self.commit = _git_commit(self.repo)
        self.session_id: str | None = None

    def select(
        self,
        *,
        task: dict[str, Any],
        round_dir: Path,
        prompt: str,
        start_rank: int,
    ) -> ControllerSelection:
        del start_rank
        round_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = round_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        final_path = round_dir / "final.txt"
        meta_path = round_dir / "client_meta.json"
        executed_now = False
        if not final_path.is_file():
            command = [
                "node",
                str(ROOT / "core/scripts/brainpilot_case_study_batch_client.mjs"),
                "--prompt",
                str(prompt_path),
                "--out",
                str(round_dir),
                "--client-dist",
                str(self.repo / "packages/client-cli/dist/index.js"),
                "--base-url",
                self.service_url,
                "--max-events",
                "5000",
            ]
            menu_path = str(task.get("public_registry_path") or "").strip()
            if menu_path:
                command.extend(["--menu", menu_path])
            if self.session_id:
                command.extend(["--session-id", self.session_id])
            env = os.environ.copy()
            env.update(
                {
                    "ANTHROPIC_API_KEY": self.secret,
                    "ANTHROPIC_MODEL": self.model,
                    "BP_THINKING_LEVEL": self.reasoning_effort,
                }
            )
            result = subprocess.run(
                command,
                cwd=self.repo,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            (round_dir / "launcher.stdout.log").write_text(
                _scrub(result.stdout, self.secret), encoding="utf-8"
            )
            (round_dir / "launcher.stderr.log").write_text(
                _scrub(result.stderr, self.secret), encoding="utf-8"
            )
            if result.returncode != 0 or not final_path.is_file():
                raise RuntimeError(
                    f"BrainPilot round failed with exit code {result.returncode}"
                )
            executed_now = True
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if executed_now:
                self.session_id = (
                    str(meta.get("session_id") or self.session_id or "") or None
                )
        text = final_path.read_text(encoding="utf-8")
        tool_audit = _brainpilot_tool_audit(round_dir / "events.json")
        return ControllerSelection(
            items=_selection_items_from_text(text),
            metadata={
                "controller": "official_brainpilot_persistent_session",
                "repository": str(self.repo),
                "repository_commit": self.commit,
                "session_id": self.session_id,
                "session_persistent": True,
                "tool_audit": tool_audit,
            },
            native_summary=text[-12000:],
        )

    def close(self) -> None:
        return None


class BiomniController:
    """Keep one official Biomni A1 instance and LangGraph thread per seed."""

    def __init__(
        self,
        *,
        secret: str,
        model: str,
        base_url: str,
        reasoning_effort: str,
        timeout_seconds: int,
        seed_dir: Path,
    ) -> None:
        self.method = "biomni_native"
        self.secret = secret
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = int(timeout_seconds)
        self.seed_dir = seed_dir
        self.repo = (BASELINES_ROOT / REPOSITORIES[self.method]).resolve()
        self.commit = _git_commit(self.repo)
        self.python = _repo_python(self.method, self.repo)
        self.process: subprocess.Popen[str] | None = None
        self.stderr_handle = None

    def _start(self) -> None:
        if self.process is not None:
            return
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = (self.seed_dir / "biomni_worker.stderr.log").open(
            "a", encoding="utf-8"
        )
        command = [
            str(self.python),
            str(ROOT / "core/scripts/biomni_case1_closed_loop_worker.py"),
            "--biomni-root",
            str(self.repo),
            "--workspace",
            str(self.seed_dir / "biomni_workspace"),
            "--out",
            str(self.seed_dir / "biomni_worker"),
            "--model",
            self.model,
            "--base-url",
            self.base_url,
            "--reasoning-effort",
            self.reasoning_effort,
        ]
        env = os.environ.copy()
        env.update(
            {
                "BIOMNI_API_KEY": self.secret,
                "OPENAI_DISABLE_RESPONSE_STORAGE": "true",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.repo,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready = self._readline_with_timeout()
        payload = json.loads(ready)
        if payload.get("type") != "ready":
            raise RuntimeError(f"Biomni worker did not become ready: {payload}")

    def _readline_with_timeout(self) -> str:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Biomni worker is not running")
        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                result_queue.put(self.process.stdout.readline())
            except BaseException as exc:  # Propagate reader failures to the caller.
                result_queue.put(exc)

        threading.Thread(target=read_line, daemon=True).start()
        try:
            result = result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Biomni worker did not respond within {self.timeout_seconds}s"
            ) from exc
        if isinstance(result, BaseException):
            raise RuntimeError("Biomni worker stdout reader failed") from result
        line = result
        if not line:
            raise RuntimeError("Biomni worker closed stdout")
        return line

    def select(
        self,
        *,
        task: dict[str, Any],
        round_dir: Path,
        prompt: str,
        start_rank: int,
    ) -> ControllerSelection:
        del task, start_rank
        self._start()
        assert self.process is not None and self.process.stdin is not None
        prompt = (
            f"{prompt}\n\nBIOMNI ACTION PROTOCOL\n"
            "Emit exactly one complete <execute>...</execute> block per response, "
            "then stop and wait for the real <observation> produced by Biomni. Never "
            "draft multiple execute blocks in one response. Before <solution>, execute "
            "at least one real query_pubmed call and inspect its observation.\n"
        )
        final_path = round_dir / "final.txt"
        if final_path.is_file():
            text = final_path.read_text(encoding="utf-8")
            resumed = True
            duration = None
        else:
            request_id = f"round-{round_dir.name}"
            request = {
                "type": "round",
                "request_id": request_id,
                "prompt": prompt,
                "out_dir": str(round_dir),
            }
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = json.loads(self._readline_with_timeout())
            if not response.get("ok") or response.get("request_id") != request_id:
                raise RuntimeError(f"Biomni worker round failed: {response}")
            text = str(response.get("final") or "")
            resumed = False
            duration = response.get("duration_seconds")
        return ControllerSelection(
            items=_selection_items_from_text(text),
            metadata={
                "controller": "official_biomni_persistent_a1",
                "repository": str(self.repo),
                "repository_commit": self.commit,
                "python": str(self.python),
                "persistent_agent": True,
                "tool_retriever_enabled": True,
                "resumed_from_round_artifact": resumed,
                "duration_seconds": duration,
                "tool_audit": _biomni_tool_audit(
                    round_dir / "biomni_execution_audit.json"
                ),
            },
            native_summary=text[-12000:],
        )

    def close(self) -> None:
        process = self.process
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=30)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.process = None
        if self.stderr_handle is not None:
            self.stderr_handle.close()
            self.stderr_handle = None


def make_framework_controller(
    method: str,
    *,
    secret: str,
    model: str,
    base_url: str,
    reasoning_effort: str,
    workflow_timeout_seconds: int,
    request_timeout_seconds: int,
    max_retries: int,
    native_retrieval_enabled: bool,
    brainpilot_url: str,
    seed_dir: Path,
):
    if method in OFFICIAL_PYTHON_METHODS:
        return OfficialPythonController(
            method=method,
            secret=secret,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            workflow_timeout_seconds=workflow_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            native_retrieval_enabled=native_retrieval_enabled,
        )
    if method == "brainpilot_native":
        return BrainPilotController(
            secret=secret,
            model=model,
            reasoning_effort=reasoning_effort,
            service_url=brainpilot_url,
            timeout_seconds=workflow_timeout_seconds,
        )
    if method == "biomni_native":
        return BiomniController(
            secret=secret,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout_seconds=workflow_timeout_seconds,
            seed_dir=seed_dir,
        )
    raise ValueError(f"unsupported autoresearch framework: {method}")


__all__ = [
    "BASELINES_ROOT",
    "BrainPilotController",
    "BiomniController",
    "ControllerSelection",
    "FRAMEWORK_METHODS",
    "NATIVE_SERVICE_METHODS",
    "OFFICIAL_PYTHON_METHODS",
    "OfficialPythonController",
    "REPOSITORIES",
    "make_framework_controller",
]
