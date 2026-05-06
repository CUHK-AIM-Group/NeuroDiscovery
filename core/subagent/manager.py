"""SubagentManager: lifecycle management for independent agent sessions.

Each subagent is an independent AgentSession with its own conversation history,
session ID, and LLM client. Communication between parent and subagent uses
thread-safe queues.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

# Module-level lock for sys.path modifications (thread-safety)
_SYS_PATH_LOCK = threading.Lock()
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubagentHandle:
    """Internal handle tracking a single subagent session."""
    session_id: str
    session: Any  # AgentSession
    thread: threading.Thread
    mode: str  # "run_and_return" | "fire_and_forget" | "interactive"
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    input_queue: queue.Queue = field(default_factory=queue.Queue)
    created_at: float = field(default_factory=time.time)
    persona: str = ""
    task: str = ""


@dataclass
class SubagentResult:
    """Result returned by a completed subagent session."""
    session_id: str
    status: str
    response: str
    history: list[dict]
    error: str = ""


class SubagentManager:
    """Manages lifecycle of subagent sessions spawned by a parent agent.

    Usage:
        manager = SubagentManager(env, workspace)
        sid = manager.spawn("analyze fMRI data", skills_filter=["fmri-skill"])
        result = manager.get_result(sid)
        print(result.response)
    """

    def __init__(
        self,
        env: dict,
        workspace: Path,
        max_concurrent: int = 4,
    ) -> None:
        self._env = env
        self._workspace = workspace
        self._max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._agents: dict[str, SubagentHandle] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        task: str,
        *,
        persona: str = "",
        skills_filter: list[str] | None = None,
        mode: str = "run_and_return",
    ) -> str:
        """Spawn a new subagent session. Returns session_id.

        Args:
            task: The task description for the subagent to execute.
            persona: Optional expert persona name (e.g. "biostatistician").
            skills_filter: Optional list of skill names to restrict the subagent to.
            mode: "run_and_return" waits for result, "fire_and_forget" returns immediately.

        Returns:
            session_id string.
        """
        session_id = uuid.uuid4().hex[:12]

        handle = SubagentHandle(
            session_id=session_id,
            session=None,
            thread=None,
            mode=mode,
            status="pending",
            persona=persona,
            task=task,
        )

        with self._lock:
            self._agents[session_id] = handle

        thread = threading.Thread(
            target=self._run_subagent,
            args=(session_id, task, persona, skills_filter),
            daemon=True,
            name=f"subagent-{session_id}",
        )
        handle.thread = thread
        thread.start()

        if mode == "fire_and_forget":
            return session_id

        return session_id

    def get_result(self, session_id: str, timeout: float = 180.0) -> SubagentResult:
        """Wait for and return the result of a subagent session."""
        with self._lock:
            handle = self._agents.get(session_id)
        if handle is None:
            return SubagentResult(
                session_id=session_id,
                status="not_found",
                response="",
                history=[],
                error=f"Subagent {session_id} not found",
            )

        try:
            result = handle.result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return SubagentResult(
                session_id=session_id,
                status="timeout",
                response="",
                history=handle.session.history if handle.session else [],
                error=f"Subagent {session_id} timed out after {timeout}s",
            )

    def send_message(self, session_id: str, message: str) -> None:
        """Send a message to an interactive-mode subagent."""
        with self._lock:
            handle = self._agents.get(session_id)
        if handle is not None:
            handle.input_queue.put(message)

    def cancel(self, session_id: str) -> None:
        """Cancel a running subagent."""
        with self._lock:
            handle = self._agents.get(session_id)
        if handle is not None:
            handle.status = "cancelled"
            # Put a sentinel in the result queue only if no result yet
            if handle.result_queue.empty():
                handle.result_queue.put(SubagentResult(
                    session_id=session_id,
                    status="cancelled",
                    response="",
                    history=handle.session.history if handle.session else [],
                    error="Cancelled by parent",
                ))

    def list_active(self) -> list[dict]:
        """List all tracked subagent sessions."""
        with self._lock:
            return [
                {
                    "session_id": h.session_id,
                    "status": h.status,
                    "mode": h.mode,
                    "persona": h.persona,
                    "task": h.task[:100],
                    "created_at": h.created_at,
                }
                for h in self._agents.values()
            ]

    def shutdown_all(self) -> None:
        """Cancel and clean up all subagent sessions."""
        with self._lock:
            for handle in self._agents.values():
                if handle.status in ("pending", "running"):
                    handle.status = "cancelled"
                    # Enqueue cancellation result only if no result yet
                    if handle.result_queue.empty():
                        handle.result_queue.put(SubagentResult(
                            session_id=handle.session_id,
                            status="cancelled",
                            response="",
                            history=handle.session.history if handle.session else [],
                            error="Cancelled by parent (shutdown)",
                        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_subagent(
        self,
        session_id: str,
        task: str,
        persona: str,
        skills_filter: list[str] | None,
    ) -> None:
        """Thread target: create an AgentSession and execute the task."""
        with self._lock:
            handle = self._agents[session_id]
        handle.status = "running"

        try:
            self._semaphore.acquire()
            self._execute_subagent(handle, task, persona, skills_filter)
        except Exception as exc:
            logger.error(f"subagent {session_id} failed: {exc}")
            handle.status = "failed"
            handle.result_queue.put(SubagentResult(
                session_id=session_id,
                status="failed",
                response="",
                history=handle.session.history if handle.session else [],
                error=str(exc),
            ))
        finally:
            self._semaphore.release()

    def _execute_subagent(
        self,
        handle: SubagentHandle,
        task: str,
        persona: str,
        skills_filter: list[str] | None,
    ) -> None:
        """Build and run an AgentSession for the given task."""
        repo_root = self._workspace
        with _SYS_PATH_LOCK:
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

        from core.agent.main import AgentSession, build_llm_client, _load_skill_loader_class

        session = AgentSession(workspace=self._workspace, benchmark_mode=False)

        # Override env with the parent's env
        session.env = dict(self._env)

        # Build LLM client
        session.set_llm_client(build_llm_client(session.env))

        # Load skills
        SkillLoader = _load_skill_loader_class()
        loader = SkillLoader(self._workspace / "skills")
        skills = loader.load_all()

        # Apply skills filter
        if skills_filter:
            filter_set = set(skills_filter)
            skills = [s for s in skills if s.get("name", "") in filter_set]

        session.skills = skills

        # Build system prompt with optional persona prefix
        system_prompt = session._build_system_prompt(skills)
        if persona:
            persona_prefix = self._build_persona_prefix(persona)
            system_prompt = persona_prefix + "\n\n" + system_prompt

        # Initialize history
        session.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        handle.session = session

        # Check if already cancelled before executing
        if handle.status == "cancelled":
            return

        # Execute
        response = session._chat()

        # Only enqueue result if not cancelled during execution
        if handle.status != "cancelled":
            handle.status = "completed"
            handle.result_queue.put(SubagentResult(
                session_id=handle.session_id,
                status="completed",
                response=response,
                history=session.history,
            ))

    @staticmethod
    def _build_persona_prefix(persona: str) -> str:
        """Build a persona-specific prefix for the system prompt."""
        # Aliases: PERSPECTIVES keys from critic_agent.py -> full persona names
        _aliases = {
            "statistical": "biostatistician",
            "clinical": "clinical_neuroscientist",
            "methodological": "methodology_expert",
        }
        resolved = _aliases.get(persona, persona)

        personas = {
            "biostatistician": (
                "You are a biostatistician specializing in neuroscience research. "
                "Focus on statistical evidence: p-values, sample sizes, effect sizes, "
                "confidence intervals, multiple comparison corrections. "
                "Flag overclaimed significance."
            ),
            "clinical_neuroscientist": (
                "You are a clinical neuroscientist. "
                "Focus on biological plausibility: molecular mechanisms, disease pathways, "
                "clinical translation feasibility. Flag biologically implausible connections."
            ),
            "methodology_expert": (
                "You are a research methodology expert. "
                "Focus on study design: causal inference validity, confounding control, "
                "selection bias, measurement validity. "
                "Flag correlational claims presented as causal."
            ),
        }
        return personas.get(resolved, f"You are {persona}.")
