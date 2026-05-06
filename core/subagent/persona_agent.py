"""PersonaAgent and MultiAgentDiscussion for persona-driven multi-agent discussions.

Each PersonaAgent wraps an independent AgentSession with a specific expert persona.
MultiAgentDiscussion orchestrates multi-round discussions between multiple personas.
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level lock for sys.path modifications (thread-safety)
_SYS_PATH_LOCK = threading.Lock()

# ── Persona definitions (extends PERSPECTIVES from critic_agent.py) ──────────

PERSONAS = {
    "biostatistician": (
        "You are a biostatistician specializing in neuroscience research. "
        "Focus on statistical evidence: p-values, sample sizes, effect sizes, "
        "confidence intervals, multiple comparison corrections. "
        "Flag overclaimed significance. Be precise and quantitative."
    ),
    "clinical_neuroscientist": (
        "You are a clinical neuroscientist. "
        "Focus on biological plausibility: molecular mechanisms, disease pathways, "
        "clinical translation feasibility. Flag biologically implausible connections. "
        "Think in terms of real patient outcomes and clinical relevance."
    ),
    "methodology_expert": (
        "You are a research methodology expert. "
        "Focus on study design: causal inference validity, confounding control, "
        "selection bias, measurement validity. "
        "Flag correlational claims presented as causal. "
        "Your judgment has the highest weight in conflicts."
    ),
}

# Aliases: PERSPECTIVES keys from critic_agent.py -> PERSONAS keys
_PERSONA_ALIASES = {
    "statistical": "biostatistician",
    "clinical": "clinical_neuroscientist",
    "methodological": "methodology_expert",
}


@dataclass
class DiscussionResult:
    """Result of a multi-agent discussion."""
    topic: str
    rounds: list[list[dict]]  # round -> list of {persona, response}
    consensus: str = ""
    divergences: list[str] = field(default_factory=list)
    all_responses: list[dict] = field(default_factory=list)


class PersonaAgent:
    """An agent with a specific expert persona for multi-agent discussion.

    Each PersonaAgent holds an independent AgentSession with its own
    conversation history, allowing cross-round memory in discussions.
    """

    def __init__(
        self,
        persona: str,
        env: dict,
        workspace: Path,
        *,
        model: str | None = None,
    ) -> None:
        self.persona = persona
        self._env = env
        self._workspace = workspace
        self._session: Any = None
        self._model = model
        self._history: list[dict] = []
        self._init_session()

    def _init_session(self) -> None:
        """Create an independent AgentSession with persona-specific system prompt."""
        with _SYS_PATH_LOCK:
            if str(self._workspace) not in sys.path:
                sys.path.insert(0, str(self._workspace))

        from core.agent.main import AgentSession, build_llm_client

        self._session = AgentSession(workspace=self._workspace, benchmark_mode=False)
        self._session.env = dict(self._env)
        self._session.set_llm_client(build_llm_client(self._session.env))

        # Build persona-specific system prompt (resolve aliases from PERSPECTIVES keys)
        resolved_name = _PERSONA_ALIASES.get(self.persona, self.persona)
        persona_text = PERSONAS.get(resolved_name, f"You are {self.persona}.")
        system_prompt = (
            f"{persona_text}\n\n"
            "You are participating in a multi-expert discussion about neuroscience "
            "research hypotheses. Respond concisely with your expert perspective. "
            "Focus on your area of expertise. Output only your analysis, no preamble."
        )
        self._history = [{"role": "system", "content": system_prompt}]
        self._session.history = self._history

    def discuss(self, topic: str, context: str = "") -> str:
        """Generate this persona's response to a discussion topic.

        Args:
            topic: The discussion topic or question.
            context: Additional context (e.g. hypothesis details).

        Returns:
            The persona's response text.
        """
        prompt = topic
        if context:
            prompt = f"{context}\n\n{topic}"

        self._history.append({"role": "user", "content": prompt})
        self._session.history = self._history

        response = self._session._chat()
        self._history.append({"role": "assistant", "content": response})
        return response

    def respond_to_others(
        self, topic: str, prior_responses: list[dict]
    ) -> str:
        """Generate a response aware of other agents' prior responses.

        Args:
            topic: The discussion topic.
            prior_responses: List of {persona, response} dicts from prior rounds.

        Returns:
            The persona's response text.
        """
        if not prior_responses:
            return self.discuss(topic)

        # Build context from prior responses
        prior_text_parts = []
        for r in prior_responses:
            persona = r.get("persona", "unknown")
            response = r.get("response", "")
            prior_text_parts.append(f"[{persona}]: {response}")
        prior_text = "\n\n".join(prior_text_parts)

        prompt = (
            f"## Discussion Topic\n{topic}\n\n"
            f"## Other Experts' Responses\n{prior_text}\n\n"
            "Based on the above, provide your expert perspective. "
            "You may agree, disagree, or add nuance. Be specific."
        )

        self._history.append({"role": "user", "content": prompt})
        self._session.history = self._history

        response = self._session._chat()
        self._history.append({"role": "assistant", "content": response})
        return response

    def get_history(self) -> list[dict]:
        """Return the full conversation history of this persona agent."""
        return list(self._history)


class MultiAgentDiscussion:
    """Orchestrates multi-agent discussion for knowledge graph research.

    Creates multiple PersonaAgent instances and runs multi-round discussions
    where each agent sees shared context and prior responses.
    """

    def __init__(
        self,
        env: dict,
        workspace: Path,
        personas: list[str] | None = None,
        max_rounds: int = 3,
        convergence_threshold: float = 0.8,
    ) -> None:
        self._env = env
        self._workspace = workspace
        self._max_rounds = max_rounds
        self._convergence_threshold = convergence_threshold

        if personas is None:
            personas = list(PERSONAS.keys())
        self._persona_names = personas

        # Create PersonaAgent instances
        self._agents: list[PersonaAgent] = []
        for name in personas:
            agent = PersonaAgent(name, env, workspace)
            self._agents.append(agent)

    def run_discussion(self, topic: str, context: str = "") -> DiscussionResult:
        """Run a multi-round discussion sequentially and return consensus.

        Each round, all agents see the shared context plus all prior round
        responses. Discussion stops when convergence is reached or max_rounds.

        Args:
            topic: The discussion topic.
            context: Additional context (e.g. hypothesis details).

        Returns:
            DiscussionResult with all rounds and consensus.
        """
        all_rounds: list[list[dict]] = []
        all_responses: list[dict] = []

        for round_num in range(self._max_rounds):
            round_responses: list[dict] = []

            for agent in self._agents:
                if round_num == 0 and not all_responses:
                    response = agent.discuss(topic, context=context)
                else:
                    response = agent.respond_to_others(topic, all_responses)

                entry = {"persona": agent.persona, "response": response, "round": round_num}
                round_responses.append(entry)
                all_responses.append(entry)

            all_rounds.append(round_responses)

            # Check convergence
            if len(all_rounds) >= 2:
                convergence = self._check_convergence(round_responses)
                if convergence >= self._convergence_threshold:
                    logger.info(f"discussion converged at round {round_num + 1} ({convergence:.2f})")
                    break

        # Generate consensus summary
        consensus = self._summarize_consensus(topic, all_responses)
        divergences = self._find_divergences(all_responses)

        return DiscussionResult(
            topic=topic,
            rounds=all_rounds,
            consensus=consensus,
            divergences=divergences,
            all_responses=all_responses,
        )

    def run_discussion_parallel(
        self, topic: str, context: str = ""
    ) -> DiscussionResult:
        """Run discussion with all agents responding in parallel per round.

        Same as run_discussion but uses ThreadPoolExecutor for parallel
        agent responses within each round.
        """
        all_rounds: list[list[dict]] = []
        all_responses: list[dict] = []

        for round_num in range(self._max_rounds):
            round_responses: list[dict] = [None] * len(self._agents)  # type: ignore

            with ThreadPoolExecutor(max_workers=len(self._agents)) as executor:
                futures = {}
                for i, agent in enumerate(self._agents):
                    if round_num == 0 and not all_responses:
                        future = executor.submit(agent.discuss, topic, context)
                    else:
                        future = executor.submit(
                            agent.respond_to_others, topic, all_responses
                        )
                    futures[future] = (i, agent.persona)

                for future in as_completed(futures):
                    idx, persona = futures[future]
                    try:
                        response = future.result()
                        entry = {"persona": persona, "response": response, "round": round_num}
                        round_responses[idx] = entry
                        all_responses.append(entry)
                    except Exception as exc:
                        logger.error(f"agent {persona} failed in round {round_num}: {exc}")
                        entry = {"persona": persona, "response": f"[error: {exc}]", "round": round_num}
                        round_responses[idx] = entry

            all_rounds.append(round_responses)

            if len(all_rounds) >= 2:
                convergence = self._check_convergence(round_responses)
                if convergence >= self._convergence_threshold:
                    logger.info(f"discussion converged at round {round_num + 1} ({convergence:.2f})")
                    break

        consensus = self._summarize_consensus(topic, all_responses)
        divergences = self._find_divergences(all_responses)

        return DiscussionResult(
            topic=topic,
            rounds=all_rounds,
            consensus=consensus,
            divergences=divergences,
            all_responses=all_responses,
        )

    def _check_convergence(self, round_responses: list[dict]) -> float:
        """Check if agents have reached consensus in this round.

        Simple heuristic: check if responses share similar key terms.
        Returns 0.0 (no convergence) to 1.0 (full convergence).
        """
        if len(round_responses) < 2:
            return 1.0

        responses = [r["response"] for r in round_responses if r and r.get("response")]
        if len(responses) < 2:
            return 0.0

        # Extract key terms from each response (simple word overlap)
        def extract_terms(text: str) -> set[str]:
            words = text.lower().split()
            # Keep words longer than 4 chars (more meaningful)
            return {w for w in words if len(w) > 4}

        term_sets = [extract_terms(r) for r in responses]
        if not any(term_sets):
            return 0.0

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                intersection = term_sets[i] & term_sets[j]
                union = term_sets[i] | term_sets[j]
                if union:
                    similarities.append(len(intersection) / len(union))

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _summarize_consensus(
        self, topic: str, all_responses: list[dict]
    ) -> str:
        """Generate a consensus summary from all responses."""
        if not all_responses:
            return ""

        # Collect the last round's responses
        last_round = max(r.get("round", 0) for r in all_responses)
        last_responses = [r for r in all_responses if r.get("round") == last_round]

        parts = [f"## Discussion Consensus on: {topic}\n"]
        for r in last_responses:
            parts.append(f"**{r['persona']}**: {r['response'][:300]}\n")

        return "\n".join(parts)

    def _find_divergences(self, all_responses: list[dict]) -> list[str]:
        """Identify key divergences between agents."""
        if len(all_responses) < 2:
            return []

        # Simple approach: look for negation patterns indicating disagreement
        divergences = []
        last_round = max(r.get("round", 0) for r in all_responses)
        last_responses = [r for r in all_responses if r.get("round") == last_round]

        disagree_markers = ["however", "but", "disagree", "contrary", "unlike", "相反", "但是", "不同意"]
        for r in last_responses:
            for marker in disagree_markers:
                if marker in r.get("response", "").lower():
                    divergences.append(
                        f"{r['persona']} expressed a dissenting view"
                    )
                    break

        return divergences
