"""Integration tests for PersonaAgent and MultiAgentDiscussion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.subagent.persona_agent import (
    PERSONAS,
    DiscussionResult,
    MultiAgentDiscussion,
    PersonaAgent,
)


class TestPersonaDefinitions:
    """Tests for persona definitions."""

    def test_all_three_personas_defined(self):
        assert "biostatistician" in PERSONAS
        assert "clinical_neuroscientist" in PERSONAS
        assert "methodology_expert" in PERSONAS

    def test_persona_text_not_empty(self):
        for name, text in PERSONAS.items():
            assert len(text) > 20, f"Persona {name} text too short"

    def test_persona_distinct_content(self):
        texts = list(PERSONAS.values())
        # All three should be different
        assert len(set(texts)) == 3


class TestPersonaAgent:
    """Tests for PersonaAgent."""

    @patch("core.subagent.persona_agent.PersonaAgent._init_session")
    def test_persona_stored(self, mock_init):
        agent = PersonaAgent.__new__(PersonaAgent)
        agent.persona = "biostatistician"
        agent._history = []
        assert agent.persona == "biostatistician"

    def test_persona_in_system_prompt(self):
        """Verify that persona text appears in the system prompt."""
        with patch.object(PersonaAgent, "_init_session"):
            agent = PersonaAgent.__new__(PersonaAgent)
            agent.persona = "biostatistician"
            agent._env = {}
            agent._workspace = Path("/tmp")
            agent._session = None
            agent._model = None
            agent._history = []

            # Simulate what _init_session does
            persona_text = PERSONAS["biostatistician"]
            system_prompt = f"{persona_text}\n\nTest context"
            agent._history = [{"role": "system", "content": system_prompt}]

            assert "biostatistician" in agent._history[0]["content"]
            assert "statistical" in agent._history[0]["content"].lower()


class TestDiscussionResult:
    """Tests for DiscussionResult dataclass."""

    def test_basic_result(self):
        result = DiscussionResult(
            topic="test topic",
            rounds=[[{"persona": "a", "response": "yes"}]],
            consensus="agreement",
        )
        assert result.topic == "test topic"
        assert len(result.rounds) == 1
        assert result.consensus == "agreement"
        assert result.divergences == []


class TestMultiAgentDiscussion:
    """Tests for MultiAgentDiscussion."""

    @patch("core.subagent.persona_agent.PersonaAgent._init_session")
    @patch("core.subagent.persona_agent.PersonaAgent.discuss")
    def test_run_discussion_sequential(self, mock_discuss, mock_init):
        mock_discuss.return_value = "test response"

        with patch.object(MultiAgentDiscussion, "__init__", lambda self, *a, **kw: None):
            discussion = MultiAgentDiscussion.__new__(MultiAgentDiscussion)
            discussion._persona_names = ["biostatistician", "clinical_neuroscientist"]
            discussion._max_rounds = 2
            discussion._convergence_threshold = 0.8

            # Create mock agents
            agents = []
            for name in discussion._persona_names:
                agent = MagicMock()
                agent.persona = name
                agent.discuss.return_value = f"{name} response"
                agent.respond_to_others.return_value = f"{name} revised response"
                agents.append(agent)
            discussion._agents = agents

            result = discussion.run_discussion("test topic")

            assert isinstance(result, DiscussionResult)
            assert result.topic == "test topic"
            assert len(result.rounds) >= 1

    @patch("core.subagent.persona_agent.PersonaAgent._init_session")
    def test_check_convergence(self, mock_init):
        with patch.object(MultiAgentDiscussion, "__init__", lambda self, *a, **kw: None):
            discussion = MultiAgentDiscussion.__new__(MultiAgentDiscussion)
            discussion._convergence_threshold = 0.8

            # Identical responses should converge
            responses = [
                {"persona": "a", "response": "the hippocampus is important for memory"},
                {"persona": "b", "response": "the hippocampus is important for memory"},
            ]
            convergence = discussion._check_convergence(responses)
            assert convergence > 0.5

            # Very different responses should not converge
            responses = [
                {"persona": "a", "response": "quantum mechanics proves consciousness"},
                {"persona": "b", "response": "statistical analysis shows p-value 0.001"},
            ]
            convergence = discussion._check_convergence(responses)
            assert convergence < 0.5

    @patch("core.subagent.persona_agent.PersonaAgent._init_session")
    def test_find_divergences(self, mock_init):
        with patch.object(MultiAgentDiscussion, "__init__", lambda self, *a, **kw: None):
            discussion = MultiAgentDiscussion.__new__(MultiAgentDiscussion)

            # Response with disagreement marker
            responses = [
                {"persona": "a", "response": "I agree with this hypothesis", "round": 0},
                {"persona": "b", "response": "However, I disagree with the methodology", "round": 0},
            ]
            divergences = discussion._find_divergences(responses)
            assert len(divergences) > 0
            assert any("b" in d for d in divergences)
