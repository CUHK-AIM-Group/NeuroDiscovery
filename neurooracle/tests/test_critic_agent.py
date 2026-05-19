"""Tests for the modified CriticAgent with independent agent support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurooracle.src.critic_agent import (
    CriticAgent,
    CriticFeedback,
    CriticResult,
    DIMENSIONS,
    PERSPECTIVES,
)
from neurooracle.src.hypothesis_engine import Hypothesis, HypothesisLink


def _make_sample_hypothesis() -> Hypothesis:
    """Create a sample hypothesis for testing."""
    return Hypothesis(
        id="test-hyp-001",
        hypothesis_type="path",
        source_id="hippocampus",
        source_name="Hippocampus",
        target_id="alzheimer",
        target_name="Alzheimer's Disease",
        path=[
            HypothesisLink(
                from_id="C0019572",
                from_name="Hippocampus",
                to_id="C0002395",
                to_name="Alzheimer's Disease",
                relation_type="is_biomarker_of",
                confidence=0.8,
                claim_id="claim-001",
                raw_text="Hippocampal volume reduction is a biomarker of AD progression.",
                evidence={"study_type": "longitudinal", "p_value": 0.001},
                source_paper={"pmid": "12345678", "year": 2020},
            )
        ],
        confidence_score=0.75,
        novelty_score=0.6,
        evidence_score=0.8,
        testability_score=0.9,
        composite_score=0.75,
        explanation="Hippocampal volume predicts AD conversion.",
        testability_reason="Testable with sMRI volumetry.",
    )


class TestCriticAgentBackwardCompatible:
    """Verify that use_independent_agents=False preserves original behavior."""

    @patch("openai.OpenAI")
    def test_default_init_no_agents(self, mock_openai):
        agent = CriticAgent(api_key="test-key")
        assert agent.use_independent_agents is False
        assert agent._persona_agents == {}

    @patch("openai.OpenAI")
    @patch.object(CriticAgent, "_call_llm")
    def test_review_uses_call_llm(self, mock_call_llm, mock_openai):
        """When use_independent_agents=False, review() calls _call_llm."""
        # Mock LLM response: 6 dimension reviews
        mock_response = """[
            {"dimension": "predicate_precision", "verdict": "pass", "score": 0.8, "issue": "", "suggestion": ""},
            {"dimension": "evidence_sufficiency", "verdict": "pass", "score": 0.7, "issue": "", "suggestion": ""},
            {"dimension": "causal_validity", "verdict": "pass", "score": 0.7, "issue": "", "suggestion": ""},
            {"dimension": "domain_coherence", "verdict": "pass", "score": 0.8, "issue": "", "suggestion": ""},
            {"dimension": "testability", "verdict": "pass", "score": 0.9, "issue": "", "suggestion": ""},
            {"dimension": "novelty_justification", "verdict": "pass", "score": 0.6, "issue": "", "suggestion": ""}
        ]"""
        mock_call_llm.return_value = mock_response

        agent = CriticAgent(api_key="test-key")
        hyp = _make_sample_hypothesis()
        result = agent.review(hyp)

        # Should have called _call_llm 3 times (once per perspective)
        assert mock_call_llm.call_count == 3
        assert isinstance(result, CriticResult)
        assert result.hypothesis_id == "test-hyp-001"

    @patch("openai.OpenAI")
    def test_init_with_missing_env_falls_back(self, mock_openai):
        """When env/workspace missing, should fall back to stateless mode."""
        agent = CriticAgent(api_key="test-key", use_independent_agents=True)
        assert agent.use_independent_agents is False  # fell back

    @patch("openai.OpenAI")
    def test_init_with_env_creates_agents(self, mock_openai):
        """When env/workspace provided, should create persona agents."""
        mock_env = {"llm_backend": {"provider": "openai", "model": "test"}}
        with patch("core.subagent.persona_agent.PersonaAgent._init_session"):
            agent = CriticAgent(
                api_key="test-key",
                env=mock_env,
                workspace=Path("/tmp"),
                use_independent_agents=True,
            )
            assert agent.use_independent_agents is True
            assert len(agent._persona_agents) == 3


class TestCriticAgentWithIndependentAgents:
    """Tests for the independent agent mode."""

    @patch("openai.OpenAI")
    @patch("core.subagent.persona_agent.PersonaAgent._init_session")
    @patch("core.subagent.persona_agent.PersonaAgent.discuss")
    def test_review_uses_persona_agents(self, mock_discuss, mock_init, mock_openai):
        """When use_independent_agents=True, review() uses PersonaAgents."""
        mock_response = """[
            {"dimension": "predicate_precision", "verdict": "pass", "score": 0.8, "issue": "", "suggestion": ""},
            {"dimension": "evidence_sufficiency", "verdict": "pass", "score": 0.7, "issue": "", "suggestion": ""},
            {"dimension": "causal_validity", "verdict": "pass", "score": 0.7, "issue": "", "suggestion": ""},
            {"dimension": "domain_coherence", "verdict": "pass", "score": 0.8, "issue": "", "suggestion": ""},
            {"dimension": "testability", "verdict": "pass", "score": 0.9, "issue": "", "suggestion": ""},
            {"dimension": "novelty_justification", "verdict": "pass", "score": 0.6, "issue": "", "suggestion": ""}
        ]"""
        mock_discuss.return_value = mock_response

        mock_env = {"llm_backend": {"provider": "openai", "model": "test"}}
        agent = CriticAgent(
            api_key="test-key",
            env=mock_env,
            workspace=Path("/tmp"),
            use_independent_agents=True,
        )

        # Mock the persona agents
        for name in agent._persona_agents:
            agent._persona_agents[name].discuss = MagicMock(return_value=mock_response)

        hyp = _make_sample_hypothesis()
        result = agent.review(hyp)

        # Should have called discuss on each persona agent
        for name, mock_agent in agent._persona_agents.items():
            mock_agent.discuss.assert_called_once()

        assert isinstance(result, CriticResult)
        assert result.hypothesis_id == "test-hyp-001"


class TestCriticFeedback:
    """Tests for CriticFeedback dataclass."""

    def test_to_dict(self):
        fb = CriticFeedback(
            dimension="testability",
            verdict="fail",
            score=0.3,
            issue="too vague",
            suggestion="be more specific",
        )
        d = fb.to_dict()
        assert d["dimension"] == "testability"
        assert d["verdict"] == "fail"
        assert d["score"] == 0.3

    def test_from_dict(self):
        d = {
            "dimension": "testability",
            "verdict": "pass",
            "score": 0.8,
            "issue": "",
            "suggestion": "",
        }
        fb = CriticFeedback.from_dict(d)
        assert fb.dimension == "testability"
        assert fb.verdict == "pass"


class TestCriticResult:
    """Tests for CriticResult dataclass."""

    def test_to_dict(self):
        result = CriticResult(
            hypothesis_id="h1",
            round=1,
            overall_verdict="pass",
            overall_score=0.8,
        )
        d = result.to_dict()
        assert d["hypothesis_id"] == "h1"
        assert d["overall_verdict"] == "pass"
        assert d["feedbacks"] == []
