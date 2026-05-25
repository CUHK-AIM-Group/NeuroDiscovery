"""Phase 3.4: Critic Agent for hypothesis quality review.

Reviews hypotheses across 6 dimensions using LLM, with iterative
refinement loop (max 3 rounds).

Usage:
    python -m neurooracle.phase3 critic --input data/hypotheses_improved.json --top 10
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .hypothesis_engine import Hypothesis, HypothesisLink

logger = logging.getLogger(__name__)

# LLM config (same as claim_extractor.py)
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

DIMENSIONS = [
    "predicate_precision",
    "evidence_sufficiency",
    "causal_validity",
    "domain_coherence",
    "testability",
    "novelty_justification",
]

# ── Three-Perspective Critic Debate ────────────────────────────────────

PERSPECTIVES = {
    "statistical": (
        "You are a biostatistician. Focus on statistical evidence: "
        "p-values, sample sizes, effect sizes, confidence intervals, "
        "multiple comparison corrections. Flag overclaimed significance."
    ),
    "clinical": (
        "You are a clinical neuroscientist. Focus on biological plausibility: "
        "molecular mechanisms, disease pathways, clinical translation feasibility. "
        "Flag biologically implausible connections."
    ),
    "methodological": (
        "You are a research methodology expert. Focus on study design: "
        "causal inference validity, confounding control, selection bias, "
        "measurement validity. Flag correlational claims presented as causal."
    ),
}

# ── Transitivity Diagnostics ───────────────────────────────────────────

PREDICATE_CONFLICT: dict[tuple[str, str], str] = {
    ("inhibits", "activates"): "inhibits and activates are contradictory",
    ("activates", "inhibits"): "activates and inhibits are contradictory",
    ("reduces", "increases"): "reduces and increases are contradictory",
    ("increases", "reduces"): "increases and reduces are contradictory",
    ("causes", "reduces"): "causes and reduces on the same node is contradictory",
    ("reduces", "causes"): "reduces and causes on the same node is contradictory",
    ("treats", "causes"): "treats and causes on the same node is contradictory",
    ("causes", "treats"): "causes and treats on the same node is contradictory",
}


@dataclass
class TransitivityViolation:
    """A transitivity violation between adjacent steps."""
    step: int  # 0-indexed, the first of the two adjacent steps
    type: str  # "node_mismatch", "semantic_incoherence", "predicate_conflict"
    detail: str = ""


@dataclass
class CriticFeedback:
    """Single dimension review result."""
    dimension: str
    verdict: str  # "pass" | "fail"
    score: float  # 0-1
    issue: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CriticFeedback:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CriticResult:
    """Full review result for one hypothesis in one round."""
    hypothesis_id: str
    round: int
    overall_verdict: str  # "pass" | "fail" | "revise"
    overall_score: float
    feedbacks: list[CriticFeedback] = field(default_factory=list)
    revised_hypothesis: Optional[Hypothesis] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feedbacks"] = [f.to_dict() for f in self.feedbacks]
        if self.revised_hypothesis:
            d["revised_hypothesis"] = self.revised_hypothesis.to_dict()
        return d


class CriticAgent:
    """LLM-based hypothesis reviewer with iterative refinement.

    When use_independent_agents=True, each perspective runs as an independent
    PersonaAgent with its own conversation history, enabling cross-round memory.
    When use_independent_agents=False (default), each perspective is a single
    stateless LLM call (original behavior).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        max_rounds: int = 3,
        pass_threshold: float = 0.6,
        env: dict | None = None,
        workspace: Path | None = None,
        use_independent_agents: bool = False,
    ):
        # Multi-key round-robin pool support
        import httpx
        keys_raw = os.environ.get("OPENAI_API_KEYS", "")
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if not keys and api_key:
            keys = [api_key]

        self._clients = [
            OpenAI(base_url=base_url, api_key=k, http_client=httpx.Client(verify=False))
            for k in keys
        ]
        self._client_idx = 0
        self._client_lock = __import__("threading").Lock()
        self.model = model
        self.max_rounds = max_rounds
        self.pass_threshold = pass_threshold
        self.use_independent_agents = use_independent_agents
        self._env = env
        self._workspace = workspace
        self._persona_agents: dict[str, Any] = {}  # name -> PersonaAgent

        if use_independent_agents:
            self._init_persona_agents()

    @property
    def client(self) -> OpenAI:
        """Round-robin client selection from key pool (thread-safe)."""
        with self._client_lock:
            c = self._clients[self._client_idx % len(self._clients)]
            self._client_idx += 1
            return c

    def _init_persona_agents(self) -> None:
        """Create independent PersonaAgent instances for each perspective."""
        from core.subagent.persona_agent import PersonaAgent

        if self._env is None or self._workspace is None:
            logger.warning(
                "Cannot init persona agents: env and workspace required. "
                "Falling back to stateless mode."
            )
            self.use_independent_agents = False
            return

        for name in PERSPECTIVES:
            persona_name = name  # "statistical", "clinical", "methodological"
            agent = PersonaAgent(persona_name, self._env, self._workspace)
            self._persona_agents[name] = agent
        logger.info(f"Initialized {len(self._persona_agents)} independent persona agents")

    def review(self, hypothesis: Hypothesis) -> CriticResult:
        """Review a single hypothesis using three-perspective debate.

        Three critics (statistical, clinical, methodological) independently
        review the hypothesis. Final verdict is the intersection:
        - All 3 PASS → PASS
        - 2 PASS + 1 FAIL → WARNING (pass with note)
        - 2+ FAIL → FAIL
        Methodological critic has highest weight in conflicts.
        """
        prompt = self._build_review_prompt(hypothesis)
        perspective_results: dict[str, list[CriticFeedback]] = {}

        def _call_and_parse(name: str, sys_prompt: str, use_agent: bool) -> list[CriticFeedback]:
            if use_agent:
                agent = self._persona_agents.get(name)
                raw = agent.discuss(prompt) if agent is not None else self._call_llm(prompt, system_prompt=sys_prompt)
            else:
                raw = self._call_llm(prompt, system_prompt=sys_prompt)
            if not raw or not raw.strip() or self._extract_json(raw) is None:
                logger.info(
                    f"empty/unparseable response for {hypothesis.id} ({name}), retrying once"
                )
                raw = self._call_llm(prompt, system_prompt=sys_prompt)
            return self._parse_review_response(raw)

        if self.use_independent_agents and self._persona_agents:
            for name in PERSPECTIVES:
                perspective_results[name] = _call_and_parse(name, PERSPECTIVES[name], use_agent=True)
        else:
            for name, sys_prompt in PERSPECTIVES.items():
                perspective_results[name] = _call_and_parse(name, sys_prompt, use_agent=False)

        # Aggregate: take the average score per dimension across perspectives
        aggregated: list[CriticFeedback] = []
        for dim in DIMENSIONS:
            dim_feedbacks = []
            for name in PERSPECTIVES:
                for f in perspective_results[name]:
                    if f.dimension == dim:
                        dim_feedbacks.append(f)
            if not dim_feedbacks:
                aggregated.append(CriticFeedback(dim, "pass", 0.5, "", ""))
                continue

            avg_score = sum(f.score for f in dim_feedbacks) / len(dim_feedbacks)
            # Majority verdict for this dimension
            passes = sum(1 for f in dim_feedbacks if f.verdict == "pass")
            verdict = "pass" if passes >= 2 else "fail"
            # Collect issues from failing perspectives
            issues = [f"{name}: {f.issue}" for name, f in zip(PERSPECTIVES, dim_feedbacks) if f.verdict == "fail" and f.issue]
            suggestions = [f"{name}: {f.suggestion}" for name, f in zip(PERSPECTIVES, dim_feedbacks) if f.verdict == "fail" and f.suggestion]
            aggregated.append(CriticFeedback(
                dimension=dim,
                verdict=verdict,
                score=avg_score,
                issue=" | ".join(issues),
                suggestion=" | ".join(suggestions),
            ))

        # Overall verdict: count how many perspectives have overall pass
        perspective_verdicts = {}
        for name, feedbacks in perspective_results.items():
            overall_score = sum(f.score for f in feedbacks) / max(len(feedbacks), 1)
            fails = [f for f in feedbacks if f.verdict == "fail"]
            if not fails:
                perspective_verdicts[name] = "pass"
            elif overall_score < self.pass_threshold:
                perspective_verdicts[name] = "fail"
            else:
                perspective_verdicts[name] = "revise"

        pass_count = sum(1 for v in perspective_verdicts.values() if v == "pass")
        fail_count = sum(1 for v in perspective_verdicts.values() if v == "fail")

        overall_score = sum(f.score for f in aggregated) / max(len(aggregated), 1)

        if pass_count >= 3:
            verdict = "pass"
        elif pass_count >= 2 and fail_count == 0:
            verdict = "pass"  # 2 pass + 1 revise
        elif fail_count >= 2:
            verdict = "fail"
        elif fail_count == 1 and perspective_verdicts.get("methodological") == "fail":
            verdict = "fail"  # methodological has highest weight
        else:
            verdict = "revise"

        return CriticResult(
            hypothesis_id=hypothesis.id,
            round=1,
            overall_verdict=verdict,
            overall_score=overall_score,
            feedbacks=aggregated,
        )

    def review_batch(
        self, hypotheses: list[Hypothesis], max_workers: int = 4
    ) -> list[CriticResult]:
        """Review multiple hypotheses in parallel."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.review, h): h for h in hypotheses}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"reviewed {result.hypothesis_id}: "
                        f"{result.overall_verdict} ({result.overall_score:.2f})"
                    )
                except Exception as e:
                    h = futures[future]
                    logger.error(f"failed to review {h.id}: {e}")
        return results

    def refine_loop(
        self, hypothesis: Hypothesis
    ) -> tuple[Hypothesis, list[CriticResult]]:
        """Iterative refinement: review → transitivity check → revise → re-review.

        After passing the 3-perspective critic, runs transitivity diagnostics.
        If transitivity violations are found, the hypothesis is sent back for
        revision with the violation details as additional feedback.

        Returns (final_hypothesis, all_review_results).
        """
        all_results = []
        current = hypothesis

        for round_num in range(1, self.max_rounds + 1):
            result = self.review(current)
            result.round = round_num
            all_results.append(result)

            # Sync review result to persona agents for cross-round memory
            if self.use_independent_agents and self._persona_agents:
                self._sync_review_to_agents(current, result)

            if result.overall_verdict == "pass":
                # Transitivity diagnostics after passing critic
                violations = self.check_transitivity(current, use_llm=True)
                if violations:
                    # Convert violations to feedback and force revise
                    trans_feedbacks = [
                        CriticFeedback(
                            dimension="domain_coherence",
                            verdict="fail",
                            score=0.3,
                            issue=f"transitivity [{v.type}]: {v.detail}",
                            suggestion="fix the semantic inconsistency between adjacent steps",
                        )
                        for v in violations
                    ]
                    result.feedbacks.extend(trans_feedbacks)
                    result.overall_verdict = "revise"
                    result.overall_score = (
                        sum(f.score for f in result.feedbacks)
                        / max(len(result.feedbacks), 1)
                    )
                    logger.info(
                        f"{current.id} transitivity violations found "
                        f"(round {round_num}): {[v.type for v in violations]}"
                    )
                    # fall through to revision below
                else:
                    current.critic_score = result.overall_score
                    current.critic_feedback = [f.to_dict() for f in result.feedbacks]
                    current.critic_rounds = round_num
                    logger.info(f"{current.id} PASSED round {round_num} ({result.overall_score:.2f})")
                    return current, all_results

            if round_num == self.max_rounds:
                current.critic_score = result.overall_score
                current.critic_feedback = [f.to_dict() for f in result.feedbacks]
                current.critic_rounds = round_num
                status = "PASSED" if result.overall_score >= self.pass_threshold else "FAILED"
                logger.info(f"{current.id} {status} after {round_num} rounds ({result.overall_score:.2f})")
                return current, all_results

            # revise
            logger.info(f"{current.id} REVISE round {round_num} ({result.overall_score:.2f})")
            revised = self._revise(current, result.feedbacks)
            if revised:
                # Sync revision to persona agents
                if self.use_independent_agents and self._persona_agents:
                    self._sync_revision_to_agents(current, revised, result.feedbacks)
                current = revised
                result.revised_hypothesis = revised

        return current, all_results

    def refine_batch(
        self,
        hypotheses: list[Hypothesis],
        max_workers: int = 4,
    ) -> list[tuple[Hypothesis, list[CriticResult]]]:
        """Refine multiple hypotheses in parallel. Uses ThreadPoolExecutor."""
        results = []

        def refine_one(i: int, h: Hypothesis):
            logger.info(f"refining {i+1}/{len(hypotheses)}: {h.id}")
            try:
                final, rounds = self.refine_loop(h)
                return (i, final, rounds)
            except Exception as e:
                logger.error(f"failed to refine {h.id}: {e}")
                h.critic_score = 0.0
                h.critic_rounds = 0
                return (i, h, [])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(refine_one, i, h): i for i, h in enumerate(hypotheses)}
            completed = []
            for future in as_completed(futures):
                completed.append(future.result())

        # Sort by original index to maintain order
        completed.sort(key=lambda x: x[0])
        results = [(final, rounds) for _, final, rounds in completed]
        return results

    # ── transitivity diagnostics ───────────────────────────────────────

    def check_transitivity(
        self, hypothesis: Hypothesis, use_llm: bool = True
    ) -> list[TransitivityViolation]:
        """Check semantic coherence between adjacent steps in the path.

        Three checks per adjacent pair:
        1. Node continuity: step_a.to_name == step_b.from_name
        2. Semantic coherence: raw_texts of two steps support a continuous chain
        3. Predicate conflict: contradictory predicates (e.g., inhibits + activates)
        """
        violations: list[TransitivityViolation] = []
        path = hypothesis.path

        for i in range(len(path) - 1):
            step_a = path[i]
            step_b = path[i + 1]

            # Check 1: node continuity
            if step_a.to_id != step_b.from_id:
                violations.append(TransitivityViolation(
                    step=i, type="node_mismatch",
                    detail=f"node mismatch: {step_a.to_name} vs {step_b.from_name}",
                ))

            # Check 2: predicate conflict (rule-based, no LLM)
            conflict_key = (step_a.relation_type, step_b.relation_type)
            if conflict_key in PREDICATE_CONFLICT:
                violations.append(TransitivityViolation(
                    step=i, type="predicate_conflict",
                    detail=PREDICATE_CONFLICT[conflict_key],
                ))

            # Check 3: semantic coherence via LLM (only if raw_texts available)
            if use_llm and step_a.raw_text and step_b.raw_text:
                coherence = self._llm_check_coherence(step_a, step_b)
                if coherence < 0.3:
                    violations.append(TransitivityViolation(
                        step=i, type="semantic_incoherence",
                        detail=(
                            f"Step {i+1} and Step {i+2} raw_texts are semantically "
                            f"incoherent (score={coherence:.2f})"
                        ),
                    ))

        return violations

    def _llm_check_coherence(
        self, step_a: HypothesisLink, step_b: HypothesisLink
    ) -> float:
        """Use LLM to judge if two adjacent steps form a coherent chain.

        Returns a score 0.0 (incoherent) to 1.0 (coherent).
        """
        prompt = f"""Rate the semantic coherence of these two adjacent reasoning steps.

Step 1: {step_a.from_name} --[{step_a.relation_type}]--> {step_a.to_name}
Evidence: {step_a.raw_text[:300]}

Step 2: {step_b.from_name} --[{step_b.relation_type}]--> {step_b.to_name}
Evidence: {step_b.raw_text[:300]}

Do these two steps form a logically coherent chain? Does the evidence in Step 1 relate to the claim in Step 2?

Output a JSON object: {{"coherence": 0.0-1.0, "reason": "brief explanation"}}
- 1.0 = fully coherent, the evidence directly supports the chain
- 0.5 = partially related but with gaps
- 0.0 = completely unrelated or contradictory

Output ONLY the JSON object."""

        try:
            raw = self._call_llm(prompt)
            data = self._extract_json(raw)
            if isinstance(data, dict) and "coherence" in data:
                return float(data["coherence"])
        except Exception as e:
            logger.debug(f"coherence check failed: {e}")
        return 0.5  # default: uncertain

    # ── Persona agent sync ──────────────────────────────────────────────────

    def _sync_review_to_agents(
        self, hypothesis: Hypothesis, result: CriticResult
    ) -> None:
        """Sync review result to persona agents for cross-round memory."""
        feedback_summary = []
        for f in result.feedbacks:
            if f.verdict == "fail":
                feedback_summary.append(f"- {f.dimension}: {f.issue}")
        feedback_text = "\n".join(feedback_summary) if feedback_summary else "No issues found."

        sync_message = (
            f"## Review Result for {hypothesis.id}\n"
            f"Verdict: {result.overall_verdict} (score: {result.overall_score:.2f})\n"
            f"Round: {result.round}\n\n"
            f"### Feedback\n{feedback_text}"
        )

        for name, agent in self._persona_agents.items():
            try:
                agent._history.append({"role": "user", "content": sync_message})
                agent._history.append({
                    "role": "assistant",
                    "content": f"Noted. I will consider this feedback in my next review.",
                })
            except Exception as exc:
                logger.debug(f"failed to sync review to {name}: {exc}")

    def _sync_revision_to_agents(
        self,
        original: Hypothesis,
        revised: Hypothesis,
        feedbacks: list[CriticFeedback],
    ) -> None:
        """Sync revision info to persona agents."""
        revision_message = (
            f"## Hypothesis Revised\n"
            f"The hypothesis {original.id} was revised based on feedback.\n"
            f"Original scores: confidence={original.confidence_score:.2f}, "
            f"novelty={original.novelty_score:.2f}\n"
            f"Revised scores: confidence={revised.confidence_score:.2f}, "
            f"novelty={revised.novelty_score:.2f}\n"
            f"Please consider the revised version in your next review."
        )

        for name, agent in self._persona_agents.items():
            try:
                agent._history.append({"role": "user", "content": revision_message})
                agent._history.append({
                    "role": "assistant",
                    "content": "Noted. I will review the revised hypothesis.",
                })
            except Exception as exc:
                logger.debug(f"failed to sync revision to {name}: {exc}")

    def _build_review_prompt(self, hypothesis: Hypothesis) -> str:
        """Build the LLM review prompt."""
        path_lines = []
        for i, link in enumerate(hypothesis.path):
            ev = link.evidence or {}
            line = f"  Step {i+1}: {link.from_name} --[{link.relation_type}]--> {link.to_name}"
            if ev.get("study_type"):
                line += f" (study: {ev['study_type']})"
            if link.raw_text:
                line += f"\n    Evidence: {link.raw_text[:200]}"
            if link.source_paper.get("pmid"):
                line += f"\n    Source: PMID {link.source_paper['pmid']} ({link.source_paper.get('year', '')})"
            path_lines.append(line)

        path_str = "\n".join(path_lines)

        return f"""You are a neuroscience research hypothesis reviewer. Review the following hypothesis across 6 dimensions.

## Hypothesis
ID: {hypothesis.id}
Type: {hypothesis.hypothesis_type}
Source: {hypothesis.source_name}
Target: {hypothesis.target_name}

## Path
{path_str}

## Current Scores
- confidence: {hypothesis.confidence_score:.2f}
- novelty: {hypothesis.novelty_score:.2f}
- evidence: {hypothesis.evidence_score:.2f}
- testability: {hypothesis.testability_score:.2f}

## Testability Info
{hypothesis.testability_reason}

## Review Dimensions

For each dimension, evaluate and output:
1. **predicate_precision**: Are the relation types specific enough? (e.g., `is_associated_with` is too vague)
2. **evidence_sufficiency**: Does each hop have supporting citations with raw text?
3. **causal_validity**: Is causality overclaimed from correlational data?
4. **domain_coherence**: Do entity types match the domain pair?
5. **testability**: Can this be tested with imaging modalities (sMRI, fMRI, dMRI, PET)?
6. **novelty_justification**: Is the novelty claim reasonable?

Output a JSON array with exactly 6 objects:
```json
[
  {{"dimension": "predicate_precision", "verdict": "pass|fail", "score": 0.0-1.0, "issue": "", "suggestion": ""}},
  ...
]
```

- verdict "pass" if the dimension meets quality standards, "fail" if not
- score: 0.0 (worst) to 1.0 (best)
- issue: describe the problem (only if fail)
- suggestion: how to fix (only if fail)

Output ONLY the JSON array, no other text."""

    def _build_revision_prompt(
        self, hypothesis: Hypothesis, feedbacks: list[CriticFeedback]
    ) -> str:
        """Build the revision prompt."""
        feedback_lines = []
        for f in feedbacks:
            if f.verdict == "fail":
                feedback_lines.append(
                    f"- {f.dimension}: {f.issue}\n  Suggestion: {f.suggestion}"
                )

        feedback_str = "\n".join(feedback_lines) if feedback_lines else "No critical issues."

        path_lines = []
        for i, link in enumerate(hypothesis.path):
            line = f"  {link.from_name} --[{link.relation_type}]--> {link.to_name}"
            path_lines.append(line)

        path_str = "\n".join(path_lines)

        return f"""The following hypothesis was reviewed and found deficient. Revise it based on the feedback.

## Original Hypothesis
Source: {hypothesis.source_name} → Target: {hypothesis.target_name}
Path:
{path_str}
Current scores: confidence={hypothesis.confidence_score:.2f}, novelty={hypothesis.novelty_score:.2f}, evidence={hypothesis.evidence_score:.2f}, testability={hypothesis.testability_score:.2f}

## Issues Found
{feedback_str}

## Instructions
Based on the feedback, provide revised scores and explanation. Output JSON:
```json
{{
  "confidence_score": 0.0-1.0,
  "novelty_score": 0.0-1.0,
  "evidence_score": 0.0-1.0,
  "testability_score": 0.0-1.0,
  "explanation": "revised explanation addressing the issues"
}}
```

If the hypothesis cannot be salvaged, set all scores to 0.0.
Output ONLY the JSON object, no other text."""

    def _revise(
        self, hypothesis: Hypothesis, feedbacks: list[CriticFeedback]
    ) -> Optional[Hypothesis]:
        """Ask LLM to revise the hypothesis based on feedback."""
        prompt = self._build_revision_prompt(hypothesis, feedbacks)
        raw = self._call_llm(prompt)
        return self._apply_revision(hypothesis, raw)

    def _apply_revision(
        self, hypothesis: Hypothesis, raw_response: str
    ) -> Optional[Hypothesis]:
        """Parse LLM revision and update hypothesis."""
        data = self._extract_json(raw_response)
        if not data or not isinstance(data, dict):
            logger.warning(f"failed to parse revision for {hypothesis.id}")
            return None

        revised = Hypothesis.from_dict(hypothesis.to_dict())
        revised.confidence_score = float(data.get("confidence_score", hypothesis.confidence_score))
        revised.novelty_score = float(data.get("novelty_score", hypothesis.novelty_score))
        revised.evidence_score = float(data.get("evidence_score", hypothesis.evidence_score))
        revised.testability_score = float(data.get("testability_score", hypothesis.testability_score))
        if "explanation" in data:
            revised.explanation = data["explanation"]

        # Recompute composite
        revised.composite_score = (
            revised.confidence_score ** 0.20
            * revised.evidence_score ** 0.20
            * revised.novelty_score ** 0.25
            * revised.testability_score ** 0.35
        )
        return revised

    def _parse_review_response(self, raw: str) -> list[CriticFeedback]:
        """Parse LLM review response into CriticFeedback list."""
        data = self._extract_json(raw)
        if not data or not isinstance(data, list):
            logger.warning(f"failed to parse review response, raw: {raw[:200]}")
            return [
                CriticFeedback(dim, "fail", 0.0, "parse_error", "")
                for dim in DIMENSIONS
            ]

        feedbacks = []
        for item in data:
            if not isinstance(item, dict):
                continue
            dim = item.get("dimension", "unknown")
            feedbacks.append(
                CriticFeedback(
                    dimension=dim,
                    verdict=item.get("verdict", "fail"),
                    score=float(item.get("score", 0.0)),
                    issue=item.get("issue", ""),
                    suggestion=item.get("suggestion", ""),
                )
            )

        # Ensure all 6 dimensions present
        found_dims = {f.dimension for f in feedbacks}
        for dim in DIMENSIONS:
            if dim not in found_dims:
                feedbacks.append(CriticFeedback(dim, "pass", 0.5, "", ""))

        return feedbacks

    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Call LLM API."""
        if not system_prompt:
            system_prompt = "You are a precise neuroscience research reviewer. Output only valid JSON."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _extract_json(text: str):
        """Extract JSON from LLM response (handles code blocks, think tags, etc.)."""
        # Strip <think> and </think> tags but KEEP their content
        # (reasoning models sometimes emit JSON across think tags incorrectly)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Try finding JSON array or object
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue

        return None
