"""Phase 3.4: Critic Agent for hypothesis quality review.

Reviews hypotheses across 6 dimensions using LLM, with iterative
refinement loop (max 3 rounds).

Usage:
    python -m core.knowledge_graph.phase3 critic --input data/hypotheses_improved.json --top 10
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional

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
    """LLM-based hypothesis reviewer with iterative refinement."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        max_rounds: int = 3,
        pass_threshold: float = 0.6,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_rounds = max_rounds
        self.pass_threshold = pass_threshold

    def review(self, hypothesis: Hypothesis) -> CriticResult:
        """Review a single hypothesis across 6 dimensions."""
        prompt = self._build_review_prompt(hypothesis)
        raw = self._call_llm(prompt)
        feedbacks = self._parse_review_response(raw)
        overall_score = sum(f.score for f in feedbacks) / max(len(feedbacks), 1)
        fails = [f for f in feedbacks if f.verdict == "fail"]

        if not fails:
            verdict = "pass"
        elif overall_score >= self.pass_threshold and any(f.verdict == "fail" for f in feedbacks):
            verdict = "revise"
        elif overall_score < self.pass_threshold:
            verdict = "fail"
        else:
            verdict = "pass"

        return CriticResult(
            hypothesis_id=hypothesis.id,
            round=1,
            overall_verdict=verdict,
            overall_score=overall_score,
            feedbacks=feedbacks,
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
        """Iterative refinement: review → revise → re-review (max rounds).

        Returns (final_hypothesis, all_review_results).
        """
        all_results = []
        current = hypothesis

        for round_num in range(1, self.max_rounds + 1):
            result = self.review(current)
            result.round = round_num
            all_results.append(result)

            if result.overall_verdict == "pass":
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
                current = revised
                result.revised_hypothesis = revised

        return current, all_results

    def refine_batch(
        self,
        hypotheses: list[Hypothesis],
        max_workers: int = 2,
    ) -> list[tuple[Hypothesis, list[CriticResult]]]:
        """Refine multiple hypotheses. Uses fewer workers due to multi-round LLM calls."""
        results = []
        for i, h in enumerate(hypotheses):
            logger.info(f"refining {i+1}/{len(hypotheses)}: {h.id}")
            try:
                final, rounds = self.refine_loop(h)
                results.append((final, rounds))
            except Exception as e:
                logger.error(f"failed to refine {h.id}: {e}")
                h.critic_score = 0.0
                h.critic_rounds = 0
                results.append((h, []))
        return results

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

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise neuroscience research reviewer. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _extract_json(text: str):
        """Extract JSON from LLM response (handles code blocks, etc.)."""
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
