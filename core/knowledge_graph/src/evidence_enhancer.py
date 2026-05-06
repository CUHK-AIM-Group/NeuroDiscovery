"""Evidence enhancement: upgrade claim evidence from full-text papers.

Given a Claim (extracted from abstract) and the full paper text,
re-extract evidence fields (p_value, effect_size, sample_size, etc.)
with higher accuracy from the complete article.

Usage:
    from core.knowledge_graph.src.evidence_enhancer import EvidenceEnhancer

    enhancer = EvidenceEnhancer()
    new_evidence = enhancer.enhance(claim, full_text)
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import OpenAI

from .claim_extractor import ClaimExtractor
from .schema import Claim, Evidence

logger = logging.getLogger(__name__)

# ── LLM Configuration ──────────────────────────────────────────────

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

ENHANCEMENT_PROMPT = """You are enhancing evidence for a neuroscience claim using the full paper text.

CLAIM:
- Subject: {subject_name}
- Predicate: {predicate}
- Object: {object_name}
- Original raw sentence: {raw_sentence}
- Current evidence: study_type={study_type}, p_value={p_value}, effect_size={effect_size}, effect_metric={effect_metric}, sample_size={sample_size}

FULL PAPER TEXT:
{full_text}

Extract the BEST evidence for this specific claim from the full text. Return a JSON object:
{{
    "study_type": "the study type (fMRI, PET, DTI, sMRI, EEG, MEG, lesion, meta_analysis, GWAS, animal_model, clinical_trial, case_control, longitudinal, cross_sectional, review, cohort, narrative_review)",
    "methodology": "specific methodology used",
    "p_value": the exact p-value number (e.g. 0.003), or comparison string like "p < 0.05", or "not_reported",
    "effect_size": the effect size number (e.g. 0.45), or "not_reported",
    "effect_metric": the metric name (e.g. "Cohen's d", "odds ratio", "AUC"), or "not_reported",
    "sample_size": the sample size integer (e.g. 150), or "not_reported",
    "replicability": "replicated", "single_study", or "controversial",
    "direction": "positive", "negative", or "not_reported"
}}

RULES:
- Focus ONLY on evidence that supports or contradicts the specific claim above
- If the full text provides more precise statistics than the abstract, use those
- If the paper reports multiple studies, use the one most relevant to this claim
- NEVER output null — use "not_reported" instead
- Output ONLY the JSON object, nothing else."""


@dataclass
class EnhancementResult:
    """Result of evidence enhancement for a single claim."""
    claim_id: str
    new_evidence: Optional[Evidence] = None
    confidence_delta: float = 0.0  # change in confidence after enhancement
    raw_response: str = ""
    error: str = ""


class EvidenceEnhancer:
    """Enhance claim evidence using full paper text."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False),
        )
        self.model = model
        self._extractor = ClaimExtractor(base_url, api_key, model)

    def enhance(
        self,
        claim: Claim,
        full_text: str,
        overwrite: bool = False,
    ) -> EnhancementResult:
        """Enhance evidence for a single claim using full paper text.

        Args:
            claim: The claim to enhance.
            full_text: Full text of the source paper.
            overwrite: If True, overwrite all fields. If False, only fill
                       "not_reported" fields (preserves existing data).

        Returns:
            EnhancementResult with the new Evidence.
        """
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."

        prompt = ENHANCEMENT_PROMPT.format(
            subject_name=claim.subject_name,
            predicate=claim.predicate,
            object_name=claim.object_name,
            raw_sentence=claim.raw_text[:300],
            study_type=claim.evidence.study_type or "not_reported",
            p_value=claim.evidence.p_value if claim.evidence.p_value is not None else "not_reported",
            effect_size=claim.evidence.effect_size if claim.evidence.effect_size is not None else "not_reported",
            effect_metric=claim.evidence.effect_metric or "not_reported",
            sample_size=claim.evidence.sample_size if claim.evidence.sample_size is not None else "not_reported",
            full_text=full_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise neuroscience evidence extraction system. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            raw_text = response.choices[0].message.content.strip()
            new_evidence = self._parse_evidence(raw_text)

            if new_evidence is None:
                return EnhancementResult(
                    claim_id=claim.id,
                    error="failed to parse LLM response",
                    raw_response=raw_text,
                )

            # merge: only overwrite if overwrite=True or field was "not_reported"
            if not overwrite:
                new_evidence = self._merge_evidence(claim.evidence, new_evidence)

            # compute confidence delta
            old_conf = claim.confidence
            new_conf = self._extractor._estimate_confidence(new_evidence)

            return EnhancementResult(
                claim_id=claim.id,
                new_evidence=new_evidence,
                confidence_delta=new_conf - old_conf,
                raw_response=raw_text,
            )

        except Exception as e:
            logger.error(f"evidence enhancement failed for claim {claim.id}: {e}")
            return EnhancementResult(claim_id=claim.id, error=str(e))

    def enhance_batch(
        self,
        claims_and_texts: list[tuple[Claim, str]],
        max_workers: int = 1,
        overwrite: bool = False,
    ) -> list[EnhancementResult]:
        """Enhance evidence for multiple claims.

        Args:
            claims_and_texts: List of (Claim, full_text) tuples.
            max_workers: Number of parallel workers.
            overwrite: If True, overwrite all fields.

        Returns:
            List of EnhancementResult (same order as input).
        """
        if max_workers <= 1:
            results = []
            for i, (claim, text) in enumerate(claims_and_texts):
                logger.info(f"enhancing [{i+1}/{len(claims_and_texts)}] claim={claim.id}")
                result = self.enhance(claim, text, overwrite)
                logger.info(f"  confidence_delta={result.confidence_delta:+.2f}")
                results.append(result)
            return results

        results: list[Optional[EnhancementResult]] = [None] * len(claims_and_texts)

        def _enhance_one(idx: int, claim: Claim, text: str) -> tuple[int, EnhancementResult]:
            logger.info(f"enhancing [{idx+1}/{len(claims_and_texts)}] claim={claim.id}")
            result = self.enhance(claim, text, overwrite)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_enhance_one, i, claim, text)
                for i, (claim, text) in enumerate(claims_and_texts)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _parse_evidence(self, raw_text: str) -> Optional[Evidence]:
        """Parse LLM JSON response into an Evidence object."""
        json_str = self._extractor._extract_json(raw_text)
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # parse numeric fields using the same logic as claim_extractor
        effect_size, _ = self._extractor._parse_numeric(data.get("effect_size"), "float")
        p_value, _ = self._extractor._parse_numeric(data.get("p_value"), "float")
        sample_size, _ = self._extractor._parse_numeric(data.get("sample_size"), "int")

        effect_metric = data.get("effect_metric", "")
        if isinstance(effect_metric, str) and effect_metric.lower() in ("not_reported", "n/a", ""):
            effect_metric = ""

        return Evidence(
            study_type=data.get("study_type", ""),
            methodology=data.get("methodology", ""),
            p_value=p_value,
            effect_size=effect_size,
            effect_metric=effect_metric,
            sample_size=sample_size,
            replicability=data.get("replicability", "single_study"),
            direction=data.get("direction", ""),
        )

    @staticmethod
    def _merge_evidence(old: Evidence, new: Evidence) -> Evidence:
        """Merge new evidence into old, only filling 'not_reported' fields.

        Preserves existing non-null values from old evidence.
        """
        return Evidence(
            study_type=old.study_type or new.study_type,
            methodology=old.methodology or new.methodology,
            p_value=old.p_value if old.p_value is not None else new.p_value,
            effect_size=old.effect_size if old.effect_size is not None else new.effect_size,
            effect_metric=old.effect_metric or new.effect_metric,
            sample_size=old.sample_size if old.sample_size is not None else new.sample_size,
            replicability=old.replicability if old.replicability != "single_study" else new.replicability,
            direction=old.direction or new.direction,
        )


def enhance_and_update(
    kg: "KnowledgeGraph",
    claim_id: str,
    full_text: str,
    enhancer: Optional[EvidenceEnhancer] = None,
    overwrite: bool = False,
) -> EnhancementResult:
    """Convenience: enhance a claim's evidence and update it in the graph.

    Args:
        kg: The knowledge graph containing the claim.
        claim_id: ID of the claim to enhance.
        full_text: Full text of the source paper.
        enhancer: Optional pre-configured EvidenceEnhancer instance.
        overwrite: If True, overwrite all evidence fields.

    Returns:
        EnhancementResult with the new evidence and confidence delta.
    """
    from .graph_manager import KnowledgeGraph  # avoid circular import

    claim = kg.get_claim(claim_id)
    if claim is None:
        return EnhancementResult(claim_id=claim_id, error=f"claim {claim_id} not found in graph")

    if enhancer is None:
        enhancer = EvidenceEnhancer()

    result = enhancer.enhance(claim, full_text, overwrite)

    if result.new_evidence is not None:
        new_conf = claim.confidence + result.confidence_delta
        kg.update_claim(
            claim_id,
            new_evidence=result.new_evidence,
            new_confidence=max(0.0, min(1.0, new_conf)),
        )

    return result
