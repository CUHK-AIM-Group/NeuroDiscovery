"""LLM-based structured claim extraction from neuroscience paper abstracts.

Uses GPT-5.5 via proxy endpoint to extract structured scientific claims
as (Subject, Predicate, Object, Evidence) triples.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import OpenAI

from .schema import Claim, Evidence, PaperRef

logger = logging.getLogger(__name__)

# ── LLM Configuration ──────────────────────────────────────────────

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

# Multi-key pool: set OPENAI_API_KEYS env var as comma-separated keys
# e.g. export OPENAI_API_KEYS="sk-aaa,sk-bbb,sk-ccc"
# Falls back to OPENAI_API_KEY if not set
_API_KEYS_RAW = os.environ.get("OPENAI_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in _API_KEYS_RAW.split(",") if k.strip()] or (
    [DEFAULT_API_KEY] if DEFAULT_API_KEY else []
)

EXTRACTION_PROMPT = """Extract ALL scientific claims from this neuroscience paper abstract as JSON array.

Each claim object fields:
- subject, subject_type, predicate, object, object_type, negated
- effect_metric, effect_size, p_value, sample_size
- study_type, methodology, replicability, direction, raw_sentence
- conditions: list of conditions under which this claim holds (e.g. ["female only", "age > 65", "resting-state fMRI"]). Empty list [] if unconditional.
- population: object with study population info, null if not reported:
  {{"mean_age": number or null, "age_range": "e.g. 18-65" or null, "n_female": int or null, "n_male": int or null, "ethnicity": str or null, "cohort_name": str or null}}

IMPORTANT rules for numeric fields:
- p_value: output the exact number if reported (e.g. 0.003), or a comparison string like "p < 0.05" or "p > 0.01", or "not_reported" if the abstract does not mention a p-value.
- effect_size: output the number (e.g. 0.45, 1.2), or "not_reported" if not mentioned.
- sample_size: output the integer (e.g. 150, 2048), or "not_reported" if not mentioned.
- effect_metric: output the metric name (e.g. "Cohen's d", "odds ratio", "AUC", "beta"), or "not_reported" if not mentioned.
- NEVER output null for these four fields — use "not_reported" instead.

Types: brain_region, disease, gene, neurotransmitter, protein, drug, network, biomarker, cognitive_function

Predicates (USE THE MOST SPECIFIC ONE):
- CAUSAL: causes, treats, inhibits, activates, increases, reduces, modulates
- PREDICTIVE: is_biomarker_of, is_risk_factor_for, predicts, distinguishes
- CORRELATIONAL: correlates_with, mediates
- VAGUE (AVOID): is_associated_with — ONLY use if the abstract text itself is vague (e.g., "X is associated with Y" without specifying direction or mechanism)

CRITICAL: Choose the most precise predicate based on the study design and language:
- RCT / intervention → "treats" or "causes"
- Longitudinal / prospective → "is_risk_factor_for" or "predicts"
- Diagnostic accuracy → "is_biomarker_of" or "distinguishes"
- Molecular mechanism → "activates", "inhibits", "increases", "reduces"
- Cross-sectional correlation → "correlates_with"
- If the abstract says "X increases Y" or "X reduces Y", use "increases" or "reduces", NOT "is_associated_with"

Study types: fMRI, PET, DTI, sMRI, EEG, MEG, lesion, meta_analysis, GWAS, animal_model, clinical_trial, case_control, longitudinal, cross_sectional, review, cohort, narrative_review

Title: {title}
PMID: {pmid}
Abstract: {abstract}

Return JSON array. Empty array [] if no claims."""


@dataclass
class ExtractionResult:
    """Result of claim extraction from a single paper."""
    paper: PaperRef
    claims: list[Claim]
    raw_response: str = ""
    error: str = ""


class ClaimExtractor:
    """Extract structured claims from paper abstracts using LLM.

    Supports multi-key round-robin to bypass per-key rate limits on proxy APIs.
    Set OPENAI_API_KEYS env var as comma-separated keys.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        api_keys: list[str] | None = None,
    ):
        self.model = model
        self.base_url = base_url

        # Build client pool from explicit keys, env pool, or single key
        keys = api_keys or API_KEY_POOL or ([api_key] if api_key else [])
        if not keys:
            raise ValueError("No API keys provided. Set OPENAI_API_KEYS or OPENAI_API_KEY env var.")

        self._clients: list[OpenAI] = []
        for k in keys:
            self._clients.append(
                OpenAI(base_url=base_url, api_key=k, http_client=httpx.Client(verify=False))
            )
        self._client_idx = 0
        self._client_lock = __import__("threading").Lock()
        logger.info(f"initialized {len(self._clients)} LLM client(s)")

    @property
    def client(self) -> OpenAI:
        """Round-robin client selection (thread-safe)."""
        with self._client_lock:
            c = self._clients[self._client_idx % len(self._clients)]
            self._client_idx += 1
            return c

    def extract_from_abstract(
        self,
        abstract: str,
        paper: PaperRef,
    ) -> ExtractionResult:
        """Extract claims from a single paper abstract.

        Args:
            abstract: The paper abstract text.
            paper: Paper metadata (PMID, title, authors, etc.).

        Returns:
            ExtractionResult with extracted claims.
        """
        # truncate very long abstracts to avoid token limits
        if len(abstract) > 2000:
            abstract = abstract[:2000] + "..."

        prompt = EXTRACTION_PROMPT.format(
            abstract=abstract,
            pmid=paper.pmid or "unknown",
            title=paper.title or "unknown",
            authors=paper.authors or "unknown",
            year=paper.year or "unknown",
            journal=paper.journal or "unknown",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise neuroscience data extraction system. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192,
            )

            raw_text = response.choices[0].message.content.strip()
            claims = self._parse_response(raw_text, paper)

            return ExtractionResult(
                paper=paper,
                claims=claims,
                raw_response=raw_text,
            )

        except Exception as e:
            logger.error(f"extraction failed for PMID {paper.pmid}: {e}")
            return ExtractionResult(
                paper=paper,
                claims=[],
                error=str(e),
            )

    def extract_batch(
        self,
        papers: list[tuple[str, PaperRef]],
        max_workers: int = 1,
    ) -> list[ExtractionResult]:
        """Extract claims from multiple papers.

        Args:
            papers: List of (abstract, PaperRef) tuples.
            max_workers: Number of parallel workers. 1 = sequential.

        Returns:
            List of ExtractionResult (same order as input).
        """
        if max_workers <= 1:
            # sequential mode
            results = []
            for i, (abstract, paper) in enumerate(papers):
                logger.info(f"extracting claims [{i+1}/{len(papers)}] PMID={paper.pmid}")
                result = self.extract_from_abstract(abstract, paper)
                logger.info(f"  extracted {len(result.claims)} claims")
                results.append(result)
            return results

        # parallel mode
        results: list[Optional[ExtractionResult]] = [None] * len(papers)

        def _extract_one(idx: int, abstract: str, paper: PaperRef) -> tuple[int, ExtractionResult]:
            logger.info(f"extracting claims [{idx+1}/{len(papers)}] PMID={paper.pmid}")
            result = self.extract_from_abstract(abstract, paper)
            logger.info(f"  extracted {len(result.claims)} claims (PMID={paper.pmid})")
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_extract_one, i, abstract, paper)
                for i, (abstract, paper) in enumerate(papers)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _parse_response(self, raw_text: str, paper: PaperRef) -> list[Claim]:
        """Parse LLM JSON response into Claim objects."""
        # try to extract JSON array from response
        json_str = self._extract_json(raw_text)
        if not json_str:
            logger.warning(f"no JSON found in response for PMID {paper.pmid}")
            return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for PMID {paper.pmid}: {e}")
            return []

        if not isinstance(data, list):
            data = [data]

        claims = []
        for item in data:
            try:
                claim = self._item_to_claim(item, paper)
                if claim:
                    claims.append(claim)
            except Exception as e:
                logger.warning(f"failed to parse claim item: {e}")
                continue

        return claims

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON array or object from LLM response text."""
        text = text.strip()

        # try direct parse first
        if text.startswith("[") or text.startswith("{"):
            return text

        # fix common LLM error: double brackets [[...]]
        if text.startswith("[["):
            text = text[1:]

        # try to find JSON in markdown code block
        match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if match:
            return match.group(1).strip()

        # try to find JSON array in text
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0)

        return None

    @staticmethod
    def _parse_numeric(value, target_type: str = "float"):
        """Parse a numeric field from LLM output.

        Handles: pure numbers, comparison strings ("p < 0.05", "n = 150"),
        and "not_reported". Returns (parsed_value, raw_string) tuple.
        """
        if value is None:
            return None, "not_reported"

        if isinstance(value, (int, float)):
            return value, str(value)

        s = str(value).strip().lower()
        if s in ("not_reported", "n/a", "na", "none", "", "null"):
            return None, "not_reported"

        # try direct numeric parse
        try:
            v = float(s) if target_type == "float" else int(s)
            return v, s
        except (ValueError, TypeError):
            pass

        # extract number from comparison strings like "p < 0.05", "n = 150", "β = -0.32"
        # strip commas from numbers like "2,048"
        s_clean = s.replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", s_clean)
        if match:
            try:
                v = float(match.group()) if target_type == "float" else int(float(match.group()))
                return v, s
            except (ValueError, TypeError):
                pass

        return None, s

    def _item_to_claim(self, item: dict, paper: PaperRef) -> Optional[Claim]:
        """Convert a single JSON item to a Claim object."""
        subject = item.get("subject", "").strip()
        obj = item.get("object", "").strip()
        predicate = item.get("predicate", "").strip()

        if not subject or not obj or not predicate:
            return None

        # parse numeric fields with range/comparison support
        effect_size, effect_size_raw = self._parse_numeric(item.get("effect_size"), "float")
        p_value, p_value_raw = self._parse_numeric(item.get("p_value"), "float")
        sample_size, sample_size_raw = self._parse_numeric(item.get("sample_size"), "int")

        # store raw strings in metadata for downstream use
        raw_stats = {}
        if p_value_raw != "not_reported":
            raw_stats["p_value_raw"] = p_value_raw
        if effect_size_raw != "not_reported":
            raw_stats["effect_size_raw"] = effect_size_raw
        if sample_size_raw != "not_reported":
            raw_stats["sample_size_raw"] = sample_size_raw

        effect_metric = item.get("effect_metric", "")
        if isinstance(effect_metric, str) and effect_metric.lower() in ("not_reported", "n/a", ""):
            effect_metric = ""

        evidence = Evidence(
            study_type=item.get("study_type", ""),
            methodology=item.get("methodology", ""),
            p_value=p_value,
            effect_size=effect_size,
            effect_metric=effect_metric,
            sample_size=sample_size,
            replicability=item.get("replicability", "single_study"),
            direction=item.get("direction", ""),
        )

        # generate claim ID
        claim_id = f"CLM:{uuid.uuid4().hex[:12]}"

        # parse conditions and population (contextualized triplets)
        conditions = item.get("conditions") or []
        if not isinstance(conditions, list):
            conditions = [str(conditions)]

        population = item.get("population")
        if isinstance(population, dict):
            # normalize numeric fields
            for key in ("mean_age", "n_female", "n_male"):
                if population.get(key) is not None:
                    try:
                        population[key] = float(population[key]) if key == "mean_age" else int(population[key])
                    except (ValueError, TypeError):
                        population[key] = None
        else:
            population = None

        return Claim(
            id=claim_id,
            subject_id="",  # will be resolved during ingestion
            subject_name=subject,
            predicate=predicate,
            object_id="",   # will be resolved during ingestion
            object_name=obj,
            negated=bool(item.get("negated", False)),
            confidence=self._estimate_confidence(evidence),
            evidence=evidence,
            source_paper=paper,
            raw_text=item.get("raw_sentence", ""),
            metadata={
                "subject_type": item.get("subject_type", ""),
                "object_type": item.get("object_type", ""),
                "conditions": conditions,
                "population": population,
                "raw_stats": raw_stats,
            },
        )

    def _estimate_confidence(self, evidence: Evidence) -> float:
        """Estimate claim confidence based on evidence quality."""
        score = 0.5  # baseline

        # p-value boost
        if evidence.p_value is not None:
            if evidence.p_value < 0.001:
                score += 0.2
            elif evidence.p_value < 0.01:
                score += 0.15
            elif evidence.p_value < 0.05:
                score += 0.1

        # sample size boost
        if evidence.sample_size is not None:
            if evidence.sample_size > 1000:
                score += 0.15
            elif evidence.sample_size > 100:
                score += 0.1
            elif evidence.sample_size > 30:
                score += 0.05

        # replicability boost
        if evidence.replicability == "replicated":
            score += 0.15
        elif evidence.replicability == "controversial":
            score -= 0.1

        # meta-analysis boost
        if evidence.study_type == "meta_analysis":
            score += 0.15

        return min(max(score, 0.0), 1.0)
