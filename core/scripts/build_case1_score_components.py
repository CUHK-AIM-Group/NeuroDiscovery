"""Build frozen, outcome-blind Case Study 1 candidate score components.

The builder intentionally reads only public candidate metadata. Experimental
effect sizes, p-values, FDR labels, and GT annotations never enter KGE,
novelty, or critic scoring.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable, Sequence

import httpx
import numpy as np
import pandas as pd

try:
    from core.scripts.canonical_kg_release import validate_canonical_kg_release
    from core.scripts.case1_method_comparison import (
        add_generator_scores,
        candidate_region_terms,
        disease_terms,
        feature_family,
        kg_query_terms_for_candidates,
        load_kg_index,
        map_group,
        resolve_ids,
        sha256_file,
    )
    from core.scripts.case1_neurodiscovery_config import feature_terms
    from core.scripts.case_study_score_components import (
        SCORE_COMPONENT_SCHEMA,
        candidate_id_sha256,
        load_score_component_bundle,
    )
    from neurooracle.src.kge.complex_scorer import ComplExScorer
except ModuleNotFoundError:
    from canonical_kg_release import validate_canonical_kg_release
    from case1_method_comparison import (
        add_generator_scores,
        candidate_region_terms,
        disease_terms,
        feature_family,
        kg_query_terms_for_candidates,
        load_kg_index,
        map_group,
        resolve_ids,
        sha256_file,
    )
    from case1_neurodiscovery_config import feature_terms
    from case_study_score_components import (
        SCORE_COMPONENT_SCHEMA,
        candidate_id_sha256,
        load_score_component_bundle,
    )
    from neurooracle.src.kge.complex_scorer import ComplExScorer


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_STUDY_ID = "case1_transdiagnostic"
DEFAULT_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_KG = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_CLAIMS = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
DEFAULT_STATE = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "CURRENT_STATE.json"
DEFAULT_COMPONENT_DIR = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_tuning"
    r"\20260810_kgdd0d4037_roi_cv_v2\score_components"
)
DEFAULT_KGE = DEFAULT_COMPONENT_DIR / "kge_full_v2_dd0d4037_complex_dim64_transductive.pt"
DEFAULT_KGE_REPORT = DEFAULT_KGE.with_suffix(".report.json")
PUBLIC_COLUMNS = (
    "modality",
    "source",
    "disease",
    "feature",
    "roi_index",
    "roi_id",
    "roi_name",
    "anatomy_key",
    "anatomy_full",
    "hemisphere",
    "network",
    "structure_class",
    "n_case",
    "n_control",
    "atlas_label_source",
    "atlas_label_weight",
)
CRITIC_DIMENSIONS = (
    "scientific_plausibility",
    "experimental_testability",
    "anatomical_specificity",
    "confound_resistance",
)
CRITIC_WEIGHTS = {
    "scientific_plausibility": 0.30,
    "experimental_testability": 0.30,
    "anatomical_specificity": 0.20,
    "confound_resistance": 0.20,
}
CRITIC_SYSTEM_PROMPT = """You are a conservative neuroimaging study-design critic.
Evaluate candidate archetypes for a preregistered transdiagnostic case-control
experiment. Judge whether each disease, imaging-feature family, and anatomical
group forms a scientifically plausible, executable, specific, and reasonably
confound-resistant test. Do not predict the result, direction, effect size, p
value, or whether the hypothesis is true. Do not use outside search. Use only
the labels supplied in the request.

Return one JSON object and no prose:
{"scores":[{"id":"A0001","scientific_plausibility":0-100,
"experimental_testability":0-100,"anatomical_specificity":0-100,
"confound_resistance":0-100}]}
Every requested id must appear exactly once. Scores must be integers from 0 to
100. Apply the same calibration across all entries in the batch."""


@dataclass(frozen=True)
class PairTemplate:
    source: str
    relation: str
    target: str


DISEASE_IMAGING_TEMPLATES = (
    PairTemplate("right", "is_biomarker_of", "left"),
    PairTemplate("left", "has_imaging_feature", "right"),
    PairTemplate("left", "is_associated_with", "right"),
    PairTemplate("right", "is_associated_with", "left"),
    PairTemplate("right", "correlates_with", "left"),
)
REGION_FEATURE_TEMPLATES = (
    PairTemplate("left", "has_imaging_feature", "right"),
    PairTemplate("right", "is_imaging_feature_of", "left"),
    PairTemplate("left", "is_associated_with", "right"),
    PairTemplate("right", "is_associated_with", "left"),
    PairTemplate("left", "correlates_with", "right"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def percentile_score(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("score input must be finite")
    unique_count = int(series.nunique(dropna=False))
    if unique_count <= 1:
        return np.full(len(series), 0.5, dtype=float)
    dense_rank = series.rank(method="dense").to_numpy(float) - 1.0
    return dense_rank / float(unique_count - 1)


def load_public_candidate_registry(path: Path) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(PUBLIC_COLUMNS) - available)
    if missing:
        raise ValueError("candidate table lacks public columns: " + ", ".join(missing))
    public = pd.read_csv(path, usecols=list(PUBLIC_COLUMNS), low_memory=False)
    public["candidate_id"] = (
        public["modality"].astype(str)
        + "|"
        + public["source"].astype(str)
        + "|"
        + public["disease"].astype(str)
        + "|"
        + public["feature"].astype(str)
        + "|"
        + public["roi_index"].astype(str)
    )
    if public["candidate_id"].duplicated().any():
        raise ValueError("candidate registry contains duplicate candidate_id values")
    public["roi_key"] = (
        public["modality"].astype(str)
        + "|"
        + public["source"].astype(str)
        + "|"
        + public["roi_index"].astype(str)
    )
    public["feature_family"] = public["feature"].map(feature_family).astype(str)
    public["map_group"] = public.apply(map_group, axis=1)
    return public


def primary_entity_maps(public: pd.DataFrame, kg: Any) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    disease_map = {
        disease: next(iter(resolve_ids(disease_terms(disease), kg, max_ids=1)), "")
        for disease in sorted(public["disease"].astype(str).unique())
    }
    feature_map = {
        feature: next(iter(resolve_ids(feature_terms(feature), kg, max_ids=1)), "")
        for feature in sorted(public["feature"].astype(str).unique())
    }
    roi_rows = public.drop_duplicates("roi_key")
    roi_map = {
        str(row.roi_key): next(
            iter(resolve_ids(candidate_region_terms(pd.Series(row._asdict())), kg, max_ids=1)),
            "",
        )
        for row in roi_rows.itertuples(index=False)
    }
    return disease_map, feature_map, roi_map


def score_unique_pairs(
    scorer: ComplExScorer,
    pairs: Sequence[tuple[str, str]],
    left_ids: dict[str, str],
    right_ids: dict[str, str],
    templates: Sequence[PairTemplate],
    *,
    chunk_size: int = 100_000,
) -> tuple[dict[tuple[str, str], float], int]:
    resolved: list[tuple[tuple[str, str], list[tuple[str, str, str]]]] = []
    output: dict[tuple[str, str], float] = {}
    for pair in pairs:
        left = left_ids.get(pair[0], "")
        right = right_ids.get(pair[1], "")
        if not left or not right:
            output[pair] = 0.5
            continue
        triples = []
        for template in templates:
            source = left if template.source == "left" else right
            target = left if template.target == "left" else right
            triples.append((source, template.relation, target))
        resolved.append((pair, triples))

    flat = [(pair, triple) for pair, triples in resolved for triple in triples]
    maxima: dict[tuple[str, str], float] = {pair: 0.0 for pair, _ in resolved}
    for start in range(0, len(flat), chunk_size):
        batch = flat[start : start + chunk_size]
        values = scorer.score_batch([triple for _, triple in batch])
        for (pair, _), value in zip(batch, values, strict=True):
            maxima[pair] = max(maxima[pair], float(value))
    output.update(maxima)
    return output, len(resolved)


def build_kge_scores(
    public: pd.DataFrame,
    scored: pd.DataFrame,
    kg: Any,
    checkpoint: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    scorer = ComplExScorer.load(checkpoint)
    disease_ids, feature_ids, roi_ids = primary_entity_maps(public, kg)

    disease_roi_pairs = sorted(
        set(zip(scored["disease"].astype(str), scored["roi_key"].astype(str), strict=False))
    )
    disease_feature_pairs = sorted(
        set(zip(scored["disease"].astype(str), scored["feature"].astype(str), strict=False))
    )
    roi_feature_pairs = sorted(
        set(zip(scored["roi_key"].astype(str), scored["feature"].astype(str), strict=False))
    )
    disease_roi, disease_roi_resolved = score_unique_pairs(
        scorer,
        disease_roi_pairs,
        disease_ids,
        roi_ids,
        DISEASE_IMAGING_TEMPLATES,
    )
    disease_feature, disease_feature_resolved = score_unique_pairs(
        scorer,
        disease_feature_pairs,
        disease_ids,
        feature_ids,
        DISEASE_IMAGING_TEMPLATES,
    )
    roi_feature, roi_feature_resolved = score_unique_pairs(
        scorer,
        roi_feature_pairs,
        roi_ids,
        feature_ids,
        REGION_FEATURE_TEMPLATES,
    )

    dr = np.fromiter(
        (
            disease_roi[(str(disease), str(roi))]
            for disease, roi in zip(scored["disease"], scored["roi_key"], strict=False)
        ),
        dtype=float,
        count=len(scored),
    )
    df = np.fromiter(
        (
            disease_feature[(str(disease), str(feature))]
            for disease, feature in zip(scored["disease"], scored["feature"], strict=False)
        ),
        dtype=float,
        count=len(scored),
    )
    rf = np.fromiter(
        (
            roi_feature[(str(roi), str(feature))]
            for roi, feature in zip(scored["roi_key"], scored["feature"], strict=False)
        ),
        dtype=float,
        count=len(scored),
    )
    raw = 0.60 * dr + 0.25 * df + 0.15 * rf
    score = percentile_score(raw)
    audit = {
        "definition": (
            "Percentile-normalized weighted maximum ComplEx link probability: "
            "0.60 disease-region + 0.25 disease-feature + 0.15 region-feature."
        ),
        "primary_entity_resolution": "highest-degree exact-or-registered-alias KG entity",
        "unique_pairs": {
            "disease_region": len(disease_roi_pairs),
            "disease_feature": len(disease_feature_pairs),
            "region_feature": len(roi_feature_pairs),
        },
        "resolved_pairs": {
            "disease_region": disease_roi_resolved,
            "disease_feature": disease_feature_resolved,
            "region_feature": roi_feature_resolved,
        },
        "raw_summary": summarize_values(raw),
        "score_summary": summarize_values(score),
    }
    return score, audit


def build_novelty_scores(scored: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    global_grounding = (
        0.30 * percentile_score(np.log1p(scored["kg_disease_degree"]))
        + 0.45 * percentile_score(np.log1p(scored["kg_region_degree"]))
        + 0.25 * percentile_score(np.log1p(scored["kg_feature_degree"]))
    )
    scoped_grounding = (
        0.30 * percentile_score(np.log1p(scored["kg_scoped_disease_degree"]))
        + 0.45 * percentile_score(np.log1p(scored["kg_scoped_region_degree"]))
        + 0.25 * percentile_score(np.log1p(scored["kg_scoped_feature_degree"]))
    )
    grounding = 0.65 * global_grounding + 0.35 * scoped_grounding

    global_attestation = (
        0.55 * percentile_score(scored["kg_pair_support"])
        + 0.25 * percentile_score(scored["kg_disease_feature_support"])
        + 0.20 * percentile_score(scored["kg_region_feature_support"])
    )
    scoped_attestation = (
        0.55 * percentile_score(scored["kg_scoped_pair_support"])
        + 0.25 * percentile_score(scored["kg_scoped_disease_feature_support"])
        + 0.20 * percentile_score(scored["kg_scoped_region_feature_support"])
    )
    attestation = 0.60 * global_attestation + 0.40 * scoped_attestation
    all_entities_resolved = (
        (scored["kg_disease_degree"].to_numpy(float) > 0)
        & (scored["kg_region_degree"].to_numpy(float) > 0)
        & (scored["kg_feature_degree"].to_numpy(float) > 0)
    )
    resolution_gate = np.where(all_entities_resolved, 1.0, 0.35)
    raw = grounding * (1.0 - attestation) * resolution_gate
    score = percentile_score(raw)
    audit = {
        "definition": (
            "Grounded evidence gap: percentile-normalized global/scoped entity "
            "grounding multiplied by one minus direct pair attestation; candidates "
            "with unresolved entities receive a 0.35 gate."
        ),
        "fully_resolved_fraction": float(np.mean(all_entities_resolved)),
        "grounding_summary": summarize_values(grounding),
        "attestation_summary": summarize_values(attestation),
        "raw_summary": summarize_values(raw),
        "score_summary": summarize_values(score),
    }
    return score, audit


def archetype_registry(scored: pd.DataFrame) -> pd.DataFrame:
    frame = (
        scored[["disease", "feature_family", "map_group"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["disease", "feature_family", "map_group"])
        .reset_index(drop=True)
    )
    frame.insert(0, "id", [f"A{index:04d}" for index in range(1, len(frame) + 1)])
    frame["archetype_key"] = (
        frame["disease"] + "|" + frame["feature_family"] + "|" + frame["map_group"]
    )
    return frame


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("critic response contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("critic response JSON must be an object")
    return value


def response_output_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(str(part.get("text") or ""))
    return "".join(parts)


def validate_critic_scores(payload: dict[str, Any], expected_ids: set[str]) -> list[dict[str, Any]]:
    rows = payload.get("scores")
    if not isinstance(rows, list):
        raise ValueError("critic response lacks scores list")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("critic score row must be an object")
        identifier = str(row.get("id") or "")
        if identifier not in expected_ids or identifier in seen:
            raise ValueError(f"unexpected or duplicate critic id: {identifier}")
        normalized: dict[str, Any] = {"id": identifier}
        for dimension in CRITIC_DIMENSIONS:
            value = float(row.get(dimension))
            if not np.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"invalid {dimension} for {identifier}")
            normalized[dimension] = value
        normalized["score_critic"] = sum(
            CRITIC_WEIGHTS[name] * normalized[name] for name in CRITIC_DIMENSIONS
        ) / 100.0
        parsed.append(normalized)
        seen.add(identifier)
    if seen != expected_ids:
        raise ValueError("critic response does not cover the requested ids exactly")
    return parsed


def critic_user_prompt(batch: pd.DataFrame) -> str:
    records = batch[["id", "disease", "feature_family", "map_group"]].to_dict("records")
    return "Score every archetype below using the frozen rubric.\n" + json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def call_critic_batch(
    batch: pd.DataFrame,
    *,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    timeout_s: float,
    max_retries: int,
    wire_api: str = "responses",
) -> list[dict[str, Any]]:
    endpoint = base_url.rstrip("/")
    expected_ids = set(batch["id"].astype(str))
    messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": critic_user_prompt(batch)},
    ]
    if wire_api == "chat":
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "enabled"},
            "reasoning_effort": reasoning_effort,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "stream": False,
        }
    else:
        if not endpoint.endswith("/responses"):
            endpoint += "/responses"
        payload = {
            "model": model,
            "input": messages,
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            body = response.json()
            output_text = (
                str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                if wire_api == "chat"
                else response_output_text(body)
            )
            return validate_critic_scores(
                extract_json_object(output_text),
                expected_ids,
            )
        except Exception as exc:  # pragma: no cover - live API path
            last_error = exc
            delays = (2, 8, 30)
            if attempt < max_retries - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])
    raise RuntimeError(f"critic batch failed after {max_retries} attempts: {last_error}")


def load_cached_critic(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def build_critic_scores(
    scored: pd.DataFrame,
    *,
    cache_path: Path,
    base_url: str,
    api_key_env: str,
    model: str,
    reasoning_effort: str,
    batch_size: int,
    workers: int,
    timeout_s: float,
    max_retries: int,
    wire_api: str = "responses",
) -> tuple[np.ndarray, dict[str, Any]]:
    registry = archetype_registry(scored)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_critic(cache_path)
    known_ids = set(registry["id"].astype(str))
    stale = sorted(set(cached) - known_ids)
    if stale:
        raise ValueError("critic cache contains stale archetype ids")
    missing = registry[~registry["id"].isin(cached)].copy()
    if not missing.empty:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing critic API key environment variable: {api_key_env}")
        batches = [missing.iloc[start : start + batch_size] for start in range(0, len(missing), batch_size)]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    call_critic_batch,
                    batch,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    wire_api=wire_api,
                ): batch_index
                for batch_index, batch in enumerate(batches)
            }
            with cache_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    batch_index = futures[future]
                    rows = future.result()
                    for row in rows:
                        row["model"] = model
                        row["reasoning_effort"] = reasoning_effort
                        row["batch_index"] = batch_index
                        row["created_at"] = utc_now()
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        cached[str(row["id"])] = row
                    handle.flush()

    if set(cached) != known_ids:
        raise ValueError("critic cache does not cover the current archetype registry")
    score_by_key = {
        row.archetype_key: float(cached[str(row.id)]["score_critic"])
        for row in registry.itertuples(index=False)
    }
    keys = (
        scored["disease"].astype(str)
        + "|"
        + scored["feature_family"].astype(str)
        + "|"
        + scored["map_group"].astype(str)
    )
    score = keys.map(score_by_key).to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("critic mapping produced non-finite candidate scores")
    audit = {
        "definition": (
            "Frozen LLM study-design critic at disease x feature-family x anatomical-group "
            "archetype level; weighted 0.30 plausibility, 0.30 testability, 0.20 "
            "specificity, and 0.20 confound resistance."
        ),
        "archetype_count": int(len(registry)),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "score_summary": summarize_values(score),
    }
    return score, audit


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "variance": float(np.var(array)),
        "max": float(np.max(array)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--kge-checkpoint", type=Path, default=DEFAULT_KGE)
    parser.add_argument("--kge-report", type=Path, default=DEFAULT_KGE_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--critic-batch-size", type=int, default=24)
    parser.add_argument("--critic-workers", type=int, default=4)
    parser.add_argument("--critic-timeout-s", type=float, default=300.0)
    parser.add_argument("--critic-max-retries", type=int, default=4)
    parser.add_argument("--allow-relocated-kg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id=CASE_STUDY_ID,
        allow_relocated_artifacts=args.allow_relocated_kg,
    )
    for path in (args.all_tests, args.kge_checkpoint, args.kge_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    public = load_public_candidate_registry(args.all_tests)
    query_terms = kg_query_terms_for_candidates(public)
    kg = load_kg_index(args.kg, query_terms)
    scored = add_generator_scores(public, kg, seed=260810)

    kge_score, kge_audit = build_kge_scores(public, scored, kg, args.kge_checkpoint)
    novelty_score, novelty_audit = build_novelty_scores(scored)
    prompt_path = args.out_dir / "critic_prompt.txt"
    prompt_path.write_text(CRITIC_SYSTEM_PROMPT + "\n", encoding="utf-8")
    critic_cache = args.out_dir / "critic_archetype_scores.jsonl"
    critic_score, critic_audit = build_critic_scores(
        scored,
        cache_path=critic_cache,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        batch_size=args.critic_batch_size,
        workers=args.critic_workers,
        timeout_s=args.critic_timeout_s,
        max_retries=args.critic_max_retries,
        wire_api=args.api,
    )

    table = pd.DataFrame(
        {
            "candidate_id": public["candidate_id"].astype(str),
            "score_kge": kge_score,
            "score_novelty": novelty_score,
            "score_critic": critic_score,
        }
    )
    table_path = args.out_dir / "score_components.csv"
    table.to_csv(table_path, index=False, float_format="%.9g")
    kge_report = json.loads(args.kge_report.read_text(encoding="utf-8"))
    manifest_path = args.out_dir / "score_components.manifest.json"
    manifest = {
        "schema_version": SCORE_COMPONENT_SCHEMA,
        "created_at": utc_now(),
        "case_study_id": CASE_STUDY_ID,
        "outcome_blind": True,
        "frozen_before_experiment": True,
        "candidate_table": str(args.all_tests.resolve()),
        "candidate_table_sha256": sha256_file(args.all_tests),
        "candidate_count": int(len(table)),
        "candidate_id_sha256": candidate_id_sha256(table["candidate_id"]),
        "score_table_sha256": sha256_file(table_path),
        "canonical_kg_release": release,
        "kg_index_stats": kg.stats,
        "components": {
            "kge": {
                "column": "score_kge",
                "model": "ComplEx",
                "checkpoint": str(args.kge_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.kge_checkpoint),
                "report": str(args.kge_report.resolve()),
                "report_sha256": sha256_file(args.kge_report),
                "kg_snapshot_sha256": release["files"]["knowledge_graph"]["sha256"],
                "kg_snapshot_status": "formal_release",
                "training_report": kge_report,
                **kge_audit,
            },
            "novelty": {
                "column": "score_novelty",
                **novelty_audit,
            },
            "critic": {
                "column": "score_critic",
                "model": args.model,
                "base_url": args.base_url,
                "wire_api": args.api,
                "reasoning_effort": args.reasoning_effort,
                "response_storage": False,
                "prompt_path": str(prompt_path.resolve()),
                "prompt_sha256": sha256_file(prompt_path),
                **critic_audit,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(manifest_path, manifest)

    # Exercise the production loader before declaring the bundle complete.
    _, audit = load_score_component_bundle(
        public,
        table_path=table_path,
        manifest_path=manifest_path,
    )
    write_json(args.out_dir / "score_components.loader_audit.json", audit)
    print(json.dumps({
        "table": str(table_path),
        "manifest": str(manifest_path),
        "candidate_count": len(table),
        "components": {
            "kge": summarize_values(kge_score),
            "novelty": summarize_values(novelty_score),
            "critic": summarize_values(critic_score),
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
