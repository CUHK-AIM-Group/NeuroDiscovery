"""Run generation-first Case Study 1 baselines.

Unlike the strong ranking baselines in case1_method_comparison.py, this script
asks each published-autoresearch baseline to generate hypotheses first, then
maps those hypotheses back to the pre-enumerated Case Study 1 candidate universe
for evaluation. The prompt never includes outcome labels, effect sizes, FDR
values, or NeuroDiscovery closed-loop feedback.
"""

from __future__ import annotations

import argparse
import httpx
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from case1_method_comparison import DEFAULT_ALL_TESTS, DEFAULT_OUT_DIR, METHOD_LABELS, load_results


METHODS = (
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
)

METHOD_INSTRUCTIONS = {
    "ai_scientist_v2": (
        "Act like an autonomous AI Scientist ideation module. Generate bold but testable "
        "neuroimaging hypotheses that could plausibly reveal cross-diagnostic findings. "
        "Prioritize novelty, clear experimental tests, and diverse disease-feature-region ideas."
    ),
    "co_scientist_style": (
        "Act like a co-scientist system with generator, critic, and ranker roles. First "
        "favor plausible mechanisms, then critique them for testability and redundancy, "
        "and return the final ranked hypotheses."
    ),
    "data_to_paper_style": (
        "Act like a data-to-paper research workflow. Generate hypotheses that are likely "
        "to be statistically analyzable, interpretable, and easy to turn into a concise "
        "data-driven result."
    ),
    "sciagents_style": (
        "Act like a KG/graph-reasoning scientific agent. Use only general biomedical and "
        "neuroanatomical knowledge from the prompt and your pretrained knowledge; do not "
        "assume access to our internal KG. Favor disease-region-feature links that would "
        "be supported by graph-neighborhood reasoning."
    ),
    "virtual_lab_style": (
        "Act like a virtual lab meeting with PI, neuroimaging scientist, psychiatrist, "
        "and statistician roles. Generate a ranked consensus list of experimentally "
        "testable disease-region-feature hypotheses."
    ),
    "openscholar_rag": (
        "Act like a retrieval-augmented literature synthesis system, but without live "
        "retrieval. Use general literature knowledge to propose evidence-backed, "
        "citation-plausible hypotheses. Do not fabricate citation identifiers."
    ),
}

DISEASE_DESCRIPTIONS = {
    "ADHD": "attention-deficit/hyperactivity disorder",
    "MDD_depression": "major depressive disorder",
    "OCD_OC_related": "obsessive-compulsive and related disorders",
    "PTSD_trauma": "post-traumatic stress/trauma-related disorders",
    "anxiety": "anxiety disorders",
    "bipolar": "bipolar disorder",
    "eating_disorder": "eating disorders",
    "psychosis_SZ_SZA": "psychosis/schizophrenia/schizoaffective disorder",
    "substance_use": "substance use disorders",
}

FEATURE_DESCRIPTIONS = {
    "corr_mean": "mean functional connectivity",
    "corr_mean_abs": "absolute mean functional connectivity",
    "corr_negative_mean": "negative functional connectivity",
    "corr_node_degree_abs_top10": "high absolute functional-connectivity node degree",
    "corr_positive_mean": "positive functional connectivity",
    "normalized_volume_fraction": "regional normalized structural volume fraction",
    "partial_mean": "mean partial functional connectivity",
    "partial_mean_abs": "absolute mean partial functional connectivity",
    "partial_negative_mean": "negative partial functional connectivity",
    "partial_positive_mean": "positive partial functional connectivity",
    "roi_alff_proxy": "regional ALFF-like temporal amplitude",
    "roi_falff_proxy": "regional fALFF-like temporal amplitude fraction",
    "roi_temporal_mean": "regional temporal mean signal",
    "roi_temporal_mean_abs": "absolute regional temporal mean signal",
    "roi_temporal_std": "regional temporal standard deviation",
    "roi_temporal_variance": "regional temporal variance",
}


@dataclass
class CandidateIndex:
    frame: pd.DataFrame
    text_by_idx: list[str]
    token_sets: list[set[str]]


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    stop = {"left", "right", "lh", "rh", "bilateral", "region", "cortex", "network", "area", "gyrus"}
    return {tok for tok in normalize_text(text).split() if len(tok) > 2 and tok not in stop}


def build_candidate_index(scored: pd.DataFrame) -> CandidateIndex:
    fields = ["roi_name", "anatomy_key", "anatomy_full", "network", "map_group", "structure_class", "hemisphere", "source"]
    text_by_idx: list[str] = []
    token_sets: list[set[str]] = []
    for _, row in scored.iterrows():
        text = " ".join(str(row.get(field) or "") for field in fields)
        text_by_idx.append(normalize_text(text))
        token_sets.append(tokens(text))
    return CandidateIndex(scored.reset_index(drop=True), text_by_idx, token_sets)


def prompt_for_method(
    method: str,
    diseases: list[str],
    features: list[str],
    n_hypotheses: int,
    seed: int,
    start_rank: int = 1,
    previous: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    disease_lines = "\n".join(f"- {d}: {DISEASE_DESCRIPTIONS.get(d, d)}" for d in diseases)
    feature_lines = "\n".join(f"- {f}: {FEATURE_DESCRIPTIONS.get(f, f)}" for f in features)
    end_rank = start_rank + n_hypotheses - 1
    previous_lines = ""
    if previous:
        compact_previous = []
        for hyp in previous[-80:]:
            compact_previous.append(
                f"- {hyp.get('disease')} | {hyp.get('feature')} | {hyp.get('anatomy_query')}"
            )
        previous_lines = "\nAvoid duplicating these already generated hypotheses:\n" + "\n".join(compact_previous)
    method_label = METHOD_LABELS[method]
    system = (
        "You generate structured neuroimaging hypotheses for an evaluation benchmark. "
        "You must not claim access to experimental outcomes. Return valid JSON only."
    )
    user = f"""
Method to emulate: {method_label}

{METHOD_INSTRUCTIONS[method]}

Task:
Generate {n_hypotheses} ranked hypotheses for ranks {start_rank}-{end_rank} of a
transdiagnostic brain-atlas discovery task.
Each hypothesis must propose one disease, one neuroimaging feature, and one anatomical
region/query that can be mapped to a brain atlas ROI.

Allowed disease codes:
{disease_lines}

Allowed feature codes:
{feature_lines}

Important constraints:
- Use exactly one allowed disease code.
- Use exactly one allowed feature code.
- Do not use observed effect sizes, p-values, FDR, labels, or any hidden results.
- Do not assume access to the full enumerated candidate table.
- Make hypotheses diverse across diseases, features, and anatomy.
- Seed for deterministic diversity: {seed}.
{previous_lines}

Return JSON with this schema:
{{
  "method": "{method}",
  "hypotheses": [
    {{
      "rank": {start_rank},
      "disease": "one allowed disease code",
      "feature": "one allowed feature code",
      "anatomy_query": "short anatomical target, e.g. anterior cingulate cortex",
      "hemisphere": "left/right/bilateral/unspecified",
      "rationale": "one concise sentence",
      "confidence": 0.0
    }}
  ]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_json(
    method: str,
    messages: list[dict[str, str]],
    model: str,
    seed: int,
    base_url: str | None,
    timeout_s: float,
    api: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SUB2API_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or SUB2API_OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("SUB2API_OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, base_url=resolved_base_url, timeout=timeout_s)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            if api == "responses":
                endpoint = (resolved_base_url or "https://api.openai.com/v1").rstrip("/")
                if not endpoint.endswith("/responses"):
                    endpoint = f"{endpoint}/responses"
                payload = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": messages[0]["content"]},
                        {"role": "user", "content": messages[1]["content"]},
                    ],
                    "store": False,
                }
                response = httpx.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout_s,
                )
                response.raise_for_status()
                body = response.text
                content_parts: list[str] = []
                for line in body.splitlines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data.get("type") == "response.content_part.done":
                        part = data.get("part") or {}
                        if part.get("type") == "output_text":
                            content_parts.append(str(part.get("text") or ""))
                if content_parts:
                    content = "".join(content_parts)
                else:
                    content = body
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    seed=seed,
                    response_format={"type": "json_object"},
                    timeout=timeout_s,
                )
                content = response.choices[0].message.content or "{}"
            parsed = extract_json(content)
            parsed["_api_model"] = model
            parsed["_api_method"] = method
            parsed["_api_wire"] = api
            return parsed
        except Exception as exc:  # pragma: no cover - API path
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"API call failed for {method}: {last_error}") from last_error


def load_or_generate(
    method: str,
    diseases: list[str],
    features: list[str],
    n_hypotheses: int,
    batch_size: int,
    seed: int,
    model: str,
    base_url: str | None,
    timeout_s: float,
    api: str,
    raw_dir: Path,
    force_api: bool,
    dry_run: bool,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{method}_seed{seed}.json"
    if path.exists() and not force_api:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypotheses = payload.get("hypotheses") or []
        if isinstance(hypotheses, list) and len(hypotheses) >= n_hypotheses:
            payload["hypotheses"] = hypotheses[:n_hypotheses]
            return payload
    batch_size = max(1, int(batch_size or n_hypotheses))
    if batch_size >= n_hypotheses:
        messages = prompt_for_method(method, diseases, features, n_hypotheses, seed)
        if dry_run:
            payload = {
                "method": method,
                "hypotheses": [
                    {
                        "rank": i + 1,
                        "disease": diseases[(i + seed) % len(diseases)],
                        "feature": features[(i * 3 + seed) % len(features)],
                        "anatomy_query": ["anterior cingulate cortex", "insula", "default mode network", "hippocampus"][i % 4],
                        "hemisphere": "unspecified",
                        "rationale": "Dry-run placeholder generated without API.",
                        "confidence": 0.5,
                    }
                    for i in range(n_hypotheses)
                ],
                "_dry_run": True,
            }
        else:
            payload = call_openai_json(method, messages, model, seed, base_url, timeout_s, api)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    aggregate: dict[str, Any] = {"method": method, "hypotheses": []}
    if path.exists() and not force_api:
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_hypotheses = existing.get("hypotheses") or []
        if isinstance(existing_hypotheses, list):
            aggregate["hypotheses"] = existing_hypotheses[:n_hypotheses]
            aggregate["_resumed_from_aggregate"] = str(path)
    hypotheses = aggregate["hypotheses"]
    assert isinstance(hypotheses, list)
    batch_dir = raw_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    while len(hypotheses) < n_hypotheses:
        start_rank = len(hypotheses) + 1
        this_n = min(batch_size, n_hypotheses - len(hypotheses))
        end_rank = start_rank + this_n - 1
        batch_path = batch_dir / f"{method}_seed{seed}_rank{start_rank:04d}_{end_rank:04d}.json"
        if batch_path.exists() and not force_api:
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
        else:
            messages = prompt_for_method(method, diseases, features, this_n, seed + start_rank, start_rank, hypotheses)
            if dry_run:
                batch_payload = {
                    "method": method,
                    "hypotheses": [
                        {
                            "rank": start_rank + i,
                            "disease": diseases[(start_rank + i + seed) % len(diseases)],
                            "feature": features[((start_rank + i) * 3 + seed) % len(features)],
                            "anatomy_query": [
                                "anterior cingulate cortex",
                                "insula",
                                "default mode network",
                                "hippocampus",
                                "amygdala",
                                "prefrontal cortex",
                            ][i % 6],
                            "hemisphere": "unspecified",
                            "rationale": "Dry-run placeholder generated without API.",
                            "confidence": 0.5,
                        }
                        for i in range(this_n)
                    ],
                    "_dry_run": True,
                }
            else:
                batch_payload = call_openai_json(method, messages, model, seed + start_rank, base_url, timeout_s, api)
            batch_payload["_rank_start"] = start_rank
            batch_payload["_rank_end"] = end_rank
            batch_path.write_text(json.dumps(batch_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_hypotheses = batch_payload.get("hypotheses") or []
        if not isinstance(batch_hypotheses, list) or not batch_hypotheses:
            raise RuntimeError(f"No hypotheses returned for {method} ranks {start_rank}-{end_rank}")
        for i, hyp in enumerate(batch_hypotheses[:this_n], start=start_rank):
            if isinstance(hyp, dict):
                hyp = dict(hyp)
                hyp["rank"] = i
                hypotheses.append(hyp)
        aggregate["hypotheses"] = hypotheses[:n_hypotheses]
        aggregate["_batched"] = True
        aggregate["_batch_size"] = batch_size
        path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{method}: generated/resumed {len(hypotheses)}/{n_hypotheses} hypotheses")
    return aggregate


def normalize_hemisphere(value: Any) -> str:
    text = normalize_text(value)
    if "left" in text or text == "lh":
        return "left"
    if "right" in text or text == "rh":
        return "right"
    if "bilateral" in text or "both" in text:
        return "bilateral"
    return "unspecified"


def map_one_hypothesis(hyp: dict[str, Any], index: CandidateIndex, used: set[str]) -> tuple[pd.Series | None, float, str]:
    disease = str(hyp.get("disease") or "")
    feature = str(hyp.get("feature") or "")
    anatomy_query = str(hyp.get("anatomy_query") or "")
    hemi = normalize_hemisphere(hyp.get("hemisphere"))
    if disease not in set(index.frame["disease"].unique()):
        return None, 0.0, "invalid_disease"
    if feature not in set(index.frame["feature"].unique()):
        return None, 0.0, "invalid_feature"
    q_tokens = tokens(anatomy_query)
    if not q_tokens:
        return None, 0.0, "missing_anatomy"

    mask = (index.frame["disease"] == disease) & (index.frame["feature"] == feature)
    if hemi in {"left", "right"}:
        hemi_mask = index.frame["hemisphere"].fillna("").astype(str).str.lower().eq(hemi)
        if bool((mask & hemi_mask).any()):
            mask = mask & hemi_mask
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) == 0:
        return None, 0.0, "no_candidate_for_disease_feature"

    best_idx = None
    best_score = -1.0
    q_norm = normalize_text(anatomy_query)
    for idx in idxs:
        candidate_id = str(index.frame.at[idx, "candidate_id"])
        if candidate_id in used:
            continue
        cand_text = index.text_by_idx[idx]
        cand_tokens = index.token_sets[idx]
        overlap = len(q_tokens & cand_tokens)
        score = overlap / max(1, len(q_tokens))
        if q_norm and q_norm in cand_text:
            score += 1.0
        if any(tok in cand_text for tok in q_tokens):
            score += 0.15
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return None, 0.0, "duplicate_or_no_unused_candidate"
    if best_score <= 0:
        return None, best_score, "anatomy_unmapped"
    return index.frame.iloc[int(best_idx)], float(best_score), "mapped"


def map_hypotheses(method: str, payload: dict[str, Any], index: CandidateIndex, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    hypotheses = payload.get("hypotheses") or []
    if not isinstance(hypotheses, list):
        hypotheses = []
    for i, hyp in enumerate(hypotheses, start=1):
        if not isinstance(hyp, dict):
            continue
        mapped, score, status = map_one_hypothesis(hyp, index, used)
        schema_format_correct = status not in {"invalid_disease", "invalid_feature", "missing_anatomy"}
        row = {
            "method": method,
            "seed": seed,
            "generated_rank": int(hyp.get("rank") or i),
            "generated_disease": hyp.get("disease"),
            "generated_feature": hyp.get("feature"),
            "generated_anatomy_query": hyp.get("anatomy_query"),
            "generated_hemisphere": hyp.get("hemisphere"),
            "generated_rationale": hyp.get("rationale"),
            "generated_confidence": hyp.get("confidence"),
            "mapping_status": status,
            "mapping_score": score,
            "schema_format_correct": schema_format_correct,
            "format_correct": status == "mapped",
        }
        if mapped is not None:
            candidate_id = str(mapped["candidate_id"])
            used.add(candidate_id)
            for col in [
                "candidate_id",
                "disease",
                "feature",
                "modality",
                "source",
                "roi_index",
                "roi_name",
                "anatomy_full",
                "hemisphere",
                "map_group",
                "is_gt_top",
                "is_strict_fdr",
                "abs_adjusted_residual_d",
            ]:
                row[f"mapped_{col}"] = mapped.get(col)
        rows.append(row)
    mapped_df = pd.DataFrame(rows).sort_values("generated_rank", kind="mergesort")
    mapped_df["valid_rank"] = np.where(mapped_df["mapping_status"].eq("mapped"), mapped_df["mapping_status"].eq("mapped").cumsum(), np.nan)
    return mapped_df


def summarize(mapped: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for method, sub in mapped.groupby("method", sort=False):
        sub = sub.sort_values("generated_rank", kind="mergesort")
        valid = sub[sub["mapping_status"].eq("mapped")].copy()
        valid["mapped_is_gt_top"] = valid["mapped_is_gt_top"].astype(bool)
        valid["mapped_is_strict_fdr"] = valid["mapped_is_strict_fdr"].astype(bool)
        status_counts = sub["mapping_status"].value_counts().to_dict()
        schema_correct = sub.get("schema_format_correct", sub["mapping_status"].ne("invalid_disease")).astype(bool)
        for budget in budgets:
            head = valid.head(budget)
            rows.append(
                {
                    "method": method,
                    "budget_format_correct": budget,
                    "budget_valid_mapped": budget,
                    "generated_total": len(sub),
                    "schema_format_correct_total": int(schema_correct.sum()),
                    "schema_format_correct_rate": float(schema_correct.mean()) if len(sub) else 0.0,
                    "format_correct_total": len(valid),
                    "format_correct_rate": len(valid) / max(1, len(sub)),
                    "valid_mapped_total": len(valid),
                    "valid_mapping_rate": len(valid) / max(1, len(sub)),
                    "anatomy_unmapped_total": int(status_counts.get("anatomy_unmapped", 0)),
                    "invalid_disease_total": int(status_counts.get("invalid_disease", 0)),
                    "invalid_feature_total": int(status_counts.get("invalid_feature", 0)),
                    "missing_anatomy_total": int(status_counts.get("missing_anatomy", 0)),
                    "gt_hits": int(head["mapped_is_gt_top"].sum()) if not head.empty else 0,
                    "strict_fdr_hits": int(head["mapped_is_strict_fdr"].sum()) if not head.empty else 0,
                    "precision": float(head["mapped_is_gt_top"].mean()) if not head.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "generation_first_seed0")
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hypotheses", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=80, help="Generate API hypotheses in resumable batches.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, for example https://provider.example/v1")
    parser.add_argument("--api", choices=("responses", "chat"), default="chat")
    parser.add_argument("--api-timeout-s", type=float, default=120.0)
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw_api"
    scored = load_results(args.all_tests, args.gt_top_frac)
    diseases = sorted(scored["disease"].dropna().unique().tolist())
    features = sorted(scored["feature"].dropna().unique().tolist())
    index = build_candidate_index(scored)

    mapped_parts = []
    for method in args.methods:
        payload = load_or_generate(
            method=method,
            diseases=diseases,
            features=features,
            n_hypotheses=args.n_hypotheses,
            batch_size=args.batch_size,
            seed=args.seed,
            model=args.model,
            base_url=args.base_url,
            timeout_s=args.api_timeout_s,
            api=args.api,
            raw_dir=raw_dir,
            force_api=args.force_api,
            dry_run=args.dry_run,
        )
        mapped_parts.append(map_hypotheses(method, payload, index, args.seed))

    mapped = pd.concat(mapped_parts, ignore_index=True)
    default_budgets = [10, 25, 50, 100, 200, 300, 400, args.n_hypotheses]
    budgets = sorted({b for b in default_budgets if 1 <= b <= args.n_hypotheses})
    summary = summarize(mapped, budgets)
    mapped.to_csv(args.out_dir / "generation_first_mapped_hypotheses.csv", index=False)
    summary.to_csv(args.out_dir / "generation_first_summary.csv", index=False)
    manifest = {
        "all_tests": str(args.all_tests),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "n_hypotheses": args.n_hypotheses,
        "batch_size": args.batch_size,
        "model": args.model,
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("SUB2API_OPENAI_BASE_URL") or "default_openai",
        "api": args.api,
        "methods": list(args.methods),
        "gt_top_frac": args.gt_top_frac,
        "prompt_policy": "generation-first; no outcome labels, effect sizes, FDR values, full candidate table, or NeuroDiscovery feedback",
    }
    (args.out_dir / "generation_first_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
