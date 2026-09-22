from __future__ import annotations

import argparse
import glob
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = ROOT / "neurooracle/data/user_study/adni_external_validation_study_v1.json"
DEFAULT_TRUTH = ROOT / "outputs/user_study/adni_external_validation_truth_v1.json"
DEFAULT_ANNOTATIONS = (
    ROOT / "neurooracle/data/user_study/adni_manual_relevance_annotations_v1.json"
)


SYSTEM_PROMPT = """You are a bilingual neuroscience literature curator.
Assess five papers for one neuroimaging hypothesis using only the supplied
title, abstract, excerpt, and metadata. Do not invent findings. Return JSON
only. Write precise, paper-specific Chinese and English explanations that let
an expert judge the hypothesis without opening the paper whenever the supplied
evidence permits it.

For every paper:
- state the population, imaging method/metric, brain region/network, direction,
  and main finding actually reported;
- explain exactly what agrees with or informs the current hypothesis;
- when the exact target region was not studied, identify the other reported
  regions and explain their anatomical or network relationship to the target;
- distinguish direct support, indirect support, conflicting evidence, and
  background-only evidence;
- state the evidence boundary and never upgrade indirect evidence to direct.
"""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _processed_ids(annotation_path: Path) -> set[str]:
    processed: set[str] = set()
    paths = [annotation_path, *sorted(annotation_path.parent.glob(f"{annotation_path.stem}_batch_*.json"))]
    for path in paths:
        if not path.exists():
            continue
        assignments = _load_json(path).get("assignments")
        if isinstance(assignments, dict):
            processed.update(str(key) for key in assignments)
    return processed


def _select_candidates(
    bank: dict[str, Any],
    truth: dict[str, Any],
    processed: set[str],
    count: int,
    status: str,
) -> list[dict[str, Any]]:
    status_by_id = {
        str(row.get("hypothesis_id")): str(row.get("status"))
        for row in truth.get("execution_results", [])
        if isinstance(row, dict)
    }
    eligible = [
        row
        for row in bank.get("hypotheses", [])
        if isinstance(row, dict)
        and str(row.get("id")) not in processed
        and (
            status == "any"
            or status_by_id.get(str(row.get("id"))) == status
        )
    ]
    if status == "any":
        eligible.sort(
            key=lambda row: (
                0 if status_by_id.get(str(row.get("id"))) == "confirmed" else 1,
                str(row.get("id")),
            )
        )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in eligible:
        candidate_tuple = row.get("metadata", {}).get("candidate_tuple", {})
        key = (
            str(candidate_tuple.get("region_name") or ""),
            str(candidate_tuple.get("feature_name") or ""),
            str(candidate_tuple.get("direction") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            return selected
    for row in eligible:
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def _paper_id(paper: dict[str, Any]) -> str:
    return str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "")


def _prompt(candidate: dict[str, Any]) -> str:
    candidate_tuple = candidate.get("metadata", {}).get("candidate_tuple", {})
    papers = []
    for paper in candidate.get("literature", []):
        papers.append(
            {
                "paper_id": _paper_id(paper),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "year": paper.get("year"),
                "journal": paper.get("journal"),
                "pmid": paper.get("pmid"),
                "doi": paper.get("doi"),
                "abstract": paper.get("abstract"),
                "excerpt": paper.get("excerpt"),
            }
        )
    input_payload = {
        "hypothesis_id": candidate.get("id"),
        "title": candidate.get("title"),
        "summary": candidate.get("summary"),
        "candidate_tuple": candidate_tuple,
        "papers": papers,
    }
    return f"""Assess this hypothesis and exactly its five papers:
{json.dumps(input_payload, ensure_ascii=False)}

Return one JSON object with:
{{
  "hypothesis_id": "...",
  "papers": [
    {{
      "paper_id": "...",
      "study_zh": "specific description of the paper's actual study and result",
      "study_en": "the same information in English",
      "support_zh": "specific support/conflict/relevance to this exact hypothesis",
      "support_en": "the same information in English",
      "strength_zh": "直接支持|中度间接支持|低度间接支持|冲突证据|仅背景相关",
      "strength_en": "direct support|moderate indirect support|weak indirect support|conflicting evidence|background only",
      "boundary_zh": "what remains untested or mismatched",
      "boundary_en": "the same information in English",
      "abstract_verified": true
    }}
  ]
}}

The papers array must contain exactly five objects in the supplied order.
Do not use generic element-count language such as "two of three elements match."
"""


def _request_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    candidate: dict[str, Any],
    timeout: float,
    retries: int,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(candidate)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 7000,
    }
    if reasoning_effort:
        request_payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed, {
                "model": raw.get("model") or model,
                "usage": raw.get("usage") or {},
                "finish_reason": raw["choices"][0].get("finish_reason"),
            }
        except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(30, 3 * attempt))
    raise RuntimeError(last_error)


def _validate(candidate: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    if str(result.get("hypothesis_id")) != str(candidate.get("id")):
        raise ValueError("hypothesis_id mismatch")
    rows = result.get("papers")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("response must contain exactly five paper assessments")
    expected = [_paper_id(paper) for paper in candidate.get("literature", [])]
    actual = [str(row.get("paper_id")) for row in rows if isinstance(row, dict)]
    if actual != expected:
        raise ValueError("paper order or identifiers do not match")
    required = (
        "study_zh",
        "study_en",
        "support_zh",
        "support_en",
        "strength_zh",
        "strength_en",
        "boundary_zh",
        "boundary_en",
    )
    for row in rows:
        if not isinstance(row, dict) or any(not str(row.get(key) or "").strip() for key in required):
            raise ValueError("an assessment is missing required bilingual fields")
        english = " ".join(
            str(row[key]) for key in ("study_en", "support_en", "strength_en", "boundary_en")
        )
        if any("\u4e00" <= char <= "\u9fff" for char in english):
            raise ValueError("Chinese text leaked into an English assessment field")
        if len(str(row["support_en"]).strip()) < 100 or len(str(row["support_zh"]).strip()) < 50:
            raise ValueError("a hypothesis-specific support explanation is too short")
        generic = f"{row['support_zh']} {row['support_en']}".lower()
        if "三个核心要素" in generic or "two of three elements" in generic:
            raise ValueError("a generic element-count explanation was returned")
    normalized_support = [
        " ".join(str(row["support_en"]).lower().split())
        for row in rows
    ]
    if len(normalized_support) != len(set(normalized_support)):
        raise ValueError("duplicate support explanations were returned")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    bank = _load_json(args.bank)
    truth = _load_json(args.truth)
    selected = _select_candidates(
        bank,
        truth,
        _processed_ids(args.annotations),
        args.batch_size,
        args.status,
    )
    if len(selected) != args.batch_size:
        raise RuntimeError(f"Only {len(selected)} eligible hypotheses were found")

    completed: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _request_json,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                candidate=candidate,
                timeout=args.timeout,
                retries=args.retries,
                reasoning_effort=args.reasoning_effort,
            ): candidate
            for candidate in selected
        }
        for future in as_completed(futures):
            candidate = futures[future]
            candidate_id = str(candidate["id"])
            try:
                result, metadata = future.result()
                completed[candidate_id] = (_validate(candidate, result), metadata)
                print(f"DONE {candidate_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures[candidate_id] = str(exc)
                print(f"FAILED {candidate_id}: {exc}", flush=True)

    raw_payload = {
        "schema_version": "1.0",
        "batch_id": args.batch_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "selected_ids": [str(row["id"]) for row in selected],
        "completed": {
            candidate_id: {"papers": rows, "metadata": metadata}
            for candidate_id, (rows, metadata) in completed.items()
        },
        "failures": failures,
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(selected)} hypotheses failed; raw output retained")

    paper_lookup = {
        _paper_id(paper): paper
        for candidate in selected
        for paper in candidate.get("literature", [])
    }
    paper_profiles: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    relation_profiles: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        candidate_id = str(candidate["id"])
        group_id = f"{args.batch_id}_{candidate_id}"
        assignments[candidate_id] = group_id
        relation_profiles[group_id] = {}
        rows, _metadata = completed[candidate_id]
        for row in rows:
            paper_id = str(row["paper_id"])
            source = paper_lookup[paper_id]
            paper_profiles.setdefault(
                paper_id,
                {
                    "study_zh": row["study_zh"],
                    "study_en": row["study_en"],
                    "abstract_verified": bool(row.get("abstract_verified"))
                    and bool(str(source.get("abstract") or "").strip()),
                    "abstract_source": "embedded_pubmed_abstract"
                    if str(source.get("abstract") or "").strip()
                    else "embedded_evidence_excerpt",
                },
            )
            relation_profiles[group_id][paper_id] = {
                key: row[key]
                for key in (
                    "support_zh",
                    "support_en",
                    "strength_zh",
                    "strength_en",
                    "boundary_zh",
                    "boundary_en",
                )
            }

    output = {
        "schema_version": "1.0",
        "current_batch": args.batch_id,
        "batches": [
            {
                "id": args.batch_id,
                "method": "api_abstract_and_excerpt_review",
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "hypothesis_count": len(selected),
                "paper_assessment_count": sum(len(rows) for rows, _ in completed.values()),
                "workers": args.workers,
            }
        ],
        "assignments": assignments,
        "paper_profiles": paper_profiles,
        "relation_profiles": relation_profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "hypotheses": len(assignments),
        "paper_assessments": sum(len(group) for group in relation_profiles.values()),
        "unique_papers": len(paper_profiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--status", default="confirmed")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=("low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--base-url", default="https://ai.aimgroup.chat/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--batch-id", default="batch11_api_confirmed_001_016")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "neurooracle/data/user_study/"
        "adni_manual_relevance_annotations_v1_batch_11_api_confirmed_001_016.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "outputs/user_study/api_batches/batch11_api_confirmed_001_016_raw.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
