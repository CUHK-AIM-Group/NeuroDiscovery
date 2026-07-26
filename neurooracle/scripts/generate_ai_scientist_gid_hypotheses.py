from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Any


SYSTEM_PROMPT = """You are reproducing the hypothesis-generation component of
the AI Scientist system described in the published Nature paper "Towards
end-to-end automation of AI research".

Your task is NOT to use NeuroOracle, NeuroDiscovery, curated KG traversal, or
domain-specific graph scoring. You may use only general scientific reasoning
and broad biomedical/neuroscience knowledge, as a general-purpose autoresearch
system would do.

Generate ranked hypotheses for a neuroscience hindcasting benchmark. Each
hypothesis should attempt to follow this strict structure:

GENE_TARGET -> IMAGING_MARKER -> DISEASE

The imaging marker must be a concrete measurement, not a modality name. Valid
examples include posterior putamen dopamine transporter binding, hippocampal
volume, entorhinal cortical thickness, amyloid PET SUVR, FDG hypometabolism,
striatal dopamine transporter specific binding ratio, regional gray matter
volume, or named-network resting-state connectivity.

Invalid examples include MRI, fMRI, PET, SPECT, CT, neuroimaging, brain scan,
or a bare anatomical region without a measured feature.

Return JSON only."""


def load_env_keys() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env.keys"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPENAI_API_KEYS"):
        first_key = os.environ["OPENAI_API_KEYS"].split(",")[0].strip()
        if first_key:
            os.environ["OPENAI_API_KEY"] = first_key


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object with a hypotheses list.")
    return payload


def proxy_config() -> dict[str, str] | None:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def http_chat_completion(
    *,
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required for HTTP chat completions.")
    url = f"{base_url}/chat/completions"
    payload_bytes = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=payload_bytes,
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 autoresearch-benchmark/1.0",
        },
        method="POST",
    )
    proxy = proxy_config()
    opener = urllib_request.build_opener(urllib_request.ProxyHandler(proxy or {}))
    try:
        with opener.open(req, timeout=float(os.environ.get("OPENAI_TIMEOUT") or 240.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:800]}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    return str(payload["choices"][0]["message"]["content"] or "")


def call_model(*, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    wire_api = os.environ.get("OPENAI_WIRE_API", "").strip().lower()
    if wire_api in {"chat_completions", "http_chat_completions"}:
        return http_chat_completion(
            model=model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": os.environ.get("OPENAI_API_KEY")}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_TIMEOUT"):
        kwargs["timeout"] = float(os.environ["OPENAI_TIMEOUT"])
    client = OpenAI(**kwargs)
    if wire_api == "responses":
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}},
        )
        return response.output_text or ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def generation_prompt(batch_n: int, start_rank: int, previous_names: list[str], seed: int) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-80:]) or "(none)"
    return f"""Generate exactly {batch_n} new hypotheses for ranks {start_rank} to {start_rank + batch_n - 1}.

This is a fixed-budget benchmark. Do not skip uncertain items. If you are
uncertain, still provide your best attempt; invalid or vague hypotheses will be
counted as failed hypotheses by the evaluator.

Replicate seed: {seed}. Use it only to diversify hypothesis selection and
ordering across independent benchmark repetitions.

Avoid duplicating these already generated names:
{previous}

Return one JSON object with this exact shape:
{{
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Every object must attempt GENE_TARGET -> IMAGING_MARKER -> DISEASE.
- Use concrete disease names, not broad phrases like neurological disorders.
- Use concrete imaging measurements, not modality names.
- Confidence should be a number from 0 to 1 and reflect your own uncertainty.
- JSON only; no markdown."""


def normalize_hypothesis(row: dict[str, Any], rank: int) -> dict[str, Any]:
    name = str(row.get("name") or row.get("Name") or f"ai_scientist_{rank:04d}")
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or f"ai_scientist_{rank:04d}"
    return {
        "rank": rank,
        "Name": name,
        "Title": str(row.get("title") or row.get("Title") or name.replace("_", " ")),
        "gene_target": str(row.get("gene_target") or row.get("gene") or ""),
        "imaging_marker": str(row.get("imaging_marker") or row.get("imaging") or ""),
        "disease": str(row.get("disease") or ""),
        "Short Hypothesis": (
            f"{row.get('gene_target') or row.get('gene') or ''} -> "
            f"{row.get('imaging_marker') or row.get('imaging') or ''} -> "
            f"{row.get('disease') or ''}. "
            f"{row.get('mechanistic_rationale') or ''}"
        ).strip(),
        "mechanistic_rationale": str(row.get("mechanistic_rationale") or ""),
        "expected_direction": str(row.get("expected_direction") or ""),
        "confidence": float(row.get("confidence") or 0.0) if str(row.get("confidence") or "").strip() else 0.0,
    }


def failure_placeholder(rank: int, reason: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "Name": f"generation_failure_{rank:04d}",
        "Title": "Generation failure",
        "gene_target": "",
        "imaging_marker": "",
        "disease": "",
        "Short Hypothesis": "",
        "mechanistic_rationale": "",
        "expected_direction": "",
        "confidence": 0.0,
        "generation_failure": reason,
    }


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_env_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set and no usable key was found in .env.keys")

    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        prompt = generation_prompt(batch_n, start_rank, names, args.seed)
        parsed_rows: list[dict[str, Any]] = []
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                text = call_model(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                payload = parse_json_object(text)
                rows = payload.get("hypotheses")
                if not isinstance(rows, list):
                    raise ValueError("JSON object does not contain a hypotheses list.")
                parsed_rows = [row for row in rows if isinstance(row, dict)]
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)
        if len(parsed_rows) < batch_n:
            parsed_rows.extend(
                {"generation_failure": last_error or "model returned too few hypotheses"}
                for _ in range(batch_n - len(parsed_rows))
            )
        for row in parsed_rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
            else:
                hyp = normalize_hypothesis(row, rank)
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed-budget AI Scientist GID hypothesis list.")
    parser.add_argument("--n", type=int, default=1000, help="Fixed number of hypotheses to generate.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", default=os.environ.get("AI_SCIENTIST_V2_MODEL", "gpt-5.4"))
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/ai_scientist_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
