"""Hypothesis tournament: pairwise LLM-judge + Bradley-Terry scoring.

Adapted from FutureHouse/Robin (utils.py:422-820, MIT License).
Stripped of Edison platform dependencies and made LLM-backend agnostic.

Usage (programmatic):
    from skills.method_design.scripts.tournament import run_tournament
    ranked = await run_tournament(
        candidates=[
            {"index": 0, "name": "GCN+SABP", "rationale": "..."},
            {"index": 1, "name": "Transformer+TopK", "rationale": "..."},
            ...
        ],
        judge_system_prompt=ARCHITECTURE_JUDGE_SYSTEM_PROMPT,
        judge_format=ARCHITECTURE_JUDGE_FORMAT,
        llm_call=my_llm_call,  # async (messages: list[dict]) -> str
    )
    # ranked is a pandas DataFrame sorted by strength_score desc

Usage (CLI):
    python skills/method-design/scripts/tournament.py \
        --candidates path/to/candidates.json \
        --output path/to/ranking.csv \
        --backend openai --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any, Awaitable, Callable

import choix
import pandas as pd

logger = logging.getLogger(__name__)

LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


ARCHITECTURE_JUDGE_SYSTEM_PROMPT = """\
You are a senior neuroimaging methods reviewer. Compare two candidate \
neural-network architecture proposals on the following criteria, in order \
of importance:

1. Methodological soundness for neuroimaging modality at hand
   (sMRI/fMRI/dMRI/EEG/MEG/PET/ASL).
2. Novelty relative to recent literature; avoid trivial re-skins.
3. Computational tractability on common research GPUs (24-80 GB).
4. Compatibility with NeuroClaw skill registry (BIDS-staged inputs,
   PyG/PyTorch outputs).
5. Clarity and falsifiability of the claimed mechanism.

You must pick exactly one Winner and one Loser. Ties are not allowed; \
break them by criterion 1, then 2, then 3.
"""

ARCHITECTURE_JUDGE_FORMAT = """\
{
  "Analysis": "<concise comparative analysis, <=200 words>",
  "Reasoning": "<which criteria decided the call>",
  "Winner": <winner_id_int>,
  "Loser":  <loser_id_int>
}
"""


def uniformly_random_pairs(
    n_candidates: int,
    seed: int = 621,
    n_games: int | None = None,
) -> list[tuple[int, int]]:
    """Sample distinct unordered pairs uniformly without replacement.

    For small n_candidates (<= 25), defaults to the full pairing
    (n*(n-1)/2 pairs); otherwise capped at 300 to bound LLM cost.
    """
    if n_candidates < 2:
        return []

    random.seed(seed)
    max_pairs = n_candidates * (n_candidates - 1) // 2
    if n_games is None:
        n_games = min(300, max_pairs)
    n_games = min(n_games, max_pairs)

    all_pairs = list(itertools.combinations(range(n_candidates), 2))
    return random.sample(all_pairs, n_games)


EXPECTED_KEYS = ("Analysis", "Reasoning", "Winner", "Loser")


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON in LLM response: {e}; raw={blob[:200]!r}") from e


async def _compare_pair(
    pair: tuple[int, int],
    candidates_df: pd.DataFrame,
    judge_system_prompt: str,
    judge_format: str,
    llm_call: LLMCall,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run a single LLM-as-judge comparison for one pair (a, b)."""
    a_id, b_id = pair
    a = candidates_df.iloc[a_id]
    b = candidates_df.iloc[b_id]

    user_prompt = f"""\
Compare the two candidates below. Respond with a single valid JSON object \
in the exact schema shown. Do not include any text before or after the JSON.

**Candidate A (id={a_id})**
Name: {a['name']}
Rationale: {a['rationale']}

--- VERSUS ---

**Candidate B (id={b_id})**
Name: {b['name']}
Rationale: {b['rationale']}

Schema:
{judge_format}
"""

    messages = [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    async with semaphore:
        try:
            raw = await llm_call(messages)
            verdict = _extract_json_object(raw)
            for k in EXPECTED_KEYS:
                if k not in verdict:
                    raise ValueError(f"Missing key {k!r} in verdict: {verdict}")
            winner = int(verdict["Winner"])
            loser = int(verdict["Loser"])
            if {winner, loser} != {a_id, b_id}:
                raise ValueError(
                    f"Verdict ids {winner},{loser} do not match pair {pair}"
                )
            return {
                "status": "success",
                "pair": pair,
                "winner": winner,
                "loser": loser,
                "analysis": verdict["Analysis"],
                "reasoning": verdict["Reasoning"],
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Pair %s failed: %s", pair, e)
            return {"status": "error", "pair": pair, "error": str(e)}


async def run_tournament(
    candidates: list[dict[str, Any]],
    llm_call: LLMCall,
    judge_system_prompt: str = ARCHITECTURE_JUDGE_SYSTEM_PROMPT,
    judge_format: str = ARCHITECTURE_JUDGE_FORMAT,
    seed: int = 621,
    max_concurrent: int = 16,
    bt_alpha: float = 0.1,
) -> pd.DataFrame:
    """Run a full pairwise tournament and return ranked candidates.

    Args:
        candidates: list of dicts with keys "name" and "rationale".
            "index" is auto-assigned by position; do not rely on caller's.
        llm_call: async callable mapping messages to raw text.
        judge_system_prompt / judge_format: override for non-architecture
            tournaments (e.g. "best research idea", "best loss formulation").
        seed: RNG seed for pair sampling.
        max_concurrent: cap on simultaneous LLM calls.
        bt_alpha: choix.ilsr_pairwise regularization.

    Returns:
        DataFrame with columns
        [index, name, rationale, strength_score], sorted desc by score.
        If <2 candidates, returns the input as-is with score=NaN.
    """
    if len(candidates) < 2:
        df = pd.DataFrame(candidates)
        df["strength_score"] = float("nan")
        return df

    df = pd.DataFrame(candidates).reset_index(drop=True)
    for col in ("name", "rationale"):
        if col not in df.columns:
            raise KeyError(f"candidates must contain '{col}' field")
    df["index"] = df.index

    pairs = uniformly_random_pairs(len(df), seed=seed)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _compare_pair(
            p, df, judge_system_prompt, judge_format, llm_call, semaphore
        )
        for p in pairs
    ]
    results = await asyncio.gather(*tasks)

    games: list[tuple[int, int]] = [
        (r["winner"], r["loser"])
        for r in results
        if r["status"] == "success"
    ]

    successes = len(games)
    failures = len(results) - successes
    logger.info(
        "Tournament complete: %d/%d pairs scored (failures=%d)",
        successes, len(pairs), failures,
    )

    if not games:
        df["strength_score"] = float("nan")
        return df

    try:
        params = choix.ilsr_pairwise(len(df), games, alpha=bt_alpha)
    except Exception:
        logger.exception("choix.ilsr_pairwise failed; returning NaN scores")
        df["strength_score"] = float("nan")
        return df

    df["strength_score"] = params
    return df.sort_values("strength_score", ascending=False).reset_index(drop=True)


def _save_ranking(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved ranking to %s", output_path)


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of candidates")
    for i, c in enumerate(data):
        if "name" not in c or "rationale" not in c:
            raise ValueError(f"Candidate {i} missing 'name' or 'rationale'")
    return data


def _build_llm_call(backend: str, model: str) -> LLMCall:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_client import build_llm_call  # noqa: E402
    return build_llm_call(backend=backend, model=model)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to candidates.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output ranking CSV path")
    parser.add_argument("--backend", choices=["openai", "anthropic"],
                        default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=621)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--bt-alpha", type=float, default=0.1)
    parser.add_argument("--prompt-mode",
                        choices=["architecture", "idea", "loss"],
                        default="architecture")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    candidates = _load_candidates(args.candidates)
    llm_call = _build_llm_call(args.backend, args.model)

    ranked = asyncio.run(run_tournament(
        candidates=candidates,
        llm_call=llm_call,
        seed=args.seed,
        max_concurrent=args.max_concurrent,
        bt_alpha=args.bt_alpha,
    ))
    _save_ranking(ranked, args.output)


if __name__ == "__main__":
    _main()
