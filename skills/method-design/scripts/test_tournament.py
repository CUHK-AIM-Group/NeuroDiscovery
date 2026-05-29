"""Offline tests for tournament.py — no network, no API keys."""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tournament import run_tournament, uniformly_random_pairs  # noqa: E402


def _mock_llm_factory(winner_by_pair: dict[tuple[int, int], int]):
    """Return an async LLM that always picks a designated winner."""
    async def _call(messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        ids = [int(m) for m in re.findall(r"id=(\d+)", user)]
        a_id, b_id = ids[0], ids[1]
        key = tuple(sorted((a_id, b_id)))
        winner = winner_by_pair.get(key, a_id)
        loser = b_id if winner == a_id else a_id
        return json.dumps({
            "Analysis": f"mock {a_id} vs {b_id}",
            "Reasoning": "mock",
            "Winner": winner,
            "Loser": loser,
        })
    return _call


def test_uniformly_random_pairs_full_coverage():
    pairs = uniformly_random_pairs(4, seed=42)
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    for a, b in pairs:
        assert a != b


def test_uniformly_random_pairs_below_min():
    assert uniformly_random_pairs(0) == []
    assert uniformly_random_pairs(1) == []


def test_run_tournament_clear_winner():
    candidates = [
        {"name": "Strong", "rationale": "Strong rationale"},
        {"name": "Mid",    "rationale": "Mid rationale"},
        {"name": "Weak",   "rationale": "Weak rationale"},
    ]
    winners = {(0, 1): 0, (0, 2): 0, (1, 2): 1}
    llm = _mock_llm_factory(winners)

    ranked = asyncio.run(run_tournament(
        candidates=candidates, llm_call=llm, max_concurrent=4,
    ))
    assert list(ranked["name"]) == ["Strong", "Mid", "Weak"]
    assert ranked["strength_score"].iloc[0] > ranked["strength_score"].iloc[1]
    assert ranked["strength_score"].iloc[1] > ranked["strength_score"].iloc[2]


def test_run_tournament_single_candidate_returns_nan():
    ranked = asyncio.run(run_tournament(
        candidates=[{"name": "Only", "rationale": "Only one"}],
        llm_call=_mock_llm_factory({}),
    ))
    assert len(ranked) == 1
    assert math.isnan(ranked["strength_score"].iloc[0])


def test_run_tournament_handles_judge_failure_gracefully():
    """If the LLM returns malformed JSON, that pair is dropped."""
    async def _bad_llm(messages):
        return "not json at all"

    candidates = [
        {"name": "A", "rationale": "a"},
        {"name": "B", "rationale": "b"},
    ]
    ranked = asyncio.run(run_tournament(
        candidates=candidates, llm_call=_bad_llm,
    ))
    assert len(ranked) == 2
    assert ranked["strength_score"].isna().all()


def test_run_tournament_missing_required_field():
    with pytest.raises(KeyError):
        asyncio.run(run_tournament(
            candidates=[{"name": "X"}, {"name": "Y"}],
            llm_call=_mock_llm_factory({}),
        ))
