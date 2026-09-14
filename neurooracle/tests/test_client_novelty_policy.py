"""Offline synthetic fixtures only; no literature/model/experiment requests."""
from copy import deepcopy

import pytest

from neurooracle.src import novelty_policy as policy


def review(label="substantive_extension"):
    return {"classification": label, "reference_ids": ["synthetic:reference"],
            "scientific_delta": "Synthetic difference, not a scientific claim.",
            "executable_test_covers_delta": True}


def candidate(hid="new", label="substantive_extension", quality=0.8):
    return {
        "hypothesis_id": hid, "structural_score": quality, "gnn_path_score": quality,
        "science": {"verdict": "pass", "critic_score": quality},
        "expert_novelties": [review(label) for _ in range(3)],
        "adjudication": {**review(label), "search_evidence_sufficient": True,
                         "registered_test_matches_hypothesis": True},
    }


@pytest.mark.parametrize("mode", policy.MODES)
@pytest.mark.parametrize("label", sorted(policy.CLASSES))
def test_preserves_classifications_and_input(mode, label):
    raw = [candidate(label=label)]
    before = deepcopy(raw)
    result = policy.select_reviewed(raw, mode)
    assert raw == before
    row = result["candidates"][0]
    assert row["literature_classification"] == label
    assert row["adjudication"] == raw[0]["adjudication"]
    assert not row["eligible_to_claim_new_finding"]
    assert bool(result["selected_ids"]) == (mode != "strict" or label in policy.PROMISING)


def test_tier_order_and_weighted_tradeoff():
    rows = [candidate("known", "exact_prior", 1), candidate("uncertain", "uncertain", 0.95),
            candidate("new", "substantive_extension", 0.6)]
    assert policy.select_reviewed(rows, "strict")["selected_ids"] == ["new"]
    result = policy.select_reviewed(rows, "novelty_first")
    assert result["selected_ids"] == ["new", "uncertain", "known"]
    assert result["candidates"][0]["selection_reason"] == "selected_known_replication_not_new_finding"
    assert policy.select_reviewed(rows, "weighted")["selected_ids"] == ["new", "uncertain", "known"]
    assert policy.select_reviewed([candidate("uncertain", "uncertain", .6), rows[0]], "weighted", 1)["selected_ids"] == ["known"]
    assert policy.select_reviewed(rows[:1], "strict")["selected_ids"] == []
    assert result["weights"] == {"novelty": .4, "structural": .2, "gnn": .2, "critic": .2}


@pytest.mark.parametrize("mode", policy.MODES)
def test_science_gate_and_empty_budget(mode):
    row = candidate()
    row["science"]["critic_score"] = 0.59
    assert policy.select_reviewed([row], mode)["selected_ids"] == []
    row = candidate()
    row["adjudication"]["registered_test_matches_hypothesis"] = False
    assert policy.select_reviewed([row], mode)["selected_ids"] == []
    assert policy.select_reviewed([candidate()], mode, 0)["selected_ids"] == []


@pytest.mark.parametrize("field,value", [
    ("reference_ids", []), ("scientific_delta", " "),
    ("executable_test_covers_delta", False), ("search_evidence_sufficient", False),
])
def test_missing_novelty_evidence_is_uncertain(field, value):
    row = candidate()
    row["adjudication"][field] = value
    row.update(novel_candidate_gate_passed=True, novelty_priority_points=1, eligible_to_claim_new_finding=True)
    result = policy.select_reviewed([row], "strict")
    assert not result["selected_ids"]
    assert result["candidates"][0]["novelty_priority_points"] == .2


def test_expert_prior_veto_cannot_be_averaged_away():
    row = candidate()
    row["expert_novelties"][2] = review("exact_prior")
    result = policy.select_reviewed([row], "strict")
    assert not result["selected_ids"]
    assert result["candidates"][0]["literature_classification"] == "substantive_extension"
    assert result["candidates"][0]["known_prior_veto"]


@pytest.mark.parametrize("mode", policy.MODES)
def test_stable_ties_and_prompt_scope(mode):
    assert policy.select_reviewed([candidate("b"), candidate("a")], mode, 1)["selected_ids"] == ["a"]
    prompt = policy.build_selection_prompt(mode)
    assert f"novelty_mode={mode}" in prompt
    assert "does not authorize experiments" in prompt
    assert "frozen/running" in prompt


@pytest.mark.parametrize("value", [None, "", "unselected", "STRICT", 1, True, [], {}])
def test_invalid_mode_is_not_silently_defaulted(value):
    with pytest.raises(ValueError):
        policy.select_reviewed([], value)


@pytest.mark.parametrize("value", [-1, True, .5, 2001, "5"])
def test_invalid_limit(value):
    with pytest.raises(ValueError):
        policy.select_reviewed([], limit=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1, True, "0.8", None])
def test_invalid_scores(value):
    row = candidate()
    row["gnn_path_score"] = value
    with pytest.raises(ValueError):
        policy.select_reviewed([row])


def test_rejects_duplicates_and_unreviewed_rows():
    for rows in ([candidate(), candidate()], [{}], ["bad"], None):
        with pytest.raises(ValueError):
            policy.select_reviewed(rows)


@pytest.mark.parametrize("field,value", [("classification", []), ("reference_ids", "fake"),
                                        ("executable_test_covers_delta", "false")])
def test_malformed_review_is_rejected(field, value):
    row = candidate()
    row["adjudication"][field] = value
    with pytest.raises(ValueError):
        policy.select_reviewed([row])
