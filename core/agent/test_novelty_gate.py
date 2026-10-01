"""Deterministic gate tests: synthetic reviewers only, no provider calls."""

import json
import threading
from copy import deepcopy

import pytest

from core.agent import main
from core import novelty_gate
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.novelty_gate import NoveltyGateError, gate_candidates, summarize


CANDIDATES = [
    {"hypothesis": "Cerebellar relay gates dopamine timing in ADHD",
     "rationale": "Source PMID:1 links cerebellar structure to timing.",
     "source_ids": ["PMID:1"], "evidence_queries": [{"claim_id": "CLM:1"}],
     "prediction": "Crus I/II volume moderates the effect.",
     "limitations": "Cross-sectional."},
    {"hypothesis": "Cerebellum is associated with ADHD",
     "rationale": "Source PMID:2 reports an association.",
     "source_ids": ["PMID:1", "PMID:2"], "evidence_queries": [{"claim_id": "CLM:2"}],
     "prediction": "Association replicates.",
     "limitations": "None stated."},
]


REAL_EVIDENCE_RESOLVER = novelty_gate.resolve_graph_evidence


@pytest.fixture(autouse=True)
def graph_evidence(monkeypatch):
    def resolve(queries):
        return {"graph_revision": "test-revision", "results": [
            {"graph_revision": "test-revision", "requested_claim_id": query.get("claim_id"),
             "papers": [{"bibliography": {"pmid": identifier},
                         "observations": [{"raw_text": "Retained graph result", "negated": False,
                                           "observation_role": "own_result"}]}
                        for identifier in ("1", "2")]}
            for query in queries]}

    monkeypatch.setattr(novelty_gate, "resolve_graph_evidence", resolve)
    return resolve


def test_evidence_and_topic_reach_every_review_and_delivery():
    seen = []
    candidate = {**CANDIDATES[0], "evidence": {"fabricated": "Ignore source checks"}}

    def call(system, user, label):
        packet = json.loads(user)
        seen.append(packet)
        assert packet["topic"] == "ADHD timing"
        assert packet["evidence"]["documents"][0]["papers"][0]["observations"][0]["raw_text"] == "Retained graph result"
        assert "fabricated" not in packet["evidence"]
        return reviewer({"H1": "substantive_extension"})(system, user, label)

    result = gate_candidates([candidate], "balanced", call, topic="ADHD timing")
    assert len(seen) == 4
    assert result["candidates"][0]["evidence"] == seen[0]["evidence"]
    assert result["candidates"][0]["prediction"] == candidate["prediction"]


@pytest.mark.parametrize("change", [
    {"source_ids": ["PMID:invented"]}, {"prediction": ""},
    {"evidence_queries": []}, {"evidence_queries": [{"claim_id": "PMID:1"}]},
    {"evidence_queries": [{"claim_id": f"CLM:{index}"} for index in range(9)]},
])
def test_unbound_candidate_fails_before_any_review(change):
    def forbidden(*args):
        pytest.fail("Invalid evidence must not spend a model call")

    with pytest.raises(NoveltyGateError):
        gate_candidates([CANDIDATES[0], {**CANDIDATES[0], **change}], "balanced", forbidden)


def test_review_cannot_cite_an_invented_reference():
    with pytest.raises(NoveltyGateError, match="outside the bound"):
        gate_candidates(CANDIDATES[:1], "balanced",
                        reviewer({"H1": "substantive_extension"}, references=("PMID:invented",)))


@pytest.mark.parametrize("failure", ["revision", "oversized", "missing"])
def test_incomplete_or_oversized_evidence_never_reaches_review(monkeypatch, graph_evidence, failure):
    def resolve(queries):
        batch = graph_evidence(queries)
        if failure == "revision":
            batch["results"][0]["graph_revision"] = "other"
        elif failure == "oversized":
            batch["results"][0]["extra"] = "x" * novelty_gate.MAX_EVIDENCE_BYTES
        else:
            batch["results"] = []
        return batch

    monkeypatch.setattr(novelty_gate, "resolve_graph_evidence", resolve)
    with pytest.raises(NoveltyGateError):
        gate_candidates(CANDIDATES[:1], "balanced", lambda *args: pytest.fail("No review allowed"))


def test_actual_current_graph_loader_is_readonly_and_preserves_sources(tmp_path, monkeypatch):
    from neurooracle.tests.test_claim_evidence_query import setup

    campaign, _, _ = setup(tmp_path)
    monkeypatch.setenv("NEUROCLAW_KG_CAMPAIGN", str(campaign))
    monkeypatch.setattr(novelty_gate, "resolve_graph_evidence", REAL_EVIDENCE_RESOLVER)
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    candidate = {**deepcopy(CANDIDATES[0]), "source_ids": ["PMID:111"],
                 "evidence_queries": [{"claim_id": "CLM:0"}]}
    result = gate_candidates([candidate], "balanced",
                            reviewer({"H1": "substantive_extension"}, references=("PMID:111",)))
    document = result["candidates"][0]["evidence"]["documents"][0]
    assert document["observation_count"] == 3
    assert {paper["bibliography"]["pmid"] for paper in document["papers"]} == {"111", "222"}
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}


def test_graph_unavailable_does_not_fall_back_to_candidate_evidence(monkeypatch):
    from core.web.claim_evidence import EvidenceUnavailable

    def unavailable(queries):
        raise EvidenceUnavailable("Stale graph input")

    monkeypatch.setattr(novelty_gate, "resolve_graph_evidence", unavailable)
    with pytest.raises(NoveltyGateError, match="Stale graph input"):
        gate_candidates([{**CANDIDATES[0], "evidence": {"verified": True}}], "balanced",
                        lambda *args: pytest.fail("No review allowed"))


def test_runtime_contract_explains_graph_binding_only_in_idea_mode(tmp_path):
    assert "evidence_queries" in AutoResearchRun(tmp_path / "idea", "idea").prompt()
    assert "evidence_queries" not in AutoResearchRun(tmp_path / "experiment", "experiment").prompt()


def reviewer(classification_by_id, *, verdict="pass", sufficient=True, covers=True,
             score=0.8, references=("PMID:1",), delta="Concrete documented difference."):
    """A stand-in reviewer that answers per candidate id and perspective label."""
    def call(system, user, label):
        packet = json.loads(user)
        hypothesis_id = label.split(" ", 1)[0]
        classification = classification_by_id.get(hypothesis_id, "uncertain")
        novel = classification in {"substantive_extension", "potential_new_relation"}
        extra = {}
        if 'chain' in packet:
            from core.idea_source_review import source_texts
            texts = source_texts(packet)
            extra['source_assessment'] = dict(bridge='compatible', support='supported', scope='compatible',
                relation='faithful', decision='retain', reason='Synthetic source fixture only.',
                anchors=[dict(source_id=s, quote=texts[s.casefold()][0]) for s in packet['source_ids'] if s.casefold() in texts])
        if 'source_refinement' in label or system.startswith('Refine one typed hypothesis'):
            return json.dumps({**{k: packet.get(k) or 'Synthetic limitation.' for k in
                                  ('hypothesis', 'prediction', 'rationale', 'limitations')}, **extra})
        return json.dumps({
            **extra,
            "classification": classification,
            "reference_ids": list(references),
            "scientific_delta": delta if novel else "",
            "executable_test_covers_delta": covers,
            "critic_score": score,
            "search_evidence_sufficient": sufficient,
            "registered_test_matches_hypothesis": True,
            "verdict": verdict,
            'response_to_peers': 'Synthetic response: retain the source-bound interpretation.',
        })
    return call


def test_novelty_first_excludes_known_conclusions_and_reports_veto():
    result = gate_candidates(CANDIDATES, "novelty_first",
                             reviewer({"H1": "substantive_extension", "H2": "exact_prior"}))
    assert result["selected_ids"] == ["H1"]
    tiers = {row["hypothesis_id"]: (row["novelty_tier"], row["known_prior_veto"]) for row in result["candidates"]}
    assert tiers == {"H1": (0, False), "H2": (2, True)}
    assert "known-prior veto raised for H2" in summarize(result)
    assert result["eligible_to_claim_new_finding"] is False


def test_novelty_first_can_select_nothing_rather_than_fill_a_quota():
    result = gate_candidates(CANDIDATES, "novelty_first",
                             reviewer({"H1": "exact_prior", "H2": "same_scientific_conclusion"}))
    assert result["selected_ids"] == []
    assert result["reviewed_candidates"] == 2


def test_the_two_novelty_gates_are_reported_separately():
    """A zero selection must be attributable to the science gate or the novelty gate."""
    # H1 clears the novelty gate but fails science; H2 fails novelty only.
    def call(system, user, label):
        hypothesis_id = label.split(" ", 1)[0]
        classification = "substantive_extension" if hypothesis_id == "H1" else "exact_prior"
        return json.dumps({
            "classification": classification, "reference_ids": ["PMID:1"],
            "scientific_delta": "Difference.", "executable_test_covers_delta": True,
            "critic_score": 0.3 if hypothesis_id == "H1" else 0.9,
            "search_evidence_sufficient": True,
            "registered_test_matches_hypothesis": True, "verdict": "pass",
        })
    result = gate_candidates(CANDIDATES[:2], "novelty_first", call)
    assert result["selected_ids"] == []
    assert result["science_gate_passed_ids"] == ["H2"]
    assert result["novel_gate_passed_ids"] == ["H1"]
    assert result["blocked_by_science_gate"] == ["H1"]
    assert result["blocked_by_novelty_gate"] == ["H2"]
    text = summarize(result)
    assert "science gate passed 1/2" in text
    assert "novelty gate passed 1" in text


def test_one_expert_prior_veto_cannot_be_averaged_away():
    def mixed(system, user, label):
        classification = "exact_prior" if label == "H1 expert:prior_art" else "substantive_extension"
        return json.dumps({"classification": classification, "reference_ids": ["PMID:1"],
                           "scientific_delta": "Difference.", "executable_test_covers_delta": True,
                           "critic_score": 0.95, "search_evidence_sufficient": True,
                           "registered_test_matches_hypothesis": True, "verdict": "pass"})
    result = gate_candidates(CANDIDATES[:1], "novelty_first", mixed)
    assert result["selected_ids"] == []
    assert result["candidates"][0]["known_prior_veto"] is True


def test_insufficient_search_evidence_is_uncertain_not_novel():
    result = gate_candidates(CANDIDATES[:1], "novelty_first",
                             reviewer({"H1": "substantive_extension"}, sufficient=False))
    assert result["selected_ids"] == []
    assert result["candidates"][0]["novelty_priority_points"] == 0.2


def test_invalid_reviewer_output_raises_instead_of_fabricating_a_verdict():
    with pytest.raises(NoveltyGateError):
        gate_candidates(CANDIDATES[:1], "novelty_first", lambda system, user, label: "not json")
    with pytest.raises(NoveltyGateError):
        gate_candidates(CANDIDATES[:1], "novelty_first",
                        lambda system, user, label: json.dumps({"classification": "substantive_extension"}))


def test_a_single_nonconforming_reply_is_reasked_then_succeeds():
    """Dropping a required key once must not disable the gate for the run."""
    attempts = {}

    def flaky(system, user, label):
        attempts[label] = attempts.get(label, 0) + 1
        if attempts[label] == 1:
            # First reply at each label: valid JSON, missing reference_ids.
            return json.dumps({"classification": "substantive_extension", "critic_score": 0.8,
                               "scientific_delta": "Delta.", "executable_test_covers_delta": True,
                               "search_evidence_sufficient": True,
                               "registered_test_matches_hypothesis": True, "verdict": "pass"})
        return reviewer({"H1": "substantive_extension"})(system, user, label)

    result = gate_candidates(CANDIDATES[:1], "novelty_first", flaky)
    assert result["selected_ids"] == ["H1"]
    assert set(attempts) == {"H1 expert:prior_art", "H1 expert:scientific_delta",
                             "H1 expert:operationalization", "H1 adjudication"}
    assert all(count == 2 for count in attempts.values())


def test_a_reply_that_is_still_nonconforming_after_the_reask_still_raises():
    calls = []

    def always_bad(system, user, label):
        calls.append((label, "previous reply" in system))
        return json.dumps({"classification": "substantive_extension", "critic_score": 0.8})

    with pytest.raises(NoveltyGateError):
        gate_candidates(CANDIDATES[:1], "novelty_first", always_bad)
    # The first non-conforming label is asked at most twice, and the re-ask
    # carries the corrective suffix; the gate then raises rather than continuing.
    assert calls == [("H1 expert:prior_art", False), ("H1 expert:prior_art", True)]


def test_gate_requires_a_valid_mode():
    with pytest.raises(ValueError):
        gate_candidates(CANDIDATES[:1], "permissive", reviewer({"H1": "substantive_extension"}))


def test_three_experts_plus_adjudication_are_separate_calls_per_candidate():
    """Three independent expert votes must come from three distinct calls."""
    labels = []

    def call(system, user, label):
        labels.append(label)
        return reviewer({"H1": "substantive_extension"})(system, user, label)

    gate_candidates(CANDIDATES[:2], "novelty_first", call)
    assert labels == [
        "H1 expert:prior_art", "H1 expert:scientific_delta", "H1 expert:operationalization",
        "H1 adjudication",
        "H2 expert:prior_art", "H2 expert:scientific_delta", "H2 expert:operationalization",
        "H2 adjudication",
    ]
    assert len(set(labels)) == len(labels)


def test_unmeasured_scores_are_disclosed_not_invented():
    result = gate_candidates(CANDIDATES[:1], "balanced",
                             reviewer({"H1": "substantive_extension"}))
    assert result["unmeasured_scores"] == ["structural_score", "gnn_path_score"]
    assert result["candidates"][0]["structural_score"] is None
    assert "not measured" in summarize(result)
    assert "falls back to novelty and review" in summarize(result)


@pytest.mark.parametrize("record,marker", [
    ({"error": "no_gateable_candidates", "message": "No candidate JSON was submitted."},
     "no_gateable_candidates"),
    ({"error": "novelty_gate_unavailable", "message": "offline"}, "novelty_gate_unavailable"),
    ({"error": "novelty_gate_attempts_exhausted", "message": "cap reached"},
     "novelty_gate_attempts_exhausted"),
    # An interrupted run can persist a gate record with no review outcome at all.
    ({}, "no review outcome"),
    ({"novelty_mode": "novelty_first"}, "no review outcome"),
])
def test_incomplete_gate_records_are_summarizable(record, marker):
    """An unavailable or incomplete gate must report its gap, never crash the summary."""
    text = summarize(record)
    assert marker in text
    assert "not a new-finding claim" in text


def test_response_survives_an_error_only_gate_record(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    run.state.update(status="stalled", summary="stalled", artifacts=[],
                     novelty_gate={"error": "novelty_gate_unavailable", "message": "no budget"})
    assert "novelty gate unavailable" in run.response()


def gate_agent(tmp_path, responses, monkeypatch, candidates):
    """A session whose submitted hypotheses file is the gate's subject.

    The candidate file is created by the real ``write_research_file`` tool, which
    opens with mode ``x``; pre-writing it here would make the write fail and the
    submission would be refused for missing execution evidence before the gate ran.
    """
    from core.test_idea_hypotheses import synthetic_candidate, synthetic_batch
    candidates = [{**synthetic_candidate(), **{key: row[key] for key in ('hypothesis', 'rationale', 'prediction')}} for row in candidates]
    monkeypatch.setattr(novelty_gate, 'resolve_graph_evidence', synthetic_batch)
    finishes = [reply(calls=[tool("finish_autoresearch", **completion(
        artifacts=["candidates.json"], evidence_ids=[1]))]) for _ in range(4)]
    sequence = [reply(calls=[tool("write_research_file", path="candidates.json", kind="hypotheses",
                                  content=json.dumps(candidates))])] + finishes + responses
    agent, calls = session(tmp_path, sequence, mode="idea")
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "70")
    agent.env["autoresearch_novelty_gate"] = True
    agent.env["autoresearch_independent_review"] = False
    agent.novelty_mode = "novelty_first"
    return agent, calls


def _gate_refusals(call):
    """The gate refusals present in one model call's conversation history."""
    return [message for message in call.get("messages", [])
            if message.get("role") == "tool" and "novelty_gate_failed" in str(message.get("content"))]


def test_gate_refuses_completion_for_known_prior_candidates(tmp_path, monkeypatch):
    agent, calls = gate_agent(tmp_path, [], monkeypatch, CANDIDATES)
    # Every expert vote answers "exact_prior", so the novelty_first gate selects nothing.
    agent._review_llm = _ReviewerClient(lambda **kw: reply(reviewer({'H1': 'exact_prior'}, covers=False)(
        kw['messages'][0]['content'], kw['messages'][1]['content'], 'H1 synthetic')))
    result = agent._chat()
    state = agent.autoresearch_state
    gate = state["novelty_gate"]
    assert gate["reviewed_candidates"] == 2
    assert gate["selected_ids"] == []
    assert {row["hypothesis_id"] for row in gate["candidates"] if row["known_prior_veto"]} == {"H1", "H2"}
    assert state["novelty_gate_status"] == "no_eligible_candidate"
    # The gate is binding until its bounded release: three refused completions.
    assert state["novelty_gate_rejections"] == 3
    assert _gate_refusals(calls[-1])  # Large chain evidence can compact earlier refusals out of context.
    assert "novelty_selection" not in state
    assert "known-prior veto raised for H1, H2" in result
    assert "eligible_to_claim_new_finding" not in state or gate["eligible_to_claim_new_finding"] is False


def test_gate_accepts_a_candidate_that_survives_review(tmp_path, monkeypatch):
    agent, calls = gate_agent(tmp_path, [], monkeypatch, CANDIDATES[:1])
    agent._review_llm = _ReviewerClient(lambda **kw: reply(reviewer({'H1': 'substantive_extension'}, score=.9)(
        kw['messages'][0]['content'], kw['messages'][1]['content'], 'H1 synthetic')))
    result = agent._chat()
    state = agent.autoresearch_state
    assert state["novelty_gate_status"] == "selection_present"
    assert state["novelty_selection"]["selected_ids"] == ["H1"]
    assert state["novelty_gate"]["candidates"][0]["known_prior_veto"] is False
    assert state["novelty_gate"]["candidates"][0]["novelty_vote_count"] == 3
    assert state["status"] == "completed"
    assert _gate_refusals(calls[-1]) == []
    assert "Novelty gate" in result


def test_gate_refusals_are_bounded_so_a_run_cannot_deadlock(tmp_path, monkeypatch):
    run = AutoResearchRun(tmp_path, "idea")
    (tmp_path / "candidates.json").write_text(json.dumps(CANDIDATES), encoding="utf-8")
    run.state.update(artifacts=[str(tmp_path / "candidates.json")], novelty_gate_required=True,
                     novelty_gate={"error": "novelty_gate_unavailable", "message": "offline"})
    agent, _ = session(tmp_path, [])
    agent.novelty_mode = "novelty_first"
    assert agent._novelty_gate_rejection(run, "offline") is not None
    assert agent._novelty_gate_rejection(run, "offline") is not None
    assert agent._novelty_gate_rejection(run, "offline") is not None
    assert agent._novelty_gate_rejection(run, "offline") is None


def test_unavailable_gate_never_claims_novelty(tmp_path, monkeypatch):
    run = AutoResearchRun(tmp_path, "idea")
    (tmp_path / "candidates.json").write_text(json.dumps(CANDIDATES), encoding="utf-8")
    run.state.update(artifacts=[str(tmp_path / "candidates.json")], novelty_gate_required=True)
    agent, _ = session(tmp_path, [])
    agent.novelty_mode = "novelty_first"
    agent._gate_idea_candidates = lambda *a, **k: (_ for _ in ()).throw(NoveltyGateError("no budget"))
    rejection = agent._novelty_gate_rejection(run, "offline")
    assert rejection is not None and rejection["error_type"] == "novelty_gate_failed"
    assert run.state["novelty_gate_status"] == "unavailable"
    assert "novelty_selection" not in run.state


def test_candidate_rewrites_cannot_buy_unlimited_gate_reviews(tmp_path, monkeypatch):
    """The gate review count is capped; rewrites after the cap are not re-reviewed."""
    run = AutoResearchRun(tmp_path, "idea")
    path = tmp_path / "candidates.json"
    run.state.update(artifacts=[str(path)], novelty_gate_required=True)
    agent, _ = session(tmp_path, [])
    agent.novelty_mode = "novelty_first"
    reviews = []
    agent._gate_idea_candidates = lambda run_, model_, cands: reviews.append(cands) or {
        "selected_ids": [], "candidates": [], "reviewed_candidates": len(cands)}
    for index in range(6):
        path.write_text(json.dumps([{**CANDIDATES[0], "rationale": f"rewrite {index}"}]), encoding="utf-8")
        agent._novelty_gate_rejection(run, "offline")
    assert len(reviews) == 3
    assert run.state["novelty_gate_attempts"] == 3
    assert run.state["novelty_gate_status"] == "unavailable"
    assert run.state["novelty_gate"]["error"] == "novelty_gate_attempts_exhausted"


def test_rewriting_a_selected_candidate_invalidates_its_old_review(tmp_path):
    run = AutoResearchRun(tmp_path, 'idea')
    path = tmp_path / 'candidates.json'
    path.write_text(json.dumps(CANDIDATES[:1]), encoding='utf-8')
    run.state.update(artifacts=[str(path)], novelty_gate_required=True,
                     novelty_selection={'selected_ids': ['H1']}, novelty_gate_digest='old-payload')
    agent, _ = session(tmp_path, [])
    reviews = []
    agent._gate_idea_candidates = lambda *args: reviews.append(args) or dict(selected_ids=[], candidates=[], reviewed_candidates=1)
    assert agent._novelty_gate_rejection(run, 'offline') is not None
    assert len(reviews) == 1 and 'novelty_selection' not in run.state


class _ReviewerClient:
    """Minimal client exposing only chat.completions.create for gate/review calls."""

    def __init__(self, create):
        self.chat = type("C", (), {"completions": type("K", (), {"create": staticmethod(create)})()})()
