from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.scripts.case_study_official_adapter_client as official_adapter
from core.scripts.case1_kg_stream import (
    case1_kg_cache_path,
    load_case1_kg_index_payload,
)
from core.scripts.case1_official_adapter_client import (
    compile_native_policy,
    run_ai_scientist,
    run_native,
    run_open_coscientist,
    sciagents_llm_config,
)
from core.scripts.case_study_official_adapter_client import (
    _open_coscientist_debate_cohorts,
    _open_coscientist_evolution_guidance,
    native_workflow_audit,
)
from core.scripts.case1_policy_audit import audit_policy_independence
from core.scripts.case1_search_policy import (
    PolicyAnchor,
    PolicyRule,
    SearchPolicy,
    build_public_registry,
    classify_observed_feedback,
    compile_policy_order,
    compile_policy_scores,
    policy_from_mapped_hypotheses,
    policy_from_payload,
)
from core.scripts.case1_method_comparison import closed_loop_neurodiscovery_order
from core.scripts.case1_native_baseline_experiment import (
    extract_json,
    map_validated,
    validate_seed_outputs,
)


def candidate_frame() -> pd.DataFrame:
    rows = []
    for disease in ("ADHD", "MDD_depression"):
        for feature in ("corr_mean", "roi_alff_proxy"):
            for roi_index, roi_name, hemisphere, group in (
                (1, "Anterior cingulate", "left", "Control"),
                (2, "Insula", "right", "Salience/VAttn"),
            ):
                rows.append(
                    {
                        "candidate_id": f"func|schaefer200|{disease}|{feature}|{roi_index}",
                        "disease": disease,
                        "feature": feature,
                        "modality": "func",
                        "source": "schaefer200",
                        "roi_index": roi_index,
                        "roi_name": roi_name,
                        "anatomy_full": roi_name,
                        "hemisphere": hemisphere,
                        "map_group": group,
                        "network": group,
                        "structure_class": "cortical",
                        "adjusted_residual_d": 0.8,
                        "p_value": 1e-6,
                        "is_gt_top": True,
                        "is_strict_fdr": True,
                        "abs_adjusted_residual_d": 0.8,
                    }
                )
    return pd.DataFrame(rows)


def test_sciagents_config_retries_provider_concurrency_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CS1_LOCAL_API_KEY", "test-key")
    monkeypatch.setenv("SCIAGENTS_LLM_MAX_RETRIES", "17")
    args = SimpleNamespace(
        model="gpt-5.5",
        base_url="http://localhost:8080/v1",
        reasoning_effort="high",
    )

    config = sciagents_llm_config(args)

    assert config["config_list"][0]["max_retries"] == 17
    assert config["reasoning_effort"] == "high"


def test_sciagents_config_leaves_provider_retries_to_router_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CS1_LOCAL_API_KEY", "test-key")
    monkeypatch.delenv("SCIAGENTS_LLM_MAX_RETRIES", raising=False)
    args = SimpleNamespace(
        model="deepseek-v4-pro",
        base_url="http://localhost:18182/v1",
        reasoning_effort="high",
    )

    config = sciagents_llm_config(args)

    assert config["config_list"][0]["max_retries"] == 0


def test_pubmed_fallback_relaxes_overconstrained_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_terms: list[str] = []

    class FakeResponse:
        def __init__(self, *, payload: dict | None = None, content: bytes = b"") -> None:
            self.payload = payload
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            assert self.payload is not None
            return self.payload

    article_xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>123</PMID><Article><ArticleTitle>Relevant study</ArticleTitle>
    <Abstract><AbstractText>Relevant abstract.</AbstractText></Abstract>
    <Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2025</Year>
    </PubDate></JournalIssue></Journal></Article></MedlineCitation>
    </PubmedArticle></PubmedArticleSet>"""

    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        del timeout
        if "esearch.fcgi" in url:
            search_terms.append(str(params["term"]))
            ids = [] if len(search_terms) == 1 else ["123"]
            return FakeResponse(payload={"esearchresult": {"idlist": ids}})
        return FakeResponse(content=article_xml)

    official_adapter._PUBMED_CACHE.clear()
    official_adapter._RETRIEVAL_AUDIT.clear()
    monkeypatch.setattr(official_adapter, "_PUBMED_LAST_REQUEST", 0.0)
    monkeypatch.setattr(official_adapter.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("requests.get", fake_get)

    papers = official_adapter._pubmed_fallback_papers(
        "resting-state fMRI transdiagnostic psychiatric disorders regional "
        "functional connectivity alterations PTSD depression ADHD OCD substance "
        "use reproducibility",
        3,
    )

    assert len(papers) == 1
    assert papers[0]["paperId"] == "PMID:123"
    assert len(search_terms) == 2
    assert len(search_terms[1].split()) < len(search_terms[0].split())
    audit = official_adapter._RETRIEVAL_AUDIT[-1]
    assert audit["query_relaxed"] is True
    assert [attempt["pmids"] for attempt in audit["search_attempts"]] == [0, 1]


def test_pubmed_relaxation_removes_benchmark_protocol_terms() -> None:
    variants = official_adapter._pubmed_query_variants(
        "Outcome-blind fMRI case-control tests for reproducible alterations in: "
        "MDD_depression left thalamus partial functional connectivity"
    )

    assert "fMRI depression thalamus functional" in variants


def test_native_workflow_audit_records_roles_tools_and_citations() -> None:
    sciagents = native_workflow_audit(
        "sciagents",
        {
            "participating_roles": ["planner", "assistant", "critic_agent"],
            "all_upstream_roles_participated": True,
            "chat_history": [
                {
                    "tool_calls": [
                        {"function": {"name": "generate_path"}},
                        {"function": {"name": "rate_novelty_feasibility"}},
                    ]
                }
            ],
        },
    )
    assert sciagents["all_upstream_roles_participated"] is True
    assert sciagents["tool_calls_by_name"] == {
        "generate_path": 1,
        "rate_novelty_feasibility": 1,
    }

    open_coscientist = native_workflow_audit(
        "open_coscientist",
        {
            "hypotheses": [
                {
                    "literature_grounding": "Supported by C1.",
                    "citation_map": {"C1": {"title": "Paper"}},
                }
            ],
            "meta_review": {"summary": "reviewed"},
            "tournament_matchups": [{"winner": 0}],
            "evolution_details": [{"index": 0}],
        },
    )
    assert open_coscientist["hypotheses_with_literature_grounding"] == 1
    assert open_coscientist["citation_records"] == 1
    assert open_coscientist["native_review_present"] is True


def test_extract_json_ignores_literal_solution_tag_before_native_artifact() -> None:
    expected = {
        "method": "biomni_native",
        "hypotheses": [
            {
                "rank": 1,
                "candidate_id": "func|schaefer200|ADHD|corr_mean|1",
                "rationale": "Registered candidate.",
                "confidence": 0.8,
            }
        ],
    }
    text = f"""
I will return JSON in the final `<solution>` tag.
<execute>
tool_result = {{"status": "ok"}}
</execute>
<solution>
{json.dumps(expected)}
</solution>
"""

    assert extract_json(text) == expected


def test_ai_scientist_checkpoints_native_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, bool, int, bool]] = []
    candidate_ids = [f"func|schaefer200|ADHD|corr_mean|{index}" for index in range(80)]
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(
        "\n".join(
            json.dumps({"candidate_id": candidate_id}) for candidate_id in candidate_ids
        )
        + "\n",
        encoding="utf-8",
    )
    package = ModuleType("ai_scientist")
    package.__path__ = []  # type: ignore[attr-defined]
    llm_module = ModuleType("ai_scientist.llm")
    ideation_module = ModuleType("ai_scientist.perform_ideation_temp_free")

    def create_client(model: str) -> tuple[object, str]:
        return object(), model

    def generate_temp_free_idea(**kwargs: object) -> list[dict[str, str]]:
        path = Path(str(kwargs["idea_fname"]))
        reload_ideas = bool(kwargs["reload_ideas"])
        ideas = (
            json.loads(path.read_text(encoding="utf-8"))
            if reload_ideas and path.exists()
            else []
        )
        calls.append(
            (
                int(kwargs["max_num_generations"]),
                reload_ideas,
                len(ideas),
                "SearchSemanticScholar" in str(kwargs["workshop_description"]),
            )
        )
        for _ in range(int(kwargs["max_num_generations"])):
            start = len(ideas) * 8
            portfolio = "\n".join(
                f"candidate_id: {candidate_id}: registered rationale"
                for candidate_id in candidate_ids[start : start + 8]
            )
            ideas.append({"idea": portfolio})
        path.write_text(json.dumps(ideas), encoding="utf-8")
        return ideas

    llm_module.create_client = create_client  # type: ignore[attr-defined]
    ideation_module.generate_temp_free_idea = (  # type: ignore[attr-defined]
        generate_temp_free_idea
    )
    monkeypatch.setitem(sys.modules, "ai_scientist", package)
    monkeypatch.setitem(sys.modules, "ai_scientist.llm", llm_module)
    monkeypatch.setitem(
        sys.modules,
        "ai_scientist.perform_ideation_temp_free",
        ideation_module,
    )

    result = run_ai_scientist(
        {
            "research_goal": "Blinded registered hypotheses.",
            "n_anchors": 80,
            "trial": 0,
            "public_registry_path": str(registry_path),
        },
        SimpleNamespace(
            repo=tmp_path,
            enable_native_retrieval=True,
            out=tmp_path,
            model="gpt-5.5",
        ),
    )

    assert len(result) == 10
    assert calls == [(1, True, index, True) for index in range(10)]


def test_open_coscientist_generates_distinct_single_candidate_hypotheses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    package = ModuleType("open_coscientist")
    package.__path__ = []  # type: ignore[attr-defined]

    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "candidate_id": "fmri|atlas|disease|feature|0",
                "modality": "fmri",
                "source": "atlas",
                "disease": "disease",
                "feature": "feature",
                "roi_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "src" / "open_coscientist"
    source_root.mkdir(parents=True)
    (source_root / "fake.py").write_text("# fake upstream source\n", encoding="utf-8")

    class FakeNodeCache:
        enabled = False

    class FakeLLMCache:
        enabled = True

        def __init__(self) -> None:
            self.cache_dir = tmp_path / "open_coscientist_trial_cache"

        def _generate_cache_key(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            return "0" * 64

        def get(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            return None

        def set(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    fake_llm_cache = FakeLLMCache()
    cache_module = ModuleType("open_coscientist.cache")
    cache_module.get_node_cache = lambda: FakeNodeCache()  # type: ignore[attr-defined]
    cache_module.get_cache = lambda: fake_llm_cache  # type: ignore[attr-defined]

    class HypothesisGenerator:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        async def generate_hypotheses(self, **kwargs: object) -> dict[str, object]:
            captured["generate"] = kwargs
            return {"hypotheses": []}

    package.HypothesisGenerator = HypothesisGenerator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "open_coscientist", package)
    monkeypatch.setitem(sys.modules, "open_coscientist.cache", cache_module)
    monkeypatch.delenv("OPEN_COSCIENTIST_MAX_CONCURRENT_LLM_CALLS", raising=False)
    monkeypatch.delenv("OPEN_COSCIENTIST_RATE_LIMIT_RETRIES", raising=False)
    monkeypatch.delenv("OPEN_COSCIENTIST_TRANSIENT_RETRIES", raising=False)
    monkeypatch.delenv("COSCIENTIST_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("COSCIENTIST_CACHE_DIR", raising=False)

    result = run_open_coscientist(
        {
            "research_goal": (
                "Blinded registered hypotheses.\n\n"
                "Return the requested framework-native artifact as a portfolio."
            ),
            "n_anchors": 20,
            "open_coscientist_overgeneration_factor": 1.0,
            "open_coscientist_tool_generation_enabled": False,
            "open_coscientist_debate_cohorts_enabled": True,
            "trial": 4,
            "public_registry_path": str(registry_path),
        },
        SimpleNamespace(
            repo=tmp_path,
            model="gpt-5.5",
            enable_native_retrieval=True,
            out=tmp_path,
            reasoning_effort="high",
        ),
    )

    assert result == {"hypotheses": []}
    assert captured["init"] == {
        "model_name": "openai/gpt-5.5",
        "max_iterations": 1,
        "initial_hypotheses_count": 20,
        "evolution_max_count": 20,
        "enable_cache": True,
        "cache_dir": str(tmp_path / "open_coscientist_trial_cache"),
    }
    generate = captured["generate"]
    assert isinstance(generate, dict)
    assert "Each individual Hypothesis object" in str(generate["research_goal"])
    assert "Return the requested framework-native artifact" not in str(
        generate["research_goal"]
    )
    assert generate["run_id"] == "cs1-4-round-0"
    assert generate["opts"]["enable_literature_review_node"] is True
    assert generate["opts"]["enable_tool_calling_generation"] is False
    cache_manifest = json.loads(
        (tmp_path / "open_coscientist_cache_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert cache_manifest["status"] == "completed"
    assert cache_manifest["identity"]["cache_policy"]["node_cache_enabled"] is False
    assert cache_manifest["identity"]["scope"]["trial"] == 4

    with pytest.raises(RuntimeError, match="cache identity mismatch"):
        run_open_coscientist(
            {
                "research_goal": "Blinded registered hypotheses.",
                "n_anchors": 20,
                "trial": 4,
                "public_registry_path": str(registry_path),
            },
            SimpleNamespace(
                repo=tmp_path,
                model="a-different-model",
                enable_native_retrieval=True,
                out=tmp_path,
                reasoning_effort="high",
            ),
        )


def test_open_coscientist_debate_cohorts_are_disjoint_and_cover_menu(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.csv"
    candidate_ids = [f"fmri|atlas|disease|feature|{index}" for index in range(12)]
    pd.DataFrame({"candidate_id": candidate_ids}).to_csv(registry_path, index=False)

    cohorts = _open_coscientist_debate_cohorts(
        {
            "public_registry_path": str(registry_path),
            "open_coscientist_debate_cohorts_enabled": True,
        },
        4,
    )

    assert set(cohorts) == {0, 1, 2, 3}
    assert all(len(cohort) == 3 for cohort in cohorts.values())
    flattened = [candidate_id for cohort in cohorts.values() for candidate_id in cohort]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(candidate_ids)


def test_open_coscientist_evolution_guidance_protects_candidate_id() -> None:
    original = {
        "workflow_plan": {
            "evolution_phase": {"refinement_priorities": ["Improve feasibility"]}
        }
    }

    guarded = _open_coscientist_evolution_guidance(original)
    phase = guarded["workflow_plan"]["evolution_phase"]

    assert original["workflow_plan"]["evolution_phase"]["refinement_priorities"] == [
        "Improve feasibility"
    ]
    assert any("immutable" in item for item in phase["refinement_priorities"])
    assert "Never change" in phase["iteration_strategy"]


def test_official_adapter_can_recompile_existing_native_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "native_result.json"
    artifact.write_text('{"hypotheses": [{"text": "native"}]}', encoding="utf-8")

    result = run_native(
        {},
        SimpleNamespace(
            method="open_coscientist",
            native_artifact=artifact,
        ),
    )

    assert result == {"hypotheses": [{"text": "native"}]}


def test_streamed_kg_index_uses_semantic_edges_only(tmp_path: Path) -> None:
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "metadata": {"version": "test"},
                "concepts": {
                    "D": {
                        "preferred_name": "Schizophrenia",
                        "aliases": ["Psychosis"],
                    },
                    "R": {
                        "preferred_name": "Anterior cingulate cortex",
                        "aliases": ["ACC"],
                    },
                    "C": {"preferred_name": "Claim provenance node"},
                    "X": {"preferred_name": "Unrelated endpoint"},
                },
                "edges": [
                    {
                        "source_id": "D",
                        "target_id": "R",
                        "relation_type": "associated_with",
                        "confidence": 0.8,
                        "metadata": {
                            "claim_case_study_ids": ["case1_transdiagnostic"]
                        },
                    },
                    {
                        "source_id": "C",
                        "target_id": "D",
                        "relation_type": "about",
                        "confidence": 1.0,
                    },
                    {
                        "source_id": "D",
                        "target_id": "X",
                        "relation_type": "contradicts",
                        "confidence": 1.0,
                    },
                    {
                        "source_id": "R",
                        "target_id": "X",
                        "relation_type": "associated_with",
                        "confidence": 0.7,
                    },
                    {
                        "source_id": "R",
                        "target_id": "C",
                        "relation_type": "predicts",
                        "confidence": 0.9,
                        "metadata": {"negated": True},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    terms = {"schizophrenia", "psychosis", "anterior cingulate cortex", "acc"}
    payload = load_case1_kg_index_payload(graph_path, terms, cache=False)

    assert payload["name_to_ids"]["psychosis"] == ("D",)
    assert payload["name_to_ids"]["acc"] == ("R",)
    assert payload["degrees"] == {"D": 1, "R": 2}
    assert payload["adjacency"] == {"D": {"R"}, "R": {"D", "X"}}
    assert payload["directed_support"] == {("D", "R"): 0.8}
    assert payload["case_study_id"] == "case1_transdiagnostic"
    assert payload["scoped_degrees"] == {"D": 1, "R": 1}
    assert payload["name_to_scoped_degree"]["psychosis"] == 1
    assert payload["name_to_scoped_degree"]["acc"] == 1
    assert payload["scoped_adjacency"] == {"D": {"R"}, "R": {"D"}}
    assert payload["scoped_directed_support"] == {("D", "R"): 0.8}
    assert payload["stats"]["case_study_semantic_edges_seen"] == 1
    assert payload["stats"]["about_edges_skipped"] == 1
    assert payload["stats"]["contradicted_edges_skipped"] == 2


def test_streamed_kg_index_cache_is_source_and_query_scoped(tmp_path: Path) -> None:
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "concepts": {"A": {"preferred_name": "Alpha"}},
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    terms = {"alpha"}

    first = load_case1_kg_index_payload(graph_path, terms)
    cache_path = case1_kg_cache_path(graph_path.resolve(), terms)
    cache_mtime = cache_path.stat().st_mtime_ns
    second = load_case1_kg_index_payload(graph_path, terms)

    assert first == second
    assert cache_path.exists()
    assert cache_path.stat().st_mtime_ns == cache_mtime
    assert case1_kg_cache_path(
        graph_path.resolve(), terms, "case1_transdiagnostic"
    ) != case1_kg_cache_path(graph_path.resolve(), terms, "brain_age")


def test_public_registry_does_not_expose_outcomes() -> None:
    registry = build_public_registry(candidate_frame())

    assert "adjusted_residual_d" not in registry
    assert "p_value" not in registry
    assert "is_gt_top" not in registry
    assert registry["candidate_id"].is_unique
    assert registry["anatomy_id"].eq("func|schaefer200|1").sum() == 4


def test_policy_compiles_to_deterministic_full_permutation() -> None:
    candidates = candidate_frame()
    anchor_id = "func|schaefer200|ADHD|corr_mean|1"
    policy = SearchPolicy(
        method="test_method",
        trial=3,
        anchors=(PolicyAnchor(anchor_id, score=0.9),),
        rules=(PolicyRule(weight=0.4, when={"disease": "ADHD"}),),
    )

    first = compile_policy_order(candidates, policy)
    first_ids = candidates.iloc[first]["candidate_id"].tolist()
    shuffled = candidates.sample(frac=1.0, random_state=17).reset_index(drop=True)
    second_ids = shuffled.iloc[compile_policy_order(shuffled, policy)][
        "candidate_id"
    ].tolist()

    assert first_ids == second_ids
    assert first_ids[0] == anchor_id
    assert len(first) == len(candidates)
    assert len(np.unique(first)) == len(candidates)


def test_unanchored_tail_is_method_informed_not_rng_dependent() -> None:
    candidates = candidate_frame()
    policy = SearchPolicy(
        method="native_method",
        trial=0,
        anchors=(PolicyAnchor("func|schaefer200|ADHD|corr_mean|1"),),
    )

    scores = compile_policy_scores(candidates, policy)
    same_disease_feature = candidates["candidate_id"].eq(
        "func|schaefer200|ADHD|corr_mean|2"
    )
    unrelated = candidates["candidate_id"].eq(
        "func|schaefer200|MDD_depression|roi_alff_proxy|2"
    )
    assert scores[same_disease_feature][0] > scores[unrelated][0]


def test_factorized_compiler_matches_naive_anchor_expansion() -> None:
    candidates = candidate_frame()
    policy = SearchPolicy(
        method="factorized",
        trial=0,
        anchors=(
            PolicyAnchor(
                "func|schaefer200|ADHD|corr_mean|1",
                score=0.8,
            ),
            PolicyAnchor(
                "func|schaefer200|MDD_depression|roi_alff_proxy|2",
                score=0.6,
            ),
        ),
        rules=(
            PolicyRule(
                weight=0.2,
                when={"hemisphere": "left"},
            ),
        ),
    )
    registry = build_public_registry(candidates)
    expected = np.zeros(len(registry), dtype=float)
    for rank, anchor in enumerate(policy.anchors):
        anchor_index = registry["candidate_id"].eq(anchor.candidate_id).to_numpy()
        row = registry.loc[anchor_index].iloc[0]
        rank_discount = 1.0 / np.sqrt(rank + 1.0)
        weight = anchor.score * rank_discount
        disease = registry["disease"].eq(row["disease"]).to_numpy()
        feature = registry["feature"].eq(row["feature"]).to_numpy()
        anatomy = registry["anatomy_id"].eq(row["anatomy_id"]).to_numpy()
        hemisphere = registry["hemisphere"].eq(row["hemisphere"]).to_numpy()
        map_group = registry["map_group"].eq(row["map_group"]).to_numpy()
        source = registry["source"].eq(row["source"]).to_numpy()
        expanded = np.maximum.reduce(
            (
                0.02 * weight * disease,
                0.03 * weight * feature,
                0.35 * weight * anatomy,
                0.05 * weight * hemisphere,
                0.08 * weight * map_group,
                0.01 * weight * source,
                0.20 * weight * (disease & feature),
                0.70 * weight * (disease & anatomy),
                0.65 * weight * (feature & anatomy),
                0.50 * weight * (disease & feature & hemisphere),
            )
        )
        expanded[anchor_index] = 3.0 + 0.5 * anchor.score + 0.5 * rank_discount
        expected = np.maximum(expected, expanded)
    expected[registry["hemisphere"].eq("left").to_numpy()] += 0.2

    assert np.allclose(compile_policy_scores(candidates, policy), expected)


def test_zero_score_tail_is_shared_lexical_order_not_method_hash() -> None:
    candidates = candidate_frame()
    first = compile_policy_order(
        candidates,
        SearchPolicy(method="method_a", trial=0),
    )
    second = compile_policy_order(
        candidates,
        SearchPolicy(method="method_b", trial=9),
    )
    expected_ids = sorted(candidates["candidate_id"].tolist())

    assert candidates.iloc[first]["candidate_id"].tolist() == expected_ids
    assert candidates.iloc[second]["candidate_id"].tolist() == expected_ids


def test_policy_rejects_duplicate_anchors_and_out_of_range_weights() -> None:
    candidates = candidate_frame()
    candidate_id = "func|schaefer200|ADHD|corr_mean|1"
    with pytest.raises(ValueError, match="must be unique"):
        compile_policy_order(
            candidates,
            SearchPolicy(
                method="duplicate",
                trial=0,
                anchors=(
                    PolicyAnchor(candidate_id),
                    PolicyAnchor(candidate_id),
                ),
            ),
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        compile_policy_order(
            candidates,
            SearchPolicy(
                method="bad_score",
                trial=0,
                anchors=(PolicyAnchor(candidate_id, score=1.1),),
            ),
        )
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        compile_policy_order(
            candidates,
            SearchPolicy(
                method="bad_rule",
                trial=0,
                rules=(PolicyRule(weight=2.0, when={"disease": "ADHD"}),),
            ),
        )


def test_prefix_quota_diversifies_full_ranking_without_dropping_candidates() -> None:
    candidates = candidate_frame()
    policy = SearchPolicy(
        method="quota",
        trial=0,
        rules=(PolicyRule(weight=0.5, when={"disease": "ADHD"}),),
        quotas={
            "prefix_size": 2,
            "max_per_value": {"disease": 1},
        },
    )
    order = compile_policy_order(candidates, policy)
    ranked = candidates.iloc[order]

    assert ranked.head(2)["disease"].nunique() == 2
    assert len(order) == len(candidates)
    assert len(np.unique(order)) == len(candidates)


def test_unknown_candidate_and_private_rule_field_are_rejected() -> None:
    candidates = candidate_frame()
    with pytest.raises(ValueError, match="unknown candidate_id"):
        compile_policy_order(
            candidates,
            SearchPolicy(method="bad", trial=0, anchors=(PolicyAnchor("missing"),)),
        )

    with pytest.raises(ValueError, match="non-public"):
        policy_from_payload(
            {
                "method": "bad",
                "rules": [{"weight": 1.0, "when": {"is_gt_top": "True"}}],
            }
        )


def test_legacy_mapped_rows_become_exact_anchors_without_failure_slots() -> None:
    mapped = pd.DataFrame(
        [
            {
                "method": "native",
                "seed": 0,
                "generated_rank": 1,
                "mapping_status": "mapped",
                "mapped_candidate_id": "func|schaefer200|ADHD|corr_mean|1",
                "generated_confidence": 0.8,
            },
            {
                "method": "native",
                "seed": 0,
                "generated_rank": 2,
                "mapping_status": "anatomy_unmapped",
                "mapped_candidate_id": np.nan,
            },
        ]
    )
    policy = policy_from_mapped_hypotheses(mapped, method="native", trial=0)
    order = compile_policy_order(candidate_frame(), policy)

    assert len(policy.anchors) == 1
    assert len(order) == len(candidate_frame())
    assert int(order.max()) < len(candidate_frame())


@pytest.mark.parametrize(
    ("adjusted_d", "p_value", "expected", "classification"),
    [
        (-0.4, 0.001, "decrease", "supported"),
        (0.4, 0.001, "decrease", "contradicted"),
        (0.4, 0.001, None, "supported"),
        (0.1, 0.001, None, "inconclusive"),
        (0.4, 0.2, None, "inconclusive"),
    ],
)
def test_observed_feedback_classification(
    adjusted_d: float,
    p_value: float,
    expected: str | None,
    classification: str,
) -> None:
    assert (
        classify_observed_feedback(
            adjusted_d,
            p_value,
            expected_direction=expected,
            alpha=0.01,
            min_abs_d=0.15,
        )
        == classification
    )


def test_neurodiscovery_order_does_not_use_offline_gt_as_feedback() -> None:
    rows = []
    for index in range(24):
        rows.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "score_neurodiscovery": 1.0 - index / 100.0,
                "disease": f"disease-{index % 3}",
                "feature_family": f"feature-{index % 4}",
                "map_group": f"group-{index % 5}",
                "roi_key": f"roi-{index % 6}",
                "source": f"source-{index % 2}",
                "adjusted_residual_d": 0.3 if index % 3 == 0 else 0.05,
                "p_value": 0.001 if index % 3 == 0 else 0.5,
                "is_gt_top": index % 2 == 0,
            }
        )
    scored = pd.DataFrame(rows)
    relabelled = scored.copy()
    relabelled["is_gt_top"] = ~relabelled["is_gt_top"]

    first = closed_loop_neurodiscovery_order(
        scored,
        np.random.default_rng(42),
        batch_size=4,
        warmup_budget=8,
        max_closed_loop_budget=20,
    )
    second = closed_loop_neurodiscovery_order(
        relabelled,
        np.random.default_rng(42),
        batch_size=4,
        warmup_budget=8,
        max_closed_loop_budget=20,
    )

    np.testing.assert_array_equal(first, second)


def test_case1_closed_loop_writes_and_reuses_experimental_claims(tmp_path: Path) -> None:
    rows = []
    for index in range(24):
        rows.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "score_neurodiscovery": 1.0 - index / 100.0,
                "disease": f"disease-{index % 3}",
                "feature": f"feature-value-{index % 4}",
                "feature_family": f"feature-{index % 4}",
                "map_group": f"group-{index % 5}",
                "roi_key": f"roi-{index % 6}",
                "anatomy_full": f"region-{index % 6}",
                "source": f"source-{index % 2}",
                "adjusted_residual_d": 0.3 if index % 3 == 0 else 0.05,
                "p_value": 0.001 if index % 3 == 0 else 0.5,
                "is_gt_top": index % 2 == 0,
            }
        )
    scored = pd.DataFrame(rows)

    order, overlay = closed_loop_neurodiscovery_order(
        scored,
        np.random.default_rng(42),
        batch_size=4,
        warmup_budget=8,
        max_closed_loop_budget=20,
        seed=17,
        trial=2,
        overlay_path=tmp_path / "overlay.jsonl",
        return_overlay_manifest=True,
    )

    assert sorted(order.tolist()) == list(range(len(scored)))
    assert overlay["records"] == 20
    assert overlay["feedback_consumed_during_ranking"] is True
    assert overlay["semantic_projection_verified"] is True
    assert overlay["records_by_status"]["inconclusive"] > 0
    records = [
        json.loads(line)
        for line in (tmp_path / "overlay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["case_study_id"] == "case1_transdiagnostic" for record in records)
    assert all(record["assertions"][0]["predicate"] == "has_imaging_marker" for record in records)


def test_native_adapter_accepts_only_exact_candidate_ids() -> None:
    scored = candidate_frame()
    public = build_public_registry(scored)
    valid_id = "func|schaefer200|ADHD|corr_mean|1"
    payload = {
        "hypotheses": [
            {
                "rank": 1,
                "candidate_id": valid_id,
                "rationale": "Native exact proposal.",
                "confidence": 0.8,
            },
            {
                "rank": 2,
                "candidate_id": "ADHD anterior cingulate corr",
                "rationale": "A paraphrase must not be fuzzy-mapped.",
                "confidence": 0.8,
            },
        ]
    }
    validated = validate_seed_outputs(
        "biomni_native",
        0,
        [(1, 2, payload, None)],
        public,
    )
    mapped = map_validated(validated, scored)

    assert mapped.loc[0, "mapping_status"] == "mapped"
    assert mapped.loc[0, "mapped_candidate_id"] == valid_id
    assert "invalid_candidate_id" in mapped.loc[1, "native_validation_status"]
    assert mapped.loc[1, "mapping_status"] == "native_validation_failed"


def test_official_native_compiler_never_fuzzy_maps_prose(tmp_path: Path) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(
        registry_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    first_id = "func|schaefer200|ADHD|corr_mean|1"
    second_id = "func|schaefer200|MDD_depression|roi_alff_proxy|2"
    payload, audit = compile_native_policy(
        method="official",
        task={
            "trial": 4,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
        },
        native_result={
            "ideas": [
                {
                    "candidate_id": first_id,
                    "confidence": 0.7,
                    "rationale": "Exact native proposal.",
                },
                {
                    "modality": "func",
                    "source": "schaefer200",
                    "disease": "MDD_depression",
                    "feature": "roi_alff_proxy",
                    "roi_index": "2",
                },
                "ADHD anterior cingulate corr should be prioritized.",
                "func|schaefer200|ADHD|unknown_feature|1",
            ]
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [
        first_id,
        second_id,
    ]
    assert audit["mapping_mode"] == "deterministic_exact_native_only"
    assert audit["invalid_exact_mentions"] == [
        "func|schaefer200|ADHD|unknown_feature|1"
    ]


def test_virtual_lab_compiler_accepts_markdown_escaped_exact_ids(
    tmp_path: Path,
) -> None:
    candidate_id = "fmri|atlas|ADHD|corr_mean_abs|1"
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({"candidate_id": candidate_id}) + "\n",
        encoding="utf-8",
    )
    task = {
        "public_registry_path": str(registry),
        "n_anchors": 1,
        "trial": 5,
    }

    policy, audit = compile_native_policy(
        method="virtual_lab",
        task=task,
        native_result={
            "summary": r"| 1 | fmri\|atlas\|ADHD\|corr_mean_abs\|1 |"
        },
    )

    assert [anchor["candidate_id"] for anchor in policy["anchors"]] == [
        candidate_id
    ]
    assert audit["mapping_mode"] == "deterministic_exact_native_only"


def test_sciagents_compiler_excludes_prompt_path_candidates(tmp_path: Path) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    prompt_id = "func|schaefer200|ADHD|corr_mean|1"
    final_id = "func|schaefer200|MDD_depression|roi_alff_proxy|2"

    payload, audit = compile_native_policy(
        method="sciagents",
        task={
            "trial": 0,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
        },
        native_result={
            "final_artifact": f"1. candidate_id: {final_id}\nTERMINATE",
            "chat_history": [
                {"role": "user", "content": f"Sampled input path: {prompt_id}"}
            ],
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [final_id]
    assert audit["artifact_scope"] == "sciagents_final_artifact"


def test_open_coscientist_compiler_uses_only_final_ranked_hypothesis_texts(
    tmp_path: Path,
) -> None:
    registry = build_public_registry(candidate_frame())
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    ranked_id = "func|schaefer200|ADHD|corr_mean|1"
    generated_id = "func|schaefer200|MDD_depression|roi_alff_proxy|2"
    planning_id = "func|schaefer200|ADHD|roi_alff_proxy|1"

    payload, audit = compile_native_policy(
        method="open_coscientist",
        task={
            "trial": 0,
            "n_anchors": 5,
            "public_registry_path": str(registry_path),
        },
        native_result={
            "hypotheses": [
                {
                    "text": (
                        "candidate_id: modality|source|disease|feature|roi_index\n"
                        f"candidate_id: {ranked_id}"
                    )
                }
            ],
            "debate_transcripts": [
                {
                    "hypothesis_text": "A generated method summary without an ID.",
                    "transcript": f"candidate_id: {generated_id}",
                }
            ],
            "research_plan": f"Consider candidate_id: {planning_id}",
            "reviews": [{"text": f"Prefer candidate_id: {planning_id}"}],
        },
    )

    assert [row["candidate_id"] for row in payload["anchors"]] == [ranked_id]
    assert audit["exact_mentions_seen"] == 1
    assert audit["invalid_exact_mentions"] == []
    assert audit["artifact_scope"] == "open_coscientist_final_ranked_hypotheses_only"


def test_brainpilot_client_recovers_validated_native_artifact_from_events(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the BrainPilot client regression test")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("formal prompt", encoding="utf-8")
    candidate_payload = {
        "method": "brainpilot_native",
        "hypotheses": [
            {
                "rank": 21,
                "candidate_id": "func|schaefer200|ADHD|corr_mean|1",
                "rationale": "Exact native candidate.",
                "confidence": 0.8,
            },
            {
                "rank": 22,
                "candidate_id": ("func|schaefer200|MDD_depression|roi_alff_proxy|2"),
                "rationale": "Second exact native candidate.",
                "confidence": 0.7,
            },
        ],
    }
    tool_args = json.dumps(
        {
            "to": "auditor",
            "msg_type": "task_delegate",
            "content": (
                "Audit this native draft:\n"
                + json.dumps(candidate_payload, ensure_ascii=False)
            ),
        }
    )
    (tmp_path / "events.json").write_text(
        json.dumps(
            [
                {
                    "type": "TOOL_CALL_ARGS",
                    "tool_call_id": "draft-review",
                    "delta": tool_args,
                },
                {
                    "type": "CUSTOM",
                    "name": "session_state",
                    "value": {"runState": {"active": False}},
                },
            ]
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[3]
        / "core"
        / "scripts"
        / "brainpilot_cs1_batch_client.mjs"
    )

    result = subprocess.run(
        [
            node,
            str(script),
            "--prompt",
            str(prompt_path),
            "--out",
            str(tmp_path),
            "--client-dist",
            str(tmp_path / "unused-client.js"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    recovered = json.loads((tmp_path / "final.txt").read_text(encoding="utf-8"))
    assert recovered == candidate_payload
    metadata = json.loads((tmp_path / "client_meta.json").read_text(encoding="utf-8"))
    assert metadata["end_reason"] == (
        "validated_native_candidate_artifact_from_existing_events"
    )


def test_policy_audit_requires_all_monitored_prefixes_to_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = candidate_frame()
    candidate_ids = candidates["candidate_id"].tolist()
    policies = (
        SearchPolicy(
            method="method_a",
            trial=0,
            anchors=(PolicyAnchor(candidate_ids[0]),),
        ),
        SearchPolicy(
            method="method_b",
            trial=0,
            anchors=(PolicyAnchor(candidate_ids[1]),),
        ),
    )
    orders = {
        "method_a": np.arange(len(candidates)),
        "method_b": np.array([0, 1, 4, 5, 2, 3, 6, 7]),
    }

    monkeypatch.setattr(
        "core.scripts.case1_policy_audit.compile_policy_order",
        lambda _registry, policy: orders[policy.method],
    )
    frame, summary = audit_policy_independence(
        candidates,
        policies,
        top_ks=(2, 4),
    )

    assert frame.loc[0, "top_2_jaccard"] == 1.0
    assert frame.loc[0, "top_4_jaccard"] < 0.98
    assert summary["passed"]


def test_policy_independence_audit_detects_only_implementation_collapse() -> None:
    candidates = candidate_frame()
    first_id = "func|schaefer200|ADHD|corr_mean|1"
    second_id = "func|schaefer200|MDD_depression|roi_alff_proxy|2"
    distinct = (
        SearchPolicy(
            method="method_a",
            trial=0,
            anchors=(PolicyAnchor(first_id),),
        ),
        SearchPolicy(
            method="method_b",
            trial=0,
            anchors=(PolicyAnchor(second_id),),
        ),
    )
    _, distinct_summary = audit_policy_independence(
        candidates,
        distinct,
        top_ks=(2,),
    )
    assert distinct_summary["passed"]

    collapsed = (
        distinct[0],
        SearchPolicy(
            method="method_b",
            trial=0,
            anchors=(PolicyAnchor(first_id),),
        ),
    )
    _, collapsed_summary = audit_policy_independence(
        candidates,
        collapsed,
        top_ks=(2,),
    )
    assert not collapsed_summary["passed"]
    assert collapsed_summary["collapsed_pairs"][0]["reason"] == (
        "identical native anchor sequence"
    )
