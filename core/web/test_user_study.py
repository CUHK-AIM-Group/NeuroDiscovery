from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path

_SERVICE_PATH = Path(__file__).resolve().parents[2] / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurodiscovery_user_study_test", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
UserStudyService = _MODULE.UserStudyService
candidate_structural_distance = _MODULE.candidate_structural_distance
candidate_comparison_task = _MODULE.candidate_comparison_task
build_progressive_pair_schedule = _MODULE.build_progressive_pair_schedule


class UserStudyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidates_path = self.root / "candidates.json"
        self.candidates_path.write_text(
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "h1",
                            "source_name": "Disorder A",
                            "target_name": "Region A",
                            "explanation": "First candidate",
                            "composite_score": 0.9,
                            "evidence_score": 0.8,
                            "path": [],
                            "literature": [
                                {
                                    "title": "A relevant paper",
                                    "pmid": "123456",
                                    "excerpts": ["A source-grounded relevant sentence."],
                                    "abstract": "A complete source abstract.",
                                }
                            ],
                        },
                        {
                            "id": "h2",
                            "source_name": "Disorder B",
                            "target_name": "Region B",
                            "explanation": "Second candidate",
                            "composite_score": 0.4,
                            "evidence_score": 0.5,
                            "path": [],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.service = UserStudyService(self.root / "study")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_database_recovers_if_study_directory_is_removed(self) -> None:
        import shutil

        shutil.rmtree(self.service.root)

        self.assertEqual(self.service.list_sessions("study-after-reset"), [])
        self.assertTrue(self.service.db_path.is_file())

    def test_manual_condition_hides_generator_scores_and_records_events(self) -> None:
        session = self.service.create_session(
            study_id="case1",
            participant_id="expert-1",
            condition="manual",
            candidate_path=self.candidates_path,
            random_seed=42,
        )
        self.assertEqual(session["status"], "active")
        self.assertEqual({item["id"] for item in session["candidates"]}, {"h1", "h2"})
        self.assertTrue(all("composite_score" not in item for item in session["candidates"]))
        h1 = next(item for item in session["candidates"] if item["id"] == "h1")
        self.assertEqual(h1["literature"][0]["title"], "A relevant paper")
        self.assertEqual(h1["literature"][0]["excerpts"], ["A source-grounded relevant sentence."])
        self.assertEqual(h1["literature"][0]["abstract"], "A complete source abstract.")
        self.assertEqual(
            self.service.append_events(
                session["session_id"],
                [{"type": "tier_assigned", "hypothesis_id": "h1", "elapsed_ms": 500, "payload": {"tier": "high"}}],
            ),
            1,
        )
        submitted = self.service.submit_session(
            session["session_id"],
            ranking=["h1", "h2"],
            active_seconds=12.5,
            wall_seconds=20,
            buckets={"h1": "high", "h2": "low"},
            completion_reason="time_limit",
        )
        self.assertEqual(submitted["status"], "completed")
        self.assertEqual(submitted["final_ranking"], ["h1", "h2"])
        self.assertEqual(submitted["completion_reason"], "time_limit")

    def test_unfinished_session_can_be_deleted_and_restarted_at_same_number(self) -> None:
        session = self.service.create_session(
            study_id="restart-study",
            participant_id="expert-restart",
            condition="manual",
            candidate_path=self.candidates_path,
        )
        self.service.append_events(
            session["session_id"],
            [
                {
                    "type": "pairwise_choice",
                    "elapsed_ms": 2500,
                    "payload": {"pair_id": "pair-001", "choice": "left"},
                }
            ],
        )

        deleted = self.service.delete_active_session(session["session_id"])

        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["session_number"], 1)
        with self.assertRaises(KeyError):
            self.service.get_session(session["session_id"], include_events=True)

        restarted = self.service.create_session(
            study_id="restart-study",
            participant_id="expert-restart",
            condition="manual",
            candidate_path=self.candidates_path,
        )
        self.assertEqual(restarted["session_number"], 1)
        self.assertNotEqual(restarted["session_id"], session["session_id"])

    def test_completed_session_cannot_be_deleted(self) -> None:
        session = self.service.create_session(
            study_id="immutable-study",
            participant_id="expert-complete",
            condition="manual",
            candidate_path=self.candidates_path,
        )
        self.service.submit_session(
            session["session_id"],
            ranking=["h1", "h2"],
            active_seconds=10,
            wall_seconds=12,
        )

        with self.assertRaisesRegex(ValueError, "unfinished active session"):
            self.service.delete_active_session(session["session_id"])
        self.assertEqual(self.service.get_session(session["session_id"])["status"], "completed")

    def test_score_visibility_conditions_share_order_and_runtime_discovery_curve(self) -> None:
        manual = self.service.create_session(
            study_id="case1",
            participant_id="expert-1",
            condition="manual",
            candidate_path=self.candidates_path,
            random_seed=42,
        )
        manual_order = [item["id"] for item in manual["candidates"]]
        self.assertTrue(all("composite_score" not in item for item in manual["candidates"]))
        self.service.submit_session(
            manual["session_id"], ranking=["h2", "h1"], active_seconds=100, wall_seconds=110
        )
        assisted = self.service.create_session(
            study_id="case1",
            participant_id="expert-2",
            condition="assisted",
            candidate_path=self.candidates_path,
            random_seed=42,
        )
        self.assertEqual([item["id"] for item in assisted["candidates"]], manual_order)
        self.assertEqual(assisted["pairs"], manual["pairs"])
        self.assertTrue(all("composite_score" in item for item in assisted["candidates"]))
        self.service.submit_session(
            assisted["session_id"], ranking=["h1", "h2"], active_seconds=60, wall_seconds=70
        )
        results_path = self.root / "runtime.json"
        results_path.write_text(
            json.dumps(
                {
                    "execution_results": [
                        {"hypothesis_id": "h1", "status": "confirmed", "duration_seconds": 20},
                        {"hypothesis_id": "h2", "status": "not_confirmed", "duration_seconds": 10},
                    ]
                }
            ),
            encoding="utf-8",
        )
        imported = self.service.import_execution_results("case1", results_path)
        self.assertEqual(imported["imported"], 2)
        results = self.service.results("case1")
        self.assertAlmostEqual(results["ranking_time_saving_percent"], 40.0)
        self.assertEqual(results["curves"]["manual"][0]["experiment_seconds"], 30.0)
        self.assertEqual(results["curves"]["assisted"][0]["experiment_seconds"], 20.0)

    def test_pair_schedule_progresses_from_easy_to_hard(self) -> None:
        def candidate(identifier: str, diseases: list[str], region: str, feature: str) -> dict:
            return {
                "id": identifier,
                "comparison_task_id": "shared-task",
                "metadata": {
                    "candidate_tuple": {
                        "disease_ids": diseases,
                        "region_id": region,
                        "feature_id": feature,
                    }
                },
            }

        candidates = [
            candidate("a", ["d1", "d2", "d3"], "r1", "f1"),
            candidate("b", ["d1", "d2", "d4"], "r1", "f1"),
            candidate("c", ["d5", "d6", "d7"], "r2", "f2"),
            candidate("d", ["d8", "d9", "d10"], "r3", "f3"),
        ]
        self.assertEqual(candidate_structural_distance(candidates[0], candidates[1]), 1)
        self.assertGreaterEqual(candidate_structural_distance(candidates[0], candidates[2]), 4)
        pairs = build_progressive_pair_schedule(candidates, max_pairs=6, seed=7)
        stages = [{"easy": 0, "medium": 1, "hard": 2}[pair["difficulty"]] for pair in pairs]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual({pair["pair_id"] for pair in pairs}, {f"pair-{i:03d}" for i in range(1, 7)})

    def test_one_hundred_question_bank_has_thirty_forty_thirty_split(self) -> None:
        candidates = []
        for region_index in range(10):
            for feature_index in range(5):
                for direction in ("increase", "decrease"):
                    candidates.append(
                        {
                            "id": f"h-{region_index}-{feature_index}-{direction}",
                            "comparison_task_id": "shared-task",
                            "metadata": {
                                "candidate_tuple": {
                                    "disease_ids": ["d1"],
                                    "region_id": f"region-{region_index}",
                                    "feature_id": f"feature-{feature_index}",
                                    "direction": direction,
                                }
                            },
                        }
                    )

        pairs = build_progressive_pair_schedule(candidates, max_pairs=100, seed=23)
        self.assertEqual(len(pairs), 100)
        self.assertEqual(
            Counter(pair["difficulty"] for pair in pairs),
            Counter({"easy": 30, "medium": 40, "hard": 30}),
        )
        self.assertTrue(all(pair["distance"] == 1 for pair in pairs[:30]))
        self.assertTrue(all(pair["distance"] == 2 for pair in pairs[30:70]))
        self.assertTrue(all(pair["distance"] >= 3 for pair in pairs[70:]))

    def test_pair_schedule_never_crosses_comparison_tasks(self) -> None:
        def candidate(identifier: str, task: str, region: str) -> dict:
            return {
                "id": identifier,
                "comparison_task_id": task,
                "metadata": {
                    "candidate_tuple": {
                        "disease_ids": ["d1", "d2", "d3"],
                        "region_id": region,
                        "feature_id": "f1",
                    }
                },
            }

        candidates = [
            candidate("a1", "task-a", "r1"),
            candidate("a2", "task-a", "r2"),
            candidate("b1", "task-b", "r1"),
            candidate("b2", "task-b", "r3"),
        ]
        tasks = {item["id"]: candidate_comparison_task(item) for item in candidates}
        pairs = build_progressive_pair_schedule(candidates, max_pairs=20, seed=9)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(tasks[pair["left_id"]] == tasks[pair["right_id"]] for pair in pairs))

    def test_manually_reviewed_schedule_is_preserved_across_random_seeds(self) -> None:
        curated_path = self.root / "curated.json"
        curated_path.write_text(
            json.dumps(
                {
                    "curation": {"status": "manually_reviewed"},
                    "hypotheses": [
                        {
                            "id": "curated-a",
                            "comparison_task_id": "task-a",
                            "source_name": "Disorder A",
                            "target_name": "Region A",
                        },
                        {
                            "id": "curated-b",
                            "comparison_task_id": "task-a",
                            "source_name": "Disorder A",
                            "target_name": "Region B",
                        },
                    ],
                    "pair_schedule": [
                        {
                            "pair_id": "pair-001",
                            "left_id": "curated-b",
                            "right_id": "curated-a",
                            "difficulty": "hard",
                            "distance": 1,
                            "manual_review": {
                                "status": "reviewed",
                                "quality": "keep",
                                "difficulty": "hard",
                                "reason": "Both sides have balanced, direct evidence.",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        first = self.service.create_session(
            study_id="curated",
            participant_id="expert-a",
            condition="manual",
            candidate_path=curated_path,
            random_seed=0,
        )
        second = self.service.create_session(
            study_id="curated",
            participant_id="expert-b",
            condition="manual",
            candidate_path=curated_path,
            random_seed=999,
        )
        self.assertEqual(first["pairs"], second["pairs"])
        self.assertEqual(first["pairs"][0]["left_id"], "curated-b")
        self.assertEqual(first["pairs"][0]["difficulty"], "hard")
        self.assertEqual(first["pairs"][0]["manual_review"]["quality"], "keep")

    def test_shipped_tcp_external_bank_passes_manual_quality_review(self) -> None:
        bank_path = (
            Path(__file__).resolve().parents[2]
            / "neurooracle"
            / "data"
            / "user_study"
            / "case1_tcp_external_expert_study_v1.json"
        )
        candidates, _, _, pairs = self.service.load_candidates(bank_path)
        payload = json.loads(bank_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "ready_for_expert_study")
        self.assertEqual(payload["curation"]["status"], "external_validation_curated")
        self.assertEqual(payload["n_hypotheses"], 182)
        self.assertEqual(len(payload["hypotheses"]), 182)
        self.assertEqual(len(candidates), 182)
        self.assertEqual(len(pairs), 120)
        self.assertEqual(
            Counter(pair["difficulty"] for pair in pairs),
            Counter({"easy": 40, "medium": 40, "hard": 40}),
        )
        self.assertEqual(
            Counter(int(pair["session_number"]) for pair in pairs),
            Counter({1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}),
        )
        used_ids = [
            candidate_id
            for pair in pairs
            for candidate_id in (pair["left_id"], pair["right_id"])
        ]
        reuse_counts = Counter(used_ids)
        self.assertEqual(len(reuse_counts), 182)
        self.assertLessEqual(max(reuse_counts.values()), 2)
        self.assertTrue(
            all(pair["manual_review"]["quality"] == "keep" for pair in pairs)
        )
        self.assertTrue(
            all(
                pair["manual_review"]["evidence_grade"] == "mixed_curated"
                for pair in pairs
            )
        )
        by_id = {candidate["id"]: candidate for candidate in candidates}
        gaps = {
            level: [
                abs(
                    float(by_id[pair["left_id"]]["composite_score"])
                    - float(by_id[pair["right_id"]]["composite_score"])
                )
                for pair in pairs
                if pair["difficulty"] == level
            ]
            for level in ("easy", "medium", "hard")
        }
        self.assertGreaterEqual(min(gaps["hard"]), 0.03)
        self.assertLess(max(gaps["hard"]), min(gaps["medium"]))
        self.assertLess(max(gaps["medium"]), min(gaps["easy"]))
        protocol = self.service.load_study_protocol(bank_path)
        self.assertEqual(protocol["completion_basis"], "active_time")
        self.assertEqual(protocol["required_sessions"], 6)
        self.assertEqual(protocol["active_seconds_per_session"], 600)
        self.assertEqual(protocol["pair_pool_per_session"], 20)
        self.assertEqual(protocol["assignment_policy"], "fixed_shared_schedule")
        self.assertEqual(protocol["shared_random_seed"], 0)
        self.assertTrue(protocol["same_questions_for_all_participants"])

    def test_shipped_bank_has_curated_literature_evidence_and_credibility(self) -> None:
        bank_path = (
            Path(__file__).resolve().parents[2]
            / "neurooracle"
            / "data"
            / "user_study"
            / "case1_tcp_external_expert_study_v1.json"
        )
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        curation = payload["literature_evidence_curation"]
        self.assertEqual(curation["associations"], 910)
        self.assertEqual(curation["evidence_sentence_min"], 1)
        self.assertEqual(curation["evidence_sentence_max"], 4)
        self.assertIn("journal-level", curation["metric_note_en"])
        scoring = payload["literature_support_scoring"]
        self.assertEqual(scoring["associations_scored"], 910)
        self.assertEqual(scoring["scale_min"], 1)
        self.assertEqual(scoring["scale_max"], 10)
        self.assertEqual(sum(scoring["score_distribution"].values()), 910)
        self.assertIn("not replication probabilities", scoring["disclaimer_en"])

        papers = [
            (hypothesis, paper)
            for hypothesis in payload["hypotheses"]
            for paper in hypothesis["literature"]
        ]
        self.assertEqual(len(papers), 910)
        for hypothesis, paper in papers:
            evidence = paper["evidence_sentences"]
            self.assertGreaterEqual(len(evidence), 1)
            self.assertLessEqual(len(evidence), 4)
            self.assertEqual(paper["excerpts"], evidence)
            abstract = " ".join(str(paper["abstract"]).split()).casefold()
            for sentence in evidence:
                self.assertIn(" ".join(sentence.split()).casefold(), abstract)
            self.assertNotEqual(" ".join(paper["excerpt"].split()).casefold(), abstract)
            assessment = paper["relevance_assessment"]
            self.assertTrue(all(assessment.values()), hypothesis["id"])
            credibility = paper["credibility"]
            self.assertIsNotNone(credibility["publication_year"])
            self.assertTrue(credibility["journal"])
            self.assertIn("status", credibility["journal_impact_factor"])
            self.assertEqual(credibility["cohort"]["source"], "manual_abstract_review")
            self.assertIn(
                credibility["p_values"]["status"],
                {
                    "reported_for_selected_evidence",
                    "reported_elsewhere_in_abstract",
                    "not_reported_in_abstract",
                },
            )
            support_score = paper["support_score"]
            self.assertEqual(
                support_score["schema_version"],
                "hypothesis-paper-support-score-v1",
            )
            self.assertIsInstance(support_score["value"], int)
            self.assertGreaterEqual(support_score["value"], 1)
            self.assertLessEqual(support_score["value"], 10)
            self.assertGreaterEqual(support_score["evidence_fit"]["score"], 1)
            self.assertLessEqual(support_score["evidence_fit"]["score"], 8)
            self.assertGreaterEqual(support_score["study_credibility"]["score"], 0)
            self.assertLessEqual(support_score["study_credibility"]["score"], 2)
            self.assertTrue(support_score["explanation_en"])
            self.assertTrue(support_score["explanation_zh"])
            self.assertFalse(support_score["is_replication_probability"])
            if support_score["polarity"] == "counterevidence":
                self.assertLessEqual(support_score["value"], 2)
            if support_score["polarity"] == "duplicate":
                self.assertEqual(support_score["value"], 1)
            if support_score["value"] >= 9:
                self.assertIn(
                    "strong direct",
                    support_score["evidence_fit"]["basis_en"].casefold(),
                )

        target = next(
            hypothesis
            for hypothesis in payload["hypotheses"]
            if hypothesis["id"]
            == "fmri|cc400_multiatlas|psychosis_SZ_SZA|roi_alff_proxy|108"
        )
        expert_paper = next(
            paper for paper in target["literature"] if paper.get("pmid") == "42044686"
        )
        self.assertEqual(expert_paper["evidence_selection"]["status"], "expert_gold")
        self.assertEqual(len(expert_paper["evidence_sentences"]), 4)
        self.assertEqual(expert_paper["credibility"]["cohort"]["n_total"], 109)
        self.assertEqual(
            expert_paper["credibility"]["p_values"]["status"],
            "reported_for_selected_evidence",
        )

    def test_timeboxed_bank_assigns_six_disjoint_sessions(self) -> None:
        bank_path = (
            Path(__file__).resolve().parents[2]
            / "neurooracle"
            / "data"
            / "user_study"
            / "case1_tcp_external_expert_study_v1.json"
        )
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        pairs = payload["pair_schedule"]
        expected = {
            number: {
                pair["pair_id"]
                for pair in pairs
                if int(pair["session_number"]) == number
            }
            for number in range(1, 7)
        }
        sessions = [
            self.service.create_session(
                study_id="six-session-protocol",
                participant_id="expert-six",
                condition="manual",
                candidate_path=bank_path,
            )
            for _ in range(6)
        ]
        self.assertEqual(
            [session["session_number"] for session in sessions],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertTrue(
            all(session["required_sessions"] == 6 for session in sessions)
        )
        self.assertTrue(
            all(session["target_active_seconds"] == 600 for session in sessions)
        )
        self.assertEqual(
            [
                {pair["pair_id"] for pair in session["pairs"]}
                for session in sessions
            ],
            [expected[number] for number in range(1, 7)],
        )
        with self.assertRaisesRegex(ValueError, "All 6 required sessions"):
            self.service.create_session(
                study_id="six-session-protocol",
                participant_id="expert-six",
                condition="assisted",
                candidate_path=bank_path,
            )

    def test_timeboxed_bank_is_identical_for_different_experts(self) -> None:
        bank_path = (
            Path(__file__).resolve().parents[2]
            / "neurooracle"
            / "data"
            / "user_study"
            / "case1_tcp_external_expert_study_v1.json"
        )
        first = self.service.create_session(
            study_id="shared-schedule",
            participant_id="expert-a",
            condition="manual",
            candidate_path=bank_path,
            random_seed=0,
        )
        second = self.service.create_session(
            study_id="shared-schedule",
            participant_id="expert-b",
            condition="manual",
            candidate_path=bank_path,
            random_seed=999,
        )
        self.assertEqual(first["session_number"], 1)
        self.assertEqual(second["session_number"], 1)
        self.assertEqual(first["pairs"], second["pairs"])
        self.assertEqual(first["random_seed"], 0)
        self.assertEqual(second["random_seed"], 0)

    def test_pair_schedule_excludes_identical_semantic_hypotheses(self) -> None:
        def candidate(identifier: str, region: str, method: str) -> dict:
            return {
                "id": identifier,
                "comparison_task_id": "bipolar-imaging",
                "metadata": {
                    "generation_method": method,
                    "candidate_tuple": {
                        "disease_ids": ["CUI:C0005586"],
                        "region_id": region,
                        "feature_id": "roi_participation_coefficient",
                        "direction": "none; infer sign during validation",
                    },
                },
            }

        candidates = [
            candidate("neurodiscovery-copy", "NN:NN_TAL:10041", "neurodiscovery"),
            candidate("brainstorm-copy", "NN:NN_TAL:10041", "llm_brainstorm"),
            candidate("distinct", "NN:NN_TAL:10042", "neurodiscovery"),
        ]
        pairs = build_progressive_pair_schedule(candidates, max_pairs=10, seed=13)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["distance"] > 0 for pair in pairs))
        self.assertNotIn(
            {"neurodiscovery-copy", "brainstorm-copy"},
            [{pair["left_id"], pair["right_id"]} for pair in pairs],
        )

    def test_pair_schedule_requires_discriminative_literature(self) -> None:
        def candidate(identifier: str, region: str, pmids: list[str]) -> dict:
            return {
                "id": identifier,
                "comparison_task_id": "bipolar-imaging",
                "literature": [{"pmid": pmid, "title": f"Paper {pmid}"} for pmid in pmids],
                "metadata": {
                    "candidate_tuple": {
                        "disease_ids": ["CUI:C0005586"],
                        "region_id": region,
                        "feature_id": "roi_local_efficiency",
                    }
                },
            }

        candidates = [
            candidate("a", "insula", ["1", "2", "3", "4", "5"]),
            candidate("same-evidence", "amygdala", ["1", "2", "3", "4", "5"]),
            candidate("distinct-evidence", "hippocampus", ["1", "2", "6", "7", "8"]),
        ]
        pairs = build_progressive_pair_schedule(candidates, max_pairs=10, seed=17)
        pair_sets = [{pair["left_id"], pair["right_id"]} for pair in pairs]
        self.assertNotIn({"a", "same-evidence"}, pair_sets)
        self.assertIn({"a", "distinct-evidence"}, pair_sets)
        discriminative = next(
            pair for pair in pairs if {pair["left_id"], pair["right_id"]} == {"a", "distinct-evidence"}
        )
        self.assertEqual(discriminative["shared_references"], 2)
        self.assertLess(discriminative["reference_jaccard"], 0.34)

    def test_manual_session_exposes_reproducible_pair_schedule(self) -> None:
        session = self.service.create_session(
            study_id="case1-pairs",
            participant_id="expert-pairs",
            condition="manual",
            candidate_path=self.candidates_path,
            random_seed=11,
        )
        self.assertEqual(len(session["pairs"]), 1)
        self.assertEqual(
            {session["pairs"][0]["left_id"], session["pairs"][0]["right_id"]},
            {"h1", "h2"},
        )

    def test_session_defaults_to_zero_random_seed_and_current_protocol(self) -> None:
        session = self.service.create_session(
            study_id="case1-default-seed",
            participant_id="expert-default",
            condition="manual",
            candidate_path=self.candidates_path,
        )
        self.assertEqual(session["random_seed"], 0)
        self.assertEqual(
            session["protocol_version"], "case1-pairwise-v8-timeboxed-6x10m"
        )

    def test_manual_session_defaults_to_one_hundred_questions(self) -> None:
        candidates = []
        for index in range(25):
            candidates.append(
                {
                    "id": f"h-{index:02d}",
                    "comparison_task_id": "shared-task",
                    "literature": [{"pmid": f"paper-{index:02d}"}],
                    "metadata": {
                        "candidate_tuple": {
                            "disease_ids": ["d1"],
                            "region_id": f"region-{index:02d}",
                            "feature_id": "feature-1",
                        }
                    },
                }
            )
        source = self.root / "large-candidates.json"
        source.write_text(json.dumps({"hypotheses": candidates}), encoding="utf-8")
        session = self.service.create_session(
            study_id="case1-100",
            participant_id="expert-100",
            condition="manual",
            candidate_path=source,
            random_seed=19,
        )
        self.assertEqual(len(session["pairs"]), 100)
        self.assertEqual(len({pair["pair_id"] for pair in session["pairs"]}), 100)


if __name__ == "__main__":
    unittest.main()
