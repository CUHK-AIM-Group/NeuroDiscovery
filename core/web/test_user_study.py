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
        )
        self.assertEqual(submitted["status"], "completed")
        self.assertEqual(submitted["final_ranking"], ["h1", "h2"])

    def test_generator_order_and_runtime_discovery_curve(self) -> None:
        manual = self.service.create_session(
            study_id="case1",
            participant_id="expert-1",
            condition="manual",
            candidate_path=self.candidates_path,
        )
        self.service.submit_session(
            manual["session_id"], ranking=["h2", "h1"], active_seconds=100, wall_seconds=110
        )
        assisted = self.service.create_session(
            study_id="case1",
            participant_id="expert-2",
            condition="assisted",
            candidate_path=self.candidates_path,
        )
        self.assertEqual([item["id"] for item in assisted["candidates"]], ["h1", "h2"])
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

    def test_three_hundred_question_bank_has_one_hundred_per_difficulty(self) -> None:
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

        pairs = build_progressive_pair_schedule(candidates, max_pairs=300, seed=23)
        self.assertEqual(len(pairs), 300)
        self.assertEqual(
            Counter(pair["difficulty"] for pair in pairs),
            Counter({"easy": 100, "medium": 100, "hard": 100}),
        )
        self.assertTrue(all(pair["distance"] == 1 for pair in pairs[:100]))
        self.assertTrue(all(pair["distance"] == 2 for pair in pairs[100:200]))
        self.assertTrue(all(pair["distance"] >= 3 for pair in pairs[200:]))

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
        self.assertEqual(session["protocol_version"], "case1-pairwise-v5")

    def test_manual_session_supports_three_hundred_questions(self) -> None:
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
            study_id="case1-300",
            participant_id="expert-300",
            condition="manual",
            candidate_path=source,
            random_seed=19,
        )
        self.assertEqual(len(session["pairs"]), 300)
        self.assertEqual(len({pair["pair_id"] for pair in session["pairs"]}), 300)


if __name__ == "__main__":
    unittest.main()
