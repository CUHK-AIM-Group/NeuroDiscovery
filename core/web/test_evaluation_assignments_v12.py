"""Assignment coverage/order regression tests with isolated synthetic sessions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from core.web.build_evaluation_assignments_v12 import (
    build_he1, build_he2, HE1_TARGET, HE2_TARGET, HE1_REVISION, HE2_POLICY, MATERIALS, RANKING,
)
from core.web.discovery_study import DiscoveryStudy
from core.web.evaluation_export import ranking_export

spec = importlib.util.spec_from_file_location("ranking_assignments_v12_test", RANKING.parents[1] / "src/user_study.py")
ranking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranking)


class AssignmentTableTests(unittest.TestCase):
    def test_builds_are_reproducible_and_equal_to_the_release(self):
        self.assertEqual(build_he1(), build_he1())
        self.assertEqual(build_he2(), build_he2())
        self.assertEqual(build_he1(), json.loads(HE1_TARGET.read_bytes()))
        self.assertEqual(build_he2(), json.loads(HE2_TARGET.read_bytes()))

    def test_he1_thirty_percent_and_equal_other_topic_totals(self):
        table = build_he1()
        pack = json.loads((MATERIALS / "cs1_discovery_pilot_v10.json").read_bytes())
        topics = {c["id"]: c["pre"]["topic"] for c in pack["cards"]}
        total = Counter()
        for index, dealt in enumerate(table["experts"].values()):
            self.assertEqual(len(set(dealt)), 10)
            counts = Counter(topics[c] for c in dealt)
            self.assertEqual(counts["跨诊断脑连接"], 3)
            self.assertEqual(counts["影像遗传学"], 3 if index % 2 == 0 else 4)
            self.assertEqual(counts["预后研究"], 4 if index % 2 == 0 else 3)
            total.update(counts)
        self.assertEqual(total, {"跨诊断脑连接": 30, "影像遗传学": 35, "预后研究": 35})
        self.assertEqual(set(table["coverage"]), set(topics))
        self.assertTrue(all(table["coverage"].values()))

    def test_he2_only_order_changes_not_membership_or_difficulty(self):
        old = json.loads((RANKING / "case1_tcp_expert_pair_assignments_v2.json").read_bytes())
        new = build_he2()
        bank = json.loads((RANKING / "case1_tcp_expert_study_v2.json").read_bytes())
        difficulty = {p["pair_id"]: p["difficulty"] for p in bank["pair_schedule"]}
        for field in ("experts", "coverage", "candidate_manifest_sha256", "pairs_per_expert", "sessions_per_expert", "pairs_per_session"):
            self.assertEqual(new[field], old[field])
        for expert, rounds in new["sessions"].items():
            for number, pairs in rounds.items():
                self.assertEqual(set(pairs), set(old["sessions"][expert][number]))
                self.assertEqual(Counter(difficulty[p] for p in pairs), {"easy": 4, "medium": 4, "hard": 4})

    def test_every_pair_once_per_band_and_exact_two_if_prefix_eight(self):
        positions = defaultdict(list)
        prefix = Counter()
        for rounds in build_he2()["sessions"].values():
            for pairs in rounds.values():
                prefix.update(pairs[:8])
                for index, pair in enumerate(pairs):
                    positions[pair].append(index // 4)
        self.assertEqual(len(positions), 240)
        self.assertTrue(all(sorted(bands) == [0, 1, 2] for bands in positions.values()))
        self.assertEqual(set(prefix.values()), {2})
        self.assertEqual(len(prefix), 240)

    def test_bad_order_or_membership_rejected(self):
        original = build_he2()
        ranking.validate_prebalanced_pair_table(original)
        for corruption in ("duplicate", "swap_bands", "coverage", "unknown_policy"):
            table = copy.deepcopy(original)
            pairs = table["sessions"]["P01"]["1"]
            if corruption == "duplicate":
                pairs[0] = pairs[1]
            elif corruption == "swap_bands":
                pairs[0], pairs[8] = pairs[8], pairs[0]
            elif corruption == "coverage":
                table["coverage"][pairs[0]] = 4
            else:
                table["order_policy"] = "random"
            with self.assertRaises(ValueError):
                ranking.validate_prebalanced_pair_table(table)


class AssignmentServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_he1_keeps_existing_shuffle_and_freezes_new_allocation(self):
        service = DiscoveryStudy(MATERIALS / "cs1_discovery_pilot_v10.json", self.directory / "he1", record_kind="test")
        with patch("core.web.discovery_study.secrets.SystemRandom") as rng:
            rng.return_value.shuffle.side_effect = lambda order: order.reverse()
            view = service.create({"code": "TEST-THIRTY", "experience": "3-5", "assignment_id": "P01"})
            rng.return_value.shuffle.assert_called_once()
        selected = set(build_he1()["experts"]["P01"])
        bank_order = [c["id"] for c in service.pack["cards"] if c["id"] in selected]
        self.assertEqual(view["session"]["order"], list(reversed(bank_order)))
        self.assertEqual(view["session"]["allocation_revision"], HE1_REVISION)
        self.assertEqual(service.get(view["session"]["id"], view["session_token"])["session"], view["session"])

    def test_he1_preexisting_six_case_allocation_is_not_changed(self):
        old = json.loads((MATERIALS / "cs1_discovery_assignments_v5.json").read_bytes())
        with patch("core.web.discovery_study.load_assignments", return_value=old):
            service = DiscoveryStudy(MATERIALS / "cs1_discovery_pilot_v10.json", self.directory / "he1", record_kind="test")
            view = service.create({"code": "TEST-OLD-SIX", "experience": "3-5", "assignment_id": "P01"})
        new = DiscoveryStudy(MATERIALS / "cs1_discovery_pilot_v10.json", self.directory / "he1", record_kind="test")
        restored = new.get(view["session"]["id"], view["session_token"])
        self.assertEqual(restored["session"], view["session"])
        self.assertEqual(restored["cards"], view["cards"])
        self.assertEqual(Counter(c["pre"]["topic"] for c in restored["cards"])["跨诊断脑连接"], 6)

    def create_ranking(self, service, **extra):
        args = dict(study_id="TEST-ORDER", participant_id="TEST-ORDER", condition="manual",
                    candidate_path=RANKING / "case1_tcp_expert_study_v2.json", assignment_id="P01")
        args.update(extra)
        return service.create_session(**args)

    def test_he2_all_sixty_sessions_follow_exact_frozen_order(self):
        service = ranking.UserStudyService(self.directory / "he2")
        expected = build_he2()
        for expert, rounds in expected["sessions"].items():
            for number, ids in rounds.items():
                view = self.create_ranking(service, participant_id="TEST-" + expert, assignment_id=expert)
                self.assertEqual(view["session_number"], int(number))
                self.assertEqual(view["target_active_seconds"], 600)
                self.assertEqual([p["pair_id"] for p in view["pairs"]], ids)
                restored = service.get_session(view["session_id"], include_events=True)
                self.assertEqual(restored["pairs"], view["pairs"])
                self.assertEqual(restored["events"][0]["payload"]["pair_order_policy"], HE2_POLICY)

    def test_he2_existing_session_and_preview_keep_original_order(self):
        service = ranking.UserStudyService(self.directory / "he2")
        old = json.loads((RANKING / "case1_tcp_expert_pair_assignments_v2.json").read_bytes())
        with patch.object(ranking, "load_pair_assignments", return_value=old):
            view = self.create_ranking(service)
        self.assertEqual(service.get_session(view["session_id"])["pairs"], view["pairs"])
        next_view = self.create_ranking(service)
        self.assertEqual([p["pair_id"] for p in next_view["pairs"]], build_he2()["sessions"]["P01"]["2"])
        preview = self.create_ranking(service, participant_id="TEST-ALL", assignment_id="ALL")
        bank = json.loads((RANKING / "case1_tcp_expert_study_v2.json").read_bytes())
        self.assertEqual([p["pair_id"] for p in preview["pairs"]],
                         [p["pair_id"] for p in bank["pair_schedule"] if p["session_number"] == 1])

    def test_he2_legacy_table_still_loads_and_export_preserves_order(self):
        legacy = RANKING / "case1_tcp_external_expert_study_v1.json"
        import hashlib
        self.assertEqual(ranking.load_pair_assignments(legacy, hashlib.sha256(legacy.read_bytes()).hexdigest())["version"], 1)
        service = ranking.UserStudyService(self.directory / "he2")
        view = self.create_ranking(service)
        # The public handoff reads the session's frozen sequence, never the current table.
        exported = ranking_export(service.get_session(view["session_id"], include_events=True))
        self.assertEqual([p["pair_id"] for p in exported["questions"]], [p["pair_id"] for p in view["pairs"]])


if __name__ == "__main__":
    unittest.main()
