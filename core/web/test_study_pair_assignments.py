"""Extension pair-share assignments: v2 (240-pair bank) and legacy v1 coverage."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from core.web import build_study_pair_assignments_v1 as v1_mod
from core.web import build_study_pair_assignments_v2 as v2_mod

_SERVICE_PATH = Path(__file__).resolve().parents[2] / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurodiscovery_user_study_assignments", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
UserStudyService = _MODULE.UserStudyService
load_pair_assignments = _MODULE.load_pair_assignments


class PairAssignmentV2TableTests(unittest.TestCase):
    def test_deterministic_and_frozen_to_the_v2_bank(self):
        self.assertEqual(v2_mod.build(), v2_mod.build())
        on_disk = json.loads(v2_mod.TARGET.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, v2_mod.build())
        self.assertEqual(on_disk["candidate_manifest_sha256"], v2_mod.BANK_SHA256)
        with self.assertRaisesRegex(ValueError, "bank changed"):
            v2_mod.build(b"{}")

    def test_three_reviewers_balanced_sessions(self):
        table = v2_mod.build()
        self.assertEqual(set(table["coverage"].values()), {3})
        self.assertEqual(len(table["coverage"]), 240)
        bank = json.loads(v2_mod.BANK.read_text(encoding="utf-8"))
        difficulty = {str(p["pair_id"]): str(p["difficulty"]) for p in bank["pair_schedule"]}
        for expert in v2_mod.EXPERTS:
            dealt = table["experts"][expert]
            self.assertEqual(len(dealt), 72)
            self.assertEqual(len(set(dealt)), 72)
            counts = {d: sum(1 for p in dealt if difficulty[p] == d) for d in ("easy", "medium", "hard")}
            self.assertEqual(counts, {"easy": 24, "medium": 24, "hard": 24}, expert)
            sessions = table["sessions"][expert]
            self.assertEqual(sorted(sessions), ["1", "2", "3", "4", "5", "6"])
            for session_pairs in sessions.values():
                self.assertEqual(len(session_pairs), 12)
                counts = {d: sum(1 for p in session_pairs if difficulty[p] == d) for d in ("easy", "medium", "hard")}
                self.assertEqual(counts, {"easy": 4, "medium": 4, "hard": 4}, expert)

    def test_pair_instances_land_on_different_experts(self):
        table = v2_mod.build()
        for pair_id in table["coverage"]:
            holders = [expert for expert, dealt in table["experts"].items() if pair_id in dealt]
            self.assertEqual(len(set(holders)), 3, pair_id)


class PairAssignmentV2ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = UserStudyService(Path(self.temp.name) / "study")

    def create(self, **extra):
        payload = dict(
            study_id="case1-tcp-external-validation-v2",
            participant_id="TEST-V2",
            condition="manual",
            candidate_path=v2_mod.BANK,
            experience="3-5",
            consent=True,
        )
        payload.update(extra)
        return self.service.create_session(**payload)

    def test_assigned_session_uses_the_share_session_one(self):
        table = v2_mod.build()
        session = self.create(assignment_id="p04")
        self.assertEqual(session["assignment_id"], "P04")
        self.assertEqual(session["required_sessions"], 6)
        self.assertEqual(session["session_number"], 1)
        self.assertEqual(len(session["pairs"]), 12)
        self.assertEqual({p["pair_id"] for p in session["pairs"]},
                         set(table["sessions"]["P04"]["1"]))

    def test_blank_or_all_keeps_the_bank_schedule(self):
        for index, extra in enumerate([{}, {"assignment_id": ""}, {"assignment_id": "ALL"}]):
            session = self.create(participant_id=f"TEST-V2-ALL-{index}", **extra)
            self.assertEqual(session["assignment_id"], "ALL")
            self.assertEqual(len(session["pairs"]), 40)
            self.assertEqual(session["required_sessions"], 6)

    def test_unknown_assignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知的分配编号"):
            self.create(assignment_id="P99")


class PairAssignmentV1LegacyTests(unittest.TestCase):
    def test_v1_table_still_binds_the_v1_bank(self):
        table = v1_mod.build()
        self.assertEqual(len(table["experts"]["P01"]), 12)
        self.assertEqual(set(table["coverage"].values()), {1})
        self.assertEqual(len(table["coverage"]), 120)
        self.assertEqual(table["candidate_manifest_sha256"], v1_mod.BANK_SHA256)


if __name__ == "__main__":
    unittest.main()
