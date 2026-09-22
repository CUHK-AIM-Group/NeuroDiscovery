"""v6 expert assignment table: deterministic dealing and server-side scoping."""
import copy
import json
import tempfile
import unittest

from core.web.build_discovery_assignments_v1 import (
    build, EXPERTS, PACK, PACK_SHA256, TARGET, CARDS_PER_EXPERT,
)
from core.web.discovery_study import DiscoveryStudy, StudyError

PROFILE = {"code": "TEST-V6A", "experience": "3-5", "consent": True}


class V6AssignmentTableTests(unittest.TestCase):
    def test_build_is_deterministic_and_frozen_to_the_v6_pack(self):
        self.assertEqual(build(), build())
        table = json.loads(TARGET.read_text(encoding="utf-8"))
        self.assertEqual(table, build())
        self.assertEqual(table["pack_id"], "cs1-discovery-capabilities-20260918-v6")
        self.assertEqual(table["pack_sha256"], PACK_SHA256)
        with self.assertRaisesRegex(ValueError, "v6 pack changed"):
            build(b"{}")

    def test_every_expert_gets_ten_distinct_cards(self):
        table = build()
        self.assertEqual(sorted(table["experts"]), EXPERTS)
        for dealt in table["experts"].values():
            self.assertEqual(len(dealt), CARDS_PER_EXPERT)
            self.assertEqual(len(set(dealt)), CARDS_PER_EXPERT)
        self.assertEqual(sum(len(dealt) for dealt in table["experts"].values()), 100)

    def test_coverage_is_three_for_ranks_1_to_32_and_two_for_33_34(self):
        table = build()
        for rank in range(1, 35):
            card = f"packet-{rank:02d}"
            expected = 3 if rank <= 32 else 2
            self.assertEqual(table["coverage"][card], expected, card)
            holders = [expert for expert, dealt in table["experts"].items() if card in dealt]
            self.assertEqual(len(holders), expected, card)


class V6AssignmentServerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(PACK, data_dir=temp.name, record_kind="test")

    def create(self, **extra):
        return self.service.create({**copy.deepcopy(PROFILE), **extra})

    def test_config_exposes_assignment_options(self):
        options = self.service.config()["assignments"]["options"]
        self.assertEqual(options[0], {"id": "ALL", "card_count": 34})
        self.assertEqual([o["id"] for o in options[1:]], EXPERTS)
        self.assertTrue(all(o["card_count"] == 10 for o in options[1:]))

    def test_assigned_session_contains_only_the_experts_cards(self):
        table = build()
        for expert in ["P03", "P10"]:
            view = self.create(assignment_id=expert)
            self.assertEqual(view["session"]["assignment_id"], expert)
            self.assertEqual(len(view["cards"]), 10)
            self.assertEqual({card["id"] for card in view["cards"]}, set(table["experts"][expert]))
            self.assertEqual(sorted(view["session"]["order"]), sorted(table["experts"][expert]))
            exported = self.service.export(view["session"]["id"], view["session_token"])
            self.assertEqual({card["id"] for card in exported["cards"]}, set(table["experts"][expert]))
            self.assertEqual(exported["session"]["assignment_id"], expert)

    def test_assignment_id_is_case_insensitive_and_blank_means_preview_all(self):
        view = self.create(assignment_id=" p07 ")
        self.assertEqual(view["session"]["assignment_id"], "P07")
        for extra in [{}, {"assignment_id": ""}, {"assignment_id": "ALL"}]:
            view = self.create(**extra)
            self.assertEqual(view["session"]["assignment_id"], "ALL")
            self.assertEqual(len(view["cards"]), 34)

    def test_unknown_assignment_is_rejected(self):
        with self.assertRaisesRegex(StudyError, "未知的分配编号"):
            self.create(assignment_id="P99")
        with self.assertRaisesRegex(StudyError, "未知的分配编号"):
            self.create(assignment_id="p01x")

    def test_chinese_name_code_matches_the_entry_form(self):
        view = self.create(code="张三")
        self.assertEqual(view["session"]["profile"]["code"], "张三")
        with self.assertRaises(StudyError):
            self.create(code="x")

    def test_create_event_records_the_assignment(self):
        view = self.create(assignment_id="P05")
        events = self.service.export(view["session"]["id"], view["session_token"])["events"]
        self.assertEqual(events[0]["kind"], "create")
        with self.service.connection() as db:
            row = db.execute("SELECT payload FROM events WHERE session_id=? AND kind='create'",
                             (view["session"]["id"],)).fetchone()
        self.assertEqual(json.loads(row["payload"])["assignment_id"], "P05")

    def test_unbound_assignment_table_is_ignored_and_mismatched_table_rejected(self):
        import shutil
        from pathlib import Path
        import core.web.discovery_study as module
        table = json.loads(TARGET.read_text(encoding="utf-8"))
        # A table naming another pack does not constrain this service.
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "pack.json"
            shutil.copy(PACK.with_name("cs1_discovery_pilot_v5.json"), pack_copy)
            shutil.copy(TARGET, Path(temp) / module.ASSIGNMENTS_NAME)
            service = DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")
            self.assertIsNone(service.assignments)
        # A table naming this pack with a wrong hash is a packaging error.
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "cs1_discovery_pilot_v6.json"
            pack_copy.write_bytes(PACK.read_bytes())
            broken = dict(table, pack_sha256="0" * 64)
            (Path(temp) / module.ASSIGNMENTS_NAME).write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")


if __name__ == "__main__":
    unittest.main()
