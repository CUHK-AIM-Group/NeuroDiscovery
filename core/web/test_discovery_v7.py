import copy
import hashlib
import json
import unittest

from core.web.build_discovery_assignments_v2 import build as build_assignments
from core.web.build_discovery_pilot_v7 import (
    CALIBRATION, TARGET as PACK_PATH, V5, V6, build as build_pack,
)
from core.web.build_reference_notes_v2 import build as build_reference_notes
from core.web.build_significance_v2 import build as build_significance
from core.web.build_translation_v4 import build as build_catalog


class DiscoveryV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
        cls.v5 = json.loads(V5.read_text(encoding="utf-8"))
        cls.v6 = json.loads(V6.read_text(encoding="utf-8"))

    def test_builder_matches_frozen_output(self):
        self.assertEqual(build_pack(), self.pack)

    def test_mixed_panel_and_upfront_labels(self):
        contexts = [card["pre"]["study_context"] for card in self.pack["cards"]]
        self.assertEqual(sum(c["alignment"] == "consistent" for c in contexts), 30)
        self.assertEqual(sum(c["alignment"] == "not_consistent" for c in contexts), 4)
        self.assertTrue(all(c["proposer"] == "NeuroDiscovery" for c in contexts))
        self.assertTrue(all(c["origin_zh"] == "本假设由 NeuroDiscovery 提出" for c in contexts))
        self.assertEqual({c["alignment_zh"] for c in contexts},
                         {"实验结果：与假设一致", "实验结果：与假设不一致"})

    def test_questions_and_scoring_are_unchanged(self):
        self.assertEqual(self.pack["questions"], self.v6["questions"])
        self.assertEqual(self.pack["scoring"], self.v6["scoring"])
        self.assertEqual(len(self.pack["questions"]), 6)
        for question in self.pack["questions"]:
            self.assertEqual([o["score"] for o in question["options"][:5]], [1, 2, 3, 4, 5])

    def test_supported_scientific_payload_is_v6_top_30(self):
        for actual, parent in zip(self.pack["cards"][:30], self.v6["cards"][:30]):
            actual = copy.deepcopy(actual)
            actual["pre"].pop("study_context")
            self.assertEqual(actual, parent)

    def test_calibration_payloads_are_real_completed_v5_cases(self):
        v5_cards = {card["id"]: card for card in self.v5["cards"]}
        v5_map = {entry["id"]: entry for entry in self.v5["organizer"]["mapping"]}
        for card, private, (old_id, hypothesis_id) in zip(
                self.pack["cards"][30:], self.pack["organizer"]["mapping"][30:], CALIBRATION):
            original = copy.deepcopy(v5_cards[old_id])
            original["id"] = card["id"]
            actual = copy.deepcopy(card)
            actual["pre"].pop("study_context")
            self.assertEqual(actual, original)
            self.assertEqual(private["hypothesis_id"], hypothesis_id)
            self.assertLess(card["post"]["ucla_joint"], 0.8)
            self.assertTrue(private["source_evidence"]["candidate"]["complete_internal_external_empirical_package"])
            self.assertEqual(v5_map[old_id]["hypothesis_id"], hypothesis_id)

    def test_assignment_burden_and_calibration_coverage(self):
        table = build_assignments(PACK_PATH.read_bytes())
        self.assertEqual(len(table["experts"]), 10)
        self.assertEqual(sum(table["coverage"].values()), 100)
        self.assertTrue(all(len(cards) == len(set(cards)) == 10 for cards in table["experts"].values()))
        self.assertTrue(all(table["coverage"][f"packet-{rank:02d}"] == 3 for rank in range(31, 35)))

    def test_bound_display_sidecars_cover_every_card(self):
        digest = hashlib.sha256(PACK_PATH.read_bytes()).hexdigest()
        notes = build_reference_notes(PACK_PATH.read_bytes())
        significance = build_significance(PACK_PATH.read_bytes())
        catalog = build_catalog()
        self.assertEqual(notes["pack_sha256"], digest)
        self.assertEqual(significance["pack_sha256"], digest)
        self.assertEqual(set(notes["notes"]), {c["id"] for c in self.pack["cards"]})
        self.assertEqual(set(significance["significance"]), {c["id"] for c in self.pack["cards"]})
        self.assertEqual(catalog["source_sha256"], digest)
        self.assertEqual(catalog["strings"]["校准材料"], "Calibration case")


if __name__ == "__main__":
    unittest.main()
