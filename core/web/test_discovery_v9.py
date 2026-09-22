"""Case-specific meaning is bilingual, session-bound, and science-preserving."""
import copy
import json
import re
import tempfile
import unittest
from core.web.build_discovery_pilot_v9 import HERE, CONTENT_REVISION, build
from core.web.discovery_study import DiscoveryStudy


class V9SignificanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = json.loads((HERE / "cs1_discovery_pilot_v9.json").read_bytes())
        cls.parent = json.loads((HERE / "cs1_discovery_pilot_v8.json").read_bytes())

    def test_every_case_has_individual_bilingual_prose(self):
        notes = [card["pre"]["significance"] for card in self.pack["cards"]]
        self.assertEqual(len(notes), 34)
        for lang in ("zh", "en"):
            self.assertEqual(len({note[lang] for note in notes}), 34)
        for note in notes:
            self.assertEqual(set(note), {"zh", "en"})
            self.assertTrue("临床" in note["zh"] or "康复" in note["zh"])
            self.assertGreater(len(note["zh"]), 100)
            self.assertLess(len(note["zh"]), 230)
            self.assertTrue(45 <= len(note["en"].split()) <= 105)
            self.assertFalse(re.search(r"[\u3400-\u9fff]", note["en"]))
        sentences = [s for note in notes for s in note["zh"].split("。") if len(s) > 25]
        self.assertEqual(len(sentences), len(set(sentences)))

    def test_all_original_hypotheses_methods_results_and_scores_unchanged(self):
        for new, old in zip(self.pack["cards"], self.parent["cards"]):
            clean = copy.deepcopy(new)
            clean["pre"].pop("significance")
            self.assertEqual(clean, old)
        for key in ("questions", "scoring", "organizer", "common_pre", "common_post"):
            self.assertEqual(self.pack[key], self.parent[key])

    def test_deterministic_build_matches_release(self):
        for name, raw in build().items():
            self.assertEqual(raw, (HERE / name).read_bytes(), name)

    def test_new_english_hypotheses_are_readable_not_variable_names(self):
        catalog = json.loads((HERE / "cs1_discovery_en_v7.json").read_bytes())
        for card in self.pack["cards"][20:]:
            sentence = catalog["strings"][card["pre"]["hypothesis_plain"]]
            self.assertNotIn("_", sentence)
            self.assertTrue("expected" in sentence or "hypothesis" in sentence)

    def test_ten_case_assignments_unchanged(self):
        old = json.loads((HERE / "cs1_discovery_assignments_v3.json").read_bytes())
        new = json.loads((HERE / "cs1_discovery_assignments_v4.json").read_bytes())
        self.assertEqual(old["experts"], new["experts"])
        self.assertEqual(old["coverage"], new["coverage"])

    def test_config_and_session_carry_same_content_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            service = DiscoveryStudy(HERE / "cs1_discovery_pilot_v9.json", data_dir=folder, record_kind="test")
            config = service.config()
            self.assertEqual(config["meta"]["content_revision"], CONTENT_REVISION)
            created = service.create({"code": "TEST-V9", "experience": "3-5", "assignment_id": "P01"})
            self.assertEqual(len(created["cards"]), 10)
            for card in created["cards"]:
                self.assertEqual(card["pre"]["significance"], config["significance"][card["id"]])
            restored = service.get(created["session"]["id"], created["session_token"])
            self.assertEqual(restored["cards"], created["cards"])

    def test_previous_material_snapshot_and_answers_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            old = DiscoveryStudy(HERE / "cs1_discovery_pilot_v8.json", data_dir=folder, record_kind="test")
            created = old.create({"code": "TEST-OLD", "experience": "3-5", "assignment_id": "P01"})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v9.json", data_dir=folder, record_kind="test")
            restored = new.get(created["session"]["id"], created["session_token"])
            self.assertEqual(created["session"], restored["session"])
            self.assertEqual(created["cards"], restored["cards"])
            self.assertTrue(all("significance" not in card["pre"] for card in restored["cards"]))


if __name__ == "__main__":
    unittest.main()
