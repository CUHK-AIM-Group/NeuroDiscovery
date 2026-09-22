"""Readable case sections are bilingual, snapshot-bound, and science-preserving."""
import copy
import json
import re
import tempfile
import unittest
from core.web.build_discovery_pilot_v10 import HERE, CONTENT_REVISION, build
from core.web.discovery_study import DiscoveryStudy


class V10ReadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = json.loads((HERE / "cs1_discovery_pilot_v10.json").read_bytes())
        cls.parent = json.loads((HERE / "cs1_discovery_pilot_v9.json").read_bytes())

    def test_individual_bilingual_sections_for_all_cases(self):
        self.assertEqual(len(self.pack["cards"]), 34)
        for field in ("rationale", "methods", "results"):
            for lang in ("zh", "en"):
                texts = [card["pre"]["reading"][field][lang] for card in self.pack["cards"]]
                self.assertEqual(len(set(texts)), 34)
                self.assertTrue(all(30 < len(text) < 750 for text in texts))
        for card in self.pack["cards"]:
            reading = card["pre"]["reading"]
            for field in ("rationale", "methods", "results", "feedback"):
                if not reading[field]:
                    continue
                self.assertEqual(set(reading[field]), {"en", "zh"})
                self.assertFalse(re.search(r"[\u3400-\u9fff]", reading[field]["en"]))
            self.assertEqual(set(reading["references"]), {ref["id"] for ref in card["pre"]["references"]})

    def test_original_case_and_scoring_fields_are_unchanged(self):
        for new, old in zip(self.pack["cards"], self.parent["cards"]):
            clean = copy.deepcopy(new)
            clean["pre"].pop("reading")
            self.assertEqual(clean, old)
        for key in ("questions", "scoring", "organizer", "common_pre", "common_post"):
            self.assertEqual(self.pack[key], self.parent[key])

    def test_feedback_is_optional_and_case_specific(self):
        for card in self.pack["cards"]:
            feedback = card["post"].get("feedback", {})
            expected = bool(feedback.get("parent_endpoint") and not feedback.get("no_parent"))
            self.assertEqual(bool(card["pre"]["reading"]["feedback"]), expected, card["id"])

    def test_deterministic_build(self):
        for name, raw in build().items():
            self.assertEqual(raw, (HERE / name).read_bytes(), name)

    def test_assignment_and_source_sidecars_remain_unchanged(self):
        for kind in ("assignments", "reference_notes", "significance"):
            old = json.loads((HERE / f"cs1_discovery_{kind}_v4.json").read_bytes())
            new = json.loads((HERE / f"cs1_discovery_{kind}_v5.json").read_bytes())
            for field in ("version", "pack_id", "pack_sha256"):
                old.pop(field, None)
                new.pop(field, None)
            self.assertEqual(old, new)

    def test_new_session_freezes_reading_and_restores_it(self):
        with tempfile.TemporaryDirectory() as folder:
            service = DiscoveryStudy(HERE / "cs1_discovery_pilot_v10.json", data_dir=folder, record_kind="test")
            self.assertEqual(service.config()["meta"]["content_revision"], CONTENT_REVISION)
            created = service.create({"code": "TEST-V10", "experience": "3-5", "assignment_id": "P01"})
            self.assertEqual(len(created["cards"]), 10)
            self.assertTrue(all(card["pre"]["reading"] for card in created["cards"]))
            self.assertEqual(created["cards"], service.get(created["session"]["id"], created["session_token"])["cards"])

    def test_previous_sessions_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            old = DiscoveryStudy(HERE / "cs1_discovery_pilot_v9.json", data_dir=folder, record_kind="test")
            created = old.create({"code": "TEST-OLD", "experience": "3-5", "assignment_id": "P01"})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v10.json", data_dir=folder, record_kind="test")
            restored = new.get(created["session"]["id"], created["session_token"])
            self.assertEqual(created["session"], restored["session"])
            self.assertEqual(created["cards"], restored["cards"])
            self.assertTrue(all("reading" not in card["pre"] for card in restored["cards"]))


if __name__ == "__main__":
    unittest.main()
