"""Multi-topic panel, conditional feedback scoring and historical compatibility."""
import copy
import hashlib
import json
import re
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from core.web.build_discovery_pilot_v8 import HERE, RETAIN, SOURCE_NAME, SOURCE_SHA, V7_SHA, build
from core.web.discovery_localization import catalog, strings_in
from core.web.discovery_scoring import score_session, applicable_item_ids, aggregate_exports
from core.web.discovery_study import DiscoveryStudy, StudyError

PACK = HERE / "cs1_discovery_pilot_v8.json"
SOURCE = Path("C:/Users/45846/Desktop/NeuroDiscovery_Eight_Topic_Records_EN_20260920.zip")


class V8PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = json.loads(PACK.read_bytes())

    def test_topic_balance_and_assignments(self):
        self.assertEqual(Counter(c["pre"]["topic"] for c in self.pack["cards"]),
                         {"跨诊断脑连接": 20, "影像遗传学": 7, "预后研究": 7})
        assignments = json.loads((HERE / "cs1_discovery_assignments_v3.json").read_bytes())
        by_id = {c["id"]: c for c in self.pack["cards"]}
        for cards in assignments["experts"].values():
            self.assertEqual(len(set(cards)), 10)
            self.assertEqual(Counter(by_id[c]["pre"]["topic"] for c in cards),
                             {"跨诊断脑连接": 6, "影像遗传学": 2, "预后研究": 2})
        self.assertEqual(sum(assignments["coverage"].values()), 100)
        self.assertEqual(set(assignments["coverage"].values()), {2, 3})

    def test_old_science_is_unchanged(self):
        raw = (HERE / "cs1_discovery_pilot_v7.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), V7_SHA)
        old = json.loads(raw)
        for card, index in zip(self.pack["cards"][:20], RETAIN):
            self.assertEqual(card["post"], old["cards"][index - 1]["post"])

    @unittest.skipUnless(SOURCE.exists(), "Author export not installed on this machine")
    def test_every_numeric_record_is_from_the_selected_source_occurrence(self):
        with zipfile.ZipFile(SOURCE) as archive:
            raw = archive.read(SOURCE_NAME)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA)
        rows = json.loads(raw)["scored_and_tested"]
        groups = {}
        for row in sorted(rows, key=lambda r: (r["seed"], r["round"], r["hypothesis_id"])):
            groups.setdefault((row["topic"], *row["measurement_pair"]), row)
        self.assertEqual(len(groups), 14)
        for card, mapping in zip(self.pack["cards"][20:], self.pack["organizer"]["mapping"][20:]):
            row = groups[(mapping["topic"], *mapping["measurement_pair"])]
            self.assertEqual(mapping["hypothesis_id"], row["hypothesis_id"])
            self.assertEqual(card["pre"]["hypothesis"], row["hypothesis"])
            self.assertEqual(len(card["post"]["experimental_results"]), 3)
            for actual, original in zip(card["post"]["experimental_results"], row["configurations"]):
                for phase in ("internal", "external"):
                    self.assertEqual(actual[phase], original[phase])
        self.assertEqual(build(SOURCE)[PACK.name], PACK.read_bytes())

    def test_new_cards_have_no_forced_feedback_requirement(self):
        for card in self.pack["cards"][20:]:
            self.assertEqual(len(applicable_item_ids(self.pack, card["id"])), 5)
            self.assertNotIn("feedback", applicable_item_ids(self.pack, card["id"]))

    def test_all_public_chinese_strings_have_english_translations(self):
        translations = catalog()[0]["strings"]
        fields = {k: self.pack[k] for k in ("public_meta", "questions", "cards")}
        # Old audit-only comments remain in the frozen scientific records but
        # are not displayed by the new renderer.
        def visible(obj):
            if isinstance(obj, dict):
                return {k: visible(v) for k, v in obj.items() if k not in {"transfer_limit", "note", "verification", "audit_note", "source_note"}}
            if isinstance(obj, list):
                return [visible(v) for v in obj]
            return obj
        missing = {s for s in strings_in(visible(fields)) if re.search("[\u4e00-\u9fff]", s) and s not in translations}
        self.assertEqual(missing, set())


class V8WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(PACK, temp.name, record_kind="test")

    def complete(self, assignment="P01", unable=False):
        view = self.service.create({"code": "TEST-" + assignment, "experience": "3-5", "assignment_id": assignment})
        token = view["session_token"]
        for i, card in enumerate(view["cards"]):
            answers = {key: "4" for key in applicable_item_ids(self.service.pack, card["id"])}
            if unable and i == 0:
                answers["novelty"] = "unable"
            view = self.service.mutate(view["session"]["id"], token, "save", {
                "request_id": f"v8-save-{i:03d}", "revision": view["session"]["revision"],
                "card_id": card["id"], "answers": answers, "note": "", "issues": []})
        result = self.service.mutate(view["session"]["id"], token, "submit", {
            "request_id": "v8-submit-000", "revision": view["session"]["revision"]})
        return result, self.service.export(view["session"]["id"], token)

    def test_can_finish_without_parent_feedback_and_export_exact_denominators(self):
        view, export = self.complete()
        self.assertEqual(view["score_summary"]["composite_mean"], 4)
        self.assertEqual(view["score_summary"]["complete_case_count"], 10)
        expected = sum(len(applicable_item_ids(self.service.pack, cid)) for cid in view["session"]["order"])
        self.assertEqual(export["summary"]["total_questions"], expected)
        self.assertEqual(export["summary"]["answered_questions"], expected)
        self.assertEqual(export["summary"]["cards_fully_answered"], 10)
        for card in export["score_summary"]["cards"].values():
            if "feedback" in card["not_applicable"]:
                self.assertEqual(card["numeric_count"], 5)
                self.assertEqual(card["composite"], 4)
                self.assertNotIn("feedback", card["unavailable"])

    def test_inapplicable_feedback_cannot_be_submitted_as_real_score(self):
        view = self.service.create({"code": "TEST-NA", "experience": "3-5", "assignment_id": "P01"})
        card = next(c for c in view["cards"] if c["pre"]["topic"] == "影像遗传学")
        with self.assertRaises(StudyError):
            self.service.mutate(view["session"]["id"], view["session_token"], "save", {
                "request_id": "v8-invalid-000", "revision": 0, "card_id": card["id"], "answers": {"feedback": "5"}})

    def test_unable_remains_unscored_not_zero(self):
        view, export = self.complete(unable=True)
        self.assertEqual(view["score_summary"]["complete_case_count"], 9)
        self.assertEqual(export["summary"]["cannot_judge"], 1)
        self.assertEqual(view["score_summary"]["dimensions"]["novelty"]["n"], 9)

    def test_different_assignments_aggregate_without_treating_unassigned_as_missing(self):
        exports = []
        for assignment in ("P01", "P02"):
            _, result = self.complete(assignment)
            result = copy.deepcopy(result)
            result["session"]["record_kind"] = "pilot"  # synthetic unit-test fixture
            exports.append(result)
        result = aggregate_exports(exports)
        self.assertEqual(result["expert_count"], 2)
        self.assertEqual(result["composite_mean"], 4)
        self.assertEqual(result["dimensions"]["feedback"]["mean"], 4)
        self.assertLess(result["dimensions"]["feedback"]["applicable_case_count"], result["case_count"])

    def test_old_six_item_scoring_does_not_change(self):
        old = json.loads((HERE / "cs1_discovery_pilot_v7.json").read_bytes())
        state = {"order": ["packet-01"], "answers": {"packet-01": {q: "4" for q in old["scoring"]["item_ids"] if q != "feedback"}}}
        self.assertIsNone(score_session(state, old)["composite_mean"])


if __name__ == "__main__":
    unittest.main()
