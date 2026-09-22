"""One unscored option; historical answers retain their original categories."""
import copy
import json
import tempfile
import unittest

from core.web.build_discovery_pilot_v4 import build, PARENT, TARGET as V4_PACK
from core.web.discovery_study import DiscoveryStudy, StudyError

PROFILE = {"code": "TEST-V4", "experience": "3-5", "consent": True}


class UnscoredOptionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(V4_PACK, data_dir=temp.name, record_kind="test")
        self.view = self.service.create(copy.deepcopy(PROFILE))
        self.sid, self.token = self.view["session"]["id"], self.view["session_token"]
        self.counter = 0

    def change(self, action, **payload):
        self.counter += 1
        self.view = self.service.mutate(self.sid, self.token, action, {
            "request_id": f"v4-check-{self.counter:03}", "revision": self.view["session"]["revision"], **payload})
        return self.view

    def fill(self, value):
        for card in self.view["cards"]:
            self.change("save", card_id=card["id"], answers={q["id"]: value for q in self.view["questions"]})

    def test_single_null_option_and_unchanged_science_numeric_anchors(self):
        old = json.loads(PARENT.read_text(encoding="utf-8"))
        self.assertEqual(build(), self.service.pack)
        self.assertEqual(old["cards"], self.service.pack["cards"])
        self.assertEqual(old["organizer"]["mapping"], self.service.pack["organizer"]["mapping"])
        self.assertEqual(self.service.pack["scoring"]["non_numeric"], ["unable"])
        for before, after in zip(old["questions"], self.view["questions"]):
            self.assertEqual(len(after["options"]), 6)
            self.assertEqual(after["options"][:5], before["options"][:5])
            self.assertEqual(after["options"][-1], {"value": "unable", "score": None, "label": "无法判断（不计分）"})
            self.assertEqual({k:v for k,v in before.items() if k != "options"}, {k:v for k,v in after.items() if k != "options"})

    def test_null_excluded_from_dimension_mean_not_zero(self):
        self.fill("4")
        self.change("save", card_id=self.view["cards"][0]["id"], answers={"novelty": "unable"})
        self.change("submit")
        scores = self.view["score_summary"]
        self.assertEqual(scores["dimensions"]["novelty"]["mean"], 4)
        self.assertEqual(scores["dimensions"]["novelty"]["n"], 9)
        self.assertEqual(scores["dimensions"]["novelty"]["unavailable_counts"], {"unable": 1})
        self.assertEqual(scores["complete_case_count"], 9)
        self.assertEqual(scores["composite_mean"], 4)

    def test_all_unable_is_completed_but_has_no_numeric_mean(self):
        self.fill("unable"); self.change("submit")
        self.assertIsNone(self.view["score_summary"]["composite_mean"])
        self.assertEqual(self.view["score_summary"]["complete_case_count"], 0)
        result = self.service.export(self.sid, self.token)
        self.assertEqual(result["session"]["stage"], "complete")
        self.assertNotIn(self.token, json.dumps(result))

    def test_obsolete_options_rejected_by_new_protocol(self):
        for value in ["insufficient", "outside", "0"]:
            with self.assertRaises(StudyError):
                self.change("save", card_id=self.view["cards"][0]["id"], answers={"novelty": value})

    def test_old_v3_options_answers_and_export_preserved(self):
        old = DiscoveryStudy(PARENT, self.service.data_dir, record_kind="test")
        view = old.create({**PROFILE, "code": "TEST-OLD-V3"})
        sid, token = view["session"]["id"], view["session_token"]
        view = old.mutate(sid, token, "save", {"request_id": "old-v3-check-01", "revision": 0,
            "card_id": view["cards"][0]["id"], "answers": {"novelty": "outside", "grounding": "insufficient"}})
        resumed = self.service.get(sid, token)
        self.assertEqual(resumed["session"], view["session"])
        self.assertEqual(resumed["questions"], view["questions"])
        self.assertEqual(len(resumed["questions"][0]["options"]), 7)
        self.assertEqual(self.service.export(sid, token)["session"]["answers"], view["session"]["answers"])

    def test_default_config_and_saved_null_resume(self):
        config = self.service.config()
        self.assertTrue(config["pack_id"].endswith("-v4"))
        self.assertTrue(all(len(q["options"]) == 6 for q in config["questions"]))
        self.change("save", card_id=self.view["cards"][0]["id"], answers={"novelty": "unable"})
        self.assertEqual(DiscoveryStudy(data_dir=self.service.data_dir).get(self.sid, self.token)["session"], self.view["session"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
