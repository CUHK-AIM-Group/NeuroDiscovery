"""Single-round protocol tests using isolated synthetic respondents only."""
import copy
import hashlib
import json
import tempfile
import unittest

from fastapi.testclient import TestClient
from core.web.build_discovery_pilot_v3 import build, PARENT, PARENT_HASH
from core.web.discovery_study import DiscoveryStudy, StudyError, create_preview_app
from core.web.discovery_scoring import aggregate_exports

PROFILE = {"code": "TEST-V3", "experience": "3-5", "consent": True}
DEFAULT_PACK = PARENT.with_name("cs1_discovery_pilot_v3.json")


class SingleRoundTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(DEFAULT_PACK, data_dir=temp.name, record_kind="test")
        self.view = self.service.create(copy.deepcopy(PROFILE))
        self.sid = self.view["session"]["id"]
        self.token = self.view["session_token"]
        self.counter = 0

    def change(self, kind, **payload):
        self.counter += 1
        self.view = self.service.mutate(self.sid, self.token, kind, {
            "revision": self.view["session"]["revision"], "request_id": f"single-test-{self.counter:03}", **payload})
        return self.view

    def fill(self, cards=None, value="3"):
        for card in cards if cards is not None else self.view["cards"]:
            self.change("save", card_id=card["id"], answers={q["id"]: value for q in self.view["questions"]}, note="合成测试，不是专家意见。")

    def test_new_snapshot_changes_protocol_not_evidence_or_scales(self):
        parent = json.loads(PARENT.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_HASH)
        self.assertEqual(build(), self.service.pack)
        self.assertEqual(parent["cards"], self.service.pack["cards"])
        self.assertEqual(parent["organizer"]["mapping"], self.service.pack["organizer"]["mapping"])
        for old, new in zip(parent["questions"], self.service.pack["questions"]):
            self.assertEqual({**old, "stage": "review"}, new)
        self.assertNotEqual(parent["protocol_version"], self.service.pack["protocol_version"])
        self.assertNotEqual(parent["scoring"]["version"], self.service.pack["scoring"]["version"])

    def test_complete_material_and_six_questions_available_from_start(self):
        self.assertEqual(self.view["session"]["stage"], "review")
        self.assertEqual(self.view["session"]["review_flow"], "single_round")
        self.assertEqual(len(self.view["cards"]), 10)
        self.assertEqual(len(self.view["questions"]), 6)
        self.assertEqual({q["stage"] for q in self.view["questions"]}, {"review"})
        self.assertTrue(all("post" in card and "pre" in card for card in self.view["cards"]))
        self.assertIn("common_post", self.view)
        self.assertNotIn("stage_a_locked_at", self.view["session"])
        self.assertNotIn("score_summary", self.view)
        self.assertTrue(all(not a for a in self.view["session"]["answers"].values()))
        for card in self.view["cards"]:
            self.assertEqual(self.view["session"]["notes"][card["id"]], {"review": ""})

    def test_no_reveal_or_partial_submission(self):
        with self.assertRaises(StudyError): self.change("reveal")
        self.fill(self.view["cards"][:1])
        self.assertEqual(self.view["session"]["stage"], "review")
        with self.assertRaises(StudyError): self.change("submit")

    def test_one_round_complete_scores_and_lock(self):
        self.fill()
        self.change("submit")
        result = self.service.export(self.sid, self.token)
        self.assertEqual(result["session"]["stage"], "complete")
        self.assertEqual(result["score_summary"]["complete_case_count"], 10)
        self.assertEqual(result["score_summary"]["composite_mean"], 3)
        self.assertNotIn("reveal", [e["kind"] for e in result["events"]])
        self.assertNotIn(self.token, json.dumps(result))
        with self.assertRaises(StudyError): self.fill(self.view["cards"][:1])

    def test_answers_editable_until_final_submission_and_resume(self):
        self.fill(self.view["cards"][:1])
        cid = self.view["cards"][0]["id"]
        self.change("save", card_id=cid, answers={"grounding": "4", "value": "outside"}, active_seconds=12)
        resumed = DiscoveryStudy(data_dir=self.service.data_dir).get(self.sid, self.token)
        self.assertEqual(resumed["session"], self.view["session"])
        self.assertEqual(resumed["session"]["answers"][cid]["value"], "outside")
        self.assertEqual(resumed["session"]["active_seconds_client_reported"], {"review": 12})

    def test_old_two_stage_session_still_hides_results_and_locks_initial_ratings(self):
        old = DiscoveryStudy(PARENT, self.service.data_dir, record_kind="test")
        prior = old.create({**PROFILE, "code": "OLD-V2"})
        sid, token = prior["session"]["id"], prior["session_token"]
        resumed = self.service.get(sid, token)
        self.assertEqual(resumed["session"], prior["session"])
        self.assertTrue(all("post" not in c for c in resumed["cards"]))
        for index, card in enumerate(resumed["cards"]):
            resumed = self.service.mutate(sid, token, "save", {"request_id": f"old-save-{index:03}",
                "revision": resumed["session"]["revision"], "card_id": card["id"],
                "answers": {q["id"]: "3" for q in resumed["questions"] if q["stage"] == "A"}})
        resumed = self.service.mutate(sid, token, "reveal", {"request_id": "old-reveal-001", "revision": resumed["session"]["revision"]})
        self.assertEqual(resumed["session"]["stage"], "B")
        with self.assertRaises(StudyError):
            self.service.mutate(sid, token, "save", {"request_id": "old-invalid-001", "revision": resumed["session"]["revision"],
                "card_id": resumed["cards"][0]["id"], "answers": {"grounding": "5"}})

    def test_v2_and_v3_cannot_be_pooled_despite_same_scoring_options(self):
        self.fill(); self.change("submit")
        new = self.service.export(self.sid, self.token)
        new["session"]["record_kind"] = "pilot"
        old = copy.deepcopy(new)
        parent = json.loads(PARENT.read_text(encoding="utf-8"))
        old["session"].update({"id": "SYNTH-OLD", "pack_id": parent["pack_id"], "protocol_version": parent["protocol_version"]})
        old["session"]["profile"]["code"] = "OTHER"
        old["scoring"] = parent["scoring"]
        with self.assertRaises(ValueError): aggregate_exports([old, new])

    def test_null_answers_not_replaced_and_tests_excluded(self):
        self.fill(value="insufficient"); self.change("submit")
        exported = self.service.export(self.sid, self.token)
        self.assertIsNone(exported["score_summary"]["composite_mean"])
        self.assertEqual(aggregate_exports([exported])["excluded"], {"test": 1})

    def test_api_single_round_and_private_source_protection(self):
        with TestClient(create_preview_app(DEFAULT_PACK, self.service.data_dir, "test")) as client:
            html = client.get("/discovery-study").text
            self.assertIn("一轮完成分配的案例", html)
            self.assertNotIn("进入阶段 A", html)
            self.assertNotIn('name="specialty"', html)
            result = client.post("/api/studies/discovery/sessions", json=PROFILE).json()
            self.assertEqual(result["session"]["stage"], "review")
            for field in ["organizer", "source_evidence", "single_round_amendment", "original_material_id"]:
                self.assertNotIn('"' + field + '"', json.dumps(result))
            self.assertEqual(client.get("/static/cs1_discovery_pilot_v3.json").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
