"""v5 adds plain-language, dataset-free hypotheses; everything else is preserved."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_discovery_pilot_v5 import build, PARENT, TARGET as V5_PACK, PLAIN_HYPOTHESES
from core.web.discovery_study import DiscoveryStudy, StudyError

PROFILE = {"code": "TEST-V5", "experience": "3-5", "consent": True}
STATIC = Path(__file__).resolve().with_name("static")


class PlainHypothesisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(V5_PACK, data_dir=temp.name, record_kind="test")
        self.view = self.service.create(copy.deepcopy(PROFILE))
        self.sid, self.token = self.view["session"]["id"], self.view["session_token"]

    def test_only_plain_hypothesis_added_and_dataset_removed(self):
        old = json.loads(PARENT.read_text(encoding="utf-8"))
        self.assertEqual(build(), self.service.pack)
        self.assertEqual(self.service.pack["pack_id"], "cs1-discovery-capabilities-20260918-v5")
        for before, after in zip(old["cards"], self.service.pack["cards"]):
            self.assertEqual(before["id"], after["id"])
            self.assertEqual(before["post"], after["post"])
            self.assertEqual(set(after["pre"]) - set(before["pre"]), {"hypothesis_plain"})
            for key in before["pre"]:
                self.assertEqual(after["pre"][key], before["pre"][key], key)
        self.assertEqual(old["questions"], self.service.pack["questions"])
        self.assertEqual(old["scoring"], self.service.pack["scoring"])
        self.assertEqual(old["common_pre"], self.service.pack["common_pre"])
        self.assertEqual(old["common_post"], self.service.pack["common_post"])

    def test_plain_statements_cover_all_cards_and_drop_dataset_framing(self):
        self.assertEqual([c["id"] for c in self.service.pack["cards"]], list(PLAIN_HYPOTHESES))
        for card in self.service.pack["cards"]:
            text = card["pre"]["hypothesis_plain"]
            self.assertTrue(text.endswith("。"))
            for banned in ["TCP", "预定", "筛选", "network_pair", "ROI_to_network", "Pearson"]:
                self.assertNotIn(banned, text, (card["id"], banned))
            self.assertIn("对照组", text)
            self.assertIn("下降方向一致", text)
        # Dataset and analysis details remain documented in the methods block.
        self.assertIn("TCP", json.dumps(self.service.pack["common_pre"], ensure_ascii=False))

    def test_frontend_prefers_plain_and_keeps_original_available(self):
        js = (STATIC / "discovery-study.js").read_text(encoding="utf-8")
        i18n = (STATIC / "discovery-study-i18n.js").read_text(encoding="utf-8")
        css = (STATIC / "discovery-study.css").read_text(encoding="utf-8")
        self.assertIn("function hypothesisHTML(pre,eyebrow,cardId)", js)
        self.assertIn("plain||pre.hypothesis", js)
        self.assertIn("card.pre.hypothesis_plain||card.pre.hypothesis", js)
        self.assertIn("程序原始表述", js)
        self.assertIn('"程序原始表述":"Original program-rendered statement"', i18n)
        self.assertIn(".hypothesis-raw", css)
        # The design section shows only the shared essentials (datasets,
        # measure, model); boundary details and row-level parameters stay in
        # the exported record per the owner request.
        self.assertIn("所有案例共用同一套数据、测量与模型。", js)
        self.assertNotIn("commonOpen", js)

    def test_english_catalog_covers_every_plain_statement(self):
        from core.web.discovery_localization import catalog
        data, _checksum = catalog()
        # The live catalog has moved to the v6 panel; it must still cover v5.
        self.assertIn("cs1_discovery_pilot_v5.json", data.get("covers_packs", [data["source_pack"]]))
        for card in self.service.pack["cards"]:
            text = card["pre"]["hypothesis_plain"]
            self.assertIn(text, data["strings"])
            self.assertFalse(any("㐀" <= c <= "鿿" for c in data["strings"][text]))

    def test_old_v4_sessions_resume_with_their_original_pack(self):
        old = DiscoveryStudy(PARENT, self.service.data_dir, record_kind="test")
        view = old.create({**PROFILE, "code": "TEST-OLD-V4"})
        sid, token = view["session"]["id"], view["session_token"]
        view = old.mutate(sid, token, "save", {"request_id": "old-v4-check-01", "revision": 0,
            "card_id": view["cards"][0]["id"], "answers": {"novelty": "4", "grounding": "unable"}})
        resumed = self.service.get(sid, token)
        self.assertEqual(resumed["session"], view["session"])
        self.assertNotIn("hypothesis_plain", resumed["cards"][0]["pre"])
        self.assertEqual(self.service.export(sid, token)["session"]["answers"], view["session"]["answers"])

    def test_answers_and_scoring_behave_exactly_as_in_v4(self):
        for card in self.view["cards"]:
            self.view = self.service.mutate(self.sid, self.token, "save", {
                "request_id": f"v5-fill-{card['id']}", "revision": self.view["session"]["revision"],
                "card_id": card["id"], "answers": {q["id"]: "4" for q in self.view["questions"]}})
        self.view = self.service.mutate(self.sid, self.token, "save", {
            "request_id": "v5-unable", "revision": self.view["session"]["revision"],
            "card_id": self.view["cards"][0]["id"], "answers": {"novelty": "unable"}})
        self.view = self.service.mutate(self.sid, self.token, "submit", {
            "request_id": "v5-submit", "revision": self.view["session"]["revision"]})
        scores = self.view["score_summary"]
        self.assertEqual(scores["dimensions"]["novelty"]["mean"], 4)
        self.assertEqual(scores["dimensions"]["novelty"]["n"], 9)
        self.assertEqual(scores["composite_mean"], 4)
        with self.assertRaises(StudyError):
            self.service.mutate(self.sid, self.token, "save", {
                "request_id": "v5-bad-option", "revision": self.view["session"]["revision"],
                "card_id": self.view["cards"][0]["id"], "answers": {"novelty": "insufficient"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
