"""Display translations must not change evidence, answers, blinding or scores."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from core.web.discovery_localization import CATALOG, catalog, presentation, strings_in
from core.web.discovery_study import DEFAULT_PACK, DiscoveryStudy, StudyError, create_preview_app


class DiscoveryLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = DiscoveryStudy(data_dir=self.temp.name, record_kind="test")
        self.view = self.service.create({"code": "TEST-LANG", "experience": "3-5", "consent": True})
        self.sid, self.token = self.view["session"]["id"], self.view["session_token"]

    def save(self, language, **extra):
        return self.service.mutate(self.sid, self.token, "save", {
            "request_id": f"language-{language}-{self.view['session']['revision']:03}",
            "revision": self.view["session"]["revision"], "card_id": self.view["cards"][0]["id"],
            "display": {"language": language, **{key: self.view["presentation"][key] for key in ["version", "sha256"]}},
            **extra,
        })

    def test_translation_source_is_exact_original_v4_not_rewritten(self):
        data, checksum = catalog()
        self.assertEqual(hashlib.sha256(DEFAULT_PACK.read_bytes()).hexdigest(), data["source_sha256"])
        self.assertEqual(hashlib.sha256(CATALOG.read_bytes()).hexdigest(), checksum)
        self.assertEqual(self.view["presentation"]["languages"], ["zh", "en"])

    def test_public_translation_dictionary_covers_all_case_narratives(self):
        dictionary = self.view["presentation"]["strings"]
        for field in ["questions", "common_pre", "common_post", "cards"]:
            for text in strings_in(self.view[field]):
                if any('\u3400' <= c <= '\u9fff' for c in text):
                    self.assertIn(text, dictionary)
                    self.assertFalse(any('\u3400' <= c <= '\u9fff' for c in dictionary[text]))

    def test_config_does_not_leak_cases_or_feedback_through_translations(self):
        public = self.service.config()
        allowed = set(strings_in({k: public[k] for k in ["meta", "questions", "issues"]}))
        self.assertLessEqual(set(public["presentation"]["strings"]), allowed)
        for card in self.view["cards"]:
            action = card["post"].get("feedback", {}).get("action_text")
            if action:
                self.assertNotIn(action, public["presentation"]["strings"])
        self.assertNotIn("organizer", json.dumps(public))

    def test_legacy_stage_a_never_gets_post_translations(self):
        old = DiscoveryStudy(DEFAULT_PACK.with_name("cs1_discovery_pilot_v2.json"), self.temp.name, record_kind="test")
        view = old.create({"code": "TEST-OLD", "experience": "3-5", "consent": True})
        self.assertEqual(view["session"]["stage"], "A")
        for card in old.pack["cards"]:
            self.assertNotIn(card["post"]["feedback"]["action_text"], view["presentation"]["strings"])
        self.assertTrue(all("post" not in c for c in view["cards"]))

    def test_toggle_preserves_answers_notes_order_evidence_and_null(self):
        cards, order = copy.deepcopy(self.view["cards"]), list(self.view["session"]["order"])
        self.view = self.save("zh", answers={"grounding": "4", "novelty": "unable"}, note="备注 / unchanged β -0.25")
        self.view = self.save("en", answers=self.view["session"]["answers"][order[0]], note="备注 / unchanged β -0.25")
        self.view = self.save("zh", answers=self.view["session"]["answers"][order[0]], note="备注 / unchanged β -0.25")
        state = self.view["session"]
        self.assertEqual(state["order"], order)
        self.assertEqual(self.view["cards"], cards)
        self.assertEqual(state["answers"][order[0]], {"grounding": "4", "novelty": "unable"})
        self.assertEqual(state["notes"][order[0]]["review"], "备注 / unchanged β -0.25")
        self.assertEqual([v["display"]["language"] for v in state["display_history"]], ["zh", "en", "zh"])
        self.assertEqual(self.service.export(self.sid, self.token)["session"], state)
        self.assertEqual(self.service.get(self.sid, self.token)["session"], state)

    def test_invalid_display_version_cannot_overwrite_saved_answers(self):
        before = self.service.get(self.sid, self.token)["session"]
        with self.assertRaises(StudyError):
            self.service.mutate(self.sid, self.token, "save", {
                "request_id": "invalid-catalog", "revision": 0, "card_id": before["order"][0],
                "answers": {"novelty": "5"}, "display": {"language": "en", "version": "fake", "sha256": "fake"}})
        self.assertEqual(self.service.get(self.sid, self.token)["session"], before)

    def test_old_answers_load_unchanged_without_inventing_language_history(self):
        original = self.view["session"]
        self.assertNotIn("display_history", original)
        self.assertEqual(self.service.get(self.sid, self.token)["session"], original)
        self.assertEqual(self.service.export(self.sid, self.token)["session"], original)

    def test_readonly_notes_are_not_translation_inputs(self):
        result = presentation({"session": {"notes": {"test": "系统如何利用既有证据来提出这项假设？"}}})
        self.assertEqual(result["strings"], {})

    def test_preview_serves_ui_dictionary_not_private_catalog(self):
        with TestClient(create_preview_app(data_dir=self.temp.name, record_kind="test")) as client:
            self.assertEqual(client.get("/static/discovery-study-i18n.js").status_code, 200)
            self.assertEqual(client.get("/static/cs1_discovery_en_v1.json").status_code, 404)
            page = client.get("/discovery-study").text
            self.assertIn('data-discovery-language="en"', page)
            self.assertIn('data-discovery-language="zh"', page)


if __name__ == "__main__":
    unittest.main()
