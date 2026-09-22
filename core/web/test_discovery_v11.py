"""Success-led selection preserves complete results and historical sessions."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

from core.web.build_discovery_pilot_v11 import (
    HERE, PARENT, PARENT_SHA, PACK_ID, ALLOCATION_REVISION, CASE1,
    OTHER_SUPPORTED, PARTIAL, SELECTED, build, matching_supported_configs,
)
from core.web.discovery_study import DiscoveryStudy


class SelectionTests(unittest.TestCase):
    def test_reproducible_outputs_and_frozen_parent(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        outputs = build()
        self.assertEqual(outputs, build())
        for name, raw in outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_exact_selection_preserves_all_case_text_and_results(self):
        old = json.loads(PARENT.read_bytes())
        new = json.loads(build()["cs1_discovery_pilot_v11.json"])
        originals = {c["id"]: c for c in old["cards"]}
        self.assertEqual([c["id"] for c in new["cards"]], list(SELECTED))
        for card in new["cards"]:
            self.assertEqual(card, originals[card["id"]])
        for field in ("questions", "issues", "common_pre", "common_post"):
            self.assertEqual(old[field], new[field])
        self.assertEqual(Counter(c["pre"]["topic"] for c in new["cards"]),
                         {"跨诊断脑连接": 4, "影像遗传学": 3, "预后研究": 3})

    def test_seven_supported_not_all_models_or_all_ten(self):
        pack = json.loads(build()["cs1_discovery_pilot_v11.json"])
        rows = pack["organizer"]["selection_evidence"]
        self.assertEqual(Counter(r["role"] for r in rows.values()),
                         {"main_supported": 7, "supplementary_partial": 3})
        for cid in CASE1:
            self.assertTrue(rows[cid]["original_external_triage_pass"])
            self.assertFalse(rows[cid]["formal_confirmation"])
        cards = {c["id"]: c for c in pack["cards"]}
        for cid in OTHER_SUPPORTED:
            self.assertTrue(matching_supported_configs(cards[cid]))
        for cid in PARTIAL:
            self.assertFalse(matching_supported_configs(cards[cid]))
        # Do not silently hide the rank-association / shorter-window results.
        for cid in OTHER_SUPPORTED + PARTIAL:
            self.assertEqual(len(cards[cid]["post"]["experimental_results"]), 3)

    def test_every_expert_gets_same_ten_with_forty_percent_connectivity(self):
        table = json.loads(build()["cs1_discovery_assignments_v7.json"])
        self.assertEqual(table["allocation_revision"], ALLOCATION_REVISION)
        for dealt in table["experts"].values():
            self.assertEqual(dealt, list(SELECTED))
            self.assertEqual(len(set(dealt) & set(CASE1)), 4)
            self.assertEqual(len(set(dealt) & set(CASE1 + OTHER_SUPPORTED)), 7)
        self.assertEqual(set(table["coverage"].values()), {10})

    def test_old_translation_strings_are_all_retained(self):
        old = json.loads((HERE / "cs1_discovery_en_v8.json").read_bytes())["strings"]
        new = json.loads(build()["cs1_discovery_en_v9.json"])["strings"]
        self.assertTrue(all(new[k] == v for k, v in old.items()))


class SessionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def test_default_config_new_session_and_export(self):
        service = DiscoveryStudy(HERE / "cs1_discovery_pilot_v11.json", data_dir=self.folder, record_kind="test")
        config = service.config()
        self.assertEqual(config["pack_id"], PACK_ID)
        self.assertEqual(config["meta"]["card_count"], 10)
        self.assertEqual(config["assignments"]["allocation_revision"], ALLOCATION_REVISION)
        self.assertNotIn("selection_evidence", str(config))
        with patch("core.web.discovery_study.secrets.SystemRandom") as rng:
            rng.return_value.shuffle.side_effect = lambda x: x.reverse()
            view = service.create({"code": "TEST-FORTY", "experience": "3-5", "assignment_id": "P01"})
            rng.return_value.shuffle.assert_called_once()
        self.assertEqual(view["session"]["order"], list(reversed(SELECTED)))
        self.assertEqual(view["session"]["allocation_revision"], ALLOCATION_REVISION)
        exported = service.export(view["session"]["id"], view["session_token"])
        self.assertEqual({c["id"] for c in exported["cards"]}, set(SELECTED))

    def test_saved_v10_session_keeps_old_materials_answers_and_allocation(self):
        old = DiscoveryStudy(PARENT, data_dir=self.folder, record_kind="test")
        view = old.create({"code": "TEST-OLD-THIRTY", "experience": "3-5", "assignment_id": "P01"})
        saved = old.mutate(view["session"]["id"], view["session_token"], "save", {
            "request_id": "save-before-upgrade", "revision": 0,
            "card_id": view["cards"][0]["id"], "answers": {"novelty": "4"}, "note": "preserve", "issues": []})
        old_export = copy.deepcopy(old.export(view["session"]["id"], view["session_token"]))
        new = DiscoveryStudy(data_dir=self.folder, record_kind="test")
        restored = new.get(view["session"]["id"], view["session_token"])
        self.assertEqual(restored["session"], saved["session"])
        self.assertEqual(restored["cards"], saved["cards"])
        self.assertEqual(restored["session"]["allocation_revision"], "he1-topic30-v6")
        self.assertEqual(new.export(view["session"]["id"], view["session_token"])["cards"], old_export["cards"])


if __name__ == "__main__":
    unittest.main()
