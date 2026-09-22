"""v6 panel: 34 externally validated, strongest-first materials; v5 stays intact."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_discovery_pilot_v6 import (
    build, PARENT as V5_PACK, TARGET as V6_PACK, SELECTION_HASH, SOURCE_HASHES, REUSE_FROM_V5,
)
from core.web.discovery_study import DiscoveryStudy

PROFILE = {"code": "TEST-V6", "experience": "3-5", "consent": True}
STATIC = Path(__file__).resolve().with_name("static")
SELECTION = json.loads((V6_PACK.with_name("sources_v6") / "selection_v6.json").read_text(encoding="utf-8"))
CJK = lambda text: any("㐀" <= c <= "鿿" for c in text)


class V6PanelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(V6_PACK, data_dir=temp.name, record_kind="test")
        self.view = self.service.create(copy.deepcopy(PROFILE))

    def test_selection_file_is_frozen_and_effect_ordered(self):
        selected = SELECTION["selected"]
        self.assertEqual(len(selected), 34)
        self.assertEqual(SELECTION["pool_size"], 38)
        self.assertEqual(len(SELECTION["dropped"]), 4)
        effects = [e["minimum_directional_standardized_beta"] for e in selected]
        joints = [e["joint_direction_frequency"] for e in selected]
        self.assertTrue(all(a > b or (a == b and x >= y)
                            for (a, x), (b, y) in zip(zip(effects, joints), zip(effects[1:], joints[1:]))))
        self.assertGreater(min(effects), max(e["minimum_directional_standardized_beta"] for e in SELECTION["dropped"]))
        self.assertGreaterEqual(min(effects), 0.15)
        self.assertGreaterEqual(min(joints), 0.8)

    def test_pack_has_34_ranked_cards_matching_selection(self):
        pack = self.service.pack
        self.assertEqual(pack["pack_id"], "cs1-discovery-capabilities-20260918-v6")
        self.assertEqual(build(), pack)
        self.assertEqual(len(pack["cards"]), 34)
        self.assertEqual(pack["public_meta"]["card_count"], 34)
        mapping = pack["organizer"]["mapping"]
        for rank, (card, private, entry) in enumerate(zip(pack["cards"], mapping, SELECTION["selected"]), 1):
            self.assertEqual(card["id"], f"packet-{rank:02}")
            self.assertEqual(private["hypothesis_id"], entry["hypothesis_id"])
            self.assertEqual(card["post"]["ucla_joint"], entry["joint_direction_frequency"])
        self.assertEqual(pack["organizer"]["v6_amendment"]["selection_v6_sha256"], SELECTION_HASH)
        self.assertEqual(pack["organizer"]["v6_amendment"]["source_hashes"], SOURCE_HASHES)

    def test_every_card_passed_external_triage_with_full_materials(self):
        for card in self.service.pack["cards"]:
            self.assertGreaterEqual(len(card["post"]["external_rows"]), 8)
            for row in card["post"]["external_rows"]:
                boot = row["bootstrap_standardized_beta"]
                self.assertIsInstance(boot["bootstrap_direction_fraction"], (int, float))
                self.assertEqual(len(boot["bootstrap_percentiles_2_5_97_5_descriptive"]), 2)
            for key in ["title", "hypothesis", "hypothesis_plain", "definition", "rationale",
                        "transfer_limit", "references", "source_note"]:
                self.assertIn(key, card["pre"], (card["id"], key))
            for banned in ["TCP", "预定", "筛选", "Pearson", "network_pair", "ROI_to_network"]:
                self.assertNotIn(banned, card["pre"]["hypothesis_plain"], (card["id"], banned))

    def test_three_retained_cards_are_byte_identical_to_v5(self):
        v5 = json.loads(V5_PACK.read_text(encoding="utf-8"))
        v5_cards = {card["id"]: card for card in v5["cards"]}
        for hypothesis_id, old_id in REUSE_FROM_V5.items():
            card = next(c for c in self.service.pack["cards"]
                        if self.service.pack["organizer"]["mapping"][self.service.pack["cards"].index(c)]["hypothesis_id"] == hypothesis_id)
            before = json.dumps({k: v for k, v in v5_cards[old_id].items() if k != "id"},
                                ensure_ascii=False, sort_keys=True)
            after = json.dumps({k: v for k, v in card.items() if k != "id"},
                               ensure_ascii=False, sort_keys=True)
            self.assertEqual(before, after, hypothesis_id)
        retired = [c["id"] for c in v5["cards"] if c["id"] not in REUSE_FROM_V5.values()]
        self.assertEqual(len(retired), 7)
        v6_ids = {m["hypothesis_id"] for m in self.service.pack["organizer"]["mapping"]}
        v5_mapping = {m["id"]: m["hypothesis_id"] for m in v5["organizer"]["mapping"]}
        self.assertTrue(all(v5_mapping[old_id] not in v6_ids for old_id in retired))

    def test_first_round_cards_carry_explicit_no_parent_feedback(self):
        first_round = [c for c in self.service.pack["cards"] if c["post"]["feedback"].get("no_parent")]
        self.assertEqual(len(first_round), 4)
        for card in first_round:
            feedback = card["post"]["feedback"]
            self.assertIsNone(feedback["parent_endpoint"])
            self.assertFalse(feedback["binding_verified"])
            self.assertIn("首轮", card["post"]["audit_note"])
        with_parents = [c for c in self.service.pack["cards"] if not c["post"]["feedback"].get("no_parent")]
        self.assertEqual(len(with_parents), 30)
        for card in with_parents:
            self.assertTrue(card["post"]["feedback"]["parent_endpoint"])
            self.assertTrue(card["post"]["feedback"]["binding_verified"])

    def test_frontend_renders_first_round_branch(self):
        js = (STATIC / "discovery-study.js").read_text(encoding="utf-8")
        i18n = (STATIC / "discovery-study-i18n.js").read_text(encoding="utf-8")
        self.assertIn("fb.no_parent||!fb.parent_endpoint", js)
        self.assertIn("首轮提议：生成该假设时同 seed 尚无已完成的 TCP 反馈，无父假设绑定。", js)
        self.assertIn('"首轮提议说明：":"First-round proposal note:"', i18n)

    def test_shared_blocks_and_meta_disclose_the_new_rule(self):
        pack = self.service.pack
        self.assertIn("38 条", pack["common_post"][0]["text"])
        self.assertIn("前 34 条", pack["common_post"][0]["text"])
        self.assertIn("不能用来估计自然发现成功率", pack["public_meta"]["selection_note"])
        self.assertEqual(pack["public_meta"]["version_label"], "v6 · 2026-09-18 · 单轮")
        # Everything else in the shared blocks is unchanged from v5.
        v5 = json.loads(V5_PACK.read_text(encoding="utf-8"))
        self.assertEqual(pack["common_pre"], v5["common_pre"])
        self.assertEqual(pack["common_post"][1:], v5["common_post"][1:])
        self.assertEqual(pack["questions"], v5["questions"])
        self.assertEqual(pack["scoring"], v5["scoring"])

    def test_english_catalog_covers_every_v6_string(self):
        from core.web.discovery_localization import catalog
        data, _checksum = catalog()
        self.assertIn("cs1_discovery_pilot_v6.json", data.get("covers_packs", [data["source_pack"]]))
        projection = {"meta": self.service.pack["public_meta"], "questions": self.service.pack["questions"],
                      "common_pre": self.service.pack["common_pre"], "common_post": self.service.pack["common_post"],
                      "cards": self.view["cards"]}
        from core.web.discovery_localization import strings_in
        for text in strings_in(projection):
            if CJK(text):
                self.assertIn(text, data["strings"])
                self.assertFalse(CJK(data["strings"][text]), text[:40])

    def test_old_v5_sessions_resume_with_their_original_pack(self):
        old = DiscoveryStudy(V5_PACK, self.service.data_dir, record_kind="test")
        view = old.create({"code": "TEST-OLD-V5", "experience": "3-5", "consent": True})
        sid, token = view["session"]["id"], view["session_token"]
        view = old.mutate(sid, token, "save", {"request_id": "old-v5-check-01", "revision": 0,
            "card_id": view["cards"][0]["id"], "answers": {"novelty": "4", "grounding": "unable"}})
        resumed = self.service.get(sid, token)
        self.assertEqual(resumed["session"], view["session"])
        self.assertEqual(len(resumed["cards"]), 10)
        self.assertEqual(self.service.export(sid, token)["session"]["answers"], view["session"]["answers"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
