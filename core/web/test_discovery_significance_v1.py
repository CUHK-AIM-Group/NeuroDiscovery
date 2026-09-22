"""v6 per-card significance sentences: templated from frozen records, pack-bound."""
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_significance_v1 import build, PACK, PACK_SHA256, TARGET
from core.web.discovery_study import DiscoveryStudy

PROFILE = {"code": "TEST-V6S", "experience": "3-5", "consent": True}


class V6SignificanceBuildTests(unittest.TestCase):
    def test_deterministic_and_frozen_to_pack_and_selection(self):
        self.assertEqual(build(), build())
        on_disk = json.loads(TARGET.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, build())
        self.assertEqual(on_disk["pack_id"], "cs1-discovery-capabilities-20260918-v6")
        self.assertEqual(on_disk["pack_sha256"], PACK_SHA256)
        with self.assertRaisesRegex(ValueError, "pack changed"):
            build(b"{}")

    def test_every_card_has_a_complete_bilingual_sentence(self):
        significance = build()["significance"]
        pack = json.loads(PACK.read_text(encoding="utf-8"))
        self.assertEqual(set(significance), {card["id"] for card in pack["cards"]})
        for card in pack["cards"]:
            entry = significance[card["id"]]
            self.assertEqual(set(entry), {"zh", "en"})
            self.assertIn(card["pre"]["title"], entry["zh"])
            self.assertTrue(entry["zh"].startswith("如果成真"), card["id"])
            self.assertIn("脑研究", entry["zh"])
            self.assertIn("临床", entry["zh"])
            self.assertTrue(entry["en"].startswith("If confirmed"), card["id"])
            self.assertIn("brain research", entry["en"])

    def test_group_wording_follows_the_frozen_selection(self):
        significance = build()["significance"]
        self.assertIn("ADHD 与精神分裂症", significance["packet-13"]["zh"])
        self.assertIn("双相障碍与精神分裂症", significance["packet-01"]["zh"])
        self.assertIn("people with ADHD", significance["packet-13"]["en"])


class V6SignificanceServerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(PACK, data_dir=temp.name, record_kind="test")

    def test_config_exposes_significance_bound_to_the_pack(self):
        significance = self.service.config().get("significance")
        self.assertIsNotNone(significance)
        self.assertEqual(len(significance), 34)
        self.assertIn("如果成真", significance["packet-01"]["zh"])
        self.assertIn("If confirmed", significance["packet-01"]["en"])

    def test_significance_absent_for_unbound_pack(self):
        import shutil
        import core.web.discovery_study as module
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "pack.json"
            shutil.copy(PACK.with_name("cs1_discovery_pilot_v5.json"), pack_copy)
            shutil.copy(TARGET, Path(temp) / module.SIGNIFICANCE_NAME)
            service = DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")
            self.assertIsNone(service.significance)
            self.assertNotIn("significance", service.config())

    def test_mismatched_significance_rejected(self):
        import core.web.discovery_study as module
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "cs1_discovery_pilot_v6.json"
            pack_copy.write_bytes(PACK.read_bytes())
            broken = json.loads(TARGET.read_text(encoding="utf-8"))
            broken["pack_sha256"] = "0" * 64
            (Path(temp) / module.SIGNIFICANCE_NAME).write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")


class V6SignificanceFrontendTests(unittest.TestCase):
    def test_hypothesis_block_renders_the_sentence(self):
        source = (Path(__file__).resolve().with_name("static") / "discovery-study.js").read_text(encoding="utf-8")
        self.assertIn("function significanceHTML(cardId)", source)
        self.assertIn("hypothesisHTML(pre,\"系统提出的假设\",card.id)", source)
        self.assertIn('class="sig"', source)
        self.assertIn('data-user-content', source)
        self.assertIn('version_label', source)  # guards older resumed sessions

    def test_significance_styled_in_both_css_layers(self):
        base = (Path(__file__).resolve().with_name("static") / "discovery-study.css").read_text(encoding="utf-8")
        embedded = (Path(__file__).resolve().with_name("static") / "study-workspace.css").read_text(encoding="utf-8")
        self.assertIn(".hypothesis .sig", base)
        self.assertIn(".hypothesis .sig", embedded)


if __name__ == "__main__":
    unittest.main()
