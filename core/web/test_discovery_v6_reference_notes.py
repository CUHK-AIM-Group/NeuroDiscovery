"""v6 per-reference display notes: templated from frozen records, pack-bound."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_reference_notes_v1 import build, PACK, PACK_SHA256, TARGET
from core.web.discovery_study import DiscoveryStudy

PROFILE = {"code": "TEST-V6N", "experience": "3-5", "consent": True}


class V6ReferenceNotesBuildTests(unittest.TestCase):
    def test_deterministic_and_frozen_to_pack_and_sources(self):
        self.assertEqual(build(), build())
        on_disk = json.loads(TARGET.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, build())
        self.assertEqual(on_disk["pack_id"], "cs1-discovery-capabilities-20260918-v6")
        self.assertEqual(on_disk["pack_sha256"], PACK_SHA256)
        with self.assertRaisesRegex(ValueError, "pack changed"):
            build(b"{}")

    def test_every_pack_reference_has_a_complete_bilingual_note(self):
        notes = build()["notes"]
        pack = json.loads(PACK.read_text(encoding="utf-8"))
        total = 0
        for card in pack["cards"]:
            card_notes = notes[card["id"]]
            self.assertEqual(set(card_notes), {ref["id"] for ref in card["pre"]["references"]}, card["id"])
            for note in card_notes.values():
                self.assertEqual(set(note), {"did_zh", "did_en", "relation_zh", "relation_en"})
                self.assertTrue(all(note[key] for key in note))
                self.assertNotIn("这是历史记录", note["did_zh"] + note["relation_zh"])
                total += 1
        self.assertEqual(total, 99)

    def test_definition_glosses_cover_all_cards_in_both_families(self):
        result = build()
        plain = result["definition_plain"]
        pack = json.loads(PACK.read_text(encoding="utf-8"))
        self.assertEqual(set(plain), {card["id"] for card in pack["cards"]})
        roi_count = pair_count = 0
        for card in pack["cards"]:
            gloss = plain[card["id"]]
            self.assertEqual(set(gloss), {"zh", "en"})
            if "ROI" in gloss["zh"] and "分区与" in gloss["zh"]:
                roi_count += 1
                self.assertIn("不代表整个脑区或整个网络", gloss["zh"])
            else:
                pair_count += 1
                self.assertIn("不针对单个分区", gloss["zh"])
        self.assertEqual((roi_count, pair_count), (27, 7))
        self.assertIn("网络内部", plain["packet-05"]["zh"])

    def test_relation_reports_overlap_and_direction_honestly(self):
        notes = build()["notes"]
        pack = json.loads(PACK.read_text(encoding="utf-8"))
        seen_directional = 0
        for card_notes in notes.values():
            for note in card_notes.values():
                if "共同降低”一致" in note["relation_zh"]:
                    seen_directional += 1
                self.assertIn("既有证据之一", note["relation_zh"])
        # 15 panel records carry predicate=reduces; all cards are joint-decrease.
        self.assertEqual(seen_directional, 15)


class V6ReferenceNotesServerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = DiscoveryStudy(PACK, data_dir=temp.name, record_kind="test")

    def test_config_exposes_notes_bound_to_the_pack(self):
        notes = self.service.config().get("reference_notes")
        self.assertIsNotNone(notes)
        self.assertEqual(len(notes), 34)
        self.assertIn("did_zh", notes["packet-01"]["P1"])
        plain = self.service.config().get("definition_plain")
        self.assertIsNotNone(plain)
        self.assertEqual(len(plain), 34)
        self.assertIn("不代表整个脑区", plain["packet-01"]["zh"])

    def test_notes_absent_for_unbound_pack(self):
        import shutil
        import core.web.discovery_study as module
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "pack.json"
            shutil.copy(PACK.with_name("cs1_discovery_pilot_v5.json"), pack_copy)
            shutil.copy(TARGET, Path(temp) / module.REFERENCE_NOTES_NAME)
            service = DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")
            self.assertIsNone(service.reference_notes)
            self.assertNotIn("reference_notes", service.config())

    def test_mismatched_notes_rejected(self):
        import core.web.discovery_study as module
        with tempfile.TemporaryDirectory() as temp:
            pack_copy = Path(temp) / "cs1_discovery_pilot_v6.json"
            pack_copy.write_bytes(PACK.read_bytes())
            broken = json.loads(TARGET.read_text(encoding="utf-8"))
            broken["pack_sha256"] = "0" * 64
            (Path(temp) / module.REFERENCE_NOTES_NAME).write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                DiscoveryStudy(pack_copy, data_dir=temp, record_kind="test")


class V6ReferenceNotesFrontendTests(unittest.TestCase):
    def test_frontend_drops_caveat_paragraphs_and_renders_notes(self):
        source = (Path(__file__).resolve().with_name("static") / "discovery-study.js").read_text(encoding="utf-8")
        render = source[source.index("function referencesHTML("):source.index("function internalHTML(")]
        self.assertNotIn("ref.note", render)
        self.assertNotIn("ref.verification", render)
        self.assertIn("referenceNotes?.[cardId]?.[ref.id]", render)
        self.assertIn('data-user-content', render)
        self.assertIn('display.language', render)
        self.assertIn("function rationaleHTML(", source)
        self.assertIn("function definitionPlainHTML(", source)
        self.assertIn("function feedbackGlossHTML(", source)
        self.assertIn("rationaleHTML(pre, card.id)", source)
        self.assertIn("definitionPlainHTML(card.id)", source)


if __name__ == "__main__":
    unittest.main()
