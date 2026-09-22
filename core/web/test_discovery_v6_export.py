"""v6 export alignment with the Expert Study (Extension) result envelope."""
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_discovery_assignments_v1 import build as build_assignments
from core.web.discovery_study import DiscoveryStudy, export_filename, REVIEW_DISPLAY_ORDER, ALIGNED_EXPORT_VERSION

PACK = Path(__file__).resolve().parent / "study_materials" / "cs1_discovery_pilot_v6.json"
LEGACY_PACK = Path(__file__).resolve().parent / "study_materials" / "cs1_discovery_pilot_v1.json"
PROFILE = {"code": "TEST-EXPORT", "experience": "3-5", "consent": True}


class V6AlignedExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.service = DiscoveryStudy(PACK, data_dir=temp.name, record_kind="test")
        self.view = self.service.create({**PROFILE, "assignment_id": "P04"})

    def fill_and_submit(self):
        data = self.view
        questions = [q for q in data["questions"] if q["stage"] == "review"]
        for index, card in enumerate(data["cards"]):
            answers = {}
            for q in questions:
                answers[q["id"]] = "unable" if index == 0 and q["id"] == "novelty" else str(1 + (index % 5))
            data = self.service.mutate(data["session"]["id"], self.view["session_token"], "save", {
                "request_id": f"test-save-{index:02d}", "revision": data["session"]["revision"],
                "card_id": card["id"], "answers": answers, "note": "边界备注" if index == 1 else "",
                "issues": [], "active_seconds": 30})
        return self.service.mutate(data["session"]["id"], self.view["session_token"], "submit", {
            "request_id": "test-submit-01", "revision": data["session"]["revision"]})

    def test_envelope_matches_the_extension_shape(self):
        self.fill_and_submit()
        result = self.service.export(self.view["session"]["id"], self.view["session_token"])
        self.assertEqual(result["schema_version"], ALIGNED_EXPORT_VERSION)
        self.assertEqual(result["schema_version"], "neurodiscovery-expert-study-results-v2")
        self.assertEqual(result["export_schema"], 2)
        self.assertEqual(result["study"]["study_id"], "cs1-discovery-capabilities-20260918-v6")
        self.assertEqual(result["participant"], {"id": "TEST-EXPORT", "experience_years": "3-5", "consent": True})
        self.assertEqual(result["session"]["assignment_id"], "P04")
        self.assertIn("active_answering_time_ms", result["timing"])
        summary = result["summary"]
        self.assertEqual(summary["total_questions"], 60)
        self.assertEqual(summary["answered_questions"], 60)
        self.assertEqual(summary["cards_fully_answered"], 10)
        self.assertEqual(summary["cannot_judge"], 1)
        self.assertEqual(summary["cannot_judge_reasons"], {"unable": 1})
        self.assertEqual(summary["notes_present"], 1)
        rows = result["question_results"]
        self.assertEqual(len(rows), 60)
        self.assertEqual([r["question_number"] for r in rows], list(range(1, 61)))
        first_card = rows[0]["card_id"]
        self.assertEqual([r["question_id"] for r in rows[:6]], REVIEW_DISPLAY_ORDER)
        self.assertTrue(all(r["card_id"] == first_card for r in rows[:6]))
        unable = [r for r in rows if r["cannot_judge"]]
        self.assertEqual(len(unable), 1)
        self.assertEqual(unable[0]["answer_label"], "无法判断（不计分）")
        # The full audit payload is preserved alongside the aligned envelope.
        self.assertIn("session", result)
        self.assertIn("answers", result["session"])
        self.assertIn("cards", result)
        self.assertIn("events", result)
        self.assertIn("score_summary", result)

    def test_export_filename_matches_the_extension_convention(self):
        self.fill_and_submit()
        result = self.service.export(self.view["session"]["id"], self.view["session_token"])
        name = export_filename(result)
        self.assertRegex(name, r"^NeuroDiscovery-user-study-TEST-EXPORT-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.*\.json$")
        saved = self.service.save_export(self.view["session"]["id"], self.view["session_token"])
        self.assertTrue(Path(saved["path"]).name.startswith("NeuroDiscovery-user-study-TEST-EXPORT-"))
        written = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
        self.assertEqual(written["schema_version"], ALIGNED_EXPORT_VERSION)

    def test_assigned_export_covers_only_the_share(self):
        self.fill_and_submit()
        result = self.service.export(self.view["session"]["id"], self.view["session_token"])
        share = set(build_assignments()["experts"]["P04"])
        self.assertEqual({r["card_id"] for r in result["question_results"]}, share)
        self.assertEqual({c["id"] for c in result["cards"]}, share)

    def test_legacy_pack_export_keeps_working(self):
        legacy = DiscoveryStudy(LEGACY_PACK, data_dir=self.root / "legacy", record_kind="test")
        created = legacy.create({"code": "TEST-OLD", "experience": "3-5", "consent": True,
                                 "specialty": "其他交叉学科", "old_study": "no", "case_exposure": "no", "mode": "test"})
        token = created["session_token"]
        view = created
        questions_a = [q for q in legacy.pack["questions"] if q["stage"] == "A"]
        for index, card in enumerate(view["cards"]):
            view = legacy.mutate(view["session"]["id"], token, "save", {
                "request_id": f"legacy-a-{index}", "revision": view["session"]["revision"],
                "card_id": card["id"],
                "answers": {q["id"]: q["options"][0]["value"] for q in questions_a}, "note": "", "issues": [],
                "active_seconds": 5})
        view = legacy.mutate(view["session"]["id"], token, "reveal",
                             {"request_id": "legacy-reveal", "revision": view["session"]["revision"]})
        result = legacy.export(view["session"]["id"], token)
        self.assertEqual(result["schema_version"], ALIGNED_EXPORT_VERSION)
        self.assertEqual(result["export_schema"], 1)
        self.assertTrue(result["question_results"])
        self.assertEqual(result["question_results"][0]["question_number"], 1)

    def test_client_download_uses_the_aligned_filename(self):
        source = (Path(__file__).resolve().with_name("static") / "discovery-study.js").read_text(encoding="utf-8")
        self.assertIn("EvaluationExport.exportResults", source)
        shared = (Path(__file__).resolve().with_name("static") / "evaluation-export.js").read_text(encoding="utf-8")
        self.assertIn("NeuroDiscovery-Human-Evaluations-", shared)
        self.assertNotIn("data.session.pack_id}_${", source)


if __name__ == "__main__":
    unittest.main()
