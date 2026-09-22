"""Visible result tables keep provenance without changing old ratings or materials."""
import copy
import tempfile
import unittest

from core.web.discovery_study import DiscoveryStudy, StudyError, HERE


class VisibleResultTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.service = DiscoveryStudy(HERE / "study_materials/cs1_discovery_pilot_v10.json", data_dir=directory.name, record_kind="test")
        self.created = self.service.create({"code": "TEST-TABLES", "experience": "3-5", "assignment_id": "P01"})

    def save(self, renderer=None):
        view = self.created
        display = {key: view["presentation"][key] for key in ("version", "sha256")}
        display["language"] = "en"
        if renderer is not None:
            display["renderer_revision"] = renderer
        payload = {"request_id": "table-display-save", "revision": 0, "card_id": view["cards"][0]["id"],
                   "answers": {"novelty": "4"}, "note": "", "issues": [], "display": display}
        return self.service.mutate(view["session"]["id"], view["session_token"], "save", payload)

    def test_all_cases_have_internal_and_external_results_from_session_start(self):
        view = self.service.create({"code": "TEST-ALL", "experience": "3-5", "assignment_id": "ALL"})
        self.assertEqual(len(view["cards"]), 34)
        self.assertEqual(view["session"]["stage"], "review")
        for card in view["cards"]:
            post = card["post"]
            if "experimental_results" in post:
                for result in post["experimental_results"]:
                    for phase in ("internal", "external"):
                        self.assertEqual(result[phase]["status"], "complete")
                        self.assertIsNone(result[phase]["failure"])
            else:
                self.assertTrue(post["internal_rows"] and post["external_rows"])

    def test_display_revision_is_recorded_but_cards_and_existing_ratings_are_not_rewritten(self):
        original = copy.deepcopy(self.created["cards"])
        saved = self.save("visible-results-v1")
        self.assertEqual(saved["cards"], original)
        self.assertEqual(saved["session"]["display_history"][-1]["display"]["renderer_revision"], "visible-results-v1")
        exported = self.service.export(saved["session"]["id"], self.created["session_token"])
        self.assertIn("visible-results-v1", str(exported))

    def test_old_client_display_payload_still_saves_without_relabelling(self):
        saved = self.save()
        self.assertNotIn("renderer_revision", saved["session"]["display_history"][-1]["display"])

    def test_analysis_revision_preserves_materials_and_is_exported(self):
        original = copy.deepcopy(self.created["cards"])
        saved = self.save("visible-results-v3")
        self.assertEqual(saved["cards"], original)
        self.assertEqual(saved["session"]["display_history"][-1]["display"]["renderer_revision"], "visible-results-v3")
        self.assertIn("visible-results-v3", str(self.service.export(saved["session"]["id"], self.created["session_token"])))

    def test_unknown_renderer_revision_is_rejected(self):
        with self.assertRaises(StudyError):
            self.save("unverified-renderer")


if __name__ == "__main__":
    unittest.main()
