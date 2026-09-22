"""Optional forward links are source-bound and never rewrite historical sessions."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_discovery_pilot_v13 import HERE, PARENT, PARENT_SHA, EVIDENCE, EVIDENCE_SHA, PACK_ID, build
from core.web.discovery_study import DiscoveryStudy

ROOT = HERE.parents[2]


class NextResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = build()
        cls.pack = json.loads(cls.outputs["cs1_discovery_pilot_v13.json"])
        cls.parent = json.loads(PARENT.read_bytes())
        cls.evidence = json.loads(EVIDENCE.read_bytes())

    def test_frozen_sources_and_deterministic_release(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        self.assertEqual(hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(), EVIDENCE_SHA)
        for relative, checksum in self.evidence["sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), checksum, relative)
        self.assertEqual(self.outputs, build())
        for name, raw in self.outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_only_forward_prose_added_no_question_or_result_changes(self):
        for old, new in zip(self.parent["cards"], self.pack["cards"]):
            after = copy.deepcopy(new)
            after["pre"]["reading"].pop("next_research", None)
            self.assertEqual(after, old, old["id"])
        for key in ("scoring", "questions", "issues", "common_pre", "common_post"):
            self.assertEqual(self.parent[key], self.pack[key])
        for key in self.parent["organizer"]:
            self.assertEqual(self.parent["organizer"][key], self.pack["organizer"][key])

    def test_seven_bilingual_notes_and_three_omissions(self):
        present = {c["id"] for c in self.pack["cards"] if c["pre"]["reading"].get("next_research")}
        self.assertEqual(present, {"packet-05", "packet-36", "packet-37", "packet-38", "packet-42", "packet-44", "packet-46"})
        self.assertEqual(set(self.evidence["omitted"]), {"packet-01", "packet-02", "packet-03"})
        for card in self.pack["cards"]:
            if card["id"] in present:
                note = card["pre"]["reading"]["next_research"]
                self.assertEqual(set(note), {"zh", "en"})
                self.assertTrue(all(len(v) > 80 for v in note.values()))
                self.assertNotRegex(note["en"], r"[\u3400-\u9fff]")

    def test_feedback_is_same_case_and_precedes_output_without_external_results(self):
        for case in self.evidence["cases"].values():
            if case["kind"] != "documented_round_input_and_output":
                continue
            request = json.loads((ROOT / case["feedback_pointer"]["path"]).read_bytes())
            index = int(case["feedback_pointer"]["json_pointer"].rsplit("/", 1)[1])
            feedback = request["payload"]["new_run_internal_feedback"]["internal_results"][index]
            self.assertEqual(feedback, case["feedback"])
            self.assertEqual(feedback["hypothesis_id"], case["hypothesis_id"])
            self.assertEqual(feedback["statement"], case["original_statement"])
            self.assertEqual(request["seed"], case["seed"])
            self.assertEqual(request["task"], case["task"])
            self.assertEqual(request["round"], 1)
            self.assertFalse(case["external_results_in_feedback"])
            self.assertFalse(feedback["external_results_present"])
            self.assertEqual(case["original_revision_of"], [])
            self.assertFalse(case["individual_parent_relationship_recorded"])
            for child in case["subsequent"]:
                self.assertFalse(child["selected"])
                self.assertEqual(child["experiment_configurations"], 0)
                self.assertEqual(child["revision_of"], [])

    def test_explicit_child_preserves_its_actual_internal_measurement(self):
        case = self.evidence["cases"]["packet-05"]
        self.assertEqual(case["kind"], "explicit_parent_and_completed_child")
        self.assertTrue(case["lineage"]["same_feedback_verified"])
        child = case["child_occurrence"]["record"]
        self.assertEqual(child["feedback_parent"], case["hypothesis_id"])
        self.assertEqual(child["round"], 1)
        self.assertEqual(child["seed"], 0)
        self.assertEqual(child["feedback"]["joint_direction_stability"], 0.982421875)
        self.assertEqual(len(child["feedback"]["components"]), 3)
        self.assertTrue(all(v["standardized_beta"] < 0 for v in child["feedback"]["components"].values()))

    def test_assignments_annotations_and_existing_translations_are_preserved(self):
        for kind, before, after in (("assignments", 8, 9), ("reference_notes", 7, 8), ("significance", 7, 8)):
            old = json.loads((HERE / f"cs1_discovery_{kind}_v{before}.json").read_bytes())
            new = json.loads(self.outputs[f"cs1_discovery_{kind}_v{after}.json"])
            for value in (old, new):
                del value["pack_id"], value["pack_sha256"]
            self.assertEqual(old, new)
        old = json.loads((HERE / "cs1_discovery_en_v11.json").read_bytes())["strings"]
        new = json.loads(self.outputs["cs1_discovery_en_v12.json"])["strings"]
        self.assertTrue(all(new[k] == v for k, v in old.items()))

    def test_legacy_answers_and_material_unchanged_new_exports_include_forward_context(self):
        with tempfile.TemporaryDirectory() as directory:
            old = DiscoveryStudy(PARENT, data_dir=directory, record_kind="test")
            view = old.create({"code": "TEST-NEXT-OLD", "experience": "3-5", "assignment_id": "P01"})
            saved = old.mutate(view["session"]["id"], view["session_token"], "save", {
                "request_id": "before-next", "revision": 0, "card_id": view["cards"][0]["id"],
                "answers": {"novelty": "4"}, "note": "retain original", "issues": []})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v13.json", data_dir=directory, record_kind="test")
            restored = new.get(view["session"]["id"], view["session_token"])
            self.assertEqual(saved["session"], restored["session"])
            self.assertEqual(saved["cards"], restored["cards"])
            self.assertEqual(new.config()["pack_id"], PACK_ID)
            self.assertEqual(new.config()["meta"]["next_research_revision"], "source-linked-next-research-v1")
            fresh = new.create({"code": "TEST-NEXT-NEW", "experience": "3-5", "assignment_id": "P01"})
            self.assertEqual(sum(bool(c["pre"]["reading"].get("next_research")) for c in fresh["cards"]), 7)
            self.assertNotIn("next_research_context", fresh)
            for card in fresh["cards"]:
                self.assertEqual(card, next(c for c in self.pack["cards"] if c["id"] == card["id"]))


if __name__ == "__main__":
    unittest.main()
