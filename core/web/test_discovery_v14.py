"""Concrete hypothesis lineage changes presentation, not results or historical answers."""
import copy
import hashlib
import json
import tempfile
import unittest

from core.web.build_discovery_pilot_v14 import HERE, PARENT, PARENT_SHA, EVIDENCE, EVIDENCE_SHA, PACK_ID, LINEAGE_REVISION, build, validate_chain
from core.web.discovery_study import DiscoveryStudy

ROOT = HERE.parents[2]


class ConcreteLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = build()
        cls.pack = json.loads(cls.outputs["cs1_discovery_pilot_v14.json"])
        cls.parent = json.loads(PARENT.read_bytes())
        cls.evidence = json.loads(EVIDENCE.read_bytes())
        cls.cards = {c["id"]: c for c in cls.pack["cards"]}

    def test_frozen_sources_and_repeatable_build(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        self.assertEqual(hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(), EVIDENCE_SHA)
        for relative, checksum in self.evidence["sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), checksum, relative)
        self.assertEqual(self.outputs, build())
        for name, raw in self.outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_only_structured_wording_changes(self):
        for old, new in zip(self.parent["cards"], self.pack["cards"]):
            restored = copy.deepcopy(new)
            restored["pre"]["reading"].pop("feedback_chain", None)
            restored["pre"]["reading"].pop("next_research_chain", None)
            self.assertEqual(restored, old, old["id"])
        for key in ("scoring", "questions", "issues", "common_pre", "common_post"):
            self.assertEqual(self.parent[key], self.pack[key])
        for key, value in self.parent["organizer"].items():
            self.assertEqual(value, self.pack["organizer"][key])

    def test_three_prior_and_seven_forward_chains(self):
        prior = {c["id"] for c in self.pack["cards"] if c["pre"]["reading"].get("feedback_chain")}
        forward = {c["id"] for c in self.pack["cards"] if c["pre"]["reading"].get("next_research_chain")}
        self.assertEqual(prior, {"packet-01", "packet-02", "packet-03"})
        self.assertEqual(forward, {"packet-05", "packet-36", "packet-37", "packet-38", "packet-42", "packet-44", "packet-46"})
        for cid in prior | forward:
            reading = self.cards[cid]["pre"]["reading"]
            chain = reading["feedback_chain" if cid in prior else "next_research_chain"]
            mode = "explicit_parent" if cid in prior | {"packet-05"} else "round_followup"
            validate_chain(chain, mode, cid in forward)
            for text in [chain[k] for k in ("source", "feedback", "change")] + chain["next"]:
                self.assertNotRegex(text["en"], r"[\u3400-\u9fff]")

    def test_prior_values_use_actual_sent_feedback_and_correct_parcels(self):
        values = {"packet-01": 0.509765625, "packet-02": 0.958984375, "packet-03": 0.94921875}
        for cid, value in values.items():
            evidence = self.evidence["prior"][cid]
            self.assertEqual(evidence["feedback_as_sent"]["joint_direction_stability"], value)
            chain = self.cards[cid]["pre"]["reading"]["feedback_chain"]
            for lang in ("zh", "en"):
                self.assertIn(f"{100 * value:.1f}%", chain["feedback"][lang])
        second = self.cards["packet-02"]["pre"]["reading"]["feedback_chain"]
        self.assertIn("dorsal attention", second["source"]["en"])
        self.assertIn("salience/ventral attention", second["next"][0]["en"])
        self.assertIn("ROI108", second["change"]["en"])
        third = self.cards["packet-03"]["pre"]["reading"]["feedback_chain"]
        self.assertIn("left", third["source"]["en"])
        self.assertIn("right", third["next"][0]["en"])

    def test_round_context_not_falsely_labelled_parent_or_completed_child(self):
        for cid, evidence in self.evidence["forward"].items():
            chain = self.cards[cid]["pre"]["reading"]["next_research_chain"]
            if cid == "packet-05":
                self.assertEqual(chain["mode"], "explicit_parent")
                self.assertIn("98.2%", chain["outcome"]["en"])
                self.assertIn("ADHD", chain["next"][0]["en"])
            else:
                self.assertFalse(evidence["individual_parent_relationship_recorded"])
                self.assertEqual(chain["mode"], "round_followup")
                self.assertTrue(all(c["experiment_configurations"] == 0 for c in evidence["subsequent"]))
                self.assertNotIn("child hypothesis", chain["change"]["en"])
                self.assertTrue(chain["outcome"]["en"])

    def test_assignments_and_annotation_values_unchanged(self):
        for kind, before, after in (("assignments", 9, 10), ("reference_notes", 8, 9), ("significance", 8, 9)):
            old = json.loads((HERE / f"cs1_discovery_{kind}_v{before}.json").read_bytes())
            new = json.loads(self.outputs[f"cs1_discovery_{kind}_v{after}.json"])
            for value in (old, new):
                del value["pack_id"], value["pack_sha256"]
            self.assertEqual(old, new)

    def test_recursive_translation_covers_multiple_proposals_and_retains_old_strings(self):
        old = json.loads((HERE / "cs1_discovery_en_v12.json").read_bytes())["strings"]
        new = json.loads(self.outputs["cs1_discovery_en_v13.json"])["strings"]
        self.assertTrue(all(new[k] == v for k, v in old.items()))
        for card in self.pack["cards"]:
            reading = card["pre"]["reading"]
            for key in ("feedback_chain", "next_research_chain"):
                if key not in reading:
                    continue
                chain = reading[key]
                for value in [chain[k] for k in ("source", "feedback", "change")] + chain["next"]:
                    self.assertEqual(new[value["zh"]], value["en"])

    def test_malformed_relationship_or_missing_translation_is_rejected(self):
        chain = copy.deepcopy(self.cards["packet-01"]["pre"]["reading"]["feedback_chain"])
        with self.assertRaises(ValueError):
            validate_chain(chain, "round_followup", False)
        del chain["next"][0]["en"]
        with self.assertRaises(ValueError):
            validate_chain(chain, "explicit_parent", False)

    def test_historical_session_frozen_new_session_has_concrete_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            old = DiscoveryStudy(PARENT, data_dir=directory, record_kind="test")
            view = old.create({"code": "TEST-LINEAGE-OLD", "experience": "3-5", "assignment_id": "P01"})
            saved = old.mutate(view["session"]["id"], view["session_token"], "save", {
                "request_id": "before-lineage", "revision": 0, "card_id": view["cards"][0]["id"],
                "answers": {"novelty": "4"}, "note": "keep original", "issues": []})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v14.json", data_dir=directory, record_kind="test")
            restored = new.get(view["session"]["id"], view["session_token"])
            self.assertEqual(saved["session"], restored["session"])
            self.assertEqual(saved["cards"], restored["cards"])
            self.assertEqual(new.config()["pack_id"], PACK_ID)
            self.assertEqual(new.config()["meta"]["lineage_detail_revision"], LINEAGE_REVISION)
            fresh = new.create({"code": "TEST-LINEAGE-NEW", "experience": "3-5", "assignment_id": "P01"})
            self.assertEqual(sum("feedback_chain" in c["pre"]["reading"] for c in fresh["cards"]), 3)
            self.assertEqual(sum("next_research_chain" in c["pre"]["reading"] for c in fresh["cards"]), 7)
            self.assertNotIn("organizer", fresh)


if __name__ == "__main__":
    unittest.main()
