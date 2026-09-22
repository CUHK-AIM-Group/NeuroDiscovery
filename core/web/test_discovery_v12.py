"""Readable context retains sample scale, every cited paper and unchanged science."""
import copy
import hashlib
import json
import tempfile
import unittest

from core.web.build_discovery_pilot_v12 import HERE, PARENT, PARENT_SHA, PACK_ID, build, study_scale
from core.web.discovery_study import DiscoveryStudy


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = build()
        cls.parent = json.loads(PARENT.read_bytes())
        cls.pack = json.loads(cls.outputs["cs1_discovery_pilot_v12.json"])
        cls.cards = {card["id"]: card for card in cls.pack["cards"]}

    def test_reproducible_version_and_frozen_parent(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        self.assertEqual(self.outputs, build())
        for name, raw in self.outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_only_presentation_changes_in_cards(self):
        for old, new in zip(self.parent["cards"], self.pack["cards"]):
            before, after = copy.deepcopy(old), copy.deepcopy(new)
            del before["pre"]["reading"]
            del after["pre"]["reading"]
            self.assertEqual(before, after, old["id"])
        for field in ("questions", "issues", "scoring", "common_pre", "common_post"):
            self.assertEqual(self.parent[field], self.pack[field])
        self.assertEqual(self.parent["organizer"]["mapping"], self.pack["organizer"]["mapping"])
        self.assertEqual(self.parent["organizer"]["selection_evidence"], self.pack["organizer"]["selection_evidence"])

    def test_all_sixteen_citations_have_study_and_case_specific_relation(self):
        total, pmids = 0, set()
        for card in self.pack["cards"]:
            details = card["pre"]["reading"]["reference_details"]
            self.assertEqual(set(details), {r["id"] for r in card["pre"]["references"]})
            for ref in card["pre"]["references"]:
                total += 1
                pmids.add(ref["pmid"])
                detail = details[ref["id"]]
                self.assertGreater(len(detail["title"]), 15)
                for key in ("did", "relation"):
                    self.assertEqual(set(detail[key]), {"zh", "en"})
                    self.assertTrue(all(len(v) > 20 for v in detail[key].values()))
        self.assertEqual((total, len(pmids)), (16, 12))
        self.assertEqual(self.pack["organizer"]["presentation_context"]["new_references_added"], 0)
        self.assertIn("Klotho", self.cards["packet-37"]["pre"]["reading"]["reference_details"]["P1"]["title"])

    def test_effective_sample_sizes_are_not_copied_across_cases(self):
        expected = {"packet-36": (313, 313), "packet-37": (326, 313), "packet-38": (313, 313),
                    "packet-42": (140, 98), "packet-44": (140, 98), "packet-46": (158, 101)}
        for cid, (ni, ne) in expected.items():
            text, facts = study_scale(self.cards[cid])
            self.assertEqual((facts["internal_n"], facts["external_n"]), (ni, ne))
            self.assertEqual(facts["configurations_per_cohort"], 3)
            for lang in ("zh", "en"):
                self.assertIn(str(ni), text[lang])
                self.assertIn(str(ne), text[lang])
        self.assertEqual(study_scale(self.cards["packet-46"])[1]["events_by_2_3_5_year_window"],
                         {"internal": [12, 21, 29], "external": [7, 12, 18]})

    def test_connectivity_scale_retains_cohort_specific_groups_not_a_pooled_total(self):
        for cid in ("packet-01", "packet-02", "packet-03", "packet-05"):
            text, facts = study_scale(self.cards[cid])
            self.assertEqual(facts["cohorts"], 4)
            self.assertEqual((facts["internal_comparisons"], facts["external_primary_comparisons"]), (2, 4))
            for row in facts["groups"]:
                for lang in ("zh", "en"):
                    self.assertIn(str(row["cases"]), text[lang])
                    self.assertIn(str(row["controls"]), text[lang])

    def test_narrative_statistics_match_the_displayed_primary_results(self):
        for card in self.pack["cards"]:
            post, text = card["post"], card["pre"]["reading"]["results"]["en"].replace("−", "-")
            if "experimental_results" not in post:
                for row in post["internal_rows"] + [r for r in post["external_rows"] if r["variant"].startswith("primary_")]:
                    self.assertIn(f"{row['standardized_beta']:.3f}", text)
            else:
                config = next(c for c in post["experimental_results"]
                              if c["model"] == "association_glm" or c["horizon_years"] == 5)
                for phase in ("internal", "external"):
                    row = config[phase]
                    value = row["primary"]["effect"] if post["result_kind"] == "imaging_genetics" else row["primary"]["hazard_ratio"]
                    self.assertIn(f"{value:.3f}", text, card["id"])
                    self.assertIn(f"{row['q_value']:.4f}", text, card["id"])

    def test_distribution_and_historical_translations_are_preserved(self):
        previous = json.loads((HERE / "cs1_discovery_assignments_v7.json").read_bytes())
        current = json.loads(self.outputs["cs1_discovery_assignments_v8.json"])
        for field in ("experts", "coverage", "allocation_revision", "topic_slots"):
            self.assertEqual(previous[field], current[field])
        old_strings = json.loads((HERE / "cs1_discovery_en_v9.json").read_bytes())["strings"]
        new_strings = json.loads(self.outputs["cs1_discovery_en_v11.json"])["strings"]
        self.assertTrue(all(new_strings[k] == v for k, v in old_strings.items()))

    def test_old_session_keeps_its_material_and_answers_and_new_session_gets_context(self):
        with tempfile.TemporaryDirectory() as directory:
            old = DiscoveryStudy(PARENT, data_dir=directory, record_kind="test")
            view = old.create({"code": "TEST-CONTEXT-OLD", "experience": "3-5", "assignment_id": "P01"})
            saved = old.mutate(view["session"]["id"], view["session_token"], "save", {
                "request_id": "before-context", "revision": 0, "card_id": view["cards"][0]["id"],
                "answers": {"novelty": "4"}, "note": "original note", "issues": []})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v12.json", data_dir=directory, record_kind="test")
            restored = new.get(view["session"]["id"], view["session_token"])
            self.assertEqual(saved["session"], restored["session"])
            self.assertEqual(saved["cards"], restored["cards"])
            self.assertEqual(new.config()["pack_id"], PACK_ID)
            self.assertEqual(new.config()["experimental_results_revision"], "visible-results-v3")
            fresh = new.create({"code": "TEST-CONTEXT-NEW", "experience": "3-5", "assignment_id": "P01"})
            for card in fresh["cards"]:
                self.assertIn("scale", card["pre"]["reading"])
                self.assertIn("reference_details", card["pre"]["reading"])
            display = {"language": "en", "version": fresh["presentation"]["version"],
                       "sha256": fresh["presentation"]["sha256"], "renderer_revision": "visible-results-v2"}
            answer = new.mutate(fresh["session"]["id"], fresh["session_token"], "save", {
                "request_id": "with-context", "revision": 0, "card_id": fresh["cards"][0]["id"],
                "answers": {"novelty": "3"}, "note": "", "issues": [], "display": display})
            self.assertEqual(answer["session"]["display_history"][-1]["display"], display)


if __name__ == "__main__":
    unittest.main()
