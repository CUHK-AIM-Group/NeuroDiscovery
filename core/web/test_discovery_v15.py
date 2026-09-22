import copy
import hashlib
import json
import tempfile
import unittest

from core.web.build_discovery_pilot_v15 import HERE, PARENT, PARENT_SHA, PACK_ID, SOURCES, WRITING, build
from core.web.discovery_study import DiscoveryStudy


class RelatedLiteratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = build()
        cls.pack = json.loads(cls.outputs["cs1_discovery_pilot_v15.json"])
        cls.parent = json.loads(PARENT.read_bytes())

    def test_frozen_parent_and_repeatable_release(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        self.assertEqual(build(), self.outputs)
        for name, raw in self.outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_all_ten_cases_have_five_distinct_reviewed_references(self):
        self.assertEqual(len(self.pack["cards"]), 10)
        notes = json.loads(self.outputs["cs1_discovery_reference_notes_v10.json"])
        self.assertEqual(notes["note_count"], 50)
        for card in self.pack["cards"]:
            refs, reading = card["pre"]["references"], card["pre"]["reading"]
            self.assertEqual(len({ref["pmid"] for ref in refs}), 5)
            self.assertEqual([ref["id"] for ref in refs], [f"P{number}" for number in range(1, 6)])
            self.assertEqual(set(notes["notes"][card["id"]]), {ref["id"] for ref in refs})
            for ref in refs:
                detail = reading["reference_details"][ref["id"]]
                self.assertTrue(detail["title"] and detail["journal"] and detail["year"])
                for key in ("did", "relation"):
                    self.assertTrue(detail[key]["zh"] and detail[key]["en"])

    def test_original_science_and_original_references_unchanged(self):
        for key in ("questions", "scoring", "common_pre", "common_post"):
            self.assertEqual(self.pack[key], self.parent[key])
        for old, new in zip(self.parent["cards"], self.pack["cards"]):
            restored = copy.deepcopy(new)
            restored["pre"]["references"] = restored["pre"]["references"][:len(old["pre"]["references"])]
            for key in ("references", "reference_details"):
                restored["pre"]["reading"][key] = {reference_id: value for reference_id, value in restored["pre"]["reading"][key].items()
                                                      if reference_id in old["pre"]["reading"][key]}
            self.assertEqual(restored, old)

    def test_source_identity_anchors_and_no_notice_holds(self):
        sources = json.loads(SOURCES.read_bytes())["papers"]
        writing = json.loads(WRITING.read_bytes())
        self.assertEqual(len(sources), 26)
        for pmid, source in sources.items():
            self.assertEqual(source["document"]["pmid"], pmid)
            self.assertEqual(source["authority"]["pmid"], pmid)
            self.assertEqual(source["source_state"], "OWN_RECORD_CACHED")
            self.assertFalse(source["document"]["comments_corrections"])
            abstract = "\n".join(part["text"] for part in source["document"]["abstract"])
            self.assertIn(writing["papers"][pmid]["anchor"].casefold(), abstract.casefold())

    def test_null_results_and_scope_differences_remain_visible(self):
        writing = json.loads(WRITING.read_bytes())["papers"]
        self.assertIn("null", writing["24260406"]["relation"]["en"])
        self.assertIn("null", writing["39011362"]["relation"]["en"])
        self.assertIn("white matter", writing["18843182"]["relation"]["en"])
        self.assertIn("not", writing["28444556"]["relation"]["en"])

    def test_historical_translations_and_sidecar_content_preserved(self):
        previous = json.loads((HERE / "cs1_discovery_en_v13.json").read_bytes())["strings"]
        current = json.loads(self.outputs["cs1_discovery_en_v14.json"])["strings"]
        for text, translation in previous.items():
            self.assertEqual(current[text], translation)
        for kind, old, new in (("assignments", 10, 11), ("significance", 9, 10)):
            previous = json.loads((HERE / f"cs1_discovery_{kind}_v{old}.json").read_bytes())
            current = json.loads(self.outputs[f"cs1_discovery_{kind}_v{new}.json"])
            for key in ("pack_id", "pack_sha256"):
                previous.pop(key)
                current.pop(key)
            self.assertEqual(current, previous)

    def test_no_shared_doi_or_pmcid_within_each_card(self):
        sources = json.loads(SOURCES.read_bytes())["papers"]
        previous = json.loads((HERE / "sources_v12/literature.json").read_bytes())["papers"]
        authorities = {paper["id"]: {key: [paper[key]] if paper.get(key) else []
                                     for key in ("doi", "pmcid")} for paper in previous}
        authorities.update({pmid: source["authority"] for pmid, source in sources.items()})
        for card in self.pack["cards"]:
            seen = set()
            for reference in card["pre"]["references"]:
                authority = authorities[reference["pmid"]]
                aliases = {(key, value.lower().strip()) for key in ("doi", "pmcid") for value in authority[key]}
                self.assertFalse(seen & aliases, card["id"])
                self.assertTrue(aliases, reference["pmid"])
                seen.update(aliases)

    def test_historical_answers_and_materials_survive_new_release(self):
        with tempfile.TemporaryDirectory() as directory:
            old = DiscoveryStudy(PARENT, data_dir=directory, record_kind="test")
            view = old.create({"code": "TEST-OLD-REFS", "experience": "3-5", "assignment_id": "P01"})
            saved = old.mutate(view["session"]["id"], view["session_token"], "save", {
                "request_id": "before-references", "revision": 0, "card_id": view["cards"][0]["id"],
                "answers": {"novelty": "4"}, "note": "preserve", "issues": []})
            new = DiscoveryStudy(HERE / "cs1_discovery_pilot_v15.json", data_dir=directory, record_kind="test")
            restored = new.get(view["session"]["id"], view["session_token"])
            self.assertEqual(restored["session"], saved["session"])
            self.assertEqual(restored["cards"], saved["cards"])
            self.assertEqual(new.config()["pack_id"], PACK_ID)
            fresh = new.create({"code": "TEST-NEW-REFS", "experience": "3-5", "assignment_id": "P01"})
            self.assertTrue(all(len(card["pre"]["references"]) == 5 for card in fresh["cards"]))
            self.assertNotIn("organizer", fresh)


if __name__ == "__main__":
    unittest.main()
