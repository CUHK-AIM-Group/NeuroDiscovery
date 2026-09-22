"""Reference-notes v1 sidecar: builder integrity + service merge/binding."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_study_bank_reference_notes_v1 import (
    BANK, BANK_SHA256, TARGET, build, _extract_cohort, _extract_p_value, _reference_key,
)

_SERVICE_PATH = Path(__file__).resolve().parents[2] / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurodiscovery_user_study_notes_test", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class BuilderTests(unittest.TestCase):
    def test_notes_on_disk_match_deterministic_rebuild(self):
        doc = build()
        on_disk = json.loads(TARGET.read_bytes().decode("utf-8-sig"))
        self.assertEqual(doc["references"], on_disk["references"])
        self.assertEqual(on_disk["schema_version"], "case1-tcp-expert-study-v2-reference-notes-v1")
        self.assertEqual(on_disk["bank_file"], BANK.name)
        self.assertEqual(on_disk["bank_sha256"], BANK_SHA256)
        self.assertEqual(hashlib.sha256(BANK.read_bytes()).hexdigest(), BANK_SHA256)

    def test_every_note_key_matches_a_bank_reference(self):
        bank = json.loads(BANK.read_bytes().decode("utf-8-sig"))
        bank_keys = set()
        for hypothesis in bank["hypotheses"]:
            for paper in hypothesis.get("literature", []):
                bank_keys.add(_reference_key(paper))
        doc = build()
        self.assertTrue(doc["references"])
        self.assertLessEqual(set(doc["references"]), bank_keys)
        # The local stores resolve abstracts for the large majority of the
        # auto-matched references; never silently ship an empty sidecar.
        self.assertGreaterEqual(len(doc["references"]), 100)

    def test_note_entries_are_well_formed_and_labelled(self):
        for key, note in build()["references"].items():
            self.assertGreaterEqual(len(note["abstract"]), 80, key)
            self.assertTrue(note["abstract_source"].startswith(("corpus_", "pubmed_")), key)
            self.assertIs(note["abstract_verified"], True, key)
            credibility = note.get("credibility") or {}
            cohort = credibility.get("cohort")
            if cohort:
                self.assertIsInstance(cohort["n_total"], int, key)
                self.assertGreaterEqual(cohort["n_total"], 10, key)
                self.assertIn("自动提取", cohort["display_zh"], key)
                self.assertIn("auto-extracted", cohort["display_en"], key)
                self.assertIn(cohort["extraction"], ("n_equals", "count_keyword"), key)
            p_values = credibility.get("p_values")
            if p_values:
                self.assertEqual(p_values["status"], "reported_elsewhere_in_abstract", key)
                self.assertRegex(p_values["display"], r"^p [<=>] ", key)

    def test_extractors_are_conservative(self):
        abstract = "We enrolled 128 participants (64 patients and 64 controls). The effect was significant (p < 0.001)."
        self.assertEqual(_extract_cohort(abstract), (128, "count_keyword"))
        self.assertEqual(_extract_p_value(abstract), "p < 0.001")
        self.assertEqual(_extract_cohort("Total N = 250 were scanned."), (250, "n_equals"))
        self.assertEqual(_extract_cohort("No counts here."), (None, ""))
        self.assertEqual(_extract_p_value("Nothing significant reported."), "")


class ServiceMergeTests(unittest.TestCase):
    def test_missing_sidecar_disables_enrichment(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bank.json"
            source.write_text("{}", encoding="utf-8")
            self.assertIsNone(_MODULE.load_reference_notes(source, "abc"))

    def test_sidecar_bound_to_another_bank_hash_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bank.json"
            source.write_text("{}", encoding="utf-8")
            _MODULE.reference_notes_path(source).write_text(
                json.dumps({"bank_sha256": "deadbeef", "references": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _MODULE.load_reference_notes(source, "abc")

    def test_merge_fills_only_gaps(self):
        candidates = [{"literature": [
            {"pmid": "1", "title": "a", "abstract": "original"},
            {"pmid": "2", "title": "b", "credibility": {"cohort": {"n_total": 50}}},
            {"pmid": "3", "title": "c"},
        ]}]
        notes = {"references": {
            "1": {"abstract": "replacement", "abstract_source": "corpus_x", "abstract_verified": True},
            "2": {"abstract": "new abstract", "abstract_source": "corpus_x", "abstract_verified": True,
                   "credibility": {"cohort": {"n_total": 99}, "p_values": {"status": "reported_elsewhere_in_abstract", "display": "p < 0.05"}}},
            "3": {"abstract": "new abstract", "abstract_source": "corpus_x", "abstract_verified": True},
        }}
        enriched = _MODULE.apply_reference_notes(candidates, notes)
        # paper1 already had an abstract and the note adds nothing; only
        # paper2 (abstract + p-values) and paper3 (abstract) change.
        self.assertEqual(enriched, 2)
        paper1, paper2, paper3 = candidates[0]["literature"]
        self.assertEqual(paper1["abstract"], "original")
        self.assertNotIn("abstract_source", paper1)
        self.assertEqual(paper2["abstract"], "new abstract")
        self.assertEqual(paper2["credibility"]["cohort"]["n_total"], 50)
        self.assertEqual(paper2["credibility"]["p_values"]["display"], "p < 0.05")
        self.assertEqual(paper3["abstract"], "new abstract")
        self.assertEqual(paper3["abstract_source"], "corpus_x")

    def test_real_session_carries_enriched_literature(self):
        if not TARGET.is_file():
            self.skipTest("reference notes sidecar not built")
        with tempfile.TemporaryDirectory() as temp:
            service = _MODULE.UserStudyService(root=temp)
            view = service.create_session(
                study_id="case1-tcp-external-validation-v1",
                participant_id="NOTES-TEST", condition="manual",
                candidate_path=BANK, experience="3-5", assignment_id="P01")
            slots = [
                paper
                for candidate in view["candidates"]
                for paper in (candidate.get("literature") or [])
            ]
            with_abstract = [p for p in slots if str(p.get("abstract") or "").strip()]
            auto_with_abstract = [
                p for p in with_abstract if p.get("manual_relevance_status") == "auto_kg_match"]
            self.assertGreater(len(auto_with_abstract), 2000)
            self.assertTrue(any((p.get("credibility") or {}).get("cohort") for p in slots))
            self.assertTrue(any((p.get("credibility") or {}).get("p_values") for p in slots))
            # Scores stay stripped for manual sessions; enrichment adds no leaks.
            for candidate in view["candidates"]:
                self.assertNotIn("composite_score", candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
