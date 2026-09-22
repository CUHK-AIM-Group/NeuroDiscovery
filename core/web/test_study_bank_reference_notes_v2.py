"""Reference-notes v2 sidecar: abstracts + cohort/p + online-searched JIF."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from core.web.build_study_bank_reference_notes_v2 import (
    BANK, BANK_SHA256, JIF_TABLE, TARGET, build, canonical_journal,
    _extract_cohort, _extract_p_value, _reference_key,
)

_SERVICE_PATH = Path(__file__).resolve().parents[2] / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurodiscovery_user_study_notes_v2_test", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

JIF_STATUSES = {"reported_online_search", "not_applicable", "not_available", "not_verified"}


class BuilderV2Tests(unittest.TestCase):
    def test_notes_on_disk_match_deterministic_rebuild(self):
        doc = build()
        on_disk = json.loads(TARGET.read_bytes().decode("utf-8-sig"))
        self.assertEqual(doc["references"], on_disk["references"])
        self.assertEqual(on_disk["schema_version"], "case1-tcp-expert-study-v2-reference-notes-v2")
        self.assertEqual(on_disk["bank_sha256"], BANK_SHA256)
        self.assertEqual(hashlib.sha256(BANK.read_bytes()).hexdigest(), BANK_SHA256)

    def test_every_bank_reference_has_an_entry(self):
        bank = json.loads(BANK.read_bytes().decode("utf-8-sig"))
        bank_keys = set()
        for hypothesis in bank["hypotheses"]:
            for paper in hypothesis.get("literature", []):
                bank_keys.add(_reference_key(paper))
        doc = build()
        self.assertEqual(set(doc["references"]), bank_keys)

    def test_entries_are_well_formed(self):
        jif_journals = json.loads(JIF_TABLE.read_bytes().decode("utf-8-sig"))["journals"]
        self.assertGreaterEqual(len(jif_journals), 55)
        reported = [j for j in jif_journals.values() if j["status"] == "reported"]
        self.assertGreaterEqual(len(reported), 50)
        for key, note in build()["references"].items():
            credibility = note.get("credibility") or {}
            jif = credibility.get("journal_impact_factor") or {}
            self.assertIn(jif.get("status"), JIF_STATUSES, key)
            if jif.get("status") == "reported_online_search":
                self.assertIsInstance(jif["value"], float, key)
                self.assertGreater(jif["value"], 0, key)
                self.assertIn(jif.get("metric_year"), ("2023", "2024"), key)
            if note.get("abstract"):
                self.assertGreaterEqual(len(note["abstract"]), 80, key)
            cohort = credibility.get("cohort")
            if cohort:
                self.assertGreaterEqual(cohort["n_total"], 10, key)
                self.assertIn("自动提取", cohort["display_zh"], key)
                self.assertIn("auto-extracted", cohort["display_en"], key)
            p_values = credibility.get("p_values")
            if p_values:
                self.assertEqual(p_values["status"], "reported_elsewhere_in_abstract", key)
                self.assertRegex(p_values["display"], r"^p [<=>] ", key)

    def test_extractors(self):
        abstract = "We enrolled 128 participants (64 patients and 64 controls). Significant (p < 0.001)."
        self.assertEqual(_extract_cohort(abstract), (128, "max_of_n_equals_and_count_keyword"))
        self.assertEqual(_extract_p_value(abstract), "p < 0.001")
        self.assertEqual(_extract_cohort("A total of 91 controls and 13 patients were scanned."), (91, "max_of_n_equals_and_count_keyword"))
        self.assertEqual(_extract_cohort("The sample (n = 45) completed the task."), (45, "max_of_n_equals_and_count_keyword"))
        self.assertEqual(_extract_cohort("No counts here."), (None, ""))
        self.assertEqual(_extract_p_value("P=.034 survived correction."), "p = 0.034")

    def test_canonical_journal_matches_bank_spellings(self):
        self.assertEqual(canonical_journal("Frontiers in Human Neuroscience"), canonical_journal("Frontiers in human neuroscience"))
        self.assertEqual(canonical_journal("JAMA Psychiatry"), "jama psychiatry")
        self.assertEqual(canonical_journal("Brain"), "brain a journal of neurology")
        self.assertEqual(canonical_journal("medRxiv; published=10.1016/j.x.2026.1"), "medrxiv")


class ServiceV2Tests(unittest.TestCase):
    def test_newest_sidecar_wins_with_older_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bank.json"
            source.write_text("{}", encoding="utf-8")
            raw = json.loads(TARGET.read_bytes().decode("utf-8-sig"))
            raw["references"] = {"x": {"abstract": "a" * 100}}
            _MODULE.reference_notes_path(source, "_reference_notes_v1.json").write_text(
                json.dumps(raw), encoding="utf-8")
            notes = _MODULE.load_reference_notes(source, BANK_SHA256)
            self.assertIsNotNone(notes)  # v1 fallback works when v2 is absent
            _MODULE.reference_notes_path(source, "_reference_notes_v2.json").write_text(
                json.dumps({**raw, "references": {"y": {}}}), encoding="utf-8")
            notes = _MODULE.load_reference_notes(source, BANK_SHA256)
            self.assertEqual(list(notes["references"]), ["y"])  # v2 wins

    def test_merge_upgrades_valueless_jif_and_keeps_existing(self):
        candidates = [{"literature": [
            {"pmid": "1", "title": "a", "credibility": {"journal_impact_factor": {"status": "unverified"}}},
            {"pmid": "2", "title": "b", "credibility": {"journal_impact_factor": {"value": 9.9}}},
            {"pmid": "3", "title": "c"},
        ]}]
        notes = {"references": {
            "1": {"credibility": {"journal_impact_factor": {"value": 3.5, "metric_year": "2024", "status": "reported_online_search"}}},
            "2": {"credibility": {"journal_impact_factor": {"value": 3.5, "metric_year": "2024", "status": "reported_online_search"}}},
            "3": {"credibility": {"journal_impact_factor": {"value": 3.5, "metric_year": "2024", "status": "reported_online_search"}}},
        }}
        enriched = _MODULE.apply_reference_notes(candidates, notes)
        # p2 already carries a numeric JIF and is preserved; p1 (value-less
        # "unverified") and p3 (no credibility) are upgraded.
        self.assertEqual(enriched, 2)
        p1, p2, p3 = candidates[0]["literature"]
        self.assertEqual(p1["credibility"]["journal_impact_factor"]["value"], 3.5)
        self.assertEqual(p2["credibility"]["journal_impact_factor"]["value"], 9.9)
        self.assertEqual(p3["credibility"]["journal_impact_factor"]["value"], 3.5)

    def test_real_session_carries_abstract_cohort_pvalue_and_jif(self):
        if not TARGET.is_file():
            self.skipTest("reference notes v2 sidecar not built")
        with tempfile.TemporaryDirectory() as temp:
            service = _MODULE.UserStudyService(root=temp)
            view = service.create_session(
                study_id="case1-tcp-external-validation-v1",
                participant_id="NOTES-V2-TEST", condition="manual",
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
            with_jif = [
                p for p in slots
                if (p.get("credibility") or {}).get("journal_impact_factor", {}).get("value") is not None]
            self.assertGreater(len(with_jif), 1900)
            preprint_slots = [
                p for p in slots
                if (p.get("credibility") or {}).get("journal_impact_factor", {}).get("status") == "not_applicable"]
            self.assertTrue(preprint_slots)
            for candidate in view["candidates"]:
                self.assertNotIn("composite_score", candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
