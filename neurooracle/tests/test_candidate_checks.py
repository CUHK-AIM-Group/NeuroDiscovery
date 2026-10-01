"""Consolidation step 2: one door, four explicit outcomes.

The equivalence test is the point of this file. It proves that routing old
candidates through the new door produces exactly the same blocking decision the
old pipeline made, so consolidation did not quietly change any verdict.
"""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest

from neurooracle.src import candidate_checks
from neurooracle.src import pilot_validation as exemptions
from neurooracle.src import pilot_validation_v2 as consistency
from neurooracle.src import pilot_validation_v3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))

STRUCTURED_QUOTE = "In patients, genotype correlated with volume (p = 0.01)."


def _binding(text):
    start = STRUCTURED_QUOTE.index(text)
    return {"quote_index": 0, "start": start, "end": start + len(text), "text": text}


def _slot(value):
    if value is None:
        return {"value": None, "bindings": [], "unknown_reason": "not reported"}
    return {"value": value, "bindings": [_binding(value)], "unknown_reason": None}


def structured_observation():
    contrast = _binding(STRUCTURED_QUOTE)
    return {
        "statement": "Genotype correlated with volume.",
        "quotes": [STRUCTURED_QUOTE],
        "role": "primary_result",
        "proposition": {"relation": "association"},
        "result": {"significance": "reported_significant"},
        "structured_result": {
            "schema": candidate_checks.STRUCTURED_SCHEMA,
            "kind": "association",
            "population": _slot("patients"),
            "exposure": _slot("genotype"),
            "outcome": _slot("volume"),
            "comparator": _slot(None),
            "timepoint": _slot(None),
            "disease_stage": _slot(None),
            "contrast": contrast,
            "atomicity": {"status": "single", "reason": "one association"},
            "test": {"status": "reported_significant", "contrast": deepcopy(contrast),
                     "evidence": _binding("p = 0.01")},
        },
    }


def legacy_observation():
    return {
        "statement": "Volume was smaller in patients.",
        "quotes": ["Volume was smaller in patients"],
        "role": "primary_result",
        "proposition": {"relation": "group_difference", "direction": "lower"},
        "result": {"significance": "not_reported"},
    }


class OutcomeContract(unittest.TestCase):
    def test_clean_legacy_candidate_is_contract_not_applicable(self):
        result = candidate_checks.check_candidate({"observations": [legacy_observation()]}, "x")
        self.assertEqual(result["verdict"], "contract_not_applicable")
        self.assertEqual(result["blocked"], [])
        self.assertEqual(result["findings"]["result_contract_fields"]["status"], "not_applicable")

    def test_structured_candidate_passes_the_contract_rule(self):
        result = candidate_checks.check_candidate({"observations": [structured_observation()]}, STRUCTURED_QUOTE)
        self.assertEqual(result["blocked"], [])
        self.assertEqual(result["findings"]["result_contract_fields"]["status"], "pass")

    def test_mixed_candidate_is_treated_as_structured_and_reports_the_gap(self):
        candidate = {"observations": [structured_observation(), legacy_observation()]}
        abstract = STRUCTURED_QUOTE + " Volume was smaller in patients"
        result = candidate_checks.check_candidate(candidate, abstract)
        self.assertEqual(result["findings"]["result_contract_fields"]["status"], "blocked")
        self.assertTrue(any("structured_result_v3_missing" in reason
                            for reason in result["blocked"]))

    def test_partial_contract_cannot_escape_validation(self):
        candidate = {"observations": [legacy_observation(), structured_observation()]}
        abstract = STRUCTURED_QUOTE + " Volume was smaller in patients"
        result = candidate_checks.check_candidate(candidate, abstract)
        self.assertEqual(result["verdict"], "blocked")

    def test_exemption_is_visible_instead_of_silently_dropped(self):
        candidate = {"observations": [legacy_observation()],
                     "study": {"samples": [{"group": "pwh", "n": 30,
                                            "raw": "Thirty virally suppressed PWH were included."}]}}
        source = "Thirty virally suppressed PWH were included. Controls were enrolled."
        result = candidate_checks.check_candidate(candidate, source, ["sample_number_not_source_bound"])
        self.assertIn("sample_number_not_source_bound", result["exempted"])
        self.assertNotIn("sample_number_not_source_bound", result["blocked"])

    def test_unexempted_structural_error_blocks(self):
        candidate = {"observations": [legacy_observation()]}
        result = candidate_checks.check_candidate(candidate, "irrelevant source", ["observation_0_quote_not_unique_literal"])
        self.assertIn("observation_0_quote_not_unique_literal", result["blocked"])
        self.assertEqual(result["verdict"], "blocked")

    def test_diagnostics_advise_but_do_not_block(self):
        observation = legacy_observation()
        observation["quotes"] = ["Controls, but not patients, showed significant activation."]
        observation["proposition"] = {"relation": "group_difference"}
        observation["result"] = {"significance": "reported_significant"}
        result = candidate_checks.check_candidate({"observations": [observation]}, "x")
        self.assertEqual(result["verdict"], "advisory")
        self.assertEqual(result["blocked"], [])
        self.assertTrue(any("groupwise_significance_requires_contrast_review" in reason
                            for reason in result["advisory"]))

    def test_input_is_never_mutated(self):
        candidate = {"observations": [legacy_observation()]}
        before = deepcopy(candidate)
        candidate_checks.check_candidate(candidate, "x", ["observation_0_quote_not_unique_literal"])
        self.assertEqual(candidate, before)

    def test_merge_is_never_allowed_by_a_passing_verdict(self):
        result = candidate_checks.check_candidate({"observations": [structured_observation()]}, STRUCTURED_QUOTE)
        self.assertFalse(result["policy"]["merge_allowed"])

    def test_advisory_reasons_are_declared_in_one_place(self):
        self.assertEqual(
            set(pilot_validation_v3.ADVISORY_REASONS),
            {"timepoint_used_as_disease_stage",
             "groupwise_significance_requires_contrast_review",
             "null_report_is_not_explicit_test_significance"})

    def test_contract_errors_never_include_advisory_reasons(self):
        observation = legacy_observation()
        observation["conditions"] = {"stage": "baseline"}
        errors = pilot_validation_v3.observation_errors(observation, "Volume was smaller in patients")
        self.assertEqual([reason for reason in errors if reason in pilot_validation_v3.ADVISORY_REASONS], [])
        self.assertIn("timepoint_used_as_disease_stage",
                      pilot_validation_v3.candidate_risks({"observations": [observation]})[0]["reason"])

    def test_rules_declare_a_severity(self):
        severities = {rule.severity for rule in candidate_checks.RULES}
        self.assertEqual(severities, {"block", "exempt", "advisory"})


FROZEN = ROOT / "tmp" / "kg_source_first_20260927"


@unittest.skipUnless((FROZEN / "blind_run" / "COMPLETED.json").is_file(),
                     "frozen first-body artifacts absent")
class FrozenEquivalenceTest(unittest.TestCase):
    """New door must reproduce the old blocking decision exactly."""

    def test_blocked_set_matches_the_old_pipeline_on_all_frozen_candidates(self):
        import paper_batch_model as structural_validator
        import paper_pilot_batch as batch

        completed = json.loads((FROZEN / "blind_run" / "COMPLETED.json").read_text(encoding="utf-8"))
        packets = {packet["work_key"]: packet for packet in
                   map(json.loads, (FROZEN / "MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines())}
        checked = 0
        for record in completed["records"]:
            body = [item for item in record["attempts"] if item["http_status"] == 200]
            if len(body) != 1 or not body[0].get("candidate_path"):
                continue
            candidate = json.loads((FROZEN / body[0]["candidate_path"]).read_text(encoding="utf-8"))
            source = batch.shim_source(packets[record["work_key"]])
            abstract = source["source_snapshot"]["abstract"]

            structural = structural_validator.validate_candidate(candidate, source)
            limitations = exemptions.validator_limitations(structural, candidate, abstract or "")
            old_blocked = sorted(set(error for error in structural if error not in set(limitations))
                                 | {"observation_%s:%s" % (item.get("observation_index"), item.get("reason"))
                                    for item in consistency.candidate_consistency_errors(candidate)})

            result = candidate_checks.check_candidate(candidate, abstract, structural)
            self.assertEqual(result["blocked"], old_blocked, record["work_key"])
            self.assertEqual(result["exempted"], sorted(set(limitations)), record["work_key"])
            self.assertEqual(result["findings"]["result_contract_fields"]["status"],
                             "not_applicable", record["work_key"])
            checked += 1
        self.assertEqual(checked, 10)


class ManifestTest(unittest.TestCase):
    def test_every_active_pipeline_path_exists(self):
        manifest = json.loads((ROOT / "neurooracle/configs/active_pipeline.json").read_text(encoding="utf-8"))
        missing = [item["path"] for item in manifest["files"]
                   if not (ROOT / item["path"]).is_file()]
        self.assertEqual(missing, [])

    def test_manifest_declares_the_five_owners_once_each(self):
        manifest = json.loads((ROOT / "neurooracle/configs/active_pipeline.json").read_text(encoding="utf-8"))
        paths = [item["path"] for item in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("neurooracle/src/candidate_checks.py", paths)
        self.assertFalse(manifest["archive_declared"]["enumerated"])


if __name__ == "__main__":
    unittest.main()
