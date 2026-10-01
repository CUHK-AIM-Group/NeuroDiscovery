from copy import deepcopy
import unittest

from neurooracle.src import pilot_validation_v3 as validation
from neurooracle.src.paper_extraction_contract_v3 import build_prompt
from neurooracle.src.source_statistics import parse_p_literal


def binding(quote, text):
    start = quote.index(text)
    return {"quote_index": 0, "start": start, "end": start + len(text), "text": text}


def example():
    quote = "In patients, genotype correlated with volume (p = 0.01)."
    def slot(text):
        if text is None:
            return {"value": None, "bindings": [], "unknown_reason": "not reported"}
        return {"value": text, "bindings": [binding(quote, text)], "unknown_reason": None}
    contrast = binding(quote, quote)
    observation = {
        "role": "primary_result", "quotes": [quote],
        "proposition": {"relation": "association"},
        "result": {"significance": "reported_significant"},
        "structured_result": {
            "schema": "source_result_v3", "kind": "association",
            "population": slot("patients"), "exposure": slot("genotype"), "outcome": slot("volume"),
            "comparator": slot(None), "timepoint": slot(None), "disease_stage": slot(None),
            "contrast": contrast, "atomicity": {"status": "single", "reason": "one association"},
            "test": {"status": "reported_significant", "contrast": deepcopy(contrast), "evidence": binding(quote, "p = 0.01")},
        },
    }
    return observation, quote


class PValueParsing(unittest.TestCase):
    def test_supported_notations(self):
        for raw, expected in [("P=0. 003", ("=", .003)), ("P = 0.\u00a0003", ("=", .003)),
                              ("p < .005", ("<", .005)), ("p ≤ 1e-3", ("<=", .001)),
                              ("p > .05", (">", .05)), ("p < or = 0.05", ("<=", .05))]:
            self.assertEqual(parse_p_literal(raw), expected)

    def test_ambiguous_or_impossible_numbers_not_guessed(self):
        for raw in ("p=0.003-.02", "p=0.003 to 0.02", "0.01 < p < 0.05", "p=0.003 5",
                    "p=0.0 03", "p=1.003", "p=-.01", "p=.01; p=.02", "p=0.003e", "p=0."):
            self.assertIsNone(parse_p_literal(raw), raw)


class ResultContract(unittest.TestCase):
    def test_valid_source_bound_contract(self):
        observation, text = example()
        before = deepcopy(observation)
        self.assertEqual(validation.observation_errors(observation, text), [])
        self.assertEqual(before, observation)

    def test_legacy_never_silently_upgraded(self):
        self.assertIn("structured_result_v3_missing", validation.observation_errors({"quotes": ["Source"]}, "Source"))

    def test_no_guessed_healthy_comparator(self):
        observation, text = example()
        observation["structured_result"]["comparator"] = {"value": "healthy controls", "bindings": [binding(text, "patients")], "unknown_reason": None}
        self.assertIn("comparator_binding_mismatch", validation.observation_errors(observation, text))

    def test_unknown_requires_reason(self):
        observation, text = example()
        observation["structured_result"]["comparator"]["unknown_reason"] = None
        self.assertIn("comparator_unknown_needs_reason", validation.observation_errors(observation, text))

    def test_real_but_unrelated_source_comparator_is_not_bound(self):
        observation, text = example()
        observation["quotes"].append("Healthy controls were also enrolled.")
        observation["structured_result"]["comparator"] = {
            "value": "Healthy controls", "unknown_reason": None,
            "bindings": [{"quote_index": 1, "start": 0, "end": 16, "text": "Healthy controls"}]}
        self.assertIn("association_comparator_not_in_bound_contrast", validation.observation_errors(observation, text + " Healthy controls were also enrolled."))

    def test_between_group_needs_comparator(self):
        observation, text = example()
        observation["structured_result"]["kind"] = "between_groups"
        self.assertIn("comparison_comparator_unresolved", validation.observation_errors(observation, text))

    def test_empty_output_requires_review(self):
        self.assertEqual(validation.validate_candidate({"observations": []}, "Source"), [{"reason": "no_observations_requires_review"}])

    def test_missing_exposure_is_not_population(self):
        observation, text = example()
        observation["structured_result"]["exposure"] = {"value": None, "bindings": [], "unknown_reason": "unknown"}
        self.assertIn("association_exposure_unresolved", validation.observation_errors(observation, text))

    def test_bad_offsets_and_boolean_index_rejected(self):
        observation, text = example()
        observation["structured_result"]["population"]["bindings"][0]["quote_index"] = False
        self.assertIn("population_binding_mismatch", validation.observation_errors(observation, text))

    def test_test_must_bind_same_comparison(self):
        observation, text = example()
        observation["structured_result"]["test"]["contrast"] = binding(text, "patients")
        self.assertIn("significance_contrast_binding_mismatch", validation.observation_errors(observation, text))

    def test_p_value_outside_clause_does_not_license_significance(self):
        observation, text = example()
        contrast = binding(text, "genotype correlated with volume")
        observation["structured_result"]["contrast"] = contrast
        observation["structured_result"]["test"]["contrast"] = deepcopy(contrast)
        self.assertIn("test_evidence_outside_comparison_clause", validation.observation_errors(observation, text))

    def test_composite_needs_review_not_automatic_split(self):
        observation, text = example()
        observation["structured_result"]["exposure"]["value"] = "OPC and RG"
        self.assertIn("exposure_possibly_composite_requires_review", validation.observation_errors(observation, text))

    def test_unresolved_atomicity_never_passes(self):
        observation, text = example()
        observation["structured_result"]["atomicity"]["status"] = "unresolved"
        self.assertIn("atomicity_unresolved_requires_review", validation.observation_errors(observation, text))

    def test_repeated_result_not_two_atomic_results(self):
        observation, text = example()
        second = deepcopy(observation)
        second["result"]["direction"] = "positive"
        errors = validation.validate_candidate({"observations": [observation, second]}, text)
        self.assertTrue(any(error["reason"] == "repeated_result_identity_requires_review" for error in errors))

    def test_malformed_contract_is_held(self):
        observation, text = example()
        observation["structured_result"]["test"] = []
        observation["structured_result"]["population"] = []
        self.assertIn("test_status_missing", validation.observation_errors(observation, text))

    def test_derived_is_not_reported(self):
        observation, text = example()
        observation["structured_result"]["test"]["status"] = "derived"
        self.assertIn("derived_test_requires_review", validation.observation_errors(observation, text))

    def test_legacy_result_cannot_disagree_with_structured_test(self):
        observation, text = example()
        observation["result"]["significance"] = "not_reported"
        self.assertIn("legacy_and_structured_test_disagree", validation.observation_errors(observation, text))

    def test_prompt_amendment_does_not_change_base(self):
        base = "Frozen prompt"
        self.assertIn("structured_result", build_prompt(base))
        self.assertEqual(base, "Frozen prompt")


class LegacyDiagnostic(unittest.TestCase):
    def test_groupwise_significance_not_between_group_evidence(self):
        observation = {"quotes": ["Controls, but not patients, showed significant activation."],
                       "proposition": {"relation": "group_difference"}, "result": {"significance": "reported_significant"}}
        self.assertIn("groupwise_significance_requires_contrast_review", validation.legacy_risks(observation))

    def test_groupwise_clause_not_blocked_as_within_group_result(self):
        observation = {"quotes": ["Controls, but not patients, showed significant activation."],
                       "proposition": {"relation": "longitudinal_change"}, "result": {"significance": "reported_significant"}}
        self.assertNotIn("groupwise_significance_requires_contrast_review", validation.legacy_risks(observation))

    def test_explicit_between_group_statement_not_automatically_blocked(self):
        observation = {"quotes": ["Activation was significantly higher in controls than patients."],
                       "proposition": {"relation": "group_difference"}, "result": {"significance": "reported_significant"}}
        self.assertEqual(validation.legacy_risks(observation), [])

    def test_baseline_is_not_stage(self):
        self.assertIn("timepoint_used_as_disease_stage", validation.legacy_risks({"conditions": {"stage": "baseline"}}))

    def test_unchanged_is_not_explicit_significance(self):
        observation = {"quotes": ["Volume remained unchanged."], "result": {"significance": "not_significant"}}
        self.assertIn("null_report_is_not_explicit_test_significance", validation.legacy_risks(observation))

    def test_actual_nonsignificance_is_allowed(self):
        observation = {"quotes": ["Volume did not differ significantly (p = 0.2)."], "result": {"significance": "not_significant"}}
        self.assertEqual(validation.legacy_risks(observation), [])


if __name__ == "__main__":
    unittest.main()
