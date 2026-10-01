import unittest

from neurooracle.src import proposition_scope_v2 as scope


def proposition(species="human", population=None, measurement=None, relation="group_difference",
                subject="hippocampal volume", object_="hippocampus", comparator=None,
                disease_stage=None, anatomy=None, modality=None, direction="lower"):
    qualifiers = {"species": species, "population": population, "measurement": measurement,
                  "comparator": comparator, "disease_stage": disease_stage,
                  "anatomy": anatomy, "modality": modality}
    return {"subject": subject, "object": object_, "relation": relation,
            "qualifiers": {slot: value for slot, value in qualifiers.items() if value},
            "scope": {}, "direction": direction, "polarity": "asserted"}


def observation(statement, direction="lower", significance="reported_significant"):
    return {"statement": statement, "quotes": [statement],
            "result": {"direction": direction, "significance": significance},
            "proposition": {"direction": direction, "polarity": "asserted"}}


class ScopeGateTests(unittest.TestCase):
    def test_different_disease_is_not_equivalent(self):
        left = proposition(population="first-episode schizophrenia patients", measurement="hippocampal volume")
        right = proposition(population="preclinical dementia", measurement="hippocampal volume")
        gate = scope.adjudication_gate(left, right)
        self.assertTrue(gate["hard_blocked"])
        self.assertEqual(gate["hint"], "related")
        self.assertIn("hard_scope_conflict:population", gate["reasons"])

    def test_same_scope_replication_is_allowed(self):
        left = proposition(population="adult schizophrenia patients", measurement="hippocampal volume",
                           comparator="healthy controls", anatomy="hippocampus")
        right = proposition(population="adult schizophrenia patients", measurement="hippocampal volume",
                            comparator="healthy controls", anatomy="hippocampus", modality="3T MRI")
        gate = scope.adjudication_gate(left, right)
        self.assertFalse(gate["hard_blocked"])
        self.assertEqual(gate["hint"], "model_must_justify")
        self.assertTrue(gate["model_justification_required"])
        self.assertIn("variant:modality", gate["reasons"])

    def test_soft_measurement_wording_difference_needs_model_not_gate(self):
        left = proposition(population="adult patients", measurement="hippocampal volume",
                           comparator="healthy controls")
        right = proposition(population="adult patients", measurement="automated region-of-interest volumetry",
                            comparator="healthy controls")
        gate = scope.adjudication_gate(left, right)
        self.assertFalse(gate["hard_conflicts"])
        self.assertFalse(gate["hard_blocked"])
        self.assertIn("soft_difference:measurement", gate["reasons"])

    def test_missing_required_scope_is_unresolved_not_equivalent(self):
        left = proposition(population="adult patients", measurement="hippocampal volume", comparator="controls")
        right = proposition(population=None, measurement="hippocampal volume", comparator="controls")
        gate = scope.adjudication_gate(left, right)
        self.assertFalse(gate["hard_blocked"])
        self.assertEqual(gate["hint"], "model_must_justify")
        self.assertIn("missing_scope:population", gate["reasons"])

    def test_disjoint_diseases_hard_block_even_with_shared_words(self):
        left = proposition(population="out patients with ptsd")
        right = proposition(population="pediatric samples with ptsd")
        gate = scope.adjudication_gate(left, right)
        self.assertEqual(gate["hint"], "model_must_justify")

    def test_disease_term_detection(self):
        self.assertIn("ptsd", scope.disease_terms("out patients with PTSD"))
        self.assertIn("development", scope.disease_terms("pediatric samples with ptsd"))
        self.assertTrue(scope.disjoint_conditions("patients with schizophrenia", "preclinical dementia"))
        self.assertFalse(scope.disjoint_conditions("persons with ptsd", "pediatric samples with ptsd"))

    def test_asymmetric_relation_cannot_be_orientation_swapped(self):
        left = proposition(relation="predicts", subject="hippocampal volume", object_="dementia")
        right = proposition(relation="predicts", subject="dementia", object_="hippocampal volume")
        self.assertFalse(scope.orientation_matches(left, right))
        self.assertEqual(scope.adjudication_gate(left, right)["hint"], "distinct")

    def test_symmetric_relation_allows_swap(self):
        left = proposition(relation="association", subject="hippocampal volume", object_="memory")
        right = proposition(relation="association", subject="memory", object_="hippocampal volume")
        self.assertTrue(scope.orientation_matches(left, right))

    def test_null_versus_assertion_is_not_counterevidence(self):
        left = proposition(population="people with HIV")
        right = proposition(population="left temporal lobe epilepsy", anatomy="left hippocampus")
        gate = scope.adjudication_gate(left, right, observation("Hippocampal volume did not differ"),
                                       observation("Left hippocampal volume was significantly smaller"))
        self.assertTrue(gate["hard_blocked"])
        self.assertEqual(gate["stance_a"], "reported_null")

    def test_matched_scope_explicit_negation_is_counterevidence(self):
        left = proposition(population="adults", measurement="volume", comparator="controls")
        right = proposition(population="adults", measurement="volume", comparator="controls")
        negation = {"statement": "Hippocampal volume was not reduced in patients.",
                    "quotes": ["Hippocampal volume was not reduced in patients."],
                    "result": {"significance": "not_reported"}, "proposition": {"polarity": "negated"}}
        self.assertTrue(scope.counterevidence_permitted(left, right, observation("volume lower"),
                                                        negation))

    def test_region_alias_maps_to_anatomy(self):
        left = {"subject": "hippocampal volume", "object": "hippocampus", "relation": "group_difference",
                "qualifiers": {}, "scope": {"region": "hippocampus", "population": "adults",
                                            "species": "human"}, "polarity": "asserted"}
        self.assertEqual(scope.scope_slots(left)["anatomy"], "hippocampus")


if __name__ == "__main__":
    unittest.main()
