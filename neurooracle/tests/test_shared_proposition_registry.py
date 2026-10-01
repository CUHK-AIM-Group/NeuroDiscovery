import unittest

from neurooracle.src import shared_proposition_registry as reg


class PropositionIdentity(unittest.TestCase):
    def test_paper_bound_fields_are_rejected_from_identity(self):
        for field in ("paper_id", "pmid", "title", "year", "sample_size", "p_value", "raw_text"):
            self.assertIn(field, reg.FORBIDDEN_KEY_FIELDS)
        with self.assertRaises(ValueError):
            reg.assert_no_forbidden_fields({"subject": "x", "p_value": 0.01})
        self.assertTrue(reg.assert_no_forbidden_fields({"subject": "x", "relation": "increases"}))

    def test_same_finding_different_papers_gets_one_key(self):
        first = reg.proposition_key("hippocampal volume", "reduces", "alzheimer disease",
                                    qualifiers={"population": "older adults"})
        second = reg.proposition_key("Hippocampal Volume", "reduces", "Alzheimer Disease",
                                     qualifiers={"population": "older adults"})
        self.assertEqual(first, second)

    def test_scope_difference_splits_the_key(self):
        adult = reg.proposition_key("hippocampal volume", "reduces", "alzheimer disease",
                                    qualifiers={"population": "older adults"})
        child = reg.proposition_key("hippocampal volume", "reduces", "alzheimer disease",
                                    qualifiers={"population": "children"})
        unspecified = reg.proposition_key("hippocampal volume", "reduces", "alzheimer disease")
        self.assertNotEqual(adult, child)
        self.assertNotEqual(adult, unspecified)

    def test_polarity_is_part_of_identity(self):
        asserted = reg.proposition_key("drug x", "reduces", "seizure frequency")
        negated = reg.proposition_key("drug x", "reduces", "seizure frequency", polarity="negated")
        self.assertNotEqual(asserted, negated)
        with self.assertRaises(ValueError):
            reg.proposition_key("drug x", "reduces", "seizure frequency", polarity="maybe")

    def test_relation_synonyms_share_a_family_but_unknown_relations_stay_distinct(self):
        self.assertEqual(reg.relation_family("correlates_with"), "association")
        self.assertEqual(reg.relation_family("is_associated_with"), "association")
        self.assertEqual(reg.relation_family("binds_to"), "binds_to")
        self.assertNotEqual(reg.relation_family("binds_to"), reg.relation_family("projects_to"))

    def test_sentence_like_or_empty_endpoints_are_refused(self):
        with self.assertRaises(ValueError):
            reg.proposition_key("", "reduces", "alzheimer disease")
        with self.assertRaises(ValueError):
            reg.proposition_key("hippocampal volume", "", "alzheimer disease")
        with self.assertRaises(ValueError):
            reg.proposition_key("hippocampal volume", "reduces", "   ")

    def test_qualifiers_reject_unknown_slots_and_keep_unknown_out(self):
        with self.assertRaises(ValueError):
            reg.normalize_qualifiers({"vibes": "high"})
        self.assertEqual(reg.normalize_qualifiers({"population": None}), {})
        self.assertEqual(reg.normalize_qualifiers({"species": None, "task": ""}), {})
        self.assertEqual(reg.normalize_qualifiers({"anatomy": ["Left Hippocampus", "left hippocampus"]}),
                         {"anatomy": ["left hippocampus"]})

    def test_identity_ignores_paper_wording_and_counts_works_once(self):
        registry = reg.SharedPropositionRegistry()
        key, created = registry.register(
            "hippocampal volume", "reduces", "alzheimer disease",
            qualifiers={"population": "older adults"},
            statement="Hippocampal volume is reduced in Alzheimer disease.",
            decision="create",
            reason="abstract states smaller hippocampal volume in AD",
        )
        self.assertTrue(created)
        reused, created_again = registry.register(
            "hippocampal volume", "reduces", "alzheimer disease",
            qualifiers={"population": "older adults"},
            reason="second abstract states the same scoped finding",
        )
        self.assertEqual(reused, key)
        self.assertFalse(created_again)
        registry.add_evidence(key, "pmid:1", "supports", reason="own result")
        registry.add_evidence(key, "pmid:2", "opposes", reason="null result in same scope")
        registry.add_evidence(key, "pmid:1", "supports", reason="duplicate cache of the same work")
        self.assertEqual(registry.works(key), ["pmid:1", "pmid:2"])
        self.assertEqual(registry.counts()["multipaper"], 1)
        self.assertEqual(registry.counts()["single_work"], 0)

    def test_every_decision_is_logged_and_merges_are_reversible(self):
        registry = reg.SharedPropositionRegistry()
        a, _ = registry.register("brain age", "predicts", "chronological age", reason="r1")
        b, _ = registry.register("cortical thickness", "predicts", "chronological age", reason="r2")
        registry.merge(a, b, reason="adjudicated equivalent after reading both sources")
        self.assertEqual(registry._merged_into[a], b)
        registry.add_evidence(a, "pmid:9", "supports", reason="evidence belongs to surviving key")
        self.assertIn("pmid:9", registry.works(b))
        registry.split(a, reason="new source shows a narrower measure")
        self.assertNotIn(a, registry._merged_into)
        actions = [entry["action"] for entry in registry.log]
        self.assertEqual(actions.count("merge"), 1)
        self.assertEqual(actions.count("split"), 1)
        self.assertTrue(all(entry["reason"] for entry in registry.log))

    def test_register_requires_a_reason_and_a_valid_decision(self):
        registry = reg.SharedPropositionRegistry()
        with self.assertRaises(ValueError):
            registry.register("aging", "is_risk_factor_for", "dementia", reason="")
        with self.assertRaises(ValueError):
            registry.register("aging", "is_risk_factor_for", "dementia", decision="guess", reason="x")
        with self.assertRaises(ValueError):
            registry.register("aging", "is_risk_factor_for", "dementia", decision="reuse", reason="x")

    def test_candidates_are_proposals_only(self):
        registry = reg.SharedPropositionRegistry()
        registry.register("brain age", "predicts", "chronological age", reason="r1")
        found = registry.candidates("brain-age", "predicts", "chronological age")
        self.assertEqual(len(found), 1)
        self.assertEqual(registry.candidates("clozapine", "treats", "schizophrenia"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
