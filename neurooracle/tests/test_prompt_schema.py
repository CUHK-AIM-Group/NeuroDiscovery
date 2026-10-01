"""Consolidation step 3: the schema is the only place shape facts live.

Two things are proved here. First, that the schema really is a single source:
the wire value, the structured_result required keys and the validators agree.
Second, that the live extraction prompt names every declared field and that its
hand-written shape block states exactly what the schema states -- that is the
evidence the plan demands before any embedded shape text is deleted.
"""
from pathlib import Path
import sys
import unittest

from neurooracle.src import pilot_validation_v3
from neurooracle.src import prompt_schema

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))

import paper_extract_v2  # noqa: E402
from neurooracle.src.paper_extraction_contract_v3 import (  # noqa: E402
    EXTRACTION_AMENDMENT_V3,
    build_prompt,
)

class RenderShapeTest(unittest.TestCase):
    def test_render_is_deterministic_and_hashes_stably(self):
        first = prompt_schema.render_shape()
        self.assertEqual(first, prompt_schema.render_shape())
        self.assertEqual(prompt_schema.shape_sha(), prompt_schema.shape_sha(prompt_schema.load_schema()))

    def test_render_names_every_property_of_every_object_definition(self):
        schema = prompt_schema.load_schema()
        rendered = prompt_schema.render_shape(schema)
        for name, definition in schema["definitions"].items():
            for key in (definition.get("properties") or {}):
                self.assertIn('"%s"' % key, rendered, "%s.%s missing from rendered shape" % (name, key))

    def test_render_states_every_enum_value(self):
        schema = prompt_schema.load_schema()
        rendered = prompt_schema.render_shape(schema)
        for values in prompt_schema.enums(schema).values():
            for value in values:
                self.assertIn("null" if value is None else str(value), rendered)


class SchemaFieldTest(unittest.TestCase):
    def test_property_names_shared_with_definition_names_are_not_lost(self):
        # Regression: subtracting definition names deleted genuine properties.
        declared = prompt_schema.schema_fields()
        for name in ("study", "proposition", "result", "conditions",
                     "qualifiers", "scope", "atomicity", "test"):
            self.assertIn(name, declared)
        self.assertNotIn("definitions", declared)

    def test_definition_only_names_are_not_reported_as_fields(self):
        declared = prompt_schema.schema_fields()
        # These are definition names only; the properties are observations/samples/statistics.
        for name in ("observation", "sample", "statistic", "slot", "binding"):
            self.assertNotIn(name, declared)


class SingleSourceTest(unittest.TestCase):
    def test_schema_wire_value_is_the_validators_wire_value(self):
        self.assertEqual(prompt_schema.wire_schema_const(), pilot_validation_v3.SCHEMA)

    def test_structured_result_required_equals_validator_fields_plus_envelope(self):
        required = set(prompt_schema.required_fields()["structured_result"])
        expected = set(pilot_validation_v3.FIELDS) | {"schema", "kind", "contrast", "atomicity", "test"}
        self.assertEqual(required, expected)

    def test_observation_required_covers_what_the_validators_read(self):
        required = prompt_schema.required_fields()["observation"]
        for key in ("statement", "quotes", "role", "proposition", "result", "statistics"):
            self.assertIn(key, required)

    def test_enums_match_the_schema(self):
        values = prompt_schema.enums()
        self.assertEqual(values["observation.role"][0], "primary_result")
        self.assertEqual(set(values["structured_result.kind"]),
                         {"between_groups", "within_group_change", "association", "interaction", "other"})
        self.assertEqual(set(values["result.significance"]),
                         {"reported_significant", "not_significant", "not_reported", "other"})
        self.assertEqual(set(values["test.status"]),
                         {"reported_significant", "reported_not_significant", "not_reported",
                          "derived", "unresolved"})


class DriftTest(unittest.TestCase):
    def test_missing_required_key_is_reported(self):
        prompt = '"statement" "quotes" "role" "proposition" "evidence_relation" ' \
                 '"scope_reason" "conditions" "result" "statistics" "limitations"'
        problems = prompt_schema.prompt_drift(prompt)
        omitted = {entry["definition"]: entry["keys"] for entry in problems
                   if entry["kind"] == "prompt_omits_required_keys"}
        self.assertIn("exposure", omitted["structured_result"])

    def test_declared_wire_value_is_not_drift(self):
        prompt = '"statement" "schema" "kind" "population" "exposure" "outcome" ' \
                 '"comparator" "timepoint" "disease_stage" "contrast" "atomicity" "test"'
        kinds = {entry["kind"] for entry in prompt_schema.prompt_drift(prompt)}
        self.assertNotIn("prompt_mentions_superseded_wire_value", kinds)

    def test_superseded_wire_value_is_drift(self):
        prompt = '"statement" "source_result_v2"'
        problems = prompt_schema.prompt_drift(prompt)
        superseded = [entry for entry in problems
                      if entry["kind"] == "prompt_mentions_superseded_wire_value"]
        self.assertEqual(superseded[0]["value"], "source_result_v2")
        self.assertEqual(superseded[0]["declared"], pilot_validation_v3.SCHEMA)

    def test_invented_field_is_reported_as_undefined(self):
        self.assertIn("severity", prompt_schema.undefined_field_mentions('"severity"'))

    def test_declared_fields_enums_and_wire_value_are_known_tokens(self):
        prompt = '"statement" "between_groups" "primary_result" "source_result_v3"'
        self.assertEqual(prompt_schema.undefined_field_mentions(prompt), [])

    def test_prose_only_mention_counts_as_missing_json_key(self):
        prompt = "Return structured_result for each result."
        self.assertIn("structured_result", prompt_schema.fields_without_json_key(prompt))


class EnvelopeShapeTest(unittest.TestCase):
    """The three base prompts use a ``JSON shape`` marker, not the amendment's."""

    def test_base_prompts_are_compared_not_reported_as_block_less(self):
        import paper_batch_model
        delta = prompt_schema.shape_delta_full(paper_batch_model.SYSTEM)
        self.assertIsNotNone(delta)
        paths = {entry["path"] for entry in delta}
        self.assertIn("observations[0]", paths)

    def test_nested_definition_keys_are_compared_through_refs(self):
        # Regression: nested $ref/oneOf was not resolved, so a nested omission
        # was invisible and the delta looked like "only structured_result".
        import paper_batch_model
        delta = prompt_schema.shape_delta_full(paper_batch_model.SYSTEM)
        nested = [entry for entry in delta if entry["path"] == "observations[0].proposition"]
        self.assertEqual(nested[0]["absent_from_prompt"], ["polarity", "qualifiers"])

    def test_live_base_prompt_delta_is_only_the_contract_key(self):
        delta = prompt_schema.shape_delta_full(paper_extract_v2.EXTRACTION_SYSTEM_V2)
        self.assertEqual([entry["absent_from_prompt"] for entry in delta], [["structured_result"]])

    def test_full_delta_is_none_without_a_block(self):
        self.assertIsNone(prompt_schema.shape_delta_full("no block here"))

    def test_public_shape_block_matches_the_private_parser(self):
        block = prompt_schema.shape_block(EXTRACTION_AMENDMENT_V3)
        self.assertEqual(block["schema"], pilot_validation_v3.SCHEMA)
        self.assertIsNone(prompt_schema.shape_block("no block here"))

    def test_replacement_is_a_behaviour_change_and_is_reported_as_one(self):
        sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))
        import check_schema_drift

        delta = check_schema_drift.replacement_delta()
        self.assertTrue(delta["behaviour_change"])
        self.assertNotEqual(delta["old_effective_sha256"], delta["new_effective_sha256_if_replaced"])
        # A naive replace restates the base prompt's own shape block.
        self.assertEqual(delta["definitions_the_render_would_restate"], ["study"])


class LivePromptTest(unittest.TestCase):
    def test_effective_prompt_is_complete_except_one_prose_only_field(self):
        effective = build_prompt(paper_extract_v2.EXTRACTION_SYSTEM_V2)
        # Pinned, not filtered: a new omission must show up as a test failure.
        self.assertEqual(prompt_schema.fields_without_json_key(effective), ["structured_result"])
        self.assertEqual(prompt_schema.prompt_drift(effective), [])

    def test_live_amendment_shape_block_states_exactly_the_schema(self):
        self.assertEqual(prompt_schema.shape_delta(EXTRACTION_AMENDMENT_V3), [])
        self.assertEqual(prompt_schema.shape_delta(build_prompt(paper_extract_v2.EXTRACTION_SYSTEM_V2)), [])

    def test_shape_delta_is_none_when_the_prompt_has_no_block(self):
        self.assertIsNone(prompt_schema.shape_delta("no block here"))

    def test_shape_delta_catches_a_changed_required_key(self):
        broken = EXTRACTION_AMENDMENT_V3.replace('"contrast":binding,', '"contrastx":binding,')
        self.assertNotEqual(prompt_schema.shape_delta(broken), [])

    def test_shape_delta_catches_a_changed_enum(self):
        broken = EXTRACTION_AMENDMENT_V3.replace('"single|unresolved"', '"single|merged"')
        diff = prompt_schema.shape_delta(broken)
        self.assertEqual([entry for entry in diff if entry["kind"] == "enum"][0]["path"],
                         "structured_result.atomicity.status")

    def test_shape_delta_catches_a_changed_wire_value(self):
        broken = EXTRACTION_AMENDMENT_V3.replace('"schema":"source_result_v3"',
                                                 '"schema":"source_result_v9"')
        diff = prompt_schema.shape_delta(broken)
        self.assertEqual([entry for entry in diff if entry["kind"] == "const"][0]["path"],
                         "structured_result.schema")

    def test_the_three_prompt_constants_are_distinct_texts(self):
        # They share an opening paragraph, so they must not be assumed
        # interchangeable: each is a separate edit, not a superset of another.
        import paper_batch_model
        import paper_pilot_batch
        texts = [paper_batch_model.SYSTEM, paper_pilot_batch.SYSTEM,
                 paper_extract_v2.EXTRACTION_SYSTEM_V2]
        self.assertEqual(len(set(texts)), 3)

    def test_effective_prompt_covers_every_field_the_legacy_route_names(self):
        # The one property that survived measurement: field-name coverage is a
        # superset, so nothing the legacy route names is lost by sharing it.
        import paper_batch_model
        effective = build_prompt(paper_extract_v2.EXTRACTION_SYSTEM_V2)
        legacy_missing = set(prompt_schema.fields_without_json_key(paper_batch_model.SYSTEM))
        effective_missing = set(prompt_schema.fields_without_json_key(effective))
        self.assertEqual(effective_missing - legacy_missing, set())
        self.assertTrue(effective_missing)


if __name__ == "__main__":
    unittest.main()
