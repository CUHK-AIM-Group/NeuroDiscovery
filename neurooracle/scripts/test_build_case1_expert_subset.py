from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).with_name("build_case1_expert_subset.py")
_SPEC = importlib.util.spec_from_file_location("case1_subset_builder_test", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
enrich_literature = _MODULE.enrich_literature
enrich_reference_abstracts = _MODULE.enrich_reference_abstracts
enrich_reference_links = _MODULE.enrich_reference_links
single_disease_case1_candidates_from = _MODULE.single_disease_case1_candidates_from
collapse_semantic_duplicates = _MODULE.collapse_semantic_duplicates
annotate_directional_hypothesis = _MODULE.annotate_directional_hypothesis


class LiteratureEnrichmentTests(unittest.TestCase):
    def test_adds_five_titled_references_with_source_excerpts(self) -> None:
        candidate = {
            "id": "hyp-1",
            "hypothesis_type": "case1_candidate",
            "path": [
                {
                    "from_name": "Bipolar Disorder",
                    "to_name": "Fusiform Gyrus | CortThick",
                    "confidence": 0.9,
                    "raw_text": "Direct fusiform finding in bipolar disorder.",
                    "source_paper": {"title": "Direct paper", "pmid": "1"},
                    "evidence": {"candidate_feature_id": "CortThick"},
                }
            ],
            "metadata": {
                "candidate_tuple": {
                    "diseases": ["Bipolar Disorder"],
                    "region_name": "Fusiform Gyrus",
                    "feature_name": "CortThick",
                }
            },
        }
        claims = []
        for index in range(2, 6):
            claims.append(
                {
                    "disease": "Bipolar Disorder",
                    "subject_name": "Fusiform cortical thickness",
                    "raw_text": f"Relevant fusiform sentence {index} in bipolar disorder.",
                    "confidence": 0.8,
                    "source_paper": {"title": f"Relevant paper {index}", "pmid": str(index)},
                }
            )
        claims.append(
            {
                "disease": "Alzheimer disease",
                "subject_name": "Fusiform cortical thickness",
                "raw_text": "An unrelated disease sentence.",
                "source_paper": {"title": "Unrelated paper", "pmid": "9"},
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            claims_path = Path(temp_dir) / "claims.jsonl"
            claims_path.write_text(
                "".join(json.dumps(item) + "\n" for item in claims),
                encoding="utf-8",
            )
            summary = enrich_literature([candidate], claims_path, max_references=5)

        self.assertEqual(summary["minimum_references"], 5)
        self.assertEqual(len(candidate["literature"]), 5)
        self.assertEqual(candidate["literature"][0]["title"], "Direct paper")
        self.assertNotIn("Unrelated paper", {item["title"] for item in candidate["literature"]})
        self.assertTrue(all(1 <= len(item["excerpts"]) <= 2 for item in candidate["literature"]))

    def test_enriches_abstracts_from_local_cache_without_network(self) -> None:
        candidate = {
            "literature": [
                {"title": "A cached paper", "pmid": "123", "excerpts": ["Relevant sentence."]},
                {"title": "A missing paper", "pmid": "456", "excerpts": ["Another sentence."]},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "abstracts.jsonl"
            cache_path.write_text(
                json.dumps(
                    {
                        "pmid": "123",
                        "abstract": "The complete cached abstract.",
                        "paper": {"title": "A cached paper", "pmid": "123"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = enrich_reference_abstracts(
                [candidate],
                abstract_sources=[cache_path],
                fetch_missing=False,
                output_path=Path(temp_dir) / "network-cache.jsonl",
            )

        self.assertEqual(summary["references_with_abstract"], 1)
        self.assertEqual(candidate["literature"][0]["abstract"], "The complete cached abstract.")
        self.assertEqual(candidate["literature"][0]["abstract_status"], "available")
        self.assertEqual(candidate["literature"][1]["abstract_status"], "missing")

    def test_prefers_direct_paper_links_and_keeps_pubmed_fallback(self) -> None:
        candidate = {
            "literature": [
                {"title": "Publisher paper", "pmid": "123", "doi": "10.1000/example"},
                {"title": "PubMed-only paper", "pmid": "456"},
                {
                    "title": "Open-access paper",
                    "pmid": "789",
                    "doi": "10.1000/open",
                    "pmcid": "PMC789",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = enrich_reference_links(
                [candidate],
                fetch_missing=False,
                output_path=Path(temp_dir) / "links.jsonl",
            )
        papers = candidate["literature"]
        self.assertEqual(papers[0]["direct_url"], "https://doi.org/10.1000/example")
        self.assertEqual(
            papers[1]["direct_url"], "https://pubmed.ncbi.nlm.nih.gov/456/"
        )
        self.assertEqual(
            papers[2]["direct_url"], "https://pmc.ncbi.nlm.nih.gov/articles/PMC789/"
        )
        self.assertEqual(summary["references_with_direct_url"], 3)

    def test_rejects_legacy_multi_disease_candidates(self) -> None:
        legacy = {
            "hypotheses": [
                {
                    "id": "legacy-cluster",
                    "hypothesis_type": "case1_candidate",
                    "metadata": {
                        "candidate_tuple": {
                            "diseases": ["Disease A", "Disease B", "Disease C"],
                            "region_id": "ROI:1",
                            "feature_id": "roi_alff",
                        }
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one disease x one ROI x one feature"):
                single_disease_case1_candidates_from(path)

    def test_collapses_same_hypothesis_from_multiple_generation_methods(self) -> None:
        def candidate(identifier: str, method: str, evidence: float) -> dict:
            return {
                "id": identifier,
                "hypothesis_type": "case1_candidate",
                "evidence_score": evidence,
                "metadata": {
                    "generation_method": method,
                    "candidate_tuple": {
                        "disease_ids": ["CUI:C0005586"],
                        "region_id": "NN:NN_TAL:10041",
                        "feature_id": "roi_participation_coefficient",
                        "direction": "none; infer sign during validation",
                    },
                },
            }

        collapsed, summary = collapse_semantic_duplicates(
            [
                candidate("brainstorm-copy", "llm_brainstorm", 0.2),
                candidate("neurodiscovery-copy", "neurodiscovery", 0.8),
            ],
            rank_key=lambda item: (-float(item.get("evidence_score") or 0), str(item["id"])),
        )
        self.assertEqual([item["id"] for item in collapsed], ["neurodiscovery-copy"])
        self.assertEqual(summary["duplicate_records_removed"], 1)
        metadata = collapsed[0]["metadata"]
        self.assertEqual(
            metadata["semantic_candidate_ids"],
            ["brainstorm-copy", "neurodiscovery-copy"],
        )
        self.assertEqual(metadata["generation_methods"], ["llm_brainstorm", "neurodiscovery"])

    def test_upgrades_direction_neutral_candidate_to_falsifiable_hypothesis(self) -> None:
        candidate = {
            "id": "case1-test",
            "hypothesis_type": "case1_candidate",
            "source_id": "D:SCZ",
            "source_name": "Schizophrenia",
            "target_name": "Thalamus | ROI participation coefficient",
            "path": [],
            "explanation": (
                "Test whether ROI participation coefficient in Thalamus differs for "
                "Schizophrenia patients. The generator does not assume direction."
            ),
            "metadata": {
                "direction_assumption": "none; infer sign during validation",
                "candidate_tuple": {
                    "disease_id": "D:SCZ",
                    "disease_name": "Schizophrenia",
                    "disease_ids": ["D:SCZ"],
                    "diseases": ["Schizophrenia"],
                    "region_id": "ROI:THALAMUS",
                    "region_name": "Thalamus",
                    "feature_id": "roi_participation_coefficient",
                    "feature_name": "ROI participation coefficient",
                },
            },
        }

        result = annotate_directional_hypothesis(candidate)

        direction = result["metadata"]["candidate_tuple"]["direction"]
        self.assertIn(direction, {"increase", "decrease"})
        self.assertEqual(result["metadata"]["direction_assumption"], direction)
        self.assertEqual(result["metadata"]["direction_source"], "generator_directional_prior")
        self.assertIn("is associated with", result["title"])
        self.assertIn("are hypothesized to show", result["summary"])
        self.assertTrue("increased" in result["summary"] or "decreased" in result["summary"])
        self.assertNotIn("differs", result["summary"])
        self.assertNotIn("infer sign", result["summary"])


if __name__ == "__main__":
    unittest.main()
