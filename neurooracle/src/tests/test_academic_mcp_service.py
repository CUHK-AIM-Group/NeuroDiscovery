from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from neurooracle.mcp.academic_service import AcademicService
from neurooracle.scripts import collect_sparse_case_study_literature_v3 as collector
from neurooracle.src import academic_literature as literature


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class AcademicLiteratureTests(unittest.TestCase):
    def test_core_record_requires_expected_identity_and_complete_abstract(self) -> None:
        lite = {
            "source": "MED",
            "id": "123",
            "pmid": "123",
            "title": "A stable title",
            "pubYear": "2020",
        }
        candidate, status = literature.normalize_search_candidate(lite)
        self.assertEqual(status, "identity_ready")
        self.assertIsNotNone(candidate)
        core = {**lite, "abstractText": "complete evidence " * 30}
        record, status = literature.normalize_core_result(
            core, expected_candidate=candidate
        )
        self.assertEqual(status, "ready")
        self.assertGreaterEqual(record["abstract_characters"], 200)
        self.assertTrue(record["search_provenance_only"])
        mismatched, status = literature.normalize_core_result(
            {**core, "id": "999", "pmid": "999"},
            expected_candidate=candidate,
        )
        self.assertIsNone(mismatched)
        self.assertEqual(status, "provider_identity_mismatch")


class AcademicServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        formal = self.root / "neurooracle" / "data" / "full_v2"
        _write_json(formal / "knowledge_graph.json", {})
        _write_jsonl(
            formal / "extracted_claims.jsonl",
            [
                {
                    "paper_id": "pmid:1",
                    "pmid": "1",
                    "title": "Existing paper",
                    "year": 2020,
                }
            ],
        )
        _write_json(
            formal / "CURRENT_STATE.json",
            {
                "taxonomy_version": "case_study_membership.v2",
                "formal_kg_statistics": {
                    "general": {"papers": 1, "claims": 1},
                    "case_studies": {"brain_age": {"papers": 1, "claims": 1}},
                    "quality": {},
                    "counting_policy": "test",
                },
            },
        )
        self.service = AcademicService(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_formal_coverage_extends_an_older_installed_registry(self) -> None:
        with patch("neurooracle.mcp.academic_service.CASE_STUDIES", ()):
            result = self.service.list_case_studies()

        self.assertEqual(result["case_study_count"], 1)
        self.assertEqual(result["case_studies"][0]["id"], "brain_age")

    def test_search_deduplicates_lite_metadata_before_core_download(self) -> None:
        search_payload = {
            "hitCount": 2,
            "nextCursorMark": "next",
            "resultList": {
                "result": [
                    {
                        "source": "MED",
                        "id": "1",
                        "pmid": "1",
                        "title": "Existing paper",
                        "pubYear": "2020",
                    },
                    {
                        "source": "MED",
                        "id": "2",
                        "pmid": "2",
                        "title": "New paper",
                        "pubYear": "2021",
                    },
                ]
            },
        }
        core_calls: list[tuple[str, str]] = []

        def fake_core(source: str, source_id: str) -> dict[str, object]:
            core_calls.append((source, source_id))
            return {
                "source": source,
                "id": source_id,
                "pmid": source_id,
                "title": "New paper",
                "pubYear": "2021",
                "abstractText": "complete abstract evidence " * 20,
            }

        with patch.object(literature, "fetch_search_page", return_value=search_payload), patch.object(
            literature, "fetch_article_core", side_effect=fake_core
        ):
            result = self.service.search_literature(
                query="brain age",
                case_study_ids=("brain_age",),
                page_size=2,
                max_results=2,
            )

        self.assertEqual(core_calls, [("MED", "2")])
        self.assertEqual(result["returned_records"], 1)
        self.assertEqual(result["excluded_by_origin"], {"formal_kg": 1})
        self.assertFalse(result["case_study_membership_assigned"])

    def test_identity_index_skips_materialized_batch_copies(self) -> None:
        campaign = (
            self.root
            / "neurooracle"
            / "data"
            / "case_study_staging"
            / "campaign"
        )
        row = {
            "paper_id": "pmid:2",
            "pmid": "2",
            "title": "Staged paper",
            "year": 2021,
        }
        _write_jsonl(campaign / "abstracts_ready_for_extraction.jsonl", [row])
        _write_jsonl(campaign / "extraction_batches" / "batch_0001.jsonl", [row])
        result = self.service.check_paper_identity(pmid="2")
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["match"]["origin"], "staging")
        self.assertEqual(result["index_sync"]["source_files"], 2)

    def test_full_collection_validation_and_path_traversal_guard(self) -> None:
        campaign = (
            self.root
            / "neurooracle"
            / "data"
            / "case_study_staging"
            / "sealed"
        )
        rows = [
            {
                "paper_id": f"pmid:{number}",
                "pmid": str(number),
                "title": f"Paper {number}",
                "year": 2020 + number,
                "abstract": "complete abstract evidence " * 20,
                "status": "abstract_verified_ready_for_claim_extraction",
                "search_provenance_only": True,
                "kg_injection": False,
            }
            for number in (2, 3)
        ]
        queue = campaign / "abstracts_ready_for_extraction.jsonl"
        batch = campaign / "extraction_batches" / "batch_0001.jsonl"
        _write_jsonl(queue, rows)
        _write_jsonl(batch, rows)
        queue_hash = hashlib.sha256(queue.read_bytes()).hexdigest()
        snapshot = self.service.formal_snapshot()["files"]
        _write_json(
            campaign / "manifest.json",
            {
                "status": "complete",
                "years": {"start": 1980, "end": 2026},
                "collected_unique_papers": 2,
                "total_batches": 1,
                "kg_injection": False,
                "formal_kg_mutated": False,
                "formal_kg_files": snapshot,
                "output_sha256": {"ready_queue": queue_hash},
            },
        )
        _write_json(
            campaign / "COLLECTION_COMPLETE.json",
            {
                "status": "complete",
                "papers": 2,
                "batches": 1,
                "ready_queue_sha256": queue_hash,
                "kg_injection": False,
            },
        )
        result = self.service.validate_collection(collection_id="sealed", mode="full")
        self.assertTrue(result["valid_for_extraction"])
        self.assertEqual(result["validated_rows"], 2)
        with self.assertRaises(ValueError):
            self.service.get_collection_status("../full_v2")

    def test_collector_fetches_core_only_after_pre_abstract_dedup(self) -> None:
        class FakeDeduplicator:
            def __init__(self) -> None:
                self.excluded = 0
                self.stages: list[tuple[str, str]] = []

            def is_seen(self, record: object, **kwargs: object) -> bool:
                paper_id = str(record.get("paper_id"))
                stage = str(kwargs.get("stage"))
                self.stages.append((paper_id, stage))
                duplicate = paper_id == "pmid:1" and stage == "post_search_pre_abstract"
                self.excluded += int(duplicate)
                return duplicate

            def accept(self, _record: object, **_kwargs: object) -> bool:
                return True

            def flush(self) -> int:
                return 0

        target = collector.Target("brain_age", 16, 0, 1, ())
        query = collector.SearchQuery("test_query", "TITLE_ABS:brain")
        lite_results = [
            {
                "source": "MED",
                "id": value,
                "pmid": value,
                "title": title,
                "pubYear": year,
            }
            for value, title, year in (
                ("1", "Existing paper", "2020"),
                ("2", "New paper", "2021"),
            )
        ]
        core_calls: list[str] = []

        def fake_core(_source: str, source_id: str) -> dict[str, object]:
            core_calls.append(source_id)
            return {
                **lite_results[1],
                "abstractText": "complete abstract evidence " * 20,
            }

        output = self.root / "collector_test"
        state_path = output / "collection_state.json"
        papers_path = output / "abstracts_ready_for_extraction.jsonl"
        links_path = output / "links.jsonl"
        state = collector.load_state(state_path)
        fake_deduplicator = FakeDeduplicator()
        with patch.object(
            collector,
            "fetch_page",
            return_value={
                "hitCount": 2,
                "resultList": {"result": lite_results},
                "nextCursorMark": "",
            },
        ), patch.object(collector.academic, "fetch_article_core", side_effect=fake_core):
            collector.collect_query(
                target=target,
                query=query,
                state=state,
                state_path=state_path,
                papers_path=papers_path,
                links_path=links_path,
                alias_to_paper={},
                paper_cases=defaultdict(set),
                paper_queries=defaultdict(set),
                links=set(),
                primary_counts=Counter(),
                total_ref=[0],
                target_total=1,
                case_limit=1,
                page_size=2,
                deduplicator=fake_deduplicator,
            )

        self.assertEqual(core_calls, ["2"])
        rows = [json.loads(line) for line in papers_path.read_text().splitlines()]
        self.assertEqual([row["paper_id"] for row in rows], ["pmid:2"])
        self.assertIn(("pmid:1", "post_search_pre_abstract"), fake_deduplicator.stages)


if __name__ == "__main__":
    unittest.main()
