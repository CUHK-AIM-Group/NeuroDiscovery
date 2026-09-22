from __future__ import annotations

import json
import sqlite3
import sys
from types import SimpleNamespace
import urllib.error

import pytest

from neurooracle.scripts.benchmark_compact_case_study_reaudit import (
    expand_compact_response,
)
from neurooracle.scripts.host_review_full_graph_case_study_reaudit import (
    expand_compact_decisions,
    export_batch,
    import_batch,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    AUDIT_CONTRACT_VERSION,
    AUDIT_NAME,
    AUDIT_VERSION,
    claim_contract_fields,
)
from neurooracle.scripts.propagate_exact_case_study_reaudit import (
    semantic_fingerprint,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    PaperIdentityUnion,
    RUBRIC_VERSION,
    trusted_embedded_review,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS
from neurooracle.scripts.run_full_graph_case_study_reaudit import (
    GATE_NAMES,
    request_json,
    run,
    validate_response,
)
from neurooracle.scripts.run_compact_case_study_reaudit_production import (
    persist_validated_batch,
)
from neurooracle.scripts.merge_parallel_host_case_study_decisions import (
    task_semantic_fingerprints,
    validate_compact_review,
)


def _gates(**enabled: bool) -> dict[str, bool]:
    return {name: bool(enabled.get(name, False)) for name in GATE_NAMES}


def test_paper_identity_union_merges_late_bridge() -> None:
    identities = PaperIdentityUnion()
    identities.add(["doi:10.1/example"])
    identities.add(["title_year:a study|2025"])
    identities.add(["pmid:123", "doi:10.1/example", "title_year:a study|2025"])
    groups = identities.groups()
    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {
        "pmid:123",
        "doi:10.1/example",
        "title_year:a study|2025",
    }


def test_validate_response_accepts_nonexclusive_membership() -> None:
    expected = [("CLM:test", "", "")]
    result = {
        "reviews": [
            {
                "claim_id": "CLM:test",
                "claim_case_study_ids": [
                    "case1_transdiagnostic",
                    "biomarker_discovery",
                    "progression_prediction",
                    "prognosis",
                ],
                "confidence": 0.95,
                "reason": "Baseline disease-linked MRI predicts a later clinical outcome.",
                "needs_secondary_review": False,
                "gates": _gates(
                    neural_disease_change_verified=True,
                    longitudinal_verified=True,
                ),
            }
        ]
    }
    rows = validate_response(expected, result)
    assert rows[0]["claim_case_study_ids"] == [
        "case1_transdiagnostic",
        "case2_pathway_mediation",
        "biomarker_discovery",
        "progression_prediction",
        "prognosis",
    ]


def test_validate_response_rejects_standalone_case2_without_component() -> None:
    expected = [("CLM:test", "", "")]
    result = {
        "reviews": [
            {
                "claim_id": "CLM:test",
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "confidence": 0.9,
                "reason": "A purported Case 2 claim without a direct component label.",
                "needs_secondary_review": True,
                "gates": _gates(),
            }
        ]
    }
    with pytest.raises(ValueError, match="requires a direct"):
        validate_response(expected, result)


def test_validate_response_rejects_reordered_claims() -> None:
    expected = [("CLM:first", "", ""), ("CLM:second", "", "")]
    result = {
        "reviews": [
            {
                "claim_id": claim_id,
                "claim_case_study_ids": [],
                "confidence": 0.9,
                "reason": "No direct formal Case Study relation is supported here.",
                "needs_secondary_review": False,
                "gates": _gates(),
            }
            for claim_id in ("CLM:second", "CLM:first")
        ]
    }
    with pytest.raises(ValueError, match="order or IDs"):
        validate_response(expected, result)


def test_request_json_rejects_unknown_wire_api_before_network() -> None:
    with pytest.raises(ValueError, match="unsupported wire API"):
        request_json(
            base_url="http://localhost:1/v1",
            api_key="not-used",
            model="gpt-test",
            reasoning_effort="xhigh",
            wire_api="unknown",
            payload={"claims_to_review": []},
            timeout=0.01,
            retries=1,
        )


def test_responses_falls_back_when_proxy_rejects_text_format(monkeypatch) -> None:
    response_body = json.dumps(
        {
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": '{"reviews": []}'}
                    ]
                }
            ],
        }
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return response_body

    class FakeOpener:
        def __init__(self):
            self.payloads = []

        def open(self, request, timeout):
            del timeout
            self.payloads.append(json.loads(request.data.decode("utf-8")))
            if len(self.payloads) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, 502, "Bad Gateway", hdrs=None, fp=None
                )
            return FakeResponse()

    opener = FakeOpener()
    monkeypatch.setattr(
        "neurooracle.scripts.run_full_graph_case_study_reaudit.urllib.request.build_opener",
        lambda *_args: opener,
    )

    result, metadata = request_json(
        base_url="http://localhost:8080/v1",
        api_key="test-key",
        model="gpt-test",
        reasoning_effort="xhigh",
        wire_api="responses",
        payload={"claims_to_review": []},
        timeout=1,
        retries=1,
    )

    assert result == {"reviews": []}
    assert "text" in opener.payloads[0]
    assert "text" not in opener.payloads[1]
    assert metadata["structured_output_requested"] is False


def test_responses_plain_json_skips_text_format_probe(monkeypatch) -> None:
    response_body = json.dumps(
        {
            "model": "gpt-test",
            "status": "completed",
            "output_text": '{"reviews": []}',
        }
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return response_body

    class FakeOpener:
        def __init__(self):
            self.payloads = []

        def open(self, request, timeout):
            del timeout
            self.payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

    opener = FakeOpener()
    monkeypatch.setattr(
        "neurooracle.scripts.run_full_graph_case_study_reaudit.urllib.request.build_opener",
        lambda *_args: opener,
    )

    result, metadata = request_json(
        base_url="http://localhost:8080/v1",
        api_key="test-key",
        model="gpt-test",
        reasoning_effort="xhigh",
        wire_api="responses",
        payload={"claims_to_review": []},
        timeout=1,
        retries=1,
        responses_plain_json=True,
    )

    assert result == {"reviews": []}
    assert len(opener.payloads) == 1
    assert "text" not in opener.payloads[0]
    assert metadata["structured_output_requested"] is False


def test_openai_sdk_transport_uses_responses_api(monkeypatch) -> None:
    calls = []

    class FakeUsage:
        def model_dump(self):
            return {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text='{"reviews": []}',
                model="gpt-test",
                usage=FakeUsage(),
                status="completed",
                incomplete_details=None,
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    result, metadata = request_json(
        base_url="http://localhost:8080/v1",
        api_key="test-key",
        model="gpt-test",
        reasoning_effort="xhigh",
        wire_api="responses",
        payload={"claims_to_review": []},
        timeout=123,
        retries=4,
        transport="openai_sdk",
    )

    assert result == {"reviews": []}
    assert calls[0]["client"]["base_url"] == "http://localhost:8080/v1"
    assert calls[0]["client"]["max_retries"] == 3
    assert calls[1]["model"] == "gpt-test"
    assert calls[1]["reasoning"] == {"effort": "xhigh"}
    assert metadata["transport"] == "openai_sdk"
    assert metadata["usage"]["total_tokens"] == 30


def test_run_splits_failed_multi_claim_request(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "reaudit.sqlite"
    connection = sqlite3.connect(ledger)
    connection.executescript(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source_paper_json TEXT NOT NULL,
            abstract TEXT NOT NULL
        );
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            paper_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            graph_ordinal INTEGER NOT NULL,
            review_status TEXT NOT NULL,
            review_json TEXT,
            reviewed_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO papers VALUES (?, ?, ?)",
        ("paper:1", '{"title":"Test paper"}', "Test abstract"),
    )
    for ordinal, claim_id in enumerate(("CLM:first", "CLM:second")):
        connection.execute(
            "INSERT INTO claims VALUES (?, ?, ?, ?, 'pending', NULL, NULL)",
            (
                claim_id,
                "paper:1",
                json.dumps(
                    {
                        "id": claim_id,
                        "subject_name": "subject",
                        "predicate": "relates_to",
                        "object_name": "object",
                        "raw_text": "General neuroscience evidence.",
                    }
                ),
                ordinal,
            ),
        )
    connection.commit()
    connection.close()

    calls: list[list[str]] = []

    def fake_request_json(**kwargs):
        claim_ids = [
            row["claim_id"] for row in kwargs["payload"]["claims_to_review"]
        ]
        calls.append(claim_ids)
        if len(claim_ids) > 1:
            raise RuntimeError("synthetic multi-claim upstream failure")
        return (
            {
                "reviews": [
                    {
                        "claim_id": claim_ids[0],
                        "claim_case_study_ids": [],
                        "confidence": 0.95,
                        "reason": "No direct formal Case Study relation is supported.",
                        "needs_secondary_review": False,
                        "gates": _gates(),
                    }
                ]
            },
            {"usage": {"total_tokens": 10}},
        )

    monkeypatch.setattr(
        "neurooracle.scripts.run_full_graph_case_study_reaudit.request_json",
        fake_request_json,
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        base_url="http://localhost:1/v1",
        api_key_env="TEST_REAUDIT_KEY",
        model="gpt-test",
        wire_api="responses",
        transport="openai_sdk",
        reasoning_effort="high",
        workers=1,
        batch_size=2,
        max_claims=2,
        abstract_chars=1000,
        paper_context_limit=10,
        timeout=1,
        retries=1,
        wave_delay=0.0,
        responses_plain_json=False,
        fail_fast=True,
        split_failed_batches=True,
    )
    monkeypatch.setenv("TEST_REAUDIT_KEY", "test-key")

    report = run(args)

    assert calls == [["CLM:first", "CLM:second"], ["CLM:first"], ["CLM:second"]]
    assert report["completed_this_run"] == 2
    assert report["attempted_this_run"] == 2
    assert report["failures"] == []
    connection = sqlite3.connect(ledger)
    statuses = connection.execute(
        "SELECT review_status FROM claims ORDER BY graph_ordinal"
    ).fetchall()
    connection.close()
    assert statuses == [("final_complete",), ("final_complete",)]


def test_expand_compact_decisions_derives_gates(tmp_path) -> None:
    task = tmp_path / "task.json"
    decisions = tmp_path / "decisions.json"
    output = tmp_path / "result.json"
    task.write_text(
        '{"task_id":"task-1","claim_ids":["CLM:test"]}', encoding="utf-8"
    )
    decisions.write_text(
        """{
          "decisions": [{
            "claim_id": "CLM:test",
            "labels": ["case1_transdiagnostic", "imaging_genetics"],
            "confidence": 0.95,
            "reason": "A molecular factor is directly linked to disease brain imaging.",
            "gate_overrides": {"longitudinal_verified": true}
          }]
        }""",
        encoding="utf-8",
    )
    args = type(
        "Args", (), {"task": task, "decisions": decisions, "output": output}
    )()
    summary = expand_compact_decisions(args)
    result = __import__("json").loads(output.read_text(encoding="utf-8"))
    gates = result["reviews"][0]["gates"]
    assert summary["claims_expanded"] == 1
    assert gates["neural_disease_change_verified"] is True
    assert gates["genetic_to_neural_verified"] is True
    assert gates["longitudinal_verified"] is True
    assert set(gates) == set(GATE_NAMES)


def test_export_batch_offset_selects_disjoint_later_slice(tmp_path) -> None:
    ledger = tmp_path / "reaudit.sqlite"
    connection = sqlite3.connect(ledger)
    connection.executescript(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source_paper_json TEXT NOT NULL,
            abstract TEXT NOT NULL
        );
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            paper_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            graph_ordinal INTEGER NOT NULL,
            review_status TEXT NOT NULL,
            review_json TEXT,
            reviewed_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO papers VALUES (?, ?, ?)",
        ("paper:1", '{"title":"Offset test"}', "Test abstract"),
    )
    for ordinal in range(5):
        claim_id = f"CLM:{ordinal}"
        connection.execute(
            "INSERT INTO claims VALUES (?, ?, ?, ?, 'pending', NULL, NULL)",
            (
                claim_id,
                "paper:1",
                json.dumps(
                    {
                        "id": claim_id,
                        "subject_name": "subject",
                        "predicate": "relates_to",
                        "object_name": "object",
                        "raw_text": f"Evidence {ordinal}.",
                    }
                ),
                ordinal,
            ),
        )
    connection.commit()
    connection.close()

    output = tmp_path / "offset_task.json"
    summary = export_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            status="unreviewed",
            count=2,
            offset=2,
            duplicate_first=False,
            unique_semantic=False,
            abstract_chars=1000,
            paper_context_limit=10,
            output=output,
        )
    )

    task = json.loads(output.read_text(encoding="utf-8"))
    assert summary["claims"] == 2
    assert task["claim_ids"] == ["CLM:2", "CLM:3"]
    assert task["audit_contract"]["audit_contract_version"] == AUDIT_CONTRACT_VERSION
    assert set(task["audit_contract"]["claims"]) == {"CLM:2", "CLM:3"}


def test_expand_compact_decisions_rejects_reordered_claims(tmp_path) -> None:
    task = tmp_path / "task.json"
    decisions = tmp_path / "decisions.json"
    task.write_text(
        '{"task_id":"task-1","claim_ids":["CLM:first","CLM:second"]}',
        encoding="utf-8",
    )
    decisions.write_text(
        """{"decisions": [
          {"claim_id":"CLM:second","labels":[],"confidence":0.9,"reason":"No direct formal relation."},
          {"claim_id":"CLM:first","labels":[],"confidence":0.9,"reason":"No direct formal relation."}
        ]}""",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"task": task, "decisions": decisions, "output": tmp_path / "result.json"},
    )()
    with pytest.raises(ValueError, match="order or IDs"):
        expand_compact_decisions(args)


def test_semantic_fingerprint_ignores_id_and_old_routing() -> None:
    left = {
        "id": "CLM:left",
        "subject_name": " Hippocampal  Volume ",
        "predicate": "is_biomarker_of",
        "object_name": "Alzheimer Disease",
        "raw_text": "Lower hippocampal volume was observed.",
        "conditions": ["Older Adults"],
        "old_scope_reaudit": {"decision": "qualifies"},
    }
    right = {
        **left,
        "id": "CLM:right",
        "subject_name": "hippocampal volume",
        "old_scope_reaudit": {"decision": "does_not_qualify"},
    }
    assert semantic_fingerprint("pmid:1", left) == semantic_fingerprint(
        "pmid:1", right
    )


def test_semantic_fingerprint_requires_same_paper_and_triple() -> None:
    payload = {
        "subject_name": "hippocampus",
        "predicate": "is_biomarker_of",
        "object_name": "Alzheimer disease",
        "raw_text": "Hippocampal atrophy was associated with Alzheimer disease.",
    }
    base = semantic_fingerprint("pmid:1", payload)
    assert base != semantic_fingerprint("pmid:2", payload)
    assert base != semantic_fingerprint(
        "pmid:1", {**payload, "object_name": "Parkinson disease"}
    )


def test_audit_contract_is_content_addressed_but_model_independent() -> None:
    payload = {
        "id": "CLM:test",
        "subject_name": "hippocampal volume",
        "predicate": "is_biomarker_of",
        "object_name": "Alzheimer disease",
        "raw_text": "Lower hippocampal volume was observed in Alzheimer disease.",
        "old_scope_reaudit": {"reviewer_id": "old-model"},
    }
    labels = ["case1_transdiagnostic", "biomarker_discovery"]
    gates = _gates(neural_disease_change_verified=True)
    first = claim_contract_fields(
        paper_key="pmid:1",
        payload=payload,
        labels=labels,
        gates=gates,
        rubric_version=RUBRIC_VERSION,
        case_study_ids=CASE_STUDY_IDS,
    )
    rerun = claim_contract_fields(
        paper_key="pmid:1",
        payload={**payload, "old_scope_reaudit": {"reviewer_id": "new-model"}},
        labels=labels,
        gates=gates,
        rubric_version=RUBRIC_VERSION,
        case_study_ids=CASE_STUDY_IDS,
    )
    changed_evidence = claim_contract_fields(
        paper_key="pmid:1",
        payload={**payload, "raw_text": "Different evidence."},
        labels=labels,
        gates=gates,
        rubric_version=RUBRIC_VERSION,
        case_study_ids=CASE_STUDY_IDS,
    )
    assert first == rerun
    assert first["audit_key"] != changed_evidence["audit_key"]


def test_trusted_embedded_review_reuses_only_exact_matching_contract() -> None:
    payload = {
        "id": "CLM:test",
        "subject_name": "hippocampal volume",
        "predicate": "is_biomarker_of",
        "object_name": "Alzheimer disease",
        "raw_text": "Lower hippocampal volume was observed in Alzheimer disease.",
    }
    labels = ["case1_transdiagnostic", "biomarker_discovery"]
    gates = _gates(neural_disease_change_verified=True)
    contract = claim_contract_fields(
        paper_key="pmid:1",
        payload=payload,
        labels=labels,
        gates=gates,
        rubric_version=RUBRIC_VERSION,
        case_study_ids=CASE_STUDY_IDS,
    )
    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "decision": "finalized",
        "confidence": 0.95,
        "decision_basis": "Direct disease-linked neural marker.",
        "gates": gates,
        **contract,
    }
    trusted = trusted_embedded_review(
        claim_id="CLM:test",
        paper_key="pmid:1",
        payload={**payload, "old_scope_reaudit": audit},
        labels=labels,
    )
    changed = trusted_embedded_review(
        claim_id="CLM:test",
        paper_key="pmid:1",
        payload={**payload, "raw_text": "Changed evidence.", "old_scope_reaudit": audit},
        labels=labels,
    )
    assert trusted is not None
    assert trusted["audit_key"] == contract["audit_key"]
    assert changed is None


def _one_claim_host_task(tmp_path):
    ledger = tmp_path / "reaudit.sqlite"
    connection = sqlite3.connect(ledger)
    connection.executescript(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source_paper_json TEXT NOT NULL,
            abstract TEXT NOT NULL
        );
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            paper_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            graph_ordinal INTEGER NOT NULL,
            review_status TEXT NOT NULL,
            review_json TEXT,
            reviewed_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO papers VALUES (?, ?, ?)",
        ("pmid:1", '{"title":"Immutable import"}', "Test abstract"),
    )
    payload = {
        "id": "CLM:test",
        "subject_name": "hippocampus",
        "predicate": "is_biomarker_of",
        "object_name": "Alzheimer disease",
        "raw_text": "Hippocampal atrophy was observed in Alzheimer disease.",
    }
    connection.execute(
        "INSERT INTO claims VALUES (?, ?, ?, 0, 'pending', NULL, NULL)",
        ("CLM:test", "pmid:1", json.dumps(payload)),
    )
    connection.commit()
    connection.close()
    task_path = tmp_path / "task.json"
    export_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            status="unreviewed",
            count=1,
            offset=0,
            duplicate_first=False,
            unique_semantic=False,
            abstract_chars=1000,
            paper_context_limit=10,
            output=task_path,
        )
    )
    task = json.loads(task_path.read_text(encoding="utf-8"))
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "reviews": [
                    {
                        "claim_id": "CLM:test",
                        "claim_case_study_ids": [
                            "case1_transdiagnostic",
                            "biomarker_discovery",
                        ],
                        "confidence": 0.95,
                        "reason": "Direct disease-linked neural marker.",
                        "needs_secondary_review": False,
                        "gates": _gates(neural_disease_change_verified=True),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return ledger, task_path, result_path


def test_import_batch_refuses_to_overwrite_finalized_claim(tmp_path) -> None:
    ledger, task, result = _one_claim_host_task(tmp_path)
    connection = sqlite3.connect(ledger)
    connection.execute(
        "UPDATE claims SET review_status='final_complete', review_json='{}'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="already-reviewed"):
        import_batch(
            SimpleNamespace(output_dir=tmp_path, task=task, result=result)
        )


def test_import_batch_can_replace_finalized_claim_with_explicit_correction(
    tmp_path,
) -> None:
    ledger, task, result = _one_claim_host_task(tmp_path)
    import_batch(SimpleNamespace(output_dir=tmp_path, task=task, result=result))

    corrected = json.loads(result.read_text(encoding="utf-8"))
    corrected["reviews"][0]["claim_case_study_ids"] = ["case1_transdiagnostic"]
    result.write_text(json.dumps(corrected), encoding="utf-8")
    output = import_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            task=task,
            result=result,
            replace_reviewed=True,
            correction_reason="manual residual QA",
        )
    )

    connection = sqlite3.connect(ledger)
    status, review_json = connection.execute(
        "SELECT review_status, review_json FROM claims WHERE claim_id='CLM:test'"
    ).fetchone()
    connection.close()
    stored = json.loads(review_json)
    assert status == "final_complete"
    assert output["replace_reviewed"] is True
    assert stored["review_stage"] == "primary_correction"
    assert stored["reviewer_id"] == "codex_host_manual_correction"
    assert stored["correction_reason"] == "manual residual QA"
    assert stored["superseded_review_status"] == "final_complete"
    assert len(stored["superseded_review_sha256"]) == 64
    assert stored["claim_case_study_ids"] == ["case1_transdiagnostic"]


def test_import_batch_replace_reviewed_requires_reason(tmp_path) -> None:
    _ledger, task, result = _one_claim_host_task(tmp_path)
    with pytest.raises(ValueError, match="correction reason"):
        import_batch(
            SimpleNamespace(
                output_dir=tmp_path,
                task=task,
                result=result,
                replace_reviewed=True,
                correction_reason="",
            )
        )


def test_import_batch_can_correct_finalized_secondary_adjudication(tmp_path) -> None:
    ledger, primary_task, primary_result = _one_claim_host_task(tmp_path)
    primary_payload = json.loads(primary_result.read_text(encoding="utf-8"))
    primary_payload["reviews"][0]["needs_secondary_review"] = True
    primary_result.write_text(json.dumps(primary_payload), encoding="utf-8")
    import_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            task=primary_task,
            result=primary_result,
        )
    )

    secondary_task = tmp_path / "secondary_task.json"
    export_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            status="secondary_pending",
            count=1,
            offset=0,
            duplicate_first=False,
            unique_semantic=False,
            abstract_chars=1000,
            paper_context_limit=10,
            output=secondary_task,
        )
    )
    task_payload = json.loads(secondary_task.read_text(encoding="utf-8"))
    secondary_result = tmp_path / "secondary_result.json"
    adjudication = json.loads(primary_result.read_text(encoding="utf-8"))
    adjudication["task_id"] = task_payload["task_id"]
    adjudication["reviews"][0]["needs_secondary_review"] = False
    secondary_result.write_text(json.dumps(adjudication), encoding="utf-8")
    import_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            task=secondary_task,
            result=secondary_result,
        )
    )

    corrected = json.loads(secondary_result.read_text(encoding="utf-8"))
    corrected["reviews"][0]["claim_case_study_ids"] = ["case1_transdiagnostic"]
    secondary_result.write_text(json.dumps(corrected), encoding="utf-8")
    output = import_batch(
        SimpleNamespace(
            output_dir=tmp_path,
            task=secondary_task,
            result=secondary_result,
            replace_reviewed=True,
            correction_reason="secondary boundary hygiene",
        )
    )

    connection = sqlite3.connect(ledger)
    status, review_json = connection.execute(
        "SELECT review_status, review_json FROM claims WHERE claim_id='CLM:test'"
    ).fetchone()
    connection.close()
    stored = json.loads(review_json)
    assert status == "final_complete"
    assert output["replace_reviewed"] is True
    assert output["reviewer_id"] == "codex_host_manual_secondary_correction"
    assert stored["review_stage"] == "secondary_correction"
    assert stored["reviewer_id"] == "codex_host_manual_secondary_correction"
    assert stored["correction_reason"] == "secondary boundary hygiene"
    assert stored["superseded_review_status"] == "final_complete"
    assert len(stored["superseded_review_sha256"]) == 64
    assert stored["claim_case_study_ids"] == ["case1_transdiagnostic"]
    assert stored["primary_review"]["review_stage"] == "primary"


def test_import_batch_refuses_changed_evidence_after_export(tmp_path) -> None:
    ledger, task, result = _one_claim_host_task(tmp_path)
    connection = sqlite3.connect(ledger)
    payload = json.loads(
        connection.execute(
            "SELECT payload_json FROM claims WHERE claim_id='CLM:test'"
        ).fetchone()[0]
    )
    payload["raw_text"] = "Evidence changed after export."
    connection.execute(
        "UPDATE claims SET payload_json=? WHERE claim_id='CLM:test'",
        (json.dumps(payload),),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="evidence changed"):
        import_batch(
            SimpleNamespace(output_dir=tmp_path, task=task, result=result)
        )


def test_expand_compact_response_restores_full_validated_schema() -> None:
    expected = [("CLM:compact", "paper:1", "{}")]
    result = {
        "b": "token",
        "r": [[
            0,
            [0, 2, 5],
            96,
            (1 << GATE_NAMES.index("neural_disease_change_verified"))
            | (1 << GATE_NAMES.index("genetic_to_neural_verified")),
            False,
            "direct genetic imaging disease marker",
        ]],
    }
    reviews = expand_compact_response(expected, result, batch_token="token")
    assert reviews[0]["claim_case_study_ids"] == [
        "case1_transdiagnostic",
        "case2_pathway_mediation",
        "biomarker_discovery",
        "imaging_genetics",
    ]
    assert reviews[0]["confidence"] == 0.96
    assert reviews[0]["gates"]["neural_disease_change_verified"] is True
    assert reviews[0]["gates"]["genetic_to_neural_verified"] is True


def test_expand_compact_response_rejects_wrong_batch_token() -> None:
    expected = [("CLM:compact", "paper:1", "{}")]
    with pytest.raises(ValueError, match="batch token"):
        expand_compact_response(
            expected,
            {"b": "wrong", "r": [[0, [], 95, 0, False, "general only"]]},
            batch_token="expected",
        )


def test_expand_compact_response_rejects_missing_mandatory_gate() -> None:
    expected = [("CLM:compact", "paper:1", "{}")]
    with pytest.raises(ValueError, match="mandatory gate"):
        expand_compact_response(
            expected,
            {"b": "token", "r": [[0, [15], 95, 0, False, "brain age construct"]]},
            batch_token="token",
        )


def test_validate_response_rejects_string_booleans() -> None:
    expected = [("CLM:test", "", "")]
    result = {
        "reviews": [
            {
                "claim_id": "CLM:test",
                "claim_case_study_ids": [],
                "confidence": 0.9,
                "reason": "No direct Case Study evidence.",
                "needs_secondary_review": "false",
                "gates": _gates(),
            }
        ]
    }
    with pytest.raises(ValueError, match="JSON boolean"):
        validate_response(expected, result)


def test_compact_production_writer_is_conditional_and_contract_sealed(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute(
        """
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            paper_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            graph_ordinal INTEGER NOT NULL,
            review_status TEXT NOT NULL,
            review_json TEXT,
            reviewed_at TEXT
        )
        """
    )
    payload = {
        "id": "CLM:writer",
        "subject_name": "hippocampal volume",
        "predicate": "associated_with",
        "object_name": "Alzheimer disease",
        "raw_text": "Hippocampal volume was associated with Alzheimer disease.",
        "negated": False,
    }
    row = ("CLM:writer", "pmid:1", json.dumps(payload))
    connection.execute(
        "INSERT INTO claims VALUES (?, ?, ?, 0, 'pending', NULL, NULL)", row
    )
    connection.commit()
    review = {
        "claim_id": "CLM:writer",
        "claim_case_study_ids": ["case1_transdiagnostic", "biomarker_discovery"],
        "confidence": 0.95,
        "reason": "Direct disease-linked neural imaging marker.",
        "needs_secondary_review": False,
        "gates": _gates(neural_disease_change_verified=True),
    }
    counts = persist_validated_batch(
        connection,
        [row],
        [review],
        model="gpt-5.5",
        reasoning_effort="high",
        reviewer_id="model:gpt-5.5:compact-test",
        api_metadata={},
        task_payload_sha256="a" * 64,
    )
    assert counts == {"final_complete": 1, "secondary_pending": 0}
    status, review_json = connection.execute(
        "SELECT review_status, review_json FROM claims WHERE claim_id='CLM:writer'"
    ).fetchone()
    stored = json.loads(review_json)
    assert status == "final_complete"
    assert stored["audit_contract_version"] == AUDIT_CONTRACT_VERSION
    assert stored["audit_key"]
    assert stored["decision_sha256"]
    with pytest.raises(RuntimeError, match="conditional ledger write"):
        persist_validated_batch(
            connection,
            [row],
            [review],
            model="gpt-5.5",
            reasoning_effort="high",
            reviewer_id="model:gpt-5.5:compact-test",
            api_metadata={},
            task_payload_sha256="a" * 64,
        )
    connection.close()


def test_parallel_shard_validation_rejects_non_boolean_gate_override() -> None:
    with pytest.raises(ValueError, match="JSON booleans"):
        validate_compact_review(
            {
                "claim_id": "CLM:test",
                "claim_case_study_ids": ["imaging_genetics"],
                "confidence": 0.9,
                "reason": "Direct genetic factor to imaging phenotype evidence.",
                "gate_overrides": {"genetic_to_neural_verified": "true"},
            }
        )


def test_parallel_shard_validation_preserves_secondary_review_flag() -> None:
    review = validate_compact_review(
        {
            "claim_id": "CLM:test",
            "claim_case_study_ids": ["imaging_genetics"],
            "confidence": 0.9,
            "reason": "Direct genetic factor to imaging phenotype evidence.",
            "needs_secondary_review": True,
        }
    )
    assert review["needs_secondary_review"] is True


def test_parallel_shard_validation_rejects_non_boolean_secondary_flag() -> None:
    with pytest.raises(ValueError, match="needs_secondary_review must be a JSON boolean"):
        validate_compact_review(
            {
                "claim_id": "CLM:test",
                "claim_case_study_ids": ["imaging_genetics"],
                "confidence": 0.9,
                "reason": "Direct genetic factor to imaging phenotype evidence.",
                "needs_secondary_review": "true",
            }
        )


def test_parallel_task_fingerprint_ignores_claim_id_but_keeps_paper() -> None:
    task = {
        "audit_payload": {
            "claims_to_review": [
                {
                    "claim_id": "CLM:a",
                    "paper_ref": "P1",
                    "subject": "APOE",
                    "predicate": "associated_with",
                    "object": "hippocampal volume",
                    "raw_text": "APOE was associated with hippocampal volume.",
                    "conditions": [],
                },
                {
                    "claim_id": "CLM:b",
                    "paper_ref": "P1",
                    "subject": " APOE ",
                    "predicate": "associated_with",
                    "object": "hippocampal volume",
                    "raw_text": "APOE was associated with hippocampal volume.",
                    "conditions": [],
                },
                {
                    "claim_id": "CLM:c",
                    "paper_ref": "P2",
                    "subject": "APOE",
                    "predicate": "associated_with",
                    "object": "hippocampal volume",
                    "raw_text": "APOE was associated with hippocampal volume.",
                    "conditions": [],
                },
            ]
        }
    }
    fingerprints = task_semantic_fingerprints(task)
    assert fingerprints["CLM:a"] == fingerprints["CLM:b"]
    assert fingerprints["CLM:a"] != fingerprints["CLM:c"]
