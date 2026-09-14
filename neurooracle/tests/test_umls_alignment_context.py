from __future__ import annotations

import json

import pytest

from neurooracle.scripts.build_umls_simplification_candidate import build_candidate, cheap, file_sha
from neurooracle.scripts.review_umls_existing_alignment import (
    check_context_artifacts, collect_context, contextual_alias_checks, read_jsonl,
)
from neurooracle.tests.test_umls_simplification_candidate import source_graph


def make_frozen_candidate(source, output):
    build = build_candidate(source, output)
    freeze = {
        "status": "READ_ONLY_CANDIDATE_VALIDATED", "counts": build["counts"],
        "formal_sources": {"graph": build["source"]},
        "artifacts": {"knowledge_graph.candidate.json": build["candidate"],
                      "umls_details.sqlite": {**cheap(output / "umls_details.sqlite"), "sha256": file_sha(output / "umls_details.sqlite")}},
    }
    (output / "CANDIDATE_FREEZE.json").write_text(json.dumps(freeze), encoding="utf-8")
    return build


def test_context_collector_is_lossless_and_includes_canonical_claim_context(source_graph, tmp_path):
    source, graph = source_graph
    candidate, output = tmp_path / "candidate", tmp_path / "review"
    make_frozen_candidate(source, candidate)
    before = {path: (cheap(path), file_sha(path)) for path in (source, candidate / "knowledge_graph.candidate.json", candidate / "umls_details.sqlite")}
    result = collect_context(candidate, output)
    assert all(result["checks"].values())
    assert result["counts"]["atoms"] == 2
    assert result["counts"]["pairs"] == 2
    assert result["counts"]["parents"] == 1
    assert result["counts"]["linked_claims"] == 1
    assert list(read_jsonl(output / "CLAIMS.jsonl")) == [graph["concepts"]["CLM:c"]]
    assert check_context_artifacts(output)["status"] == "CONTEXT_COLLECTED_AND_VERIFIED"
    for path, fingerprint in before.items():
        assert (cheap(path), file_sha(path)) == fingerprint
    with pytest.raises(ValueError, match="protected"):
        collect_context(candidate, candidate / "review")
    with pytest.raises(FileExistsError):
        collect_context(candidate, output)


def test_collector_refuses_changed_source_before_creating_output(source_graph, tmp_path):
    source, _ = source_graph
    candidate, output = tmp_path / "candidate", tmp_path / "review"
    make_frozen_candidate(source, candidate)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protected source changed"):
        collect_context(candidate, output)
    assert not output.exists()


def test_context_claim_cues_are_not_generalized_to_every_ct_token():
    row = {"atom_id": "A", "target_id": "IF:cortical_thickness", "source_mention_id": "P",
           "parent_text": "CT and MRI", "target_definition": "cortical thickness", "atom_name": "CT",
           "context": {"claim_ids": ["C"]}}
    result = contextual_alias_checks([row], {"C": {"metadata": {"raw_text": "Computed tomography (CT) was compared with MRI."}}})
    assert result[0]["cues"] == ["computed_tomography_in_claim", "explicit_CT_expansion"]
    assert result[0]["graph_change_applied"] is False
    unknown = contextual_alias_checks([row], {"C": {"metadata": {"raw_text": "CT and MRI were compared."}}})
    assert unknown[0]["cues"] == []
    assert unknown[0]["claim_evidence"] == []
