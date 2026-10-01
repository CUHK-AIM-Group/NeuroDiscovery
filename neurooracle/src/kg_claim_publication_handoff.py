"""Review handoff and a narrow adapter to the existing supplemental publisher.

Model suggestions never create host reading declarations. Publication actions
must come from a separately supplied, explicitly approved source-review ledger.
The existing identity/publication/counting implementation remains authoritative.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import tempfile

from neurooracle.src import kg_claim_compact_protocol as compact
from neurooracle.src import kg_claim_support_contract as contract

VERSION = "kg.claim_publication_handoff.v1"


def review_queue(store):
    """Small in-memory view for inspection; bulk exports use export_reviews."""
    papers, claims, cases = {}, {}, {}
    for case, con in store.cases():
        source_key, claim_key = contract.digest(case["source"]), contract.digest(con)
        papers[source_key] = case["source"]
        claims[claim_key] = con
        cases[case["case_id"]] = {"source_sha256": source_key, "contract_sha256": claim_key}
    results = {r["case_id"]: r for r in store.results()}
    groups = {}
    for row in store.db.execute("SELECT id,state,error FROM cases ORDER BY id"):
        binding = cases[row["id"]]
        result = results.get(row["id"])
        item = {"case_id": row["id"], **binding, "capture_state": row["state"],
                "capture_error": row["error"], "suggestion": result,
                "review_status": "source_adjudication_required" if result else "no_usable_answer",
                "publication_action": None}
        groups.setdefault(binding["contract_sha256"], []).append(item)
    return {"version": VERSION, "papers": papers, "claims": claims, "groups": groups,
            "case_count": len(cases), "summary": store.summary(),
            "automatic_approvals": 0, "production_writes": 0,
            "note": "Software capture and model suggestions are not scientific approval. Missing, invalid and unknown-transport cases remain in this handoff."}


def _jsonl(path, records):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    count, sha = 0, hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as stream:
            for record in records:
                line = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                stream.write(line); sha.update(line); count += 1
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        return {"path": str(path.resolve()), "sha256": sha.hexdigest(), "rows": count}
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()


def export_reviews(store, directory):
    """Stream a consistent DB snapshot with bounded memory and a manifest last.

    Paper/contract text appears once in separate keyed files. Normal progress
    uses counters; this full export is an explicit review/delivery boundary.
    """
    out = Path(directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    def documents(table):
        for row in store.db.execute(f"SELECT sha,payload FROM {table} ORDER BY sha"):
            value = json.loads(row["payload"])
            compact.require(contract.digest(value) == row["sha"], "stored review document changed")
            yield {"sha256": row["sha"], "value": value}
    def entries():
        for row in store.db.execute("SELECT id,paper_sha,contract_sha,state,error,result FROM cases ORDER BY id"):
            result = json.loads(row["result"]) if row["result"] is not None else None
            if result is not None:
                compact.require(result["case_id"] == row["id"] and result["source_sha256"] == row["paper_sha"]
                    and result["contract_sha256"] == row["contract_sha"] and result["raw_result_sha256"] == contract.digest(result["raw_result"]),
                    "stored review result binding changed")
            yield {"case_id": row["id"], "source_sha256": row["paper_sha"], "contract_sha256": row["contract_sha"],
                   "capture_state": row["state"], "capture_error": row["error"], "suggestion": result,
                   "review_status": "source_adjudication_required" if result else "no_usable_answer",
                   "publication_action": None}
    with store.db:
        store.db.execute("BEGIN")
        summary = store.summary()
        files = {"papers": _jsonl(out / "REVIEW_PAPERS.jsonl", documents("papers")),
                 "contracts": _jsonl(out / "REVIEW_CONTRACTS.jsonl", documents("contracts")),
                 "cases": _jsonl(out / "REVIEW_CASES.jsonl", entries())}
    compact.require(files["papers"]["rows"] == summary["papers"] and files["contracts"]["rows"] == summary["contracts"]
                    and files["cases"]["rows"] == summary["requested_cases"], "review export counters differ")
    manifest = {"version": VERSION, "format": "source_deduplicated_jsonl", "files": files,
                "summary": summary, "automatic_approvals": 0, "production_writes": 0,
                "reading": "Iterate actual file lines, not Unicode splitlines. Join case source_sha256 and contract_sha256 to keyed files.",
                "review_status": "All suggestions still need explicit source adjudication; missing work remains listed."}
    # The same atomic writer is used for the tiny manifest, without loading
    # any of the three potentially large JSONL files back into memory.
    _jsonl(out / "REVIEW_QUEUE.json", iter([manifest]))
    return manifest


def approved_supplemental_observation(*, case, claim, suggestion, packet, group_id,
                                       approved_ledger, expected_ledger_sha256, registry):
    """Pass an explicit host source action through the real publisher helper.

    The caller supplies a trusted ledger binding, not a model-produced approval.
    Historical canonical-only reviews may be replayed unchanged. Newly scoped
    contracts additionally require the ledger to bind their exact contract hash.
    This creates an in-memory observation; it never adopts a graph version.
    """
    from core.web.claim_layer_support_v1 import make_observation

    compact.require(contract.digest(approved_ledger) == expected_ledger_sha256, "approved ledger changed")
    compact.require(group_id in approved_ledger and packet["group_id"] == group_id, "review group mismatch")
    decision = approved_ledger[group_id]
    compact.require(not decision.get("hold") and decision.get("adjudicator") == "current_task_host", "explicit host source approval required")
    compact.require(decision.get("reviewer", {}).get("role") == "current_task_host", "reviewer provenance")
    compact.require(bool(decision.get("reading_provenance")), "source reading provenance required")
    compact.require(decision.get("packet_sha256") == contract.digest(packet), "review packet changed")
    compact.require(packet.get("complete_current_relation_closure") is True, "original relation closure required")
    compact.require(set(decision["original_claim_ids"]) == {o["claim_id"] for o in packet["observations"]}, "original observation closure differs")
    contract.validate_contract(claim)
    compact.require(case["proposition"] == claim["proposition"], "case contract mismatch")
    canonical = decision["canonical_claim"]
    compact.require({k: canonical[k] for k in ("subject_name", "predicate", "object_name")} == claim["proposition"], "reviewed canonical differs")
    if claim != contract.contract(claim["proposition"]):
        compact.require(decision.get("contract_bindings", {}).get(case["case_id"]) == contract.digest(claim), "additional scope lacks explicit review binding")
    compact.require(suggestion["case_id"] == case["case_id"] and suggestion["source_sha256"] == contract.digest(case["source"])
                    and suggestion["contract_sha256"] == contract.digest(claim), "suggestion input changed")
    normalized = compact.normalize(suggestion["raw_result"], case, claim, input_format=suggestion["input_format"])
    compact.require(normalized == suggestion, "suggestion changed")
    compact.require(suggestion["review_priority"] == "support" and suggestion["full_support_suggestion"], "suggestion needs scope resolution")
    pmid = str(case["source"]["pmid"])
    doc = packet["owning_sources"][pmid]
    compact.require(all(doc.get(key) == value for key, value in case["source"].items()), "owning document differs from model source")
    compact.require(pmid in decision["owning_abstracts_read"], "owning abstract not explicitly reviewed")
    compact.require(decision.get("owning_source_abstract_hashes", {}).get(pmid) == contract.digest(doc["abstract"]), "reviewed abstract changed")
    action = decision.get("supplemental_sources", {}).get(pmid)
    compact.require(isinstance(action, dict) and action.get("support") == "supports", "no approved supplemental action")
    compact.require(action.get("own_abstract_sha256") == contract.digest(doc["abstract"]), "action abstract changed")
    compact.require(contract.locate_quote(case["source"], action.get("anchor")), "approved anchor is not literal source text")
    return make_observation(decision["target_shared_claim_id"], canonical,
                            decision["original_claim_ids"], doc, deepcopy(action),
                            decision["review_id"], decision["note"],
                            decision["reading_provenance"], registry)
