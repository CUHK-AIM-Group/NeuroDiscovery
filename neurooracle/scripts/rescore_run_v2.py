"""Offline re-score of a completed extraction run against the current validators.

Why this exists
---------------
The extraction run persists every raw response body and every accepted
candidate, but the *verdict* (structural pass, consistency conflicts) was
computed with whatever validator code was loaded at send time. When a validator
representational gap is found afterwards, re-sending every paper would double
spend for an answer we already have on disk. This script rebuilds each candidate
from the stored response body and re-evaluates it with the current
``paper_batch_model.validate_candidate``, ``pilot_validation.validator_limitations``
and ``pilot_validation_v2`` checks, writing a versioned report.

It never sends a request and never overwrites a run artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))
sys.path.insert(0, str(ROOT / "neurooracle" / "src"))

import paper_pilot_batch as batch  # noqa: E402
from paper_batch_model import validate_candidate  # noqa: E402
from pilot_validation import validator_limitations  # noqa: E402
import pilot_validation_v2 as pv2  # noqa: E402


def load_packets(pilot_dir):
    return {packet["work_key"]: packet
            for packet in batch.load_subset(pilot_dir / batch.SUBSET_NAME)}


def candidate_from_attempt(ledger_dir, attempt):
    """Recover the candidate object from the attempt's stored response body."""
    stem = attempt["attempt_id"][4:].split(":")[0]
    body = ledger_dir / "responses" / (stem + ".json")
    if not body.exists():
        return None
    try:
        envelope = json.loads(body.read_bytes().decode("utf-8", "replace"))
    except ValueError:
        return None
    content = (envelope.get("message") or {}).get("content") or envelope.get("content") or ""
    try:
        return batch.normalized_candidate(batch.extract_json_object(content))
    except (ValueError, TypeError):
        return None


# A paper that was never accepted can carry several HTTP 200 attempts, and none
# of them is canonical. The fair question is "did some enabled slot ever return a
# candidate that the current validators accept", so the best verdict wins.
_RANK = {"CANDIDATE_READY": 0, "HELD_CONSISTENCY": 1, "HELD_STRUCT": 2, "PARSE_FAIL": 3}


def _score_attempt(packet, ledger_dir, attempt):
    candidate = candidate_from_attempt(ledger_dir, attempt)
    if candidate is None:
        return {"status": "PARSE_FAIL", "observation_count": 0, "hard_errors": [],
                "validator_limitations": [], "fragment_errors": [],
                "consistency_conflicts": []}
    source = batch.shim_source(packet)
    structural = validate_candidate(candidate, source)
    limitations = validator_limitations(
        structural, candidate, source["source_snapshot"]["abstract"] or "")
    hard = [error for error in structural if error not in limitations]
    conflicts = pv2.candidate_consistency_errors(candidate)
    status = "CANDIDATE_READY"
    if hard:
        status = "HELD_STRUCT"
    elif conflicts:
        status = "HELD_CONSISTENCY"
    return {"status": status,
            "observation_count": len(candidate.get("observations") or []),
            "hard_errors": hard, "validator_limitations": limitations,
            "fragment_errors": batch.fragment_errors(candidate),
            "consistency_conflicts": conflicts}


def score(connection, packets, ledger_dir):
    attempts = [dict(row) for row in connection.execute("select * from attempts order by rowid")]
    by_paper = {}
    for attempt in attempts:
        if attempt["http_status"] != 200:
            continue
        by_paper.setdefault(attempt["work_key"], []).append(attempt)
    records = []
    for work_key, paper_attempts in sorted(by_paper.items()):
        packet = packets.get(work_key)
        if packet is None:
            continue
        scored = [_score_attempt(packet, ledger_dir, attempt) for attempt in paper_attempts]
        best = min(scored, key=lambda item: _RANK[item["status"]])
        best["work_key"] = work_key
        best["attempts_scored"] = len(scored)
        records.append(best)
    # A paper with no HTTP 200 body (unknown outcome or auth-rejected only) is
    # still part of the run denominator; it must not silently leave the report.
    scored_keys = {record["work_key"] for record in records}
    for work_key in sorted({attempt["work_key"] for attempt in attempts}):
        if work_key not in scored_keys:
            records.append({"work_key": work_key, "status": "NO_SUCCESSFUL_BODY",
                            "observation_count": 0, "hard_errors": [],
                            "validator_limitations": [], "fragment_errors": [],
                            "consistency_conflicts": [], "attempts_scored": 0})
    return sorted(records, key=lambda record: record["work_key"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir if args.pilot_dir.is_absolute() else ROOT / args.pilot_dir
    run_dir = pilot_dir / "run"
    ledger_dir = run_dir / args.run_tag
    packets = load_packets(pilot_dir)
    connection = sqlite3.connect(run_dir / ("PILOT_%s.sqlite" % args.run_tag))
    connection.row_factory = sqlite3.Row
    try:
        records = score(connection, packets, ledger_dir)
    finally:
        connection.close()

    counted = Counter(record["status"] for record in records)
    report = {
        "run_tag": args.run_tag,
        "papers_scored": len(records),
        "status_counts": dict(counted),
        "ready_rate": (round(counted["CANDIDATE_READY"] / len(records), 4) if records else None),
        "note": "Offline re-score with the current validators; no request was sent "
                "and no run artifact was overwritten.",
        "records": records,
    }
    out = args.out or (ledger_dir / "RESCORE_CURRENT.json")
    batch.write_json(out, report)
    print(json.dumps({"run_tag": args.run_tag, "papers": len(records),
                      "status_counts": dict(counted), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
