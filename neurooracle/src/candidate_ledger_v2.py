"""Load accepted extraction candidates from the attempt ledger.

The first loader read responses by job and kept whichever attempt it saw last,
so a failed predecessor could overwrite an accepted result, and one job could
silently yield a different candidate than the pair packets referenced. This
loader selects eligible attempts per job, refuses ambiguity, and binds each
candidate to its recorded work, source and input digests.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3

ELIGIBLE_STATUSES = frozenset({"CANDIDATE_READY"})
REQUIRED_FINISH_REASON = "stop"


def parse_validation(row):
    raw = row.get("validation_json")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def attempt_eligible(row):
    """Return (eligible, reason). Only complete, error-free attempts qualify."""
    if row.get("status") not in ELIGIBLE_STATUSES:
        return False, "status_%s" % row.get("status")
    if not row.get("candidate_path"):
        return False, "no_candidate_path"
    if row.get("finish_reason") != REQUIRED_FINISH_REASON:
        return False, "finish_reason_%s" % row.get("finish_reason")
    validation = parse_validation(row)
    if validation is None:
        return False, "no_validation_record"
    if "hard_errors" not in validation:
        return False, "validation_missing_hard_errors"
    if validation.get("hard_errors"):
        return False, "hard_errors"
    return True, None


def resolve_candidate_path(root, value):
    return Path(root) / Path(str(value).replace("\\", "/"))


def load_accepted_candidates(db_path, root, packets_by_job=None):
    """Return accepted candidates plus explicit rejection and ambiguity records."""
    connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("select * from attempts order by started_at")]
    finally:
        connection.close()

    eligible, rejected = {}, []
    for row in rows:
        eligible_attempt, reason = attempt_eligible(row)
        if eligible_attempt:
            eligible.setdefault(row["job_id"], []).append(row)
        else:
            rejected.append({"job_id": row.get("job_id"), "work_key": row.get("work_key"),
                             "status": row.get("status"), "reason": reason})

    accepted, ambiguous = {}, []
    for job_id, attempts in eligible.items():
        digests = {}
        for row in attempts:
            path = resolve_candidate_path(root, row["candidate_path"])
            digests.setdefault(path, []).append(row)
        if len(digests) > 1:
            ambiguous.append({"job_id": job_id, "work_key": attempts[0].get("work_key"),
                              "candidates": [str(path) for path in digests]})
        path = max(digests, key=lambda item: len(digests[item]))
        row = digests[path][-1]
        candidate = json.loads(path.read_bytes())
        binding = {"job_id": job_id, "work_key": row.get("work_key"),
                   "source_sha256": row.get("source_sha256"), "input_sha256": row.get("input_sha256"),
                   "candidate_path": str(path), "attempts": len(attempts)}
        problems = []
        if not isinstance(candidate, dict):
            problems.append("candidate_not_object")
        else:
            if row.get("work_key") and candidate.get("paper_id") != row["work_key"]:
                problems.append("candidate_paper_id_mismatch")
            if packets_by_job is not None:
                packet = packets_by_job.get(job_id)
                if packet is None:
                    problems.append("job_not_in_subset")
                else:
                    for key in ("work_key", "source_sha256", "input_sha256"):
                        if packet.get(key) != row.get(key):
                            problems.append("packet_%s_mismatch" % key)
        if problems:
            ambiguous.append({"job_id": job_id, "work_key": row.get("work_key"), "problems": problems})
            continue
        binding["structural_pass"] = bool(parse_validation(row).get("structural_pass"))
        binding["candidate"] = candidate
        accepted[job_id] = binding

    validation_counts = Counter(parse_validation(row).get("structural_pass")
                                for row in rows if parse_validation(row) is not None)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "ambiguous": ambiguous,
        "stats": {
            "attempt_rows": len(rows),
            "eligible_jobs": len(eligible),
            "accepted_jobs": len(accepted),
            "raw_structural_pass_papers": validation_counts.get(True, 0),
            "records_without_hard_errors": validation_counts.get(False, 0),
        },
    }


def observation_at(candidate, index):
    observations = (candidate or {}).get("observations") or []
    if not isinstance(index, int) or index < 0 or index >= len(observations):
        return None
    return observations[index]
