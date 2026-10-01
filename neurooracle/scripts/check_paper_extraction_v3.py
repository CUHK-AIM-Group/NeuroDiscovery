"""Offline candidate inspection through the single check door.

Never sends requests, never promotes, never rewrites a candidate. Reports the
four outcomes of ``neurooracle/src/candidate_checks.py`` instead of calling
individual rule modules, so there is one place that decides.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))
from neurooracle.src import candidate_checks
from paper_batch_model import validate_candidate as structural_errors_for


def shim_for(packet):
    """Minimal source view the structural validator expects."""
    return {"job": {"work_key": packet.get("paper_id") or packet.get("work_key"),
                    "job_id": packet.get("job_id")},
            "source_snapshot": {"abstract": packet.get("abstract")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="JSON packet with abstract or packet.abstract")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate_bytes = args.candidate.read_bytes()
    source_bytes = args.source.read_bytes()
    candidate = json.loads(candidate_bytes)
    source = json.loads(source_bytes)
    packet = source.get("packet", source)
    abstract = packet.get("abstract")
    if not isinstance(abstract, str):
        raise ValueError("Source needs an abstract string; supply the exact transported source")
    paper_id = packet.get("paper_id") or source.get("work_key")
    if not paper_id or candidate.get("paper_id") != paper_id:
        raise ValueError("Candidate/source identity mismatch")
    structural = structural_errors_for(candidate, shim_for(packet))
    result = candidate_checks.check_candidate(candidate, abstract, structural)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "checks_sha256": hashlib.sha256((ROOT / "neurooracle/src/candidate_checks.py").read_bytes()).hexdigest(),
        "verdict": result["verdict"],
        "blocked": result["blocked"], "advisory": result["advisory"], "exempted": result["exempted"],
        "findings": result["findings"], "policy": result["policy"],
        "status": "HELD" if result["verdict"] == "blocked" else "CHECK_ONLY",
        "production_merge_allowed": False, "independent_scientific_validation": False,
        "model_calls": 0, "old_candidate_modified": False,
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "verdict": report["verdict"],
                      "blocked": len(result["blocked"]), "advisory": len(result["advisory"]),
                      "model_calls": 0}))


if __name__ == "__main__":
    main()
