"""Read-only regression on exposed first-body outputs; no re-extraction or promotion."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neurooracle/scripts"))
import paper_pilot_batch as batch
from paper_extract_v2 import EXTRACTION_SYSTEM_V2
from neurooracle.src.paper_extraction_contract_v3 import build_prompt
from neurooracle.src import candidate_checks


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(pilot):
    summary = read(pilot / "BLIND_COMPARISON_SUMMARY.json")
    if sha(pilot / "FROZEN_REFERENCE.json") != summary["reference_sha256"]:
        raise ValueError("Reference changed")
    completed = read(pilot / "blind_run/COMPLETED.json")
    packets = {packet["work_key"]: packet for packet in map(json.loads, (pilot / "MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines())}
    records = []
    for record in completed["records"]:
        successful = [attempt for attempt in record["attempts"] if attempt["http_status"] == 200]
        if len(successful) != 1:
            raise ValueError("Expected frozen first-body experiment")
        attempt = successful[0]
        path = pilot / attempt["candidate_path"]
        if sha(path) != attempt["candidate_sha256"] or sha(pilot / attempt["body_path"]) != attempt["body_sha256"]:
            raise ValueError("Model evidence changed")
        candidate = read(path)
        packet = packets[record["work_key"]]
        structural = batch.validate_candidate(candidate, batch.shim_source(packet))
        verdict = candidate_checks.check_candidate(
            candidate, batch.normalized_packet(packet)["abstract"], structural)
        records.append({
            "work_key": record["work_key"], "candidate_sha256": sha(path),
            "original_status": record["status"],
            "original_structural_errors": attempt["validation"]["errors"],
            "current_blocked": verdict["blocked"],
            "advisory": verdict["advisory"],
            "exempted": verdict["exempted"],
            "verdict": verdict["verdict"],
            "findings": verdict["findings"],
            "status_promoted": False,
        })
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(), "model_calls": 0,
        "independent_scientific_validation": False, "exposed_development_set": True,
        "production_writes": 0, "original_reference_sha256": summary["reference_sha256"],
        "records": records,
        "advisory_reason_counts": dict(Counter(reason.split(":", 1)[-1] for record in records
                                               for reason in record["advisory"])),
        "verdict_counts": dict(Counter(record["verdict"] for record in records)),
        "limits": "Legacy source diagnostics cover specific risks only. V3 schema absence is not a model error on old v2 requests. No semantic entailment, equivalence or accuracy certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "tmp/kg_source_first_20260927")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.pilot)
    prompt = build_prompt(EXTRACTION_SYSTEM_V2)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    files = ["neurooracle/src/source_statistics.py", "neurooracle/src/pilot_validation_v3.py",
             "neurooracle/src/paper_extraction_contract_v3.py", "neurooracle/src/candidate_checks.py",
             "neurooracle/scripts/paper_batch_model.py", "neurooracle/configs/active_pipeline.json",
             "neurooracle/scripts/check_paper_extraction_v3.py", "neurooracle/scripts/audit_source_first_repairs_v3.py"]
    report["code_sha256"] = {name: sha(ROOT / name) for name in files}
    with (args.out_dir / "OFFLINE_REGRESSION.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with (args.out_dir / "PROMPT_V3.json").open("x", encoding="utf-8") as handle:
        json.dump({"system": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest(), "active_sender_changed": False, "model_calls": 0}, handle, ensure_ascii=False, indent=2)
    with (args.out_dir / "CLEANUP_MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump({"keep": ["OFFLINE_REGRESSION.json", "PROMPT_V3.json", "CLEANUP_MANIFEST.json"], "deleted": [], "deletable_now": [], "reason": "Small reproducible audit evidence; no copied responses or databases", "independent_scientific_validation": False}, handle, indent=2)
    print(json.dumps({"papers": len(report["records"]), "verdicts": report["verdict_counts"],
                      "advisory_reasons": report["advisory_reason_counts"], "model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
