"""Freeze the 120-paper unseen CORPUS experiment cohort (read-only selection).

Per ``docs/KG_PILOT_REPLAN_20260927.md`` §4.1 this selects three coherent
hippocampal sub-topics, 40 papers each, from the frozen source index:

* ``mci_ad_structure``   -- MCI/AD hippocampal structure
* ``tle_memory``         -- temporal-lobe-epilepsy hippocampus and memory
* ``schizophrenia_structure`` -- schizophrenia hippocampal structure

Guarantees:

* ``fold='corpus'`` only; the frozen 40 heldout papers are never read or written;
* every previously exposed paper (200-paper pilot, 40-paper pilot, v2val16
  re-extraction, root manual batches) is excluded by ``job_id``;
* version duplicates are excluded by ``work_key``, so one work is counted once;
* selection is deterministic: papers are ordered by ``job_id`` and taken up to
  the per-cohort cap, so the same inputs reproduce the same cohort;
* nothing is chosen by old claims or by an expectation of merging.

The first 12 papers (in frozen order) are the smoke set; they are part of the
same 120 and are not a separate sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "tmp" / "kg_paper_reconstruction_20260920" / "BATCH_MODEL" / "INPUTS.sqlite"
DEFAULT_PROTOCOL = ROOT / "tmp" / "kg_paper_reconstruction_20260920" / "BATCH_MODEL" / "EVALUATION_PROTOCOL.json"
DEFAULT_OUT = ROOT / "tmp" / "kg_cohort_v3_20260927"

# job_id sources that must never be reused as "unseen".
EXPOSURE_SOURCES = {
    "exposed_200": ROOT / "tmp" / "kg_pilot_20260927_v2" / "SUBSET_MANIFEST.json",
    "exposed_40": ROOT / "tmp" / "kg_pilot_20260927" / "SUBSET_MANIFEST.json",
}
ROOT_BATCHES = [
    ROOT / "tmp" / "kg_paper_reconstruction_20260920" / "ROOT_SOURCE_BATCH_01.json",
    ROOT / "tmp" / "kg_paper_reconstruction_20260920" / "ROOT_SOURCE_BATCH_02.json",
    ROOT / "tmp" / "kg_paper_reconstruction_20260920" / "ROOT_SOURCE_BATCH_03.json",
]
V2VAL16_LEDGER = ROOT / "tmp" / "kg_pilot_20260927_v2" / "run" / "PILOT_v2val16.sqlite"

MIN_WORDS = 80
SMOKE_SIZE = 12

# Topic = (condition terms, measurement terms). A paper joins a cohort when its
# abstract mentions a hippocampal term AND one condition term AND one
# measurement term. Topics are ordered; the first matching cohort wins.
TOPICS = {
    "mci_ad_structure": (
        ("mild cognitive impairment", "mci ", "amnestic", "alzheimer", "preclinical", "dementia"),
        ("volume", "atrophy", "structure", "structural", "thickness", "subfield",
         "morphometry", "size"),
    ),
    "tle_memory": (
        ("temporal lobe epilepsy", "tle", "epilepsy"),
        ("memory", "recall", "verbal", "learning"),
    ),
    "schizophrenia_structure": (
        ("schizophrenia", "psychosis", "schizoaffective", "psychotic"),
        ("volume", "atrophy", "structure", "structural", "thickness", "subfield",
         "morphometry", "size"),
    ),
}
COHORT_ORDER = list(TOPICS)
HIPPOCAMPAL = ("hippocamp",)


def walk_job_ids(node, found):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("job_id", "heldout_job_id") and isinstance(value, str):
                found.add(value)
            walk_job_ids(value, found)
    elif isinstance(node, list):
        for item in node:
            walk_job_ids(item, found)


def load_exclusions(protocol_path):
    exposed = set()
    detail = {}
    for name, path in EXPOSURE_SOURCES.items():
        found = set()
        if path.is_file():
            walk_job_ids(json.loads(path.read_text(encoding="utf-8")), found)
        detail[name] = len(found)
        exposed |= found
    for path in ROOT_BATCHES:
        found = set()
        if path.is_file():
            walk_job_ids(json.loads(path.read_text(encoding="utf-8")), found)
        detail[path.name] = len(found)
        exposed |= found
    if V2VAL16_LEDGER.is_file():
        connection = sqlite3.connect(V2VAL16_LEDGER.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for (job_id,) in connection.execute(
                    "select distinct job_id from attempts where job_id is not null"):
                exposed.add(job_id)
        finally:
            connection.close()
        detail["v2val16_ledger"] = "queried"
    heldout = set()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for record in protocol.get("heldout_papers") or []:
        if record.get("job_id"):
            heldout.add(record["job_id"])
    return exposed, heldout, detail


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select(inputs_path, excluded, per_cohort, scan_limit):
    connection = sqlite3.connect(inputs_path.resolve().as_uri() + "?mode=ro", uri=True)
    cohorts = {name: [] for name in COHORT_ORDER}
    seen_work = set()
    scanned = 0
    short = 0
    try:
        cursor = connection.execute(
            "select job_id, work_key, source_sha256, input_sha256, input_zlib,"
            " utf8_bytes, abstract_words, fold from inputs"
            " where fold='corpus' order by job_id limit ?",
            (scan_limit,),
        )
        for job_id, work_key, source_sha, input_sha, blob, utf8_bytes, words, fold in cursor:
            scanned += 1
            if len(cohorts["mci_ad_structure"]) >= per_cohort and \
                    len(cohorts["tle_memory"]) >= per_cohort and \
                    len(cohorts["schizophrenia_structure"]) >= per_cohort:
                break
            if job_id in excluded or work_key in seen_work:
                continue
            if not words or words < MIN_WORDS:
                short += 1
                continue
            if not blob:
                continue
            try:
                packet = json.loads(zlib.decompress(blob).decode("utf-8"))
            except Exception:
                continue
            text = " ".join(
                str(packet.get(field) or "") for field in ("title", "abstract"))
            lowered = text.lower()
            if not any(term in lowered for term in HIPPOCAMPAL):
                continue
            for name in COHORT_ORDER:
                if len(cohorts[name]) >= per_cohort:
                    continue
                condition, measurement = TOPICS[name]
                if any(term in lowered for term in condition) and \
                        any(term in lowered for term in measurement):
                    cohorts[name].append({
                        "job_id": job_id, "work_key": work_key,
                        "source_sha256": source_sha, "input_sha256": input_sha,
                        "utf8_bytes": utf8_bytes, "abstract_words": words,
                        "fold": fold, "packet": packet,
                    })
                    seen_work.add(work_key)
                    break
    finally:
        connection.close()
    return cohorts, {"scanned": scanned, "short_abstracts": short}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-cohort", type=int, default=40)
    parser.add_argument("--scan", type=int, default=0, help="0 = scan the whole corpus")
    args = parser.parse_args()

    inputs = args.inputs if args.inputs.is_absolute() else ROOT / args.inputs
    protocol = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded, heldout, exposure_detail = load_exclusions(protocol)
    print("excluded job_ids:", len(excluded), "heldout:", len(heldout))

    scan_limit = args.scan if args.scan else 10_000_000
    cohorts, stats = select(inputs, excluded | heldout, args.per_cohort, scan_limit)

    # Interleave the cohorts round-robin so the first SMOKE_SIZE papers span all
    # three sub-topics instead of coming from a single cohort. This keeps the
    # smoke set representative without changing which 120 papers are selected.
    ordered = []
    depth = max((len(cohorts[name]) for name in COHORT_ORDER), default=0)
    for index in range(depth):
        for name in COHORT_ORDER:
            if index < len(cohorts[name]):
                ordered.append(dict(cohorts[name][index], cohort=name))
    smoke = [record["job_id"] for record in ordered[:SMOKE_SIZE]]

    # ``SUBSET_PACKETS.jsonl`` is the runner's canonical input name; the cohorts
    # are written with a descriptive alias as well.
    packets_path = out_dir / "SUBSET_PACKETS.jsonl"
    with packets_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "COHORT_PACKETS.jsonl").write_bytes(packets_path.read_bytes())

    manifest = {
        "cohort_version": "v3_20260927",
        "source": "INPUTS.sqlite fold=corpus; ordered by job_id; deterministic",
        "old_claims_consulted": False,
        "min_abstract_words": MIN_WORDS,
        "per_cohort_requested": args.per_cohort,
        "cohort_counts": {name: len(cohorts[name]) for name in COHORT_ORDER},
        "total_selected": len(ordered),
        "smoke_size": SMOKE_SIZE,
        "smoke_job_ids": smoke,
        "exposure_sources": {key: (value if isinstance(value, str) else value)
                             for key, value in exposure_detail.items()},
        "excluded_job_ids": len(excluded),
        "heldout_excluded": len(heldout),
        "scan_stats": stats,
        "inputs_file": str(inputs.relative_to(ROOT)),
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "packets_sha256": hashlib.sha256(packets_path.read_bytes()).hexdigest(),
        "selection_rule": "hippocampal term AND (condition terms) AND (measurement terms); "
                          "first matching cohort; one work per work_key",
        "papers": [{key: value for key, value in record.items() if key != "packet"}
                   for record in ordered],
    }
    (out_dir / "COHORT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("cohort_counts", "total_selected", "scan_stats")},
                     ensure_ascii=False, indent=1))
    print("wrote", out_dir / "COHORT_MANIFEST.json")


if __name__ == "__main__":
    main()
