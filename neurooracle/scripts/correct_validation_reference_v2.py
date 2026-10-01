"""Write the corrected root validation reference (strict-equivalence policy).

The frozen ``ROOT_VALIDATION_REFERENCE.json`` (2026-09-27) was edited after
model outputs were seen, and its policy counted a strictly-nested verdict
(``narrower_than``/``broader_than``) as a merge. Two problems follow:

* it is a development regression, not a frozen blind validation;
* nesting is a containment edge between two DISTINCT nodes, so it must never be
  credited as one shared proposition.

This script does NOT overwrite the old file. It emits a separate corrected
reference that:

* keeps only ``equivalent`` as a strict merge positive;
* records nesting on its own ``hierarchy`` list;
* records the corrected verdict, the reason for every change, the original
  verdict, the source candidate file hash and each side's source anchor
  (work key, observation index, source hash) so every label is traceable;
* records an explicit independence disclaimer, because it is still root's own
  reading and cannot serve as external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "tmp" / "kg_pilot_20260927_v2"

STRICT_MERGE_VERDICTS = {"equivalent"}
HIERARCHY_VERDICTS = {"narrower_than", "broader_than"}
NOT_A_MERGE_VERDICTS = {"related_to", "distinct", "unresolved"}

# label -> corrected verdict. Only labels whose verdict changes are listed; all
# other labels keep the original verdict. Nesting is retained as a verdict but
# reclassified as a hierarchy relation, so it is not a merge.
CORRECTED_VERDICTS = {
    "V007": "narrower_than",
    "V011": "narrower_than",
    "V014": "narrower_than",
    "V016": "narrower_than",
    "V029": "narrower_than",
}

# Why each corrected verdict changed (or why an equivalent is retained with a
# required caveat). This is the root's re-reading under the strict policy.
CORRECTION_REASONS = {
    "V007": "Strict subgroup (amyloid-positive early MCI, substructures); the "
            "containment edge is not one shared proposition, so it is no longer "
            "a merge positive. Side A states the claim as a premise.",
    "V011": "Limbic MCI is a strict subtype of MCI; subtype containment is a "
            "hierarchy edge, not equivalence.",
    "V014": "Amyloid-positive MCI with substructures is a strict subgroup; "
            "containment only.",
    "V016": "No proved containment between amyloid-positive and limbic MCI. "
            "A subtype relation is a hierarchy edge, not equivalence.",
    "V029": "Limbic MCI subtype plus a mislabelled direction; containment only.",
    "V009": "Retained as strict equivalence: the same group difference in the "
            "same scope. The discrimination label is a direction typing error, "
            "and side B is background, so it is not independent replication.",
    "V021": "Retained as strict equivalence only provisionally: trauma-exposed "
            "vs trauma-exposed-and-non-exposed comparators differ, so this is "
            "flagged for a matched-comparator check rather than treated as "
            "settled. Side B is a meta-analysis (secondary).",
    "V035": "Retained as strict equivalence only provisionally: total recall vs "
            "delayed auditory memory are not automatically interchangeable, so "
            "the task family is flagged rather than silently merged.",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor(side):
    return {"work_key": side.get("work_key"), "observation_index": side.get("observation_index"),
            "source_sha256": side.get("source_sha256"), "role": side.get("role"),
            "quote_count": len(side.get("quotes") or [])}


def build(pilot_dir):
    reference_path = pilot_dir / "ROOT_VALIDATION_REFERENCE.json"
    candidates_path = pilot_dir / "run" / "VALIDATION_CANDIDATES.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    by_label = {pair["label"]: pair for pair in candidates["pairs"]}

    decisions = []
    for original in reference["decisions"]:
        label = original["label"]
        verdict = CORRECTED_VERDICTS.get(label, original["verdict"])
        pair = by_label.get(label, {})
        decisions.append({
            "label": label,
            "kind": original["kind"],
            "original_verdict": original["verdict"],
            "verdict": verdict,
            "changed": verdict != original["verdict"],
            "reclassified": (verdict in HIERARCHY_VERDICTS) and original["merge"],
            "merge": verdict in STRICT_MERGE_VERDICTS,
            "hierarchy": verdict in HIERARCHY_VERDICTS,
            "original_merge": original["merge"],
            "correction_reason": CORRECTION_REASONS.get(label, "unchanged"),
            "flags": original["flags"],
            "shared_conditions": original["shared_conditions"],
            "family": original["family"],
            "gate_hint": original["gate_hint"],
            "gate_hard_blocked": original["gate_hard_blocked"],
            "side_a_anchor": anchor(pair.get("side_a") or {}),
            "side_b_anchor": anchor(pair.get("side_b") or {}),
        })

    strict = [item["label"] for item in decisions if item["merge"]]
    hierarchy = [item["label"] for item in decisions if item["hierarchy"]]
    changed = [item["label"] for item in decisions if item["changed"]]
    # Nesting that used to be counted as a merge and is no longer merge credit.
    downgraded = [item["label"] for item in decisions if item["reclassified"]]

    return {
        "actor": "root_direct_source_reading",
        "status": "corrected_development_reference",
        "independence": ("NOT independent validation. Root wrote the extraction "
                         "prompt and the scope gate, and edited the original "
                         "reference after seeing model outputs, so agreement with "
                         "this file bounds recall only against root itself."),
        "policy": ("Only ``equivalent`` is a strict merge positive. "
                   "``narrower_than``/``broader_than`` are hierarchy relations "
                   "between two distinct nodes and are never merge credit. "
                   "``related_to``/``distinct``/``unresolved`` are not merges."),
        "unit": "pair",
        "provenance": {
            "original_reference_file": str(reference_path.relative_to(ROOT)),
            "original_reference_sha256": sha256_file(reference_path),
            "candidates_file": str(candidates_path.relative_to(ROOT)),
            "candidates_sha256": sha256_file(candidates_path),
        },
        "counts": {
            "pairs": len(decisions),
            "strict_merge_positives": len(strict),
            "strict_merge_labels": strict,
            "hierarchy_labels": hierarchy,
            "changed_labels": changed,
            "reclassified_from_merge_to_hierarchy": downgraded,
            "nested_no_longer_merge": downgraded,
            "original_merge_allowed": sum(1 for item in decisions if item["original_merge"]),
        },
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--out", default="ROOT_VALIDATION_REFERENCE_CORRECTED.json")
    args = parser.parse_args()
    pilot_dir = args.pilot_dir if args.pilot_dir.is_absolute() else ROOT / args.pilot_dir
    result = build(pilot_dir)
    out = pilot_dir / args.out
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
