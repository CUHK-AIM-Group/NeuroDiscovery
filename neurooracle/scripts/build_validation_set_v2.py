"""Build an adjudication validation set from accepted candidates (offline).

The old recall layer required both endpoints to match lexically, so genuine
cross-paper replications (e.g. "MCI has lower hippocampus" vs "hippocampal
volume is lower in amnestic MCI") were never even proposed. This builder uses
generous, source-bound blocking instead:

* both sides must mention a shared specific condition (disease/age/genetic risk);
* the anatomy/measurement endpoints must share a token family.

It emits a candidate pool with the full source abstract for each side so a root
reviewer can read source-first. It also emits deliberately hard negatives
(similar wording, different condition or measurement) to test over-merging.
Nothing here decides equivalence; it only proposes pairs to review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neurooracle" / "scripts"))
sys.path.insert(0, str(ROOT / "neurooracle" / "src"))

import candidate_ledger_v2 as ledger  # noqa: E402
import paper_adjudicate_v2 as adj  # noqa: E402
import proposition_scope_v2 as scope  # noqa: E402
import shared_proposition_registry as reg  # noqa: E402

ANATOMY_FAMILIES = {
    "hippocampus": ("hippocamp", "hippocampal", "subiculum", "ca1", "cornu ammonis"),
    "amygdala": ("amygdala", "amygdalar"),
    "entorhinal": ("entorhinal",),
    "cortex": ("cortex", "cortical", "cingulate", "frontal", "temporal", "parietal"),
    "white_matter": ("white matter", "fasciculus", "cingulum", "tract", "fornix"),
    "ventricle": ("ventricle", "ventricular"),
    "retina": ("retina", "retinal", "ganglion cell", "nerve fiber"),
    "whole_brain": ("whole brain", "total brain", "brain volume"),
}

MEASUREMENT_FAMILIES = {
    "volume": ("volume", "volumetry", "atrophy", "size", "volumetric", "morphometry", "voxel"),
    "shape": ("shape", "deformation", "surface"),
    "metabolite": ("naa", "creatine", "mI", "choline", "spectroscopy", "metabolite"),
    "dti": ("fractional anisotropy", "diffusivity", "diffusion", "tractography", "mean diffusivity"),
    "perfusion": ("cerebral blood flow", "perfusion", "arterial spin", "rcbf"),
    "cognition": ("memory", "cognitive", "neuropsycholog", "p300", "adas", "discriminat"),
    "pet": ("pet", "suvr", "amyloid", "tau", "fdg", "av-1451"),
    "eeg": ("eeg", "event-related", "erp", "amplitude"),
}


def families(text, vocabulary):
    if not text:
        return frozenset()
    lowered = str(text).lower()
    return frozenset(label for label, aliases in vocabulary.items()
                     if any(alias in lowered for alias in aliases))


def endpoints(proposition):
    text = " ".join([str(proposition.get("subject") or ""), str(proposition.get("object") or ""),
                     str((proposition.get("qualifiers") or {}).get("measurement") or ""),
                     str((proposition.get("qualifiers") or {}).get("anatomy") or "")])
    return (families(text, ANATOMY_FAMILIES),
            families(text, MEASUREMENT_FAMILIES))


def load_observations(root, pilot_dir, packets_path):
    """Load candidates through the same loader and observation collector the
    live adjudication uses, so the validation set is not built from a second,
    drifted code path.
    """
    packets = [json.loads(line) for line in Path(packets_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    packets_by_job = {packet["job_id"]: packet for packet in packets}
    loaded = ledger.load_accepted_candidates(pilot_dir / "run/PILOT_ollama200.sqlite", root, packets_by_job)
    items = adj.collect_observations(loaded["accepted"])
    items, _dropped = adj.deduplicate(items)
    enriched = {}
    for key, item in items.items():
        proposition = item["proposition"]
        anatomy, measurement = endpoints(proposition)
        packet = packets_by_job[item["job_id"]]
        enriched[key] = dict(item, index=key[1],
                             condition=scope.disease_terms((proposition.get("qualifiers") or {}).get("population")),
                             anatomy=anatomy, measurement=measurement,
                             family=scope.safe_family(proposition),
                             source_abstract=packet["packet"]["abstract"],
                             source_sha256=packet["source_sha256"])
    return enriched, packets_by_job


def block_pairs(items, min_conditions=1):
    items = {key: item for key, item in items.items()}  # keys unchanged; kept for clarity
    pairs = []
    for left_key, right_key in itertools.combinations(sorted(items), 2):
        if left_key[0] == right_key[0]:
            continue
        left, right = items[left_key], items[right_key]
        if left["family"] is None or left["family"] != right["family"]:
            continue
        if scope.orientation_matches(left["proposition"], right["proposition"]) is not True and \
                scope.concept_orientation_matches(left["proposition"], right["proposition"]) is not True:
            continue
        shared = left["condition"] & right["condition"] & scope.SPECIFIC_CONDITION_LABELS
        if len(shared) < min_conditions:
            continue
        if not (left["anatomy"] & right["anatomy"]):
            continue
        if not (left["measurement"] & right["measurement"]):
            continue
        pairs.append({"key": [left_key, right_key], "shared_conditions": sorted(shared),
                      "anatomy": sorted(left["anatomy"] & right["anatomy"]),
                      "measurement": sorted(left["measurement"] & right["measurement"])})
    return pairs


def side(items, key, packets_by_job):
    """Build one validation side with the PRODUCTION observation-side builder.

    The earlier hand-rolled side omitted the ``scope`` block, the publication
    notices/types and ``scope_reason``, so a validation run scored a different
    input than the live adjudicator. ``adj.observation_side`` is the single
    source of truth, so both paths always see the same fields.
    """
    item = items[key]
    built = adj.observation_side(item, packets_by_job)
    built["observation_index"] = key[1]
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("tmp/kg_pilot_20260927_v2"))
    parser.add_argument("--packets", type=Path, default=Path("tmp/kg_pilot_20260927_v2/SUBSET_PACKETS.jsonl"))
    parser.add_argument("--out", default="VALIDATION_CANDIDATES.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir if args.pilot_dir.is_absolute() else ROOT / args.pilot_dir
    packets_path = args.packets if args.packets.is_absolute() else ROOT / args.packets
    items, packets_by_job = load_observations(ROOT, pilot_dir, packets_path)
    pairs = block_pairs(items)
    pairs.sort(key=lambda pair: (-len(pair["shared_conditions"]), -len(pair["measurement"]), pair["key"]))
    if args.limit:
        pairs = pairs[:args.limit]

    # Collapse to distinct study pairs so one study comparison is not counted
    # many times through different observation pairings.
    by_work_pair = {}
    for pair in pairs:
        left, right = items[pair["key"][0]], items[pair["key"][1]]
        work_pair = tuple(sorted((left["work_key"], right["work_key"])))
        score = (len(pair["shared_conditions"]), len(pair["measurement"]))
        if work_pair not in by_work_pair or score > by_work_pair[work_pair][0]:
            by_work_pair[work_pair] = (score, pair)
    pairs = [pair for _, pair in sorted(by_work_pair.values(), key=lambda item: item[1]["key"])]

    records = []
    for index, pair in enumerate(pairs, 1):
        left, right = items[pair["key"][0]], items[pair["key"][1]]
        gate = scope.adjudication_gate(left["proposition"], right["proposition"],
                                       left["observation"], right["observation"])
        records.append({"label": "V%03d" % index, "shared_conditions": pair["shared_conditions"],
                        "kind": "shared_condition_candidate",
                        "shared_anatomy": pair["anatomy"], "shared_measurement": pair["measurement"],
                        "family": left["family"], "gate": gate,
                        "side_a": side(items, pair["key"][0], packets_by_job),
                        "side_b": side(items, pair["key"][1], packets_by_job)})

    # Hard negatives: identical anatomy + measurement wording but a DISJOINT
    # specific condition. A correct system must not merge these.
    negatives = []
    seen_work_pairs = {tuple(sorted(record["shared_conditions"])) and tuple(sorted(
        (record["side_a"]["work_key"], record["side_b"]["work_key"]))) for record in records}
    for left_key, right_key in itertools.combinations(sorted(items), 2):
        if left_key[0] == right_key[0]:
            continue
        left, right = items[left_key], items[right_key]
        if left["family"] is None or left["family"] != right["family"]:
            continue
        if not (left["anatomy"] & right["anatomy"]) or not (left["measurement"] & right["measurement"]):
            continue
        specific_left = left["condition"] & scope.SPECIFIC_CONDITION_LABELS
        specific_right = right["condition"] & scope.SPECIFIC_CONDITION_LABELS
        if not specific_left or not specific_right or (specific_left & specific_right):
            continue
        work_pair = tuple(sorted((left["work_key"], right["work_key"])))
        if work_pair in seen_work_pairs:
            continue
        seen_work_pairs.add(work_pair)
        negatives.append((left_key, right_key))
    # Diversify: cap how many negatives share the same first work, so the set is
    # not dominated by one paper.
    per_work, diversified = Counter(), []
    for left_key, right_key in sorted(negatives):
        work = items[left_key]["work_key"]
        if per_work[work] >= 2:
            continue
        per_work[work] += 1
        diversified.append((left_key, right_key))
    for offset, (left_key, right_key) in enumerate(diversified[:12], len(records) + 1):
        left, right = items[left_key], items[right_key]
        gate = scope.adjudication_gate(left["proposition"], right["proposition"],
                                       left["observation"], right["observation"])
        records.append({"label": "V%03d" % offset, "kind": "hard_negative_disjoint_condition",
                        "shared_conditions": [], "shared_anatomy": sorted(left["anatomy"] & right["anatomy"]),
                        "shared_measurement": sorted(left["measurement"] & right["measurement"]),
                        "family": left["family"], "gate": gate,
                        "side_a": side(items, left_key, packets_by_job),
                        "side_b": side(items, right_key, packets_by_job)})

    result = {"observations": len(items), "candidate_pairs": len(records),
              "kind_counts": dict(Counter(record["kind"] for record in records)),
              "gate_hints": dict(Counter(record["gate"]["hint"] for record in records)),
              "instruction": "Root reads source_abstract and records a verdict per label; "
                             "gate/model outputs are not answers.",
              "pairs": records}
    out_path = pilot_dir / "run" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"observations": result["observations"], "candidate_pairs": result["candidate_pairs"],
                      "gate_hints": result["gate_hints"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
