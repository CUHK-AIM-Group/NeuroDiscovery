"""Select the v6 expert-study panel: externally validated, strongest effects first.

Owner decision (2026-09-18): every material in the panel must have passed the
frozen external triage, and the panel shows the strongest results because the
system itself presents its best candidates to users. The panel size 34 comes
from the study design: 10 experts x 10 materials each = 100 reviews, every
material reviewed by 3 experts -> ceil-free 33.3 -> 34 materials (100/3
rounded up so each expert still gets 10).

Rule (frozen here, disclosed in the pack):
  pool  = every canonical hypothesis whose original_external_assessment has
          pragmatic_triage_pass == true, in either the completed first
          discovery campaign (old_r1 handoff) or the completed r7 expansion
          handoff;
  order = minimum_directional_standardized_beta descending, then
          joint_direction_frequency descending, then hypothesis_id ascending
          (deterministic tie-break only);
  take  = top 34. No other input (anchors, redundancy, curator preference)
          influences selection or order.

This script only reads frozen handoff metadata; it never runs experiments.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials" / "sources_v6"
TARGET = HERE / "selection_v6.json"

SOURCE_HASHES = {
    "old_r1/candidate_index.json": "a16ff8cc00d87dcf2c1d6eeafb2d0ad4ae3ea8e55f7d8f061d2f7ad309ce1b74",
    "r7/candidate_index.json": "40964c8ba056a2cc44615857030f48a177e24baff50a25a2a6b168bc2f74f19d",
    "r7/inherited_baseline.json": "e0b42e64bd59fc7095bfd1d0657a0ca9ccde40d32a95729e2364fc2189c3bb29",
}

PANEL_SIZE = 34
SELECTION_RULE_ZH = (
    "候选池为两批已完成发现活动中外部整理条件全部通过的假设（旧批 11 条、扩充批 27 条，共 38 条）；"
    "按最小方向标准化效应降序排列，并列时按联合方向频率降序，再并列按假设标识字典序；取前 34 条。"
    "未使用锚点、冗余分组或整理者偏好影响选取与排序。"
)


def load_checked(rel, expected):
    raw = (HERE / rel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {rel}; build a separately reviewed new version.")
    return json.loads(raw)


def records_of(doc):
    if isinstance(doc, list):
        return doc
    return doc["records"]


def pool_entry(batch, record):
    assessment = record["original_external_assessment"]
    return {
        "batch": batch,
        "hypothesis_id": record["canonical_hypothesis_id"],
        "endpoint": record["endpoint"],
        "named": list(assessment["named"]),
        "minimum_directional_standardized_beta": assessment["minimum_directional_standardized_beta"],
        "joint_direction_frequency": assessment["joint_direction_frequency"],
        "direction": record["canonical_record"]["direction"],
    }


def main():
    old_records = records_of(load_checked("old_r1/candidate_index.json", SOURCE_HASHES["old_r1/candidate_index.json"]))
    r7_records = records_of(load_checked("r7/candidate_index.json", SOURCE_HASHES["r7/candidate_index.json"]))
    baseline = load_checked("r7/inherited_baseline.json", SOURCE_HASHES["r7/inherited_baseline.json"])

    pool = []
    for record in old_records:
        assessment = record.get("original_external_assessment") or {}
        if assessment.get("pragmatic_triage_pass") is True:
            pool.append(pool_entry("old_r1", record))
    for record in r7_records:
        assessment = record.get("original_external_assessment") or {}
        if assessment.get("pragmatic_triage_pass") is True:
            pool.append(pool_entry("r7_expansion", record))

    # Cross-check the old-batch pool against the sealed inherited baseline.
    sealed = {h["hypothesis_id"]: h for h in baseline["summary"]["hypothesis_assessments"]}
    for entry in pool:
        if entry["batch"] != "old_r1":
            continue
        check = sealed.get(entry["hypothesis_id"])
        if not check or check["pragmatic_triage_pass"] is not True:
            raise ValueError(f"Old-batch pass not in sealed baseline: {entry['hypothesis_id']}")
        if abs(check["minimum_directional_standardized_beta"] - entry["minimum_directional_standardized_beta"]) > 1e-12:
            raise ValueError(f"Baseline effect mismatch: {entry['hypothesis_id']}")

    pool.sort(key=lambda e: (-e["minimum_directional_standardized_beta"],
                             -e["joint_direction_frequency"], e["hypothesis_id"]))
    if len(pool) < PANEL_SIZE:
        raise ValueError("Not enough externally validated hypotheses; do not manufacture cases.")
    for rank, entry in enumerate(pool, 1):
        entry["rank"] = rank
    selected, dropped = pool[:PANEL_SIZE], pool[PANEL_SIZE:]

    out = {
        "created": "2026-09-18",
        "panel_size": PANEL_SIZE,
        "study_design_basis": "10 experts x 10 materials = 100 reviews; each material reviewed by 3 experts; ceil(100/3) = 34.",
        "selection_rule_zh": SELECTION_RULE_ZH,
        "source_hashes": SOURCE_HASHES,
        "pool_size": len(pool),
        "selected": selected,
        "dropped": [{**e, "drop_reason": "rank_below_panel_size"} for e in dropped],
    }
    with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"pool": len(pool), "selected": len(selected),
                      "weakest_selected": selected[-1]["minimum_directional_standardized_beta"],
                      "strongest_dropped": dropped[0]["minimum_directional_standardized_beta"] if dropped else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
