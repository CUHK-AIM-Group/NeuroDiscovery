"""Score the gate/model adjudication against the root validation reference.

Only a STRICT ``equivalent`` counts as a merge positive. A strictly nested
``narrower_than``/``broader_than`` is a *hierarchy* relation between two
DISTINCT nodes, not the same proposition, so it is reported in a separate table
and never inflates merge recall. This corrects the earlier scorer, which credited
nesting as a merge and therefore reported a recall that was not strict
equivalence.

Errors reported:

* ``missed_merge`` -- root says strictly equivalent but the pipeline does not.
* ``over_merge``   -- root does NOT say equivalent but the pipeline does. The
  dangerous error: it inflates multi-paper support by erasing the science.

Honest denominators (2026-09-27 correction): a pair with no pipeline verdict is
``not_scored`` and stays in the reported total, so HOLD/failed/not-extracted/
not-recalled pairs cannot silently leave the denominator. With no strict
positive, recall is ``None`` (N/A), not ``0.0``; with no attempted merge,
precision is ``None`` (N/A), not ``1.0``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Strict equivalence is the ONLY verdict that permits reusing one claim.
STRICT_MERGE_VERDICTS = {"equivalent"}
# Hierarchy verdicts describe a narrower/broader relation between two DISTINCT
# nodes. Reported separately; never added to the merge rate.
HIERARCHY_VERDICTS = {"narrower_than", "broader_than"}

# Backwards-compatible aliases. ``ROOT_MERGE_VERDICTS`` now means strict only.
ROOT_MERGE_VERDICTS = STRICT_MERGE_VERDICTS
EXACT_MERGE_VERDICTS = STRICT_MERGE_VERDICTS


CORRECTED_REFERENCE = "ROOT_VALIDATION_REFERENCE_CORRECTED.json"
ORIGINAL_REFERENCE = "ROOT_VALIDATION_REFERENCE.json"


def load_reference(pilot_dir, name=CORRECTED_REFERENCE):
    """Load the corrected reference when present; otherwise the original.

    The corrected file applies the strict-equivalence policy. The original is
    kept on disk for auditability and is used only if the corrected file has not
    been generated yet.
    """
    path = pilot_dir / name
    if not path.is_file():
        path = pilot_dir / ORIGINAL_REFERENCE
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["label"]: item for item in payload["decisions"]}


def pipeline_verdicts(pilot_dir, gate_file, model_file):
    """Return {label: (verdict, source)} for the gate baseline and, if given, the model run."""
    verdicts = {}
    if gate_file:
        payload = json.loads((pilot_dir / "run" / gate_file).read_text(encoding="utf-8"))
        for label, record in payload.get("baseline_verdicts", {}).items():
            verdicts[label] = (record["verdict"], "gate")
    for name in ([model_file] if isinstance(model_file, str) else (model_file or [])):
        payload = json.loads((pilot_dir / "run" / name).read_text(encoding="utf-8"))
        for label, record in payload.get("verdicts", {}).items():
            verdict = record["verdict"]
            verdict = verdict.get("verdict") if isinstance(verdict, dict) else verdict
            verdicts[label] = (verdict, record.get("provider", "model"))
    return verdicts


def score(reference, verdicts, merge_verdicts=STRICT_MERGE_VERDICTS):
    """Compare pipeline verdicts to the reference.

    ``root_merge`` is true only for a STRICT equivalence; a nested verdict on
    either side is tracked as ``root_hierarchy``/``pipeline_hierarchy`` but is
    never a merge. A missing pipeline verdict is ``not_scored`` and stays in the
    denominator counting.
    """
    rows, counts = [], Counter()
    for label in sorted(reference):
        root = reference[label]
        verdict, source = verdicts.get(label, (None, None))
        root_merge = root["verdict"] in merge_verdicts
        root_hierarchy = root["verdict"] in HIERARCHY_VERDICTS
        pipeline_merge = verdict in merge_verdicts
        pipeline_hierarchy = verdict in HIERARCHY_VERDICTS
        if verdict is None:
            error = "not_scored"
        elif root_merge and pipeline_merge:
            error = "correct_merge"
        elif root_merge and not pipeline_merge:
            error = "missed_merge"
        elif not root_merge and pipeline_merge:
            error = "over_merge"
        else:
            error = "correct_no_merge"
        counts[error] += 1
        rows.append({"label": label, "kind": root["kind"], "root": root["verdict"],
                     "root_merge": root_merge, "root_hierarchy": root_hierarchy,
                     "pipeline": verdict, "source": source, "error": error,
                     "pipeline_hierarchy": pipeline_hierarchy,
                     "gate_hint": root["gate_hint"],
                     "gate_hard_blocked": root["gate_hard_blocked"],
                     "shared_conditions": root["shared_conditions"]})
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("tmp/kg_pilot_20260927_v2"))
    parser.add_argument("--gate-file", default="VALIDATION_GATE.json")
    parser.add_argument("--model-file", default=None, nargs="*",
                        help="one or more model result files in run/; later files override earlier ones")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir if args.pilot_dir.is_absolute() else ROOT / args.pilot_dir
    reference = load_reference(pilot_dir)
    verdicts = pipeline_verdicts(pilot_dir, args.gate_file, args.model_file)
    rows, counts = score(reference, verdicts)

    def precision_recall(rows_):
        """Strict metrics with honest N/A: no positive -> recall None, no
        attempted merge -> precision None. ``not_scored`` stays in the total."""
        merges = [row for row in rows_ if row["error"] == "correct_merge"]
        attempted = [row for row in rows_ if row["pipeline"] in STRICT_MERGE_VERDICTS]
        positives = [row for row in rows_ if row["root_merge"]]
        return {
            "pipeline_merges": len(attempted),
            "pipeline_merge_labels": [row["label"] for row in attempted],
            "correct_merges": len(merges),
            "root_positives": len(positives),
            "recall": (round(len(merges) / len(positives), 4) if positives else None),
            "precision": (round(len(merges) / len(attempted), 4) if attempted else None),
            "denominator_pairs": len(rows_),
            "not_scored": sum(1 for row in rows_ if row["error"] == "not_scored"),
        }

    def hierarchy_table(rows_):
        """Nesting is reported separately and is never a merge credit."""
        return {
            "root_hierarchy": [row["label"] for row in rows_ if row["root_hierarchy"]],
            "pipeline_hierarchy": [row["label"] for row in rows_ if row["pipeline_hierarchy"]],
            "both_hierarchy": [row["label"] for row in rows_
                               if row["root_hierarchy"] and row["pipeline_hierarchy"]],
            "hierarchy_treated_as_merge": False,
        }

    def blocked_positives(rows_):
        return {
            "hard_blocked": [row["label"] for row in rows_
                             if row["root_merge"] and row["gate_hard_blocked"]],
            "model_eligible": [row["label"] for row in rows_
                               if row["root_merge"] and not row["gate_hard_blocked"]],
        }

    positives = [row["label"] for row in rows if row["root_merge"]]
    negatives = [row["label"] for row in rows if not row["root_merge"]]
    hard_negatives = [row["label"] for row in rows if row["kind"] == "hard_negative_disjoint_condition"]
    result = {
        "gate_file": args.gate_file,
        "model_file": args.model_file,
        "counts": dict(counts),
        "strict_equivalence": precision_recall(rows),
        "hierarchy_separate": hierarchy_table(rows),
        "root_positives_by_reachability": blocked_positives(rows),
        "root_positives": len(positives),
        "root_negatives": len(negatives),
        "hard_negatives": len(hard_negatives),
        "missed_merge_labels": [row["label"] for row in rows if row["error"] == "missed_merge"],
        "over_merge_labels": [row["label"] for row in rows if row["error"] == "over_merge"],
        "not_scored_labels": [row["label"] for row in rows if row["error"] == "not_scored"],
        "hard_negative_over_merge": [row["label"] for row in rows
                                     if row["error"] == "over_merge" and row["kind"] == "hard_negative_disjoint_condition"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.out:
        out = pilot_dir / args.out
        out.write_text(json.dumps({"summary": result, "rows": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
