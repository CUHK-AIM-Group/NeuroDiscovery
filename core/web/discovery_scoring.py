"""Deterministic, version-bound scoring. No scores are assigned to v1 categories."""
import argparse
import json
from collections import Counter
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def applicable_item_ids(pack, card_id):
    scoring = pack.get("scoring") or {}
    return scoring.get("applicable_items_by_card", {}).get(
        card_id, scoring.get("item_ids", [q["id"] for q in pack["questions"]]))


def score_session(state, pack):
    if not pack.get("scoring"):
        return None
    questions = {q["id"]: q for q in pack["questions"]}
    ids = pack["scoring"]["item_ids"]
    cards = {}
    for card_id in state["order"]:
        answers = state["answers"][card_id]
        scores, unavailable = {}, {}
        applicable = applicable_item_ids(pack, card_id)
        for key in ids:
            if key not in applicable:
                scores[key] = None
                continue
            options = {o["value"]: o["score"] for o in questions[key]["options"]}
            value = answers.get(key)
            if value is not None and value not in options:
                raise ValueError("Unknown rating; refusing to recode it.")
            scores[key] = options.get(value)
            if scores[key] is None:
                unavailable[key] = value if value is not None else "unanswered"
        values = [x for x in scores.values() if x is not None]
        cards[card_id] = {"items": scores, "unavailable": unavailable,
                          "numeric_count": len(values),
                          "applicable_items": list(applicable),
                          "not_applicable": [key for key in ids if key not in applicable],
                          "composite": mean(values) if len(values) == len(applicable) else None}
    dimensions = {}
    for key in ids:
        values = [c["items"][key] for c in cards.values() if c["items"][key] is not None]
        missing = Counter(c["unavailable"][key] for c in cards.values() if key in c["unavailable"])
        dimensions[key] = {"label": questions[key]["capability"], "mean": mean(values),
                           "n": len(values), "unavailable_counts": dict(missing),
                           "applicable_count": sum(key in c["applicable_items"] for c in cards.values()),
                           "not_applicable_count": sum(key in c["not_applicable"] for c in cards.values()),
                           "distribution": {str(i): values.count(i) for i in range(1, 6)}}
    composites = [c["composite"] for c in cards.values() if c["composite"] is not None]
    return {"scoring_version": pack["scoring"]["version"], "scale": [1, 5],
            "cards": cards, "dimensions": dimensions,
            "composite_mean": mean(composites), "complete_case_count": len(composites),
            "case_count": len(cards), "all_cases_complete": len(composites) == len(cards)}


def aggregate_exports(exports):
    """Organiser-only descriptive summary; excludes tests/incomplete and duplicate exports.

    One completed session per anonymous expert code is required. No unregistered
    choice among repeated human submissions or material versions is made here.
    """
    accepted, excluded, seen = [], Counter(), set()
    for item in exports:
        state = item["session"]
        identity = (state["id"], state["revision"])
        if identity in seen:
            excluded["duplicate_export"] += 1
            continue
        seen.add(identity)
        if state.get("record_kind", state.get("profile", {}).get("mode")) == "test":
            excluded["test"] += 1
        elif state["stage"] != "complete":
            excluded["incomplete_session"] += 1
        elif not item.get("scoring"):
            excluded["legacy_unscored"] += 1
        else:
            accepted.append(item)
    if not accepted:
        return {"expert_count": 0, "excluded": dict(excluded), "composite_mean": None}
    signatures = {(x["session"]["pack_id"], x["session"]["pack_hash"],
                   x["session"]["protocol_version"], x["scoring"]["version"]) for x in accepted}
    if len(signatures) != 1:
        raise ValueError("Different material/rubric versions must be analysed separately.")
    codes = [x["session"]["profile"]["code"] for x in accepted]
    if len(set(codes)) != len(codes):
        raise ValueError("Repeated expert code: resolve repeat submissions before aggregation.")
    scores = [score_session(x["session"], {"questions": x["questions"], "scoring": x["scoring"]}) for x in accepted]
    variable_panel = bool(accepted[0]["scoring"].get("applicable_items_by_card"))
    card_ids = sorted(set().union(*(x["cards"] for x in scores))) if variable_panel else sorted(scores[0]["cards"])
    if not variable_panel and any(sorted(x["cards"]) != card_ids for x in scores):
        raise ValueError("Mismatched case panels.")
    item_ids = accepted[0]["scoring"]["item_ids"]
    cases, dimensions = {}, {}
    for cid in card_ids:
        assigned = [x["cards"][cid] for x in scores if cid in x["cards"]]
        values = [x["composite"] for x in assigned if x["composite"] is not None]
        excluded_key = "excluded_incomplete_applicable" if variable_panel else "excluded_incomplete_six"
        cases[cid] = {"mean": mean(values), "n": len(values), excluded_key: len(assigned) - len(values)}
    for key in item_ids:
        per_case, all_values, missing = {}, [], Counter()
        for cid in card_ids:
            assigned = [x["cards"][cid] for x in scores if cid in x["cards"]]
            values = [x["items"][key] for x in assigned if x["items"][key] is not None]
            missing.update(x["unavailable"][key] for x in assigned if key in x["unavailable"])
            all_values.extend(values)
            per_case[cid] = {"mean": mean(values), "n": len(values),
                             "applicable": any(key in x["applicable_items"] for x in assigned)}
        available = [x["mean"] for x in per_case.values() if x["mean"] is not None]
        applicable_count = sum(x["applicable"] for x in per_case.values())
        dimensions[key] = {"label": scores[0]["dimensions"][key]["label"], "case_means": per_case,
                           "mean": mean(available) if len(available) == applicable_count else None,
                           "applicable_case_count": applicable_count,
                           "available_case_mean": mean(available), "cases_covered": len(available),
                           "n_ratings": len(all_values), "unavailable_counts": dict(missing),
                           "distribution": {str(i): all_values.count(i) for i in range(1, 6)}}
    composite_values = [c["mean"] for c in cases.values() if c["mean"] is not None]
    return {"pack_id": accepted[0]["session"]["pack_id"], "pack_hash": accepted[0]["session"]["pack_hash"],
            "scoring_version": accepted[0]["scoring"]["version"], "expert_count": len(scores),
            "case_count": len(card_ids), "cases": cases, "dimensions": dimensions, "excluded": dict(excluded),
            "composite_mean": mean(composite_values) if len(composite_values) == len(card_ids) else None,
            "available_case_composite_mean": mean(composite_values), "composite_cases_covered": len(composite_values),
            "interpretation": accepted[0]["scoring"]["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description="Summarize completed expert JSON exports; no public answer endpoint.")
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate_exports([json.loads(p.read_text(encoding="utf-8")) for p in args.exports])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
