"""Reviewed, claim-specific endpoint corrections, never global node merges."""
from copy import deepcopy
import hashlib
import json

CT_OLD = "CLM_CONCEPT:cortical_thickness"
CT_TARGET = "IF:cortical_thickness"
TRS_OLD = "CLM_CONCEPT:treatment_resistant_schizophrenia_9fc807b69dbc"
TRS_TARGET = "CUI:C3544321"
CT_CLAIMS = frozenset({
    "CLM:CASE1MAN:23328446:826", "CLM:CASE1MAN:24179811:9598", "CLM:CASE1MAN:27198485:931",
    "CLM:CASE1MAN:29454511:787", "CLM:CASE1MAN:29960671:013", "CLM:CASE1MAN:30118825:930",
    "CLM:CASE1MAN:31503209:861", "CLM:CASE1MAN:32958675:645", "CLM:CASE1MAN:32958675:646",
    "CLM:CASE1MAN:34971699:948", "CLM:CASE1MAN:35739320:047", "CLM:CASE1MAN:36846964:9408",
    "CLM:CASE1MAN:38123240:919", "CLM:CASE1MAN:40332668:927", "CLM:CASE1MAN:41621354:056",
    "CLM:d08ebe9e1c42008b",
})
TRS_PMIDS = frozenset({"33515249", "38645076", "39754499", "31911624", "38354479", "25716781",
    "26948188", "28086761", "36268829", "30170114", "40943517", "36165228", "10932483", "11202015",
    "26320028", "34496461", "35041733", "12839431", "25102584", "40047832", "31551822", "27853387"})
TRS_CLAIMS = frozenset("CLM:case3_manual_" + pmid + "_001" for pmid in TRS_PMIDS)
DISEASE_TYPES = frozenset({"disease_state", "psychiatric_disorder_subtype", "disease_subtype", "schizophrenia_subtype",
    "neuropsychiatric disease subtype", "schizophrenia subtype", "disorder", "psychiatric_condition",
    "clinical condition", "psychiatric_disorder"})


def digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def choose_events(collection):
    selected = []
    for old, target, ids, name, types in (
        (CT_OLD, CT_TARGET, CT_CLAIMS, "cortical thickness", {"IMAGING_MARKER"}),
        (TRS_OLD, TRS_TARGET, TRS_CLAIMS, "treatment-resistant schizophrenia", DISEASE_TYPES),
    ):
        for item in collection["incidents"][old]:
            if item["claim_id"] not in ids:
                continue
            side = item["side"]
            if item[side+"_name"] != name or item["qualifiers"][side+"_type"] not in types:
                raise ValueError("reviewed endpoint surface/type changed")
            if not item["pmid"] or not item["raw_text"].strip():
                raise ValueError("reviewed stored source is missing")
            selected.append({"claim_id": item["claim_id"], "side": side, "old_id": old, "target_id": target,
                "claim_sha256": item["record_sha256"], "name": name, "declared_type": item["qualifiers"][side+"_type"],
                "review_scope": "stored_endpoint_identity_only_not_whole_claim_scientific_approval"})
    if {item["claim_id"] for item in selected} != CT_CLAIMS | TRS_CLAIMS or len(selected) != 38:
        raise ValueError("reviewed 38-claim scope changed")
    return selected


def change_claim(record, event):
    if digest(record) != event["claim_sha256"]:
        raise ValueError("reviewed claim changed")
    md = record["metadata"]
    side = event["side"]
    field = side + "_id"
    opposite = "object_id" if side == "subject" else "subject_id"
    if md.get(field) != event["old_id"] or md.get(opposite) in (event["old_id"], event["target_id"]):
        raise ValueError("stale or self-loop-producing endpoint correction")
    inner = md.get("metadata") or {}
    if field in inner and inner[field] != event["old_id"]:
        raise ValueError("conflicting nested endpoint")
    after = deepcopy(record)
    after["metadata"][field] = event["target_id"]
    if field in inner:
        after["metadata"]["metadata"][field] = event["target_id"]
    return after


def change_edge(record, events):
    md = record.get("metadata") or {}
    if record["relation_type"] == "about":
        event = events.get(record["source_id"])
        if event is None or record["target_id"] != event["old_id"]:
            return record
        if md.get("claim_id") not in (None, "", event["claim_id"]):
            raise ValueError("about edge has conflicting owner")
        field = "target_id"
    else:
        event = events.get(md.get("claim_id"))
        if event is None:
            return record
        field = "source_id" if event["side"] == "subject" else "target_id"
        if record[field] != event["old_id"]:
            raise ValueError("owned science edge endpoint disagrees with claim")
    after = deepcopy(record)
    after[field] = event["target_id"]
    if after["source_id"] == after["target_id"]:
        raise ValueError("correction creates a self loop")
    return after


def nonidentity_claim(record):
    """All fields other than the two routing IDs, including unknown audit data."""
    result = deepcopy(record)
    for md in (result["metadata"], result["metadata"].get("metadata") or {}):
        md.pop("subject_id", None)
        md.pop("object_id", None)
    return result
