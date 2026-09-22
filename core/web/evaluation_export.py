"""Participant-scoped handoff files. Read saved answers; never modify study results."""
from collections import Counter
from datetime import datetime, timezone


WORKFLOW_REVISION = "unified-evaluation-export-v1"


def _iso(value):
    if not value:
        return None
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    return str(value)


def ranking_export(session):
    """Reconstruct final choices from saved events, including pre-export UI versions.

    Checkpoints contribute only presentation/timing metadata, never answers.
    Generator scores, validation labels, paths and access credentials are excluded.
    """
    histories, checkpoint, seen = {}, {}, set()
    active_ms = 0
    for event in session.get("events", []):
        payload = event.get("payload") or {}
        event_id = payload.get("_event_id")
        if event_id and event_id in seen:
            continue
        if event_id:
            seen.add(event_id)
        active_ms = max(active_ms, event.get("client_elapsed_ms") or 0)
        if event.get("event_type") == "evaluation_checkpoint":
            checkpoint = payload
        if event.get("event_type") != "pairwise_choice" or not payload.get("pair_id"):
            continue
        histories.setdefault(payload["pair_id"], []).append({
            key: payload.get(key) for key in (
                "choice", "undecided_reason", "winner_id", "decision_ms",
                "changed_answer", "previous_choice", "previous_undecided_reason",
                "last_answered_at", "recovered_from_local_draft")
        } | {"recorded_at": _iso(event.get("created_at"))})

    candidates = {item["id"]: {key: item.get(key) for key in ("id", "title", "summary")}
                  for item in session.get("candidates", [])}
    stats = checkpoint.get("question_stats") or {}
    questions = []
    for number, pair in enumerate(session.get("pairs", []), 1):
        pair_id = pair["pair_id"]
        attempts = histories.get(pair_id, [])
        last = attempts[-1] if attempts else {}
        display = stats.get(pair_id) or {}
        questions.append({
            "question_number": number, "pair_id": pair_id,
            "difficulty": pair.get("difficulty"),
            "left_hypothesis": candidates.get(pair["left_id"], {"id": pair["left_id"]}),
            "right_hypothesis": candidates.get(pair["right_id"], {"id": pair["right_id"]}),
            "answer": last.get("choice"), "cannot_judge_reason": last.get("undecided_reason"),
            "winner_id": last.get("winner_id"),
            "first_presented_at": display.get("first_presented_at"),
            "last_presented_at": display.get("last_presented_at"),
            "view_count": display.get("view_count"),
            "first_answered_at": (attempts[0].get("last_answered_at") or attempts[0]["recorded_at"]) if attempts else None,
            "last_answered_at": last.get("last_answered_at") or last.get("recorded_at"),
            "answer_submission_count": len(attempts),
            "modification_count": sum(bool(item.get("changed_answer")) for item in attempts),
            "total_decision_time_ms": sum(item.get("decision_ms") or 0 for item in attempts),
            "attempts": attempts,
        })
    counts = Counter(item["answer"] for item in questions if item["answer"])
    completed = session["status"] == "completed"
    return {
        "schema_version": "neurodiscovery-ranking-handoff-v1",
        "participant": {"id": session["participant_id"], "experience_years": session.get("participant_experience")},
        "study": {key: session.get(key) for key in ("study_id", "case_study", "protocol_version")},
        "session": {key: session.get(key) for key in (
            "session_id", "status", "session_number", "required_sessions", "condition",
            "completion_reason", "random_seed", "candidate_manifest_hash", "graph_snapshot_hash")},
        "timing": {
            "created_at": _iso(session.get("started_at")),
            "completed_at": _iso(session.get("submitted_at")) if completed else None,
            "target_active_time_ms": 1000 * (session.get("target_active_seconds") or 600),
            "active_answering_time_ms": round(1000 * (session.get("active_seconds") or 0) if completed else max(active_ms, checkpoint.get("active_ms") or 0)),
            "total_study_open_time_ms": round(1000 * (session.get("wall_seconds") or 0)) if completed else checkpoint.get("open_ms"),
        },
        "language": checkpoint.get("language"),
        "summary": {"answered": sum(counts.values()), "available_questions": len(questions),
                    "left_selected": counts["left"], "right_selected": counts["right"],
                    "cannot_judge": counts["undecided"]},
        "questions": questions,
        "final_ranking": session.get("final_ranking") if completed else None,
    }


def build_evaluation_export(payload, discovery, ranking=None):
    code = str(payload.get("participant_code") or "").strip()
    if not code or len(code) > 100:
        raise ValueError("A participant name/code is required.")
    discovery_refs = payload.get("discovery_sessions") or []
    ranking_refs = payload.get("ranking_session_ids") or []
    ranking_studies = payload.get("ranking_study_ids") or []
    if not isinstance(discovery_refs, list) or not isinstance(ranking_refs, list) or len(discovery_refs) + len(ranking_refs) > 100:
        raise ValueError("Invalid session references.")
    if not isinstance(ranking_studies, list) or len(ranking_studies) > 10:
        raise ValueError("Invalid ranking study references.")
    discovery_results, ranking_results, seen = [], [], set()
    for ref in discovery_refs:
        if not isinstance(ref, dict):
            raise ValueError("Invalid discovery session reference.")
        session_id = str(ref.get("id") or "")
        if session_id in seen:
            continue
        result = discovery.export(session_id, str(ref.get("token") or ""))
        if result["session"]["profile"]["code"].strip() != code:
            raise ValueError("Session participant does not match the export name/code.")
        discovery_results.append(result)
        seen.add(session_id)
    if (ranking_refs or ranking_studies) and ranking is None:
        raise ValueError("Human Evaluation 2 is unavailable in this service. Export from the desktop client.")
    scopes = {(str(study_id), code) for study_id in ranking_studies}
    for session_id in set(ranking_refs):
        anchor = ranking.get_session(str(session_id))
        if anchor["participant_id"].strip() != code:
            raise ValueError("Session participant does not match the export name/code.")
        scopes.add((anchor["study_id"], anchor["participant_id"]))
    # Include previous timed sessions, not just the one currently on screen.
    ranking_seen = set()
    for study_id, participant_id in sorted(scopes):
        for row in ranking.list_sessions(study_id):
            if row["participant_id"].strip() != participant_id.strip() or row["session_id"] in ranking_seen:
                continue
            ranking_results.append(ranking_export(ranking.get_session(row["session_id"], include_events=True)))
            ranking_seen.add(row["session_id"])
    if not discovery_results and not ranking_results:
        raise ValueError("No saved evaluations were found for this participant.")
    discovery_complete = bool(discovery_results) and all(item["session"]["stage"] == "complete" for item in discovery_results)
    required = max((item["session"]["required_sessions"] or 1 for item in ranking_results), default=0)
    completed_numbers = {item["session"]["session_number"] for item in ranking_results if item["session"]["status"] == "completed"}
    ranking_complete = bool(required) and set(range(1, required + 1)) <= completed_numbers and all(item["session"]["status"] == "completed" for item in ranking_results)
    return {
        "schema_version": "neurodiscovery-human-evaluations-v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "participant_code": code,
        "completion_status": "complete" if discovery_complete and ranking_complete else "partial",
        "human_evaluation_1": {
            "status": "complete" if discovery_complete else "in_progress" if discovery_results else "not_included",
            "session_count": len(discovery_results), "sessions": discovery_results,
        },
        "human_evaluation_2": {
            "status": "complete" if ranking_complete else "in_progress" if ranking_results else "not_included",
            "completed_sessions": len(completed_numbers), "required_sessions": required or None,
            "session_count": len(ranking_results), "sessions": ranking_results,
        },
    }
