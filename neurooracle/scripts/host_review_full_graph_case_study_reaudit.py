"""Export/import host-agent semantic review batches for the full KG ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    compact_json,
    connect,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    claim_contract_fields,
    claim_input_contract_fields,
    contract_context,
)
from neurooracle.scripts.run_full_graph_case_study_reaudit import review_payload, validate_response
from neurooracle.scripts.propagate_exact_case_study_reaudit import (
    semantic_fingerprint,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS
from neurooracle.src.case_study_membership_policy import (
    GATE_NAMES,
    GATE_REQUIREMENTS,
    RUBRIC_VERSION,
)


LABEL_GATE_MAP = GATE_REQUIREMENTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_id(claim_ids: list[str]) -> str:
    digest = hashlib.sha256("\n".join(claim_ids).encode("utf-8")).hexdigest()[:16]
    return f"host_reaudit_{digest}"


def export_batch(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = args.output_dir.resolve() / "reaudit.sqlite"
    connection = connect(ledger_path)
    try:
        offset = int(getattr(args, "offset", 0))
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if offset and (args.duplicate_first or args.unique_semantic):
            raise ValueError(
                "offset cannot be combined with duplicate-first or unique-semantic"
            )
        if args.status == "secondary_pending":
            raw_rows = connection.execute(
                """
                SELECT claim_id, paper_key, payload_json FROM claims
                WHERE review_status='secondary_pending'
                ORDER BY graph_ordinal LIMIT ? OFFSET ?
                """,
                (args.count, offset),
            ).fetchall()
        else:
            if args.duplicate_first:
                counts: Counter[str] = Counter()
                for row in connection.execute(
                    """
                    SELECT paper_key, payload_json FROM claims
                    WHERE review_status IN ('pending', 'primary_failed')
                    ORDER BY graph_ordinal
                    """
                ):
                    counts[
                        semantic_fingerprint(str(row[0]), json.loads(str(row[1])))
                    ] += 1
                raw_rows = []
                seen: set[str] = set()
                for row in connection.execute(
                    """
                    SELECT claim_id, paper_key, payload_json FROM claims
                    WHERE review_status IN ('pending', 'primary_failed')
                    ORDER BY graph_ordinal
                    """
                ):
                    fingerprint = semantic_fingerprint(
                        str(row[1]), json.loads(str(row[2]))
                    )
                    if counts[fingerprint] < 2 or fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    raw_rows.append(row)
                    if len(raw_rows) >= args.count:
                        break
            elif args.unique_semantic:
                raw_rows = []
                seen: set[str] = set()
                for row in connection.execute(
                    """
                    SELECT claim_id, paper_key, payload_json FROM claims
                    WHERE review_status IN ('pending', 'primary_failed')
                    ORDER BY graph_ordinal
                    """
                ):
                    fingerprint = semantic_fingerprint(
                        str(row[1]), json.loads(str(row[2]))
                    )
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    raw_rows.append(row)
                    if len(raw_rows) >= args.count:
                        break
            else:
                raw_rows = connection.execute(
                    """
                    SELECT claim_id, paper_key, payload_json FROM claims
                    WHERE review_status IN ('pending', 'primary_failed')
                    ORDER BY graph_ordinal LIMIT ? OFFSET ?
                    """,
                    (args.count, offset),
                ).fetchall()
        rows = [(str(row[0]), str(row[1]), str(row[2])) for row in raw_rows]
        if not rows:
            raise RuntimeError("no unreviewed claims remain")
        payload = review_payload(
            connection,
            rows,
            abstract_chars=args.abstract_chars,
            paper_context_limit=args.paper_context_limit,
        )
        claim_ids = [row[0] for row in rows]
        claim_input_contracts = {
            claim_id: claim_input_contract_fields(
                paper_key=paper_key,
                payload=json.loads(payload_json),
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            )
            for claim_id, paper_key, payload_json in rows
        }
        task = {
            "schema_version": "full_graph_case_study_host_task.v2",
            "rubric_version": RUBRIC_VERSION,
            "audit_contract": {
                **contract_context(
                    rubric_version=RUBRIC_VERSION,
                    case_study_ids=CASE_STUDY_IDS,
                ),
                "claims": claim_input_contracts,
            },
            "task_id": task_id(claim_ids),
            "review_stage": "secondary" if args.status == "secondary_pending" else "primary",
            "created_at": utc_now(),
            "claim_ids": claim_ids,
            "audit_payload": payload,
            "expected_result": {
                "schema_version": "full_graph_case_study_host_result.v1",
                "task_id": task_id(claim_ids),
                "reviews": "same validated objects and order as claims_to_review",
            },
        }
        if args.status == "secondary_pending":
            task["primary_reviews"] = {
                claim_id: json.loads(
                    connection.execute(
                        "SELECT review_json FROM claims WHERE claim_id=?", (claim_id,)
                    ).fetchone()[0]
                )
                for claim_id in claim_ids
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "task": str(args.output.resolve()),
            "task_id": task["task_id"],
            "claims": len(rows),
            "first_claim": claim_ids[0],
            "last_claim": claim_ids[-1],
        }
    finally:
        connection.close()


def import_batch(args: argparse.Namespace) -> dict[str, Any]:
    task = json.loads(args.task.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if task.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("task rubric_version does not match the active frozen rubric")
    if result.get("task_id") != task.get("task_id"):
        raise ValueError("task_id mismatch")
    claim_ids = [str(value) for value in task.get("claim_ids") or []]
    payload_claims = (task.get("audit_payload") or {}).get("claims_to_review") or []
    if claim_ids != [str(row.get("claim_id") or "") for row in payload_claims]:
        raise ValueError("task claim inventory is internally inconsistent")
    expected_rows = [(claim_id, "", "") for claim_id in claim_ids]
    reviews = validate_response(expected_rows, result)
    ledger_path = args.output_dir.resolve() / "reaudit.sqlite"
    connection = connect(ledger_path)
    try:
        current = connection.execute(
            f"SELECT claim_id, paper_key, payload_json, review_status, review_json FROM claims WHERE claim_id IN ({','.join('?' for _ in claim_ids)})",
            claim_ids,
        ).fetchall()
        if len(current) != len(claim_ids):
            raise ValueError("one or more task claims are absent from the ledger")
        secondary = task.get("review_stage") == "secondary"
        replace_reviewed = bool(getattr(args, "replace_reviewed", False))
        correction_reason = str(getattr(args, "correction_reason", "") or "").strip()
        if replace_reviewed and not correction_reason:
            raise ValueError("replace-reviewed requires a non-empty correction reason")
        if replace_reviewed:
            # A secondary correction is valid only after the adjudication has
            # already finalized the claim. Primary corrections retain their
            # broader historical behavior for repairing any reviewed state.
            allowed_statuses = (
                {"final_complete"}
                if secondary
                else {"pending", "primary_failed", "secondary_pending", "final_complete"}
            )
        else:
            allowed_statuses = (
                {"secondary_pending"} if secondary else {"pending", "primary_failed"}
            )
        review_stage = (
            ("secondary_correction" if secondary else "primary_correction")
            if replace_reviewed
            else ("secondary_adjudication" if secondary else "primary")
        )
        reviewer_id = (
            (
                "codex_host_manual_secondary_correction"
                if secondary
                else "codex_host_manual_correction"
            )
            if replace_reviewed
            else ("codex_host_manual_secondary" if secondary else "codex_host_manual")
        )
        current_by_id = {
            str(claim_id): (
                str(paper_key),
                json.loads(payload_json),
                str(status),
                str(review_json) if review_json is not None else None,
            )
            for claim_id, paper_key, payload_json, status, review_json in current
        }
        disallowed = {
            claim_id: status
            for claim_id, (_paper_key, _payload, status, _review_json) in current_by_id.items()
            if status not in allowed_statuses
        }
        if disallowed:
            raise ValueError(f"task contains already-reviewed claims: {disallowed}")
        task_contracts = ((task.get("audit_contract") or {}).get("claims") or {})
        if set(task_contracts) != set(claim_ids):
            raise ValueError("task is missing the immutable per-claim audit contract")
        for claim_id in claim_ids:
            paper_key, payload, _status, _review_json = current_by_id[claim_id]
            expected_contract = claim_input_contract_fields(
                paper_key=paper_key,
                payload=payload,
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            )
            if task_contracts[claim_id] != expected_contract:
                raise ValueError(f"claim evidence changed after task export: {claim_id}")
        reviewed_at = utc_now()
        for review in reviews:
            if secondary and review["needs_secondary_review"]:
                raise ValueError("secondary adjudication must return needs_secondary_review=false")
            stored = {
                **review,
                "rubric_version": RUBRIC_VERSION,
                "review_stage": review_stage,
                "reviewer_id": reviewer_id,
                "reasoning_effort": "host_session_semantic_review",
                "reviewed_at": reviewed_at,
                "task_id": task["task_id"],
            }
            paper_key, payload, previous_status, previous_review_json = current_by_id[
                review["claim_id"]
            ]
            if replace_reviewed:
                stored.update(
                    {
                        "correction_reason": correction_reason,
                        "superseded_review_status": previous_status,
                        "superseded_review_sha256": (
                            hashlib.sha256(previous_review_json.encode("utf-8")).hexdigest()
                            if previous_review_json is not None
                            else None
                        ),
                    }
                )
            stored.update(
                claim_contract_fields(
                    paper_key=paper_key,
                    payload=payload,
                    labels=review["claim_case_study_ids"],
                    gates=review["gates"],
                    rubric_version=RUBRIC_VERSION,
                    case_study_ids=CASE_STUDY_IDS,
                )
            )
            if secondary:
                stored["primary_review"] = task["primary_reviews"][review["claim_id"]]
            cursor = connection.execute(
                """
                UPDATE claims SET review_status=?, review_json=?, reviewed_at=?
                WHERE claim_id=? AND review_status=?
                """,
                (
                    "final_complete"
                    if secondary or not review["needs_secondary_review"]
                    else "secondary_pending",
                    compact_json(stored),
                    reviewed_at,
                    review["claim_id"],
                    previous_status,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"claim status changed during immutable import: {review['claim_id']}"
                )
        connection.commit()
        return {
            "task_id": task["task_id"],
            "claims_imported": len(reviews),
            "reviewer_id": reviewer_id,
            "replace_reviewed": replace_reviewed,
            "remaining_unreviewed": connection.execute(
                "SELECT COUNT(*) FROM claims WHERE review_status!='final_complete'"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def expand_compact_decisions(args: argparse.Namespace) -> dict[str, Any]:
    """Expand human-authored compact decisions into the strict result schema.

    This is deliberately mechanical: it never assigns a Case Study label. It
    only derives the mandatory gate corresponding to each explicitly authored
    label and applies optional, explicit gate overrides (for example, a
    longitudinal study that does not qualify for prognosis).
    """
    task = json.loads(args.task.read_text(encoding="utf-8"))
    compact = json.loads(args.decisions.read_text(encoding="utf-8"))
    claim_ids = [str(value) for value in task.get("claim_ids") or []]
    decisions = compact.get("reviews") or compact.get("decisions") or []
    if claim_ids != [str(row.get("claim_id") or "") for row in decisions]:
        raise ValueError("compact decision order or IDs do not match the task")

    reviews: list[dict[str, Any]] = []
    for decision in decisions:
        labels = decision.get("claim_case_study_ids", decision.get("labels", []))
        gates = {name: False for name in GATE_NAMES}
        for label in labels:
            gate = LABEL_GATE_MAP.get(str(label))
            if gate:
                gates[gate] = True
        overrides = decision.get("gate_overrides") or {}
        unknown = set(overrides) - set(GATE_NAMES)
        if unknown:
            raise ValueError(f"unknown gate overrides: {sorted(unknown)}")
        gates.update({name: bool(value) for name, value in overrides.items()})
        reviews.append(
            {
                "claim_id": str(decision["claim_id"]),
                "claim_case_study_ids": [str(value) for value in labels],
                "confidence": float(decision["confidence"]),
                "reason": str(decision["reason"]),
                "needs_secondary_review": bool(
                    decision.get("needs_secondary_review", False)
                ),
                "gates": gates,
            }
        )

    result = {
        "schema_version": "full_graph_case_study_host_result.v1",
        "task_id": str(task.get("task_id") or ""),
        "reviews": reviews,
    }
    expected_rows = [(claim_id, "", "") for claim_id in claim_ids]
    validate_response(expected_rows, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "task_id": result["task_id"],
        "claims_expanded": len(reviews),
        "result": str(args.output.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export_parser.add_argument("--count", type=int, default=20)
    export_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Skip this many eligible ledger rows before exporting. Useful for "
            "reserving a disjoint later slice while an API worker processes the "
            "earliest pending rows."
        ),
    )
    export_parser.add_argument(
        "--status", choices=("unreviewed", "secondary_pending"), default="unreviewed"
    )
    export_parser.add_argument("--abstract-chars", type=int, default=2500)
    export_parser.add_argument("--paper-context-limit", type=int, default=20)
    export_parser.add_argument(
        "--unique-semantic",
        action="store_true",
        help="export one representative per exact same-paper semantic fingerprint",
    )
    export_parser.add_argument(
        "--duplicate-first",
        action="store_true",
        help="export representatives of pending exact-duplicate groups first",
    )
    export_parser.add_argument("--output", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    import_parser.add_argument("--task", type=Path, required=True)
    import_parser.add_argument("--result", type=Path, required=True)
    import_parser.add_argument(
        "--replace-reviewed",
        action="store_true",
        help=(
            "replace an existing primary review after a manual QA correction; "
            "the immutable input contract is still verified"
        ),
    )
    import_parser.add_argument(
        "--correction-reason",
        default="",
        help="required audit-trail reason when --replace-reviewed is used",
    )
    expand_parser = subparsers.add_parser("expand")
    expand_parser.add_argument("--task", type=Path, required=True)
    expand_parser.add_argument("--decisions", type=Path, required=True)
    expand_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "export":
        output = export_batch(args)
    elif args.command == "import":
        output = import_batch(args)
    else:
        output = expand_compact_decisions(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
