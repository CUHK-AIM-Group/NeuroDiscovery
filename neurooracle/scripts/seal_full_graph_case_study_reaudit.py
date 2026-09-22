"""Seal finalized re-audit rows with immutable evidence and decision hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    CONTRACT_FIELD_NAMES,
    claim_contract_fields,
    contract_context,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    RUBRIC_VERSION,
    compact_json,
    connect,
    set_meta,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


def seal(connection: Any, *, apply: bool) -> dict[str, Any]:
    scanned = 0
    already_sealed = 0
    to_update: list[tuple[str, str]] = []
    for claim_id, paper_key, payload_json, review_json in connection.execute(
        """
        SELECT claim_id, paper_key, payload_json, review_json
        FROM claims WHERE review_status='final_complete' ORDER BY graph_ordinal
        """
    ):
        scanned += 1
        review = json.loads(review_json)
        if review.get("rubric_version") != RUBRIC_VERSION:
            raise ValueError(f"final review uses a different rubric: {claim_id}")
        expected = claim_contract_fields(
            paper_key=str(paper_key),
            payload=json.loads(payload_json),
            labels=review.get("claim_case_study_ids") or [],
            gates=review.get("gates") or {},
            rubric_version=RUBRIC_VERSION,
            case_study_ids=CASE_STUDY_IDS,
        )
        present = {
            key: str(review.get(key) or "") for key in CONTRACT_FIELD_NAMES
        }
        if any(present.values()):
            if any(
                present[key] != expected[key] for key in CONTRACT_FIELD_NAMES
            ):
                raise ValueError(f"existing immutable contract mismatch: {claim_id}")
            already_sealed += 1
            continue
        review.update(expected)
        to_update.append((compact_json(review), str(claim_id)))
    if apply:
        connection.executemany(
            "UPDATE claims SET review_json=? WHERE claim_id=? AND review_status='final_complete'",
            to_update,
        )
        set_meta(
            connection,
            "audit_contract",
            contract_context(
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            ),
        )
        connection.commit()
    return {
        "mode": "apply" if apply else "dry_run",
        "finalized_rows_scanned": scanned,
        "already_sealed": already_sealed,
        "rows_to_seal": len(to_update),
        "rows_sealed": len(to_update) if apply else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    connection = connect(args.output_dir.resolve() / "reaudit.sqlite")
    try:
        report = seal(connection, apply=args.apply)
    finally:
        connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
