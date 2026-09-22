"""Apply root adjudications for documented Case-2 cross-QA disagreements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPEC_SCHEMA = "case2_manual_qa_adjudication_spec.v1"
COMPONENTS = {
    "genetic_or_pathway",
    "brain_imaging_or_physiology",
    "longitudinal_clinical_or_cognitive_outcome",
    "mediation_or_causal_chain",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = _load(args.spec)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError("QA adjudication spec schema mismatch")
    supplement_hash = _hash(root / "source_supplements.json")
    if supplement_hash != spec.get("source_supplements_sha256"):
        raise ValueError("source supplement hash mismatch")
    for qa in spec.get("qa_files") or []:
        qa_path = root / str(qa["name"])
        if _hash(qa_path) != qa.get("sha256"):
            raise ValueError(f"QA file hash mismatch: {qa_path.name}")

    manifest = _load(root / "manifest.json")
    locations: dict[int, tuple[int, dict[str, Any]]] = {}
    for shard in manifest.get("shards") or []:
        shard_index = int(shard["shard_index"])
        task = _load(root / f"shard_{shard_index:02d}_task.json")
        for paper in task.get("papers") or []:
            locations[int(paper["paper_index"])] = (shard_index, paper)

    changed: dict[int, dict[str, Any]] = {}
    for item in spec.get("adjudications") or []:
        paper_index = int(item["paper_index"])
        shard_index, paper = locations[paper_index]
        if paper.get("paper_key") != item.get("paper_key"):
            raise ValueError(f"paper key mismatch for {paper_index}")
        path = root / f"shard_{shard_index:02d}_result.json"
        result = changed.setdefault(shard_index, _load(path))
        decision = next(
            row for row in result["decisions"] if int(row["paper_index"]) == paper_index
        )
        if decision.get("paper_decision") != item.get("expected_current_decision"):
            raise ValueError(f"unexpected current decision for paper {paper_index}")
        new_decision = str(item["paper_decision"])
        if new_decision not in {"retain", "remove"}:
            raise ValueError(f"invalid adjudication for paper {paper_index}")
        evidence = item.get("component_evidence") or {
            key: [] for key in COMPONENTS
        }
        if set(evidence) != COMPONENTS:
            raise ValueError(f"component evidence mismatch for paper {paper_index}")
        claim_ids = [
            str(claim["claim_id"])
            for claim in paper.get("currently_case2_labelled_claims") or []
        ]
        retained = {str(value) for value in item.get("retain_claim_ids") or []}
        if not retained <= set(claim_ids):
            raise ValueError(f"unknown retained claim for paper {paper_index}")
        if new_decision == "retain" and (
            not retained or any(not evidence[key] for key in COMPONENTS)
        ):
            raise ValueError(f"retained paper {paper_index} lacks full evidence")
        if new_decision == "remove" and retained:
            raise ValueError(f"removed paper {paper_index} retains a claim")

        decision["paper_decision"] = new_decision
        decision["confidence"] = float(item["confidence"])
        decision["component_evidence"] = evidence
        decision["rationale"] = str(item["rationale"])
        decision["claim_decisions"] = [
            {
                "claim_id": claim_id,
                "decision": "retain" if claim_id in retained else "remove",
                "reason": (
                    "Cross-QA and source adjudication verified this claim as part "
                    "of the complete Case-2 chain."
                    if claim_id in retained
                    else "Cross-QA/source adjudication did not verify this claim as "
                    "part of a complete Case-2 chain."
                ),
            }
            for claim_id in claim_ids
        ]
        decision["qa_adjudication"] = {
            "reviewer_id": str(spec.get("reviewer_id") or "codex_root_manual"),
            "source_supplements_sha256": supplement_hash,
            "qa_files": [dict(value) for value in spec.get("qa_files") or []],
            "decision_basis": str(item.get("decision_basis") or "cross-QA adjudication"),
        }

    for shard_index, result in sorted(changed.items()):
        if "result_sha256" in result:
            unhashed = {
                key: value for key, value in result.items() if key != "result_sha256"
            }
            result["result_sha256"] = _canonical_hash(unhashed)
        path = root / f"shard_{shard_index:02d}_result.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "adjudicated_papers": len(spec.get("adjudications") or []),
                "changed_shards": sorted(changed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
