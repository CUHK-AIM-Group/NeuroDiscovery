"""Apply reviewer-authored source resolutions to unresolved Case-2 decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPEC_SCHEMA = "case2_manual_source_resolution_spec.v1"
COMPONENTS = {
    "genetic_or_pathway",
    "brain_imaging_or_physiology",
    "longitudinal_clinical_or_cognitive_outcome",
    "mediation_or_causal_chain",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _file_hash(path: Path) -> str:
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
        raise ValueError("source resolution spec schema mismatch")
    supplement_path = root / "source_supplements.json"
    supplement_hash = _file_hash(supplement_path)
    if supplement_hash != spec.get("source_supplements_sha256"):
        raise ValueError("source supplement hash does not match resolution spec")
    supplements = _load(supplement_path)
    supplement_by_key = {
        str(entry["paper_key"]): entry for entry in supplements.get("entries") or []
    }

    manifest = _load(root / "manifest.json")
    paper_locations: dict[int, tuple[int, dict[str, Any]]] = {}
    for shard in manifest.get("shards") or []:
        shard_index = int(shard["shard_index"])
        task = _load(root / f"shard_{shard_index:02d}_task.json")
        for paper in task.get("papers") or []:
            paper_locations[int(paper["paper_index"])] = (shard_index, paper)

    changed_by_shard: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for resolution in spec.get("resolutions") or []:
        paper_index = int(resolution["paper_index"])
        if paper_index in seen:
            raise ValueError(f"duplicate source resolution for paper {paper_index}")
        seen.add(paper_index)
        if paper_index not in paper_locations:
            raise ValueError(f"paper {paper_index} is not in the campaign")
        shard_index, paper = paper_locations[paper_index]
        paper_key = str(paper["paper_key"])
        if paper_key != resolution.get("paper_key"):
            raise ValueError(f"paper key mismatch for {paper_index}")
        if paper_key not in supplement_by_key:
            raise ValueError(f"paper {paper_index} has no source supplement")
        result_path = root / f"shard_{shard_index:02d}_result.json"
        result = changed_by_shard.setdefault(shard_index, _load(result_path))
        decision = next(
            row for row in result["decisions"] if int(row["paper_index"]) == paper_index
        )
        if decision.get("paper_decision") != "unresolved_source":
            raise ValueError(f"paper {paper_index} is no longer unresolved_source")

        new_decision = str(resolution.get("paper_decision") or "")
        if new_decision not in {"retain", "remove"}:
            raise ValueError(f"invalid resolved decision for paper {paper_index}")
        evidence = resolution.get("component_evidence") or {
            key: [] for key in COMPONENTS
        }
        if set(evidence) != COMPONENTS:
            raise ValueError(f"component evidence mismatch for paper {paper_index}")
        claim_ids = [
            str(claim["claim_id"])
            for claim in paper.get("currently_case2_labelled_claims") or []
        ]
        retained_claim_ids = {
            str(value) for value in (resolution.get("retain_claim_ids") or [])
        }
        if not retained_claim_ids <= set(claim_ids):
            raise ValueError(f"unknown retained claim for paper {paper_index}")
        if new_decision == "retain":
            if not retained_claim_ids or any(not evidence[key] for key in COMPONENTS):
                raise ValueError(f"retained paper {paper_index} lacks chain evidence")
        elif retained_claim_ids:
            raise ValueError(f"removed paper {paper_index} retains a claim")

        decision["paper_decision"] = new_decision
        decision["confidence"] = float(resolution["confidence"])
        decision["component_evidence"] = evidence
        decision["rationale"] = str(resolution["rationale"])
        decision["claim_decisions"] = [
            {
                "claim_id": claim_id,
                "decision": "retain" if claim_id in retained_claim_ids else "remove",
                "reason": (
                    "This claim forms or states part of the source-resolved complete "
                    "paper-level Case-2 chain."
                    if claim_id in retained_claim_ids
                    else "Recovered source evidence does not verify this claim as part "
                    "of a complete paper-level Case-2 chain."
                ),
            }
            for claim_id in claim_ids
        ]
        supplement = supplement_by_key[paper_key]
        decision["source_resolution"] = {
            "reviewer_id": str(spec.get("reviewer_id") or "codex_root_manual"),
            "source_supplements_sha256": supplement_hash,
            "source_text_sha256": supplement.get("source_text_sha256"),
            "decision_basis": "manual PubMed abstract adjudication",
        }

    expected = {int(value) for value in spec.get("expected_unresolved_paper_indices") or []}
    if seen != expected:
        raise ValueError(
            f"resolution coverage mismatch: expected={sorted(expected)}, seen={sorted(seen)}"
        )

    for shard_index, result in sorted(changed_by_shard.items()):
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
                "resolved_papers": len(seen),
                "changed_shards": sorted(changed_by_shard),
                "retained": sum(
                    item["paper_decision"] == "retain"
                    for item in spec.get("resolutions") or []
                ),
                "removed": sum(
                    item["paper_decision"] == "remove"
                    for item in spec.get("resolutions") or []
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
