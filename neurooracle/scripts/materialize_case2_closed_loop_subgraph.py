"""Stream a deterministic Case-2-scoped graph from the canonical claim store.

The canonical multi-gigabyte graph cannot be safely materialized by the legacy
``json.load`` loader on the benchmark host.  This projection retains every
claim carrying the canonical Case Study 2 membership, its two endpoints and
the corresponding claim-derived semantic edge.  It never reads experiment
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASE_STUDY_ID = "case2_pathway_mediation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scope_ids(claim: Mapping[str, Any]) -> set[str]:
    values = claim.get("claim_case_study_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(value).strip() for value in values if value}


def _endpoint_tags(declared_type: str, name: str) -> list[str]:
    value = str(declared_type or "").strip().casefold()
    label = str(name or "").casefold()
    tags: set[str] = set()
    if any(token in value for token in ("gene", "genetic", "variant", "pathway")):
        tags.add("gene")
    if any(token in value for token in ("imaging", "biomarker", "marker")):
        tags.add("biomarker")
    if any(token in value for token in ("connect", "network")):
        tags.add("connectivity")
    if any(token in value for token in ("brain", "region", "anatom")):
        tags.add("neuroanatomy")
    if any(token in value for token in ("outcome", "cognit", "clinical", "score", "symptom")):
        tags.update(("cognitive_function", "treatment_outcome"))
    if "disease" in value or "disorder" in value:
        tags.add("disease")
    if any(token in label for token in ("mri", "pet", "suvr", "amyloid", "tau", "volume", "thickness", "connectivity")):
        tags.add("imaging_feature")
    if not tags:
        tags.add("dataset_variable")
    return sorted(tags)


def _endpoint_node(endpoint_id: str, name: str, declared_type: str) -> dict[str, Any]:
    return {
        "id": endpoint_id,
        "preferred_name": str(name or endpoint_id),
        "semantic_types": [],
        "domain_tags": _endpoint_tags(declared_type, name),
        "source_vocab": "canonical_claim_endpoint",
        "definition": "",
        "aliases": [],
        "external_ids": {},
        "spatial_mapping": None,
        "metadata": {"declared_claim_type": str(declared_type or "")},
    }


def _compact_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    nested = dict(claim.get("metadata") or {})
    compact_nested = {
        key: nested.get(key)
        for key in (
            "subject_type", "object_type", "subject_canonical_hint",
            "object_canonical_hint", "claim_case_study_ids",
            "paper_case_study_ids", "case_study_membership_schema_version",
        )
        if nested.get(key) not in (None, "", [])
    }
    return {
        "id": str(claim.get("id") or ""),
        "subject_id": str(claim.get("subject_id") or ""),
        "subject_name": str(claim.get("subject_name") or ""),
        "predicate": str(claim.get("predicate") or "associated_with"),
        "object_id": str(claim.get("object_id") or ""),
        "object_name": str(claim.get("object_name") or ""),
        "negated": bool(claim.get("negated", False)),
        "confidence": float(claim.get("confidence") or 0.0),
        "evidence": claim.get("evidence") or {},
        "source_paper": claim.get("source_paper") or {},
        "raw_text": str(claim.get("raw_text") or ""),
        "metadata": compact_nested,
        "paper_case_study_ids": list(claim.get("paper_case_study_ids") or []),
        "claim_case_study_ids": list(claim.get("claim_case_study_ids") or []),
    }


def _claim_node(compact: Mapping[str, Any]) -> dict[str, Any]:
    preferred = " ".join(
        str(compact.get(key) or "")
        for key in ("subject_name", "predicate", "object_name")
    ).strip()
    return {
        "id": compact["id"],
        "preferred_name": preferred or str(compact["id"]),
        "semantic_types": [],
        "domain_tags": ["claim"],
        "source_vocab": "claim_extraction",
        "definition": str(compact.get("raw_text") or ""),
        "aliases": [],
        "external_ids": {},
        "spatial_mapping": None,
        "metadata": dict(compact),
    }


def _edge(compact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": compact["subject_id"],
        "target_id": compact["object_id"],
        "relation_type": compact["predicate"] or "associated_with",
        "source": "claim_extraction",
        "confidence": float(compact.get("confidence") or 0.0),
        "evidence_ref": str((compact.get("source_paper") or {}).get("pmid") or ""),
        "metadata": {"claim_id": compact["id"]},
    }


def materialize(claims_path: Path, output_path: Path, *, source_pin: Mapping[str, Any] | None = None) -> dict[str, Any]:
    claims_path = claims_path.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_bytes") == claims_path.stat().st_size
            and manifest.get("output_sha256") == sha256_file(output_path)
        ):
            return manifest
        raise ValueError("Existing scoped graph does not match its manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep staging basenames deliberately short.  The frozen benchmark lives on
    # a deep UNC path where expanding the full output basename can cross the
    # legacy Windows MAX_PATH boundary even though the final artifact does not.
    work = output_path.parent / f".c2g_{os.getpid()}"
    claim_nodes_path = work.with_suffix(work.suffix + ".claims.jsonl")
    edges_path = work.with_suffix(work.suffix + ".edges.jsonl")
    if any(path.exists() for path in (work, claim_nodes_path, edges_path)):
        raise FileExistsError("Scoped-graph staging artifact already exists")
    endpoints: dict[str, dict[str, Any]] = {}
    claim_count = 0
    invalid_endpoint_claims = 0
    with claims_path.open("r", encoding="utf-8") as source, claim_nodes_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as claim_out, edges_path.open("w", encoding="utf-8", newline="\n") as edge_out:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            claim = json.loads(line)
            if CASE_STUDY_ID not in _scope_ids(claim):
                continue
            compact = _compact_claim(claim)
            claim_id = str(compact["id"])
            subject_id = str(compact["subject_id"])
            object_id = str(compact["object_id"])
            if not claim_id or not subject_id or not object_id or subject_id == object_id:
                invalid_endpoint_claims += 1
                continue
            nested = dict(claim.get("metadata") or {})
            endpoints.setdefault(
                subject_id,
                _endpoint_node(subject_id, str(compact["subject_name"]), str(nested.get("subject_type") or "")),
            )
            endpoints.setdefault(
                object_id,
                _endpoint_node(object_id, str(compact["object_name"]), str(nested.get("object_type") or "")),
            )
            claim_out.write(json.dumps({claim_id: _claim_node(compact)}, ensure_ascii=False, separators=(",", ":")) + "\n")
            edge_out.write(json.dumps(_edge(compact), ensure_ascii=False, separators=(",", ":")) + "\n")
            claim_count += 1
            if claim_count % 25000 == 0:
                print(f"materialized claim records: {claim_count}", flush=True)

    metadata = {
        "version": "case2-scoped-claim-projection-v1",
        "case_study_id": CASE_STUDY_ID,
        "source_claim_store": str(claims_path),
        "source_bytes": claims_path.stat().st_size,
        "source_sha256": (source_pin or {}).get("sha256"),
        "claim_count": claim_count,
        "endpoint_count": len(endpoints),
        "invalid_endpoint_claims": invalid_endpoint_claims,
        "result_fields_used": [],
    }
    with work.open("w", encoding="utf-8", newline="\n") as output:
        output.write('{"metadata":')
        output.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
        output.write(',"concepts":{')
        first = True
        for endpoint_id in sorted(endpoints):
            if not first:
                output.write(",")
            first = False
            output.write(json.dumps(endpoint_id, ensure_ascii=False))
            output.write(":")
            output.write(json.dumps(endpoints[endpoint_id], ensure_ascii=False, separators=(",", ":")))
        with claim_nodes_path.open("r", encoding="utf-8") as claim_in:
            for line in claim_in:
                payload = json.loads(line)
                claim_id, node = next(iter(payload.items()))
                if not first:
                    output.write(",")
                first = False
                output.write(json.dumps(claim_id, ensure_ascii=False))
                output.write(":")
                output.write(json.dumps(node, ensure_ascii=False, separators=(",", ":")))
        output.write('},"edges":[')
        first = True
        with edges_path.open("r", encoding="utf-8") as edge_in:
            for line in edge_in:
                if not first:
                    output.write(",")
                first = False
                output.write(line.strip())
        output.write("]}\n")
    os.replace(work, output_path)
    # Temporary line stores are recoverable intermediates and intentionally
    # retained until the final graph hash exists; then remove only these exact files.
    claim_nodes_path.unlink()
    edges_path.unlink()
    manifest = {
        "schema_version": "neurooracle.case2_scoped_graph_manifest.v1",
        **metadata,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = materialize(args.claims, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
