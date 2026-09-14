"""Validate canonical KG files and regenerate CURRENT_STATE.json and README.md."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.audit_case_study_taxonomy_v2 import audit_graph
from neurooracle.scripts.count_case_study_kg_stats import count_case_studies


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO / "neurooracle" / "data" / "full_v2"


def scan_claim_store(path: Path) -> dict[str, int]:
    rows = 0
    rows_with_legacy_fields = 0
    rows_without_canonical_fields = 0
    invalid_rows = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rows += 1
            try:
                claim = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            metadata = claim.get("metadata") or {}
            if any(
                key in claim or key in metadata
                for key in ("paper_scope", "case3_tasks", "case3_subtasks")
            ):
                rows_with_legacy_fields += 1
            if not all(
                key in claim and key in metadata
                for key in ("paper_case_study_ids", "claim_case_study_ids")
            ):
                rows_without_canonical_fields += 1
    return {
        "rows": rows,
        "rows_with_legacy_fields": rows_with_legacy_fields,
        "rows_without_canonical_membership_fields": rows_without_canonical_fields,
        "invalid_json_rows": invalid_rows,
    }


def build_state(data_dir: Path) -> dict[str, Any]:
    graph = data_dir / "knowledge_graph.json"
    extracted = data_dir / "extracted_claims.jsonl"
    stats = count_case_studies(graph)
    full_graph_audit = audit_graph(graph, example_limit=1)
    claim_store = scan_claim_store(extracted)
    failures = {
        "graph_claims_with_legacy_fields": stats["quality"]["claims_with_legacy_scope_fields"],
        "graph_invalid_membership_subset": stats["quality"]["claim_membership_not_subset_of_paper_membership"],
        "claim_store_rows_with_legacy_fields": claim_store["rows_with_legacy_fields"],
        "claim_store_rows_without_canonical_fields": claim_store[
            "rows_without_canonical_membership_fields"
        ],
        "claim_store_invalid_json_rows": claim_store["invalid_json_rows"],
        "full_graph_structured_legacy_fields": sum(
            full_graph_audit["structured_legacy_field_occurrences"].values()
        ),
    }
    if any(failures.values()):
        raise ValueError(f"canonical full_v2 validation failed: {failures}")
    state = {
        "status": "canonical_current",
        "taxonomy_version": "case_study_membership.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instruction": (
            "Use only the two canonical files listed here for current KG state. "
            "Case studies are 17 peer scopes; hindcasting is a validation protocol."
        ),
        "canonical_files": {
            "knowledge_graph": {
                "path": str(graph.resolve()),
                "bytes": graph.stat().st_size,
                "last_write_utc": datetime.fromtimestamp(
                    graph.stat().st_mtime, timezone.utc
                ).isoformat(),
            },
            "extracted_claims": {
                "path": str(extracted.resolve()),
                "bytes": extracted.stat().st_size,
                "last_write_utc": datetime.fromtimestamp(
                    extracted.stat().st_mtime, timezone.utc
                ).isoformat(),
            },
        },
        "formal_kg_statistics": {
            "general": stats["general"],
            "case_studies": stats["case_studies"],
            "quality": stats["quality"],
            "counting_policy": stats["counting_policy"],
        },
        "extracted_claim_store": claim_store,
        "full_graph_taxonomy_audit": full_graph_audit,
        "validation_protocols": {
            "hindcasting": {
                "supported_case_studies": list(stats["case_studies"]),
            }
        },
        "archived_legacy_taxonomy": str(
            (data_dir.parent / "archive" / "legacy_case3_taxonomy_20260801").resolve()
        ),
    }
    # A deterministic policy projection is not a semantic re-audit, so retain
    # its explicit transition marker across routine state refreshes.  The
    # claim-level provenance remains canonical in both formal stores; this
    # summary merely keeps CURRENT_STATE self-explanatory for other sessions.
    current_state_path = data_dir / "CURRENT_STATE.json"
    if current_state_path.is_file():
        try:
            previous_state = json.loads(current_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_state = {}
        mapping = (
            previous_state.get("case2_component_mapping")
            if isinstance(previous_state, dict)
            else None
        )
        if (
            isinstance(mapping, dict)
            and mapping.get("schema_version")
            == "neurooracle.case2_component_mapping.v1"
            and mapping.get("target_case_study_id") == "case2_pathway_mediation"
        ):
            state["case2_component_mapping"] = mapping
        # These are durable provenance records for already-applied formal
        # mutations. Routine coverage refreshes must not erase them.
        for key in (
            "selective_case_study_membership_update",
            "infrastructure_taxonomy_update",
            "infrastructure_taxonomy_version",
            "spatial_mapping_schema_update",
            "spatial_mapping_schema_version",
            "spatial_mapping_data_update",
            "spatial_mapping_data_version",
            "umls_atomic_mapping_update",
            "umls_atomic_mapping_version",
        ):
            value = previous_state.get(key) if isinstance(previous_state, dict) else None
            if isinstance(value, (dict, str)):
                state[key] = value
    return state


def render_readme(state: dict[str, Any]) -> str:
    stats = state["formal_kg_statistics"]
    lines = [
        "# Current NeuroOracle KG",
        "",
        "This directory is intentionally minimal. Only these files are authoritative:",
        "",
        "- `knowledge_graph.json` — formal current knowledge graph and source of truth.",
        "- `extracted_claims.jsonl` — synchronized extraction/audit store.",
        "- `CURRENT_STATE.json` — machine-readable validation and coverage statistics.",
        "",
        "The taxonomy contains 17 peer Case Study IDs. Hindcasting is an independent",
        "validation protocol that can be run for every Case Study; there is no Case Study 3 ID.",
        "General is the shared corpus scope and is not a Case Study ID.",
        "",
        "## Current formal KG coverage",
        "",
        "| Case Study | Unique papers | Claims |",
        "|---|---:|---:|",
    ]
    for case_study_id, row in stats["case_studies"].items():
        lines.append(
            f"| `{case_study_id}` | {row['papers']:,} | {row['claims']:,} |"
        )
    general = stats["general"]
    lines.extend(
        [
            "",
            f"General corpus: **{general['papers']:,} papers / {general['claims']:,} claims**.",
            "",
            "Membership is non-exclusive. Claim counts use `claim_case_study_ids`; paper",
            "counts use the union stored in `paper_case_study_ids`. After any KG mutation,",
            "run `python -m neurooracle.scripts.refresh_full_v2_state` before reporting coverage.",
            "The refresh also audits non-Claim concepts and edges for structured legacy fields.",
        ]
    )
    infrastructure = state.get("infrastructure_taxonomy_update")
    if isinstance(infrastructure, dict):
        spatial = infrastructure.get("spatial_reference") or {}
        models = infrastructure.get("ml_model") or {}
        lines.extend(
            [
                "",
                "## Experiment infrastructure vocabulary",
                "",
                f"- Spatial references: **{int(spatial.get('after', 0)):,}** "
                "(`atlas` remains a supported query alias).",
                f"- ML models: **{int(models.get('after', 0)):,}**.",
                "- Stable `ATLAS:*` identifiers are retained for compatibility.",
                "- This infrastructure update did not change claim or paper membership.",
            ]
        )
    spatial_mapping = state.get("spatial_mapping_data_update")
    if isinstance(spatial_mapping, dict):
        coverage = spatial_mapping.get("coverage") or {}
        lines.extend(
            [
                "",
                "## Spatial mapping coverage",
                "",
                f"- Structured spatial references: **{int(coverage.get('spatial_reference_nodes', 0)):,}/36**.",
                f"- Atlas-specific brain ROI nodes: **{int(coverage.get('brain_roi_nodes', 0)):,}**.",
                f"- Resource-mapped EEG electrodes: **{int(coverage.get('sensor_elements_resource_mapped', 0)):,}**.",
                f"- Exact ROI crosswalks added: **{int(coverage.get('new_crosswalk_edges', 0)):,}**.",
                "- Spatial mapping did not change claim or paper membership.",
            ]
        )
    umls_mapping = state.get("umls_atomic_mapping_update")
    if isinstance(umls_mapping, dict):
        coverage = umls_mapping.get("coverage") or {}
        rewrite = umls_mapping.get("rewrite") or {}
        lines.extend(
            [
                "",
                "## UMLS atomic mention mapping",
                "",
                f"- Release: **{umls_mapping.get('umls_release', 'unknown')}**.",
                f"- Preserved `CLM_CONCEPT` nodes: **{int(rewrite.get('source_clm_concepts', 0)):,}**.",
                f"- Added eligible `CLM_ATOM` nodes: **{int(rewrite.get('new_atomic_mentions', 0)):,}**.",
                f"- Exact semantically compatible mapped mentions: **{int(coverage.get('mapped_atomic_biomedical_mentions', 0)):,}**.",
                f"- Added `maps_to` edges: **{int(rewrite.get('new_mapping_edges', 0)):,}**.",
                "- `needs_review` remains an explicit edge status; no fuzzy or embedding mapping was promoted.",
                "- The UMLS migration did not change claims, papers, or case-study membership.",
            ]
        )
    lines.extend(
        [
            "",
            "Legacy Case 3 taxonomy files and historical outputs are retained only under:",
            "",
            f"`{state['archived_legacy_taxonomy']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    state = build_state(data_dir)
    if not args.check_only:
        (data_dir / "CURRENT_STATE.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (data_dir / "README.md").write_text(render_readme(state), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
