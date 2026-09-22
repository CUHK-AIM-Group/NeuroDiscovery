"""Build and freeze outcome-blind NeuroDiscovery prior components for CS2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.scripts.case2_search_policy import (
    build_public_registry,
    candidate_id_from_fields,
)
from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import CASE2_ROOT
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import load_hypotheses
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni_longitudinal import (
    build_ranked_candidates,
)
from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    DEFAULT_PATHWAY_ROOT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = (
    CASE2_ROOT / "protocols" / "case2_adni_endpoint_holdout_v2"
)
DEFAULT_HYPOTHESES = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "experiments"
    / "case2"
    / "pathway_mediation_20260816_kg89E40DD8_confirmatory_v1"
    / "hypotheses_raw.json"
)
PRE_OUTCOME_PROTOCOL_STATUSES = {
    "frozen_before_confirmation_association_access",
    "frozen_before_phase_heldout_association_access",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_protocol_freeze(protocol_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = protocol_root / "protocol_freeze_manifest.json"
    lock_path = protocol_root / "FREEZE.lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in PRE_OUTCOME_PROTOCOL_STATUSES:
        raise ValueError("Case 2 protocol is not in a frozen pre-outcome state")
    if manifest.get("freeze_id") != lock.get("freeze_id"):
        raise ValueError("Case 2 protocol manifest and lock use different freeze IDs")
    if _sha256(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("Case 2 protocol manifest changed after it was locked")
    registry_path = Path(manifest["artifacts"]["public_candidate_registry"])
    expected = manifest["lock_material"]["registry_sha256"]
    if _sha256(registry_path) != expected:
        raise ValueError("Frozen Case 2 candidate registry hash mismatch")
    return manifest, registry_path


def build_prior_components(
    *,
    registry: pd.DataFrame,
    exposures: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
    hypotheses: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    public = build_public_registry(registry)
    ranked, support, mapping_audit = build_ranked_candidates(
        hypotheses,
        exposures,
        pathway_catalog,
        public,
    )
    ranked["candidate_id"] = [
        candidate_id_from_fields(*values)
        for values in ranked.loc[
            :, ["exposure", "modality", "marker", "outcome"]
        ].itertuples(index=False, name=None)
    ]
    if set(ranked["candidate_id"]) != set(public["candidate_id"]):
        raise ValueError("NeuroDiscovery prior does not cover the frozen candidate registry")
    if ranked["candidate_id"].duplicated().any():
        raise ValueError("NeuroDiscovery prior contains duplicate candidate IDs")
    return ranked, support, mapping_audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    freeze, registry_path = _verify_protocol_freeze(args.protocol_root)
    freeze_id = str(freeze["freeze_id"])
    output_root = args.output_root or (
        CASE2_ROOT
        / "experiments"
        / "case2_adni_confirmatory_v1"
        / freeze_id[:12]
        / "outcome_blind_generator_inputs"
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(registry_path)
    exposures_path = args.protocol_root / "selected_pathway_exposures.csv"
    pathway_catalog_path = args.pathway_root / "pathway_catalog.csv"
    exposures = pd.read_csv(exposures_path)
    pathway_catalog = pd.read_csv(pathway_catalog_path)
    hypotheses = load_hypotheses(args.hypotheses)
    ranked, support, mapping_audit = build_prior_components(
        registry=registry,
        exposures=exposures,
        pathway_catalog=pathway_catalog,
        hypotheses=hypotheses,
    )

    ranked_path = output_root / "neurodiscovery_prior_components.parquet"
    ranked_csv_path = output_root / "neurodiscovery_prior_components.csv"
    support_path = output_root / "candidate_hypothesis_support.parquet"
    audit_path = output_root / "hypothesis_mapping_audit.csv"
    ranked.to_parquet(ranked_path, index=False, compression="zstd")
    ranked.to_csv(ranked_csv_path, index=False)
    support.to_parquet(support_path, index=False, compression="zstd")
    mapping_audit.to_csv(audit_path, index=False)

    pins = {
        "protocol_freeze_manifest": _sha256(
            args.protocol_root / "protocol_freeze_manifest.json"
        ),
        "public_candidate_registry": _sha256(registry_path),
        "hypotheses_raw": _sha256(args.hypotheses),
        "selected_pathway_exposures": _sha256(exposures_path),
        "pathway_catalog": _sha256(pathway_catalog_path),
        "prior_components": _sha256(ranked_path),
        "candidate_support": _sha256(support_path),
        "mapping_audit": _sha256(audit_path),
    }
    phase_heldout = (
        freeze.get("status") == "frozen_before_phase_heldout_association_access"
    )
    manifest = {
        "schema_version": "neurooracle.case2_outcome_blind_prior.v1",
        "status": (
            "frozen_before_phase_heldout_association_access"
            if phase_heldout
            else "frozen_before_confirmation_association_access"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_freeze_id": freeze_id,
        "candidate_count": int(len(ranked)),
        "hypothesis_count": int(len(hypotheses)),
        "meaningfully_mapped_hypotheses": int(
            mapping_audit["meaningfully_mapped"].sum()
        ),
        "outcome_blind": True,
        "confirmation_holdout_axis": (
            "cohort_phase" if phase_heldout else "endpoint"
        ),
        "experimental_statistics_accessed": [],
        "source_fields": [
            "KG hypothesis score",
            "pathway semantic alignment",
            "imaging-marker semantic alignment",
            "outcome semantic alignment",
            "independent supporting-hypothesis count",
        ],
        "artifacts": {
            "prior_components": str(ranked_path),
            "candidate_support": str(support_path),
            "mapping_audit": str(audit_path),
        },
        "sha256": pins,
    }
    (output_root / "prior_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 13:06 HKT
