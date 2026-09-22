"""Migrate the authoritative TCP biomarker experiment into closed-loop tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.scripts.case_study_candidate_tables import export_table_bundle
from core.scripts.case_study_closed_loop import sha256_file


DEFAULT_RUN = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260803_fullv2_membership_v2"
)
DEFAULT_FROZEN = (
    DEFAULT_RUN
    / "external_method_comparison/frozen_tcp_rankings/frozen_tcp_candidates.csv.gz"
)
DEFAULT_EXTERNAL = (
    DEFAULT_RUN
    / "external_method_comparison/case1_external_candidate_labels.csv"
)
DEFAULT_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_ORDERS = (
    DEFAULT_RUN
    / "external_method_comparison/frozen_tcp_rankings/frozen_tcp_orders.npz"
)
DEFAULT_ORDER_MANIFEST = (
    DEFAULT_RUN
    / "external_method_comparison/frozen_tcp_rankings/frozen_tcp_rankings_manifest.json"
)
DEFAULT_OUTPUT = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v1"
    r"\biomarker_discovery\tables"
)


def atlas_from_source(value: object) -> str:
    source = str(value or "")
    return source.removesuffix("_multiatlas")


def build_tables(
    *,
    frozen_path: Path,
    all_tests_path: Path,
    external_path: Path,
    output_dir: Path,
    gt_fraction: float,
    orders_path: Path,
    order_manifest_path: Path,
) -> dict[str, object]:
    if not (0 < gt_fraction <= 1):
        raise ValueError("gt_fraction must be in (0, 1]")
    frozen = pd.read_csv(frozen_path, low_memory=False)
    required = {
        "candidate_id",
        "disease",
        "source",
        "feature",
        "feature_family",
        "anatomy_full",
        "score_neurodiscovery",
        "adjusted_residual_d",
    }
    missing = sorted(required - set(frozen.columns))
    if missing:
        raise ValueError(f"frozen TCP candidates are missing columns: {missing}")
    frozen["atlas"] = frozen["source"].map(atlas_from_source)
    frozen["anatomy"] = frozen["anatomy_full"].fillna(frozen["roi_name"])

    public_columns = [
        "candidate_id",
        "disease",
        "atlas",
        "anatomy",
        "feature_family",
        "feature",
        "modality",
        "source",
        "roi_index",
        "roi_id",
        "roi_name",
        "anatomy_full",
        "network",
        "score_neurodiscovery_global",
        "score_case_study_support",
        "score_neurodiscovery",
        "kg_scoped_disease_degree",
        "kg_scoped_region_degree",
        "kg_scoped_pair_support",
    ]
    public = frozen[[column for column in public_columns if column in frozen]].copy()

    absolute_effect = pd.to_numeric(
        frozen["adjusted_residual_d"], errors="coerce"
    ).abs()
    finite = np.isfinite(absolute_effect.to_numpy(dtype=float))
    n_gt = max(1, int(math.ceil(int(finite.sum()) * gt_fraction)))
    ranks = absolute_effect.rank(method="first", ascending=False)
    validated = finite & (ranks.to_numpy(dtype=float) <= n_gt)
    all_tests = pd.read_csv(
        all_tests_path,
        usecols=[
            "modality",
            "source",
            "disease",
            "feature",
            "roi_index",
            "q_fdr_global",
        ],
        low_memory=False,
    )
    all_tests["candidate_id"] = (
        all_tests["modality"].astype(str)
        + "|"
        + all_tests["source"].astype(str)
        + "|"
        + all_tests["disease"].astype(str)
        + "|"
        + all_tests["feature"].astype(str)
        + "|"
        + all_tests["roi_index"].astype(str)
    )
    if all_tests["candidate_id"].duplicated().any():
        raise ValueError("source all-tests table contains duplicate candidate IDs")
    q_lookup = all_tests.set_index("candidate_id")["q_fdr_global"]
    q_values = pd.to_numeric(
        frozen["candidate_id"].map(q_lookup), errors="coerce"
    )
    if q_values.isna().sum() > 0:
        raise ValueError(
            f"{int(q_values.isna().sum())} frozen candidates have no all-tests FDR row"
        )
    strict = q_values < 0.05
    internal = pd.DataFrame(
        {
            "candidate_id": frozen["candidate_id"].astype(str),
            "validated": validated,
            "strict_validated": strict.to_numpy(dtype=bool),
            "effect_size": pd.to_numeric(
                frozen["adjusted_residual_d"], errors="coerce"
            ),
            "absolute_effect_size": absolute_effect,
            "gt_rank": ranks,
            "q_fdr_global": q_values,
            "gt_definition": f"top_{gt_fraction:.6f}_absolute_adjusted_cohen_d",
        }
    )

    source_external = pd.read_csv(external_path, low_memory=False)
    unknown = set(source_external["candidate_id"].astype(str)) - set(
        public["candidate_id"].astype(str)
    )
    if unknown:
        raise ValueError(
            f"external labels contain {len(unknown)} candidates absent from TCP registry"
        )
    external = pd.DataFrame(
        {
            "candidate_id": source_external["candidate_id"].astype(str),
            "executable": pd.to_numeric(
                source_external["n_external_datasets"], errors="coerce"
            ).fillna(0).gt(0),
            "validated": source_external["external_label"]
            .astype(str)
            .eq("confirmed"),
            "external_label": source_external["external_label"].astype(str),
            "executable_datasets": source_external["executable_datasets"],
            "confirmed_datasets": source_external["confirmed_datasets"],
            "n_external_datasets": source_external["n_external_datasets"],
            "n_confirmed_datasets": source_external["n_confirmed_datasets"],
        }
    )

    provenance = {
        "discovery_dataset": "TCP",
        "external_datasets": ["UCLA", "COBRE", "HCP-EP", "ADHD200"],
        "source_frozen_candidates": str(frozen_path.resolve()),
        "source_frozen_candidates_sha256": sha256_file(frozen_path.resolve()),
        "source_external_labels": str(external_path.resolve()),
        "source_external_labels_sha256": sha256_file(external_path.resolve()),
        "source_all_tests": str(all_tests_path.resolve()),
        "source_all_tests_sha256": sha256_file(all_tests_path.resolve()),
        "source_frozen_orders": str(orders_path.resolve()),
        "source_frozen_orders_sha256": sha256_file(orders_path.resolve()),
        "source_order_manifest": str(order_manifest_path.resolve()),
        "source_order_manifest_sha256": sha256_file(order_manifest_path.resolve()),
        "candidate_order_preserved": True,
        "gt_fraction": gt_fraction,
        "gt_count": n_gt,
        "external_rule": (
            "contrast x feature-family BH-FDR q<0.05, |Cohen d|>0.15, "
            "and direction concordant with TCP"
        ),
    }
    kg_audit = {
        "source": "authoritative_case1_frozen_candidate_scores",
        "uses_experimental_outcomes": False,
        "case_study_id": "case1_transdiagnostic",
    }
    manifest = export_table_bundle(
        task="biomarker_discovery",
        public=public,
        internal=internal,
        external=external,
        factor_fields=("disease", "atlas", "anatomy", "feature_family", "feature"),
        output_dir=output_dir,
        provenance=provenance,
        kg_audit=kg_audit,
    )
    # The original 70 orders remain immutable and are referenced, not copied.
    closure = {
        **manifest,
        "source_ranking_manifest": json.loads(
            order_manifest_path.read_text(encoding="utf-8")
        ),
    }
    closure_path = output_dir / "biomarker_closure_manifest.json"
    closure_path.write_text(
        json.dumps(closure, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--orders", type=Path, default=DEFAULT_ORDERS)
    parser.add_argument(
        "--order-manifest", type=Path, default=DEFAULT_ORDER_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt-fraction", type=float, default=0.01)
    args = parser.parse_args()
    manifest = build_tables(
        frozen_path=args.frozen,
        all_tests_path=args.all_tests,
        external_path=args.external,
        output_dir=args.output_dir,
        gt_fraction=args.gt_fraction,
        orders_path=args.orders,
        order_manifest_path=args.order_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
