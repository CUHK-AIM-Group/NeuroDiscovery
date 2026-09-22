from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurooracle.scripts.run_case2_adni_dataset_constrained_panel import (
    build_kg_candidate_ranking,
    parse_variant_dosages,
)


def test_parse_variant_dosages_normalizes_ids_and_records_alleles(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "target.raw"
    raw_path.write_text(
        "FID IID PAT MAT SEX PHENOTYPE rs3865444_C rs4680_G\n"
        "1 333_002_S_0295 0 0 1 -9 1.25 0.50\n"
        "2 sub-ADNI002S0413 0 0 2 -9 0.00 2.00\n",
        encoding="utf-8",
    )

    dosages, metadata = parse_variant_dosages(raw_path)

    assert dosages["subject_id"].tolist() == ["002_S_0295", "002_S_0413"]
    assert dosages["cd33_rs3865444_dosage"].tolist() == [1.25, 0.0]
    assert dosages["comt_rs4680_dosage"].tolist() == [0.5, 2.0]
    assert metadata.set_index("variant_id").loc["rs3865444", "counted_allele"] == "C"
    assert metadata.set_index("variant_id").loc["rs4680", "counted_allele"] == "G"


def _result_row(
    *,
    exposure: str,
    outcome: str,
    marker_id: str,
    atlas: str,
    roi_name: str,
    network: str,
    feature: str,
    sobel_p: float,
) -> dict[str, object]:
    return {
        "exposure": exposure,
        "outcome": outcome,
        "marker_id": marker_id,
        "source": atlas,
        "atlas": atlas,
        "roi_index": 1,
        "roi_id": "1",
        "roi_name": roi_name,
        "parcel_name": roi_name,
        "hemisphere": "L",
        "network": network,
        "structure_class": "cortical",
        "feature": feature,
        "sobel_p": sobel_p,
        "a_path_p": 0.2,
        "a_path_std": 0.1,
        "abs_a_path_std": 0.1,
        "indirect_effect_std": 0.1,
        "abs_indirect_effect_std": 0.1,
        "sobel_q_family": 0.5,
    }


def test_kg_candidate_ranking_selects_claim_matched_markers_only() -> None:
    rows = [
        _result_row(
            exposure="apoe_e4_dosage",
            outcome="MMSE",
            marker_id="ho_hipp_corr",
            atlas="harvard_oxford_merged",
            roi_name="Left Hippocampus",
            network="",
            feature="corr_mean",
            sobel_p=0.99,
        ),
        _result_row(
            exposure="apoe_e4_dosage",
            outcome="MMSE",
            marker_id="ho_occipital_corr",
            atlas="harvard_oxford_merged",
            roi_name="Occipital Pole",
            network="",
            feature="corr_mean",
            sobel_p=0.001,
        ),
        _result_row(
            exposure="cd33_rs3865444_dosage",
            outcome="ADAS13",
            marker_id="s400_degree",
            atlas="schaefer_400_7net",
            roi_name="Default PFC",
            network="Default",
            feature="corr_node_degree_abs_top10",
            sobel_p=0.80,
        ),
        _result_row(
            exposure="comt_rs4680_dosage",
            outcome="MMSE",
            marker_id="msdl_dmn",
            atlas="msdl_39",
            roi_name="L DMN",
            network="DMN",
            feature="corr_mean",
            sobel_p=0.70,
        ),
    ]

    ranked = build_kg_candidate_ranking(pd.DataFrame(rows))

    assert set(ranked["marker_id"]) == {
        "ho_hipp_corr",
        "s400_degree",
        "msdl_dmn",
    }
    assert "ho_occipital_corr" not in set(ranked["marker_id"])
    assert ranked["kg_rank"].tolist() == list(range(1, len(ranked) + 1))


def test_kg_ranking_is_outcome_blind() -> None:
    rows = [
        _result_row(
            exposure="apoe_e4_dosage",
            outcome="MMSE",
            marker_id="left",
            atlas="harvard_oxford_merged",
            roi_name="Left Hippocampus",
            network="",
            feature="corr_mean",
            sobel_p=0.99,
        ),
        _result_row(
            exposure="apoe_e4_dosage",
            outcome="MMSE",
            marker_id="right",
            atlas="harvard_oxford_merged",
            roi_name="Right Hippocampus",
            network="",
            feature="corr_mean",
            sobel_p=0.001,
        ),
    ]
    original = build_kg_candidate_ranking(pd.DataFrame(rows))
    rows[0]["sobel_p"], rows[1]["sobel_p"] = rows[1]["sobel_p"], rows[0]["sobel_p"]
    swapped = build_kg_candidate_ranking(pd.DataFrame(rows))

    assert original["marker_id"].tolist() == swapped["marker_id"].tolist()

# Last Updated At: 2026-07-31 03:52 HKT
