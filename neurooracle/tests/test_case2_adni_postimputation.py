import csv
import gzip

from neurooracle.scripts.postprocess_case2_adni_imputation import (
    apoe_label,
    common_variant_records,
    extract_subject_id,
    merge_missnp_path,
    prepare_prs_score_files,
    read_pvar,
)


def write_pvar(path, rows):
    path.write_text(
        "\n".join(
            [
                "##filedate=20260731",
                "#CHROM\tPOS\tID\tREF\tALT\tFILTER\tINFO",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_common_variant_records_intersect_by_key_and_skip_missing_ids(tmp_path):
    left = tmp_path / "left.pvar"
    right = tmp_path / "right.pvar"
    write_pvar(
        left,
        [
            "19\t45411941\trs429358\tT\tC\tPASS\tDR2=0.99;AF=0.2",
            "19\t45412079\t.\tC\tT\tPASS\tDR2=0.90;AF=0.05",
        ],
    )
    write_pvar(
        right,
        [
            "19\t45411941\trs429358\tT\tC\tPASS\tDR2=0.91;AF=0.3",
            "19\t45412079\trs7412\tC\tT\tPASS\tDR2=0.95;AF=0.06",
        ],
    )

    records = common_variant_records(
        {"left": read_pvar(left), "right": read_pvar(right)}
    )

    assert [record["canonical_id"] for record in records] == ["rs429358"]
    assert records[0]["min_dr2"] == 0.91


def test_apoe_label_and_subject_normalization():
    assert extract_subject_id("ADNI3_129_S_6146") == "129_S_6146"
    assert extract_subject_id("32_009_S_0751") == "009_S_0751"
    assert apoe_label("1", "0") == "e3/e4"
    assert apoe_label("0", "1") == "e2/e3"
    assert apoe_label("", "1") == "incomplete"


def test_merge_missnp_path_matches_plink1_naming(tmp_path):
    prefix = tmp_path / "chr19_common_dr2_0p8_merged_bed"
    assert merge_missnp_path(prefix).name == "chr19_common_dr2_0p8_merged_bed-merge.missnp"


def test_prepare_prs_score_files_filters_thresholds_and_duplicates(tmp_path):
    gwas = tmp_path / "gwas.tsv.gz"
    rows = [
        {
            "hm_rsid": "rs1",
            "hm_effect_allele": "A",
            "hm_beta": "0.1",
            "p_value": "1e-9",
        },
        {
            "hm_rsid": "rs2",
            "hm_effect_allele": "G",
            "hm_beta": "0.2",
            "p_value": "0.01",
        },
        {
            "hm_rsid": "rs1",
            "hm_effect_allele": "A",
            "hm_beta": "0.3",
            "p_value": "0.02",
        },
    ]
    with gzip.open(gwas, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hm_rsid", "hm_effect_allele", "hm_beta", "p_value"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    paths = prepare_prs_score_files(gwas, tmp_path / "scores", [5e-8, 0.05])

    strict_lines = paths[5e-8].read_text(encoding="utf-8").strip().splitlines()
    loose_lines = paths[0.05].read_text(encoding="utf-8").strip().splitlines()
    assert strict_lines == ["ID\tA1\tBETA", "rs1\tA\t0.1"]
    assert loose_lines == ["ID\tA1\tBETA", "rs1\tA\t0.1", "rs2\tG\t0.2"]


# Last Updated At: 2026-07-31 01:48 HKT
