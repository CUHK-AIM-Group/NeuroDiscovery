from __future__ import annotations

from neurooracle.src.adni_genetics_harmonization import build_reference_match_set


def test_reference_match_requires_position_and_alleles(tmp_path) -> None:
    bim = tmp_path / "input.bim"
    pvar = tmp_path / "reference.pvar"
    bim.write_text(
        "1 rsExact 0 100 A G\n"
        "1 rsSwap 0 200 T C\n"
        "1 rsPosition 0 301 A C\n"
        "1 rsComplement 0 400 A C\n",
        encoding="utf-8",
    )
    pvar.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n"
        "1\t100\trsExact\tA\tG\n"
        "1\t200\trsSwap\tC\tT\n"
        "1\t300\trsPosition\tA\tC\n"
        "1\t400\trsComplement\tT\tG\n",
        encoding="utf-8",
    )
    summary = build_reference_match_set(bim, pvar, tmp_path / "out")
    assert summary["exact_matches"] == 2
    assert summary["excluded"]["position_mismatch"] == 1
    assert summary["excluded"]["allele_mismatch_or_complement"] == 1
    assert (tmp_path / "out" / "reference_ref_alleles.tsv").read_text(
        encoding="utf-8"
    ) == "rsExact\tA\nrsSwap\tC\n"
