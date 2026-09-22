from __future__ import annotations

from neurooracle.src.adni_genetics_harmonization import (
    _plink_chromosome,
    _ucsc_chromosome,
)


def test_chromosome_conversions() -> None:
    assert _ucsc_chromosome("1") == "chr1"
    assert _ucsc_chromosome("23") == "chrX"
    assert _ucsc_chromosome("MT") == "chrM"
    assert _plink_chromosome("chr1") == "1"
    assert _plink_chromosome("chrM") == "MT"
