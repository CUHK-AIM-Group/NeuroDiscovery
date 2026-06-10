from __future__ import annotations

import pytest

from neurooracle.src.case_studies import (
    CASE1,
    CASE2,
    CASE3,
    case_study_by_name,
    list_case_study_names,
)


def test_case_study_registry_uses_final_numbering():
    names = list_case_study_names()

    assert names[:3] == (
        "case1_transdiagnostic",
        "case2_pathway_mediation",
        "case3_hindcasting",
    )
    assert case_study_by_name("case1_transdiagnostic") is CASE1
    assert case_study_by_name("case2_pathway_mediation") is CASE2
    assert case_study_by_name("case3_hindcasting") is CASE3


def test_case_study_registry_rejects_legacy_aliases():
    for old_name in (
        "cs2_transdiagnostic",
        "cs3_pathway_mediation",
        "cs_gamma_hindcasting",
        "cs_y_hindcasting",
    ):
        with pytest.raises(KeyError):
            case_study_by_name(old_name)
