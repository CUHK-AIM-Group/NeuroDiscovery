from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_bulk_cleanup import cleanup_summary,coverage_summary


def test_cleanup_summary_rejects_unapproved_categories():
    assert cleanup_summary({'empty_hint:a':2,'canonical_audit_duplicate:b':3})=={'empty_hint':2,'canonical_audit_duplicate':3}
    with pytest.raises(Exception):cleanup_summary({'scientific_value:a':1})
    with pytest.raises(Exception):cleanup_summary({'empty_hint:a':True})


def test_coverage_comparison_distinguishes_occurrences_and_value_coverage():
    fields=['metadata.raw_text','metadata.evidence.p_value','metadata.evidence.effect_size','metadata.evidence.sample_size']
    after=[dict(scope='node/claim',field=f,denominator=10,present=10,nonempty_pct=20.0) for f in fields]
    before=deepcopy(after)+[dict(scope='node/claim',field='metadata.metadata.subject_atlas',denominator=10,present=9,nonempty_pct=0.0)]
    result=coverage_summary(before,after,9)
    assert result['removed_occurrences']==9 and result['source_average_nested_fields']==0.9
    assert result['current_average_nested_fields']==0.0 and result['claims']==10
    changed=deepcopy(after);changed[1]['nonempty_pct']=100.0
    with pytest.raises(Exception):coverage_summary(before,changed,9)


def test_report_retirement_is_exact_and_does_not_claim_graph_problem_free():
    code=(Path(__file__).resolve().parents[1]/'scripts/report_kg_bulk_cleanup.py').read_text(encoding='utf8')
    assert 'validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)' in code
    assert 'original_archives_and_public_evidence_deleted=False' in code
    assert '整体优化未完成' in code and '不穷尽全图' in code
    assert 'automation_update(' not in code
