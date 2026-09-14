from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_scoped_structure import remaining_findings


def test_resolved_findings_replaced_by_residual_scope_with_current_hash():
    ids=['CLM:CASE1MAN:22306803:8568','CLM:ead63272cd396884','CLM:pending']
    prior=dict(current_claim_hashes={cid:'old' for cid in ids},findings=[dict(claim_id=cid,classification='old') for cid in ids],hippocampal_scope='keep distinct')
    mri=[dict(claim_id='CLM:CASE1MAN:28526817:9725',claim_sha256='mri')]
    scope=[dict(claim_id='CLM:10532161a8b2296ab45e58e9db202527',claim_sha256='acc')]
    events=[dict(claim_id=cid,current_node_sha256='new') for cid in ids[:2]]
    out=remaining_findings(prior,mri,scope,events)
    assert ids[0] not in out['current_claim_hashes']
    assert out['current_claim_hashes'][ids[1]]=='new'
    assert out['current_claim_hashes']['CLM:pending']=='old'
    assert out['queue_counts_may_overlap'] and not out['issue_register_is_exhaustive']
    assert prior['current_claim_hashes'][ids[1]]=='old'


def test_report_never_recreates_automation_or_deletes_source_anchors():
    code=(Path(__file__).resolve().parents[1]/'scripts/report_kg_scoped_structure.py').read_text(encoding='utf8')
    assert 'automation_update(' not in code
    assert 'validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)' in code
    assert 'original_archives_and_public_evidence_deleted=False' in code
    assert 'physical_source_anchor_retirement_still_pending=True' in code
