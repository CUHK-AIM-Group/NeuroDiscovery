"""R38 adapter over the verified streaming engine; never replay R37 artifacts.

Only new process-local callbacks/paths are configured. Historical source files
remain frozen and hash-checked. The source is the current R37 accepted graph.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
import apply_kg_literal_endpoint_repair as engine
from kg_accepted_candidate_lineage import require
from plan_kg_measurement_reuse import public_proof
from neurooracle.src import kg_measurement_reuse as repair

OUTPUT=journal.OUTPUT/'round38_measurement_reuse'
SOURCE=journal.OUTPUT/'round23_source_deletion_candidate/knowledge_graph.candidate.json'
FILES=list(dict.fromkeys([*engine.FILES,*[journal.REPO/p for p in (
    'neurooracle/scripts/apply_kg_measurement_reuse.py','neurooracle/scripts/plan_kg_measurement_reuse.py',
    'neurooracle/scripts/inspect_kg_measurement_reuse.py','neurooracle/src/kg_measurement_reuse.py',
    'neurooracle/tests/test_kg_measurement_reuse.py','neurooracle/tests/test_kg_measurement_reuse_pipeline.py')]]))
_original_validate=engine.validate
_original_apply=engine.apply
_original_progress=engine.progress


def bindings():return [journal.fingerprint(p) for p in FILES]


def load_plan(baseline):
    fp=journal.fingerprint(OUTPUT/'PLAN.json');plan=journal.read_json(fp['path'])
    require(plan['version']=='kg.measurement_reuse.v1' and plan['status']=='REVIEWED_NOT_APPLIED','wrong R38 plan')
    require(plan['graph']==baseline['current_graph'] and plan['source_acceptance']==baseline['current_acceptance'] and plan['detail_store']==baseline['current_detail_store'],'R38 source baseline changed')
    require(plan['gene_endpoint_source_count']==488==plan['changed_endpoints']+plan['held_endpoints'],'R38 source scope changed')
    for item in [*plan['code'],plan['remaining_queue'],plan['public_source_fetch']]:engine.small_check(item)
    proof=public_proof()
    require({k:v for k,v in proof.items() if k!='abstract'}==plan['public_measurement_evidence'],'owning public scientific proof differs')
    return fp,plan


def test_evidence():
    counts,seen=Counter(),set()
    for suite in ET.parse(OUTPUT/'TEST_RESULTS.xml').getroot().iter('testsuite'):
        counts.update({k:int(suite.get(k,0)) for k in ('tests','failures','errors','skipped')})
        for case in suite.findall('testcase'):
            key=(case.get('classname'),case.get('name'));require(key not in seen,'duplicate regression case');seen.add(key)
    require(counts['tests']==len(seen)>=525 and not any(counts[k] for k in ('failures','errors','skipped')),'complete distinct regression required')
    return dict(counts)


def reviewed_claim(row,changes):return repair.reviewed_claim(row,changes,_proof)


def verified_reverse(row,event):
    original=repair.reverse_claim(row,event)
    reproduced,_=repair.reviewed_claim(original,event['changes'],_proof)
    require(all(event[k]==v for k,v in reproduced.items()),'independent scientific/identity review differs')
    return original


def progress(phase,**values):
    if phase=='BUILD_LITERAL_IDENTITY_AND_EXACT_REFERENCES':
        c=journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
        require(c['active_process']['state']==str(OUTPUT/'RUN_STATE.json'),'wrong new writer state path')
        c['phase']='R38完整网络测量、明确测量表达及有来源的统计值修复'
        journal.atomic_json(journal.OUTPUT/'CAMPAIGN.json',c)
    _original_progress(phase,**values)


def validate():
    _original_validate()
    # All original-record inverse, entity, bibliography and publication proofs
    # have run. Publish their existing reader capabilities before adoption.
    result=journal.read_json(OUTPUT/'VALIDATED.json');checks=result['checks']
    for key in ('all_nonidentity_claim_fields_preserved','records_match_approved_identity_only_transform'):
        checks.pop(key,None)
    checks.update(verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,
        publication_status_witnesses_validated=True,owning_network_definition_and_numeric_source_reproduced=True,
        only_approved_three_numeric_and_method_repairs=True,all_other_scientific_fields_preserved=True,
        nominal_p_not_claimed_multiple_comparison_corrected=True,regression_not_replaced_by_univariate_correlation=True)
    journal.atomic_json(OUTPUT/'VALIDATED.json',result)


def apply():
    _original_apply()
    receipt=journal.read_json(OUTPUT/'CURRENT_ACCEPTANCE.json')
    plan=journal.read_json(OUTPUT/'PLAN.json')
    receipt.update(status='CURRENT_MEASUREMENT_REUSE_APPLIED',public_measurement_evidence=plan['public_measurement_evidence'])
    receipt['changed']['source_proven_numeric_or_method_repairs']=plan['science_changed_claims']
    journal.atomic_json(OUTPUT/'CURRENT_ACCEPTANCE.json',receipt)
    runtime=journal.read_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json')
    runtime['acceptance']=journal.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')
    journal.atomic_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json',runtime)
    c=journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
    c.update(phase='R38网络指标复用、测量漏检及三条统计值/方法修复已验收',
        current_acceptance=runtime['acceptance'],current_runtime_acceptance=journal.fingerprint(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json'),
        next_steps=['6条网络指标已按本篇完整来源对应，3条上界误作精确值已修正；不把单篇复用称为多篇证据。',
            '保留剩余未证实类型、复合/分子/重复候选及混合引用；21语义与1结构仍待核。',
            '当前派生资料在R38；无模型训练、正式full_v2同步、历史KG或记录前像。'])
    journal.atomic_json(journal.OUTPUT/'CAMPAIGN.json',c)
    night=journal.read_json(journal.OUTPUT/'NIGHT_WINDOW_20260909.json')
    night.update(latest_acceptance=c['current_acceptance'],latest_runtime_acceptance=c['current_runtime_acceptance'],
        latest_completed_batch='R38',latest_completed_runtime_batch='R38')
    journal.atomic_json(journal.OUTPUT/'NIGHT_WINDOW_20260909.json',night)
    state=journal.read_json(OUTPUT/'RUN_STATE.json');state['changed']=receipt['changed']
    journal.atomic_json(OUTPUT/'RUN_STATE.json',state)
    print('R38_ADOPTED',receipt['changed'],flush=True)


def configuration():
    return dict(OUTPUT=OUTPUT,SOURCE=SOURCE,TEMP=SOURCE.with_name(SOURCE.name+'.measurement.tmp'),
        CATALOG=OUTPUT/'CURRENT_SHARED_RELATIONS.jsonl',DATABASE=OUTPUT/'CURRENT_PAPER_CENSUS.sqlite',
        FILES=FILES,bindings=bindings,load_plan=load_plan,test_evidence=test_evidence,reviewed_claim=reviewed_claim,
        change_claim=repair.change_claim,reverse_claim=verified_reverse,progress=progress,validate=validate,apply=apply)


def configure():
    global _proof
    _proof=public_proof()
    for name,value in configuration().items():setattr(engine,name,value)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','build','resume'))
    phase=parser.parse_args().phase;configure()
    if phase=='run':engine.build();validate();apply()
    elif phase=='build':engine.build()
    else:engine.resume()
