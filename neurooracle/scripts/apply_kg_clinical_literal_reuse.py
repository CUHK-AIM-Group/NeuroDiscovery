"""R39 configured streaming engine; current clinical references only, no deletions."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
import apply_kg_literal_endpoint_repair as engine
from apply_kg_measurement_reuse import FILES as PRIOR_FILES
from kg_accepted_candidate_lineage import require
from neurooracle.src import kg_clinical_literal_reuse as repair

OUTPUT=j.OUTPUT/'round39_literal_duplicates'
SOURCE=j.OUTPUT/'round23_source_deletion_candidate/knowledge_graph.candidate.json'
FILES=list(dict.fromkeys([*PRIOR_FILES,*[j.REPO/p for p in (
    'neurooracle/scripts/apply_kg_clinical_literal_reuse.py','neurooracle/scripts/plan_kg_clinical_literal_reuse.py',
    'neurooracle/scripts/inspect_kg_literal_duplicates.py','neurooracle/src/kg_clinical_literal_reuse.py',
    'neurooracle/tests/test_kg_clinical_literal_reuse.py','neurooracle/tests/test_kg_clinical_literal_pipeline.py')]]))
_validate=engine.validate
_apply=engine.apply
_progress=engine.progress


def bindings():return [j.fingerprint(p) for p in FILES]


def load_plan(baseline):
    fp=j.fingerprint(OUTPUT/'PLAN.json');plan=j.read_json(fp['path'])
    require(plan['version']=='kg.clinical_literal_reuse.v1' and plan['status']=='REVIEWED_NOT_APPLIED','wrong current plan')
    require(plan['graph']==baseline['current_graph'] and plan['source_acceptance']==baseline['current_acceptance'] and plan['detail_store']==baseline['current_detail_store'],'source baseline differs')
    require(plan['gene_endpoint_source_count']==460 and plan['repaired_gene_endpoints']==4 and plan['held_endpoints']==456,'gene scope differs')
    require(plan['changed_claims']==6 and not plan['new_literals'] and not plan['added_literal_nodes'] and not plan['detail_rows_modified'],'clinical-only boundary differs')
    for item in [*plan['code'],plan['remaining_queue'],plan['source_inspection']]:engine.small_check(item)
    inspection=j.read_json(plan['source_inspection']['path'])
    require(inspection['graph']==plan['graph'] and inspection['source_full_sha_verified'] and inspection['detail_full_sha_verified'],'inspection not current')
    engine.small_check(inspection['code'])
    for item in inspection['artifacts'].values():engine.small_check(item)
    return fp,plan


def test_evidence():
    counts,seen=Counter(),set()
    for suite in ET.parse(OUTPUT/'TEST_RESULTS.xml').getroot().iter('testsuite'):
        counts.update({k:int(suite.get(k,0)) for k in ('tests','failures','errors','skipped')})
        for case in suite.findall('testcase'):
            key=(case.get('classname'),case.get('name'));require(key not in seen,'duplicate regression case');seen.add(key)
    require(counts['tests']==len(seen)>=590 and not any(counts[k] for k in ('failures','errors','skipped')),'distinct complete regression required')
    return dict(counts)


def progress(phase,**values):
    if phase=='BUILD_LITERAL_IDENTITY_AND_EXACT_REFERENCES':
        c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['active_process']['state']==str(OUTPUT/'RUN_STATE.json'),'wrong new writer')
        c['phase']='R39完整同名测量的临床引用复用，保留明细来源锚点';j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    _progress(phase,**values)


def validate():
    _validate()
    result=j.read_json(OUTPUT/'VALIDATED.json')
    result['checks'].update(verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,
        publication_status_witnesses_validated=True,clinical_literal_family_source_scope_verified=True,
        all_source_anchor_nodes_and_detail_rows_preserved=True,no_physical_entity_deletion_claimed=True)
    j.atomic_json(OUTPUT/'VALIDATED.json',result)


def apply():
    _apply()
    receipt=j.read_json(OUTPUT/'CURRENT_ACCEPTANCE.json');plan=j.read_json(OUTPUT/'PLAN.json')
    receipt.update(status='CURRENT_CLINICAL_LITERAL_REUSE_APPLIED',literal_family_proofs=plan['family_proofs'])
    receipt['changed'].update(repaired_gene_endpoints=4,legacy_clinical_endpoint_references_unified=2,source_anchor_nodes_retired=0)
    j.atomic_json(OUTPUT/'CURRENT_ACCEPTANCE.json',receipt)
    runtime=j.read_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json');runtime['acceptance']=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')
    j.atomic_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json',runtime)
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    c.update(phase='R39六条完整测量临床引用已复用；明细来源锚点保留',current_acceptance=runtime['acceptance'],
        current_runtime_acceptance=j.fingerprint(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json'),
        next_steps=['两组完整指标的临床引用共用既有实体；来源锚点和明细记录保留，不声称物理节点已合并删除。',
            '双VCP端点只剩一条about的结构问题、海马体积范围及其他待核继续保留。',
            '当前派生资料R39；无模型/训练、正式full_v2同步、历史KG或记录前像。'])
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    night=j.read_json(j.OUTPUT/'NIGHT_WINDOW_20260909.json')
    night.update(latest_acceptance=c['current_acceptance'],latest_runtime_acceptance=c['current_runtime_acceptance'],latest_completed_batch='R39',latest_completed_runtime_batch='R39')
    j.atomic_json(j.OUTPUT/'NIGHT_WINDOW_20260909.json',night)
    state=j.read_json(OUTPUT/'RUN_STATE.json');state['changed']=receipt['changed'];j.atomic_json(OUTPUT/'RUN_STATE.json',state)
    print('R39_ADOPTED',receipt['changed'],flush=True)


def configuration():
    return dict(OUTPUT=OUTPUT,SOURCE=SOURCE,TEMP=SOURCE.with_name(SOURCE.name+'.clinical.tmp'),
        CATALOG=OUTPUT/'CURRENT_SHARED_RELATIONS.jsonl',DATABASE=OUTPUT/'CURRENT_PAPER_CENSUS.sqlite',
        FILES=FILES,bindings=bindings,load_plan=load_plan,test_evidence=test_evidence,reviewed_claim=repair.reviewed_claim,
        change_claim=repair.change_claim,reverse_claim=repair.verified_reverse,progress=progress,validate=validate,apply=apply)


def configure():
    for name,value in configuration().items():setattr(engine,name,value)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','build','resume'))
    phase=parser.parse_args().phase;configure()
    if phase=='run':engine.build();validate();apply()
    elif phase=='build':engine.build()
    else:engine.resume()

