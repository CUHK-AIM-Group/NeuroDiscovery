"""R39 clinical-reference report, real queries and guarded old derivative cleanup."""
import argparse
from pathlib import Path
import re
import sqlite3
import sys
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from report_current_kg import check_html
from report_kg_literal_endpoint_repair import comparison
from report_kg_measurement_reuse import validate_retirement_paths
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=j.OUTPUT/'round39_literal_duplicates'
PREVIOUS=j.OUTPUT/'round38_measurement_reuse'


def audited_families(plan):
    result=[]
    for family in plan['family_proofs']:
        ids=set(family['incident_claim_ids'])
        ids.update(e['claim_id'] for e in plan['events'] if any(ch['name']==family['name'] for ch in e['changes']))
        require(family['source_anchor_nodes_and_detail_rows_preserved'],'source anchors must remain')
        result.append(dict(name=family['name'],canonical_id=family['target_id'],reviewed_claim_ids=sorted(ids),reviewed_claim_count=len(ids),
            physical_nodes_deleted=0,not_an_exhaustive_same_name_scientific_equivalence=True))
    return result


def pending():
    require(not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'already adopted')
    section=f'''<section id="clinical-literal-pending"><h2>R39 正在验收：完整指标的临床引用复用</h2>
<p>595项回归通过。计划修改6条claim、10条关联边，其中4处错误VCP指向；不增删节点/边或metadata字段。VMHC和灰质体积变化两组已审临床引用共用现有实体，仍被明细库引用的来源锚点保留，不声称物理节点已删除。</p>
<p>同名但类型不明的记录、海马体积合计/双侧分别测量的范围问题、双VCP端点仅有一条about的问题均不强行合并。当前仍以R38验收为准。</p>
<p>{j.link(OUTPUT/'PLAN.json','限定修复计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','595项测试')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');marker='<section id="current-measurement-reuse">'
    require(marker in page,'current HTML section missing')
    if 'id="clinical-literal-pending"' in page:page=re.sub(r'<section id="clinical-literal-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:page=page.replace(marker,section+'\n'+marker,1)
    check_html(page);j.atomic_text(j.REPORT,page)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R39 not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json');plan=j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status']=='CURRENT_CLINICAL_LITERAL_REUSE_APPLIED' and receipt['counts']==c['counts'],'current receipt differs')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_coverage','current_paper_issues','current_paper_census','current_issues','current_structure_holds')],
        receipt['provenance_plan'],receipt['identity_audit'],receipt['tests'],*receipt['code']]:
        require(j.fingerprint(fp['path'])==fp,'current evidence changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance']==c['current_acceptance'],'runtime binding differs')
    suite=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests'))==3 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report tests failed')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    families=audited_families(plan);require(sorted(f['reviewed_claim_count'] for f in families)==[3,5],'reviewed family membership differs')
    known={r['claim_id']:r['claim_sha256'] for r in rows(OUTPUT/'CLAIM_SCOPE.jsonl')}
    known.update({e['claim_id']:e['current_node_sha256'] for e in plan['events']})
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid in {cid for f in families for cid in f['reviewed_claim_ids']}:
        require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(known[cid],),'current clinical claim hash differs')
    deferred_ids=['CLM:CASE1MAN:22306803:8568','CLM:CASE1MAN:36716140:8023','CLM:593b77e75c8f89d8','CLM:ead63272cd396884']
    for cid in deferred_ids:require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(known[cid],),'deferred source changed')
    db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2))
    inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'query count differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        raw_groups=len(groups),default_minimum_two_source_groups=len(default),inclusive_minimum_two_source_groups=len(inclusive),
        reviewed_clinical_families=families,source_nodes_and_detail_rows_retired=0,no_paper_or_consensus_inferred=True))
    comp=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp.exists():
        old=state['baseline']['current_shared_relations'];require(j.fingerprint(old['path'])==old,'previous index differs')
        j.atomic_json(comp,dict(graph=c['current_graph'],at=j.utc_now(),**comparison(groups,rows(old['path']))))
    compared=j.read_json(comp);require(compared['graph']==c['current_graph'],'comparison not current')
    j.atomic_json(OUTPUT/'DEFERRED_SCOPE_FINDINGS.json',dict(at=j.utc_now(),graph=c['current_graph'],
        current_claim_hashes={cid:known[cid] for cid in deferred_ids},
        findings=[dict(claim_id=deferred_ids[0],classification='confirmed_endpoint_collision_and_incomplete_about_closure',
            detail='Both endpoint IDs are VCP although complete names are an imaging measure and a COMT-by-externalizing interaction; one about edge only. Requires joint endpoint and missing-reference repair, not one-side rewrite.',
            explicit_source_pmid='22306803',public_evidence=j.fingerprint(OUTPUT/'PUBLIC_SOURCE_FETCH.json')),
            dict(claim_id=deferred_ids[1],classification='compound_scope_and_reference_review',detail='Prenatal exposure and cognitive-performance outcome combined in one endpoint name; do not truncate to generic cognition.'),
            dict(claim_id=deferred_ids[2],classification='frailty_complete_name_and_mixed_reference_review'),
            dict(claim_id=deferred_ids[3],classification='internal_direction_conflict_requires_own_source_review',
                detail='Current raw text says negatively associated, while evidence.direction is positive. Not corrected in an identity-only batch.')],
        hippocampal_scope='Combined left+right volume versus separately reported bilateral volumes needs semantic review; exact label alone is insufficient.',
        graph_or_scientific_fields_modified_by_this_register=False,record_preimages_saved=False))
    retention_path=OUTPUT/'RETENTION_RESULT.json'
    if not retention_path.exists():
        targets=[j.read_json(state['baseline']['current_paper_census']['path'])['database'],
            *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current={Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths=validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)
        for fp,path in zip(targets,paths):j.guards(fp);require(sha256(path)==fp['sha256'],'old derivative changed')
        require(j.read_json(control)==c,'current writer advanced')
        for path in paths:path.unlink()
        j.atomic_json(retention_path,dict(status='COMPLETED',at=j.utc_now(),removed=targets,bytes_removed=sum(f['bytes'] for f in targets),
            graph_backups=0,record_preimages_saved=False,original_archives_and_public_evidence_deleted=False,
            recoverability='No old copies retained; current derivatives can be rebuilt, not a historical KG rollback.'))
    ret=j.read_json(retention_path);at=j.local_time(receipt['at']);counts=c['counts'];rel=c['relation_evidence_counts']
    coverage=rows(c['current_coverage']['path']);held=rows(c['current_gene_holds']['path'])
    require(len(held)==456 and c['pending_gene_link_endpoint_events']==456,'current held scope differs')
    report=f'''# R39：完整指标的临床引用统一，来源锚点保留

{at}（香港）已应用当前工作KG。595项回归与3项报告测试、独立全图和实际查询通过；没有模型、训练或正式full_v2同步。

本批审核的VMHC相关5条、灰质体积变化相关3条claim，各自共用1个现有完整指标实体。实际改动6条claim和10条边引用，包括4处错误VCP指向及2处旧临床节点引用。不改变原始文本、谓词、否定、数值、条件、论文身份或证据成员；不增加metadata字段。

这不是物理删除了重复节点。两个原临床锚点仍各有一条非核心、未映射的明细原子记录依赖，本批保留全部源节点与明细数据，避免悬空引用。后续若物理合并，必须联合迁移明细记录并验收。名称相同但类型不明的记录未纳入本批；初次只读计划因此停止，最终计划严格限制在已审范围，没有放宽类型判断。

## 当前数据

- 节点{counts['nodes']:,}、claim{counts['claims']:,}、边{counts['edges']:,}，均不增减。
- 节点metadata键并集{len(receipt['metadata_key_unions']['nodes'])}、边{len(receipt['metadata_key_unions']['edges'])}；{len(coverage)}行覆盖统计完整重算，未补填空值。
- 共享关系组{rel['shared_groups']:,}、成员claim{rel['indexed_claims']:,}；至少两个已核实论文来源的组默认{len(default)}，含已确认撤稿来源{len(inclusive)}。共享成员进入{len(compared['newly_shared_claim_ids'])}、退出{len(compared['no_longer_shared_claim_ids'])}。实体复用不一定增加同一关系的多论文证据，不把节点归一当新增研究。
- 连通分量{receipt['connected_components']:,}、孤立点{receipt['isolated_nodes']:,}、自环{receipt['self_loops']:,}。无临床引用的来源锚点仍被明细库引用，不因成为图内孤立点而随意删除。
- 剩余456处基因端点、{c['pending_gene_link_claims']}条claim待核；它们并非全部已确认错误。既有21语义与1frailty结构队列仍保留，新增发现单列，未假装队列已穷尽。

## 一并查明的后续问题

1. CLM:CASE1MAN:22306803:8568的影像指标和COMT基因型×外化行为交互均被压到VCP，只保留了一条about边。原论文确实研究COMT val158met与外化特质共同影响104名男性的脑激活，但不同基因型/特质组合涉及不同区域。需要同时恢复完整两端与缺失引用，不能只修一个ID。[原论文](https://pubmed.ncbi.nlm.nih.gov/22306803/)。本批未改这条科学记录。
2. bilateral hippocampal volume候选中既有左右海马合计体积，也有左右分别测量的描述。完整名称没有编码这种差异，不自动全局合并；缺失类型/左右范围需与原文核对。
3. CLM:ead63272cd396884的当前原文写negative，direction字段却为positive；登记为内部方向矛盾，另行核实本篇来源，未混入身份修复。
4. 产前烟草暴露与认知表现的复合端点、frailty混合引用、NAcc书目/年龄侧别范围仍待核。不截短复合表达、不把组内或关联结果变成因果，不统一翻转否定。

## 验收与保留

源图与计划边界完整SHA、候选独立全图、全节点/边原记录逆向摘要、当前905,184条claim/书目/共享成员普查、实体/论文/出版见证、全量metadata覆盖、明细库与新普查完整SHA均通过。实际读取器能力在采用前发布，实际共享查询通过。

旧R38普查、共享索引、书目待核与覆盖表共{ret['bytes_removed']:,}字节已删除，没有历史KG或记录前像。保留当前KG及必要派生资料、原始档案、公开来源和小型哈希/数量凭据。

KG SHA-256：{c['current_graph']['sha256']}，{c['current_graph']['bytes']:,}字节。
使用incremental-integrity-checks：平时查原子状态/可信指纹，构建和采用边界全量核验，避免为普通进度重复扫描巨型文件。总体目标尚未完成；夜间接续窗口至2026-09-09 10:00HKT。
'''
    next_plan='''# R40 / 夜间窗口收尾

以CAMPAIGN.json为准：当前R39工作图在R23路径；R39持有当前census/index/coverage/holds/publication review，R30实体terms、R34论文registry。R38旧四项派生数据已删除，不重跑旧构建。595项回归+3项报告测试与独立全图、实际查询通过。

当前新发现见DEFERRED_SCOPE_FINDINGS.json（当前claim哈希已与新普查核对）。R39/EXACT_REFERENCES.jsonl与NODE_SCOPE.jsonl是R38源图检查见证，不是可直接当当前图使用的无版本快照；需经过当前源指纹/对应记录哈希确认后复用。

1. 本批6claim/10边引用、4处VCP误指向修复，无节点/边/metadata字段增删。VMHC和灰质体积两组已审临床引用共用既有实体，原锚点及非核心未映射明细原子保留。实体物理合并要联合迁移umls_details.sqlite，不能只删图节点；不得假称物理去重完成。
2. 双VCP端点CLM:CASE1MAN:22306803:8568：原图两端相同，完整引用只有about序号62977；另一侧是COMT genotype by externalizing behavior interaction，不是VCP。R39/pubmed_22306803.xml与PUBLIC_SOURCE_FETCH.json为明确PMID的公开原文。需要同时决定完整端点、补缺失about并独立验收，不能沿用恰好两条about/零新边的旧引擎契约。
3. 三个海马体积基因端点及两个同名节点的多种测量范围、产前烟草/认知复合端点、frailty和NAcc问题继续保留；方向矛盾CLM:ead63272cd396884必须逐篇核实，不批量改positive/negative。
4. 456处基因端点待核；21语义与1结构旧队列仍在当前版本，新增发现另列，不外推全图无问题。当前目标控制blocked已记录，不绕过控制数据库、不把未完成目标标为完成。
5. 截止2026-09-09 10:00HKT不启动无法按时安全验收的新KG写入；完成正在运行批次再记录成果与下一步。此窗口结束时删除本轮kg心跳，并明确仍未解决项；不要删除旧源档案或擅自同步正式full_v2。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report);j.atomic_text(OUTPUT/'NEXT_BATCH_PLAN.md',next_plan)
    section=f'''<section id="current-clinical-reuse"><h2>当前 R39：6条claim与10条边的完整指标引用已统一</h2>
<div class="notice">595项回归、3项报告测试、独立全图和实际查询通过。4处VCP误指向修复；不增删节点/边或metadata字段，无模型或训练。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}</td></tr>
<tr><td>本批已审临床引用</td><td>VMHC相关5条、灰质体积相关3条，各自共用1个既有指标节点；实际修改6条</td></tr>
<tr><td>metadata键并集</td><td>节点120 / 边36；新增0</td></tr><tr><td>共享组 / 成员claim</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认至少两篇已核实来源{len(default)}组</td></tr>
<tr><td>待核</td><td>456处基因端点、{c['pending_gene_link_claims']}条claim；旧21语义/1结构及新增范围问题保留</td></tr></table>
<p>临床引用已统一，但源节点还被明细库依赖，故本批没有物理删除这两个来源锚点。物理合并需联动明细迁移；同名不等于所有研究测量范围相同。</p>
<p class="warning">已查明一条关系两端误指VCP且只剩一条about；还发现海马体积范围差异和一条原文/方向字段矛盾，均有当前哈希和下一步说明，未强行改科学结论。整体目标尚未完成。</p>
<p>旧R38四项派生文件{ret['bytes_removed']:,}字节已删除，无旧KG或记录回退副本。当前SHA-256：<code>{c['current_graph']['sha256']}</code></p>
<p>{' · '.join(j.link(OUTPUT/n,label) for n,label in [('REPORT.md','完整说明与出处'),('CURRENT_ACCEPTANCE.json','当前验收'),('CURRENT_RUNTIME_ACCEPTANCE.json','当前运行时'),('PLAN.json','限定修复计划'),('QUERY_VALIDATION.json','实际查询'),('DEFERRED_SCOPE_FINDINGS.json','新增待核发现'),('CURRENT_GENE_ENDPOINT_HOLDS.jsonl','456处端点待核'),('CURRENT_METADATA_COVERAGE.jsonl','当前覆盖'),('CENSUS.json','当前普查'),('TEST_RESULTS.xml','595项测试'),('NEXT_BATCH_PLAN.md','接续与窗口收尾')])}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');page=re.sub(r'<section id="clinical-literal-pending">.*?</section>','',page,count=1,flags=re.S)
    if 'id="current-clinical-reuse"' in page:page=re.sub(r'<section id="current-clinical-reuse">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-measurement-reuse">';require(marker in page,'current insertion missing');page=page.replace(marker,section+'\n'+marker,1)
        page=re.sub(r'(<section id="current-measurement-reuse">.*?</section>)',lambda m:'<details><summary>R38 历史批次：网络指标与统计解析修复</summary>'+m[1].replace('当前 R38：','历史 R38：',1)+'</details>',page,count=1,flags=re.S)
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix();page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已替换，仅保留R39当前文件</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R39工作图，正式full_v2未同步。</p>',page,count=1)
    page=page.replace('R38已完成整体验收；后续批次通过验收后才替换当前工作图。','R39已完成整体验收；后续批次通过验收后才替换当前工作图。')
    qa=check_html(page);j.atomic_text(j.REPORT,page);j.atomic_text(j.OUTPUT/'HANDOFF.md','# Current data and runtime — R39\n\n'+report+'\n'+next_plan)
    log_path=j.OUTPUT/'WORK_LOG.json';log=j.read_json(log_path)
    if not any(e.get('id')=='r39-clinical-literal-reuse-20260909' for e in log['events']):
        log['events'].append(dict(id='r39-clinical-literal-reuse-20260909',at=receipt['at'],status='completed',title='R39完整指标临床引用复用与依赖查明',
            body='6claim、10边引用、4处VCP修复；595+3测试及独立全图/实际查询通过。',changes='无节点/边增删、无科学字段改写；来源锚点保留，旧派生资料删除。',
            artifacts=[str(OUTPUT/n) for n in ('REPORT.md','CURRENT_ACCEPTANCE.json','DEFERRED_SCOPE_FINDINGS.json','NEXT_BATCH_PLAN.md')]))
        j.atomic_json(log_path,log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(retention_path);j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),retention=j.fingerprint(retention_path),report_code=j.fingerprint(Path(__file__)),
        report_tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),report_test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_clinical_literal_report.py'),
        retirement_dependency=j.fingerprint(j.REPO/'neurooracle/scripts/report_kg_measurement_reuse.py')))
    print('R39_PUBLISHED',dict(**counts,shared_groups=len(groups),default_groups=len(default),retired_bytes=ret['bytes_removed']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pending',action='store_true')
    pending() if parser.parse_args().pending else main()

