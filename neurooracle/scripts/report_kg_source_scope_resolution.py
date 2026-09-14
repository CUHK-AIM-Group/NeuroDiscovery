"""Publish current R44 source closure, real query evidence and single-version retention."""
from copy import deepcopy
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
from neurooracle.src import kg_source_scope_resolution as scope
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=j.OUTPUT/'round44_source_resolution'
PREVIOUS=j.OUTPUT/'round43_gene_boundary'
REVIEW=j.OUTPUT/'round42_source_scope'


def coverage_delta(before,after):
    a={r['field']:r for r in before if r['scope']=='node/claim'}
    b={r['field']:r for r in after if r['scope']=='node/claim'}
    n=a['metadata.raw_text']['denominator'];m=b['metadata.raw_text']['denominator']
    require(n-m==1 and set(b)<=set(a),'coverage claim/field delta differs')
    changed=[]
    for field,old in a.items():
        current=b.get(field,dict(present=0,nonempty=0,types={}))
        dp=old['present']-current['present'];dn=old['nonempty']-current['nonempty']
        require(dp in {0,1} and 0<=dn<=dp,'more than one removed claim in coverage delta')
        old_types=old.get('types',{});new_types=current.get('types',{})
        require(set(new_types)<=set(old_types) and all(new_types.get(t,0)<=v for t,v in old_types.items()),'coverage type grew')
        if dp:changed.append(dict(field=field,present_removed=dp,nonempty_removed=dn))
    return dict(previous_claims=n,current_claims=m,field_deltas=changed,new_claim_fields=0,
        average_nested_fields=sum(r['present'] for f,r in b.items() if f.startswith('metadata.metadata.'))/m,
        p_value_nonempty_pct=b['metadata.evidence.p_value']['nonempty_pct'],effect_size_nonempty_pct=b['metadata.evidence.effect_size']['nonempty_pct'],
        sample_size_nonempty_pct=b['metadata.evidence.sample_size']['nonempty_pct'])


def collect_hold_hashes(records,scope_hashes):
    result=dict(scope_hashes)
    for r in records:
        cid=r['claim_id'];h=r.get('claim_sha256') or r.get('current_node_sha256') or scope_hashes.get(cid)
        require(h and (cid not in result or result[cid]==h),'inconsistent current hold hash')
        result[cid]=h
    return result


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='source_scope_resolution','no active R44 writer')
    section=f'''<section id="current-source-resolution"><h2>R44正在来源修复与独立全图验收</h2>
<p>846项联合回归通过。计划删除1条qMRI严重程度肯定声明及2条about边，退役3条旧衰弱重复分支边；复用已存在的完整病毒蛋白节点，修复VCP的{p['changed_edges']}条关联边。没有新增节点、科学断言或metadata字段。</p>
<p>当前仍以R43验收为准。原始论文、其他claim、所有非目标概念及离线明细保留。三篇既有引用的公开标题/DOI已核实，不能据此把这些claim的科学结论自动标为已验证。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','完整范围及排重计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','846项联合回归')} · {j.link(OUTPUT/'VIRAL_REUSE_SOURCE_FETCH.json','已有病毒节点的公开来源')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');marker='<section id="current-gene-boundary">'
    if 'id="current-source-resolution"' in page:page=re.sub(r'<section id="current-source-resolution">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        require(marker in page,'HTML current anchor missing');page=page.replace(marker,section+'\n'+marker,1)
    qa=check_html(page);j.atomic_text(j.REPORT,page)
    hand=j.OUTPUT/'HANDOFF.md';text=hand.read_text(encoding='utf8')
    header='''# 最新：R44正在写入/独立验收，R43仍为当前已验收版本

当前单写进程以CAMPAIGN.json及round44_source_resolution/RUN_STATE.json为准；禁止重复启动、修改冻结代码或开始其他KG写入。846项回归通过，PLAN.json revision 2已完整源SHA、全部删除ID精确引用、实际论文/明细与既有病毒节点3处引用复核。

R44仅删除qMRI严重程度错误肯定claim及2条about边、退役衰弱旧分支3条边；复用已有完整病毒蛋白节点改正VCP的2条边。无新增节点、无新科学断言、无模型/训练/正式同步。R43的旧R41四项派生资料已清理，R43实际查询与HTML验证已通过。

成功后运行report_kg_source_scope_resolution.py（先跑对应报告测试并生成REPORT_TEST_RESULTS.xml），核对当前全部待核hash、实际查询、覆盖与单版本保留，更新HTML和HANDOFF。R44 apply已直接导出完整查询能力标志，不应再次修改已冻结的R43脚本来解决查询凭据。若进程异常先核实真实终止和BUILD_STATE/VALIDATED，只有具备完整构建状态才可resume；不得从失联观察直接重启。

下一步仍需对BACE1浓度/通路、产前复合宾语、ACC观察性谓词/截断摘录、小鼠VCP及21条旧语义问题继续按来源复核。扩大待核队列不是确认错误数。窗口到2026-09-10T02:50Z，届时不再启动新批次，只验收收口并停用kg-9-10。

---

'''
    if not text.startswith('# 最新：R44正在'):j.atomic_text(hand,header+text)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R44_PENDING_PUBLISHED',flush=True)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R44 not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status']=='CURRENT_SOURCE_SCOPE_RESOLUTION_APPLIED' and c['counts']==p['expected_counts'],'wrong adoption/counts')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_scope_findings','current_issues','current_structure_holds',
        'current_coverage','current_paper_census','current_paper_issues','current_census_normalization','current_gene_review_summary')],
        receipt['tests'],receipt['provenance_plan'],receipt['identity_audit'],*receipt['code']]:require(j.fingerprint(fp['path'])==fp,'current frozen proof changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance']==c['current_acceptance'],'runtime binding differs')
    suite=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests'))>=6 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report tests incomplete')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    held=rows(c['current_gene_holds']['path']);issues=rows(c['current_issues']['path']);structure=rows(c['current_structure_holds']['path'])
    findings=j.read_json(c['current_scope_findings']['path']);hashes=collect_hold_hashes([*held,*issues,*structure],findings['current_claim_hashes'])
    require(scope.MYELIN not in hashes and scope.FRAILTY not in {r['claim_id'] for r in structure},'closed issue still held')
    if p['events']:require(not any(r['claim_id']==scope.VIRUS and r['side']=='subject' for r in held),'viral identity still held')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,h in hashes.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(h,),'current hold/census differs')
    require(db.execute('SELECT cid FROM claims WHERE cid=?',(scope.MYELIN,)).fetchone() is None,'removed claim still current')
    require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(scope.FRAILTY,)).fetchone()==(p['frailty_claim_sha256'],),'frailty scientific claim changed')
    for e in p['events']:require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(e['claim_id'],)).fetchone()==(e['current_node_sha256'],),'identity claim not current')
    remaining_paper_claims=db.execute('SELECT COUNT(*) FROM claims c JOIN papers p ON p.sig=c.paper_sig WHERE p.pmid=?',('28526817',)).fetchone()[0]
    require(remaining_paper_claims>=1,'own original paper unexpectedly disappeared');db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2))
    inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'actual shared query differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        shared_groups=len(groups),default_two_verified_sources=len(default),including_retracted=len(inclusive),current_held_claim_hashes_checked=len(hashes),
        original_myelin_paper_remaining_claims=remaining_paper_claims,no_scientific_consensus_or_old_audit_revalidation_inferred=True))
    comp_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        old=state['baseline']['current_shared_relations'];require(j.fingerprint(old['path'])==old,'old shared index changed')
        j.atomic_json(comp_path,dict(graph=c['current_graph'],**comparison(groups,rows(old['path']))))
    comp=j.read_json(comp_path);require(comp['graph']==c['current_graph'],'comparison not current')
    cov_path=OUTPUT/'COVERAGE_COMPARISON.json'
    if not cov_path.exists():
        old=state['baseline']['current_coverage'];require(j.fingerprint(old['path'])==old,'old coverage changed')
        j.atomic_json(cov_path,dict(graph=c['current_graph'],**coverage_delta(rows(old['path']),rows(c['current_coverage']['path']))))
    cov=j.read_json(cov_path);require(cov['graph']==c['current_graph'],'coverage not current')
    ret_path=OUTPUT/'RETENTION_RESULT.json'
    if not ret_path.exists():
        targets=[j.read_json(state['baseline']['current_paper_census']['path'])['database'],
            *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current={Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths=validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)
        for fp,path in zip(targets,paths):j.guards(fp);require(sha256(path)==fp['sha256'],'obsolete derivative changed')
        require(j.read_json(control)==c,'campaign advanced')
        for path in paths:path.unlink()
        j.atomic_json(ret_path,dict(status='COMPLETED',at=j.utc_now(),removed=targets,bytes_removed=sum(fp['bytes'] for fp in targets),
            graph_backups=0,record_preimages_saved=False,original_archives_and_public_evidence_deleted=False,
            recoverability='No historical copies. Current derivatives can be rebuilt from current KG; no old KG rollback.'))
    ret=j.read_json(ret_path);at=j.local_time(receipt['at']);n=c['counts'];rel=c['relation_evidence_counts']
    report=f'''# R44：来源范围与重复旧分支收口

{at}（香港时间）已采用并完成独立全图、实际查询和当前待核哈希核对。{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试通过。没有模型、训练、重新抽取或正式full_v2同步。

## 一批完成的三件事

删除1条将LPFC髓鞘与抑郁症状严重程度肯定关联的人工泛化claim及其2条about边。完整原文报告该严重程度关系不显著；不能将发作次数关系移植过来，也不从null结果重新制造claim。原论文仍有{remaining_paper_claims}条其他claim，公开全文和原始数据没有删除。见[原始论文](https://www.nature.com/articles/s41598-017-02062-y)。

衰弱claim及当前WMH/general frailty规范3条边全部保留，退役3条旧分支边。它们与保留分支的所有非端点内容完全相同，已逐字段和哈希证明；不是删除另外三条独立科学证据。一般、身体、认知衰弱概念没有全局合并，所有概念及明细依赖保留。见[原始论文记录](https://pubmed.ncbi.nlm.nih.gov/41297452/)。

病毒VCP身份修复{p['changed_claims']}条claim及{p['changed_edges']}条既有边，复用既存完整蛋白节点：{p['reused_existing_viral_node']}。新增普通节点{p['added_literal_nodes']}。已核对目标的3处完整名称引用和各自公开标题/DOI；原claim仍保留VCP写法，不新增全局VCP别名，不影响人类或小鼠valosin-containing protein节点。旧审核对象及这三条旧claim的科学结论没有重新签发或自动升级。见[本篇病毒VCP来源](https://pubmed.ncbi.nlm.nih.gov/29534078/)。

## 当前图与metadata

节点{n['nodes']:,}、claim{n['claims']:,}、边{n['edges']:,}。本轮节点减少1、边减少5；共享关系组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}。默认至少两篇已核实且未确认撤稿来源的组{len(default)}，含确认撤稿时{len(inclusive)}。共享不是科学共识，也不等于独立实验数。

metadata字段并集仍为节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}，没有新增字段。claim内层平均{cov['average_nested_fields']:.2f}项；p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}%/{cov['effect_size_nonempty_pct']:.2f}%/{cov['sample_size_nonempty_pct']:.2f}%。本轮覆盖变化只来自删除这1条claim及分母变化，没有填值、删掉其他claim字段或修改统计null的审核含义。

扩大端点待核还剩{len(held):,}处、{c['pending_gene_link_claims']:,}条claim，不是全为已确认错误。旧语义待核{len(issues)}条，当前登记的衰弱结构待核已关闭；各队列可能重叠且不穷尽全图。BACE1浓度/通路、产前复合宾语、ACC观察性谓词/截断摘录、小鼠VCP及其他范围问题继续保留。

## 验证和单版本

源图完整SHA与指定删除/改动范围核对；候选图独立完整读取，全部保留源记录摘要经逆变换再现，删除claim的精确引用为0。独立重构3条旧分支的原哈希，证明只退役内容重复的旧端点表达。所有存续论文、引文、否定、条件、科学统计和审核对象不变，当前普查/全量覆盖/共享索引逐项通过。必要明细和新普查完整SHA通过。

已删除被替代的R43四项派生资料，共{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。没有旧KG或完整记录前像可回退；当前派生資料可从当前KG重建。当前图仍在R23路径，当前普查、索引、覆盖和待核在R44；terms在R30，paper registry在R34。正式full_v2仍905,274条claim，累计差额{c['formal_sync_gap_claims']}条，未同步。

使用incremental-integrity-checks管理状态读取，实际构建/采用边界执行完整验证。当前SHA-256：{c['current_graph']['sha256']}。夜间目标和kg-9-10继续，窗口截至2026-09-10T02:50Z（香港10:50）；整体KG未全部优化完。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-source-resolution"><h2>当前R44：来源与旧分支修复已验收</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图及实际查询通过。删除1条不受来源支持的claim及2条about边，退役3条旧重复分支边，修复病毒VCP的2条关联边；没有新增节点或metadata字段。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨已核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim内层平均{cov['average_nested_fields']:.2f}项</td></tr>
<tr><td>扩大端点待核</td><td>{len(held):,}处，并非全部确认错误；旧语义待核{len(issues)}条</td></tr></table>
<p>病毒蛋白复用了既有完整名称节点，不再造重复节点。原论文仍有{remaining_paper_claims}条其他claim，全部非目标节点与离线明细保留，衰弱的规范关系仍在；旧审核没有自动升级。</p>
<p class="warning">整体KG尚未全部优化完，BACE1、产前复合宾语、ACC、小鼠VCP等仍待核。只做KG，无模型/训练/正式同步；旧R43四项派生资料已移除，无逐步回退副本。</p>
<p>{j.link(OUTPUT/'REPORT.md','完整来源和数量说明')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','实际查询与存续论文')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','字段覆盖变化')} · {j.link(OUTPUT/'CURRENT_SCOPE_FINDINGS.json','继续待核')} · {j.link(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl','当前扩大端点队列')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');require('id="current-source-resolution"' in page,'pending HTML missing')
    page=re.sub(r'<section id="current-source-resolution">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    page=page.replace('<h2>当前R43：2,550处影像指标误指向已修复</h2>','<h2>R43历史批次：2,550处影像指标修复成果已保留</h2>')
    page=page.replace('当前已验收R43，详见下方最新结果；后续科学范围批次仍在推进，整体KG尚未全部优化完。',
        '当前已验收R44，详见下方最新结果；后续来源和端点复核继续，整体KG尚未全部优化完。')
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已由R44当前文件替换</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R44工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r44-source-resolution'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=receipt['at'],status='completed',title='R44来源声明、旧衰弱分支与病毒VCP复用',
            body='删除1条不受来源支持的claim及2条about；退役3条重复旧分支；修复病毒VCP的2条边，复用现有节点。',
            changes=f"{receipt['test_counts']['tests']}项回归和独立全图、来源、当前查询验证；单当前KG，旧派生资料移除。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);c['latest_source_review']=j.fingerprint(OUTPUT/'REPORT.md');j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),coverage=j.fingerprint(cov_path),retention=j.fingerprint(ret_path),
        report_code=j.fingerprint(Path(__file__)),tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_source_scope_report.py')))
    print('R44_PUBLISHED',dict(counts=n,relations=rel,default_groups=len(default),pending_endpoints=len(held),remaining_old_semantic=len(issues),
        old_structure_holds=len(structure),retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pending',action='store_true')
    pending() if p.parse_args().pending else main()
