"""Publish R46 real queries, current holds, coverage and single-version retention."""
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
from report_kg_source_scope_resolution import collect_hold_hashes
from neurooracle.src import kg_research_statement_retirement as scope
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=j.OUTPUT/'round46_research_retirement'
PREVIOUS=j.OUTPUT/'round44_source_resolution'
REVIEW=j.OUTPUT/'round45_semantic_sources'


def coverage_delta(before,after,removed=4):
    a={r['field']:r for r in before if r['scope']=='node/claim'}
    b={r['field']:r for r in after if r['scope']=='node/claim'}
    n=a['metadata.raw_text']['denominator'];m=b['metadata.raw_text']['denominator']
    require(n-m==removed and set(b)<=set(a),'coverage claim/field delta differs')
    require(all(r['denominator']==n for r in a.values()) and all(r['denominator']==m for r in b.values()),'mixed claim denominators')
    changed=[]
    for field,old in a.items():
        current=b.get(field,dict(present=0,nonempty=0,types={}))
        dp=old['present']-current['present'];dn=old['nonempty']-current['nonempty']
        require(0<=dp<=removed and 0<=dn<=dp,'unexplained field removal')
        old_types=old.get('types',{});new_types=current.get('types',{})
        require(set(new_types)<=set(old_types) and all(new_types.get(t,0)<=v for t,v in old_types.items()),'coverage type grew')
        require(sum(old_types.values())-sum(new_types.values())==dp,'type-count loss differs from presence loss')
        if dp:changed.append(dict(field=field,present_removed=dp,nonempty_removed=dn))
    for field in ('p_value','effect_size','sample_size'):
        key='metadata.evidence.'+field
        require(a[key]['nonempty']==b[key]['nonempty'],'a nonempty statistic was removed from these four null-statistic claims')
    return dict(previous_claims=n,current_claims=m,field_deltas=changed,new_claim_fields=0,
        average_nested_fields=sum(r['present'] for f,r in b.items() if f.startswith('metadata.metadata.'))/m,
        p_value_nonempty_pct=b['metadata.evidence.p_value']['nonempty_pct'],effect_size_nonempty_pct=b['metadata.evidence.effect_size']['nonempty_pct'],
        sample_size_nonempty_pct=b['metadata.evidence.sample_size']['nonempty_pct'])


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='research_statement_retirement','no active R46 writer')
    section=f'''<section id="current-research-retirement"><h2>R46正在移除4条被压平的研究意图声明</h2>
<p>887项联合回归通过；当前仍以已验收R44为准。计划删除4条claim及其12条边，不新增科学声明或metadata字段。原始requires、is_designed_to_test、may_explain、may_reduce及公开摘要已逐条对应，不按综述体裁或假说关键词批量删除。</p>
<p>四条各为其论文当前唯一claim；成功后活动论文普查减少4条书目记录，但原始论文资料保留。其他17条限定语义问题与基因端点继续审查。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','限定范围和删除引用')} · {j.link(OUTPUT/'TEST_RESULTS.xml','887项联合回归')} · {j.link(REVIEW/'SOURCE_INSPECTION.json','当前全图只读预检')} · {j.link(REVIEW/'PUBLIC_SOURCE_REVIEW.md','逐篇来源判断')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');marker='<section id="semantic-source-review">'
    if 'id="current-research-retirement"' in page:page=re.sub(r'<section id="current-research-retirement">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        require(marker in page,'HTML current anchor missing');page=page.replace(marker,section+'\n'+marker,1)
    qa=check_html(page);j.atomic_text(j.REPORT,page)
    header='''# 最新：R46构建/独立验收中，R44仍为当前已验收图

当前单写进程由CAMPAIGN.json与round46_research_retirement/RUN_STATE.json登记。887项联合回归通过；禁止重复启动或编辑R46冻结FILES、PLAN和TEST_RESULTS.xml。四条指定研究意图误压为肯定关联，删除4个claim和12条拥有者边；其四篇论文各自只有这一条当前claim，所以活动普查同步减少4条书目，原始文献不删。

R45已全图SHA、当前论文普查SHA、470条当前记录/引用、21篇完整公开摘要及历史449队列当前447项交集检查。只读投影不是节点前像；首次预检因DOI大小写表示差异停止，修正为严格PMID+不区分大小写DOI后重新完整扫描通过。源论文身份和科学内容没有放宽或补造。

成功后运行report_kg_research_statement_retirement.py，先跑相应报告测试并保存REPORT_TEST_RESULTS.xml。必须通过实际共享查询、全部存续待核哈希、四个删除ID/活动书目均不在当前census和覆盖变化，再只退役R44的四项大型派生文件。当前R44图2,590,961节点/905,183claim/3,012,005边；R46未采用前不得使用计划数当结果。R46计划数2,590,957/905,179/3,011,993。

后续重点：447处旧端点中的明确字内别名误指向，以及其他扩大队列。不能按缺少显式IMAGING_MARKER类型就永久搁置，也不能把宽泛影像/临床表达一律当测量；保留完整名称与科学范围，已有同名节点先排重。BACE1浓度/通路、产前复合宾语、ACC、小鼠VCP、17条限定语义问题仍未闭合。窗口至2026-09-10T02:50Z（香港10:50），到点不再开始新写批次，验收收口并停用kg-9-10；目标仍在进行。

---

'''
    hand=j.OUTPUT/'HANDOFF.md';prior=hand.read_text(encoding='utf8')
    if not prior.startswith('# 最新：R46构建'):j.atomic_text(hand,header+prior)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R46_PENDING_PUBLISHED',flush=True)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R46 not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status']=='CURRENT_RESEARCH_STATEMENT_RETIREMENT_APPLIED' and c['counts']==p['expected_counts'],'wrong adoption/counts')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_scope_findings','current_issues','current_structure_holds',
        'current_coverage','current_paper_census','current_paper_issues','current_census_normalization','current_gene_review_summary')],
        receipt['tests'],receipt['provenance_plan'],receipt['identity_audit'],*receipt['code']]:require(j.fingerprint(fp['path'])==fp,'current frozen proof changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance']==c['current_acceptance'],'runtime binding differs')
    suite=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests'))>=6 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report tests incomplete')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    held=rows(c['current_gene_holds']['path']);issues=rows(c['current_issues']['path']);structure=rows(c['current_structure_holds']['path'])
    findings=j.read_json(c['current_scope_findings']['path']);hashes=collect_hold_hashes([*held,*issues,*structure],findings['current_claim_hashes'])
    require(not set(scope.REVIEWS)&set(hashes) and len(issues)==c['additional_semantic_or_detail_review_candidates']==17,'closed issue still held')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,h in hashes.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(h,),'current hold/census differs')
    for cid in scope.REVIEWS:require(db.execute('SELECT cid FROM claims WHERE cid=?',(cid,)).fetchone() is None,'removed claim still current')
    for retired in p['retired_active_bibliographies']:
        require(db.execute('SELECT sig FROM papers WHERE sig=?',(retired['bibliography_sha256'],)).fetchone() is None,'retired bibliography still active')
        require(db.execute('SELECT COUNT(*) FROM claims WHERE legacy_key=?',(retired['legacy_key'],)).fetchone()[0]==0,'retired current source key still active')
    db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2))
    inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'actual shared query differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        shared_groups=len(groups),default_two_verified_sources=len(default),including_retracted=len(inclusive),current_held_claim_hashes_checked=len(hashes),
        deleted_claims_absent=4,retired_active_bibliographies_absent=4,original_source_documents_not_deleted=True,
        no_scientific_consensus_or_old_audit_revalidation_inferred=True))
    comp_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        old=state['baseline']['current_shared_relations'];require(j.fingerprint(old['path'])==old,'old shared index changed')
        require(groups==rows(old['path']),'unexpected shared membership change from four singleton retirements')
        j.atomic_json(comp_path,dict(graph=c['current_graph'],**comparison(groups,rows(old['path']))))
    require(j.read_json(comp_path)['graph']==c['current_graph'],'comparison not current')
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
    report=f'''# R46：四条研究意图不再冒充实测关联

{at}（香港时间）已采用，{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图和实际查询通过。

本轮只删除四条把研究议程、设计和假说压成肯定关联的claim及12条拥有者边（4条科学边、8条about边）。分别为[研究要求](https://pubmed.ncbi.nlm.nih.gov/12788246/)、[TMAP-3设计](https://pubmed.ncbi.nlm.nih.gov/12716235/)、[表观遗传解释假说](https://pubmed.ncbi.nlm.nih.gov/14601038/)和[拟议认知治疗模型](https://pubmed.ncbi.nlm.nih.gov/12534655/)。原始requires、is_designed_to_test、may_explain、may_reduce、完整来源和当前哈希已逐项对应；不是按文章体裁批量排除。

四篇论文各自只有这一条当前claim，故活动论文普查减少4条书目和4个来源键。原始论文资料、公开摘要、非目标概念和离线明细未删除；没有造新结论、填统计值、把其他假说标成错误或修改存续审核章。

当前{n['nodes']:,}节点、{n['claims']:,}条claim、{n['edges']:,}条边。metadata字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}，新增0。claim内层平均{cov['average_nested_fields']:.2f}项；p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}%/{cov['effect_size_nonempty_pct']:.2f}%/{cov['sample_size_nonempty_pct']:.2f}%。四条删除记录的这些统计均空，所以非空统计个数不变，只有分母变化。

共享组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}，全部成员与前轮相同。默认至少两篇已核实未确认撤稿来源{len(default)}组，含确认撤稿{len(inclusive)}组；不是共识或独立实验数。

剩余17条旧语义待核，不能统一翻成无关联。扩大端点待核{len(held):,}处；R45当前源复核确认历史449队列仍有447处。BACE1浓度/通路、产前复合宾语、ACC观察性谓词/截断摘录、小鼠VCP等继续处理。各队列可重叠且并不穷尽全图。

所有保留节点和边的源记录摘要由独立全图重新计算，逐一保留科学字段和审核含义，删除ID精确引用为0。所有当前claim/论文/共享成员、引用、元数据覆盖、出版状态验证通过；详情库和当前普查完整SHA通过。实际查询及所有存续待核claim哈希已验证。

被替代的R44四项大型派生文件已精确退役，共{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。只保留当前工作KG，没有历史KG或完整记录前像可回退；当前派生资料可从当前KG重建。当前图仍在R23路径，当前普查/索引/覆盖/待核在R46；terms在R30，paper registry在R34。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。无模型、KGE、训练、重新抽取或正式同步。incremental-integrity-checks用于普通状态读取，实际修改边界完整验证。本次目标继续至2026-09-10T02:50Z（香港10:50），全KG未全部优化完。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-research-retirement"><h2>当前R46：四条研究意图声明已精确移除</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图及实际查询通过。删除4条claim和12条边；无新增节点、metadata字段、科学结论或审核章。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨已核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}，新增0</td></tr>
<tr><td>待核</td><td>17条旧语义；扩大端点{len(held):,}处，非全部确认错误</td></tr></table>
<p>四条各为其论文当前唯一claim，因此活动书目索引减少4条；原始文献资料仍在。其它存续科学记录和审核含义不变。旧R44四项派生文件已清理，无旧KG回退副本。</p>
<p class="warning">历史447处端点、其他扩大队列及BACE1、产前、ACC、小鼠VCP范围继续审查；整体KG尚未全部优化完。</p>
<p>{j.link(OUTPUT/'REPORT.md','完整结果与来源')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','实际查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','覆盖变化')} · {j.link(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl','17条当前语义待核')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8');require('id="current-research-retirement"' in page,'pending HTML missing')
    page=re.sub(r'<section id="current-research-retirement">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    page=page.replace('<h2>当前R44：来源与旧分支修复已验收</h2>','<h2>R44历史批次：来源和旧分支修复成果保留</h2>')
    page=page.replace('当前已验收R44，详见下方最新结果；后续来源和端点复核继续，整体KG尚未全部优化完。','当前已验收R46，详见下方最新结果；端点与限定语义继续审查，整体KG尚未全部优化完。')
    page=page.replace('目前没有据此写入或删除KG。','此为R45审查时的状态；其中四条已在R46按完整闭包移除，参见最新结果。')
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已由R46当前文件替换</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R46工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n\n'+(REVIEW/'PUBLIC_SOURCE_REVIEW.md').read_text(encoding='utf8'))
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r46-research-statement-retirement'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=receipt['at'],status='completed',title='R46精确移除四条研究意图误压声明',
            body='删除4个claim和12条边；四篇各只有一条当前claim，活动书目同步减少4条，原文资料仍在。',
            changes=f"{receipt['test_counts']['tests']}项回归、独立全图和真实查询通过；存续科学/audit不变，17条旧语义继续待核。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);c['latest_source_review']=j.fingerprint(OUTPUT/'REPORT.md');j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),coverage=j.fingerprint(cov_path),retention=j.fingerprint(ret_path),
        report_code=j.fingerprint(Path(__file__)),tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_research_retirement_report.py')))
    print('R46_PUBLISHED',dict(counts=n,relations=rel,default_groups=len(default),pending_endpoints=len(held),remaining_old_semantic=len(issues),retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pending',action='store_true')
    pending() if p.parse_args().pending else main()
