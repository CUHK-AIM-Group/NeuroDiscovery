"""Publish R42/R43 evidence and retire exactly four replaced R41 derivatives."""
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
from neurooracle.src.correlation_grouping import POLICY
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=j.OUTPUT/'round43_gene_boundary'
REVIEW=j.OUTPUT/'round42_source_scope'
PREVIOUS=j.OUTPUT/'round41_bulk_cleanup'


def review_counts(plan,old_rows):
    old={(r['claim_id'],r['side']) for r in old_rows}
    fixed={(e['claim_id'],ch['side']) for e in plan['events'] for ch in e['changes']}
    require(len(old)==plan['old_watchlist_endpoints'] and len(fixed)==plan['changed_endpoints'],'review scope counts differ')
    require(plan['expanded_review_endpoints']==len(old)+plan['newly_registered_lexical_candidates'],'expanded queue differs')
    require(plan['held_endpoints']+len(fixed)==plan['expanded_review_endpoints'],'remaining queue differs')
    return dict(original_watchlist=len(old),original_watchlist_repaired=len(old&fixed),
        original_watchlist_remaining=len(old-fixed),expanded_lexical_candidates=plan['newly_registered_lexical_candidates'],
        newly_discovered_repaired=len(fixed-old),expanded_remaining=plan['held_endpoints'],not_all_confirmed_errors=True)


def coverage_summary(before,after):
    a={r['field']:r for r in before if r['scope']=='node/claim'}
    b={r['field']:r for r in after if r['scope']=='node/claim'}
    require(a==b,'claim metadata coverage changed')
    n=b['metadata.raw_text']['denominator']
    return dict(claim_fields_unchanged=len(b),claims=n,
        average_nested_fields=sum(r['present'] for f,r in b.items() if f.startswith('metadata.metadata.'))/n,
        p_value_nonempty_pct=b['metadata.evidence.p_value']['nonempty_pct'],
        effect_size_nonempty_pct=b['metadata.evidence.effect_size']['nonempty_pct'],
        sample_size_nonempty_pct=b['metadata.evidence.sample_size']['nonempty_pct'])


def publish_section(section,at=None):
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="current-gene-boundary"' in page:
        page=re.sub(r'<section id="current-gene-boundary">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-bulk-cleanup">';require(marker in page,'HTML insertion point missing')
        page=page.replace(marker,section+'\n'+marker,1)
    if at:
        page=page.replace('当前已验收版本仍是下方R41；新批次正在核实，尚未宣称修改完成。',
            '当前已验收R43，详见下方最新结果；后续科学范围批次仍在推进，整体KG尚未全部优化完。')
        page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R43工作图；正式full_v2未同步。</p>',page,count=1)
        page=page.replace('<h2>当前R41：批量metadata精简和反向相关索引已应用</h2>',
            '<h2>R41历史批次：metadata精简成果已保留在当前R43</h2>')
    qa=check_html(page);j.atomic_text(j.REPORT,page)
    return qa


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='gene_boundary_repair','no active R43 writer')
    run=j.read_json(OUTPUT/'RUN_STATE.json')
    heading='候选图已构建，正在独立全图验收' if (OUTPUT/'BUILD_STATE.json').exists() else '正在构建与独立全图验收'
    section=f'''<section id="current-gene-boundary"><h2>R43{heading}</h2>
<p>800项回归已通过。计划修复{p['changed_endpoints']:,}处明确影像指标的基因误指向及{p['changed_edges']:,}条既有关联边；复用{p['reused_existing_nodes']}个节点、按完整名称共用新增{p['added_literal_nodes']:,}个普通概念节点。没有新增claim、科学断言边或metadata字段。</p>
<p>当前仍以R41验收为准。全基因端点扩大筛查发现旧449处以外的8,038处词面待核候选，并非8,038个确认错误；本批后预计还有5,937处扩大队列待核。</p>
<p>七篇原始论文已经核实：qMRI发作次数/严重程度、BACE1浓度/通路、VCP缩写物种及衰弱旧分支继续单独处理。ACC表面积有完整摘要支持，不因摘录截断直接删除。</p>
<p>状态更新：{j.local_time(run['at'])}（香港时间）。后续三项来源收口已有18项准备测试，尚未生成R44冻结计划或改图。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','限定修复计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','800项回归')} · {j.link(REVIEW/'REPORT.md','来源复核与扩大普查')} · {j.link(j.OUTPUT/'round44_source_resolution/README.md','R44接续准备')}</p></section>'''
    publish_section(section)
    hand=j.OUTPUT/'HANDOFF.md';text=hand.read_text(encoding='utf8')
    prefix='''# 最新进度：R42只读复核完成，R43正在独立验收

以CAMPAIGN.json和round43_gene_boundary/RUN_STATE.json为准，只有一个KG写进程，禁止并发写入或改动已冻结代码。R43由apply_kg_gene_boundary_repair.py run执行，800项回归已通过；2,550处端点、3,714条边、2,373新普通节点、44既有节点复用。未验收前当前仍为R41。

R42/REPORT.md记录七篇来源复核及17,722处全基因端点普查。新增8,038处待核不是确认错误；R43后预计扩大待核5,937处。后续优先qMRI错误严重程度、衰弱重复旧分支、VCP病毒/人类/小鼠范围、复合暴露和BACE1浓度/通路，不要把ACC表面积误判为无来源。

构建结束后执行report_kg_gene_boundary_repair.py完成查询、当前待核哈希检查、报告和严格限定的R41四项旧派生文件清理；不要重跑旧图构建。新夜间窗口仍截至2026-09-10T02:50Z，结束后只验收收口并停用kg-9-10。

---

'''
    if not text.startswith('# 最新进度：R42只读复核完成'):j.atomic_text(hand,prefix+text)
    if 'round44_source_resolution/README.md' not in hand.read_text(encoding='utf8'):
        j.atomic_text(hand,'R44准备已写入round44_source_resolution/README.md：18项测试通过，计划器尚未执行，apply/独立验收/报告实现尚待接续。先完成R43验收与发布，不并发写图。\n\n'+hand.read_text(encoding='utf8'))
    print('R43_PENDING_PUBLISHED',flush=True)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R43 is not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json');plan=j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status']=='CURRENT_GENE_BOUNDARY_REPAIR_APPLIED' and receipt['relation_grouping']==POLICY,'wrong adoption/index policy')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_scope_findings','current_coverage',
        'current_issues','current_structure_holds','current_paper_census','current_paper_issues','current_census_normalization')],
        receipt['tests'],receipt['provenance_plan'],*receipt['code']]:require(j.fingerprint(fp['path'])==fp,'current evidence/code changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance']==c['current_acceptance'],'runtime binding differs')
    suite=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests'))>=5 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report tests failed')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    held=rows(c['current_gene_holds']['path']);require(len(held)==plan['held_endpoints'],'remaining queue differs')
    scopes=j.read_json(c['current_scope_findings']['path'])['current_claim_hashes']
    hashes=dict(scopes)
    for r in [*held,*rows(c['current_issues']['path']),*rows(c['current_structure_holds']['path'])]:
        cid=r['claim_id'];h=r.get('claim_sha256') or r.get('current_node_sha256') or scopes.get(cid)
        require(h and (cid not in hashes or hashes[cid]==h),'inconsistent current holds')
        hashes[cid]=h
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,h in hashes.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(h,),'held claim not current')
    db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2))
    inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'actual query differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        shared_groups=len(groups),default_two_verified_sources=len(default),including_retracted=len(inclusive),
        current_held_claim_hashes_checked=len(hashes),no_scientific_consensus_inferred=True))
    comp_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        old=state['baseline']['current_shared_relations'];require(j.fingerprint(old['path'])==old,'old shared index changed')
        j.atomic_json(comp_path,dict(at=j.utc_now(),graph=c['current_graph'],**comparison(groups,rows(old['path']))))
    comp=j.read_json(comp_path);require(comp['graph']==c['current_graph'],'comparison not current')
    cov_path=OUTPUT/'COVERAGE_COMPARISON.json'
    if not cov_path.exists():
        old=state['baseline']['current_coverage'];require(j.fingerprint(old['path'])==old,'old coverage changed')
        j.atomic_json(cov_path,dict(graph=c['current_graph'],**coverage_summary(rows(old['path']),rows(c['current_coverage']['path']))))
    cov=j.read_json(cov_path);require(cov['graph']==c['current_graph'],'coverage not current')
    queue=review_counts(plan,rows(state['baseline']['current_gene_holds']['path']))
    j.atomic_json(OUTPUT/'REVIEW_QUEUE_SUMMARY.json',dict(graph=c['current_graph'],**queue))
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
            recoverability='No old copies. Current derivatives are rebuildable, but no historical KG rollback.'))
    ret=j.read_json(ret_path);at=j.local_time(receipt['at']);counts=c['counts'];rel=c['relation_evidence_counts']
    report=f'''# R43：2,550处完整影像指标脱离历史基因误指向

{at}（香港时间）已应用并独立验收。800项联合回归、4项查询契约测试及{suite.get('tests')}项报告测试通过。没有模型、训练、重新抽取或正式full_v2同步。实际查询曾因验收凭据未导出三项验证标志而被安全拒绝；已依据既有完整证明和重新核对的公开来源补齐凭据，图谱本体和读取安全检查未放宽。

## 实际修改

修复{plan['changed_claims']:,}条claim的{plan['changed_endpoints']:,}处端点及{plan['changed_edges']:,}条既有关联边，复用{plan['reused_existing_nodes']}个现有完整同名节点，补齐{plan['added_literal_nodes']:,}个按完整名称共用的普通概念节点。名称大小写、左右侧、联合指标和条件未截短；节点ID不按论文或claim加盐。基因节点本身及其离线明细均保留。

当前节点{counts['nodes']:,}，claim{counts['claims']:,}，边{counts['edges']:,}。节点增加是将原先错误挤在基因节点上的不同指标分开，不是新增论文或科学结论。同名证据复用，不能为了降低节点数而保留错误身份。

全部原始引文、论文、谓词、否定、类型、统计数值、研究条件和旧审核对象保持原样。源图完整SHA、新图独立全量扫描、逆变换全部源节点/边记录摘要、所有claim/论文/共享成员普查、全量metadata覆盖及明细库完整SHA均通过。

## 共享关系与字段

共享组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}；默认至少两篇已核实且未确认撤稿来源的组{len(default)}，含确认撤稿时{len(inclusive)}。新入共享索引{len(comp['newly_shared_claim_ids'])}条，不再共享{len(comp['no_longer_shared_claim_ids'])}条；详见RELATION_MEMBERSHIP_COMPARISON.json。保留所有claim，修复身份可能拆开原来的假共享，不以共享率越高越好。

节点/边metadata顶层字段并集仍为{c['node_metadata_field_union']}/{c['edge_metadata_field_union']}，不是每条记录的字段数。claim的{cov['claim_fields_unchanged']}项字段覆盖逐项不变，内层metadata平均{cov['average_nested_fields']:.2f}项，p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}%/{cov['effect_size_nonempty_pct']:.2f}%/{cov['sample_size_nonempty_pct']:.2f}%。没有补值或新增字段。

## 还未完成

原449处队列中本批修复{queue['original_watchlist_repaired']}处，原队列剩{queue['original_watchlist_remaining']}处；此外全基因扫描登记8,038处词面待核，本批修复其中{queue['newly_discovered_repaired']:,}处。扩大后还剩{queue['expanded_remaining']:,}处，不能与之前449直接比较成“错误变多”，也不能把所有待核当成错误。

七篇来源的结论和后续科学范围问题详见../round42_source_scope/REPORT.md。qMRI严重程度/发作次数、BACE1浓度/通路、VCP同缩写及物种、复合暴露宾语和衰弱旧分支仍需分别收口。ACC表面积确有来源支持，但观察性谓词和截断摘录仍待核。旧21语义、1结构与本轮队列可能重叠，清单不穷尽全图。

## 单当前版本与接续

当前图仍在R23/knowledge_graph.candidate.json；本轮索引、普查、覆盖和待核清单在R43，terms在R30、paper registry在R34。旧R41四项派生文件共{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）已删除。没有旧KG或完整记录前像可回退；当前派生资料可从当前KG重建，原始文献未删除。

incremental-integrity-checks用于普通进度的状态/原生指纹核对；真实构建和采用边界保留完整验证。当前SHA-256：{c['current_graph']['sha256']}。

本次goal和heartbeat kg-9-10仍有效，窗口截至2026-09-10T02:50Z（香港10:50）。到点不再开新写批次，收口并停用本次跟进；不把一批验收误报为全KG已无问题。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-gene-boundary"><h2>当前R43：2,550处影像指标误指向已修复</h2>
<div class="notice">800项联合回归、4项查询契约测试、{suite.get('tests')}项报告测试、独立全图验收和实际共享查询通过。原始科学字段和审核信息不变；无模型或训练。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}</td></tr>
<tr><td>端点 / 关联边修复</td><td>{plan['changed_endpoints']:,} / {plan['changed_edges']:,}，不新增claim或科学断言边</td></tr>
<tr><td>完整同名节点</td><td>复用{plan['reused_existing_nodes']}个；补齐{plan['added_literal_nodes']:,}个共用普通节点，不按论文重复新增</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认至少两个已核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim覆盖逐项不变</td></tr></table>
<p class="warning">扩大后的待核队列还剩{plan['held_endpoints']:,}处，并非全部确认错误。旧449处队列中还剩{queue['original_watchlist_remaining']}处；新数字包括本轮全基因扫描发现的候选。科学范围问题继续处理，不能宣布全KG已无问题。</p>
<p>节点略增是将不同影像指标从错误基因身份中分开；相同完整名称仍然共用。旧R41四项派生文件已删除，保持单当前KG，无逐步回退前像。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批完整说明')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','实际查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','当前字段覆盖')} · {j.link(OUTPUT/'REVIEW_QUEUE_SUMMARY.json','待核数口径')} · {j.link(REVIEW/'REPORT.md','来源结论及下一批')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已由R43当前文件替换</span>',page)
    j.atomic_text(j.REPORT,page)
    qa=publish_section(section,at)
    j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n\n下一批实现状态与验收注意事项见round44_source_resolution/README.md。不要并发写图或重跑历史批次。\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r43-gene-boundary-repair'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=receipt['at'],status='completed',title='R42全基因与七篇来源复核；R43批量影像身份修复',
            body='2,550处端点及3,714条边改正历史基因指向；扩大待核5,937处不等于确认错误。',
            changes='800项联合回归和独立全图、来源、逆变换、覆盖、普查及实际查询通过。当前单KG，旧R41派生资料移除。',
            artifacts=[str(OUTPUT/'REPORT.md'),str(OUTPUT/'CURRENT_ACCEPTANCE.json'),str(REVIEW/'REPORT.md')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);c['current_gene_review_summary']=j.fingerprint(OUTPUT/'REVIEW_QUEUE_SUMMARY.json')
    c['latest_source_review']=j.fingerprint(REVIEW/'REPORT.md');j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),coverage=j.fingerprint(cov_path),
        retention=j.fingerprint(ret_path),report_code=j.fingerprint(Path(__file__)),tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),
        test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_gene_boundary_report.py')))
    print('R43_PUBLISHED',dict(counts=counts,relations=rel,default_groups=len(default),queue=queue,retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pending',action='store_true')
    pending() if p.parse_args().pending else main()
