"""R56/R57 source-defined scientific corrections, honest queues and current-only retention."""
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
from report_kg_gene_boundary_repair import coverage_summary as strict_coverage
from report_kg_literal_endpoint_repair import comparison
from report_kg_measurement_reuse import validate_retirement_paths
from report_kg_source_scope_resolution import collect_hold_hashes
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations
from neurooracle.src.kg_scientific_definition_repair import ENIGMA,FES,DPD_A,DPD_B,update_scope_findings

OUTPUT=j.OUTPUT/'round57_scientific_definition_repair'
REVIEW=j.OUTPUT/'round56_scientific_definition_review'
PREVIOUS=j.OUTPUT/'round55_existing_literal_reuse'
LAST_HISTORIC=('CLM:CASE1MAN:15038994:11676','subject')
FIXED_HISTORIC={(ENIGMA,'subject'),(FES,'subject')}
HISTORIC_THREE={LAST_HISTORIC,*FIXED_HISTORIC}


def queue_summary(plan,before,after):
    old={(r['claim_id'],r['side']) for r in before};current={(r['claim_id'],r['side']) for r in after}
    changed={(e['claim_id'],ch['side']) for e in plan['events'] for ch in e['changes']}
    fixed=changed&old
    require(len(old)==len(before)==plan['old_watchlist_endpoints'],'old queue cardinality differs')
    require(len(current)==len(after)==plan['held_endpoints'],'current queue cardinality differs')
    require(len(changed)==plan['changed_endpoints'] and fixed==FIXED_HISTORIC
        and len(fixed)==plan['repaired_queued_endpoints'] and current==old-fixed,'exact queued/nonqueued subtraction differs')
    require(HISTORIC_THREE<=old and current&HISTORIC_THREE=={LAST_HISTORIC}
        and plan['original_watchlist_449_remaining_after']==1 and plan['original_watchlist_449_repaired']==2,'historic remaining claim must stay open')
    return dict(original_watchlist=449,original_watchlist_remaining_before=3,original_watchlist_repaired_this_batch=2,
        original_watchlist_remaining=1,expanded_repaired_this_batch=2,expanded_remaining=len(current),
        expanded_claims=len({cid for cid,_ in current}),not_all_confirmed_errors=True)


def coverage_summary(before,after):
    """Exactly three empty direction values acquire source-supported direction."""
    adjusted=deepcopy(before)
    direction=[r for r in adjusted if r['scope']=='node/claim' and r['field']=='metadata.evidence.direction']
    require(len(direction)==1,'one claim direction coverage row required')
    row=direction[0];old_nonempty=row['nonempty']
    require(row['empty_or_null']>=3 and row['denominator']>0,'direction baseline invalid')
    row['nonempty']+=3;row['empty_or_null']-=3
    row['nonempty_pct']=round(100*row['nonempty']/row['denominator'],6)
    result=strict_coverage(adjusted,after)
    result['claim_fields_unchanged']-=1
    result.update(claim_fields_with_coverage_change=1,changed_direction_values=3,
        direction_nonempty_before=old_nonempty,direction_nonempty_after=row['nonempty'],
        nonempty_change_field='metadata.evidence.direction',
        statistical_field_coverage_unchanged=True,no_metadata_fields_added=True)
    return result


def scope_validation(plan,before,after,graph,plan_fp):
    expected=update_scope_findings(before,plan)
    expected.update(at=after['at'],graph=graph,latest_scientific_definition_plan=plan_fp)
    require(after==expected,'scientific issue register not exactly reproduced')
    require(len(after['findings'])==len(after['current_claim_hashes'])==6
        and after['old_audits_not_revalidated'] and not after['issue_register_is_exhaustive'],
        'scientific findings or audit boundary differ')


def reviewed_source_report(plan):
    require(j.fingerprint(plan['source_review']['path'])==plan['source_review'],'R56 proof changed')
    inspection=j.read_json(plan['source_review']['path'])
    text=f'''# R56：测量定义与科学表达的原文复核

R56对当前R53的31条有限记录、288个端点/候选节点和4,233处现有关联做了完整源SHA与引用检查。R55完成后，R57重新核对当前图；新增的一条既有关联CLM:CASE1MAN:33668432:9546严格凭R55计划、全源逆验证、当前节点哈希、科学字段摘要和完整名称衔接，旧关联不丢弃。第一次预检曾因遗漏这条新增关联而安全停止，没有修改KG；修正检查程序后新增11项回归。

## 已核实的有限科学修改

- BACE1：原文测量血浆BACE1浓度，不是通路活性。修正subject名称/身份及粗粒度PATHWAY类型；当前抽取七类契约用IMAGING_MARKER容纳影像及非影像测量，这不表示血浆是用影像测得。负相关、MCI群体、原统计和原审核保持。[PubMed原摘要](https://pubmed.ncbi.nlm.nih.gov/36847009/)
- ACC：原文支持不利环境与前扣带皮层表面积的正向观察性关联，不足以支持因果“增加”。只修正谓词和方向；原始截断摘录、p值等不补写，旧审核不升级。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10507487/)
- DPD：论文分别报告左右海马与抑郁严重程度的负相关；不是把左右相加或平均。两条记录统一测量和严重程度名称并明确negative，保留各自旧审核与claim ID；同一论文只算一个来源。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10784022/)
- 儿童体适能/记忆：该分析使用左右海马体积总和，修正为sum；不自动补齐不同分析的样本量。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3953557/)
- CLDN5：当前血液位点与体积关联分别检验左右侧；文章另一个后续中介分析使用平均值，不能混用。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13317736/)
- ENIGMA：体积定义为双侧平均；仅纠正这个端点，复合rs7294919/TESC表达客体仍另列待核。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3635491/)
- FES：每侧分别报告，因此纠正为左右分别测量；保留原整队列N244，不猜测成某个对照比较的N。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8821164/)

所有全文都核对本篇front PMID/DOI，不能拿参考文献中的相同数字当作本篇身份。新增公开全文3篇，原摘要和此前全文均保留在原目录。没有付费或认证绕过，没有重新抽取。

## 仍不确定

历史PMID15038994只见摘要，双侧聚合定义不明，暂不决定平均/总和/分别。BACE1海马聚合和旧通路审核、ACC截断摘录的历史溯源、产前暴露复合客体、ENIGMA复合客体继续登记。两条端粒记录N170与本篇摘要的30患者/60对照不一致，性别分层分母未知，不能直接替换为90或清空。[端粒摘要](https://pubmed.ncbi.nlm.nih.gov/30384145/)

旧audit作为历史记录保存，不声称通过新科学审核。候选当前科学登记6条，不等于6条都应删除，也不穷尽全KG问题。

检查时间：{inspection['at']}；当前源计划：{plan['at']}。无模型、KGE、训练、重新抽取或正式full_v2同步。
'''
    j.atomic_text(REVIEW/'REPORT.md',text)


def publish_section(section,at=None,current=False):
    page=j.REPORT.read_text(encoding='utf8')
    marker='<section id="current-scientific-definition">'
    if marker in page:
        page=re.sub(r'<section id="current-scientific-definition">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        old='<section id="current-existing-literal">'
        require(old in page,'HTML R55 anchor missing');page=page.replace(old,section+'\n'+old,1)
    if current:
        page=page.replace('<h2>当前R55：复用已有完整名称节点已验收</h2>','<h2>R55历史批次：复用已有完整名称节点已验收</h2>')
        page=page.replace('当前已验收R55，详见下方最新结果；科学范围及其余身份继续审查，整体KG尚未全部优化完。',
            '当前已验收R57，详见下方最新结果；已修正字段与仍待核问题分开列出，整体KG尚未全部优化完。')
        page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R57工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);return qa


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='scientific_definition_repair','no R57 writer')
    require(c['current_graph']==p['graph'] and c['current_acceptance']==p['source_acceptance'],'pending source differs')
    reviewed_source_report(p)
    section=f'''<section id="current-scientific-definition"><h2>R57构建/独立验收中：8条原文支持的科学定义纠正</h2>
<p>1,214项联合回归通过。计划8条claim、9处端点/{p['changed_edges']}条既有边；复用2个节点、补齐3个共用普通节点。不新增claim或边，不删记录，不改原始摘录、统计数值、研究条件、来源和旧audit。</p>
<p>限定修正BACE1浓度/通路、ACC观察性关联、DPD统一表达，以及海马平均/总和/两侧分别报告。1个谓词、3个方向值和1个粗粒度类型值改变，没有新增metadata字段。</p>
<p class="warning">当前仍以R55为准；计划不是完成数。历史449预计剩1处，扩大队列2,206预计剩2,204处；其余7处本批端点不在该队列，不能重复计功。复合客体、样本分母和历史审核继续保留问题登记。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','冻结计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','1214项回归')} · {j.link(REVIEW/'REPORT.md','原文与修改边界')}</p></section>'''
    qa=publish_section(section)
    header=f'''# 最新：R57科学定义纠正正在构建与独立验收，R55仍为当前图

唯一KG写进程见CAMPAIGN.json及R57/RUN_STATE.json。1214项联合回归通过；不要改冻结FILES、PLAN.json或TEST_RESULTS.xml，不启动第二个KG writer。
计划8条claim/9端点/{p['changed_edges']}既有边，新增3普通节点、复用2节点，无claim/边删除或新增；原始摘录/统计/audit保留。R55新增的33668432:9546关联已凭正式R55验收重新绑定，未放宽目标全关联检查。

成功采用后运行test_kg_scientific_definition_report.py保存REPORT_TEST_RESULTS.xml，再执行report_kg_scientific_definition_repair.py，完成真实查询、全部待核当前hash、DPD同关系同论文、2206减2差集、6项科学登记和只增加3个direction非空值的字段覆盖比较；之后仅退役R55四项大型派生。当前KG仍在R23，不保留旧KG前像。

当前R55：2594104节点/905179claim/3011990边；扩大2206端点/2061claim；历史449剩3；旧语义17/结构0。共享2587组/5868成员；默认至少两核实未撤稿来源519。R53四项旧派生766614821B已退役。无模型/训练/full_v2同步。

窗口到2026-09-10T02:50Z（香港10:50），到点不新开writer，完成必要验收收口并停用kg-9-10。不要把一批验收误报为全KG完成。

---

'''
    hand=j.OUTPUT/'HANDOFF.md';old=hand.read_text(encoding='utf8')
    if not old.startswith('# 最新：R57'):j.atomic_text(hand,header+old)
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r56-source-defined-scientific-review'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=p['at'],status='completed',title='R56原文测量定义及科学表达复核',
            body='31条当前科学记录/288个节点/4233关联；七篇有限来源形成8条字段修正规则，另公开获取三篇全文。',
            changes='R55新增一条现有关联已重新绑定；1214项测试通过，修改边界不包含原始证据、统计或旧audit。',
            artifacts=[str(REVIEW/'REPORT.md'),str(REVIEW/'SOURCE_INSPECTION.json'),str(OUTPUT/'PLAN.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R57_PENDING_PUBLISHED',flush=True)


def main():
    control = j.OUTPUT/'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R57 not current')
    receipt = j.read_json(c['current_acceptance']['path']); state = j.read_json(OUTPUT/'BUILD_STATE.json'); p = j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status'] == 'CURRENT_SCIENTIFIC_DEFINITION_REPAIR_APPLIED', 'wrong adoption')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_scope_findings','current_issues','current_structure_holds',
        'current_coverage','current_paper_census','current_paper_issues','current_census_normalization','current_gene_review_summary')],
        receipt['tests'],receipt['provenance_plan'],receipt['identity_audit'],*receipt['code']]: require(j.fingerprint(fp['path']) == fp, 'current frozen proof changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance'] == c['current_acceptance'], 'runtime binding differs')
    suite = ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests')) >= 10 and all(int(suite.get(k)) == 0 for k in ('failures','errors','skipped')), 'report tests incomplete')
    census = j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    held = rows(c['current_gene_holds']['path']); issues = rows(c['current_issues']['path']); structure = rows(c['current_structure_holds']['path'])
    findings = j.read_json(c['current_scope_findings']['path'])
    scope_validation(p,j.read_json(state['baseline']['current_scope_findings']['path']),findings,c['current_graph'],receipt['provenance_plan'])
    hashes = collect_hold_hashes([*held,*issues,*structure], findings['current_claim_hashes'])
    db = sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro', uri=True)
    try:
        for cid, h in hashes.items(): require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone() == (h,), 'held claim/census differs')
        for event in p['events']:
            require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(event['claim_id'],)).fetchone() == (event['current_node_sha256'],event['new_relation_id']), 'repaired claim/census differs')
        dpd = [db.execute('SELECT paper_sig,relation_id FROM claims WHERE cid=?',(cid,)).fetchone() for cid in (DPD_A,DPD_B)]
        require(dpd[0] is not None and dpd[0]==dpd[1] and all(dpd[0]), 'DPD must share relation and one source paper')
    finally: db.close()
    require(len(issues) == c['additional_semantic_or_detail_review_candidates'] == 17 and not structure, 'unrelated issue set changed')
    queue = queue_summary(p, rows(state['baseline']['current_gene_holds']['path']), held)
    actual_summary = j.read_json(c['current_gene_review_summary']['path'])
    require(all(actual_summary.get(k) == v for k,v in queue.items()) and actual_summary['graph'] == c['current_graph'], 'current summary not derived from queue')
    groups = list(current_shared_relations(control)); default = list(find_shared_relations(control, minimum_papers=2))
    inclusive = list(find_shared_relations(control, minimum_papers=2, include_retracted=True))
    require(len(groups) == c['relation_evidence_counts']['shared_groups'], 'actual shared query differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json', dict(at=j.utc_now(), graph=c['current_graph'], acceptance=c['current_acceptance'],
        shared_groups=len(groups), default_two_verified_sources=len(default), including_retracted=len(inclusive),
        current_held_claim_hashes_checked=len(hashes), repaired_claim_hashes_and_relation_ids_checked=len(p['events']),
        scientific_findings_current_hashes=len(findings['current_claim_hashes']),dpd_same_relation_single_source_paper=True,
        no_scientific_consensus_or_old_audit_revalidation_inferred=True))
    comp_path = OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        old = state['baseline']['current_shared_relations']; require(j.fingerprint(old['path']) == old, 'old shared index changed')
        j.atomic_json(comp_path, dict(graph=c['current_graph'], **comparison(groups, rows(old['path']))))
    comp = j.read_json(comp_path); require(comp['graph'] == c['current_graph'], 'comparison not current')
    cov_path = OUTPUT/'COVERAGE_COMPARISON.json'
    if not cov_path.exists():
        old = state['baseline']['current_coverage']; require(j.fingerprint(old['path']) == old, 'old coverage changed')
        j.atomic_json(cov_path, dict(graph=c['current_graph'], **coverage_summary(rows(old['path']), rows(c['current_coverage']['path']))))
    cov = j.read_json(cov_path); require(cov['graph'] == c['current_graph'], 'coverage not current')
    ret_path = OUTPUT/'RETENTION_RESULT.json'
    if not ret_path.exists():
        targets = [j.read_json(state['baseline']['current_paper_census']['path'])['database'],
            *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current = {Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths = validate_retirement_paths(targets, current, previous=PREVIOUS, root=j.OUTPUT)
        for fp,path in zip(targets,paths): j.guards(fp); require(sha256(path) == fp['sha256'], 'obsolete derivative changed')
        require(j.read_json(control) == c, 'campaign advanced')
        for path in paths: path.unlink()
        j.atomic_json(ret_path, dict(status='COMPLETED', at=j.utc_now(), removed=targets, bytes_removed=sum(fp['bytes'] for fp in targets),
            graph_backups=0, record_preimages_saved=False, original_archives_and_public_evidence_deleted=False,
            recoverability='No old copies. Current derivatives are rebuildable from the current KG; no old KG rollback.'))
    ret=j.read_json(ret_path);at=j.local_time(receipt['at']);n=c['counts'];rel=c['relation_evidence_counts']
    reviewed_source_report(p)
    report=f'''# R57：8条原文支持的科学定义与表达修正已验收

{at}（香港时间）采用；{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图与真实查询通过。

## 实际修改

8条claim / 9处端点 / {p['changed_edges']}条既有边。新增3个按完整名称共用的普通节点，复用2个已有节点；没有新claim或新边，没有删除claim/边/概念。有限变更含1个谓词、3个方向值、1个粗粒度声明类型值，没有新增metadata字段。

BACE1从通路改为血浆浓度；ACC从“增加”改为正向观察性关联；两条DPD记录统一为分别两侧海马体积与DPD抑郁严重程度的负相关。儿童分析用两侧总和，ENIGMA用两侧平均，CLDN5与FES当前分析用左右分别报告。来源和精确范围见R56/REPORT.md，不能将不同测量定义合并。

DPD两个claim和各自旧审核仍保留，但当前关系及来源纸本身份完全相同，只计一篇论文，不宣称两项独立研究。原始raw_text、摘录位置、统计值、研究条件、来源书目、置信度和旧audit均完整保留；科学字段已改不意味着旧科学审核被重新批准。

## 当前图与metadata

{n['nodes']:,}节点 / {n['claims']:,}条claim / {n['edges']:,}条边；metadata字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}，不是每条记录携带这么多字段。

claim {cov['claim_fields_unchanged']}项字段覆盖不变；只有direction这一字段新增3个非空值：{cov['direction_nonempty_before']:,}→{cov['direction_nonempty_after']:,}。内层平均{cov['average_nested_fields']:.2f}字段；p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}% / {cov['effect_size_nonempty_pct']:.2f}% / {cov['sample_size_nonempty_pct']:.2f}%，统计覆盖没有补值，也不是100%。

共享{rel['shared_groups']:,}组 / {rel['indexed_claims']:,}条成员；至少两篇已核实且未确认撤稿来源的实际默认查询{len(default)}组，含已确认撤稿时{len(inclusive)}组。新进入共享{len(comp['newly_shared_claim_ids'])}条，离开共享{len(comp['no_longer_shared_claim_ids'])}条；汇聚不是科学共识或独立队列证明。

## 尚未解决

历史449处剩1处（PMID15038994海马聚合不明），本批只关闭原队列中的2处。扩大队列2,206→{queue['expanded_remaining']:,}处 / {queue['expanded_claims']:,}条claim；另外7处科学端点不在该队列，不能重复计为队列消减。旧语义17、结构待核0。待核队列不是确认错误总数，也不穷尽全KG。

科学范围登记6条：BACE1海马聚合/旧审核、ACC旧截断摘录溯源、产前复合客体、ENIGMA复合客体、两条端粒性别分析样本分母。前两项已修正部分当前字段，但历史问题未伪装消失。端粒原N170暂未改，不能用摘要总数90替代性别特定分母。ENIGMA没有丢弃复合客体中的任何成分。

## 验收与保留策略

当前源完整SHA、精确8条公开来源字段规则、全部原记录正反向摘要、所有目标原有关联、全claim/论文/共享成员普查、metadata覆盖、明细/普查库完整SHA及真实读取均通过。规划期曾两次被安全门拦下：一次需要解释R55新接入关联；另一次需要核实科学边历史属性。均先修正检查与测试，未跳过校验或改图后补理由。

仅退役R55四项旧派生文件{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。当前KG和详情仍在R23，本轮派生与证据指针在R57；无旧KG或完整前像回退，当前派生可重建。原始来源未删。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。无模型、KGE、训练、重新抽取。incremental-integrity-checks只用于常规状态，真实写入边界仍全量验证。HTML做静态链接检查，未做浏览器视觉验收。

本次窗口到2026-09-10T02:50Z（香港10:50）；继续审查，不把R57验收宣称为整个KG无问题。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-scientific-definition"><h2>当前R57：8条科学定义与关系表达已有限修正</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图及真实查询通过。原始摘录、统计值和旧审核完整保留；无模型或训练。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>本批变更</td><td>8 claim、9端点、{p['changed_edges']}既有边；新增3共用普通节点，复用2节点；没有新增或删除claim/边</td></tr>
<tr><td>科学字段</td><td>1谓词、3方向、1粗粒度类型及明确测量名称；平均、总和与分别两侧不混并</td></tr>
<tr><td>扩大队列 / 历史449</td><td>{queue['expanded_remaining']:,}处 / {queue['expanded_claims']:,}条claim；历史449还剩1处</td></tr>
<tr><td>旧语义 / 科学登记</td><td>17条 / 6条有限问题与历史溯源记录，不穷尽全KG</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata</td><td>字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；仅direction非空数增加3，统计覆盖不变</td></tr></table>
<p>两条DPD记录已统一关系表达，原claim和不同旧审核保留，来源只计一篇论文。BACE1修正为浓度；ACC为正向观察性关联；未将旧审核当成新科学批准。</p>
<p class="warning">海马聚合不明、复合客体、端粒样本分母及其余身份待核仍存在；不得直接猜数或删去未知成分。本批9处端点中只有2处属于扩大队列。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批结果')} · {j.link(REVIEW/'REPORT.md','原文与边界')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','真实查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','覆盖变化')} · {j.link(OUTPUT/'CURRENT_SCOPE_FINDINGS.json','6项科学登记')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已由R57当前文件替换</span>',page)
    j.atomic_text(j.REPORT,page);qa=publish_section(section,at,current=True)
    j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r57-finite-source-defined-scientific-repair'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=receipt['at'],status='completed',title='R57八条科学定义与关系表达修正',
            body=f"8条claim/9端点/{p['changed_edges']}既有边，复用2、新增3普通节点；原证据/统计/audit保留。",
            changes=f"{receipt['test_counts']['tests']}项回归、独立逆变换及真实查询通过；历史449剩1、扩大{queue['expanded_remaining']}、科学登记6。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);c['latest_source_review']=j.fingerprint(OUTPUT/'REPORT.md');j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,
        acceptance=c['current_acceptance'],runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        coverage=j.fingerprint(cov_path),retention=j.fingerprint(ret_path),report_code=j.fingerprint(Path(__file__)),
        tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_scientific_definition_report.py')))
    print('R57_PUBLISHED',dict(counts=n,relations=rel,default_groups=len(default),queue=queue,retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pending',action='store_true')
    pending() if parser.parse_args().pending else main()
