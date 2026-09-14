"""R40 current report and actual queries; retire only replaced R39 derivatives."""
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

OUTPUT=j.OUTPUT/'round40_scoped_structure'
PREVIOUS=j.OUTPUT/'round39_literal_duplicates'


def remaining_findings(previous,mri_scope,scope,events):
    """Resolved structural/direction items cannot remain labelled unresolved."""
    hashes=dict(previous['current_claim_hashes'])
    hashes.update({r['claim_id']:r['claim_sha256'] for r in [*mri_scope,*scope]})
    hashes.update({e['claim_id']:e['current_node_sha256'] for e in events})
    findings=[r for r in previous['findings'] if r['claim_id'] not in {'CLM:CASE1MAN:22306803:8568','CLM:ead63272cd396884'}]
    findings.extend([
        dict(claim_id='CLM:ead63272cd396884',classification='broader_pathway_endpoint_still_requires_scope_review',
             detail='The positive/negative field conflict is repaired. The source measured plasma BACE1 concentration; this does not validate the broader pathway label or old scope audit.'),
        dict(claim_id='CLM:CASE1MAN:28526817:9725',classification='symptom_severity_vs_episode_count_requires_source_review',
             detail='The source reports LPFC myelin versus number of depressive episodes, while the current object names symptom severity. Held without endpoint or conclusion changes.'),
        dict(claim_id='CLM:10532161a8b2296ab45e58e9db202527',classification='incomplete_ACC_evidence_span_requires_source_review',
             detail='Current raw text is truncated around statistical values and mentions thickness; it does not independently establish the surface-area assertion.')])
    return dict(current_claim_hashes={r['claim_id']:hashes[r['claim_id']] for r in findings},findings=findings,
        physical_source_anchor_retirement_still_pending=True,
        hippocampal_scope=previous['hippocampal_scope'],issue_register_is_exhaustive=False,queue_counts_may_overlap=True)


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['active_process'] and c['active_process']['state']==str(OUTPUT/'RUN_STATE.json'),'not current R40 writer')
    plan=j.read_json(OUTPUT/'PLAN.json');require(plan['graph']==c['current_graph'],'source already advanced')
    section=f'''<section id="scoped-structure-pending"><h2>R40 正在构建与整体验收</h2>
<p>658项联合回归通过。计划修复7条claim、7处基因误指向与9条既有边引用，补1条中性的about结构引用；仅1处有来源的方向字段修正。新增7个完整表达节点，不按文章建点，不新增metadata字段。</p>
<p>当前仍以已验收R39为准。五条MRI来源支持身份修复；qMRI发作次数/症状严重程度、同名节点范围问题继续保留。不涉及模型、训练或正式同步，夜间定时接续没有重建。</p>
<p>{j.link(OUTPUT/'PLAN.json','完整限定计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','658项测试')} · {j.link(OUTPUT/'ENDPOINT_TRIAGE.jsonl','456处端点的源版本分层')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="scoped-structure-pending"' in page:page=re.sub(r'<section id="scoped-structure-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="night-window-20260909">';require(marker in page,'report anchor missing')
        page=page.replace(marker,section+'\n'+marker,1)
    check_html(page);j.atomic_text(j.REPORT,page)
    print('R40_PENDING_REPORT_UPDATED',flush=True)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R40 not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json');plan=j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status']=='CURRENT_SCOPED_STRUCTURE_REPAIR_APPLIED','wrong receipt')
    for fp in [c['current_runtime_acceptance'],c['current_gene_holds'],c['current_coverage'],c['current_paper_census'],
        receipt['tests'],receipt['provenance_plan'],*receipt['code']]:require(j.fingerprint(fp['path'])==fp,'frozen current evidence differs')
    runtime=j.read_json(c['current_runtime_acceptance']['path']);require(runtime['acceptance']==c['current_acceptance'],'runtime binding differs')
    tests=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(tests.get('tests'))==2 and all(int(tests.get(k))==0 for k in ('failures','errors','skipped')),'report tests failed')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    coverage={r['field']:r for r in rows(c['current_coverage']['path']) if r['scope']=='node/claim'}
    require(all(r['denominator']==c['counts']['claims'] for r in coverage.values()),'coverage denominator differs')
    sparse_coverage='、'.join(f"{label} {coverage[field]['nonempty_pct']:.2f}%" for label,field in (
        ('p值','metadata.evidence.p_value'),('效应量','metadata.evidence.effect_size'),('样本量','metadata.evidence.sample_size')))
    deferred=remaining_findings(j.read_json(PREVIOUS/'DEFERRED_SCOPE_FINDINGS.json'),rows(OUTPUT/'MRI_CLAIM_SCOPE.jsonl'),rows(OUTPUT/'CLAIM_SCOPE.jsonl'),plan['events'])
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,h in {**deferred['current_claim_hashes'],**{e['claim_id']:e['current_node_sha256'] for e in plan['events']}}.items():
        require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(h,),'current claim hash differs')
    db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2));inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'current query count differs')
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        shared_groups=len(groups),default_two_verified_sources=len(default),including_retracted=len(inclusive),
        repaired_claims_checked_in_current_census=len(plan['events']),no_independent_experiment_or_consensus_inferred=True))
    comp_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        fp=state['baseline']['current_shared_relations'];require(j.fingerprint(fp['path'])==fp,'previous shared index changed')
        j.atomic_json(comp_path,dict(graph=c['current_graph'],at=j.utc_now(),**comparison(groups,rows(fp['path']))))
    comp=j.read_json(comp_path);require(comp['graph']==c['current_graph'],'comparison not current')
    j.atomic_json(OUTPUT/'CURRENT_SCOPE_FINDINGS.json',dict(at=j.utc_now(),graph=c['current_graph'],**deferred,
        resolved=dict(collapsed_double_VCP_endpoint='CLM:CASE1MAN:22306803:8568',direction_conflict='CLM:ead63272cd396884'),
        public_sources=[j.fingerprint(OUTPUT/'PUBLIC_SOURCE_FETCH.json'),j.fingerprint(OUTPUT/'MRI_SOURCE_FETCH.json')]))
    ret_path=OUTPUT/'RETENTION_RESULT.json'
    if not ret_path.exists():
        targets=[j.read_json(state['baseline']['current_paper_census']['path'])['database'],
            *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current={Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths=validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)
        for fp,path in zip(targets,paths):j.guards(fp);require(sha256(path)==fp['sha256'],'replaced derivative changed')
        require(j.read_json(control)==c,'current batch advanced')
        for path in paths:path.unlink()
        j.atomic_json(ret_path,dict(status='COMPLETED',at=j.utc_now(),removed=targets,bytes_removed=sum(fp['bytes'] for fp in targets),
            graph_backups=0,record_preimages_saved=False,original_archives_and_public_evidence_deleted=False,
            recoverability='No historical copies retained. Current derivatives are rebuildable from the current KG; not a historical rollback.'))
    ret=j.read_json(ret_path);at=j.local_time(receipt['at']);counts=c['counts'];rel=c['relation_evidence_counts']
    report=f'''# R40：结构闭合、明确MRI端点与方向字段

{at}（香港时间）已通过整体验收并采用到当前工作KG。658项联合回归、2项报告测试通过。没有模型、KGE、训练、重新抽取、正式full_v2同步或自动化重建。

## 一批完成的修改

- 7条claim，修复7处错误基因端点；修改9条既有边引用，新增1条about结构引用。新增7个按完整名称复用的普通节点，不按论文/claim编号建点。没有新增metadata字段，也没有新增科学断言边。
- 原CLM:CASE1MAN:22306803:8568的影像指标及COMT×外化行为交互均误指VCP。现恢复两个完整端点和第二条about。只有中性结构引用可从本条既有about派生，不复制带独立证据或未知角色的边。[本篇来源](https://pubmed.ncbi.nlm.nih.gov/22306803/)。
- 五条明确MRI指标：双侧丘脑枕高信号、对侧杏仁核T2信号、愉快词DLPFC功能侧化、颈胸髓后柱/后外侧柱T2信号、额颞扩散MRI侧化指标。每篇来源均核实，完整复合表达/侧别保留；混合符号和不同测量没有归成统一方向。
- CLM:ead63272cd396884的evidence.direction由positive改为negative，与原文及本篇摘要一致；不翻转negated，不填样本量/p值。[本篇来源](https://pubmed.ncbi.nlm.nih.gov/36847009/)。这只修复方向字段，不等于已确认其“pathway”端点范围和旧scope审核全部正确。

## 当前规模与关系复用

节点{counts['nodes']:,}，claim{counts['claims']:,}，边{counts['edges']:,}；metadata键并集节点{c['node_metadata_field_union']}、边{c['edge_metadata_field_union']}。共享关系组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}；默认至少两个已核实来源的组{len(default)}，含确认撤稿{len(inclusive)}。来源数不是独立实验或共识。成员变化见RELATION_MEMBERSHIP_COMPARISON.json，不把修复错误身份夸成新增跨论文证据。

metadata并非全部高覆盖。以全部{counts['claims']:,}条claim为分母，非空率为：{sparse_coverage}；原文片段{coverage['metadata.raw_text']['nonempty_pct']:.2f}%。这些是非空统计，不代表数值有效或实验可用。下一批可审查读取依赖后集中精简空模板和重复表达，但必须保留确有内容的科学证据，不能因字段稀疏就整类删除。

## 一并检查但未强行修改

源版本全部456处端点重新分层：294处表达未满足旧测量门槛、136处未明确影像类型、15处分子/遗传语境，其他11处为候选类型、同名、方法或引用闭合问题。分类是待核原因，不是错误结论。修复后剩余{c['pending_gene_link_endpoint_events']}处，涉及{c['pending_gene_link_claims']}条claim。

三个同名候选节点带treatment_outcome类型，原关联claim还存在未明确类型/测量范围问题；其中ACC原文片段截断，未据此强并。qMRI本篇报告LPFC髓鞘与抑郁发作次数，当前端点却写症状严重程度，保留本篇科学范围复审。[qMRI来源](https://pubmed.ncbi.nlm.nih.gov/28526817/)。

R39两个来源锚点仍被明细库依赖，未物理删除。海马双侧合计/分别测量、产前暴露复合端点、frailty等待核保留。旧21语义/1结构与新增范围登记可能重叠，也不穷尽全图。

## 验收与保留

完整源KG SHA、独立候选全图、所有原节点/边逆向摘要、缺失about的同条来源重建、当前claim/论文/共享成员普查、全量metadata覆盖、书目/实体/出版见证、明细库和当前普查完整SHA通过。658项回归和真实查询通过；不是仅以测试替代数据校验。

旧R39四项派生文件共{ret['bytes_removed']:,}字节已删除，无旧KG或记录前像。当前KG、明细、索引与公开/原始证据保留；删除的旧派生资料无回退副本，当前版本派生资料可重建。

初次未应用候选因一个交互节点的域标签不能被现有解析器识别而停止。原工作图完整SHA确认未变后移除未应用产物，改用现有dataset_variable域，并新增14项实际解析/保存重载测试，确认完整名称复用不再建点。随后重建并独立验收；没有为让测试通过而改变现有类型映射。

当前KG SHA-256：{c['current_graph']['sha256']}。
使用incremental-integrity-checks：普通进度复用可信状态，本次构建/采用边界完整校验。原夜间窗口继续保持关闭，整体优化目标尚未完成。
'''
    next_plan='''# R41 接续

以CAMPAIGN.json为准；当前R40数据与运行时均已验收，工作KG仍在R23路径，当前census/index/coverage/issues在R40，terms在R30、paper registry在R34。R39四项旧派生文件已移除，不重跑历史构建。只有本轮当前KG，无记录前像。

优先按CURRENT_SCOPE_FINDINGS.json处理本篇证据与端点范围：qMRI发作次数与症状严重程度、BACE1浓度与pathway、ACC截断证据。剩余449处基因端点不要凭类型缺失或宽泛词就全部删除。三组同名测量的节点类型/左右范围与R39两个来源锚点的明细依赖，要联合核实后才能归并或物理精简。

旧21语义/1结构、海马测量范围、产前暴露复合端点和书目冲突仍在；清单非穷尽、可能重叠。后续重点回到关系复用与重复字段的安全清理，不以单篇claim或共享率作为删除依据。不运行模型/训练，不同步正式full_v2，不重建夜间自动化。

当前claim的p值、效应量、样本量非空率分别约0.62%、1.51%、5.45%。审查消费者和序列化约定后，优先批量精简空模板、重复字段与过时流程信息；保留真实科学值。低覆盖本身不构成整类删除依据，合并前先确认不同条件和测量范围。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report);j.atomic_text(OUTPUT/'NEXT_BATCH_PLAN.md',next_plan)
    section=f'''<section id="current-scoped-structure"><h2>当前 R40：结构闭合、MRI端点和方向字段已修复</h2>
<div class="notice">658项回归、2项报告测试、独立全图与实际查询通过。7条claim、7处基因指向、9条既有边引用已修复；补1条about，新增7个完整表达节点，metadata字段新增0。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}</td></tr>
<tr><td>metadata键并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}</td></tr><tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认多已核实来源组{len(default)}</td></tr>
<tr><td>claim科学字段非空率</td><td>{sparse_coverage}；有值不等于已科学核实</td></tr>
<tr><td>剩余端点待核</td><td>{c['pending_gene_link_endpoint_events']}处，{c['pending_gene_link_claims']}条claim</td></tr></table>
<p>双VCP端点恢复为完整影像指标和COMT×外化行为交互；五条MRI指标不再误指基因。BACE1方向字段改为负向，negated、原文及条件不变。没有新增科学断言或模型训练。</p>
<p class="warning">同名节点、qMRI发作次数/症状严重程度、BACE1路径范围、ACC截断证据和明细依赖继续待核。实体误指向修复不等于跨论文关系归并完成，原夜间接续保持关闭。</p>
<p>{j.link(OUTPUT/'REPORT.md','完整说明与来源')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前数据验收')} · {j.link(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json','当前运行时')} · {j.link(OUTPUT/'PLAN.json','限定修复计划')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','实际查询')} · {j.link(OUTPUT/'CURRENT_SCOPE_FINDINGS.json','当前范围待核')} · {j.link(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl','当前449处端点')} · {j.link(OUTPUT/'CURRENT_METADATA_COVERAGE.jsonl','当前覆盖')} · {j.link(OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json','共享成员变化')} · {j.link(OUTPUT/'NEXT_BATCH_PLAN.md','下一批接续')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    page=re.sub(r'<section id="scoped-structure-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    require('id="current-scoped-structure"' in page,'pending report section missing')
    page=page.replace('<h2>夜间窗口已结束：当前 R39，整体优化尚未完成</h2>','<h2>历史：9月9日夜间窗口交接（R39）</h2>')
    page=re.sub(r'(<section id="current-clinical-reuse">.*?</section>)',r'<details><summary>R39 历史批次：完整临床指标引用</summary>\1</details>',page,count=1,flags=re.S)
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已替换，仅保留R40当前文件</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R40工作图，正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n'+next_plan)
    log_path=j.OUTPUT/'WORK_LOG.json';log=j.read_json(log_path)
    event_id='r40-scoped-structure'
    if not any(e.get('id')==event_id for e in log['events']):
        log['events'].append(dict(id=event_id,at=receipt['at'],status='completed',title='R40结构闭合、五条MRI端点与方向修复',
            body='7claim、7端点、9既有边，补1about；658+2测试及独立全图通过。',
            changes='456源端点已分层，449待核；无模型训练或正式同步，旧派生资料删除，夜间窗口不重建。',
            artifacts=[str(OUTPUT/n) for n in ('REPORT.md','CURRENT_ACCEPTANCE.json','CURRENT_SCOPE_FINDINGS.json','NEXT_BATCH_PLAN.md')]))
        j.atomic_json(log_path,log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);c['current_scope_findings']=j.fingerprint(OUTPUT/'CURRENT_SCOPE_FINDINGS.json');j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),retention=j.fingerprint(ret_path),
        report_code=j.fingerprint(Path(__file__)),report_tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),
        report_test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_scoped_structure_report.py'),
        current_scope_findings=c['current_scope_findings']))
    print('R40_PUBLISHED',dict(counts=counts,relations=rel,default_groups=len(default),retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pending',action='store_true')
    pending() if p.parse_args().pending else main()
