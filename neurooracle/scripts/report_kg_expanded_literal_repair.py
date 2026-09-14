"""R52/R53 exact queue accounting, real queries and single-current retention."""
from pathlib import Path
import re
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from report_current_kg import check_html
from report_kg_gene_boundary_repair import coverage_summary
from report_kg_literal_endpoint_repair import comparison
from report_kg_measurement_reuse import validate_retirement_paths
from report_kg_source_scope_resolution import collect_hold_hashes
from neurooracle.src.shared_relation_catalog import current_shared_relations, find_shared_relations

OUTPUT = j.OUTPUT / 'round53_expanded_literal_repair'
REVIEW = j.OUTPUT / 'round52_expanded_literal_review'
PREVIOUS = j.OUTPUT / 'round50_literal_scope_reuse'


HISTORIC_THREE={('CLM:CASE1MAN:15038994:11676','subject'),('CLM:CASE1MAN:22504417:9443','subject'),('CLM:e228496be8a5','subject')}


def queue_summary(plan,before,after):
    old={(r['claim_id'],r['side']) for r in before};current={(r['claim_id'],r['side']) for r in after}
    fixed={(e['claim_id'],ch['side']) for e in plan['events'] for ch in e['changes']}
    require(len(old)==len(before)==plan['old_watchlist_endpoints'],'old queue cardinality differs')
    require(len(current)==len(after)==plan['held_endpoints'],'current queue cardinality differs')
    require(len(fixed)==plan['changed_endpoints'] and fixed<=old and current==old-fixed,'exact queue subtraction differs')
    require(HISTORIC_THREE<=old and HISTORIC_THREE<=current and not fixed&HISTORIC_THREE
        and plan['original_watchlist_449_remaining_after']==3 and plan['original_watchlist_449_repaired']==0,'historic three must not be silently closed')
    return dict(original_watchlist=449,original_watchlist_remaining_before=3,original_watchlist_repaired_this_batch=0,
        original_watchlist_remaining=3,expanded_repaired_this_batch=len(fixed),expanded_remaining=len(current),
        expanded_claims=len({cid for cid,_ in current}),not_all_confirmed_errors=True)


def reviewed_source_report(plan):
    inspection=j.read_json(plan['source_review']['path']);classification=j.read_json(plan['classification']['path'])
    require(j.fingerprint(plan['source_review']['path'])==plan['source_review'] and j.fingerprint(plan['classification']['path'])==plan['classification'],'R52 proof changed')
    text=f'''# R52：扩大待核队列的完整名称、源基因与所属引用审查

在R50构建期间，以仍然已验收的R48为只读源，完整SHA扫描5490处队列端点、5005条claim、213个当前基因，发现2465个已有同名/别名/大小写候选节点并检查5037处既有CLM_CONCEPT关联端点。保存的是必要投影和摘要，没有完整记录前像。

5298处通过字内别名及所属引用的初始结构门槛；这不是5298条科学结论验证，也不代表都能自动改。之后依原始声明类型、完整词面和排除规则，分离现有同名候选、分子/基因/药物范围、未声明类型、混合引用及不在本批非分子词面范围的情况。R50改过但仍有另一端待核的3条claim也排除，未使用旧claim摘要直接改新图。

有限候选{classification['candidate_endpoints']}处/{classification['candidate_claims']}条claim，完整名称{classification['candidate_names']}个；原始类型分别为1995 outcome、862 imaging_marker、102 disease、18 individual_data、2 cognitive_task。这些仍是原有来源类别，不是新推断的本体类型。例如MATT落在matter内部、MENT落在improvement内部，这些整段临床/测量名称不是对应的单独基因。

R53又在实际当前R50图上完整扫描所有候选claim、213基因、名字/别名/大小写及当前边，核对R52所属边摘要，按R50已删3边后的序号重新定位。检查同批大小写变体和生成ID冲突；现有同名候选绝不通过再创建一个节点绕过。最终计划{plan['changed_endpoints']}处端点、{plan['changed_edges']}条既有边、{plan['added_literal_nodes']}个按完整名称共用的普通概念，无复用未经范围确认的旧节点。

完整名称、左右侧、群体、限定词、类型、否定、统计、证据、来源和旧audit不变。新节点无metadata、别名、external_ids或语义类型，不将描述自动变成具体分子或本体实体。不增加claim/科学断言，也不删除claim/边/源锚点/详情。

历史449仍有3处双侧海马定义待核；本批修复只计入扩大队列，预计剩{plan['held_endpoints']}处。旧17语义和产前/BACE1/ACC科学范围继续；未解决清单可重叠，不穷尽全KG。

R52检查：{inspection['at']}；R53当前源计划：{plan['at']}。无模型、KGE、训练或重新抽取；正式full_v2未同步。
'''
    j.atomic_text(REVIEW/'REPORT.md',text)


def scientific_source_note():
    folder=j.OUTPUT/'round51_scientific_scope_review'
    manifest=j.read_json(folder/'PUBLIC_SOURCE_FETCH.json')
    old=j.read_json(j.OUTPUT/'round49_remaining_scoped_literals/PUBLIC_SOURCE_FETCH.json')
    sources=[r['response'] for r in manifest['fulltexts'] if r['status']=='PUBLIC_FULLTEXT_FETCHED']
    sources.extend(r['response'] for r in old['fulltexts'] if r['pmid'] in {'36716140','37721751'} and r['status']=='PUBLIC_FULLTEXT_FETCHED')
    require(len(sources)==4,'reviewed fulltext evidence incomplete')
    for fp in [manifest['prior_public_manifest'],manifest['original_abstract_response'],*sources]:
        require(j.fingerprint(fp['path'])==fp,'public scientific evidence changed')
    j.atomic_json(folder/'SCIENTIFIC_REVIEW_RECEIPT.json',dict(at=j.utc_now(),status='SOURCE_REVIEW_NOT_KG_MODIFICATION',
        report=j.fingerprint(folder/'REPORT.md'),public_fetch=j.fingerprint(folder/'PUBLIC_SOURCE_FETCH.json'),sources=sources,
        reviewed_body_sections={'22504417':['body','online methods'],'35145436':['Materials and Methods','Results','Discussion'],
            '37721751':['Methods','Results','Discussion and Limitations','Conclusions'],'36716140':['RESULTS','DISCUSSION','Conclusion']},
        current_claims_not_revalidated=True,old_audits_not_upgraded=True,new_findings_not_added_to_current_issue_register=True,
        graph_modified=False,record_preimages_saved=False))
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r51-primary-measurement-and-scientific-scope-review'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=j.utc_now(),status='completed',title='R51海马定义与科学范围原文复核（未改图）',
            body='区分左右平均、分别报告与未明聚合；ACC有正向表面积关联但不能推断因果；产前暴露分层和BACE1测量仍待核。',
            changes='2份新增全文获取；4篇选定正文部分复核。新增两条DPD重复线索和端粒样本量分母线索，尚未正式改值或加入当前问题清单。',
            artifacts=[str(folder/'REPORT.md'),str(folder/'SCIENTIFIC_REVIEW_RECEIPT.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)


def publish_section(section,at=None,current=None):
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="current-expanded-literal"' in page:
        page=re.sub(r'<section id="current-expanded-literal">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-reviewed-literal">'
        require(marker in page,'HTML current anchor missing');page=page.replace(marker,section+'\n'+marker,1)
    if current:
        page=page.replace('<h2>当前R50：完整名称复用、物种身份及旧分支已验收</h2>','<h2>R50历史批次：现有名称复用与旧分支清理成果保留</h2>')
        page=page.replace('当前已验收R50，详见下方最新结果；海马定义和科学限定范围继续审查，整体KG尚未全部优化完。',
            '当前已验收R53，详见下方最新结果；同名身份及科学范围继续审查，整体KG尚未全部优化完。')
        page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R53工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);return qa


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='expanded_literal_repair','no R53 writer')
    require(c['current_graph']==p['graph'] and c['current_acceptance']==p['source_acceptance'],'pending source differs')
    reviewed_source_report(p)
    scientific_source_note()
    section=f'''<section id="current-expanded-literal"><h2>R53构建/独立验收中：扩大队列的非分子完整描述身份纠正</h2>
<p>1,073项联合回归通过。计划纠正{p['changed_claims']:,}条claim、{p['changed_endpoints']:,}处端点及{p['changed_edges']:,}条既有边；{p['added_literal_nodes']:,}个完整名称共用普通概念，不按论文重复创建。无新增claim/科学边，无删除claim/边或原始科学字段修改。</p>
<p class="warning">当前已验收版本仍为R50，计划数尚非结果。扩大队列本批前5,490处，预计剩{p['held_endpoints']:,}处；历史449仍剩3处海马定义，本批不混记历史修复数。</p>
<p>分子、基因、药物、未声明类型、现有同名/别名/大小写及混合引用继续待核。新概念metadata/aliases/external_ids/semantic_types均为空；不是新增生物学推断。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','当前源计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','1073项联合回归')} · {j.link(REVIEW/'REPORT.md','R52完整复核边界')} · {j.link(j.OUTPUT/'round51_scientific_scope_review/REPORT.md','R51海马与科学范围复核：未改图')}</p></section>'''
    qa=publish_section(section)
    hand=j.OUTPUT/'HANDOFF.md';header=f'''# 最新：R53正在构建/独立验收，R50仍是当前已验收图

唯一KG写进程见CAMPAIGN.json和round53_expanded_literal_repair/RUN_STATE.json。1073项回归已通过；不要修改冻结FILES、PLAN、TEST_RESULTS.xml，不启动另一个KG写进程。计划2839条claim/2979端点/4308既有边/2751完整名称普通概念；无claim/边删除，无科学内容或旧audit修改。

成功采用后，先运行test_kg_expanded_literal_report.py并保存REPORT_TEST_RESULTS.xml，再运行report_kg_expanded_literal_repair.py完成真实查询、全待核哈希、5490队列差集、各claim字段覆盖及R50四项旧大型派生文件退役。source KG仍在R23路径，不备份旧KG或完整记录前像。

R52完整SHA审查5490端点/5005claim/213基因，2465既有名称候选、5037CLM_CONCEPT引用。5298过初始结构/字面门槛不是科学验证。2979非分子候选在实际R50全图重新绑定，源基因与完整词面/声明类型/引用全核；同名/别名、分子、药物、未知类型和混合分支不强并。

R50已完成21端点/32既有边、复用13新增2、精确删3旧about。当前2591353节点/905179claim/3011990边，metadata并集120/36，共享2587组5868成员，默认跨核实未撤稿来源519组。历史449剩3，扩大队列5490；R53预计扩大剩2511。R50报告已发布，R48旧四派生已退役766623822B。

R51已有2份新增海马全文XML。22504417明确mean bilateral，35145436分别left/right；15038994只有摘要未确认聚合。ACC全文明确观察性且不能因果推断，表面积方向确为正；产前全文区分暴露时期和家长教育层；BACE1仍是浓度测量而非通路。细节暂未更改，R51报告尚待整理。

窗口到2026-09-10T02:50Z（香港10:50）；到点不新开写批次，完成必要当前验收并停用kg-9-10。目标仍继续，不把本批当全KG已完成。无模型、KGE、训练或重新抽取，不同步full_v2。

---

'''
    previous=hand.read_text(encoding='utf8')
    previous=previous.replace('细节暂未更改，R51报告尚待整理。','细节暂未更改；R51/REPORT.md及SCIENTIFIC_REVIEW_RECEIPT.json已整理，未写回KG。')
    if not previous.startswith('# 最新：R53'):j.atomic_text(hand,header+previous)
    else:j.atomic_text(hand,previous)
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r52-expanded-current-source-identity-review'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=p['at'],status='completed',title='R52扩大队列完整源复核；R53最新图再次核对',
            body='5490端点/5005claim/213源基因；分类与测试完成，不等于科学结论验证。',
            changes='计划2979非分子端点；存在同名/别名、分子、药物、未声明类型和混合引用保持待核。',
            artifacts=[str(REVIEW/'REPORT.md'),str(REVIEW/'SOURCE_INSPECTION.json'),str(REVIEW/'CANDIDATE_CLASSIFICATION.json'),str(OUTPUT/'PLAN.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R53_PENDING_PUBLISHED',flush=True)


def main():
    control = j.OUTPUT/'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R53 not current')
    receipt = j.read_json(c['current_acceptance']['path']); state = j.read_json(OUTPUT/'BUILD_STATE.json'); p = j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status'] == 'CURRENT_EXPANDED_LITERAL_REPAIR_APPLIED', 'wrong adoption')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_scope_findings','current_issues','current_structure_holds',
        'current_coverage','current_paper_census','current_paper_issues','current_census_normalization','current_gene_review_summary')],
        receipt['tests'],receipt['provenance_plan'],receipt['identity_audit'],*receipt['code']]: require(j.fingerprint(fp['path']) == fp, 'current frozen proof changed')
    require(j.read_json(c['current_runtime_acceptance']['path'])['acceptance'] == c['current_acceptance'], 'runtime binding differs')
    suite = ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests')) >= 6 and all(int(suite.get(k)) == 0 for k in ('failures','errors','skipped')), 'report tests incomplete')
    census = j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    held = rows(c['current_gene_holds']['path']); issues = rows(c['current_issues']['path']); structure = rows(c['current_structure_holds']['path'])
    findings = j.read_json(c['current_scope_findings']['path'])
    hashes = collect_hold_hashes([*held,*issues,*structure], findings['current_claim_hashes'])
    db = sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro', uri=True)
    try:
        for cid, h in hashes.items(): require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone() == (h,), 'held claim/census differs')
        for event in p['events']:
            require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(event['claim_id'],)).fetchone() == (event['current_node_sha256'],event['new_relation_id']), 'repaired claim/census differs')
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
    ret = j.read_json(ret_path); at = j.local_time(receipt['at']); n = c['counts']; rel = c['relation_evidence_counts']
    reviewed_source_report(p)
    scientific_source_note()
    report=f'''# R53：扩大队列非分子完整描述身份批量纠正

{at}（香港时间）已采用；{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图和真实共享查询通过。

修复{p['changed_claims']:,}条claim的{p['changed_endpoints']:,}处端点及{p['changed_edges']:,}条既有边，使用{p['added_literal_nodes']:,}个按完整名称共用的普通概念节点。所有这些描述此前只因字内别名而指向单个基因。新ID不按论文或claim加盐，完整词面和来源声明类型保持；本批没有复用任何未确认范围的同名节点。

当前{n['nodes']:,}节点、{n['claims']:,}条claim、{n['edges']:,}条边。节点增加是将错误挤在基因上的不同描述分开，不是新增论文/claim/科学结论。没有删除claim/边/原概念/详情。新节点metadata、aliases、external_ids、semantic_types均为空，不推断新的本体或分子身份。既有节点和所有非目标记录保持不变。

metadata字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}不变；claim {cov['claim_fields_unchanged']}项覆盖逐项相同，内层平均{cov['average_nested_fields']:.2f}字段。p值/效应量/样本量非空率仍为{cov['p_value_nonempty_pct']:.2f}% / {cov['effect_size_nonempty_pct']:.2f}% / {cov['sample_size_nonempty_pct']:.2f}%，不补值、不声称全覆盖。原始名称、类型、统计、否定、条件、引文、来源和旧audit完全保留。

扩大队列从5,490处移除本批{queue['expanded_repaired_this_batch']:,}处已纠正端点，剩{queue['expanded_remaining']:,}处/{queue['expanded_claims']:,}条claim。历史449仍有3处海马测量定义，未被本批混算或关闭。现有同名/别名、分子/药物、未声明类型及混合引用待核；17旧语义、产前/BACE1/ACC科学范围保持开放，不把身份纠正当作科学结论验证。

共享关系组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}；至少两篇已核实且未确认撤稿来源默认{len(default)}组，含已确认撤稿时{len(inclusive)}组。新入共享索引{len(comp['newly_shared_claim_ids'])}条，不再共享{len(comp['no_longer_shared_claim_ids'])}条；分组不等于科学共识、独立队列或研究质量。

R52旧投影只作证据，R53在实际R50完整图SHA边界重新检查候选原记录、213源基因、全部名字/别名/大小写和当前所属边。独立候选全图逆变换复现全部源节点/边摘要，科学字段与旧audit未变；全部claim/书目/共享成员普查、metadata覆盖、明细和普查库完整SHA、实际查询及所有存续待核claim哈希均通过。

精确退役R50四项被替代大型派生文件{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。只有当前工作KG，不保留旧KG或完整记录前像；当前派生数据可重建。原始文献/详情未删除。工作KG仍在R23路径，最新普查/索引/待核/覆盖在R53，terms在R30、paper registry在R34。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。无模型、KGE、训练或重新抽取。incremental-integrity-checks用于常规进度读取，实际修改边界全量验证。HTML完成静态链接检查，未做浏览器视觉验收。目标继续到2026-09-10T02:50Z（香港10:50），整体KG尚未全部优化完。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-expanded-literal"><h2>当前R53：扩大队列完整描述身份已批量纠正</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图和实际查询通过。{p['changed_endpoints']:,}处端点/{p['changed_edges']:,}条既有边已纠正，科学字段和旧审核不变。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>完整名称节点</td><td>{p['added_literal_nodes']:,}个共享普通概念，不按论文重复造节点；无新增科学断言</td></tr>
<tr><td>扩大队列</td><td>5,490 → {queue['expanded_remaining']:,}处；不是全部确认错误</td></tr>
<tr><td>历史449 / 旧语义</td><td>仍剩3处海马定义 / 17条限定语义</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨已核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim各字段覆盖不变</td></tr></table>
<p>新节点metadata/aliases/external_ids/semantic_types均为空。节点增加用于分离错误基因身份，不新增claim/科学结论。原概念、详情与原始文献保持。</p>
<p class="warning">分子、药物、未声明类型、同名/别名/大小写及混合引用继续待核；产前/BACE1/ACC和海马定义未自动关闭。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批完整结果')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','真实查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','逐字段覆盖')} · {j.link(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json','待核口径')} · {j.link(REVIEW/'REPORT.md','R52复核边界')} · {j.link(j.OUTPUT/'round51_scientific_scope_review/REPORT.md','R51科学范围复核：未改图')}</p></section>'''
    page = j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative = Path(fp['path']).relative_to(j.REPO).as_posix()
        page = re.sub(rf'<a href="{re.escape(relative)}">.*?</a>', '<span>旧派生资料已由R53当前文件替换</span>', page)
    j.atomic_text(j.REPORT, page); qa = publish_section(section, at, current=True)
    j.atomic_text(j.OUTPUT/'HANDOFF.md', report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log = j.read_json(j.OUTPUT/'WORK_LOG.json'); eid = 'r53-finite-expanded-literal-repair'
    if not any(e.get('id') == eid for e in log['events']):
        log['events'].append(dict(id=eid, at=receipt['at'], status='completed', title='R53扩大队列非分子完整描述身份批量纠正',
            body=f"{p['changed_endpoints']}处端点/{p['changed_edges']}既有边；复用{p['reused_existing_nodes']}，补齐{p['added_literal_nodes']}共用普通节点。",
            changes=f"{receipt['test_counts']['tests']}项回归及全图逆变换、普查、覆盖、真实查询通过；历史449现剩{queue['original_watchlist_remaining']}，科学/audit未改。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json', log)
    require(j.read_json(control) == c, 'campaign advanced'); j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result'] = j.fingerprint(ret_path); c['latest_source_review'] = j.fingerprint(OUTPUT/'REPORT.md'); j.atomic_json(control, c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json', dict(at=j.utc_now(), html=j.fingerprint(j.REPORT), checks=qa,
        acceptance=c['current_acceptance'], runtime=c['current_runtime_acceptance'], queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        coverage=j.fingerprint(cov_path), retention=j.fingerprint(ret_path), report_code=j.fingerprint(Path(__file__)),
        tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'), test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_expanded_literal_report.py')))
    print('R53_PUBLISHED', dict(counts=n, relations=rel, default_groups=len(default), queue=queue, retired_bytes=ret['bytes_removed'], checks=qa), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending', action='store_true')
    pending() if parser.parse_args().pending else main()
