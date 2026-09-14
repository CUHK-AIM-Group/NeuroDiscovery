"""R49/R50 exact queue accounting, real queries and single-current retention."""
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

OUTPUT = j.OUTPUT / 'round50_literal_scope_reuse'
REVIEW = j.OUTPUT / 'round49_remaining_scoped_literals'
PREVIOUS = j.OUTPUT / 'round48_historic_literal_repair'


def queue_summary(plan, prior, before, after):
    from neurooracle.src.kg_reviewed_literal_reuse import BROAD_REASSIGN
    old={(r['claim_id'],r['side']) for r in before};current={(r['claim_id'],r['side']) for r in after}
    historic={(r['claim_id'],r['side']) for r in prior['endpoint_reviews'].values()}
    prior_fixed={(e['claim_id'],ch['side']) for e in prior['events'] for ch in e['changes']}
    require(len(historic)==447 and prior_fixed<=historic and len(prior_fixed)==425,'R48 historic review differs')
    historic-=prior_fixed
    fixed={(e['claim_id'],ch['side']) for e in plan['events'] if e['claim_id'] not in BROAD_REASSIGN for ch in e['changes']}
    broad={e['claim_id'] for e in plan['events'] if e['claim_id'] in BROAD_REASSIGN}
    require(broad==BROAD_REASSIGN and len(plan['events'])==21,'two nonhistoric reassignments differ')
    require(len(old)==len(before)==plan['expanded_review_endpoints'] and len(current)==len(after)==plan['held_endpoints'],'queue cardinality differs')
    require(len(historic)==22 and historic<=old and fixed<=historic and len(fixed)==plan['original_watchlist_repaired']==19,'finite historic scope differs')
    require(current==old-fixed and len(current&historic)==plan['original_watchlist_remaining_after']==3,'queue subtraction differs')
    return dict(original_watchlist=449,original_watchlist_remaining_before=22,original_watchlist_repaired_this_batch=19,
        original_watchlist_remaining=3,expanded_remaining=len(current),expanded_claims=len({cid for cid,_ in current}),not_all_confirmed_errors=True)


def reviewed_source_report(plan):
    inspection=j.read_json(plan['source_review']['path'])
    require(j.fingerprint(plan['source_review']['path'])==plan['source_review'],'R49 proof changed')
    require(j.fingerprint(plan['public_manifest']['path'])==plan['public_manifest'],'public fetch proof changed')
    text=f'''# R49：现有节点、剩余端点及原始来源的定向复核

只读检查覆盖75条claim、16个原有完整名称节点、36处关联claim端点及214条claim所属边。16节点各有离线atoms依赖（慢性脑损伤节点2条），不能因为暂时无活跃claim就直接删源锚点或详情。R49按R46完整源SHA、普查SHA验收；R48只做不相交的425处端点纠正。R50计划再扫描实际R48全图，逐项核对21条待改claim、全部所属边、36处既有引用、名字/别名/大小写与小鼠蛋白accession候选。R49旧序号未直接拿来改图。

来源刷新按75条claim自带明确PMID获取65篇公共摘要，4篇取得自有全文XML；全文身份只检查front/article-meta，不从参考文献抓PMID/DOI。BACE1及小鼠VCP未取得自有PMC全文。获取不等于科学审查全部通过，未给旧audit、paper registry或scientific conclusion自动升级。

三项题名差异保留：33901490现有题名截断；37111100破折号/句末标点；42300138的PTSD缩写/全称。没有自动写回文献身份。四篇全文属于36716140、36990674、37721751、40448221。相关摘要和特定身份片段已核对，但本报告不声称65篇科学结论或四篇全文正文全部审查完。

R50计划中，19处历史端点和2处宽泛/区域限定脑结构节点混用有精确身份依据；复用13节点，只新建儿童皮层体积与面积完整描述、物种特异小鼠VCP蛋白2节点。既有collybistin交互、菌群、损伤及测量节点的来源metadata不移除、不重新解释为疾病限定。现有7类粗粒度atom角色有意将血液/CSF等biomarker也归入imaging_marker，不能据此断定是生物学影像类型错误。

小鼠VCP依据本篇[PMID15885483](https://pubmed.ncbi.nlm.nih.gov/15885483/)的鼠源细胞和蛋白实验范围，以及[UniProt Q01853](https://www.uniprot.org/uniprotkb/Q01853/entry)。目标使用T116蛋白、Mus musculus物种及accession；不再指向人VCP基因。现有CUI:C1613884虽叫VCP且为蛋白，但本地节点及详情均无自有物种/accession证据，不猜测为同一鼠蛋白。新蛋白的external_ids保留UniProt与NCBI_Taxonomy两个必要身份值，metadata字典为空。

三条待删about分别属于产前暴露、脉络膜巩膜界面厚度、菌群claim；旧认知目标枝与其保留的当前认知枝除目标ID外逐字段完全相同，无附加证据或审核。原概念及离线atoms不删；候选图重新证明完整所属引用，并从保留的中性about重建旧枝摘要，未保留完整前像。

仍未解决：3处双侧海马体积定义；产前复合客体；BACE1测量浓度/通路标签；ACC观察性关系及截断证据；17条旧限定语义。原始方向、限定条件、来源、统计和audit不因本次身份纠正被改写或认为已通过科学复核。

R49检查：{inspection['at']}；R50实际当前源计划：{plan['at']}。正式full_v2未同步；无模型、KGE、训练或重新抽取。
'''
    j.atomic_text(REVIEW/'REPORT.md',text)


def publish_section(section,at=None,current=None):
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="current-reviewed-literal"' in page:
        page=re.sub(r'<section id="current-reviewed-literal">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-historic-literal">'
        require(marker in page,'HTML current anchor missing');page=page.replace(marker,section+'\n'+marker,1)
    page=page.replace('</section>\\n<section id="current-historic-literal">','</section>\n<section id="current-historic-literal">')
    if current:
        page=page.replace('<h2>当前R48：历史完整描述身份已批量纠正</h2>','<h2>R48历史批次：完整描述身份纠正成果保留</h2>')
        page=page.replace('当前已验收R48，详见下方最新结果；同名范围、混合引用和限定语义继续审查，整体KG尚未全部优化完。',
            '当前已验收R50，详见下方最新结果；海马定义和科学限定范围继续审查，整体KG尚未全部优化完。')
        page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R50工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);return qa


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='reviewed_literal_reuse','no active R50 writer')
    require(c['current_graph']==p['graph'] and c['current_acceptance']==p['source_acceptance'],'pending source differs')
    reviewed_source_report(p)
    section=f'''<section id="current-reviewed-literal"><h2>R50构建和独立验收中：现有完整名称复用与物种身份</h2>
<p>1,004项联合回归通过。计划纠正21处端点、32条既有边，复用13节点、新增2节点，精确删除3条没有额外信息的旧about。无新增claim或科学断言；原始科学字段和旧审核不变。</p>
<p class="warning">当前已验收版本仍是R48。历史449队列本批前剩22处，若验收通过将剩3处；扩大待核预计剩5,490处，不代表全部确认错误。产前仅纠正主体身份，不解决复合客体。</p>
<p>小鼠VCP目标为有UniProt与物种身份的蛋白，不是人基因；不合并缺乏物种信息的旧VCP蛋白。已有源锚点和详情均保留。节点/边metadata字段并集不增，小鼠蛋白新增两个必要external_ids身份值。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','当前源计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','1004项联合回归')} · {j.link(REVIEW/'REPORT.md','R49来源与复用边界')}</p></section>'''
    qa=publish_section(section)
    hand=j.OUTPUT/'HANDOFF.md'
    prefix=f'''# 最新：R50正在构建/独立验收，R48仍为已验收图

唯一写进程见CAMPAIGN.json和round50_literal_scope_reuse/RUN_STATE.json。1004项回归通过；禁止并发写或修改冻结FILES、PLAN、TEST_RESULTS.xml。计划21处端点/32条既有边、复用13节点、新增2节点、删除3条旧about。19处属于历史449队列，2处是宽泛/区域限定节点混用；不能混算历史修复数。

成功采用后通过test_kg_reviewed_literal_reuse_report.py保存REPORT_TEST_RESULTS.xml，运行report_kg_reviewed_literal_reuse.py完成实际查询、所有待核哈希和队列差集、覆盖核对；仅精确退役R48四项被替代大型派生文件。当前源位置仍在R23。不改科学结论，不同步full_v2。产前主体身份虽修复，复合客体仍待核。

R49只读复核75条claim、16原节点及36关联端点，65自有PMID摘要/4自有全文XML已获取；科学审查不因获取自动通过。3处海马定义、3科学范围及17旧语义继续，扩大队列不是已确认错误清单。

窗口到2026-09-10T02:50Z（香港10:50），到点不新开写批次，完成必要验收并停用kg-9-10。目标继续，不把R50当全KG全部完成。

---

'''
    previous=hand.read_text(encoding='utf8')
    if not previous.startswith('# 最新：R50'):j.atomic_text(hand,prefix+previous)
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r49-current-scope-source-review'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=p['at'],status='completed',title='R49现有范围与原始来源复核；R50重新绑定当前源',
            body='75条claim、16源节点、36关联端点；65摘要/4全文XML获取，非全部科学验证。',
            changes='计划19历史端点+2宽泛范围重指向；3旧about精确退役，科学与audit不改。',
            artifacts=[str(REVIEW/'REPORT.md'),str(REVIEW/'SOURCE_INSPECTION.json'),str(REVIEW/'PUBLIC_SOURCE_FETCH.json'),str(OUTPUT/'PLAN.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R50_PENDING_PUBLISHED',flush=True)


def main():
    control = j.OUTPUT/'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R50 not current')
    receipt = j.read_json(c['current_acceptance']['path']); state = j.read_json(OUTPUT/'BUILD_STATE.json'); p = j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status'] == 'CURRENT_REVIEWED_LITERAL_REUSE_APPLIED', 'wrong adoption')
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
    queue = queue_summary(p, j.read_json(p['prior_disjoint_identity_plan']['path']), rows(state['baseline']['current_gene_holds']['path']), held)
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
    report=f'''# R50：完整名称复用、小鼠蛋白身份与三条旧分支清理

{at}（香港时间）采用；{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图和真实共享查询通过。

纠正21条claim的21处端点、32条既有边；复用13现有节点，仅新建儿童皮层体积/表面积完整描述和物种特异小鼠VCP蛋白2节点。19处为历史可疑基因端点，另2处是宽泛脑结构描述从区域限定节点重指向同名宽泛节点。删除3条已证实没有附加信息的旧about；没有删除claim、科学断言、原概念或详情，也没有新增claim/科学断言。

小鼠蛋白按自有论文实验范围及UniProt Q01853/NCBI Taxonomy 10090明确为T116蛋白；旧人VCP基因及无物种证据VCP蛋白不改、不强并。新节点metadata为空，external_ids增加两项必要身份值。其它端点只是完整名称身份恢复，不推断新的生物学类型。既有节点metadata、claim原始名称/类型/统计/条件/否定/证据/来源和旧audit不变。

当前{n['nodes']:,}节点、{n['claims']:,}条claim、{n['edges']:,}条边。metadata字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}不变；claim {cov['claim_fields_unchanged']}项字段覆盖逐项相同，内层平均{cov['average_nested_fields']:.2f}项；p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}% / {cov['effect_size_nonempty_pct']:.2f}% / {cov['sample_size_nonempty_pct']:.2f}%，没有补值或声称100%覆盖。

历史449队列：R44处理2处、R48修复425处、R50修复19处，尚余3处海马测量范围待核。扩大队列剩{queue['expanded_remaining']:,}处/{queue['expanded_claims']:,}条claim，不是全部确认错误。产前主体身份纠正不解决复合客体；BACE1测量/通路、ACC观察性与证据截断、17条旧限定语义仍开放。旧源锚点有离线atoms依赖，未据活跃claim数直接删除。

共享关系组{rel['shared_groups']:,}、成员{rel['indexed_claims']:,}；默认至少两篇已核实且未确认撤稿来源{len(default)}组，含确认撤稿时{len(inclusive)}组。新入共享{len(comp['newly_shared_claim_ids'])}条，不再共享{len(comp['no_longer_shared_claim_ids'])}条；不把共享率升高当质量提升或科学共识。

当前源全SHA、独立候选全图、所有保留源节点/边逆向摘要、21条精确身份与所属引用重现、3旧about从保留中性枝重建摘要、全部claim/书目/共享成员普查、metadata覆盖、详情与新普查库完整SHA均通过。实际读取器查询和全部存续待核哈希也通过。没有完整KG历史备份或节点/边前像。

旧R48四项大型派生文件精确退役{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB），当前派生数据可由当前图重建；原始文献和详情未删。工作图仍在R23路径，最新索引/普查/覆盖/待核在R50，terms在R30、paper registry在R34。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。无模型、KGE、训练或重新抽取。incremental-integrity-checks用于常规进度读取，实际修改边界全量验证。HTML完成静态链接检查，未声称浏览器视觉验收。目标继续至2026-09-10T02:50Z（香港10:50）；全KG尚未全部优化完。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-reviewed-literal"><h2>当前R50：完整名称复用、物种身份及旧分支已验收</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图与实际查询通过。纠正21处端点/32条既有边，复用13节点、新建2节点、删除3条无附加信息的旧about。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>历史449队列</td><td>R44处理2、R48修复425、本批19；剩3处海马定义待核</td></tr>
<tr><td>扩大待核 / 旧语义</td><td>{queue['expanded_remaining']:,}处 / 17条，非全部确认错误</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim各字段覆盖不变</td></tr></table>
<p>新增节点metadata为空；小鼠蛋白的UniProt及物种external_ids为必要身份值。既有节点metadata、源锚点和详情未变。所有原始科学字段与旧审核保持，不涉及模型/训练。</p>
<p class="warning">产前仅修复主体身份，复合客体仍开放；BACE1、ACC、海马定义和限定语义继续，不把身份修复说成科学验证。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批完整结果')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','真实查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','逐字段覆盖')} · {j.link(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json','待核口径')} · {j.link(REVIEW/'REPORT.md','R49来源复核')}</p></section>'''
    page = j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative = Path(fp['path']).relative_to(j.REPO).as_posix()
        page = re.sub(rf'<a href="{re.escape(relative)}">.*?</a>', '<span>旧派生资料已由R50当前文件替换</span>', page)
    j.atomic_text(j.REPORT, page); qa = publish_section(section, at, current=True)
    j.atomic_text(j.OUTPUT/'HANDOFF.md', report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log = j.read_json(j.OUTPUT/'WORK_LOG.json'); eid = 'r50-reviewed-complete-name-species-reuse'
    if not any(e.get('id') == eid for e in log['events']):
        log['events'].append(dict(id=eid, at=receipt['at'], status='completed', title='R50现有名称复用与小鼠物种身份、旧分支退役',
            body='21处端点/32既有边；复用13节点、新增2节点，3中性旧about精确退役。',
            changes=f"{receipt['test_counts']['tests']}项回归及全图逆变换、普查、覆盖、真实查询通过；历史449现剩{queue['original_watchlist_remaining']}，科学/audit未改。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json', log)
    require(j.read_json(control) == c, 'campaign advanced'); j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result'] = j.fingerprint(ret_path); c['latest_source_review'] = j.fingerprint(OUTPUT/'REPORT.md'); j.atomic_json(control, c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json', dict(at=j.utc_now(), html=j.fingerprint(j.REPORT), checks=qa,
        acceptance=c['current_acceptance'], runtime=c['current_runtime_acceptance'], queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        coverage=j.fingerprint(cov_path), retention=j.fingerprint(ret_path), report_code=j.fingerprint(Path(__file__)),
        tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'), test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_reviewed_literal_reuse_report.py')))
    print('R50_PUBLISHED', dict(counts=n, relations=rel, default_groups=len(default), queue=queue, retired_bytes=ret['bytes_removed'], checks=qa), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending', action='store_true')
    pending() if parser.parse_args().pending else main()
