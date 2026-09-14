"""R47/R48 exact queue accounting, real queries and single-current retention."""
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

OUTPUT = j.OUTPUT / 'round48_historic_literal_repair'
REVIEW = j.OUTPUT / 'round47_historic_mentions'
PREVIOUS = j.OUTPUT / 'round46_research_retirement'


def queue_summary(plan, before, after):
    old = {(r['claim_id'], r['side']) for r in before}
    current = {(r['claim_id'], r['side']) for r in after}
    historic = {(r['claim_id'], r['side']) for r in plan['endpoint_reviews'].values()}
    fixed = {(e['claim_id'], ch['side']) for e in plan['events'] for ch in e['changes']}
    require(len(old) == len(before) == plan['old_watchlist_endpoints'], 'old queue cardinality differs')
    require(len(current) == len(after) == plan['held_endpoints'], 'current queue cardinality differs')
    require(len(historic) == 447 and historic <= old and fixed <= historic, 'finite historic scope differs')
    require(len(fixed) == plan['changed_endpoints'] and current == old-fixed, 'queue subtraction differs')
    require(len(current & historic) == plan['original_watchlist_449_remaining_after'], 'historic remainder differs')
    return dict(original_watchlist=449, original_watchlist_remaining_before=447,
        original_watchlist_repaired_this_batch=len(fixed), original_watchlist_remaining=len(current & historic),
        expanded_remaining=len(current), expanded_claims=len({cid for cid, _ in current}), not_all_confirmed_errors=True)


def reviewed_source_report(plan):
    inspection = j.read_json(plan['source_review']['path'])
    require(j.fingerprint(plan['source_review']['path']) == plan['source_review'], 'R47 proof changed')
    text = f'''# R47历史449队列：447处当前端点的完整引用复核

R47在R44源图完整SHA边界完成；随后R46只删除四条无关研究意图claim，保留记录摘要全量不变。R48计划又重新扫描实际R46全图、447条端点、两个基因、现有同名节点及全部39处关联claim端点，并重算当前边序号。旧R47的边序号未拿来改当前图。

复核结果为443处拥有完整的当前引用结构、3处混合about分支、1处真实VCP的物种/基因-蛋白范围问题。443不是科学结论已验证数。两种错误身份来自FANCE的FACE、VCP的TERA别名出现在surface、alterations、lateral、interaction等词内部；原始完整描述不是这两个单独基因。

候选完整名称在旧图中对应18个已有节点、39处关联claim端点。逐一检查后发现同名不一定同范围：双侧海马体积有多种测量定义；脑结构改变存在宽泛/区域限定混用；部分节点带病例或来源范围metadata。R48只自动复用两个已按完整名称建立、无额外metadata的皮层表面积节点，其它同名、别名、大小写候选留待专项判断，不再造一个重复节点绕过去。

其余无已有全名或大小写/别名冲突的描述使用普通claim_concept完整字面节点，ID只取完整名称，不带论文或claim盐。保留大小写、左右侧、联合指标、群体与所有限定词；不将宽泛描述解释为单个指标，不把基因交互表达改成单个基因，不新增metadata字段、原始结论或审核章。

R48计划修复{plan['changed_endpoints']}处端点、{plan['changed_edges']}条已有边；复用{plan['reused_existing_nodes']}节点，补齐{plan['added_literal_nodes']}个按完整名称共用的普通节点。计划完成后历史449队列预计剩{plan['original_watchlist_449_remaining_after']}处，扩大队列预计剩{plan['held_endpoints']}处。这里只是计划，是否实际采用以R48/CURRENT_ACCEPTANCE.json和CAMPAIGN.json为准。

边界：只恢复有限历史清单中的身份，不外推整个扩大待核队列。原始科学字段、证据、类型和旧audit完全保留。三个混合引用、真实小鼠VCP、同名测量范围，以及BACE1、产前、ACC和17条限定语义问题仍需单独处理。R47只保存哈希和必要投影，未保存完整节点/边前像。

完整源检查完成：{inspection['at']}；R48当前源重绑定：{plan['at']}。
'''
    j.atomic_text(REVIEW / 'REPORT.md', text)


def publish_section(section, at=None, current=None):
    page = j.REPORT.read_text(encoding='utf8')
    if 'id="current-historic-literal"' in page:
        page = re.sub(r'<section id="current-historic-literal">.*?</section>', lambda _: section, page, count=1, flags=re.S)
    else:
        marker = '<section id="current-research-retirement">'
        require(marker in page, 'HTML current anchor missing'); page = page.replace(marker, section+'\n'+marker, 1)
    if current:
        page = page.replace('<h2>当前R46：四条研究意图声明已精确移除</h2>', '<h2>R46历史批次：四条研究意图声明移除成果保留</h2>')
        page = page.replace('当前已验收R46，详见下方最新结果；端点与限定语义继续审查，整体KG尚未全部优化完。',
            '当前已验收R48，详见下方最新结果；同名范围、混合引用和限定语义继续审查，整体KG尚未全部优化完。')
        page = re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R48工作图；正式full_v2未同步。</p>', page, count=1)
    qa = check_html(page); j.atomic_text(j.REPORT, page)
    return qa


def pending():
    c = j.read_json(j.OUTPUT / 'CAMPAIGN.json'); p = j.read_json(OUTPUT / 'PLAN.json')
    require(c['active_process'] and c['active_process']['kind'] == 'historic_literal_repair', 'no active R48 writer')
    require(c['current_graph'] == p['graph'] and c['current_acceptance'] == p['source_acceptance'], 'pending source differs')
    reviewed_source_report(p)
    section = f'''<section id="current-historic-literal"><h2>R48构建和独立验收中：历史完整描述身份纠正</h2>
<p>934项联合回归通过。计划修复{p['changed_endpoints']}处端点、{p['changed_edges']}条既有边，复用{p['reused_existing_nodes']}个节点并补齐{p['added_literal_nodes']}个按完整名称共用的普通概念。无新增claim、科学断言边或metadata字段；所有科学内容及旧审核不变。</p>
<p class="warning">当前已验收版本仍为R46，计划数尚未当作结果。历史449处目前剩447处，本批若验收通过将剩{p['original_watchlist_449_remaining_after']}处；扩大待核预计剩{p['held_endpoints']}处，不代表全部为错误。</p>
<p>3处混合引用、真实VCP物种、已有同名范围和其它科学范围另行处理，不强行合并。R47旧边序号未复用；计划重新扫描了当前R46源图、名字、别名、大小写冲突和引用。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','当前源修复计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','934项联合回归')} · {j.link(REVIEW/'REPORT.md','R47完整复核与复用边界')}</p></section>'''
    qa = publish_section(section)
    header = f'''# 最新：R48正在构建/独立验收，R46仍为当前已验收图

唯一写进程见CAMPAIGN.json和round48_historic_literal_repair/RUN_STATE.json。934项回归通过；禁止重启并发写进程或修改R48冻结FILES、PLAN、TEST_RESULTS.xml。计划{p['changed_endpoints']}端点/{p['changed_edges']}既有边、{p['added_literal_nodes']}普通完整名称节点/{p['reused_existing_nodes']}现有节点复用；不改类型、引文、统计、科学结论或审核对象。

成功采用后运行report_kg_historic_literal_repair.py，先通过报告测试并保存REPORT_TEST_RESULTS.xml，再验证真实查询、全部待核哈希、449与扩大队列逐项差集、所有claim字段覆盖。精确退役R46四项被替代大型派生文件，不保留旧KG或完整记录前像。

R47：447处完整复核，443处引用完整，3处混合about，1处真实VCP物种问题。已有18同名节点/39关联端点全部检查。R48仅复用两个无额外metadata的表面积节点；同名范围歧义不绕过新建。历史原449本批预计剩{p['original_watchlist_449_remaining_after']}，扩大队列预计剩{p['held_endpoints']}，两者不能混算。

窗口到2026-09-10T02:50Z（香港10:50），到点不开始新写批次，完成必要验收收口并停用kg-9-10。目标继续，不把本批当全KG已完成。下一项可处理混合引用及安全同名复用，以及17条限定语义与BACE1/产前/ACC/小鼠VCP。

---

'''
    hand = j.OUTPUT/'HANDOFF.md'; previous = hand.read_text(encoding='utf8')
    if not previous.startswith('# 最新：R48'): j.atomic_text(hand, header+previous)
    log = j.read_json(j.OUTPUT/'WORK_LOG.json'); eid = 'r47-current-historic-closure-review'
    if not any(e.get('id') == eid for e in log['events']):
        log['events'].append(dict(id=eid, at=p['at'], status='completed', title='R47历史队列完整引用复核；R48当前源重新绑定',
            body='447处审查：443引用完整，3混合about，1真实VCP物种；18已有节点、39关联claim端点全查。',
            changes='R48重新扫描实际R46全图和当前边序号；具体范围见冻结计划，身份修复尚在验收。',
            artifacts=[str(REVIEW/'REPORT.md'), str(REVIEW/'SOURCE_INSPECTION.json'), str(OUTPUT/'PLAN.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json', log)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json', dict(at=j.utc_now(), html=j.fingerprint(j.REPORT), checks=qa, plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R48_PENDING_PUBLISHED', flush=True)


def main():
    control = j.OUTPUT/'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R48 not current')
    receipt = j.read_json(c['current_acceptance']['path']); state = j.read_json(OUTPUT/'BUILD_STATE.json'); p = j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status'] == 'CURRENT_HISTORIC_LITERAL_REPAIR_APPLIED', 'wrong adoption')
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
    report = f'''# R48：历史完整描述的基因误指向批量纠正

{at}（香港时间）已采用；{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图和真实共享查询通过。

修复{p['changed_claims']}条claim的{p['changed_endpoints']}处端点及{p['changed_edges']}条既有边。复用{p['reused_existing_nodes']}个既有完整名称节点，补齐{p['added_literal_nodes']}个按完整名称共用的普通claim_concept节点。名称ID不按论文或claim加盐，不增加claim、科学边、类型推断或metadata字段。相同完整名称共用，近义、大小写、左右侧和联合范围不擅自混并。

这些描述原先错误挤在FANCE/VCP基因上，来自FACE/TERA的字内误匹配。恢复完整名称的身份，不等于验证每条研究结论。原有词面、类型、否定、证据、统计、条件、来源和旧audit都完全保留；所有非目标既有节点和边记录不变。泛化描述和基因交互仍作为完整描述，不强制当影像指标或单个基因。

当前{n['nodes']:,}节点、{n['claims']:,}条claim、{n['edges']:,}条边。节点增加是分离不同概念的错误基因身份，并非新增论文或科学结论。metadata并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim的{cov['claim_fields_unchanged']}项字段覆盖逐项不变，内层平均{cov['average_nested_fields']:.2f}项。p值/效应量/样本量非空率仍为{cov['p_value_nonempty_pct']:.2f}%/{cov['effect_size_nonempty_pct']:.2f}%/{cov['sample_size_nonempty_pct']:.2f}%，没有补值。

历史449队列在R44后剩447处，本批修复{queue['original_watchlist_repaired_this_batch']}，现在剩{queue['original_watchlist_remaining']}处。扩大待核队列当前{queue['expanded_remaining']}处/{queue['expanded_claims']}条claim，不是全部确认错误。3处混合引用、真实VCP物种、其它同名节点范围以及17条旧语义、BACE1、产前、ACC等仍需继续处理。各清单可重叠且不穷尽全KG。

共享关系组{rel['shared_groups']:,}，成员{rel['indexed_claims']:,}；默认至少两篇已核实未确认撤稿来源{len(default)}组，含确认撤稿时{len(inclusive)}组。新入共享索引{len(comp['newly_shared_claim_ids'])}条，不再共享{len(comp['no_longer_shared_claim_ids'])}条；身份修复可以拆开假共享，不以共享率越高越好，也不等于共识数。

R47旧图投影经R46保留记录证明衔接，再由R48实际当前全图SHA、447处原记录、基因/同名节点、39处关联端点和当前边序号重新检查。构建源SHA、独立候选全图逆变换源节点/边摘要、所有claim/书目/共享成员普查、metadata覆盖、明细库完整SHA均通过；实际查询及全部存续待核哈希也通过。

旧R46四项大型派生文件已精确退役{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。原始文献和明细保留，只有当前工作KG，没有历史KG或完整记录前像可回退；当前派生数据可重建。当前图仍在R23路径，当前索引/普查/覆盖/待核在R48，terms在R30、paper registry在R34。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。无模型、KGE、训练或重新抽取。incremental-integrity-checks用于普通状态读取，实际修改边界全量验证。目标继续至2026-09-10T02:50Z（香港10:50）；全KG尚未全部优化完。
'''
    j.atomic_text(OUTPUT/'REPORT.md', report)
    section = f'''<section id="current-historic-literal"><h2>当前R48：历史完整描述身份已批量纠正</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图及真实查询通过。修复{p['changed_endpoints']}处端点/{p['changed_edges']}条既有边；原始科学字段和旧审核保持不变。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>完整名称节点</td><td>复用{p['reused_existing_nodes']}；补齐{p['added_literal_nodes']}个共用普通概念，不按论文重复创建</td></tr>
<tr><td>历史449待核队列</td><td>本批前447；修复{p['changed_endpoints']}；现剩{queue['original_watchlist_remaining']}处</td></tr>
<tr><td>扩大待核 / 旧语义</td><td>{queue['expanded_remaining']:,}处 / 17条，不等于全部确认错误</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨已核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim各字段覆盖不变，新增字段0</td></tr></table>
<p>同名歧义先待核，不通过再造节点绕过。节点略增用于分开被挤在基因上的不同描述；科学结论和论文数量不增加。旧R46四项派生文件已清理，不保留旧KG回退前像。</p>
<p class="warning">混合引用、同名范围、真实VCP物种及BACE1、产前、ACC、限定语义继续审查，整体KG尚未全部优化完。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批完整结果')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','真实查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','逐字段覆盖')} · {j.link(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json','当前待核口径')} · {j.link(REVIEW/'REPORT.md','R47复核边界')}</p></section>'''
    page = j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative = Path(fp['path']).relative_to(j.REPO).as_posix()
        page = re.sub(rf'<a href="{re.escape(relative)}">.*?</a>', '<span>旧派生资料已由R48当前文件替换</span>', page)
    j.atomic_text(j.REPORT, page); qa = publish_section(section, at, current=True)
    j.atomic_text(j.OUTPUT/'HANDOFF.md', report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log = j.read_json(j.OUTPUT/'WORK_LOG.json'); eid = 'r48-finite-historic-literal-repair'
    if not any(e.get('id') == eid for e in log['events']):
        log['events'].append(dict(id=eid, at=receipt['at'], status='completed', title='R48历史完整描述基因身份批量纠正',
            body=f"{p['changed_endpoints']}处端点/{p['changed_edges']}既有边；复用{p['reused_existing_nodes']}，补齐{p['added_literal_nodes']}共用普通节点。",
            changes=f"{receipt['test_counts']['tests']}项回归及全图逆变换、普查、覆盖、真实查询通过；历史449现剩{queue['original_watchlist_remaining']}，科学/audit未改。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json', log)
    require(j.read_json(control) == c, 'campaign advanced'); j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result'] = j.fingerprint(ret_path); c['latest_source_review'] = j.fingerprint(OUTPUT/'REPORT.md'); j.atomic_json(control, c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json', dict(at=j.utc_now(), html=j.fingerprint(j.REPORT), checks=qa,
        acceptance=c['current_acceptance'], runtime=c['current_runtime_acceptance'], queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        coverage=j.fingerprint(cov_path), retention=j.fingerprint(ret_path), report_code=j.fingerprint(Path(__file__)),
        tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'), test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_historic_literal_report.py')))
    print('R48_PUBLISHED', dict(counts=n, relations=rel, default_groups=len(default), queue=queue, retired_bytes=ret['bytes_removed'], checks=qa), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending', action='store_true')
    pending() if parser.parse_args().pending else main()
