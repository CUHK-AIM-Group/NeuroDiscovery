"""R54/R55 exact existing-node reuse accounting, queries and single-current retention."""
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

OUTPUT = j.OUTPUT / 'round55_existing_literal_reuse'
REVIEW = j.OUTPUT / 'round54_existing_literal_review'
PREVIOUS = j.OUTPUT / 'round53_expanded_literal_repair'


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
    inspection=j.read_json(plan['source_review']['path'])
    require(j.fingerprint(plan['source_review']['path'])==plan['source_review'],'R54 proof changed')
    text=f'''# R54：唯一完整名称节点的复用范围检查

从R52现有同名候选中，先排除多个候选、别名/大小写而非精确主名、分子身份、未声明类型、定义或外部身份冲突和额外metadata范围。382处端点/266个目标进入当前R50完整SHA复核；扫描全部当前claim引用及非claim连接，检查946处既有关联，不删任何节点。

其中75处因既有claim使用不同完整名称而保留，2处因无当前claim关联而保留；305处端点/304条claim可以复用220个已有完整名称节点。即使字面相同也不强行忽略区域、群体或不同既有名称。

R55在R53已验收后重扫实际当前图：只将R53修改同条claim另一侧导致的哈希变化重新绑定；所选端点、原始科学及旧audit未变。R53全记录逆验证、当前claim哈希、213个源基因、所有名字/别名/大小写、全既有目标关联和当前所属边都必须相符。R55独立候选扫描还须逆变换并再次核对全部源记录及这些目标关联。

本批无新节点、全局节点合并或科学字段修改。既有节点metadata、定义、类型、外部身份和离线详情均保留；未来来源无权自动继承旧claim的科学审核。历史449剩3处海马定义不在本批；扩大队列2511预计降至{plan['held_endpoints']}处。科学范围和未确定问题不会被自动关闭。

R54检查：{inspection['at']}；R55当前源计划：{plan['at']}。无模型、训练或正式full_v2同步。
'''
    j.atomic_text(REVIEW/'REPORT.md',text)


def publish_section(section,at=None,current=None):
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="current-existing-literal"' in page:
        page=re.sub(r'<section id="current-existing-literal">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-expanded-literal">'
        require(marker in page,'HTML current R53 anchor missing');page=page.replace(marker,section+'\\n'+marker,1)
    if current:
        page=page.replace('<h2>当前R53：扩大队列完整描述身份已批量纠正</h2>','<h2>R53历史批次：扩大队列完整描述身份已批量纠正</h2>')
        page=page.replace('当前已验收R53，详见下方最新结果；同名身份及科学范围继续审查，整体KG尚未全部优化完。',
            '当前已验收R55，详见下方最新结果；科学范围及其余身份继续审查，整体KG尚未全部优化完。')
        page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
            f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R55工作图；正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);return qa


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(OUTPUT/'PLAN.json')
    require(c['active_process'] and c['active_process']['kind']=='existing_literal_reuse','no R55 writer')
    require(c['current_graph']==p['graph'] and c['current_acceptance']==p['source_acceptance'],'pending source differs')
    reviewed_source_report(p)
    state=j.read_json(OUTPUT/'RUN_STATE.json')
    section=f'''<section id="current-existing-literal"><h2>R55构建/独立验收中：复用已有完整名称节点</h2>
<p>1,125项联合回归通过。计划纠正{p['changed_claims']:,}条claim、{p['changed_endpoints']:,}处端点和{p['changed_edges']:,}条既有边，复用{p['reused_existing_nodes']:,}个已有节点；不新增节点或科学断言、不删claim/边、不改科学内容和旧audit。</p>
<p class="warning">当前已验收仍为R53。扩大队列2,511处预计剩{p['held_endpoints']:,}处；历史449仍剩3处海马定义。正在验收的计划数不是完成结果。</p>
<p>唯一精确主名、无身份冲突、全部现有关联名称一致和无额外非claim关系，四项均须通过；R53另一侧变更须重新绑定当前源。</p>
<p>{j.link(OUTPUT/'RUN_STATE.json','实时状态')} · {j.link(OUTPUT/'PLAN.json','当前源计划')} · {j.link(OUTPUT/'TEST_RESULTS.xml','1125项联合回归')} · {j.link(REVIEW/'REPORT.md','R54复用边界')}</p></section>'''
    qa=publish_section(section)
    header=f'''# 最新：R55已有节点复用正在构建/独立验收，R53仍为当前已验收图

唯一KG写进程见CAMPAIGN.json和R55/RUN_STATE.json。1125项联合回归通过；不要改冻结FILES、PLAN.json或TEST_RESULTS.xml，不启动第二个KG写进程。计划304条claim/305端点/{p['changed_edges']}既有边，复用220已有节点，新增0；完整原始科学/audit/现有节点/详情不变。

成功采用后运行test_kg_existing_literal_reuse_report.py保存REPORT_TEST_RESULTS.xml，随后report_kg_existing_literal_reuse.py完成实际共享查询、全待核当前hash、2511减305差集、逐claim字段覆盖和R53四项大型派生退役。当前KG在R23，绝不保留旧KG前像。

R53已验收2594104节点/905179claim/3011990边；历史449仍剩3、扩大2511端点/2351claim、旧语义17。共享2587/5868，默认跨核实未撤稿来源519。R53报告已发布，R50旧四派生已退役766612472B。R51科学原文报告已完成但未修改科学字段。

R54全R50源检382端点/266目标/946现有关联；305端点/220目标允许复用，75异名关联+2无当前关联保持待核。R55最新R53全SHA又检当前源claim、213源基因、精确名字/别名/大小写、全部目标关联及所属边，旧R50哈希仅在R53明确另一侧变更下重新绑定。无模糊或全局合并，无模型、训练或full_v2同步。

窗口到2026-09-10T02:50Z（香港10:50）；到点不新开写批次，完成必要验收并停用kg-9-10。目标继续，本批不是全KG完成。

---

'''
    hand=j.OUTPUT/'HANDOFF.md';old=hand.read_text(encoding='utf8')
    if not old.startswith('# 最新：R55'):j.atomic_text(hand,header+old)
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r54-existing-literal-full-incidence-review'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=p['at'],status='completed',title='R54已有完整名称节点全连接复核',
            body='382端点/266目标/946现有关联；305端点复用220节点进入R55，异名75及无关联2保留。',
            changes='当前R53重新绑定与全源验证，尚未将计划数记作完成；无新节点或科学内容改变。',
            artifacts=[str(REVIEW/'REPORT.md'),str(REVIEW/'SOURCE_INSPECTION.json'),str(OUTPUT/'PLAN.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    j.atomic_json(OUTPUT/'PENDING_REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,plan=j.fingerprint(OUTPUT/'PLAN.json')))
    print('R55_PENDING_PUBLISHED',flush=True)


def main():
    control = j.OUTPUT/'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R55 not current')
    receipt = j.read_json(c['current_acceptance']['path']); state = j.read_json(OUTPUT/'BUILD_STATE.json'); p = j.read_json(OUTPUT/'PLAN.json')
    require(receipt['status'] == 'CURRENT_EXISTING_LITERAL_REUSE_APPLIED', 'wrong adoption')
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
    report=f'''# R55：已有完整名称节点复用已验收

{at}（香港时间）采用；{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图与真实共享查询通过。

{p['changed_claims']:,}条claim的{p['changed_endpoints']:,}处端点及{p['changed_edges']:,}条既有边已纠正，直接复用{p['reused_existing_nodes']:,}个已有节点。新增节点0、删除claim/边/概念0。不是把基因节点全局改成测量：源基因、已有普通节点、原metadata、旧audit及详情全部原样保留。

当前{n['nodes']:,}节点、{n['claims']:,}条claim、{n['edges']:,}条边。metadata字段并集节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim {cov['claim_fields_unchanged']}项字段覆盖逐项相同，内层平均{cov['average_nested_fields']:.2f}字段。p值/效应量/样本量非空率{cov['p_value_nonempty_pct']:.2f}% / {cov['effect_size_nonempty_pct']:.2f}% / {cov['sample_size_nonempty_pct']:.2f}%，没有补值或声称100%覆盖。

扩大队列2,511 → {queue['expanded_remaining']:,}处/{queue['expanded_claims']:,}条claim。历史449仍剩3处海马定义；17旧限定语义、产前/BACE1/ACC科学范围继续开放。剩余队列不是全部已确认错误，也不穷尽全KG的问题。

共享关系{rel['shared_groups']:,}组、{rel['indexed_claims']:,}条成员；默认至少两篇已核实且未确认撤稿来源{len(default)}组，含已确认撤稿{len(inclusive)}组。新入共享索引{len(comp['newly_shared_claim_ids'])}条，不再共享{len(comp['no_longer_shared_claim_ids'])}条。这是关系与来源汇聚，不等于独立研究队列、科学共识或科学审核升级。

本批只接受唯一精确完整主名、无分子/外部/额外范围冲突、所有现有关联名称一致且没有额外非claim关系的目标。R53若改过同一claim另一端，先凭已验收全源逆变换桥接，再用实际当前图的全SHA、claim/节点/全部名字别名/源基因/所属边和目标全关联复查。独立候选扫描逆变换复现全部源节点/边摘要，再检查全部目标原有关联。全部claim/书目/共享成员普查、metadata覆盖、明细/普查库全SHA和实际查询均通过；原始科学证据及审核含义没有改变。

仅退役R53四项被替代的大型派生文件{ret['bytes_removed']:,}字节（{ret['bytes_removed']/1024**2:.2f}MiB）。当前派生可由当前KG重建，旧KG没有保留回退；原始来源及详情未删。工作图仍在R23路径，最新索引/普查/队列/覆盖在R55。正式full_v2未同步，claim累计差额{c['formal_sync_gap_claims']}。

SHA-256：{c['current_graph']['sha256']}。没有模型、KGE、训练或重新抽取。incremental-integrity-checks用于常规状态，修改边界仍全量验证。HTML通过静态链接检查，未做浏览器视觉验收。继续到2026-09-10T02:50Z（香港10:50），不宣称整个KG已完全优化。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report)
    section=f'''<section id="current-existing-literal"><h2>当前R55：复用已有完整名称节点已验收</h2>
<div class="notice">{receipt['test_counts']['tests']}项联合回归、{suite.get('tests')}项报告测试、独立全图与实际查询通过。修正{p['changed_endpoints']:,}处端点/{p['changed_edges']:,}条既有边，复用{p['reused_existing_nodes']:,}个已有节点。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{n['nodes']:,} / {n['claims']:,} / {n['edges']:,}</td></tr>
<tr><td>新增 / 删除</td><td>0 新节点，0 claim/边/概念删除，科学字段和旧audit不变</td></tr>
<tr><td>扩大待核队列</td><td>2,511 → {queue['expanded_remaining']:,}处 / {queue['expanded_claims']:,}条claim；不是全部已确认错误</td></tr>
<tr><td>历史449 / 旧语义</td><td>仍剩3处海马定义 / 17条限定语义</td></tr>
<tr><td>共享组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；默认跨核实未撤稿来源{len(default)}组</td></tr>
<tr><td>metadata字段并集</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}；claim各字段覆盖不变</td></tr></table>
<p>不是仅凭同名合并：唯一完整主名、全部现有关联范围、无外部身份冲突、当前源记录及完整边都通过。既有概念/源基因/详情/文献保留，不全局合并概念。</p>
<p class="warning">科学范围、海马定义及其余分子/别名/未声明类型等继续待核；不把身份纠正说成科学结论验证。</p>
<p>{j.link(OUTPUT/'REPORT.md','本批结果')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','真实查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','逐字段覆盖')} · {j.link(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json','待核口径')} · {j.link(REVIEW/'REPORT.md','R54复用范围')}</p></section>'''
    page = j.REPORT.read_text(encoding='utf8')
    for fp in ret['removed']:
        relative = Path(fp['path']).relative_to(j.REPO).as_posix()
        page = re.sub(rf'<a href="{re.escape(relative)}">.*?</a>', '<span>旧派生资料已由R55当前文件替换</span>', page)
    j.atomic_text(j.REPORT, page); qa = publish_section(section, at, current=True)
    j.atomic_text(j.OUTPUT/'HANDOFF.md', report+'\n\n'+(REVIEW/'REPORT.md').read_text(encoding='utf8'))
    log = j.read_json(j.OUTPUT/'WORK_LOG.json'); eid = 'r55-finite-existing-literal-reuse'
    if not any(e.get('id') == eid for e in log['events']):
        log['events'].append(dict(id=eid, at=receipt['at'], status='completed', title='R55复用已有完整名称节点',
            body=f"{p['changed_endpoints']}处端点/{p['changed_edges']}既有边；复用{p['reused_existing_nodes']}，补齐{p['added_literal_nodes']}共用普通节点。",
            changes=f"{receipt['test_counts']['tests']}项回归及全图逆变换、普查、覆盖、真实查询通过；历史449现剩{queue['original_watchlist_remaining']}，科学/audit未改。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','COVERAGE_COMPARISON.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json', log)
    require(j.read_json(control) == c, 'campaign advanced'); j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result'] = j.fingerprint(ret_path); c['latest_source_review'] = j.fingerprint(OUTPUT/'REPORT.md'); j.atomic_json(control, c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json', dict(at=j.utc_now(), html=j.fingerprint(j.REPORT), checks=qa,
        acceptance=c['current_acceptance'], runtime=c['current_runtime_acceptance'], queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        coverage=j.fingerprint(cov_path), retention=j.fingerprint(ret_path), report_code=j.fingerprint(Path(__file__)),
        tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'), test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_existing_literal_reuse_report.py')))
    print('R55_PUBLISHED', dict(counts=n, relations=rel, default_groups=len(default), queue=queue, retired_bytes=ret['bytes_removed'], checks=qa), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending', action='store_true')
    pending() if parser.parse_args().pending else main()
