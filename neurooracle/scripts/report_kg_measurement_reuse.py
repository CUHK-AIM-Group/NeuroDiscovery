"""R38 actual query, same HTML and strictly scoped current-derivative retirement."""
import argparse
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from report_current_kg import check_html
from report_kg_literal_endpoint_repair import comparison
from neurooracle.src.shared_relation_catalog import current_shared_relations, find_shared_relations

OUTPUT = journal.OUTPUT / 'round38_measurement_reuse'
PREVIOUS = journal.OUTPUT / 'round37_relation_scope'
PAIRS = [('CLM:951f2bb246a3', 'CLM:CASE1MAN:39829963:284'),
         ('CLM:1b4579c82247', 'CLM:CASE1MAN:39829963:286')]
DERIVATIVES = ('CURRENT_PAPER_CENSUS.sqlite', 'CURRENT_SHARED_RELATIONS.jsonl',
               'CURRENT_PAPER_ISSUES.jsonl', 'CURRENT_METADATA_COVERAGE.jsonl')


def restored_pairs(groups):
    result = []
    for pair in PAIRS:
        found = [g for g in groups if set(pair) <= {m['claim_id'] for m in g['members']}]
        require(len(found) == 1, 'owning same-paper pair not restored exactly once')
        members = [m for m in found[0]['members'] if m['claim_id'] in pair]
        require(all(m['paper_key'] == 'pmid:39829963' and m['paper_status'] == 'verified' for m in members),
                'restored pair source identity differs')
        result.append(dict(claim_ids=list(pair), relation_id=found[0]['id'],
                           source_key='pmid:39829963', new_independent_sources=0))
    return result


def validate_retirement_paths(targets, current, previous=PREVIOUS, root=journal.OUTPUT):
    allowed = {(previous / name).resolve() for name in DERIVATIVES}
    actual = [Path(fp['path']).resolve(strict=True) for fp in targets]
    require(len(actual) == 4 and len(set(actual)) == 4 and set(actual) == allowed, 'wrong retirement set')
    require(all(p.is_relative_to(root.resolve()) and p not in current and p.is_file() for p in actual),
            'unsafe obsolete derivative target')
    return actual


def pending():
    require(not (OUTPUT / 'CURRENT_ACCEPTANCE.json').exists(), 'already adopted')
    section = f'''<section id="measurement-reuse-pending"><h2>R38 正在全图验收</h2>
<p>557项回归测试通过。计划修复28条claim、33条关联边，复用3个已有完整指标节点，增加23个按完整名称共用的普通节点；没有新增metadata字段。</p>
<p>6条网络面积指标已通过本篇影像方法和原文核实。其中3条把上界r &lt; 0.12误作精确值：COP按结果表修正为负相关及名义p值，LAN/DAN保留上界并清除伪精确值。不改原文、否定、研究条件和论文身份。通过独立全图验收前，当前仍以R37为准。</p>
<p>{journal.link(OUTPUT/'RUN_STATE.json', '原子进度')} · {journal.link(OUTPUT/'PLAN.json', '28条修复计划')} · {journal.link(OUTPUT/'TEST_RESULTS.xml', '557项测试')}</p></section>'''
    page = journal.REPORT.read_text(encoding='utf8')
    if 'id="measurement-reuse-pending"' in page:
        page = re.sub(r'<section id="measurement-reuse-pending">.*?</section>', lambda _: section, page, count=1, flags=re.S)
    else:
        marker = '<section id="current-literal-endpoints">'
        require(marker in page, 'current insertion point missing')
        page = page.replace(marker, section + '\n' + marker, 1)
    check_html(page); journal.atomic_text(journal.REPORT, page)
    print('R38_PENDING_LOGGED; R37 remains accepted', flush=True)


def main():
    control = journal.OUTPUT / 'CAMPAIGN.json'; c = journal.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(journal.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json') == c['current_acceptance'], 'R38 not current')
    receipt = journal.read_json(c['current_acceptance']['path']); state = journal.read_json(OUTPUT/'BUILD_STATE.json')
    plan = journal.read_json(OUTPUT/'PLAN.json'); census = journal.read_json(c['current_paper_census']['path'])
    require(receipt['status'] == 'CURRENT_MEASUREMENT_REUSE_APPLIED' and receipt['test_counts']['tests'] == 557, 'unexpected receipt')
    require(receipt['counts'] == c['counts'] and receipt['relation_counts'] == c['relation_evidence_counts'], 'control counts differ')
    require((len(receipt['metadata_key_unions']['nodes']), len(receipt['metadata_key_unions']['edges'])) ==
            (c['node_metadata_field_union'], c['edge_metadata_field_union']) == (120,36), 'metadata summary differs')
    for fp in [*[c[k] for k in ('current_runtime_acceptance','current_gene_holds','current_coverage','current_paper_issues',
            'current_paper_census','current_issues','current_structure_holds','current_census_normalization')],
            receipt['provenance_plan'], receipt['identity_audit'], *receipt['code'], receipt['tests']]:
        require(journal.fingerprint(fp['path']) == fp, 'current frozen evidence changed')
    require(journal.read_json(c['current_runtime_acceptance']['path'])['acceptance'] == c['current_acceptance'], 'runtime binding differs')
    suite = ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests')) >= 6 and all(int(suite.get(k)) == 0 for k in ('failures','errors','skipped')), 'report tests failed')
    groups = list(current_shared_relations(control))
    default = list(find_shared_relations(control, minimum_papers=2))
    inclusive = list(find_shared_relations(control, minimum_papers=2, include_retracted=True))
    restored = restored_pairs(groups)
    require(len(groups) == receipt['relation_counts']['shared_groups'] and len(default) == 518 and len(inclusive) == 520,
            'shared/source counts differ from reviewed unchanged source scope')
    query = dict(at=journal.utc_now(), graph=c['current_graph'], acceptance=c['current_acceptance'],
        raw_groups=len(groups), default_minimum_two_source_groups=len(default), inclusive_minimum_two_source_groups=len(inclusive),
        restored_same_paper_pairs=restored, index_and_registries_hash_verified=True, graph_guard='accepted full verification + native fingerprint',
        no_new_claims_papers_or_consensus_inferred=True)
    journal.atomic_json(OUTPUT/'QUERY_VALIDATION.json', query)
    compare_path = OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not compare_path.exists():
        old = state['baseline']['current_shared_relations']
        require(journal.fingerprint(old['path']) == old, 'previous index changed')
        journal.atomic_json(compare_path, dict(at=journal.utc_now(), graph=c['current_graph'], **comparison(groups, rows(old['path']))))
    compared = journal.read_json(compare_path)
    require(compared['graph'] == c['current_graph'], 'comparison not current')
    held = rows(c['current_gene_holds']['path']); coverage = rows(c['current_coverage']['path'])
    require(len(held) == plan['held_endpoints'] == 460 and len({r['claim_id'] for r in held}) == c['pending_gene_link_claims'], 'hold closure differs')
    audit = journal.read_json(receipt['identity_audit']['path'])
    journal.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    retention_path = OUTPUT/'RETENTION_RESULT.json'
    if not retention_path.exists():
        old_db = journal.read_json(state['baseline']['current_paper_census']['path'])['database']
        targets = [old_db, *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current = {Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations',
            'current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths = validate_retirement_paths(targets, current)
        for fp, path in zip(targets, paths):
            journal.guards(fp); require(sha256(path) == fp['sha256'], 'obsolete derivative changed')
        require(journal.read_json(control) == c, 'writer advanced before retirement')
        for path in paths: path.unlink()
        journal.atomic_json(retention_path, dict(status='COMPLETED', at=journal.utc_now(), removed=targets,
            bytes_removed=sum(fp['bytes'] for fp in targets), source_kg_backups=0, record_preimages_saved=False,
            original_archive_and_public_evidence_deleted=False,
            recoverability='No historical copies retained. Current derivatives are rebuildable from current KG; no historical KG rollback.'))
    retention = journal.read_json(retention_path)
    at = journal.local_time(receipt['at']); counts = c['counts']; relation = c['relation_evidence_counts']
    report = f'''# R38：网络指标复用与有来源的统计值修复

{at}（香港）已采用到当前工作KG；独立全图、557项回归及实际查询通过。仅KG质量治理，没有模型、KGE、训练或正式full_v2同步。

## 本批一并解决

- 28条claim的28处错误基因指向、33条边端点修复；复用3个已有完整指标节点，建立23个按完整名称共用的普通CLM_CONCEPT节点。完整名称、大小写、侧别及限定词保留，不按每篇论文创建节点，不声称新的UMLS等价身份。
- 6条原标为network的功能网络面积指标，经本篇影像方法与完整摘要核实，不再错误连接FANCE。其余22处为明确IMAGING类型的测量表达漏检，如皮层变薄、复数volumes及灰质损失；只是扩大具体测量识别，不缩短或模糊合并名称。[Mamah等，2024，PubMed](https://pubmed.ncbi.nlm.nih.gov/39829963/)。
- 3条原记录把摘要共同上界r &lt; 0.12当作单个指标的精确r=0.12，并混用了logistic regression方法名。COP相关记录按本篇Table 2的ASR Thought Problems列修为r=-0.088、名义p=0.007，方法为Pearson相关；LAN/DAN预测记录清除伪精确效应值，原始统计字段保留r &lt; 0.12，方法对齐stepwise regression。不拿单变量相关替代回归系数，不把名义p值说成多重校正通过。[原文与Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11740805/#tbl2)（Mamah等，CC BY 4.0）。
- 原始引文、predicate、否定、条件、样本量及论文身份不变；只有获批3条的特定数值、方法名和原始统计解析值变动。没有删除claim、科学证据或新造科学关系边。
- 上一批暂时分开的COP与FPN两组同篇表述已重新对应。实际新增进入共享集合的claim {len(compared['newly_shared_claim_ids'])}条，退出{len(compared['no_longer_shared_claim_ids'])}条；这不代表新增论文或独立实验。共享索引归一关系，claim仍保留逐篇/逐条件证据。

## 当前总量

| 项目 | 当前 |
|---|---:|
| 节点 | {counts['nodes']:,}（+23） |
| claim节点 | {counts['claims']:,}（不变） |
| 边 | {counts['edges']:,}（不变） |
| 全部细粒度关系组 | {relation['all_fine_grained_relation_groups']:,} |
| 共享关系组 / 成员claim | {relation['shared_groups']:,} / {relation['indexed_claims']:,} |
| 至少两个已核实论文来源的组 | 默认{len(default)}；含已确认撤稿来源{len(inclusive)} |
| metadata键并集 | 节点{len(receipt['metadata_key_unions']['nodes'])} / 边{len(receipt['metadata_key_unions']['edges'])}；新增0 |
| 覆盖统计行数 | {len(coverage)}；独立全量重算，不补空值制造100% |
| 连通分量 / 孤立点 / 自环 | {receipt['connected_components']:,} / {receipt['isolated_nodes']:,} / {receipt['self_loops']:,} |

来源身份：verified {audit['claim_status']['verified']:,}，conflict {audit['claim_status']['conflict']:,}，unverified {audit['claim_status']['unverified']:,}。未核实并非错误，论文不同也不自动表示独立实验。不能把全部原始paper key当已核实论文数。

## 尚未完成及下一步

剩余460处基因端点、{len({r['claim_id'] for r in held})}条claim待核，并非全部已确认错误。21条语义、1条frailty结构及NAcc书目/年龄侧别范围问题仍保留。未知类型、分子机制、同名多候选、复合表达和混合引用不强行归并；不批量翻转否定、不把研究设计写成结果。

优先用现有原始输入与公开原文解决完整同名多候选和混合引用；不为提高共享率牺牲原始证据。当前census/index/coverage/holds/publication review在R38，实体terms仍R30，论文registry仍R34。当前KG仍在R23路径，不要重跑旧构建。

## 验收与单版本保留

源图完整SHA、候选独立全图、所有原节点/边逆向摘要、每个批准字段与完整引用事件、905,184条claim/书目/共享普查、实体与论文/出版见证、全量metadata覆盖、明细库与当前普查SHA均验证。能力标志在采用前发布，实际查询不需修补KG。

旧R37普查数据库、共享索引、书目待核与覆盖表共{retention['bytes_removed']:,}字节已删除，无历史KG或记录前像。只保留当前KG及必要派生资料；原始档案、公共证据和小型哈希/数量验收凭据保留。无旧版本回退副本。

KG SHA-256：{c['current_graph']['sha256']}，{c['current_graph']['bytes']:,}字节。

使用incremental-integrity-checks：进度用原子状态与可信指纹，构建/采用边界执行全量核验；不为状态汇报重复扫描巨型KG。整体目标未完成；本轮夜间窗口截至2026-09-09 10:00HKT。
'''
    next_plan = '''# R39接续计划

当前以CAMPAIGN.json为准：R38当前工作图已验收；R23图路径不变，R38持有当前census/index/coverage/holds/publication review，R30实体terms、R34论文registry保留。R37四个旧派生文件已删除，不重跑R37/R38或旧验收。

1. 本批修复28端点、33边及3条特定数值/方法解析；28条原文/条件/来源保留。COP/FPN同篇共享已恢复，不是新增独立论文。当前557项回归、独立全图、实际查询均已通过。
2. 460端点待核，原因表见PLAN.json；优先7处多候选、2处混合about引用的完整事件核查。不可任意挑canonical节点，不通过截短/名字相似/多数票归并。复合指标保持完整。
3. 21语义与1frailty结构保留。R37公开来源与R38源文均只读，可继续逐条核实；不得把假说、设计、not_specific等泛化为因果或用统一negated翻转。
4. NAcc条目CLM:CASE1MAN:OA:W31756730:11949同时有书目和成人双侧/青少年左侧语境问题；R37/PMC6969163.xml为原文，不从合成ID猜PMID或只改标题。明确源文重建整条含义需单独审批计划/验收。
5. 同一HTML持续记录。活动writer只检查/接续，不修改冻结代码或启第二writer。只保留当前KG，无模型/训练、正式full_v2同步、历史KG/记录前像。
6. 夜间窗口至2026-09-09 10:00HKT。完成安全批次后及时记录未解决项；当前目标控制blocked已记录，不绕过目标数据库、不假称全图已无问题。
'''
    journal.atomic_text(OUTPUT/'REPORT.md', report); journal.atomic_text(OUTPUT/'NEXT_BATCH_PLAN.md', next_plan)
    section = f'''<section id="current-measurement-reuse"><h2>当前 R38：28条指标、33条边引用、3条统计解析修复</h2>
<div class="notice">557项回归、独立全图和实际查询通过；本批已采用。原文、否定、研究条件和论文身份保留。没有模型训练或正式同步。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}；节点+23，claim与边数不变</td></tr>
<tr><td>本批修复</td><td>28处错误基因指向、33条关联边；复用3个已有完整指标节点</td></tr>
<tr><td>数值与方法</td><td>COP：r=-0.088，名义p=0.007；LAN/DAN清除伪精确0.12并保留上界。不混用相关与回归结果。</td></tr>
<tr><td>metadata键并集</td><td>节点120 / 边36；本批新增0</td></tr>
<tr><td>共享组 / 成员claim</td><td>{relation['shared_groups']:,} / {relation['indexed_claims']:,}；默认至少两篇已核实来源组518</td></tr>
<tr><td>待核</td><td>460处基因端点、{len({r['claim_id'] for r in held})}条claim；21语义与1结构另保留</td></tr></table>
<p>已恢复COP/FPN两组同篇表述的关系复用；新增进入共享集合{len(compared['newly_shared_claim_ids'])}条、退出{len(compared['no_longer_shared_claim_ids'])}条，不代表新增论文或独立实验。新增节点按完整名称共用，不按论文或claim创建；不添加UMLS身份断言。</p>
<p class="warning">整体尚未完成：来源未核实、复合指标、未知类型、分子机制与混合引用仍需证据，不能为共享率强行合并。</p>
<p>旧R37四项派生资料{retention['bytes_removed']:,}字节已删除，不保留历史KG或记录回退副本。当前SHA-256：<code>{c['current_graph']['sha256']}</code></p>
<p>{' · '.join(journal.link(OUTPUT/n, label) for n,label in [('REPORT.md','完整说明与公开来源'),('CURRENT_ACCEPTANCE.json','当前验收'),('CURRENT_RUNTIME_ACCEPTANCE.json','当前运行时'),('PLAN.json','完整修复计划'),('QUERY_VALIDATION.json','实际查询'),('RELATION_MEMBERSHIP_COMPARISON.json','共享变化'),('CURRENT_GENE_ENDPOINT_HOLDS.jsonl','460处待核'),('CURRENT_METADATA_COVERAGE.jsonl','当前覆盖表'),('CENSUS.json','当前普查'),('TEST_RESULTS.xml','557项回归'),('NEXT_BATCH_PLAN.md','下一批计划')])}</p></section>'''
    page = journal.REPORT.read_text(encoding='utf8')
    page = re.sub(r'<section id="measurement-reuse-pending">.*?</section>', '', page, count=1, flags=re.S)
    if 'id="current-measurement-reuse"' in page:
        page = re.sub(r'<section id="current-measurement-reuse">.*?</section>', lambda _: section, page, count=1, flags=re.S)
    else:
        marker = '<section id="current-literal-endpoints">'; require(marker in page, 'HTML insertion missing')
        page = page.replace(marker, section + '\n' + marker, 1)
        page = re.sub(r'(<section id="current-literal-endpoints">.*?</section>)',
            lambda m: '<details><summary>R37 历史批次：718条完整指标</summary>' + m[1].replace('当前 R37：','历史 R37：',1) + '</details>', page, count=1, flags=re.S)
    for fp in retention['removed']:
        relative = Path(fp['path']).relative_to(journal.REPO).as_posix()
        page = re.sub(rf'<a href="{re.escape(relative)}">.*?</a>', '<span>旧派生资料已替换，仅保留R38当前文件</span>', page)
    page = re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',
        f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R38工作图，正式full_v2未同步。</p>', page, count=1)
    page = page.replace('R37已完成整体验收；后续批次通过验收后才替换当前工作图。','R38已完成整体验收；后续批次通过验收后才替换当前工作图。')
    qa = check_html(page); journal.atomic_text(journal.REPORT, page)
    journal.atomic_text(journal.OUTPUT/'HANDOFF.md', '# Current data and runtime — R38\n\n' + report + '\n' + next_plan)
    log_path = journal.OUTPUT/'WORK_LOG.json'; log = journal.read_json(log_path)
    if not any(e.get('id') == 'r38-measurement-reuse-20260909' for e in log['events']):
        log['events'].append(dict(id='r38-measurement-reuse-20260909', at=receipt['at'], status='completed',
            title='R38完整网络指标复用与三条统计解析修复', body='28端点、33边引用、23普通节点、复用3；557项回归及独立全图/实际查询通过。',
            changes='3条获批统计值/方法修复；原文与来源不变，metadata新增0；旧R37派生资料删除。',
            artifacts=[str(OUTPUT/n) for n in ('REPORT.md','CURRENT_ACCEPTANCE.json','QUERY_VALIDATION.json','NEXT_BATCH_PLAN.md')]))
        journal.atomic_json(log_path, log)
    require(journal.read_json(control) == c, 'campaign advanced')
    journal.guards([c['current_graph'], c['current_detail_store'], c['formal_sources']])
    c['retention_result'] = journal.fingerprint(retention_path); journal.atomic_json(control, c)
    journal.atomic_json(OUTPUT/'REPORT_VALIDATION.json', dict(at=journal.utc_now(), html=journal.fingerprint(journal.REPORT),
        checks=qa, acceptance=c['current_acceptance'], queries=journal.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),
        retention=journal.fingerprint(retention_path), report_code=journal.fingerprint(Path(__file__)),
        report_tests=journal.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),
        report_test_code=journal.fingerprint(journal.REPO/'neurooracle/tests/test_kg_measurement_reuse_report.py')))
    print('R38_PUBLISHED', dict(**counts, groups=len(groups), default_groups=len(default), removed_bytes=retention['bytes_removed']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending', action='store_true')
    pending() if parser.parse_args().pending else main()
