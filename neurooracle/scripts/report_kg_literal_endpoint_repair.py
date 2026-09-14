"""Update the same overnight HTML and retire superseded current derivatives."""
import argparse
from collections import Counter
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from report_current_kg import check_html
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=journal.OUTPUT/"round37_relation_scope"
PREVIOUS=journal.OUTPUT/"round36_explicit_pmid_provenance"


def pending():
    require(not (OUTPUT/"CURRENT_ACCEPTANCE.json").exists(),"already adopted")
    plan=journal.read_json(OUTPUT/"PLAN.json")
    section=f'''<section id="literal-endpoint-pending"><h2>R37 正在验收：完整影像指标与错误基因指向</h2>
<p>492项回归通过。计划修复{plan['changed_claims']}条claim、{plan['changed_edges']}条关联边的端点；补齐{plan['added_literal_nodes']}个完整指标节点，复用{plan['reused_existing_nodes']}个现有节点。只新增普通概念节点，不新增metadata字段、claim或科学断言边。</p>
<p>完整名称严格保留大小写、左右侧、联合指标和限定范围。普通指标节点只表示原claim的完整表达，不冒充UMLS身份。当前仍以R36验收为准；独立全图与实际查询通过后才标为本批完成。</p>
<p>488处剩余基因指向含类型、分子机制、名称范围及引用待核，并非488条已证明错误。21条语义、1条结构待核暂不自动改写。</p>
<p>{journal.link(OUTPUT/'RUN_STATE.json','原子进度')} · {journal.link(OUTPUT/'PLAN.json','完整名称与引用修复计划')} · {journal.link(OUTPUT/'TEST_RESULTS.xml','492项测试')}</p></section>'''
    page=journal.REPORT.read_text(encoding="utf8")
    if 'id="literal-endpoint-pending"' in page:
        page=re.sub(r'<section id="literal-endpoint-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-explicit-pmid">'; require(marker in page,"HTML insertion point missing")
        page=page.replace(marker,section+'\n'+marker,1)
    check_html(page); journal.atomic_text(journal.REPORT,page)
    print("R37_PENDING_LOGGED; R36 remains accepted")


def comparison(current,previous):
    old_members={m['claim_id'] for g in previous for m in g['members']}
    new_members={m['claim_id'] for g in current for m in g['members']}
    old_sets={tuple(sorted(m['claim_id'] for m in g['members'])) for g in previous}
    new_sets={tuple(sorted(m['claim_id'] for m in g['members'])) for g in current}
    return dict(previous_shared_claims=len(old_members),current_shared_claims=len(new_members),
        newly_shared_claim_ids=sorted(new_members-old_members),no_longer_shared_claim_ids=sorted(old_members-new_members),
        unchanged_member_sets=len(old_sets&new_sets),changed_or_added_member_sets=len(new_sets-old_sets),
        changed_or_removed_member_sets=len(old_sets-new_sets),
        no_longer_shared_groups=[dict(relation_id=g.get('id'),subject_name=g.get('subject_name'),predicate=g.get('predicate'),
            object_name=g.get('object_name'),claim_ids=sorted(m['claim_id'] for m in g['members']),
            source_keys=sorted({m['paper_key'] for m in g['members'] if 'paper_key' in m}))
            for g in previous if any(m['claim_id'] in old_members-new_members for m in g['members'])],
        interpretation="counts of full endpoint identity groups; no scientific consensus or independent cohort inference")


def main():
    c=journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None,"writer active")
    require(journal.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],"R37 not current")
    receipt=journal.read_json(c['current_acceptance']['path']); state=journal.read_json(OUTPUT/'BUILD_STATE.json')
    from xml.etree import ElementTree as ET
    report_tests=OUTPUT/'REPORT_TEST_RESULTS.xml'
    suite=ET.parse(report_tests).getroot().find('testsuite')
    require(int(suite.get('tests'))==4 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report comparison tests failed')
    plan=journal.read_json(OUTPUT/'PLAN.json'); census=journal.read_json(c['current_paper_census']['path'])
    query=journal.read_json(OUTPUT/'QUERY_CONTRACT_VALIDATION.json')
    require(receipt['query_contract_validation']==journal.fingerprint(OUTPUT/'QUERY_CONTRACT_VALIDATION.json'),"query contract binding differs")
    groups=list(current_shared_relations(journal.OUTPUT/'CAMPAIGN.json'))
    default=list(find_shared_relations(journal.OUTPUT/'CAMPAIGN.json',minimum_papers=2))
    inclusive=list(find_shared_relations(journal.OUTPUT/'CAMPAIGN.json',minimum_papers=2,include_retracted=True))
    require((len(groups),len(default),len(inclusive))==(query['raw_groups'],query['default_minimum_two_source_groups'],query['inclusive_minimum_two_source_groups']),"published query changed")
    compare_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not compare_path.exists():
        old=state['baseline']['current_shared_relations']; require(journal.fingerprint(old['path'])==old,"previous index changed")
        journal.atomic_json(compare_path,dict(at=journal.utc_now(),graph=c['current_graph'],**comparison(groups,rows(old['path']))))
    compared=journal.read_json(compare_path)
    held=rows(c['current_gene_holds']['path']); audit=journal.read_json(receipt['identity_audit']['path'])
    coverage=rows(c['current_coverage']['path'])
    require(len(held)==plan['held_endpoints'] and len({r['claim_id'] for r in held})==c['pending_gene_link_claims'],"current gene queue differs")
    retention_path=OUTPUT/'RETENTION_RESULT.json'
    if not retention_path.exists():
        old_census=journal.read_json(state['baseline']['current_paper_census']['path'])['database']
        targets=[old_census,*[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        allowed={PREVIOUS/n for n in ('CURRENT_PAPER_CENSUS.sqlite','CURRENT_SHARED_RELATIONS.jsonl','CURRENT_PAPER_ISSUES.jsonl','CURRENT_METADATA_COVERAGE.jsonl')}
        current={Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        for fp in targets:
            path=Path(fp['path']).resolve(strict=True)
            require(path in {p.resolve() for p in allowed} and path.is_relative_to(journal.OUTPUT.resolve()) and path not in current and path.is_file(),"unsafe obsolete derivative target")
            journal.guards(fp); require(sha256(path)==fp['sha256'],"obsolete derivative changed")
        for fp in targets: Path(fp['path']).unlink()
        journal.atomic_json(retention_path,dict(status='COMPLETED',at=journal.utc_now(),removed=targets,bytes_removed=sum(fp['bytes'] for fp in targets),
            source_kg_backups=0,record_preimages_saved=False,original_archive_and_public_evidence_deleted=False,
            recoverability='Old derivatives have no retained copies. Current derivatives can be rebuilt from current KG; no historical KG rollback.'))
    retention=journal.read_json(retention_path)
    at=journal.local_time(receipt['at']); counts=c['counts']; relation=c['relation_evidence_counts']
    report=f'''# R37：718条完整影像指标脱离错误基因指向

已于{at}（香港）完成当前工作KG原子替换、独立全图验收与实际查询。不是模型、KGE或训练；正式full_v2没有同步。

## 一批完成

- 修复718条claim的718处端点ID，同步908条边记录。基因FANCE、VCP节点及其其他引用保持原样；没有删除claim或科学证据。
- 688个原图不存在的完整指标表达使用普通CLM_CONCEPT节点，复用1个严格同名、同类型、引用范围一致的现有节点。同名指标在本批内共用同一个节点，不按论文/claim建节点；总节点仅增加约0.027%。
- 节点保留完整原名，大小写、左右侧、联合指标和限定条件不折叠；新增节点无metadata内容、UMLS编码或外部身份断言。它们是原claim指标表达的存储节点，不是声称新建了688个本体实体。
- 190条claim原有科学关系边，528条只通过两条about边连接端点；保持各自原有表达，未为后一类制造额外科学边。
- 全量扫描完整名称和别名、所有关联引用；只修复明确IMAGING_MARKER且名称为具体测量、非分子/方法混合的记录。未知类型不补猜，分子机制不重分类，模糊同义词不强行归并。

## 当前情况

- 节点{counts['nodes']:,}（+688）；claim{counts['claims']:,}、边{counts['edges']:,}均不变。
- 节点metadata键并集{len(receipt['metadata_key_unions']['nodes'])}，边{len(receipt['metadata_key_unions']['edges'])}；本批新增metadata字段0。{len(coverage)}行覆盖统计完整重算，未通过补填制造100%覆盖。
- 全部细粒度关系组{relation['all_fine_grained_relation_groups']:,}，共享组{relation['shared_groups']:,}、共享成员{relation['indexed_claims']:,}；至少两个已核实论文来源的组默认{len(default)}，包含已确认撤稿来源时{len(inclusive)}。
- {len(compared['newly_shared_claim_ids'])}条claim进入共享组，{len(compared['no_longer_shared_claim_ids'])}条暂不再与原组共享；这不是删claim或新增论文。具体是同一已核实论文PMID39829963的两组双重表述：一条明确影像类型已修复，另一条类型尚未确认仍待核，因修复进度不同暂时拆开。不能据此断定研究关系不同；优先接续核实这两条，而不是把减少共享一律当成质量收益。多论文来源组没有减少。
- 已核实来源claim{audit['claim_status']['verified']:,}，书目冲突{audit['claim_status']['conflict']:,}，未核实{audit['claim_status']['unverified']:,}；本批不改论文身份和来源数量。论文不同不等于实验相互独立。
- 连通分量{receipt['connected_components']:,}、孤立节点{receipt['isolated_nodes']:,}、自环{receipt['self_loops']:,}。错误基因中心分离会改变连通结构，不应为了连通率保留错误边。

## 尚未完成

488处仍连到这两个基因的端点、涉及{len({r['claim_id'] for r in held})}条claim。它们是待核队列，不是全部已证明错误；原因可能包括类型不清、名称缺少具体测量、分子机制、现有名称候选不唯一或混合引用。当前队列已重绑当前claim哈希。

21条语义与1条结构待核继续保留。已收集23篇相关PubMed记录和NAcc研究PMC全文，尚未应用科学内容修订。例如成人双侧与青少年左侧NAcc结论不可合并为跨年龄统一双侧结论；正式科学修订需另批核对。[原研究](https://pubmed.ncbi.nlm.nih.gov/31756730/)。

这批处理的是“不同指标错误共用基因节点”，而非只处理重复拆分。[VCP官方定义](https://medlineplus.gov/genetics/gene/vcp/)也明确其为编码蛋白的基因。完整原始指标表达不应冒充该基因；但本批没有宣称每篇研究的科学结论均已重新验证。

## 验收与单版本保留

492项回归、4项查询契约和4项报告比较测试通过（合计500项不同测试）。源图完整SHA、独立候选全图、全记录逆向摘要、每条获批claim/边反向重建的原哈希、全体905,184条claim与当前书目/共享成员普查、所有实体和论文/出版证据、全量metadata覆盖、UMLS明细及当前普查完整SHA均验证。

实际共享查询需要3个能力标志；它们对应的实体、论文和出版状态核验已在冻结验收代码里执行，本批另用小型查询凭据发布并完成实际查询测试，没有为控制字段重扫或重写KG。此前控制层metadata键数121与验收120的不一致也已纠正。

旧R36普查数据库、共享索引、书目待核与覆盖表共{retention['bytes_removed']:,}字节已删除，无旧版本回退副本；当前KG及派生数据、原始档案、公共证据保留。没有历史KG或记录前像。

当前KG SHA-256：{c['current_graph']['sha256']}；{c['current_graph']['bytes']:,}字节。
使用incremental-integrity-checks：进度读取原子状态与可信指纹，构建/验收执行完整核验；控制字段沿用已完成的可信证据，不为状态汇报重复扫描巨型文件。整体目标尚未完成，夜间窗口仍到2026-09-09 10:00HKT。
'''
    next_plan='''# R38接续：剩余端点与科学陈述范围

R37已完成718端点与908边引用修复；勿重跑R37或更早已采用构建。以CAMPAIGN.json为准。

1. 当前图仍在R23路径；当前census/index/coverage/holds/publication review在R37，论文registry仍R34，实体terms仍R30。R36旧派生文件已删除。
2. 优先CURRENT_GENE_ENDPOINT_HOLDS.jsonl的488处端点；316处名称具体测量不充分不等于没有对应指标。可补充原始输入和论文定义，但不按短词/多数票/病例数合并，不把未知类型猜为影像。已有7处同名多候选和2处混合引用可按完整事件与引用闭合单独核对。
   首先核对CLM:1b4579c82247与CLM:951f2bb246a3，均PMID39829963；其完整名称分别为frontoparietal network cortical surface area与cingulo-opercular network cortical surface area。对应CASE1MAN:39829963:286与:284已经修复，因此原两组同篇共享暂时拆开，不能解释成证明两条不同研究。用本篇原始方法/结果与完整原始输入核实缺失类型后决定引用，不按另一条的类型多数票照抄。
3. 21语义、1frailty结构与额外NAcc范围问题继续待核；R37/SCIENTIFIC_REVIEW_TARGETS.jsonl、PUBLIC_SOURCE_FETCH.json及PMC6969163.xml提供只读出处。任何修订先逐段核对原文，不 blanket翻转negated、不把设计/假说写成结果、不用临床组内改善替代组间因果差异。
4. Frailty同名候选中有physical frailty与完整名不一致、既有DRUG类型错误；只修重复引用不能等同于全局实体合并。NAcc条目同时有错误书目和年龄/左右侧语境，不能只改标题或猜合成ID后缀。
5. 与初始动机一致：关系复用与逐篇claim应分开统计。已有部分历史批次每篇一条claim抽取上限，不外推全图，不重跑抽取/训练。不为共享率提升损失完整联合指标、独立论文、否定或研究条件。
6. 同一HTML持续记录。活动writer只检查/接续；不改冻结代码、不启动第二writer。只保留当前KG，无模型训练或正式full_v2同步。窗口到10:00HKT。目标控制blocked已另记录，不绕过数据库或把整体目标假标完成。
'''
    journal.atomic_text(OUTPUT/'REPORT.md',report); journal.atomic_text(OUTPUT/'NEXT_BATCH_PLAN.md',next_plan)
    section=f'''<section id="current-literal-endpoints"><h2>当前 R37：718条完整指标与908条边引用已修复</h2>
<div class="notice">492项回归、4项查询契约测试、独立全图和实际查询通过。原始论文、科学文本、否定和研究条件保留。无模型训练或正式同步。</div>
<table><tr><th>项目</th><th>当前结果</th></tr>
<tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}；只增加688个完整指标节点</td></tr>
<tr><td>本批修复</td><td>718处错误基因指向、908条关联边；复用1个现有节点</td></tr>
<tr><td>metadata键并集</td><td>节点120 / 边36；不新增metadata字段</td></tr>
<tr><td>共享组 / 共享成员</td><td>{relation['shared_groups']:,} / {relation['indexed_claims']:,}；保留全部原始claim</td></tr>
<tr><td>至少两个已核实来源的组</td><td>默认{len(default)}；显式包含已确认撤稿来源{len(inclusive)}</td></tr>
<tr><td>剩余基因指向</td><td>488处、{len({r['claim_id'] for r in held})}条claim待核；并非全部已确认错误</td></tr></table>
<p>这里修的是不同指标错误共用FANCE/VCP基因节点。严格复用完整同名指标；缺少节点时一个完整名称共用一个普通概念节点，不按论文创建。新增节点没有metadata内容或UMLS身份断言；节点增幅约0.027%。不能为了少几个节点保留错误的基因关系。</p>
<p>本批{len(compared['newly_shared_claim_ids'])}条进入共享组、{len(compared['no_longer_shared_claim_ids'])}条暂不再共享。具体为PMID39829963同篇的两组表述：一条已修、另一条类型未核实而暂时拆开，列为优先接续项，不表示证明是不同研究。多论文来源组未减少；不删claim、不增论文，不把来源数当独立实验数或共识。</p>
<p class="warning">仍有21条语义、1条结构及书目冲突。已收集相关原文但未擅自修改科学结论，整体目标尚未完成。</p>
<p>旧R36派生资料{retention['bytes_removed']:,}字节已删除，无历史KG或记录回退副本；当前数据、原档案与公共证据保留。当前SHA-256：<code>{c['current_graph']['sha256']}</code></p>
<p>{' · '.join(journal.link(OUTPUT/name,label) for name,label in [('REPORT.md','完整说明及公开出处'),('CURRENT_ACCEPTANCE.json','当前数据验收'),('CURRENT_RUNTIME_ACCEPTANCE.json','当前运行时'),('PLAN.json','完整名称与引用计划'),('RELATION_MEMBERSHIP_COMPARISON.json','共享成员变化'),('CURRENT_GENE_ENDPOINT_HOLDS.jsonl','488处待核'),('CURRENT_METADATA_COVERAGE.jsonl','当前覆盖表'),('CENSUS.json','当前普查'),('QUERY_CONTRACT_VALIDATION.json','实际查询与读取契约'),('TEST_RESULTS.xml','492项回归'),('QUERY_CONTRACT_TEST_RESULTS.xml','4项查询测试'),('NEXT_BATCH_PLAN.md','下一批计划')])}</p></section>'''
    page=journal.REPORT.read_text(encoding='utf8')
    page=re.sub(r'<section id="literal-endpoint-pending">.*?</section>','',page,count=1,flags=re.S)
    if 'id="current-literal-endpoints"' in page:
        page=re.sub(r'<section id="current-literal-endpoints">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-explicit-pmid">'; require(marker in page,'current insertion missing')
        page=page.replace(marker,section+'\n'+marker,1)
        page=re.sub(r'(<section id="current-explicit-pmid">.*?</section>)',lambda m:'<details><summary>R36 历史批次：显式PMID与引用修复</summary>'+m[1].replace('当前 R36：','历史 R36：',1)+'</details>',page,count=1,flags=re.S)
    for old in retention['removed']:
        relative=Path(old['path']).relative_to(journal.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已替换，仅保留R37当前文件</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R37工作图，正式full_v2未同步。</p>',page,count=1)
    page=page.replace('R36已完成整体验收；后续批次通过验收后才替换当前工作图。','R37已完成整体验收；后续批次通过验收后才替换当前工作图。')
    qa=check_html(page); journal.atomic_text(journal.REPORT,page)
    journal.atomic_text(journal.OUTPUT/'HANDOFF.md','# Current data and runtime — R37 literal endpoints\n\n'+report+'\n'+next_plan)
    log_path=journal.OUTPUT/'WORK_LOG.json'; log=journal.read_json(log_path)
    if not any(e.get('id')=='r37-literal-endpoints-20260909' for e in log['events']):
        log['events'].append(dict(id='r37-literal-endpoints-20260909',at=receipt['at'],status='completed',title='R37完整影像指标与错误基因指向',
            body=f'718claim、908边引用，688完整指标节点；492+4测试与全图/实际查询通过。剩488处端点待核。',
            changes='无科学文本或来源变更，无metadata新增，无旧KG/记录前像；旧当前派生资料已删除。',
            artifacts=[str(OUTPUT/n) for n in ('REPORT.md','CURRENT_ACCEPTANCE.json','CENSUS.json','NEXT_BATCH_PLAN.md')]))
        journal.atomic_json(log_path,log)
    journal.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    require(journal.read_json(journal.OUTPUT/'CAMPAIGN.json')==c,'campaign advanced')
    c['retention_result']=journal.fingerprint(retention_path); journal.atomic_json(journal.OUTPUT/'CAMPAIGN.json',c)
    journal.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=journal.utc_now(),html=journal.fingerprint(journal.REPORT),checks=qa,
        acceptance=c['current_acceptance'],queries=journal.fingerprint(OUTPUT/'QUERY_CONTRACT_VALIDATION.json'),retention=journal.fingerprint(retention_path),
        report_code=journal.fingerprint(Path(__file__)),report_tests=journal.fingerprint(report_tests),
        report_test_code=journal.fingerprint(journal.REPO/'neurooracle/tests/test_kg_literal_endpoint_report.py')))
    print('R37_PUBLISHED',dict(nodes=counts['nodes'],claims=counts['claims'],edges=counts['edges'],default_groups=len(default),removed_bytes=retention['bytes_removed']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--pending',action='store_true')
    pending() if parser.parse_args().pending else main()
