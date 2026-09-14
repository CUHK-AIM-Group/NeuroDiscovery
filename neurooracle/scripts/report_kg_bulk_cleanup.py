"""Publish R41 exact cleanup results and retire only replaced R40 derivatives."""
from collections import Counter
from pathlib import Path
import json
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
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations

OUTPUT=j.OUTPUT/'round41_bulk_cleanup'
PREVIOUS=j.OUTPUT/'round40_scoped_structure'


def cleanup_summary(removals):
    total=Counter()
    for key,n in removals.items():
        require(type(n) is int and n>=0,'invalid removal count')
        reason=key.split(':',1)[0]
        require(reason in {'empty_hint','empty_raw_stats','canonical_science_duplicate','canonical_audit_duplicate'},'unknown removal category')
        total[reason]+=n
    return dict(total)


def coverage_summary(before,after,removed):
    def lookup(records):return {r['field']:r for r in records if r['scope']=='node/claim'}
    old,new=lookup(before),lookup(after)
    scientific=[field for field in old if field.startswith('metadata.evidence.') or field.startswith('metadata.source_paper.') or field in (
        'metadata.raw_text','metadata.subject_id','metadata.object_id','metadata.subject_name','metadata.object_name','metadata.predicate','metadata.negated')]
    require(all(old[f]==new.get(f) for f in scientific),'scientific/paper field coverage changed')
    prefix='metadata.metadata.'
    a=sum(r['present'] for f,r in old.items() if f.startswith(prefix));b=sum(r['present'] for f,r in new.items() if f.startswith(prefix))
    require(a-b==removed,'coverage does not reproduce all removals')
    n=new['metadata.raw_text']['denominator']
    return dict(source_nested_occurrences=a,current_nested_occurrences=b,removed_occurrences=a-b,
        source_average_nested_fields=a/n,current_average_nested_fields=b/n,claims=n,
        p_value_nonempty_pct=new['metadata.evidence.p_value']['nonempty_pct'],
        effect_size_nonempty_pct=new['metadata.evidence.effect_size']['nonempty_pct'],
        sample_size_nonempty_pct=new['metadata.evidence.sample_size']['nonempty_pct'],
        unchanged_scientific_and_paper_coverage_fields=len(scientific))


def pending():
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['active_process'] and c['active_process']['kind']=='bulk_cleanup','no active R41 writer')
    p=j.read_json(OUTPUT/'PLAN.json')
    section=f'''<section id="bulk-cleanup-pending"><h2>R41 正在集中清理 metadata 与反向相关索引</h2>
<p>{p['test_counts']['tests']}项联合回归已通过。空提示、空raw_stats和完全相同的规范副本集中精简；3组反向correlates_with表达归入同一关系索引，所有原claim和边保留。</p>
<p>当前KG仍以已验收R40为准。统计空值、真实数值、条件、原文及审核封印不改；449处端点不按名称猜测强并。无模型、训练、正式同步或定时任务重建。</p>
<p>{j.link(OUTPUT/'PLAN.json','限定计划')} · {j.link(OUTPUT/'RUN_STATE.json','实时构建状态')} · {j.link(OUTPUT/'SOURCE_INSPECTION.json','全图核对')} · {j.link(OUTPUT/'TEST_RESULTS.xml','联合回归')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    if 'id="bulk-cleanup-pending"' in page:page=re.sub(r'<section id="bulk-cleanup-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    else:
        marker='<section id="current-scoped-structure">';require(marker in page,'R40 report anchor missing')
        page=page.replace(marker,section+'\n'+marker,1)
    check_html(page);j.atomic_text(j.REPORT,page);print('R41_PENDING_PUBLISHED',flush=True)


def main():
    control=j.OUTPUT/'CAMPAIGN.json';c=j.read_json(control)
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer still active')
    require(j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json')==c['current_acceptance'],'R41 is not current')
    receipt=j.read_json(c['current_acceptance']['path']);state=j.read_json(OUTPUT/'BUILD_STATE.json')
    require(receipt['status']=='CURRENT_BULK_CLEANUP_APPLIED' and receipt['relation_grouping']==POLICY,'wrong adoption/index policy')
    for fp in [c['current_runtime_acceptance'],c['current_gene_holds'],c['current_scope_findings'],c['current_coverage'],c['current_paper_census'],
        receipt['tests'],receipt['provenance_plan'],*receipt['code']]:require(j.fingerprint(fp['path'])==fp,'current evidence/code binding differs')
    runtime=j.read_json(c['current_runtime_acceptance']['path']);require(runtime['acceptance']==c['current_acceptance'],'runtime receipt differs')
    suite=ET.parse(OUTPUT/'REPORT_TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(suite.get('tests'))==3 and all(int(suite.get(k))==0 for k in ('failures','errors','skipped')),'report tests failed')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,h in state['held_hashes'].items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(h,),'current hold binding differs')
    db.close()
    groups=list(current_shared_relations(control));default=list(find_shared_relations(control,minimum_papers=2))
    inclusive=list(find_shared_relations(control,minimum_papers=2,include_retracted=True))
    require(len(groups)==c['relation_evidence_counts']['shared_groups'],'actual shared query differs')
    by_id={g['id']:g for g in groups};reverse=j.read_json(OUTPUT/'REVERSED_CORRELATION_GROUPS.json');query_proofs=[]
    for proposed in reverse['groups']:
        rid=relation_id(json.loads(proposed['key']));g=by_id[rid]
        require({m['claim_id'] for m in g['members']}=={m['claim_id'] for m in proposed['members']},'reversed group members differ')
        kw=dict(predicate='correlates_with',minimum_papers=1,verified_papers_only=False,include_retracted=True)
        forward={r['id'] for r in find_shared_relations(control,subject=g['subject_name'],object_name=g['object_name'],**kw)}
        backward={r['id'] for r in find_shared_relations(control,subject=g['object_name'],object_name=g['subject_name'],**kw)}
        require(rid in forward and forward==backward,'bidirectional correlation query differs')
        query_proofs.append(dict(relation_id=rid,claim_count=g['claim_count'],source_keys=g['paper_count'],verified_sources=g['verified_paper_count'],both_query_directions_passed=True))
    j.atomic_json(OUTPUT/'QUERY_VALIDATION.json',dict(at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
        shared_groups=len(groups),default_two_verified_sources=len(default),including_retracted=len(inclusive),
        reverse_groups=query_proofs,current_held_claim_hashes_checked=len(state['held_hashes']),no_scientific_consensus_inferred=True))
    comp_path=OUTPUT/'RELATION_MEMBERSHIP_COMPARISON.json'
    if not comp_path.exists():
        old=state['baseline']['current_shared_relations'];require(j.fingerprint(old['path'])==old,'old shared index changed')
        j.atomic_json(comp_path,dict(at=j.utc_now(),graph=c['current_graph'],**comparison(groups,rows(old['path']))))
    comp=j.read_json(comp_path);require(comp['graph']==c['current_graph'],'comparison not current')
    coverage_path=OUTPUT/'COVERAGE_COMPARISON.json'
    if not coverage_path.exists():
        old=state['baseline']['current_coverage'];require(j.fingerprint(old['path'])==old,'old coverage changed')
        j.atomic_json(coverage_path,dict(graph=c['current_graph'],**coverage_summary(rows(old['path']),rows(c['current_coverage']['path']),receipt['removed_field_occurrences'])))
    cov=j.read_json(coverage_path);require(cov['graph']==c['current_graph'],'coverage comparison not current')
    totals=cleanup_summary(receipt['removals']);require(sum(totals.values())==receipt['removed_field_occurrences'],'removal aggregate differs')
    ret_path=OUTPUT/'RETENTION_RESULT.json'
    if not ret_path.exists():
        targets=[j.read_json(state['baseline']['current_paper_census']['path'])['database'],
            *[state['baseline'][k] for k in ('current_shared_relations','current_paper_issues','current_coverage')]]
        current={Path(c[k]['path']).resolve() for k in ('current_graph','current_detail_store','current_shared_relations','current_paper_issues','current_coverage','current_entity_terms','current_paper_identities')}
        current.add(Path(census['database']['path']).resolve())
        paths=validate_retirement_paths(targets,current,previous=PREVIOUS,root=j.OUTPUT)
        for fp,path in zip(targets,paths):j.guards(fp);require(sha256(path)==fp['sha256'],'old derivative changed')
        require(j.read_json(control)==c,'campaign advanced')
        for path in paths:path.unlink()
        j.atomic_json(ret_path,dict(status='COMPLETED',at=j.utc_now(),removed=targets,bytes_removed=sum(fp['bytes'] for fp in targets),
            graph_backups=0,record_preimages_saved=False,original_archives_and_public_evidence_deleted=False,
            recoverability='No old copies retained. Current derivatives can be rebuilt from current KG; not a historical rollback.'))
    ret=j.read_json(ret_path);counts=c['counts'];rel=c['relation_evidence_counts'];at=j.local_time(receipt['at'])
    test_count=receipt['test_counts']['tests'];held=c['pending_gene_link_endpoint_events']
    report=f'''# R41：批量精简重复metadata与反向相关索引

{at}（香港时间）应用到当前工作KG。{test_count}项联合回归、3项报告测试、独立全图验收与实际查询通过。仅KG；没有模型、训练、重新抽取、正式full_v2同步或定时任务重建。

## 实际修改

{receipt['changed']['claim_nodes']:,}条claim的存储得到精简，共省去{receipt['removed_field_occurrences']:,}处字段：空提示{totals.get('empty_hint',0):,}处、空raw_stats对象{totals.get('empty_raw_stats',0):,}处、与外层科学字段完全一致的内层副本{totals.get('canonical_science_duplicate',0):,}处、与scope_reaudit规范值完全一致的审核副本{totals.get('canonical_audit_duplicate',0):,}处。审核对象本身、来源批次、旧ID和不同值没有一概删除。

当前节点{counts['nodes']:,}、边{counts['edges']:,}、claim{counts['claims']:,}，三项数量不变。全部边、非claim节点和节点ID不变。图文件减少{receipt['graph_bytes_reduced']:,}字节（{receipt['graph_bytes_reduced']/1024**2:.2f} MiB逻辑大小），不是新增论文或减少科学证据。

节点/边metadata顶层字段并集仍为{c['node_metadata_field_union']}/{c['edge_metadata_field_union']}；这并不是每条记录都有这么多字段。claim内层metadata平均字段数由{cov['source_average_nested_fields']:.2f}降至{cov['current_average_nested_fields']:.2f}，详见COVERAGE_COMPARISON.json。

## 关系重复

全图53,499条correlates_with中，发现3组端点ID及完整名称相同、只有主客顺序反向的表达，涉及7条claim。新索引采用对称归组；原始7条claim、原始边方向、否定、统计值、条件、文章来源全部保留。predicts、causes、treats等不对称关系不这样处理，is_associated_with也没有顺带合并。

共享关系组从{state['baseline']['relation_evidence_counts']['shared_groups']:,}变为{rel['shared_groups']:,}，成员从{state['baseline']['relation_evidence_counts']['indexed_claims']:,}变为{rel['indexed_claims']:,}。新纳入共享索引{len(comp['newly_shared_claim_ids'])}条claim；没有丢失原共享成员。默认至少两个已核实且未被确认撤稿来源的组{len(default)}，包括确认撤稿时{len(inclusive)}。其中两组反向表达来自同一篇文章，不算新的跨论文证据；文章数也不等于独立实验数。

## 未强删的内容

p值、效应量、样本量非空率仍为{cov['p_value_nonempty_pct']:.2f}%、{cov['effect_size_nonempty_pct']:.2f}%、{cov['sample_size_nonempty_pct']:.2f}%。它们的null值参与既有审核证据摘要，直接省略会改变审核输入；本轮保持原样，没有补值或重新签发审核。真实值、未知扩展字段、类型不同或内容冲突的副本均保留。科学/书目字段覆盖逐项不变。

{held}处基因端点待核、既有21条语义与1条结构问题、qMRI发作次数/症状严重程度、BACE1浓度/pathway、ACC截断证据、海马范围和明细依赖继续保留；队列可能重叠，也不穷尽全图。不得说整体KG优化或跨论文归并已经全部完成。

## 验收、读写与保留

源图完整SHA验证，候选图独立全量读取；科学字段及原审核对象摘要一致，所有非claim节点/边记录摘要一致，当前claim/论文/共享成员普查和全量覆盖一致。明细库与新普查完整SHA通过，悬空端点、重复节点ID与失效边所有者均为0。三个归组的正反查询均通过，保存/重载不重新长出已精简字段；原默认有向归组兼容保留，只有声明新策略的图采用对称索引。

旧R40四项派生文件共{ret['bytes_removed']:,}字节已删除，无旧KG或记录前像。当前KG、明细和索引保留，当前派生资料可重建，没有旧版本回退副本。原始文献、公开来源及正式full_v2未修改。

使用incremental-integrity-checks：进度核对复用可信状态，构建和采用边界执行完整验证。当前KG SHA-256：{c['current_graph']['sha256']}。
'''
    next_plan='''# 下一批：继续科学范围与实体身份收口

以CAMPAIGN.json为准；当前工作KG仍在R23路径，当前索引、普查、覆盖和问题清单均在R41。terms仍在R30，paper registry仍在R34。R40四项大派生资料已删除，不重跑历史构建。

当前图采用relation_grouping对称相关策略，保留完整端点名称；不要把correlates_with规则扩到因果、预测或更宽泛关联。metadata精简的空提示和完全相同副本规则已接入保存路径、证据去重和旧版比较读者。冲突、唯一值、统计null和审核封印仍保留。

优先集中核实CURRENT_SCOPE_FINDINGS.json中的五条来源范围问题及449处端点，不能把类型缺失直接认定为错误。对物理锚点归并/删除要同时处理明细库依赖。需要更进一步省略统计空值时，先设计不破坏既有审核输入的明确存储契约，不能悄悄改变旧审核含义。

不运行模型、KGE或训练；不重建夜间任务；不擅自同步正式full_v2。只有当前工作版本，无回退前像；整体优化未全部完成。
'''
    j.atomic_text(OUTPUT/'REPORT.md',report);j.atomic_text(OUTPUT/'NEXT_BATCH_PLAN.md',next_plan)
    section=f'''<section id="current-bulk-cleanup"><h2>当前R41：批量metadata精简和反向相关索引已应用</h2>
<div class="notice">{receipt['changed']['claim_nodes']:,}条claim精简，省去{receipt['removed_field_occurrences']:,}处空提示或相同副本；节点、边、claim数量全部不变。{test_count}+3项测试、独立全图验收及实际查询通过。</div>
<table><tr><th>项目</th><th>当前结果</th></tr><tr><td>节点 / claim / 边</td><td>{counts['nodes']:,} / {counts['claims']:,} / {counts['edges']:,}</td></tr>
<tr><td>claim内层metadata平均字段数</td><td>{cov['source_average_nested_fields']:.2f} → {cov['current_average_nested_fields']:.2f}</td></tr>
<tr><td>共享关系组 / 成员</td><td>{rel['shared_groups']:,} / {rel['indexed_claims']:,}；3组反向相关表达收拢，7条原claim均保留</td></tr>
<tr><td>图文件减少</td><td>{receipt['graph_bytes_reduced']/1024**2:.2f} MiB（逻辑文件大小）</td></tr>
<tr><td>metadata顶层字段种类</td><td>节点{c['node_metadata_field_union']} / 边{c['edge_metadata_field_union']}，不是每条记录的字段数</td></tr></table>
<p>统计值、原文、条件、审核封印和全部原始边不变；p值/效应量/样本量非空率仍为{cov['p_value_nonempty_pct']:.2f}%/{cov['effect_size_nonempty_pct']:.2f}%/{cov['sample_size_nonempty_pct']:.2f}%。</p>
<p class="warning">还有{held}处端点及已登记的科学范围问题待核，整体优化未完成。无模型或训练，正式full_v2未同步，夜间任务没有重建。</p>
<p>{j.link(OUTPUT/'REPORT.md','完整说明')} · {j.link(OUTPUT/'CURRENT_ACCEPTANCE.json','当前验收')} · {j.link(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json','当前运行时')} · {j.link(OUTPUT/'QUERY_VALIDATION.json','实际正反查询')} · {j.link(OUTPUT/'COVERAGE_COMPARISON.json','覆盖与字段负担')} · {j.link(OUTPUT/'CURRENT_METADATA_COVERAGE.jsonl','当前全量覆盖')} · {j.link(OUTPUT/'CURRENT_SCOPE_FINDINGS.json','当前范围待核')} · {j.link(OUTPUT/'NEXT_BATCH_PLAN.md','接续计划')}</p></section>'''
    page=j.REPORT.read_text(encoding='utf8')
    page=re.sub(r'<section id="bulk-cleanup-pending">.*?</section>',lambda _:section,page,count=1,flags=re.S)
    require('id="current-bulk-cleanup"' in page,'pending HTML section missing')
    page=re.sub(r'(<section id="current-scoped-structure">.*?</section>)',r'<details><summary>R40历史批次：结构和MRI端点修复</summary>\1</details>',page,count=1,flags=re.S)
    for fp in ret['removed']:
        relative=Path(fp['path']).relative_to(j.REPO).as_posix()
        page=re.sub(rf'<a href="{re.escape(relative)}">.*?</a>','<span>旧派生资料已替换，仅保留R41当前文件</span>',page)
    page=re.sub(r'<h1>KG 当前状态与工作记录</h1><p>更新：.*?</p>',f'<h1>KG 当前状态与工作记录</h1><p>更新：{at}（香港时间）。当前R41工作图，正式full_v2未同步。</p>',page,count=1)
    qa=check_html(page);j.atomic_text(j.REPORT,page);j.atomic_text(j.OUTPUT/'HANDOFF.md',report+'\n'+next_plan)
    log=j.read_json(j.OUTPUT/'WORK_LOG.json');eid='r41-bulk-cleanup'
    if not any(e.get('id')==eid for e in log['events']):
        log['events'].append(dict(id=eid,at=receipt['at'],status='completed',title='R41批量metadata精简与反向相关索引',
            body=f"{receipt['changed']['claim_nodes']:,}条claim省去{receipt['removed_field_occurrences']:,}处空/重复字段；3组反向相关索引归组。",
            changes=f"{test_count}+3测试及独立全图通过；所有节点ID和边不变，科学/审核对象原样保留，旧派生资料删除。",
            artifacts=[str(OUTPUT/f) for f in ('REPORT.md','CURRENT_ACCEPTANCE.json','COVERAGE_COMPARISON.json','QUERY_VALIDATION.json')]))
        j.atomic_json(j.OUTPUT/'WORK_LOG.json',log)
    require(j.read_json(control)==c,'campaign advanced');j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c['retention_result']=j.fingerprint(ret_path);j.atomic_json(control,c)
    j.atomic_json(OUTPUT/'REPORT_VALIDATION.json',dict(at=j.utc_now(),html=j.fingerprint(j.REPORT),checks=qa,acceptance=c['current_acceptance'],
        runtime=c['current_runtime_acceptance'],queries=j.fingerprint(OUTPUT/'QUERY_VALIDATION.json'),coverage_comparison=j.fingerprint(coverage_path),
        retention=j.fingerprint(ret_path),report_code=j.fingerprint(Path(__file__)),report_tests=j.fingerprint(OUTPUT/'REPORT_TEST_RESULTS.xml'),
        report_test_code=j.fingerprint(j.REPO/'neurooracle/tests/test_kg_bulk_cleanup_report.py')))
    print('R41_PUBLISHED',dict(changed=receipt['changed'],removed_fields=receipt['removed_field_occurrences'],saved_bytes=receipt['graph_bytes_reduced'],
        counts=counts,relations=rel,default_groups=len(default),retired_bytes=ret['bytes_removed'],checks=qa),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pending',action='store_true')
    pending() if p.parse_args().pending else main()
