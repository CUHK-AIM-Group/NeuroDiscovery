"""Cold readback and descriptive delivery for the finite low-hint trial."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import json
import statistics
import subprocess
import sys

from neurobench.runner.exec_dev.common import file_map, load_case, read_json, receipt_digest, sha, write_json

ARMS = ('A', 'B0', 'B1')
LABELS = {'A': 'OFF', 'B0': 'Current', 'B1': 'Candidate'}


def events(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify(root, delivery):
    manifest, gate = read_json(root / 'MANIFEST.json'), read_json(root / 'GATE.json')
    checks, errors = [], []

    def check(name, ok, detail=None):
        record = {'check': name, 'passed': bool(ok)}
        if detail is not None:
            record['detail'] = detail
        checks.append(record)
        if not ok:
            errors.append(record)

    check('manifest_gate', sha(root / 'MANIFEST.json') == gate['manifest_sha256'])
    packet = Path(manifest['packet'])
    check('packet_index', sha(packet / 'INDEX.json') == manifest['packet_index_sha256'])
    qualification = packet.parent / 'qualification_r1/QUALIFICATION.json'
    check('fixture_qualification', read_json(qualification)['passed'] and sha(qualification) == manifest['qualification_sha256'])
    for p, h in manifest['source_bindings'].items():
        check('implementation:' + p, Path(p).is_file() and sha(p) == h)
    for p, h in manifest['p0_bindings'].items():
        check('prior_preserved:' + p, Path(p).is_file() and sha(p) == h)
    check('methods', file_map(root / 'methods') == manifest['methods'])
    for map_name in ('SOURCE_MAP.json', 'EXTRA_SOURCE_MAP.json'):
        for item in read_json(root / 'source_snapshot' / map_name)['files']:
            check('snapshot:' + item['snapshot'], sha(item['snapshot']) == item['sha256'])
    amendment = read_json(root / 'acceptance_roles_r2/AMENDMENT.json')
    check('roles_r2_source', sha(amendment['verifier_source']) == amendment['verifier_sha256'])
    check('roles_r2_public_task', sha(packet / 'public/cases/N07/task.md') == amendment['public_task_sha256'])
    qualified = root / 'acceptance_roles_r2/qualification/QUALIFIED.json'
    check('roles_r2_qualification', read_json(qualified)['passed'] and sha(qualified) == amendment['qualification_sha256'])
    check('finite_run_ended', (root / 'RUN_END.json').is_file())
    check('slot_identity', len(manifest['slots']) == 72 == len({s['id'] for s in manifest['slots']}))
    raw = [read_json(p) for p in sorted((root / 'attempts').glob('*/RESULT.json'))]
    check('all_72_original_slots', len(raw) == 72 and {r['id'] for r in raw} == {s['id'] for s in manifest['slots']})
    contexts, workitems, rows = {}, [], []
    for result in raw:
        slot = result['id']; base = root / 'attempts' / slot
        case = load_case(packet, result['case_id'], manifest['packet_index_sha256'])
        start = read_json(base / 'START.json')
        check(slot + ':start_identity', start['slot']['id'] == slot)
        check(slot + ':input_binding', start['public_inputs'] == case.contract['input_sha256'])
        if case.id not in contexts:
            contexts[case.id] = start['context']
        check(slot + ':common_context', start['context'] == contexts[case.id])
        work = root / 'workspaces' / slot
        check(slot + ':inputs_still_unchanged', file_map(work / 'inputs') == start['public_inputs'])
        check(slot + ':context_still_unchanged', file_map(work / 'context') == start['context'])
        for marker in ('injected_fault', 'recoverable_fault', 'must_block', 'mode=repair', 'mode=probe'):
            check(slot + ':no_public_marker:' + marker, marker not in (base / 'PROMPT.txt').read_text(encoding='utf-8'))
        requests, uses, chars, hits, tool_errors, terminal_errors = [], Counter(), 0, 0, [], []
        for e in events(base / 'events.jsonl'):
            kind = e.get('event')
            if kind == 'model_request':
                requests.append({'body': e['body'], 'endpoint': e['endpoint'], 'raw_usage': None})
                body = e['body']
                if result['model'] == 'gpt-5.6-sol':
                    check(slot + ':requested_high', body.get('reasoning', {}).get('effort') == 'high')
                    check(slot + ':requested_exact_gpt', body.get('model') == 'gpt-5.6-sol')
                else:
                    check(slot + ':requested_high', body.get('reasoning_effort') == 'high' and body.get('thinking', {}).get('type') == 'enabled')
                    check(slot + ':requested_exact_flash_alias', body.get('model') == 'deepseek-flash')
            elif kind == 'native_stream_event':
                payload = e.get('payload') or {}
                native = payload.get('response', payload)
                if requests and native.get('usage'):
                    requests[-1]['raw_usage'] = native['usage']
                if payload.get('type') == 'response.completed':
                    check(slot + ':native_high_echo', native.get('reasoning', {}).get('effort') == 'high')
                if payload.get('type') in ('response.failed', 'response.incomplete', 'error'):
                    terminal_errors.append({'type': payload.get('type'), 'code': (native.get('error') or {}).get('code')})
            elif kind == 'native_usage' and requests:
                value = e['payload']; native = requests[-1]['raw_usage']
                check(slot + ':usage_present', isinstance(native, dict))
                if native:
                    a = native.get('input_tokens', native.get('prompt_tokens'))
                    b = native.get('output_tokens', native.get('completion_tokens'))
                    check(slot + ':usage_matches_wire', (a, b) == (value['input_tokens'], value['output_tokens']))
            elif kind == 'tool_request':
                uses[e['name']] += 1
            elif kind == 'tool_result':
                response = e.get('result', {})
                if e['name'] in ('read_file', 'read_skill', 'read_skill_section'):
                    chars += len(response.get('content', ''))
                hits += int(bool(response.get('cache_hit')))
                if response.get('success') is False:
                    tool_errors.append({'name': e['name'], 'error': response.get('error', response.get('stderr', ''))[-1500:]})
        check(slot + ':request_count', len(requests) == result['native_request_count'])
        check(slot + ':round_budget', result['native_request_count'] <= 12)
        check(slot + ':tokens', sum(u['total_tokens'] for u in result['native_usage']) == result['tokens_observed'])
        if result['model'] != 'gpt-5.6-sol':
            check(slot + ':native_flash_echo', all(m.casefold() in ('deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017') for m in result['native_models']))
        for i, submission in enumerate(result['submissions']):
            output, receipts = Path(submission['output']), Path(submission['receipts'])
            check(slot + ':submission_hashes', file_map(output) == submission['output_hashes'])
            check(slot + ':receipt_hashes', receipt_digest(receipts) == submission['receipts_sha256'])
            workitems.append((slot, i, case.id, submission, result['verdicts'][i]))
        repro = result.get('split_reproducibility', {'passed': True})
        if case.id == 'N07' and result['submissions']:
            recorded = read_json(Path(repro['evidence']))
            check(slot + ':replay_binding', recorded['passed'] == repro['passed'])
            if repro['passed']:
                from neurobench.runner.exec_dev.common import read_csv
                split = lambda p: {r['subject_id']: r['split'] for r in read_csv(p)}
                check(slot + ':actual_replayed_split', split(base / 'split_reproducibility/output/split.csv') == split(work / 'output/split.csv'))
        expected = bool(result['verdicts']) and result['verdicts'][-1]['verdict'] == 'PASS' and result['submissions'][-1]['elapsed_seconds'] <= 120 and result['inputs_unchanged'] and result['context_unchanged'] and repro['passed']
        check(slot + ':final_selection', expected == result['final_pass'])
        row = {k: result[k] for k in ('id', 'model', 'case_id', 'arm', 'family_id', 'input_kind', 'status', 'error', 'elapsed_seconds', 'native_request_count', 'tokens_observed', 'tool_calls', 'first_pass', 'final_pass')}
        row.update(verdict_reason=result['verdicts'][-1].get('reason', '') if result['verdicts'] else result['error'],
                   submitted=bool(result['submissions']), material_read_chars=chars, cache_hits=hits,
                   submission_count=len(result['submissions']), tool_usage=dict(uses), tool_errors=tool_errors,
                   native_terminal_errors=terminal_errors, split_reproducibility=repro)
        row.update(raw_first_pass=result['first_pass'], raw_final_pass=result['final_pass'],
                   raw_verdict_reason=row['verdict_reason'], acceptance_revision='exec-brief-roles-r2')
        rows.append(row)

    def cold(item):
        slot, i, cid, submission, expected = item
        verdict = delivery / 'cold_oracles' / slot / f'{i + 1}.json'
        cmd = [sys.executable, '-X', 'utf8', '-m', 'neurobench.runner.exec_brief.oracle', '--packet', str(packet), '--case', cid,
               '--index-sha', manifest['packet_index_sha256'], '--output', submission['output'], '--receipts', submission['receipts'],
               '--receipts-sha', submission['receipts_sha256'], '--verdict', str(verdict)]
        r = subprocess.run(cmd, capture_output=True, timeout=90)
        actual = read_json(verdict) if verdict.is_file() else {'verdict': 'ORACLE_ERROR', 'error': r.stderr.decode('utf-8', errors='replace')[-500:]}
        return slot, i, actual, expected

    by_slot, raw_by_slot = {r['id']: r for r in rows}, {r['id']: r for r in raw}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for slot, i, actual, expected in pool.map(cold, workitems):
            check(slot + f':cold_oracle_{i + 1}', actual == expected, {'actual': actual, 'recorded': expected} if actual != expected else None)
            original = raw_by_slot[slot]
            if original['case_id'] == 'N07':
                from neurobench.runner.exec_brief_acceptance_r2 import evaluate as evaluate_roles
                case = load_case(packet, 'N07', manifest['packet_index_sha256'])
                submission = original['submissions'][i]
                corrected = evaluate_roles(case, Path(submission['output']), Path(submission['receipts']), submission['receipts_sha256'])
                write_json(delivery / 'acceptance_r2' / slot / f'{i + 1}.json', corrected)
                check(slot + f':roles_r2_original_verdict_{i + 1}', corrected['raw_verdict'] == expected)
                check(slot + f':roles_r2_source_still_immutable_{i + 1}', file_map(submission['output']) == submission['output_hashes'])
                passed = corrected['verdict'] == 'PASS' and original['split_reproducibility']['passed']
                row = by_slot[slot]
                if i == 0:
                    row['first_pass'] = passed
                if i == len(original['submissions']) - 1:
                    row['final_pass'] = passed and submission['elapsed_seconds'] <= 120 and original['inputs_unchanged'] and original['context_unchanged']
                    row['verdict_reason'] = corrected.get('reason', '')
    report = {'passed': not errors, 'checks': len(checks), 'errors': errors,
              'case_results': len(rows), 'cold_submissions': len(workitems),
              'prior_bindings_preserved': len(manifest['p0_bindings']),
              'scope': 'Cold source/input/native-request/usage/output/receipt/raw selection consistency, plus uniformly applied, qualified roles-r2 acceptance correction. Engineering only, not expert science or independent backend attestation.',
              'records': checks}
    write_json(delivery / 'VERIFIED.json', report)
    return rows, report


def write_table(path, rows):
    fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows])


def unfinished_diagnostics(root, delivery, rows):
    """Post-terminal arithmetic only, never a replacement submission or success."""
    from neurobench.runner.exec_dev.common import finish
    import shutil
    manifest = read_json(root / 'MANIFEST.json'); packet = Path(manifest['packet'])
    records = []
    for row in rows:
        if row['submitted'] or row['case_id'] == 'N12':
            continue
        case = load_case(packet, row['case_id'], manifest['packet_index_sha256'])
        source = root / 'attempts' / row['id'] / 'unfinished_output'
        original = file_map(source)
        record = {'id': row['id'], 'raw_status': row['status'], 'excluded_from_main_results': True,
                  'original_output_hashes': original}
        missing = [p for p in case.contract['required_outputs'] if p not in original]
        if missing:
            record.update(verdict='NOT_EVALUATED', missing_artifacts=missing)
        else:
            target = delivery / 'unfinished_diagnostics' / row['id'] / 'output'
            shutil.copytree(source, target)
            finish(case, target, 'complete')
            from neurobench.runner.exec_brief_acceptance_r2 import evaluate
            receipts = root / 'attempts' / row['id'] / 'trusted_process_receipts'
            record.update(verdict=evaluate(case, target, receipts, receipt_digest(receipts)),
                          posthoc_host_generated_manifest=True, completion_intent_not_inferred=True)
        if file_map(source) != original:
            raise RuntimeError('unfinished_source_mutated')
        records.append(record)
    write_json(delivery / 'UNFINISHED_DIAGNOSTICS.json', {'records': records, 'main_selection_changed': False,
               'scope': 'Post-terminal diagnostic under a verifier-authored manifest, not on-time delivery or a recovered model attempt.'})
    return records


def deliver(root, delivery_name='delivery_r1'):
    if Path(delivery_name).name != delivery_name or not delivery_name.startswith('delivery_r'):
        raise ValueError('delivery_revision_must_be_a_local_name')
    delivery = root / delivery_name; delivery.mkdir(exist_ok=False)
    rows, audit = verify(root, delivery)
    if not audit['passed']:
        raise RuntimeError('cold_verification_failed:' + str(audit['errors'][:3]))
    diagnostic = unfinished_diagnostics(root, delivery, rows)
    summary, pairs = [], []
    models = ('gpt-5.6-sol', 'deepseek-v4.1-flash')
    for model in models:
        for arm in ARMS:
            group = [r for r in rows if r['model'] == model and r['arm'] == arm]
            normal = [r for r in group if r['case_id'] != 'N12']
            blocked = [r for r in group if r['case_id'] == 'N12']
            counter = sum((Counter(r['tool_usage']) for r in group), Counter())
            summary.append({'model': model, 'arm': arm, 'processable_pass': sum(r['final_pass'] for r in normal), 'processable_total': len(normal),
                            'first_submission_pass': sum(r['first_pass'] for r in normal),
                            'within_budget_recovered': sum(r['final_pass'] and not r['first_pass'] for r in normal),
                            'submission_count': sum(r['submission_count'] for r in group),
                            'correct_block': sum(r['final_pass'] for r in blocked), 'block_total': len(blocked),
                            'all_case_pass': sum(r['final_pass'] for r in group), 'median_all_case_seconds': statistics.median(r['elapsed_seconds'] for r in group),
                            'native_requests': sum(r['native_request_count'] for r in group), 'observed_tokens': sum(r['tokens_observed'] for r in group),
                            'read_chars': sum(r['material_read_chars'] for r in group), 'tool_usage': dict(counter), 'cache_hits': sum(r['cache_hits'] for r in group),
                            'status_counts': dict(Counter(r['status'] for r in group))})
        for off, on in (('A', 'B0'), ('A', 'B1'), ('B0', 'B1')):
            mapping = {(r['case_id'], r['arm']): r for r in rows if r['model'] == model}
            outcomes = Counter()
            for cid in (f'N{i:02d}' for i in range(1, 12)):
                a, b = mapping[cid, off]['final_pass'], mapping[cid, on]['final_pass']
                outcomes['both_pass' if a and b else 'on_only' if b else 'off_only' if a else 'both_fail'] += 1
            pairs.append({'model': model, 'comparison': on + '-' + off, 'processable_pairs': 11, **{k: outcomes[k] for k in ('both_pass', 'on_only', 'off_only', 'both_fail')}})
    write_json(delivery / 'RESULTS.json', {'rows': rows, 'summary': summary, 'pairs': pairs, 'descriptive_development_only': True})
    write_table(delivery / 'CASE_RESULTS.csv', rows); write_table(delivery / 'SUMMARY.csv', summary); write_table(delivery / 'PAIRS.csv', pairs)
    plot(delivery, summary)
    lines = ['# 精简题面后的 12 例复测', '', '12 个已暴露开发案例 × 2 模型 × OFF / 当前版 / 候选版，共 72 个原定槽位，各执行一次；请求 high，单题 120 秒、12 轮、48 次工具调用。', '',
             'DeepSeek 请求模型为 `deepseek-flash`，官方模型目录名称为 DeepSeek-V4.1-Flash，原生回显为 deepseek-flash。官方文档将当前别名对应到 V4.1-Flash-1017：[接口说明](https://api-docs.deepseek.com/quick_start/agent_integrations/)。GPT 沿用已授权 XNova Responses 通道和原生 GPT-5.6-Sol/high 回显。服务商回显不等于独立后端权重鉴定。', '',
             'OFF 组也能使用共同的 Python/QC 执行接口和 read_file 读取同一方法库。当前版增加技能搜索/读取、公开契约与静态提案检查；候选版再增加输入元数据整理、相关片段读取及会话缓存，均为本轮试验适配器。此处比较的是这些接口的增量，不是完整 NeuroClaw 产品与裸模型的差异。', '',
             '| 模型 | NR 关闭：完成/可处理 | 当前版 | 候选版 |', '|---|---:|---:|---:|']
    for model in models:
        s = [next(x for x in summary if x['model'] == model and x['arm'] == a) for a in ARMS]
        lines.append('| ' + ('GPT-5.6-Sol' if model.startswith('gpt') else 'DeepSeek V4.1 Flash') + ' | ' + ' | '.join(f"{x['processable_pass']}/{x['processable_total']}" for x in s) + ' |')
    lines += ['', f"要求阻断的案例共 {sum(s['correct_block'] for s in summary)}/6 正确处理，另行报告，不混入可处理题完成率。", '',
              '| 模型 / 组别 | 全 12 例耗时中位数（秒） | 模型请求数 |', '|---|---:|---:|']
    for s in summary:
        lines.append(f"| {s['model']} / {LABELS[s['arm']]} | {s['median_all_case_seconds']:.1f} | {s['native_requests']} |")
    lines += ['', '耗时包含失败与未知的全部尝试，不仅统计成功题。主表采用公开语义对应的 roles-r2 验收和预算内最终提交；首份提交通过率、运行内恢复数量另列于 SUMMARY.csv。CASE_RESULTS.csv 同时保留 raw_first_pass、raw_final_pass 和原判定原因。N07 题面写 validation，原 r1 验收器只接受 val；r2 仅归一化等价角色名称，对所有组统一复核，原输入、原始产物、原 r1 判定和时间/隔离/比例/重放要求均保留。该修订另有 9 项别名兼容与错误反例检查，没有重跑模型。原始状态、错误、工具调用、原生流与未完成输出均保留，没有终止后的重发。', '',
              '本轮修改了题面、输入别名与出题标记、部分多解验收、UTF-8 主机、受保护的产物字段和自动提交接口；DeepSeek 也换了模型。因此不能把与上一轮的差异归因于删提示或模型更换中的某一项。OFF、当前版和候选版在本轮共享相同输入、通用执行接口、方法库和预算。', '',
              'N07 还在干净目录重放所有记录的 Python 调用并核对划分一致性。六类任务每类两个，包含相近 QC 变体，仍是小型工程开发面板；改写已暴露的 12 例不会产生独立盲测集。不能由此建立总体显著性、科学准确性或稳定优越性。', '',
              f"准备阶段通过 48 项合法参考/替代解与错误反例检查及 5 项协议回归。冷复核通过 {audit['checks']} 项一致性核对、复算 {audit['cold_submissions']} 个冻结提交，并保留 {audit['prior_bindings_preserved']} 个历史绑定。冷复核使用冻结的独立产物验收程序，不能替代独立领域专家判断。", '',
              '[逐题结果](CASE_RESULTS.csv) · [配对不一致项](PAIRS.csv) · [汇总](SUMMARY.csv) · [图](COMPARISON.png) · [核对凭据](VERIFIED.json)', '', '## 未通过项', '']
    failures = [r for r in rows if not r['final_pass']]
    lines += [f"- {r['id']}: {r['status']}；{r['verdict_reason'] or r['error'] or '未通过重放或期限检查'}。" for r in failures] or ['无。']
    lines += ['', '## 过程与失败解释', '']
    for s in summary:
        tools = s['tool_usage']
        lines.append(f"- {s['model']} / {LABELS[s['arm']]}：compile_task={tools.get('compile_task', 0)}，技能正文/片段读取={tools.get('read_skill', 0) + tools.get('read_skill_section', 0)}，片段缓存命中={s['cache_hits']}；调用次数是过程证据。")
    for r in failures:
        if r['native_terminal_errors']:
            lines.append(f"- {r['id']} 原始状态保留为 {r['status']}；原生流记录 {r['native_terminal_errors']}，应与未观测到服务端终态的超时分开解释。")
        if any(e['name'] == 'finish_result' for e in r['tool_errors']):
            lines.append(f"- {r['id']} 曾把 Python 辅助函数 finish_result 当成未注册工具调用，收到 tool_not_exposed；这属于接口使用问题。")
    numeric_only = [r for r in diagnostic if isinstance(r.get('verdict'), dict) and r['verdict'].get('verdict') == 'PASS']
    lines += ['', f"未提交产物的终止后诊断中，{len(numeric_only)} 份在评估程序补建清单后通过数值/产物检查，仍不计为按时交付，见 UNFINISHED_DIAGNOSTICS.json。该检查不恢复任务、不请求模型，也不改写原输出。", '']
    (delivery / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(root / 'CURRENT_DELIVERY_ENTRYPOINTS.json', {k: str(delivery / v) for k, v in {'report': 'REPORT.md', 'results': 'RESULTS.json', 'case_results': 'CASE_RESULTS.csv', 'summary': 'SUMMARY.csv', 'pairs': 'PAIRS.csv', 'plot': 'COMPARISON.png', 'verification': 'VERIFIED.json'}.items()})
    write_json(root / 'DELIVERY_BINDINGS.json', {'scope': 'Entire trial tree except this binding and later readback; report implementation separately bound',
                                               'files': file_map(root), 'report_source': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}})
    print(json.dumps({'verified': audit['passed'], 'checks': audit['checks'], 'summary': summary, 'pairs': pairs}, ensure_ascii=False), flush=True)


def plot(delivery, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    colors = ['#708596', '#2F7691', '#DC864D']
    for ax, model, label in zip(axes, ('gpt-5.6-sol', 'deepseek-v4.1-flash'), ('GPT-5.6-Sol', 'DeepSeek V4.1 Flash')):
        group = [next(s for s in summary if s['model'] == model and s['arm'] == a) for a in ARMS]
        bars = ax.bar(np.arange(3), [s['processable_pass'] for s in group], color=colors, width=.6)
        for b, s in zip(bars, group):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .2, f"{s['processable_pass']}/11", ha='center', fontweight='bold')
        ax.set_xticks(range(3), [LABELS[a] for a in ARMS]); ax.set_title(label, pad=15)
        ax.set_ylim(0, 12.3); ax.set_yticks(range(0, 12, 2)); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.text(.5, -.18, 'Median all-case time: ' + ' / '.join(f"{s['median_all_case_seconds']:.1f}s" for s in group), transform=ax.transAxes, ha='center', fontsize=9)
    axes[0].set_ylabel('Verified on-time deliveries (11 processable cases)')
    fig.suptitle('Low-hint development trial | high reasoning | 120 seconds per case', y=.99, fontsize=13)
    fig.text(.5, .015, '12 exposed engineering cases; blocking scored separately; descriptive only.', ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.25, wspace=.18)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(delivery / ('COMPARISON.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--delivery', default='delivery_r1')
    arguments = parser.parse_args()
    deliver(arguments.root.resolve(), arguments.delivery)
