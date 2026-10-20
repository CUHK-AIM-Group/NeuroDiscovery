"""Cold readback for the GPT-6.1-Sol replacement, with archived Flash comparison."""
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
    check('slot_identity', len(manifest['slots']) == 36 == len({s['id'] for s in manifest['slots']}))
    raw = [read_json(p) for p in sorted((root / 'attempts').glob('*/RESULT.json'))]
    check('all_36_original_slots', len(raw) == 36 and {r['id'] for r in raw} == {s['id'] for s in manifest['slots']})
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
                if result['model'] == 'gpt-6.1-sol':
                    check(slot + ':requested_high', body.get('reasoning', {}).get('effort') == 'high')
                    check(slot + ':requested_exact_gpt', body.get('model') == 'gpt-6.1-sol')
                    check(slot + ':same_gpt_route', e['endpoint'] == 'https://api.xnova.online/v1/responses')
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
                    check(slot + ':native_exact_model_echo', native.get('model') == 'gpt-6.1-sol')
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
        if result['model'] != 'gpt-6.1-sol':
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


def summarize(rows, models):
    summary, pairs = [], []
    for model in models:
        group_rows = [r for r in rows if r['model'] == model]
        for arm in ARMS:
            group = [r for r in group_rows if r['arm'] == arm]
            normal = [r for r in group if r['case_id'] != 'N12']
            blocked = [r for r in group if r['case_id'] == 'N12']
            assert len(group) == 12 and len(normal) == 11 and len(blocked) == 1
            tools = sum((Counter(r['tool_usage']) for r in group), Counter())
            summary.append({
                'model': model, 'arm': arm, 'processable_pass': sum(r['final_pass'] for r in normal),
                'processable_total': 11, 'first_submission_pass': sum(r['first_pass'] for r in normal),
                'within_budget_recovered': sum(r['final_pass'] and not r['first_pass'] for r in normal),
                'correct_block': sum(r['final_pass'] for r in blocked), 'block_total': 1,
                'all_case_pass': sum(r['final_pass'] for r in group),
                'median_all_case_seconds': statistics.median(r['elapsed_seconds'] for r in group),
                'native_requests': sum(r['native_request_count'] for r in group),
                'observed_tokens': sum(r['tokens_observed'] for r in group),
                'read_chars': sum(r['material_read_chars'] for r in group),
                'cache_hits': sum(r['cache_hits'] for r in group), 'tool_usage': dict(tools),
                'status_counts': dict(Counter(r['status'] for r in group)),
                'result_origin': group[0]['result_origin'],
            })
        mapping = {(r['case_id'], r['arm']): r['final_pass'] for r in group_rows}
        for off, on in (('A', 'B0'), ('A', 'B1'), ('B0', 'B1')):
            outcomes = Counter()
            for cid in (f'N{i:02d}' for i in range(1, 12)):
                a, b = mapping[cid, off], mapping[cid, on]
                outcomes['both_pass' if a and b else 'on_only' if b else 'off_only' if a else 'both_fail'] += 1
            pairs.append({'model': model, 'comparison': on + '-' + off, 'processable_pairs': 11,
                          **{k: outcomes[k] for k in ('both_pass', 'on_only', 'off_only', 'both_fail')}})
    return summary, pairs


def plot(delivery, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    models = ('gpt-6.1-sol', 'deepseek-v4.1-flash')
    titles = ('GPT-6.1-Sol | new 36 attempts', 'DeepSeek V4.1 Flash | archived 36')
    for ax, model, title in zip(axes, models, titles):
        group = [next(s for s in summary if s['model'] == model and s['arm'] == a) for a in ARMS]
        bars = ax.bar(range(3), [s['processable_pass'] for s in group],
                      color=['#708596', '#2F7691', '#DC864D'], width=.6)
        for bar, s in zip(bars, group):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .2,
                    str(s['processable_pass']) + '/11', ha='center', fontweight='bold')
        ax.set_xticks(range(3), [LABELS[a] for a in ARMS])
        ax.set_title(title, pad=15, fontsize=11)
        ax.set_ylim(0, 12.3); ax.set_yticks(range(0, 12, 2))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.text(.5, -.18, 'Median all-case time: ' + ' / '.join(
            f"{s['median_all_case_seconds']:.1f}s" for s in group),
            transform=ax.transAxes, ha='center', fontsize=9)
    axes[0].set_ylabel('Verified on-time deliveries (11 processable cases)')
    fig.suptitle('Model replacement | high reasoning | same public tasks and budgets', y=.99, fontsize=12)
    fig.text(.5, .015, '12 exposed development cases; archived Flash results were not rerun; descriptive only.',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.25, wspace=.18)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(delivery / ('COMPARISON.' + extension), dpi=180)
    plt.close(fig)


def deliver(root):
    from neurobench.runner.exec_brief_report import write_table, unfinished_diagnostics
    delivery = root / 'delivery_r1'
    delivery.mkdir(exist_ok=False)
    rows, audit = verify(root, delivery)
    if not audit['passed']:
        raise RuntimeError('cold_verification_failed:' + str(audit['errors'][:3]))
    manifest = read_json(root / 'MANIFEST.json')
    previous_file = Path(manifest['comparison_reference'])
    assert sha(previous_file) == manifest['comparison_reference_sha256']
    previous = read_json(previous_file)
    for row in rows:
        row.update(result_origin='new_gpt61_attempt', source_results=str(root / 'attempts' / row['id'] / 'RESULT.json'))
    diagnostic = unfinished_diagnostics(root, delivery, rows)
    archived = []
    for row in previous['rows']:
        if row['model'] == 'deepseek-v4.1-flash':
            copy = dict(row)
            copy.update(result_origin='archived_flash_no_new_request', source_results=str(previous_file))
            archived.append(copy)
    assert len(archived) == 36
    combined = rows + archived
    models = ('gpt-6.1-sol', 'deepseek-v4.1-flash')
    summary, pairs = summarize(combined, models)
    write_json(delivery / 'RESULTS.json', {'rows': combined, 'summary': summary, 'pairs': pairs,
        'new_attempts': 36, 'archived_flash_attempts': 36, 'descriptive_development_only': True,
        'archived_source_sha256': sha(previous_file)})
    write_table(delivery / 'CASE_RESULTS.csv', combined)
    write_table(delivery / 'SUMMARY.csv', summary)
    write_table(delivery / 'PAIRS.csv', pairs)
    plot(delivery, summary)
    lines = ['# GPT-6.1-Sol 替换测试', '',
        '按用户要求将本次 GPT 比较模型换为 GPT-6.1-Sol，新增 12 个案例 × 三组，共 36 次尝试。DeepSeek V4.1 Flash 的 36 个原始尝试沿用上一轮冻结结果，本次没有重新请求。表中合并展示 72 个槽位，不表示本次执行了 72 次。', '',
        '三组均请求 high。A 为 NR 关闭，B0 为当前 NR 开启，B1 为候选 NR 开启。每题仍为 120 秒、12 轮、48 次工具调用；公开题面、数据、共同执行工具、方法库和 NR 提示与上一轮逐字一致。候选版仍是试验适配器。', '',
        'GPT 沿用已授权 XNova Responses 接口，精确请求 gpt-6.1-sol 并核对原生模型和 high 回显。模型目录及两次合成请求的工具调用往返测试通过；服务商回显不是独立后端权重鉴定。[官方模型说明](https://developers.openai.com/api/docs/models/gpt-6.1-sol)。', '',
        '| 模型 | A：NR 关闭 | B0：当前版开启 | B1：候选版开启 |',
        '|---|---:|---:|---:|']
    for model in models:
        group = [next(s for s in summary if s['model'] == model and s['arm'] == a) for a in ARMS]
        label = 'GPT-6.1-Sol（本次新增）' if model.startswith('gpt') else 'DeepSeek V4.1 Flash（沿用）'
        lines.append('| ' + label + ' | ' + ' | '.join(str(s['processable_pass']) + '/11' for s in group) + ' |')
    lines += ['', '表中只计 11 个可处理案例。缺失输入的阻断案例另计，共 ' +
              str(sum(s['correct_block'] for s in summary)) + '/6 正确处理。', '',
              '| 模型 / 组别 | 全 12 例耗时中位数（秒） | 请求数 | 观测 token 数 |',
              '|---|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['model']} / {LABELS[s['arm']]} | {s['median_all_case_seconds']:.1f} | {s['native_requests']} | {s['observed_tokens']} |")
    lines += ['', '耗时包括全部成功、失败和未知尝试。token 数只计收到的原生用量；未知请求可能缺失用量，不能据此声称实际总费用。SUMMARY.csv 同时记录首份提交通过数、预算内修复数、原生请求数、工具用量和原始状态。', '',
              '本次开始前已固定沿用经过九项资格检查的 roles-r2 等价划分标签验收。原 r1 判定、原始产物和时间要求保留，N07 仍要求干净进程重放。没有终止后模型重发，没有将离线补建清单的产物计为按时交付。', '',
              f"新 GPT 尝试通过 {audit['checks']} 项冷核对，复算 {audit['cold_submissions']} 份冻结提交；原有 {audit['prior_bindings_preserved']} 个输入/实现绑定保持不变。完整上一轮交付文件另在本次开始前和完成后核对。", '',
              '这是相同已暴露开发任务上的模型替换测试。Flash 沿用较早时间的记录；GPT 也与上一轮处于不同时间段。结果不构成盲测、稳定优势、科学专家验收或消除渠道波动后的因果比较。', '',
              '[逐题结果](CASE_RESULTS.csv) · [汇总](SUMMARY.csv) · [配对结果](PAIRS.csv) · [图](COMPARISON.png) · [冷核对](VERIFIED.json)', '', '## 新 GPT 的未通过项', '']
    failures = [r for r in rows if not r['final_pass']]
    lines += [f"- {r['id']}：{r['status']}；{r['verdict_reason'] or r['error'] or '未通过重放或期限检查'}。" for r in failures] or ['无。']
    lines += ['', '## 工具使用', '']
    for s in summary:
        if s['model'].startswith('gpt'):
            tools = s['tool_usage']
            lines.append(f"- {LABELS[s['arm']]}：compile_task={tools.get('compile_task', 0)}，技能正文/片段读取={tools.get('read_skill', 0) + tools.get('read_skill_section', 0)}，缓存命中={s['cache_hits']}。")
    numeric_only = [r for r in diagnostic if isinstance(r.get('verdict'), dict) and r['verdict'].get('verdict') == 'PASS']
    lines += ['', f"未提交产物的离线诊断中，{len(numeric_only)} 份在评估程序补建清单后通过产物检查；它们仍不计为按时交付，见 UNFINISHED_DIAGNOSTICS.json。", '']
    (delivery / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(root / 'CURRENT_DELIVERY_ENTRYPOINTS.json', {k: str(delivery / v) for k, v in {
        'report': 'REPORT.md', 'results': 'RESULTS.json', 'case_results': 'CASE_RESULTS.csv',
        'summary': 'SUMMARY.csv', 'pairs': 'PAIRS.csv', 'plot': 'COMPARISON.png', 'verification': 'VERIFIED.json'}.items()})
    print(json.dumps({'verified': True, 'checks': audit['checks'], 'summary': summary, 'pairs': pairs}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    arguments = parser.parse_args()
    deliver(arguments.root.resolve())


