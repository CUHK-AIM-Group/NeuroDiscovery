"""Cold source/native/artifact readback for the Flash budget comparison."""
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
    check('flash_only', manifest['models'] == ['deepseek-v4.1-flash'])
    check('selected_budget', manifest['budget'] == {'task_seconds': 300, 'model_rounds': 24, 'tool_calls': 96, 'python_call_seconds': 30, 'max_submissions': 2})
    preflight = read_json(root / 'PREFLIGHT.json')
    check('preflight_passed', preflight['passed'] and preflight['manifest_sha256'] == sha(root / 'MANIFEST.json'))
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
                    check(slot + ':official_flash_route', e['endpoint'] == 'https://api.deepseek.com/chat/completions')
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
        check(slot + ':round_budget', result['native_request_count'] <= manifest['max_model_rounds'])
        check(slot + ':tool_budget', result['tool_calls'] <= manifest['max_tool_calls'])
        feedback = [e['payload'] for e in events(base / 'events.jsonl') if e['event'] == 'budget_feedback_before_model']
        check(slot + ':common_budget_feedback', len(feedback) == result['native_request_count'] and
              all(f['model_round'] == i and f['max_model_rounds'] == 24 and 0 <= f['tool_calls_remaining'] <= 96
                  for i, f in enumerate(feedback, 1)))
        common_schemas = read_json(base / 'SCHEMAS.json')[:6]
        check(slot + ':registered_finish', common_schemas[5]['name'] == 'finish_result')
        if '_common_schemas' not in contexts:
            contexts['_common_schemas'] = common_schemas
        check(slot + ':identical_common_schemas', common_schemas == contexts['_common_schemas'])
        check(slot + ':tokens', sum(u['total_tokens'] for u in result['native_usage']) == result['tokens_observed'])
        if result['model'] != 'gpt-6.1-sol':
            check(slot + ':native_flash_echo', all(m.casefold() in ('deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017') for m in result['native_models']))
        for i, submission in enumerate(result['submissions']):
            output, receipts = Path(submission['output']), Path(submission['receipts'])
            check(slot + ':submission_hashes', file_map(output) == submission['output_hashes'])
            check(slot + ':receipt_hashes', receipt_digest(receipts) == submission['receipts_sha256'])
            check(slot + ':submission_budget', 0 <= submission['elapsed_seconds'] <= 300 and
                  1 <= submission['model_round'] <= 24 and submission['tool_calls'] <= 96)
            workitems.append((slot, i, case.id, submission, result['verdicts'][i]))
        repro = result.get('split_reproducibility', {'passed': True})
        if case.id == 'N07' and result['submissions']:
            recorded = read_json(Path(repro['evidence']))
            check(slot + ':replay_binding', recorded['passed'] == repro['passed'])
            if repro['passed']:
                from neurobench.runner.exec_dev.common import read_csv
                split = lambda p: {r['subject_id']: r['split'] for r in read_csv(p)}
                check(slot + ':actual_replayed_split', split(base / 'split_reproducibility/output/split.csv') == split(work / 'output/split.csv'))
        expected = bool(result['verdicts']) and result['verdicts'][-1]['verdict'] == 'PASS' and result['submissions'][-1]['elapsed_seconds'] <= manifest['deadline_seconds_per_task'] and result['inputs_unchanged'] and result['context_unchanged'] and repro['passed']
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
                    row['final_pass'] = passed and submission['elapsed_seconds'] <= manifest['deadline_seconds_per_task'] and original['inputs_unchanged'] and original['context_unchanged']
                    row['verdict_reason'] = corrected.get('reason', '')
    report = {'passed': not errors, 'checks': len(checks), 'errors': errors,
              'case_results': len(rows), 'cold_submissions': len(workitems),
              'prior_bindings_preserved': len(manifest['p0_bindings']),
              'scope': 'Cold source/input/native-request/usage/output/receipt/raw selection consistency, plus uniformly applied, qualified roles-r2 acceptance correction. Engineering only, not expert science or independent backend attestation.',
              'records': checks}
    write_json(delivery / 'VERIFIED.json', report)
    return rows, report


def historical_preservation(root):
    records = []
    for item in read_json(root / 'HISTORICAL_BINDINGS.json')['packages']:
        base = Path(item['root'])
        binding = Path(item['binding'])
        assert sha(binding) == item['binding_sha256'], str(binding)
        files = read_json(binding)['files']
        failures = [p for p, h in files.items() if not (base / p).is_file() or sha(base / p) != h]
        records.append({'root': str(base), 'files': len(files), 'passed': not failures, 'failures': failures})
    result = {'passed': all(r['passed'] for r in records), 'packages': records,
              'files': sum(r['files'] for r in records)}
    write_json(root / 'HISTORICAL_PRESERVATION_FINAL.json', result)
    return result


def observed_old_caps(root, rows, packet, index_sha):
    from neurobench.runner.exec_brief_acceptance_r2 import evaluate as evaluate_roles
    for row in rows:
        raw = read_json(root / 'attempts' / row['id'] / 'RESULT.json')
        case = load_case(packet, row['case_id'], index_sha)
        passed_submissions = []
        for i, submission in enumerate(raw['submissions']):
            corrected = evaluate_roles(case, Path(submission['output']), Path(submission['receipts']), submission['receipts_sha256'])
            if corrected['verdict'] == 'PASS' and raw['split_reproducibility']['passed'] and raw['inputs_unchanged'] and raw['context_unchanged']:
                passed_submissions.append(submission)
        within = any(s['elapsed_seconds'] <= 120 and s['model_round'] <= 12 and s['tool_calls'] <= 48 for s in passed_submissions)
        row.update(observed_pass_within_old_caps=within,
                   pass_beyond_old_caps=bool(row['final_pass'] and not within),
                   final_submission_seconds=raw['submissions'][-1]['elapsed_seconds'] if raw['submissions'] else None,
                   final_submission_round=raw['submissions'][-1]['model_round'] if raw['submissions'] else None)


def plot(delivery, previous, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(3))
    old = [next(s for s in previous['summary'] if s['model'] == 'deepseek-v4.1-flash' and s['arm'] == a)['processable_pass'] for a in ARMS]
    current = [s['processable_pass'] for s in summary]
    ax.bar([n - .2 for n in x], old, .38, color='#9aa9b5', label='Previous protocol | 120s / 12 rounds / 48 tools')
    bars = ax.bar([n + .2 for n in x], current, .38, color='#327f96', label='New common protocol | 300s / 24 rounds / 96 tools')
    for n, value in zip(x, old):
        ax.text(n - .2, value + .15, f'{value}/11', ha='center')
    for b, s in zip(bars, summary):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + .15, f"{s['processable_pass']}/11", ha='center')
    ax.set_xticks(x, ['A: NR OFF', 'B0: current NR ON', 'B1: candidate NR ON'])
    ax.set_ylim(0, 12.5)
    ax.set_yticks(range(0, 12, 2))
    ax.set_ylabel('Verified deliveries / 11 processable cases')
    ax.set_title('DeepSeek V4.1 Flash | high | 12 exposed development cases', pad=16)
    ax.legend(loc='lower left', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
    fig.text(.5, .02, 'Before/after changes both protocol and budget; descriptive only. Required block scored separately.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.86)
    for ext in ('png','svg','pdf'):
        fig.savefig(delivery / ('COMPARISON.' + ext), dpi=180)
    plt.close(fig)


def deliver(root):
    from neurobench.runner.exec_brief_report import write_table, unfinished_diagnostics
    from neurobench.runner.exec_brief_gpt61_report import summarize
    delivery = root / 'delivery_r1'
    delivery.mkdir(exist_ok=False)
    rows, audit = verify(root, delivery)
    if not audit['passed']:
        raise RuntimeError('cold_verification_failed:' + str(audit['errors'][:3]))
    manifest = read_json(root / 'MANIFEST.json')
    packet = Path(manifest['packet'])
    previous_path = Path(manifest['comparison_reference'])
    assert sha(previous_path) == manifest['comparison_reference_sha256']
    previous = read_json(previous_path)
    observed_old_caps(root, rows, packet, manifest['packet_index_sha256'])
    for row in rows:
        row.update(result_origin='new_flash_v3_budget', source_results=str(root / 'attempts' / row['id'] / 'RESULT.json'))
    summary, pairs = summarize(rows, ('deepseek-v4.1-flash',))
    for s in summary:
        group = [r for r in rows if r['arm'] == s['arm'] and r['case_id'] != 'N12']
        s.update(observed_processable_pass_within_old_caps=sum(r['observed_pass_within_old_caps'] for r in group),
                 processable_pass_beyond_old_caps=sum(r['pass_beyond_old_caps'] for r in group))
    diagnostic = unfinished_diagnostics(root, delivery, rows)
    preservation = historical_preservation(root)
    if not preservation['passed']:
        raise RuntimeError('historical_binding_changed')
    result = {'rows': rows, 'summary': summary, 'pairs': pairs, 'new_native_attempts': 36,
              'old_flash_summary': [s for s in previous['summary'] if s['model'] == 'deepseek-v4.1-flash'],
              'budget': manifest['budget'], 'analysis_predeclared': manifest['analysis_predeclared'],
              'exposed_development_only': True}
    write_json(delivery / 'RESULTS.json', result)
    write_table(delivery / 'CASE_RESULTS.csv', rows)
    write_table(delivery / 'SUMMARY.csv', summary)
    write_table(delivery / 'PAIRS.csv', pairs)
    plot(delivery, previous, summary)
    lines = ['# DeepSeek Flash 共用协议与预算复测', '',
             '本轮新增 12 个案例 × A/B0/B1，共 36 次原生尝试，官方 DeepSeek V4.1 Flash 接口，请求 high。A 为 NR 关闭，B0 为当前 NR 开启，B1 为候选 NR 开启。GPT 没有重测。', '',
             '三组共同使用 300 秒、24 轮、96 次工具调用、每次 Python 最多 30 秒、最多两份结构提交。公开题面、输入、科学验收和方法库保留原绑定；共用 finish_result、文件清单诊断与预算反馈已更新。复杂任务的 900 秒、48 轮、192 次工具调用档位尚未执行。', '',
             '| 组别 | 旧轮次完成 | 本轮完成 | 本轮在旧上限内已完成 | 本轮超过旧上限才完成 |',
             '|---|---:|---:|---:|---:|']
    for s in summary:
        old = next(v for v in previous['summary'] if v['model'] == 'deepseek-v4.1-flash' and v['arm'] == s['arm'])
        lines.append(f"| {s['arm']} {LABELS[s['arm']]} | {old['processable_pass']}/11 | {s['processable_pass']}/11 | {s['observed_processable_pass_within_old_caps']}/11 | {s['processable_pass_beyond_old_caps']} |")
    lines += ['', '主表只计 11 个可处理案例。缺失输入案例另计，本轮正确阻断 ' + str(sum(s['correct_block'] for s in summary)) + '/3。', '',
              '“本轮在旧上限内”同时要求真实合格提交在 120 秒、12 轮、48 次调用以内。这是新协议和新预算提示下的实际观察，不等于使用旧预算重新运行的反事实结果。本次同时变更共用协议与预算，旧轮次与本轮的差异不能单独归因于 NR。', '',
              '| 组别 | 全 12 例耗时中位数 | 原生请求数 | 首份提交通过 | 原始状态 |', '|---|---:|---:|---:|---|']
    for s in summary:
        lines.append(f"| {s['arm']} | {s['median_all_case_seconds']:.1f}s | {s['native_requests']} | {s['first_submission_pass']}/11 | {s['status_counts']} |")
    lines += ['', '所有失败、未知、原始 r1 验收结果与未完成输出均保留。已在请求前固定 roles-r2 等价划分标签规则；N07 保留所有记录 Python 调用的干净重放要求，任何先前失败的调用也可能使整体重放失败。没有终止后模型重试，也没有把补建清单的离线产物诊断算成模型成功。', '',
              f"冷核对通过 {audit['checks']} 项，重算 {audit['cold_submissions']} 份冻结提交；复核 {preservation['files']} 个历史包文件未变。原生记录核对 high 参数、Flash 别名、官方渠道、用量、输入、工具协议、提交时刻与产物哈希。服务商回显不代表独立后端权重鉴定。", '',
              '这些仍是 12 个已暴露工程开发案例，包含相近变体。结果用于确定执行接口和预算是否足够，不能推断新任务总体成功率、科学专家准确率或稳定优势。', '',
              '[逐题结果](CASE_RESULTS.csv) · [汇总](SUMMARY.csv) · [配对结果](PAIRS.csv) · [图](COMPARISON.png) · [冷核对](VERIFIED.json)', '',
              '## 未通过项', '']
    failed = [r for r in rows if not r['final_pass']]
    lines += [f"- {r['case_id']}/{r['arm']}: {r['status']}；{r['verdict_reason'] or r['error'] or '重放或预算不合格'}。" for r in failed] or ['无。']
    lines += ['', '## 可观测 NR 使用', '']
    for s in summary:
        lines.append(f"- {s['arm']}：{s['tool_usage']}；片段缓存命中 {s['cache_hits']}。")
    (delivery / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(root / 'CURRENT_DELIVERY_ENTRYPOINTS.json', {k: str(delivery / v) for k, v in {
        'report':'REPORT.md', 'results':'RESULTS.json', 'summary':'SUMMARY.csv', 'case_results':'CASE_RESULTS.csv',
        'plot':'COMPARISON.png', 'verification':'VERIFIED.json'}.items()})
    # The caller performs visual checks and the final seal after this process exits.
    print(json.dumps({'passed': True, 'checks': audit['checks'], 'summary': summary, 'pairs': pairs}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    deliver(parser.parse_args().root.resolve())

