"""Cold verification and descriptive report for the fixed executable trial.

Re-evaluates saved submissions in new processes. Never sends model requests,
repairs candidate outputs or changes the frozen execution configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

from neurobench.runner.exec_dev.common import file_map, read_json, receipt_digest, sha, write_json


def verify(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    m = read_json(root / 'MANIFEST.json')
    gate = read_json(root / 'GATE.json')
    checks, problems = [], []
    def check(name, passed, detail=None):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
        if not passed:
            problems.append(name)
    check('manifest', sha(root / 'MANIFEST.json') == gate['manifest_sha256'])
    check('terminal_controller_receipt', (root / 'RUN_END.json').is_file())
    for p, digest in m['source_bindings'].items():
        check('runtime_source:' + p, sha(p) == digest)
    for p, digest in m['p0_bindings'].items():
        check('p0_original:' + p, sha(p) == digest)
    packet = Path(m['packet'])
    check('packet_index', sha(packet / 'INDEX.json') == m['packet_index_sha256'])
    check('methods', file_map(root / 'methods') == m['methods'])
    amendments = []
    for folder in sorted((root / 'cap_amendments').glob('*')):
        a = read_json(folder / 'AMENDMENT.json'); amendments.append(a)
        check('amendment_binding:' + folder.name, sha(folder / 'AMENDMENT.json') == read_json(folder / 'GATE.json')['amendment_sha256'])
        check('amendment_runner:' + folder.name, sha(Path(__file__).with_name('exec_trial_pending.py')) == a['source_sha256'])
        check('amendment_parent:' + folder.name, a['manifest_sha256'] == sha(root / 'MANIFEST.json'))
        check('amendment_terminal:' + folder.name, (folder / 'RUN_END.json').is_file())
        check('same_task_budgets:' + folder.name, a['case_wall_seconds_unchanged'] == 120 and a['max_rounds_unchanged'] == 12 and a['post_terminal_retries'] == 0)
        expected_old_tokens = 0
        for sid, digest in a['original_results_preserved'].items():
            original = root / 'attempts' / sid / 'RESULT.json'
            check('pre_amendment_result:' + sid, sha(original) == digest)
            expected_old_tokens += read_json(original)['tokens_observed']
        check('old_worker_stop_proof:' + folder.name, expected_old_tokens == a['observed_tokens_before'] and expected_old_tokens >= a['old_observed_token_cap_for_model'])
        ids = {s['id'] for s in a['pending_original_slots_only']}
        check('only_unattempted_original_slots:' + folder.name, not ids.intersection(a['original_results_preserved']) and all(s in m['slots'] for s in a['pending_original_slots_only']))
    rows, native_requests = [], 0
    contexts, account_intervals = {}, {}
    for slot in m['slots']:
        attempt = root / 'attempts' / slot['id']
        result_path = attempt / 'RESULT.json'
        if not result_path.is_file():
            check('no_ambiguous_live_attempt:' + slot['id'], not attempt.exists())
            rows.append({**slot, 'status': 'NOT_DISPATCHED', 'first_pass': False, 'final_pass': False,
                         'error': 'no_attempt_under_finite_campaign', 'elapsed_seconds': None,
                         'native_requests': 0, 'tokens_observed': None, 'usage_complete': False,
                         'final_reason': 'not_dispatched', 'claimed_status': None, 'submission_count': 0})
            continue
        r = read_json(result_path)
        start = read_json(attempt / 'START.json')
        account_intervals.setdefault(slot['model'], []).append((start['at_utc'], r['finished_at_utc']))
        check('slot_identity:' + slot['id'], all(r[k] == v for k, v in slot.items()) and start['slot'] == slot)
        work = root / 'workspaces' / slot['id']
        contract = read_json(work / 'context/CONTRACT.json')
        actual_context = file_map(work / 'context')
        check('public_inputs_preserved:' + slot['id'], file_map(work / 'inputs') == start['public_inputs'] == contract['input_sha256'])
        check('public_context_preserved:' + slot['id'], actual_context == start['context'])
        previous = contexts.setdefault(slot['case_id'], actual_context)
        check('identical_public_context:' + slot['id'], actual_context == previous)
        prompt = (attempt / 'PROMPT.txt').read_text(encoding='utf-8')
        expected = m['common_prompt'] + m['arm_prompts'][slot['arm']] + '\nPUBLIC TASK:\n' + (packet / 'public/cases' / slot['case_id'] / 'task.md').read_text(encoding='utf-8') + '\nPUBLIC CONTRACT:\n' + json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        check('frozen_prompt:' + slot['id'], prompt == expected)
        events = [json.loads(l) for l in (attempt / 'events.jsonl').read_text(encoding='utf-8').split('\n') if l.strip()]
        requests = [e for e in events if e['event'] == 'model_request']
        schemas = read_json(attempt / 'SCHEMAS.json')
        names = {'read_file', 'run_python', 'run_qc_tool', 'inspect_outputs', 'submit'}
        if slot['arm'] != 'A':
            names |= {'search_skills', 'read_skill', 'describe_task_contract', 'validate_proposal'}
        if slot['arm'] == 'B1':
            names |= {'compile_task', 'read_skill_section'}
        check('arm_tool_surface:' + slot['id'], {s['name'] for s in schemas} == names)
        native_requests += len(requests)
        check('request_budget:' + slot['id'], len(requests) <= r['native_request_count'] <= m['max_model_rounds'])
        for i, e in enumerate(requests):
            body = e['body']
            check('request_model:' + slot['id'] + ':' + str(i), body['model'] == slot['model'])
            check('request_high:' + slot['id'] + ':' + str(i), (body.get('reasoning') or {}).get('effort') == 'high' or body.get('reasoning_effort') == 'high')
            wanted = 'https://api.xnova.online/v1/responses' if slot['model'] == 'gpt-5.6-sol' else 'https://api.deepseek.com/chat/completions'
            check('request_endpoint:' + slot['id'] + ':' + str(i), e['endpoint'] == wanted)
            first = body.get('input', body.get('messages'))[0]['content']
            check('logged_model_context:' + slot['id'] + ':' + str(i), first == prompt)
            actual_schemas = [{k: v for k, v in s.items() if k != 'type'} for s in body['tools'][0]['tools']] if slot['model'] == 'gpt-5.6-sol' else [s['function'] for s in body['tools']]
            check('logged_tool_schemas:' + slot['id'] + ':' + str(i), actual_schemas == schemas)
        if slot['model'] == 'gpt-5.6-sol':
            native = [e['payload']['response'] for e in events if e['event'] == 'native_stream_event' and e['payload'].get('type') == 'response.completed']
            check('native_GPT_high_echo:' + slot['id'], all(v.get('model') == slot['model'] and (v.get('reasoning') or {}).get('effort') == 'high' for v in native))
        if r['status'] == 'SUBMITTED':
            check('native_completion_and_usage:' + slot['id'], r['usage_complete'] and len(r['native_models']) == len(requests) and all(v == slot['model'] for v in r['native_models']))
        check('submission_budget:' + slot['id'], len(r['submissions']) <= 2)
        check('all_submission_verdicts_present:' + slot['id'], len(r['submissions']) == len(r['verdicts']))
        cold = []
        for s, original in zip(r['submissions'], r['verdicts']):
            target = destination / 'verdicts' / (slot['id'] + '__' + str(s['number']) + '.json')
            check('immutable_output:' + slot['id'] + ':' + str(s['number']), file_map(s['output']) == s['output_hashes'])
            check('immutable_receipts:' + slot['id'] + ':' + str(s['number']), receipt_digest(s['receipts']) == s['receipts_sha256'])
            cmd = [sys.executable, '-X', 'utf8', '-m', 'neurobench.runner.exec_dev.oracle', '--packet', str(packet), '--case', slot['case_id'],
                   '--output', s['output'], '--index-sha', m['packet_index_sha256'], '--receipts', s['receipts'],
                   '--receipts-sha', s['receipts_sha256'], '--verdict', str(target)]
            completed = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=90)
            if not target.is_file():
                write_json(target, {'verdict': 'ORACLE_ERROR', 'reason': 'cold_subprocess_failed', 'stderr': completed.stderr[-1000:]})
            v = read_json(target); cold.append(v)
            check('cold_verdict:' + slot['id'] + ':' + str(s['number']), v == original)
        first_pass = bool(cold) and cold[0]['verdict'] == 'PASS' and r['submissions'][0]['elapsed_seconds'] <= 120
        final_pass = bool(cold) and cold[-1]['verdict'] == 'PASS' and r['submissions'][-1]['elapsed_seconds'] <= 120 and r['inputs_unchanged'] and r['context_unchanged']
        check('final_outcome:' + slot['id'], final_pass == r['final_pass'])
        final_reason = (cold[-1].get('reason', cold[-1]['verdict']) if cold else r['error'] or r['status'])
        if cold and cold[-1]['verdict'] == 'PASS' and not final_pass:
            final_reason = 'late_submission_or_changed_public_context'
        claim = r['submissions'][-1]['public_diagnostics'].get('claimed_status') if cold else None
        unfinished = None
        if not cold and (attempt / 'unfinished_output').is_dir():
            # Diagnostic only. Never upgrade an output that missed submission/budget.
            target = destination / 'unfinished_diagnostics' / (slot['id'] + '.json')
            receipts = attempt / 'trusted_process_receipts'
            cmd = [sys.executable, '-X', 'utf8', '-m', 'neurobench.runner.exec_dev.oracle', '--packet', str(packet), '--case', slot['case_id'],
                   '--output', str(attempt / 'unfinished_output'), '--index-sha', m['packet_index_sha256'],
                   '--receipts', str(receipts), '--receipts-sha', receipt_digest(receipts), '--verdict', str(target)]
            completed = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=90)
            unfinished = read_json(target) if target.is_file() else {'verdict': 'ORACLE_ERROR', 'reason': 'diagnostic_process_failure'}
        tool_requests = [e for e in events if e['event'] == 'tool_request']
        tool_results = [e for e in events if e['event'] == 'tool_result']
        check('tools_recorded:' + slot['id'], len(tool_requests) == len(tool_results))
        transport_errors = [e for e in tool_results if e['name'] == 'run_python' and
                            (e['result'].get('error_type') == 'TypeError' and 'NoneType' in str(e['result'].get('error'))
                             or 'UnicodeEncodeError' in str(e['result'].get('stderr'))
                             or 'UnicodeDecodeError' in str(e['result'].get('stderr')))]
        remote_errors = [e['payload'].get('response', {}).get('error', e['payload'].get('error')) for e in events
                         if e['event'] == 'native_stream_event' and e['payload'].get('type') in ('response.failed', 'response.incomplete', 'error')]
        failure_class = ('verified_delivery' if final_pass else 'oracle_measurement_error' if cold and cold[-1]['verdict'] == 'ORACLE_ERROR'
                         else 'submitted_contract_failure' if cold and cold[-1]['verdict'] == 'FAIL'
                         else 'task_round_budget' if r['error'] == 'max_model_rounds'
                         else 'task_deadline_native_completion_unknown' if r['status'] == 'UNKNOWN' and r['elapsed_seconds'] >= 119
                         else 'upstream_native_failure' if remote_errors else r['status'])
        def content_chars(value):
            if isinstance(value, dict):
                return sum(len(v) if k == 'content' and isinstance(v, str) else content_chars(v) for k, v in value.items())
            if isinstance(value, list):
                return sum(content_chars(v) for v in value)
            return 0
        material_chars = sum(content_chars(e['result']) for e in tool_results if e['name'] in ('read_file', 'read_skill', 'read_skill_section'))
        rows.append({**slot, 'status': r['status'], 'first_pass': first_pass, 'final_pass': final_pass,
                     'error': r['error'], 'elapsed_seconds': r['elapsed_seconds'], 'native_requests': len(requests),
                     'tokens_observed': r['tokens_observed'], 'usage_complete': r['usage_complete'], 'final_reason': final_reason,
                     'claimed_status': claim, 'submission_count': len(cold), 'tool_calls': r['tool_calls'],
                     'final_oracle_verdict': cold[-1]['verdict'] if cold else None,
                     'unfinished_oracle_diagnostic_not_scored': unfinished,
                     'material_read_chars': material_chars,
                     'executor_encoding_error_calls': len(transport_errors), 'native_remote_errors': remote_errors,
                     'outcome_class': failure_class,
                     'tool_counts': dict(Counter(e['name'] for e in tool_requests)),
                     'tool_errors': [{'name': e['name'], 'error': e['result'].get('error'), 'error_type': e['result'].get('error_type')}
                                     for e in tool_results if e['result'].get('success') is False],
                     'complete_claim_failed_contract': claim == 'complete' and bool(cold) and cold[-1]['verdict'] == 'FAIL' and r['status'] == 'SUBMITTED'})
    for model, intervals in account_intervals.items():
        intervals.sort()
        check('one_inflight_per_account:' + model, all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:])))
    check('finite_native_request_budget', native_requests <= m['max_native_requests'])
    check('unique_planned_slots', len(rows) == len(m['slots']) == len({r['id'] for r in rows}) == 72)
    summary = []
    for model in m['models']:
        for arm in ('A', 'B0', 'B1'):
            group = [r for r in rows if r['model'] == model and r['arm'] == arm]
            processable = [r for r in group if r['input_kind'] != 'must_block']
            blocked = [r for r in group if r['input_kind'] == 'must_block']
            times = [r['elapsed_seconds'] for r in group if r['elapsed_seconds'] is not None]
            summary.append({'model': model, 'arm': arm, 'planned': len(group), 'submitted': sum(r['status'] == 'SUBMITTED' for r in group),
                            'processable_n': len(processable), 'first_pass_n': sum(r['first_pass'] for r in processable),
                            'final_pass_n': sum(r['final_pass'] for r in processable), 'correct_block_n': sum(r['final_pass'] for r in blocked),
                            'must_block_n': len(blocked), 'all_contract_pass_n': sum(r['final_pass'] for r in group),
                            'median_seconds_all_attempts': statistics.median(times) if times else None,
                            'total_seconds_all_attempts': sum(times), 'native_requests': sum(r['native_requests'] for r in group),
                            'tokens_observed': sum(r['tokens_observed'] or 0 for r in group), 'all_usage_complete': all(r['usage_complete'] for r in group),
                            'material_read_chars': sum(r.get('material_read_chars', 0) for r in group),
                            'encoding_affected_runs': sum(r.get('executor_encoding_error_calls', 0) > 0 for r in group),
                            'statuses': dict(Counter(r['status'] for r in group)),
                            'oracle_errors': sum(r.get('final_oracle_verdict') == 'ORACLE_ERROR' for r in group),
                            'complete_claim_failed_contract': sum(r.get('complete_claim_failed_contract', False) for r in group)})
    pairs = []
    for model in m['models']:
        for left, right in (('A', 'B0'), ('A', 'B1'), ('B0', 'B1')):
            comparisons = []
            for cid in sorted({r['case_id'] for r in rows if r['input_kind'] != 'must_block'}):
                a = next(r for r in rows if (r['model'], r['arm'], r['case_id']) == (model, left, cid))
                b = next(r for r in rows if (r['model'], r['arm'], r['case_id']) == (model, right, cid))
                comparisons.append((a['final_pass'], b['final_pass']))
            pairs.append({'model': model, 'off_or_current': left, 'on_or_candidate': right, 'paired_processable_cases': len(comparisons),
                          'gains': sum(not a and b for a, b in comparisons), 'losses': sum(a and not b for a, b in comparisons),
                          'ties': sum(a == b for a, b in comparisons),
                          'delta_percentage_points': 100 * sum(int(b) - int(a) for a, b in comparisons) / len(comparisons)})
    write_json(destination / 'RESULTS.json', {'rows': rows, 'summary': summary, 'pairs': pairs})
    def export_csv(name, records):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (destination / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in records)
    export_csv('CASE_RESULTS.csv', rows); export_csv('SUMMARY.csv', summary); export_csv('PAIRS.csv', pairs)
    table = ['| 模型 | 条件 | 可处理案例通过 | 正确阻断 | 全部契约通过 | 所有尝试耗时中位数 |',
             '|---|---|---:|---:|---:|---:|']
    for s in summary:
        table.append(f"| {s['model']} | {s['arm']} | {s['final_pass_n']}/{s['processable_n']} | {s['correct_block_n']}/{s['must_block_n']} | {s['all_contract_pass_n']}/12 | {s['median_seconds_all_attempts']:.1f}s |" if s['median_seconds_all_attempts'] is not None else f"| {s['model']} | {s['arm']} | 未派发 | 未派发 | 0/12 | 未知 |")
    lines = ['# 十二例可执行开发对照', '', 'A：NeuroRuntime关闭，保留通用执行工具；B0：当前 NeuroRuntime；B1：候选任务契约整理、输入元数据和技能片段读取。三组均显式请求high思考强度，关闭的是NeuroRuntime能力。',
             '', '每个模型每条件12例，包括11例可处理和1例必须阻断。相同120秒墙钟、最多12轮模型请求和48次工具调用，公共输入/方法、执行工具和最多一次公开结构检查修复一致。模型不会看到数值 oracle 或参考解。', '', *table, '',
             'GPT 使用此前授权的 XNova Responses 通道，DeepSeek 使用官方 API。模型身份依据返回字段，high 是请求设置，GPT 还检查原生 effort 回显；DeepSeek 没有独立 high 执行证明。每账户单任务在途，条件顺序轮换。', '',
             '首次提交和最终提交均保留。进程/服务异常、未派发和隐藏验收失败分别记录，未自动补跑。主指标分母固定为每条件11例可处理案例，必须阻断例单独统计。耗时含该条件所有尝试，原生 token 不可观测部分不填补为已知成本。', '',
             '这12例是已暴露的工程开发材料，任务文本包含明确算法、参数和产物要求，不能证明科研正确性或推广收益。候选能力仅在本试验适配器内实现，未替换生产 NeuroRuntime。CLI 比较及新家族保留集尚未执行。', '',
             '## 配对变化', '', '| 模型 | 比较 | 改善 / 退步 / 相同 | 完成率变化 |', '|---|---|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['model']} | {p['on_or_candidate']} − {p['off_or_current']} | {p['gains']} / {p['losses']} / {p['ties']} | {p['delta_percentage_points']:+.1f}个百分点 |")
    lines += ['', '## 未通过的计划槽位', '']
    for r in rows:
        if not r['final_pass']:
            lines.append(f"- {r['model']} {r['case_id']} {r['arm']}: {r['status']}; {r['final_reason']}")
            if r.get('unfinished_oracle_diagnostic_not_scored'):
                lines.append('  停止时产物的事后诊断（未正式提交，不计通过）: ' + json.dumps(r['unfinished_oracle_diagnostic_not_scored'], ensure_ascii=False))
    if (root / 'E06_BOUNDARY_ASSESSMENT.json').is_file():
        issue = read_json(root / 'E06_BOUNDARY_ASSESSMENT.json')
        lines += ['', '## E06 的边界敏感性', '',
                  f"116个节点、密度0.1时，公开公式直接按浮点顺序计算为 {issue['literal_left_associative_float']}，ceil得到{issue['literal_float_ceil']}；先求整数候选边数或使用十进制数学值则为{issue['decimal_mathematical_ceil']}。DeepSeek候选版提交了{issue['actual_candidate_edges']}条边。冻结oracle严格要求667，原始FAIL未修改。", '',
                  '该结果暴露了计算规范需要明确的舍入边界。应保留原始数值失败，同时避免仅据这一条边的差异解释一般图分析能力或候选版收益；后续修订需在新的输入/协议版本中明确计算规则。其他条件的E06结果仍按同一冻结规则记录。']
    if (root / 'transport_qualification/RESULT.json').is_file():
        affected = [r['id'] for r in rows if r.get('executor_encoding_error_calls', 0)]
        lines += ['', '## 执行器测量问题', '',
                  'Windows子进程用-I时忽略PYTHONIOENCODING，原执行器的子进程可能输出GBK，父进程却按UTF-8读取。已观察到解码线程异常；独立中文/符号测试复现，显式-X utf8后通过。transport_qualification/保留原始字节和修正验证。这轮未中途修改执行器或补跑旧输入。', '',
                  '工具记录中检测到编码症状的槽位: ' + (', '.join(affected) if affected else '无匹配标记，另见执行工具终端的解码异常记录') + '。这些是测量限制，不能直接归为模型算法错误。', '',
                  '扩样前应统一任务文件位置，固定UTF-8进程传输，明确边数舍入规则，并调整公共轮数上限或明确其独立约束。原始严格完成率保留，当前结果只能作为开发诊断，不能据此确认候选版收益或删掉不利案例。']
    lines += ['', '逐案例、首次/最终提交、错误和用量见 CASE_RESULTS.csv；完整原始请求、工具记录与产物保留在 ../attempts/ 和 ../workspaces/。', '',
              f"独立冷复核：{len(checks)}项检查，{len(problems)}项不一致。验收程序逐个重新读取已提交产物并复算；P0的{len(m['p0_bindings'])}个既有文件绑定和当前运行源码一并复查。", '']
    for a in amendments:
        lines += ['阶段累计token修订：' + a['model'] + '原分配阈值为' + str(a['old_observed_token_cap_for_model']) +
                  '，触发后单独将该模型的阶段停止阈值改为' + str(a['new_observed_token_cap_for_model']) +
                  '，继续原计划中尚未派发的' + str(len(a['pending_original_slots_only'])) +
                  '个槽位。每例120秒/12轮、原始失败与UNKNOWN不变，未重跑旧槽位。修订与旧结果SHA绑定保留在 ../cap_amendments/。', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    result_bindings = file_map(destination)
    write_json(destination / 'VERIFIED.json', {'passed': not problems, 'check_count': len(checks), 'problems': problems, 'checks': checks,
                                             'planned_slots': len(rows), 'native_requests': native_requests, 'artifact_bindings': result_bindings,
                                             'verifier_sha256': sha(__file__), 'manifest_sha256': sha(root / 'MANIFEST.json'),
                                             'verification_scope': 'source/data/receipt/outcome consistency, not scientific acceptance',
                                             'performance_claim_ready': False,
                                             'expansion_requires': ['fixed_UTF8_execution_transport', 'explicit_graph_rounding_contract', 'review_common_round_and_delivery_protocol', 'new_holdout_families']})
    print(json.dumps({'passed': not problems, 'checks': len(checks), 'problems': problems, 'summary': summary, 'pairs': pairs}, ensure_ascii=False), flush=True)
    return 0 if not problems else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    a = p.parse_args()
    raise SystemExit(verify(a.root.resolve(), a.destination.resolve()))
