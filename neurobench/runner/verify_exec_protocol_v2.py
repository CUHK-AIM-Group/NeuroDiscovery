"""Offline qualification; no live model request or historical score mutation."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import argparse
import json
import shutil
import time
import unittest

from neurobench.runner.exec_dev.common import file_map, load_case, read_json, sha, write_json
from neurobench.runner.exec_brief_acceptance_r2 import evaluate
from neurobench.runner.exec_protocol_v2 import ProtocolTools, PYTHON_HOST, REVISION
from neurobench.runner.exec_trial import REPO, utc


def qualify(root):
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'START.json', {'at_utc': utc(), 'user_request': '可以',
        'accepted_proposal': 'Repair the common finish tool, structural diagnostics and budget feedback for all arms.',
        'scope': 'Offline implementation qualification; no authorization inferred for a new live campaign.',
        'new_native_requests': 0})
    log, prints = StringIO(), StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('neurobench.tests.test_exec_protocol_v2')
    with redirect_stdout(prints):
        tested = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    test_result = {'passed': tested.wasSuccessful(), 'tests': tested.testsRun,
        'failures': len(tested.failures), 'errors': len(tested.errors), 'skipped': len(tested.skipped),
        'runner_log': log.getvalue(), 'mocked_attempt_stdout': prints.getvalue(),
        'native_calls_are_mocked': True, 'new_native_requests': 0}
    write_json(root / 'UNIT_TEST_RESULT.json', test_result)
    if not tested.wasSuccessful():
        raise RuntimeError('protocol_unit_regressions_failed')
    previous = REPO / 'tmp/benchmark_exec_brief_trial_20261019_r1'
    manifest = read_json(previous / 'MANIFEST.json')
    packet = Path(manifest['packet'])
    slots = ('deepseek-v4.1-flash__N03__B0', 'deepseek-v4.1-flash__N07__B0',
             'deepseek-v4.1-flash__N07__B1', 'deepseek-v4.1-flash__N11__B0')
    original_bindings = {}
    checks = []
    for slot in slots:
        original = previous / 'attempts' / slot
        result = read_json(original / 'RESULT.json')
        assert result['status'] == 'BUDGET_STOP' and not result['final_pass'] and not result['submissions']
        case = load_case(packet, result['case_id'], manifest['packet_index_sha256'])
        old_products = original / 'unfinished_output'
        before = file_map(old_products)
        original_bindings[str(original / 'RESULT.json')] = sha(original / 'RESULT.json')
        original_bindings.update({str(old_products / p): h for p, h in before.items()})
        common = None
        for arm in ('A', 'B0', 'B1'):
            target = root / 'copied_products' / slot / arm
            work, private = target / 'work', target / 'attempt'
            private.mkdir(parents=True)
            old_work = previous / 'workspaces' / slot
            for name in ('inputs', 'context'):
                shutil.copytree(old_work / name, work / name)
            shutil.copytree(old_products, work / 'output')
            (work / 'scratch').mkdir()
            (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
            tools = ProtocolTools(case, work, private, arm, time.monotonic() + 120)
            prefix = tools.schemas()[:6]
            if common is not None:
                assert prefix == common, 'common_schema_varied_by_arm'
            common = prefix
            # A host-side boundary test, not twelve simulated model requests.
            tools.begin_round(12)
            diagnostic = tools.call('inspect_outputs', {})
            assert diagnostic['state'] == 'awaiting_finalization' and diagnostic['files'] == before
            finalized = tools.call('finish_result', {'status': 'complete',
                'summary': 'Offline protocol qualification of copied historical products.'})
            assert finalized['public_check_pass'] and tools.terminal and len(tools.submissions) == 1
            submission = tools.submissions[0]
            verdict = evaluate(case, Path(submission['output']), Path(submission['receipts']), submission['receipts_sha256'])
            assert verdict['verdict'] == 'PASS', verdict
            record = {'source_slot': slot, 'arm': arm, 'passed': True,
                'real_model_run': False, 'new_native_requests': 0,
                'simulated_tool_round_index': 12, 'tool_calls': tools.tool_count,
                'original_artifact_hashes': before, 'public_diagnostic_before_finish': diagnostic,
                'finish_result': finalized, 'product_verdict': verdict,
                'full_historical_N07_replay_claimed': False, 'historical_score_changed': False}
            write_json(target / 'QUALIFIED.json', record)
            checks.append({'slot': slot, 'arm': arm, 'passed': True, 'evidence': str(target / 'QUALIFIED.json')})
        assert file_map(old_products) == before, 'original_products_modified'
    for path, expected in original_bindings.items():
        assert sha(path) == expected, 'original_evidence_modified:' + path
    write_json(root / 'HISTORICAL_PRODUCTS.json', {'passed': True, 'checks': checks,
        'source_bindings': original_bindings, 'new_native_requests': 0,
        'scope': 'Host finalization and existing product oracle on copies; not a new model completion or score recovery. N07 full historical replay is not established.'})
    result = {'passed': True, 'protocol': REVISION, 'unit_tests': tested.testsRun,
        'copied_product_checks': len(checks), 'all_arms_same_common_schema_and_budget': True,
        'public_task_files_changed': False, 'scientific_acceptance_changed': False,
        'old_scores_changed': False, 'new_native_requests': 0, 'live_comparison_started': False}
    write_json(root / 'QUALIFIED.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    qualify(arguments.root.resolve())
