"""Forward-only campaign-cap extension for originally planned unattempted slots.

Uses the unchanged frozen task runner. Does not resume, repair or resend any
attempt. May overlap the other provider only after this provider's old worker
has deterministically stopped at its fixed accumulated-token cap.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time

from neurobench.runner.exec_dev.common import read_json, sha, write_json
from neurobench.runner.exec_trial import attempt, check, utc


def main(root, model, action):
    m = check(root)
    if model not in m['models']:
        raise ValueError('model_not_in_original_trial')
    folder = root / 'cap_amendments' / model
    if action == 'prepare':
        results = []
        pending = []
        for slot in (s for s in m['slots'] if s['model'] == model):
            target = root / 'attempts' / slot['id']
            if (target / 'RESULT.json').is_file():
                results.append(read_json(target / 'RESULT.json'))
            elif target.exists():
                raise ValueError('ambiguous_attempt_never_resend:' + slot['id'])
            else:
                pending.append(slot)
        used = sum(r['tokens_observed'] for r in results)
        old_cap = m['observed_total_token_stop'] // 2
        if used < old_cap:
            raise ValueError('old_worker_not_proven_token_stopped')
        # The frozen old worker breaks unconditionally before the next dispatch
        # once used >= old_cap, and never restarts itself. Exclusive slot dirs
        # provide an additional duplicate-dispatch guard.
        folder.mkdir(parents=True, exist_ok=False)
        amendment = {'reason': 'Original per-model accumulated-token allocation exhausted before all 12 planned cases. Extend only campaign safety allowance, preserving every per-case budget and outcome.',
                     'authorized_scope': '先测试 12 例，看看对比结果', 'at_utc': utc(), 'model': model,
                     'manifest_sha256': sha(root / 'MANIFEST.json'), 'source_sha256': sha(__file__),
                     'old_observed_token_cap_for_model': old_cap, 'new_observed_token_cap_for_model': 3000000,
                     'original_results_preserved': {r['id']: sha(root / 'attempts' / r['id'] / 'RESULT.json') for r in results},
                     'observed_tokens_before': used, 'old_worker_stop_proof': 'frozen worker unconditional budget break; accrued complete-result count exceeds old cap; no unfinished attempt',
                     'pending_original_slots_only': pending, 'case_wall_seconds_unchanged': 120, 'max_rounds_unchanged': 12,
                     'post_terminal_retries': 0, 'other_provider_unchanged': True,
                     'maximum_campaign_allowance_after_amendment': 4500000,
                     'original_stage_deadline_seconds_unchanged': 4800, 'native_request_limit_unchanged': 864}
        write_json(folder / 'AMENDMENT.json', amendment)
        write_json(folder / 'GATE.json', {'amendment_sha256': sha(folder / 'AMENDMENT.json')})
        print(json.dumps({'prepared': True, 'model': model, 'pending_original_slots': len(pending), 'observed_tokens_before': used}), flush=True)
        return
    amendment = read_json(folder / 'AMENDMENT.json')
    assert sha(folder / 'AMENDMENT.json') == read_json(folder / 'GATE.json')['amendment_sha256']
    assert sha(__file__) == amendment['source_sha256']
    assert sha(root / 'MANIFEST.json') == amendment['manifest_sha256']
    for sid, digest in amendment['original_results_preserved'].items():
        assert sha(root / 'attempts' / sid / 'RESULT.json') == digest
    with (folder / 'RUN_START.json').open('x', encoding='utf-8') as f:
        json.dump({'at_utc': utc(), 'pid': os.getpid()}, f)
    original_start = datetime.fromisoformat(read_json(root / 'RUN_START.json')['at_utc'])
    used = amendment['observed_tokens_before']
    completed = []
    for slot in amendment['pending_original_slots_only']:
        elapsed = (datetime.now(timezone.utc) - original_start).total_seconds()
        if used >= amendment['new_observed_token_cap_for_model'] or elapsed + 120 > 4800:
            print(json.dumps({'stopped_before_dispatch': slot['id'], 'reason': 'amended_finite_campaign_cap'}), flush=True)
            break
        if (root / 'attempts' / slot['id']).exists():
            raise ValueError('slot_already_exposed_never_resend:' + slot['id'])
        result = attempt(root, slot)
        completed.append(slot['id']); used += result['tokens_observed']
        if result['error'].startswith(('native_http_401', 'native_http_403', 'native_http_429')):
            break
    write_json(folder / 'RUN_END.json', {'at_utc': utc(), 'completed_original_slots': completed,
                                       'observed_total_tokens_for_model': used, 'no_terminal_retries': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    main(a.root.resolve(), a.model, a.action)
