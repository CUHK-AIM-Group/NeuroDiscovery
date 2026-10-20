"""Prepare and qualify a finite Flash run; never starts task dispatch itself."""
from pathlib import Path
import argparse
import json
import shutil
import time

from neurobench.runner.artifacts import EventLog
from neurobench.runner.exec_dev.common import read_json, sha, write_json
from neurobench.runner.exec_budget_flash_trial import prepare, REPO


def preflight(root):
    unit_root = REPO / 'tmp/benchmark_exec_budget_flash_preflight_20261019_r1'
    unit = read_json(unit_root / 'UNIT_TEST_RESULT.json')
    assert unit['passed'] and unit['tests'] == 22
    prepare(root, '重测 deepseek\n我们是不是要调高限时、伦次、工具次数，新加的任务会比较复杂')
    shutil.copytree(unit_root, root / 'qualification_budget')
    old = REPO / 'tmp/benchmark_exec_brief_trial_20261019_r1'
    shutil.copytree(old / 'acceptance_roles_r2', root / 'acceptance_roles_r2')
    manifest = read_json(root / 'MANIFEST.json')
    snapshots = []
    for p, h in manifest['source_bindings'].items():
        source = Path(p)
        target = root / 'source_snapshot/files' / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == h
        snapshots.append({'source': p, 'snapshot': str(target), 'sha256': h})
    write_json(root / 'source_snapshot/SOURCE_MAP.json', {'files': snapshots})
    write_json(root / 'source_snapshot/EXTRA_SOURCE_MAP.json', {'files': []})
    packages = []
    for name, seal in [('benchmark_exec_brief_trial_20261019_r1','DELIVERY_BINDINGS.json'),
                       ('benchmark_exec_brief_gpt61_trial_20261019_r1','DELIVERY_BINDINGS.json'),
                       ('benchmark_exec_protocol_v2_20261019_r1','BINDINGS.json')]:
        base = REPO / 'tmp' / name
        binding = base / seal
        files = read_json(binding)['files']
        assert all((base / p).is_file() and sha(base / p) == h for p, h in files.items()), name
        packages.append({'root': str(base), 'binding': str(binding), 'binding_sha256': sha(binding), 'files': len(files)})
    write_json(root / 'HISTORICAL_BINDINGS.json', {'packages': packages})
    # Read the provider catalogue using the same credential/transport as this trial.
    import httpx
    from neurobench.runner.high_channels import credential
    with httpx.Client(trust_env=False, timeout=30) as client:
        response = client.get('https://api.deepseek.com/models', headers={'Authorization':'Bearer '+credential('deepseek')})
        response.raise_for_status()
        catalogue = response.json()
    write_json(root / 'preflight_native/MODELS.json', catalogue)
    ids = [r['id'] for r in catalogue['data']]
    assert 'deepseek-flash' in ids, 'official_flash_alias_absent'
    # Exactly two bounded, non-case requests establish a real tool response roundtrip.
    from neurobench.runner.exec_brief.channel import flash_round
    from neurobench.runner.exec_protocol_v2 import FINISH_SCHEMA
    log = EventLog(root / 'preflight_native/SYNTHETIC_EVENTS.jsonl')
    history = [{'role':'user','content':'Synthetic protocol integration check. No dataset or research task. Call finish_result exactly once with status="blocked" and reason="synthetic check". After the tool result reply DONE without further tools.'}]
    deadline = time.monotonic() + 120
    try:
        first, usage1 = flash_round(history, [FINISH_SCHEMA], log, deadline)
        assert len(first['calls']) == 1 and first['calls'][0]['name'] == 'finish_result'
        call = first['calls'][0]
        assert call['arguments']['status'] == 'blocked' and call['arguments'].get('reason')
        history.append(first['message'])
        history.append({'role':'tool','tool_call_id':call['call_id'],'name':'finish_result','content':json.dumps({'success':True,'terminal':True,'synthetic':True})})
        second, usage2 = flash_round(history, [FINISH_SCHEMA], log, deadline)
        assert not second['calls'] and usage1['usage_complete'] and usage2['usage_complete']
        write_json(root / 'preflight_native/SYNTHETIC_RESULT.json', {'passed':True,'native_requests':2,
            'first_model':first['model'],'second_model':second['model'],'usage':[usage1,usage2],
            'case_data_exposed':False,'host_submission_created':False,'text':second['text']})
    finally:
        log.close()
    result = {'passed':True, 'manifest_sha256':sha(root/'MANIFEST.json'), 'unit_tests':22,
              'snapshot_sources':len(snapshots),'historical_files_preserved':sum(p['files'] for p in packages),
              'catalogue_alias_available':True,'synthetic_roundtrip_requests':2,
              'task_attempts_started':0,'common_budget':manifest['budget']}
    write_json(root / 'PREFLIGHT.json', result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    preflight(p.parse_args().root.resolve())
