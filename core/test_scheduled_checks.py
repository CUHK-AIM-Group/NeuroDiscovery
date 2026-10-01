from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from core.scheduled_checks import ScheduledChecks


def test_no_implicit_check_and_bounded_read_only_observation(tmp_path):
    status_file = tmp_path / 'status.json'
    original = b'{"status":"running"}'
    status_file.write_bytes(original)
    checks = ScheduledChecks(tmp_path / 'runtime.db')
    task = checks.create('chat', tmp_path, 'status.json', interval=1, max_checks=2)
    checks.tick(task['next_check'] - 1)
    assert checks.list('chat')[0]['checks'] == 0
    checks.tick(task['next_check'])
    result = checks.list('chat')[0]
    assert result['checks'] == 1 and result['status'] == 'active'
    assert result['result']['reported_status'] == 'running'
    assert result['max_repairs'] == 0
    checks.tick(result['next_check'])
    result = checks.list('chat')[0]
    assert result['checks'] == 2 and result['status'] == 'completed'
    assert 'not a scientific completion' in result['result']['reason']
    assert status_file.read_bytes() == original


def test_restart_and_missing_file_pause_without_replay(tmp_path):
    checks = ScheduledChecks(tmp_path / 'runtime.db')
    task = checks.create('chat', tmp_path, 'missing.json', interval=1)
    checks.tick(task['next_check'])
    assert checks.list('chat')[0]['status'] == 'paused'
    assert checks.list('chat')[0]['checks'] == 1
    (tmp_path / 'missing.json').write_text('{"status":"running"}')
    checks.control(task['id'], 'chat', 'resume')
    checks.pause_on_restart()
    checks.tick(task['deadline'] - 1)
    assert checks.list('chat')[0]['status'] == 'paused'
    assert checks.list('chat')[0]['checks'] == 1
    with pytest.raises(ValueError):
        checks.control(task['id'], 'wrong-chat', 'resume')
    checks.control(task['id'], 'chat', 'cancel')
    with pytest.raises(ValueError):
        checks.control(task['id'], 'chat', 'resume')


def test_concurrent_checkers_claim_once(tmp_path):
    (tmp_path / 'status.json').write_text('{"status":"running"}')
    checks = ScheduledChecks(tmp_path / 'runtime.db')
    task = checks.create('chat', tmp_path, 'status.json', interval=1)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(checks.tick, [task['next_check']] * 4))
    assert checks.list('chat')[0]['checks'] == 1


@pytest.mark.parametrize('path', ['../outside.json', 'C:/outside.json', 'status.json:stream', ''])
def test_status_path_must_be_workspace_relative(tmp_path, path):
    with pytest.raises(ValueError):
        ScheduledChecks(tmp_path / 'runtime.db').create('chat', tmp_path, path)


@pytest.mark.parametrize('field,value', [('interval', 0), ('interval', True), ('hours', 25), ('max_checks', 145)])
def test_limits_fail_closed(tmp_path, field, value):
    with pytest.raises(ValueError):
        ScheduledChecks(tmp_path / 'runtime.db').create('chat', tmp_path, 'status.json', **{field: value})


def test_terminal_status_is_reported_not_independently_verified(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps({'status': 'completed', 'instruction': 'run something'}))
    checks = ScheduledChecks(tmp_path / 'runtime.db')
    task = checks.create('chat', tmp_path, 'status.json')
    checks.tick(task['next_check'])
    result = checks.list('chat')[0]
    assert result['status'] == 'completed'
    assert 'not independent acceptance' in result['result']['scope']
    assert 'instruction' not in result['result']
