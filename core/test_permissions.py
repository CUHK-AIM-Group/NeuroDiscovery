import pytest

from core.permissions import tool_permission
from core.runtime_store import RuntimeStore
from core.agent.test_autoresearch_execution import session


@pytest.mark.parametrize('mode,command,expected', [
    ('ask', 'pwd', 'ask'), ('risk', 'pwd', 'allow'),
    ('risk', 'python analysis.py', 'ask'), ('risk', 'pwd; Remove-Item data', 'ask'),
    ('risk', 'pwd > results.txt', 'ask'), ('never', 'python analysis.py', 'allow'),
    ('read_only', 'pwd', 'deny'), ('invalid', 'pwd', 'deny'),
])
def test_shell_approval_modes(tmp_path, mode, command, expected):
    assert tool_permission(mode, 'run_shell_command', {'command': command}, tmp_path) == expected


@pytest.mark.parametrize('mode', ['ask', 'risk', 'never', 'read_only'])
def test_permission_does_not_remove_workspace_and_delegation_boundaries(tmp_path, mode):
    assert tool_permission(mode, 'read_workspace_file', {'path': 'result.md'}, tmp_path) == 'allow'
    assert tool_permission(mode, 'read_workspace_file', {'path': '../outside'}, tmp_path) == 'deny'
    assert tool_permission(mode, 'spawn_subagent', {}, tmp_path) == 'deny'
    assert tool_permission(mode, 'unknown_tool', {}, tmp_path) == 'deny'


def test_never_ask_skips_approval_but_honors_stop(tmp_path):
    agent, _ = session(tmp_path, [], mode='off')
    agent._runtime_store = RuntimeStore(tmp_path / 'runtime.db')
    agent._runtime_request_id = 'synthetic-request'
    agent.permission_mode = 'never'
    assert agent._tool_authorized('run_shell_command', {'command': 'synthetic only'}, 'one')
    assert agent._runtime_store.approval('synthetic-request', 'one') == 'missing'
    agent._cancel_event.set()
    assert not agent._tool_authorized('run_shell_command', {'command': 'synthetic only'}, 'two')
