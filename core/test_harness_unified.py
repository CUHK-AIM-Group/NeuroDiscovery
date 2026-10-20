"""Exercise the unified tools through real files, shared guards and MCP sessions."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from core import mcp_stdio, tool_registry
from core.harness_skills import default_capability
from core.tools_impl import register_core_tools


@pytest.fixture
def method(tmp_path):
    path = tmp_path / 'skills' / 'connectivity' / 'SKILL.md'
    path.parent.mkdir(parents=True)
    path.write_text('---\nname: connectivity\ndescription: functional connectivity\n---\n\n'
                    '# Overview\nGeneric guidance.\n\n# Correlation\nCompute correlation from observations.\n\n'
                    '# Example\nPath("workflow-ran").touch()\n', encoding='utf-8')
    return path


def host(tmp_path):
    register_core_tools()
    return mcp_stdio.HarnessHost(tmp_path, skills_dir=tmp_path / 'skills')


def dispatch(owner, name, arguments):
    return tool_registry.dispatch(owner, name, arguments, owner.workspace)


def test_sections_are_bounded_source_bound_and_cache_results_are_detached(tmp_path, method):
    cap = default_capability(tmp_path / 'skills', include_cwd=False)
    first = cap.load_section('connectivity', 'correlation', max_chars=55)
    assert first['success'] and 'Correlation' in first['content']
    assert len(first['content']) <= 55 and not first['cache_hit']
    original = first['content']
    first['content'] = 'caller mutation'
    second = cap.load_section('connectivity', 'correlation', max_chars=55)
    assert second['cache_hit'] and second['content'] == original
    assert second['source_sha256'] == hashlib.sha256(method.read_bytes()).hexdigest()
    assert second['content_sha256'] == hashlib.sha256(original.encode()).hexdigest()
    assert not second['workflow_executed'] and not (tmp_path / 'workflow-ran').exists()
    method.write_text(method.read_text(encoding='utf-8') + '\n\nCorrelation new evidence.\n', encoding='utf-8')
    third = cap.load_section('connectivity', 'correlation', max_chars=55)
    assert not third['cache_hit'] and third['source_sha256'] != second['source_sha256']


@pytest.mark.parametrize('query,limit', [('', 50), ('x' * 4097, 50), ('correlation', 0), ('correlation', 20001)])
def test_section_arguments_are_enforced_by_handler(tmp_path, method, query, limit):
    result = dispatch(host(tmp_path), 'read_skill_section', {'name': 'connectivity', 'query': query, 'max_chars': limit})
    assert result['success'] is False


def test_compile_preserves_public_requirements_and_reports_missing_inputs(tmp_path, method):
    np = pytest.importorskip('numpy')
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    source = inputs / 'observations.npy'
    np.save(source, np.arange(24, dtype='float32').reshape(6, 4))
    original = source.read_bytes()
    public = {'title': 'Functional connectivity', 'outputs': ['fc.npy']}
    result = dispatch(host(tmp_path), 'compile_task', {
        'task_text': 'Estimate functional connectivity.', 'public_contract': public,
        'input_paths': ['inputs/observations.npy', 'inputs/missing.npy']})
    assert result['success'] and result['requirements'] == public
    assert result['input_metadata']['inputs/observations.npy']['shape'] == [6, 4]
    assert result['input_metadata']['inputs/observations.npy']['dtype'] == 'float32'
    assert result['input_metadata']['inputs/observations.npy']['sha256'] == hashlib.sha256(original).hexdigest()
    assert result['input_metadata']['inputs/missing.npy'] == {'exists': False, 'status': 'missing'}
    assert result['method_candidates'][0]['name'] == 'connectivity'
    assert source.read_bytes() == original and not (tmp_path / 'output').exists()
    assert result['workflow_executed'] is False and result['scientific_correctness_verified'] is False


def test_compile_observes_nifti_space_without_loading_voxels(tmp_path, method, monkeypatch):
    np = pytest.importorskip('numpy'); nib = pytest.importorskip('nibabel')
    path = tmp_path / 'image.nii.gz'
    affine = np.diag([2., 3., 4., 1.])
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype='uint8'), affine), path)
    def no_voxel_read(*args, **kwargs):
        raise AssertionError('must only inspect headers')
    monkeypatch.setattr(nib.Nifti1Image, 'get_fdata', no_voxel_read)
    result = dispatch(host(tmp_path), 'compile_task', {'task_text': 'Inspect image', 'input_paths': ['image.nii.gz']})
    assert result['success']
    metadata = result['input_metadata']['image.nii.gz']
    assert metadata['shape'] == [3, 4, 5] and metadata['affine'] == affine.tolist()


def test_compile_bounds_hashing_and_refuses_workspace_escape(tmp_path, method, monkeypatch):
    from core import harness_tasks
    monkeypatch.setattr(harness_tasks, 'MAX_HASH_BYTES', 8)
    (tmp_path / 'large.bin').write_bytes(b'0123456789')
    owner = host(tmp_path)
    result = dispatch(owner, 'compile_task', {'task_text': 'Inspect input', 'input_paths': ['large.bin']})
    assert result['success']
    assert result['input_metadata']['large.bin']['sha256'] is None
    assert result['input_metadata']['large.bin']['hash_status'] == 'omitted_size_limit'
    refused = dispatch(owner, 'compile_task', {'task_text': 'Inspect input', 'input_paths': ['../outside.bin']})
    assert refused['success'] is False and 'outside_workspace' in refused['error']


def test_compile_uses_bound_task_and_rejects_changed_task_file(tmp_path, method):
    owner = host(tmp_path)
    task = tmp_path / 'task.txt'; task.write_text('Functional connectivity', encoding='utf-8')
    owner._evaluation_task_file = task
    owner._evaluation_task_sha256 = hashlib.sha256(task.read_bytes()).hexdigest()
    result = dispatch(owner, 'compile_task', {'task_text': 'Attempt to replace task'})
    assert result['task_text'] == 'Functional connectivity'
    task.write_text('Changed', encoding='utf-8')
    assert dispatch(owner, 'compile_task', {})['success'] is False


def test_mcp_reuses_cache_within_one_session_but_not_across_sessions(tmp_path, method, monkeypatch):
    monkeypatch.setenv('NEURORUNTIME_MCP_SKILLS_DIR', str(tmp_path / 'skills'))
    monkeypatch.delenv('NEURORUNTIME_MCP_TOOLS', raising=False)
    previous_workspace = mcp_stdio.os.environ.get('NEURORUNTIME_MCP_WORKSPACE')
    request = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
               'params': {'name': 'read_skill_section', 'arguments': {'name': 'connectivity', 'query': 'correlation'}}}
    frame = json.dumps(request) + '\n'
    out = io.StringIO(); mcp_stdio.serve([frame, frame], out, workspace=tmp_path)
    values = [json.loads(json.loads(line)['result']['content'][0]['text']) for line in out.getvalue().splitlines()]
    assert [v['cache_hit'] for v in values] == [False, True]
    other = io.StringIO(); mcp_stdio.serve([frame], other, workspace=tmp_path)
    assert json.loads(json.loads(other.getvalue())['result']['content'][0]['text'])['cache_hit'] is False
    assert mcp_stdio.os.environ.get('NEURORUNTIME_MCP_WORKSPACE') == previous_workspace


def test_direct_mcp_and_benchmark_share_result_and_evidence(tmp_path, method, monkeypatch):
    from neurobench.runner.drivers import SKILL_TOOLS, _dispatch_tool, _tool_schemas
    args = {'name': 'connectivity', 'query': 'correlation'}
    direct = dispatch(host(tmp_path), 'read_skill_section', args)
    monkeypatch.delenv('NEURORUNTIME_MCP_TOOLS', raising=False)
    mcp = mcp_stdio.call_tool('read_skill_section', args, host=host(tmp_path))
    http = _dispatch_tool('read_skill_section', args, tmp_path, skills_dir=tmp_path / 'skills')
    assert direct == http == json.loads(mcp['content'][0]['text'])
    assert {'compile_task', 'read_skill_section'} <= set(SKILL_TOOLS)
    assert {s['function']['name'] for s in _tool_schemas()} == set(SKILL_TOOLS)
    owner = host(tmp_path)
    dispatch(owner, 'read_skill_section', args)
    assert owner._tool_events[-1]['skills_used'] == ['connectivity']


def test_restricted_mcp_surface_does_not_advertise_new_tools(monkeypatch):
    monkeypatch.setenv('NEURORUNTIME_MCP_TOOLS', 'search_skills,read_skill')
    response = mcp_stdio.dispatch({'method': 'initialize', 'params': {}})
    assert 'compile_task' not in response['instructions'] and 'read_skill_section' not in response['instructions']
