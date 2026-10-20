"""A skill read is bound guidance, with identical evidence over HTTP and MCP."""
from __future__ import annotations
import hashlib
from pathlib import Path

from core.harness_skills import default_capability


def fixture_skill(tmp_path: Path):
    path=tmp_path/'skills'/'fixture-method'/'SKILL.md'
    path.parent.mkdir(parents=True)
    text='---\r\nname: fixture-method\r\ndescription: method fixture\r\n---\r\n'
    # Loading an example must never perform its workflow side effect.
    text+='神经影像\r\n```python\r\nfrom pathlib import Path\r\nPath("workflow-ran").touch()\r\n```\r\n'
    path.write_bytes(text.encode('utf-8'))
    return path,text


def test_hashes_bind_original_bytes_and_all_normalized_pages(tmp_path):
    path,text=fixture_skill(tmp_path)
    cap=default_capability(path.parents[1],include_cwd=False)
    pages=[];offset=0
    while True:
        page=cap.load('fixture-method',max_chars=31,offset=offset);pages.append(page)
        if not page['truncated']:break
        offset=page['next_offset']
    normalized=text.replace('\r\n','\n')
    assert ''.join(p['content'] for p in pages)==normalized
    assert all(p['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() for p in pages)
    assert all(p['content_sha256']==hashlib.sha256(normalized.encode('utf-8')).hexdigest() for p in pages)
    assert all(p['content_kind']=='method_guidance' and p['workflow_executed'] is False for p in pages)
    assert not (tmp_path/'workflow-ran').exists()


def test_http_and_mcp_consumer_return_and_log_identical_method_evidence(tmp_path):
    from neurobench.runner.drivers import _dispatch_tool
    from core import tool_registry
    from core.mcp_stdio import HarnessHost
    from core.tools_impl import register_core_tools
    path,_=fixture_skill(tmp_path);skills=path.parents[1];register_core_tools()
    http=_dispatch_tool('read_skill',{'name':'fixture-method'},tmp_path,skills_dir=skills)
    host=HarnessHost(tmp_path,skills_dir=skills)
    mcp=tool_registry.dispatch(host,'read_skill',{'name':'fixture-method'},tmp_path)
    assert http==mcp
    assert host._tool_events[-1]['result']==mcp
    assert mcp['workflow_executed'] is False
    assert not (tmp_path/'workflow-ran').exists()


def test_hash_changes_when_method_source_changes(tmp_path):
    path,_=fixture_skill(tmp_path);cap=default_capability(path.parents[1],include_cwd=False)
    first=cap.load('fixture-method');path.write_bytes(path.read_bytes()+b'New method detail\n')
    second=cap.load('fixture-method')
    assert first['source_sha256']!=second['source_sha256']
    assert first['content_sha256']!=second['content_sha256']
    assert second['content'].endswith('New method detail\n')
